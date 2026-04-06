# /// script
# dependencies = [
#   'torch',
#   'numpy',
#   'gymnasium',
#   'tqdm',
#   'fire',
#   'wandb',
#   'einops',
#   'accelerate',
#   'memmap-replay-buffer',
#   'x-mlps-pytorch',
#   'moviepy'
# ]
# ///

from __future__ import annotations

import os
import shutil
import glob
import math

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm
from collections import deque

import fire
import wandb

from accelerate import Accelerator
from memmap_replay_buffer import ReplayBuffer

from einops import rearrange
from x_mlps_pytorch import MLP
from rectified_flow_pytorch.value_flow import ValueFlow
from rectified_flow_pytorch.rectified_flow import Unet
from assoc_scan import AssocScan
import torch.nn.functional as F

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

# sinusoidal positional embedding

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim = -1)

# mlp actor

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ImageEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env, size = 64):
        super().__init__(env)
        self.size = size

    def observation(self, obs):
        img = self.env.render()
        img_t = torch.tensor(img, dtype = torch.float32).permute(2, 0, 1) / 255.0
        return F.interpolate(img_t.unsqueeze(0), size = (self.size, self.size)).squeeze(0).numpy()

# mlp critic backbone

class CriticBackbone(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 256, z_dim = 1, use_unet = False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(64),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.z_mlp = nn.Linear(z_dim, hidden_dim)
        self.state_mlp = nn.Linear(state_dim, hidden_dim)
        self.action_emb = nn.Embedding(action_dim, hidden_dim)

        self.to_return_velocity = MLP(hidden_dim, hidden_dim, hidden_dim, 1)
        self.to_state_velocity = MLP(hidden_dim, hidden_dim, hidden_dim, state_dim)

        self.unet = Unet(dim = 32, channels = 3, accept_cond = True, dim_cond = 1) if use_unet else None

    def forward(self, x = None, times = None, cond = None, action = None):
        if x.ndim == 4 and exists(self.unet):
            # purely compute the flow trajectory matching natively for images
            return self.unet(x, times = times, cond = cond), torch.zeros(x.shape[0], 1, device = x.device)

        times = rearrange(times, 'b 1 -> b') if times.ndim == 2 else times
        cond = rearrange(cond, 'b -> b 1') if cond.ndim == 1 else cond
        state = rearrange(x, 'b ... -> b (...)') if x.ndim > 2 else x

        h = self.time_mlp(times) + self.z_mlp(cond) + self.state_mlp(state)

        if exists(action):
            action = rearrange(action, 'b 1 -> b') if action.ndim > 1 else action
            h = h + self.action_emb(action.long())

        return self.to_state_velocity(h), self.to_return_velocity(h)

# gae

def calc_gae(rewards, values, last_value, masks, gamma=0.99, lam=0.95, use_accelerated=None):
    assert values.shape[-1] == rewards.shape[-1]
    use_accelerated = default(use_accelerated, rewards.is_cuda)

    values = F.pad(values, (0, 1), value = float(last_value))
    values_current, values_next = values[..., :-1], values[..., 1:]

    delta = rewards + gamma * values_next * masks - values_current
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)
    gae = scan(gates, delta)

    return gae, gae + values_current

# training

def main(
    num_episodes = 3000,
    update_every = 10,
    gamma = 0.99,
    lam = 0.95,
    lr = 3e-4,
    clip_ratio = 0.2,
    ppo_epochs = 10,
    batch_size = 64,
    record_every = 100,
    max_steps = 700,
    video_folder = './results_ppo_state',
    wandb_project = 'value-flows',
    wandb_run_name = 'LunarLander-FlowPPO-State',
    num_flow_steps = 10,
    gate_by_aleatoric_uncertainty = True,
    aleatoric_temperature = 1.0,
    gate_by_delight = True,
    delight_temperature = 1.0,
    prob_state_generation = 0.25,
    image_state = False,
    cpu = False
):
    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    if accelerator.is_main_process:
        wandb.init(project = wandb_project, name = wandb_run_name, config = locals())
        shutil.rmtree(video_folder, ignore_errors = True)
        shutil.rmtree('./replay_data_ppo', ignore_errors = True)

    env = gym.make('LunarLander-v3', render_mode = 'rgb_array', max_episode_steps = max_steps)
    env = RecordVideo(env, video_folder = video_folder, episode_trigger = lambda ep: divisible_by(ep, record_every), disable_logger = True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = PPOActor(state_dim, action_dim)

    critic_backbone = CriticBackbone(state_dim, action_dim, use_unet = image_state)

    critic = ValueFlow(
        model = critic_backbone,
        gamma = gamma,
        num_flow_steps = num_flow_steps,
        prob_state_generation = prob_state_generation,
        ema_kwargs = dict(update_every = 1)
    )

    buffer = ReplayBuffer(
        './replay_data_ppo',
        max_episodes = update_every,
        max_timesteps = max_steps,
        circular = True,
        fields = dict(
            state = ('float', (state_dim,)),
            image_state = ('float', (3, 64, 64)),
            action = 'int',
            log_prob = 'float',
            reward = 'float',
            done = 'float',
            value = 'float',
            adv = 'float',
            target_ret = 'float'
        )
    )

    actor_optim = Adam(actor.parameters(), lr = lr)
    critic_optim = Adam(critic.model.parameters(), lr = lr)
    actor, actor_optim, critic, critic_optim = accelerator.prepare(actor, actor_optim, critic, critic_optim)

    if accelerator.is_main_process:
        print(f'recording samples to {video_folder}')

    pbar = tqdm(range(1, num_episodes + 1))
    recent_rewards = deque(maxlen = 20)

    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0.
        done = truncated = False

        with buffer.one_episode():
            while not (done or truncated):
                state_t = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)

                if image_state:
                    img = env.render()
                    img_t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                    img_scaled = F.interpolate(img_t.unsqueeze(0), size=(64, 64)).squeeze(0).numpy()
                else:
                    img_scaled = np.zeros((3, 64, 64), dtype=np.float32)

                with torch.no_grad():
                    logits = actor(state_t)
                    dist = Categorical(logits = logits)
                    action = dist.sample()
                    lprob = dist.log_prob(action)

                act_v = action.item()
                next_state, reward, done, truncated, _ = env.step(act_v)
                score = float(reward)

                buffer.store(
                    state = state,
                    image_state = img_scaled,
                    action = act_v,
                    log_prob = lprob.item(),
                    reward = score,
                    done = float(done),
                    value = 0.,          # deferred to end of episode
                    adv = 0.,
                    target_ret = 0.
                )

                state = next_state
                episode_reward += score

        buffer.flush()

        ep_idx = (buffer.episode_index - 1) % buffer.max_episodes
        T = buffer.episode_lens[ep_idx]

        b_data = {k: torch.tensor(v[ep_idx, :T], dtype = torch.float32, device = device) for k, v in buffer.data.items()}
        states_t, rewards_t, dones_t = b_data['state'], b_data['reward'], b_data['done']

        with torch.no_grad():
            state_t = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
            all_states = torch.cat((states_t, state_t), dim=0)

            chunk_size = 512
            all_values = []
            for i in range(0, all_states.shape[0], chunk_size):
                chunk = all_states[i:i+chunk_size]
                chunk_vals = critic.sample_q_value(chunk, num_samples = 1, zero_variance = True).squeeze(-1)
                all_values.append(chunk_vals)

            all_values = torch.cat(all_values, dim=0)

            values_t = all_values[:-1]
            last_value = all_values[-1].item()

            buffer.data['value'][ep_idx, :T] = values_t.cpu().numpy()

        advs, returns = calc_gae(rewards_t, values_t, last_value, 1. - dones_t, gamma, lam)

        buffer.data['adv'][ep_idx, :T] = advs.cpu().numpy()
        buffer.data['target_ret'][ep_idx, :T] = returns.cpu().numpy()

        recent_rewards.append(episode_reward)
        avg_reward = np.mean(recent_rewards)
        pbar.set_description(f'reward {avg_reward:.1f}')

        log_data = dict(cumulative_reward = episode_reward, episode = episode)

        # ppo update on collected buffer

        if divisible_by(episode, update_every):
            num_eps = min(buffer.num_episodes, buffer.max_episodes)

            all_states, all_image_states, all_actions, all_old_lp, all_advs, all_rets = [], [], [], [], [], []
            for index in range(num_eps):
                ln = buffer.episode_lens[index]
                all_states.append(torch.tensor(buffer.data['state'][index, :ln], device=device))
                if image_state:
                    all_image_states.append(torch.tensor(buffer.data['image_state'][index, :ln], device=device))
                all_actions.append(torch.tensor(buffer.data['action'][index, :ln], device=device))
                all_old_lp.append(torch.tensor(buffer.data['log_prob'][index, :ln], device=device))
                all_advs.append(torch.tensor(buffer.data['adv'][index, :ln], device=device))
                all_rets.append(torch.tensor(buffer.data['target_ret'][index, :ln], device=device))

            b_states = torch.cat(all_states, dim=0)
            b_image_states = torch.cat(all_image_states, dim=0) if image_state else None
            b_actions = torch.cat(all_actions, dim=0).long()
            b_old_lp = torch.cat(all_old_lp, dim=0)
            b_adv = torch.cat(all_advs, dim=0)
            b_ret = torch.cat(all_rets, dim=0)

            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            dataset_size = len(b_states)
            indices = np.arange(dataset_size)

            actor_losses = []
            critic_losses = []
            dcfm_losses = []
            bcfm_losses = []
            state_gen_losses = []

            total_steps = ppo_epochs * ((dataset_size + batch_size - 1) // batch_size)
            ppo_pbar = tqdm(total=total_steps, desc='ppo update', leave=False)

            for _ in range(ppo_epochs):
                np.random.shuffle(indices)

                for start in range(0, dataset_size, batch_size):
                    idx = indices[start:start + batch_size]
                    mb_states, mb_actions, mb_old_lp, mb_adv, mb_ret = b_states[idx], b_actions[idx], b_old_lp[idx], b_adv[idx], b_ret[idx]
                    mb_image_states = b_image_states[idx] if image_state else None

                    # actor

                    logits = actor(mb_states)
                    dist = Categorical(logits = logits)
                    new_lp = dist.log_prob(mb_actions)
                    ratio = torch.exp(new_lp - mb_old_lp)

                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1. - clip_ratio, 1. + clip_ratio) * mb_adv
                    entropy = dist.entropy().mean()

                    # uncertainty & delightful gating mechanisms

                    gating = torch.ones_like(mb_adv)

                    if gate_by_aleatoric_uncertainty and not image_state:
                        uncertainty = critic.get_aleatoric_uncertainty(mb_states, mb_actions, num_samples = 16)
                        aleatoric_gating = torch.sigmoid(-aleatoric_temperature * (uncertainty - uncertainty.mean())) * 2.0
                        gating = gating * aleatoric_gating.detach()

                    if gate_by_delight:
                        # Delightful Policy Gradient - Ian Osband (https://arxiv.org/abs/2603.01234)
                        surprisal = -mb_old_lp
                        delight = mb_adv * surprisal
                        delight_gating = torch.sigmoid(delight_temperature * delight) * 2.0
                        gating = gating * delight_gating.detach()

                    actor_loss = (-torch.min(surr1, surr2) * gating).mean() - 0.01 * entropy

                    actor_optim.zero_grad()
                    accelerator.backward(actor_loss)
                    actor_optim.step()

                    # critic

                    critic_optim.zero_grad()

                    critic_loss, (loss_dcfm, loss_bcfm, loss_state_gen) = critic(
                        state = mb_states,
                        action = mb_actions,
                        explicit_target_return = mb_ret,
                        loss_weight = gating,
                        flow_state = mb_image_states
                    )

                    accelerator.backward(critic_loss)
                    critic_optim.step()
                    critic.target_model.update()

                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    dcfm_losses.append(loss_dcfm.item())
                    bcfm_losses.append(loss_bcfm.item())
                    state_gen_losses.append(loss_state_gen.item())
                    ppo_pbar.update(1)

            ppo_pbar.close()

            log_data.update(
                loss_actor = np.mean(actor_losses),
                loss_critic = np.mean(critic_losses),
                loss_dcfm = np.mean(dcfm_losses),
                loss_bcfm = np.mean(bcfm_losses),
                loss_state_gen = np.mean(state_gen_losses)
            )

        if divisible_by(episode, record_every) and accelerator.is_main_process:
            mp4_files = glob.glob(os.path.join(video_folder, '*.mp4'))
            if mp4_files:
                log_data.update(video = wandb.Video(max(mp4_files, key = os.path.getctime), fps = 30, format = 'mp4'))

            # explicitly sample states across spread out returns
            spread_out_returns = torch.tensor([-200., 0., 100., 200., 300.], device=device)
            with torch.no_grad():
                shape_gen = (3, 64, 64) if image_state else state_dim
                # Wait, shape_gen needs to unpack `state_dim` cleanly!
                s_shape = (3, 64, 64) if image_state else (state_dim,)
                sampled_states = critic.sample_state(spread_out_returns, state_shape=s_shape)

            if image_state:
                wandb_images = []
                for ret, img_t in zip(spread_out_returns.cpu().numpy(), sampled_states.cpu()):
                    img_np = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    wandb_images.append(wandb.Image(img_np, caption=f"Return: {ret}"))
                log_data.update(sampled_images=wandb_images)
            else:
                table = wandb.Table(columns=["target_return", "x", "y", "vx", "vy", "angle", "v_angle", "leg1", "leg2"])
                for ret, state_arr in zip(spread_out_returns.cpu().numpy(), sampled_states.cpu().numpy()):
                    table.add_data(ret, *state_arr.tolist())
                log_data.update(sampled_states = table)

            # colocate the raw generated state array with the local recordings and upload natively
            file_path = os.path.join(video_folder, f'sampled_states_episode_{episode}.npy')
            np.save(file_path, sampled_states.cpu().numpy())
            wandb.save(file_path, base_path=video_folder)

        if accelerator.is_main_process:
            wandb.log(log_data)

    env.close()

    if accelerator.is_main_process:
        wandb.finish()

if __name__ == '__main__':
    fire.Fire(main)
