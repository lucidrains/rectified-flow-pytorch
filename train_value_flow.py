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
import random

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Adam

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm
from collections import deque

import fire
import wandb

from einops import rearrange, repeat
from x_mlps_pytorch import MLP
from accelerate import Accelerator
from memmap_replay_buffer import ReplayBuffer

from rectified_flow_pytorch.value_flow import ValueFlow

# helpers

def exists(val):
    return val is not None

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

# critic backbone with objective routing

class CriticBackbone(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 256, z_dim = 1):
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

    def forward(self, x = None, times = None, cond = None, action = None):
        times = rearrange(times, 'b 1 -> b') if times.ndim == 2 else times
        cond = rearrange(cond, 'b -> b 1') if cond.ndim == 1 else cond
        state = rearrange(x, 'b ... -> b (...)') if x.ndim > 2 else x

        batch, device = x.shape[0], x.device

        h = self.time_mlp(times) + self.z_mlp(cond) + self.state_mlp(state)

        if exists(action):
            action = rearrange(action, 'b 1 -> b') if action.ndim > 1 else action
            h = h + self.action_emb(action.long())

        return self.to_state_velocity(h), self.to_return_velocity(h)

# training

def main(
    num_episodes = 2500,
    batch_size = 128,
    gamma = 0.99,
    lr = 3e-4,
    epsilon_start = 1.0,
    epsilon_end = 0.05,
    epsilon_decay_steps = 500,
    record_every = 100,
    max_steps = 700,
    video_folder = './results',
    wandb_project = 'value-flows',
    wandb_run_name = 'LunarLander-ValueFlows',
    train_every = 20,
    train_steps = 50,
    num_flow_steps = 10,
    lambda_bcfm = 1.0,
    prob_state_generation = 0.25,
    gate_by_aleatoric_uncertainty = True,
    aleatoric_temperature = 1.0,
    cpu = False
):
    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    if accelerator.is_main_process:
        wandb.init(
            project = wandb_project,
            name = wandb_run_name,
            config = dict(
                num_episodes = num_episodes,
                batch_size = batch_size,
                gamma = gamma,
                lr = lr,
                epsilon_decay_steps = epsilon_decay_steps,
                train_every = train_every,
                train_steps = train_steps,
                num_flow_steps = num_flow_steps,
                lambda_bcfm = lambda_bcfm,
                prob_state_generation = prob_state_generation,
                gate_by_aleatoric_uncertainty = gate_by_aleatoric_uncertainty,
                aleatoric_temperature = aleatoric_temperature,
                cpu = cpu
            )
        )

        if os.path.exists(video_folder):
            shutil.rmtree(video_folder, ignore_errors = True)

    env = gym.make('LunarLander-v3', render_mode = 'rgb_array', max_episode_steps = max_steps)
    env = RecordVideo(
        env,
        video_folder = video_folder,
        episode_trigger = lambda ep: divisible_by(ep, record_every),
        disable_logger = True
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    critic_backbone = CriticBackbone(state_dim, action_dim, hidden_dim = 256)

    critic = ValueFlow(
        model = critic_backbone,
        gamma = gamma,
        num_flow_steps = num_flow_steps,
        lambda_bcfm = lambda_bcfm,
        use_symlog = True,
        prob_state_generation = prob_state_generation,
        ema_kwargs = dict(beta = 0.995, update_every = 1)
    )

    optimizer = Adam(critic.model.parameters(), lr = lr)

    critic, optimizer = accelerator.prepare(critic, optimizer)

    buffer = ReplayBuffer(
        './replay_data_q',
        max_episodes = num_episodes,
        max_timesteps = max_steps,
        fields = dict(
            state = ('float', (state_dim,)),
            action = 'int',
            reward = 'float',
            next_state = ('float', (state_dim,)),
            done = 'bool'
        )
    )

    if accelerator.is_main_process:
        print(f'recording samples to {video_folder}')

    pbar = tqdm(range(1, num_episodes + 1))
    recent_rewards = deque(maxlen = 20)
    all_rewards = []

    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0.
        done = truncated = False

        epsilon = max(epsilon_end, epsilon_start - (episode - 1) * (epsilon_start - epsilon_end) / epsilon_decay_steps)

        with buffer.one_episode():
            while not (done or truncated):
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_t = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
                        actions_t = torch.arange(action_dim, device = device, dtype = torch.long)
                        states_expanded = state_t.expand(action_dim, *state_t.shape[1:])

                        q_all = critic.sample_q_value(states_expanded, actions_t, num_samples = 1, zero_variance = True)
                        action = q_all.argmax(dim = -1).item()

                next_state, reward, current_done, truncated, _ = env.step(action)

                buffer.store(
                    state = state,
                    action = action,
                    reward = float(reward),
                    next_state = next_state,
                    done = current_done
                )

                state = next_state
                episode_reward += reward
                done = current_done

        buffer.flush()

        recent_rewards.append(episode_reward)
        all_rewards.append(episode_reward)
        avg_reward = np.mean(recent_rewards)

        pbar.set_description(f'reward {avg_reward:.1f} | ε {epsilon:.2f}')

        dcfm_losses = []
        bcfm_losses = []
        sgen_losses = []

        if divisible_by(episode, train_every) and buffer.num_episodes > 1:
            dataset = buffer.dataset(timestep_level = True)
            dl = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)
            dl = accelerator.prepare(dl)
            dl_iter = iter(dl)

            for _ in tqdm(range(train_steps), desc = 'learning', leave = False):
                try:
                    batch = next(dl_iter)
                except StopIteration:
                    dl_iter = iter(dl)
                    batch = next(dl_iter)

                states = batch['state'].to(device)
                actions = batch['action'].to(device)
                rewards = batch['reward'].to(device)
                next_states = batch['next_state'].to(device)
                dones = batch['done'].to(device).float()

                # greedy next actions

                with torch.no_grad():
                    act_range = torch.arange(action_dim, device = device, dtype = torch.long)
                    act_expanded = repeat(act_range, 'a -> (a b)', b = batch_size)
                    ns_expanded = repeat(next_states, 'b ... -> (a b) ...', a = action_dim)

                    q_all = critic.sample_q_value(ns_expanded, act_expanded, num_samples = 1, zero_variance = True)
                    q_all = rearrange(q_all, '(a b) -> b a', a = action_dim)
                    next_actions = q_all.argmax(dim = -1)

                optimizer.zero_grad()

                gating = None
                if gate_by_aleatoric_uncertainty:
                    with torch.no_grad():
                        uncertainty = critic.get_aleatoric_uncertainty(states, actions, num_samples = 16)
                        gating = torch.sigmoid(-aleatoric_temperature * (uncertainty - uncertainty.mean())) * 2.0

                loss, (l_dcfm, l_bcfm, l_sgen) = critic(
                    state = states,
                    action = actions,
                    reward = rewards,
                    next_state = next_states,
                    next_action = next_actions,
                    dones = dones,
                    loss_weight = gating
                )

                accelerator.backward(loss)
                optimizer.step()
                critic.target_model.update()

                dcfm_losses.append(l_dcfm.item())
                bcfm_losses.append(l_bcfm.item())
                sgen_losses.append(l_sgen.item())

        log_data = dict(cumulative_reward = episode_reward, epsilon = epsilon, episode = episode)

        if dcfm_losses:
            log_data.update(
                loss_dcfm = np.mean(dcfm_losses),
                loss_bcfm = np.mean(bcfm_losses),
                loss_state_gen = np.mean(sgen_losses)
            )

        if divisible_by(episode, record_every):
            mp4_files = glob.glob(os.path.join(video_folder, '*.mp4'))
            if mp4_files:
                log_data.update(video = wandb.Video(max(mp4_files, key = os.path.getctime), fps = 30, format = 'mp4'))

        if accelerator.is_main_process:
            wandb.log(log_data)

        if avg_reward > 200:
            print(f'\nSolved in {episode} episodes with reward {avg_reward:.2f}')
            break

    env.close()

    if accelerator.is_main_process:
        wandb.finish()
        np.save('rewards_value_flows.npy', np.array(all_rewards))

if __name__ == '__main__':
    fire.Fire(main)
