# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rectified-flow-pytorch[examples_ql]",
#     "memmap-replay-buffer>=0.1.1",
#     "wandb",
# ]
# ///

# fql   - https://arxiv.org/abs/2502.02538
# t-sac - https://arxiv.org/abs/2503.03660
# aac   - https://arxiv.org/abs/2605.10044

from __future__ import annotations

import fire
import random
from shutil import rmtree
from collections import deque

from accelerate import Accelerator

import wandb
from tqdm import tqdm

import torch
from torch import nn, tensor, cat
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Module

import gymnasium as gym

from einops import rearrange, repeat

from ema_pytorch import EMA

from x_mlps_pytorch import Feedforwards

from memmap_replay_buffer import ReplayBuffer
from torch_einops_utils import pack_with_inverse, mask_after, masked_mean, lens_to_mask
from assoc_scan import AssocScan
from x_transformers import Decoder

from rectified_flow_pytorch.nano_flow import NanoFlow

# helpers

def divisible_by(num, den):
    return (num % den) == 0

def bernoulli(p):
    return random.random() < p

def sample_categorical(logits, low = 1):
    probs = F.softmax(logits, dim = -1)
    sampled = torch.multinomial(probs, 1)
    return (sampled + 1).clamp(min = low).item()

# agent networks

class TransformerActor(Module):
    """ transformer decoder based actor - produces chunks of actions over time """

    def __init__(
        self,
        *,
        dim_state,
        num_cont_actions,
        dim_hidden = 256,
        depth = 3,
        heads = 4,
        max_chunk_size = 16
    ):
        super().__init__()
        self.num_cont_actions = num_cont_actions

        self.state_proj = nn.Linear(dim_state, dim_hidden)
        self.noise_proj = nn.Linear(num_cont_actions, dim_hidden)

        self.pos_emb = nn.Embedding(max_chunk_size, dim_hidden)

        self.decoder = Decoder(
            dim = dim_hidden,
            depth = depth,
            heads = heads
        )

        self.to_actions = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, num_cont_actions)
        )

    def forward(self, states, noise):
        # noise: (b, num_actions) for single action or (b, seq, num_actions) for chunk

        state_tokens = rearrange(self.state_proj(states), 'b d -> b 1 d')

        if noise.ndim == 2:
            noise = rearrange(noise, 'b d -> b 1 d')

        seq = noise.shape[1]
        noise_tokens = self.noise_proj(noise)

        positions = torch.arange(seq, device = noise.device)
        noise_tokens = noise_tokens + self.pos_emb(positions)

        tokens = cat((state_tokens, noise_tokens), dim = 1)
        out = self.decoder(tokens)

        actions = self.to_actions(out[:, 1:])

        if seq == 1:
            return rearrange(actions, 'b 1 d -> b d')

        return actions

class FlowActor(Module):
    """ mlp velocity field for behavior cloning flow """

    def __init__(self, **kwargs):
        super().__init__()
        self.ff = Feedforwards(**kwargs)

    def forward(self, noised_data, times, cond):
        b = noised_data.shape[0]

        if times.shape[0] == 1 and b > 1:
            times = repeat(times, '1 -> b 1', b = b)
        else:
            times = rearrange(times, 'b -> b 1')

        return self.ff((noised_data, cond, times))

class TransformerCritic(Module):
    """ transformer based critic - Dong Tian et al. https://arxiv.org/abs/2503.03660 """

    def __init__(
        self,
        *,
        dim_state,
        num_cont_actions,
        dim_hidden = 256,
        depth = 3,
        heads = 8,
        dim_out = 1,
        max_seq_len = 16
    ):
        super().__init__()
        self.dim_out = dim_out
        self.max_seq_len = max_seq_len

        self.state_proj = nn.Linear(dim_state, dim_hidden)
        self.action_proj = nn.Linear(num_cont_actions, dim_hidden)

        self.pos_emb = nn.Embedding(max_seq_len, dim_hidden)

        self.decoder = Decoder(
            dim = dim_hidden,
            depth = depth,
            heads = heads
        )

        self.to_values = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, state, cont_actions):
        cont_actions, inverse_seq = pack_with_inverse(cont_actions, 'b * d')

        b, seq, _ = cont_actions.shape
        assert seq <= self.max_seq_len, f'sequence length {seq} exceeds max_seq_len {self.max_seq_len}'

        state_tokens = rearrange(self.state_proj(state), 'b d -> b 1 d')
        action_tokens = self.action_proj(cont_actions)

        positions = torch.arange(seq, device = cont_actions.device)
        action_tokens = action_tokens + self.pos_emb(positions)

        tokens = cat((state_tokens, action_tokens), dim = 1)
        out = self.decoder(tokens)

        values = self.to_values(out[:, 1:])
        values = inverse_seq(values)

        if self.dim_out == 1:
            values = rearrange(values, '... 1 -> ...')

        return values

# agent

class Agent(Module):
    def __init__(
        self,
        accelerator,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        batch_size,
        lr,
        weight_decay,
        betas,
        discount_factor,
        ema_decay,
        action_chunk_size = 4,
        bc_distill_weight = 1.,
        pessimism_strength = 0.05,
        oob_penalty_weight = 1.,
        actor_sample_steps_at_rollout = 10,
        max_grad_norm = 1.
    ):
        super().__init__()

        self.accelerator = accelerator
        self.num_actions = num_actions

        # bc flow actor

        flow_actor = FlowActor(
            dim = actor_hidden_dim,
            dim_in = state_dim + num_actions + 1,
            depth = 2,
            dim_out = num_actions,
            final_norm = True
        )

        self.bc_flow = NanoFlow(
            flow_actor,
            data_shape = (num_actions,),
            times_cond_kwarg = 'times'
        )

        # one step actor (transformer decoder)

        self.one_step_actor = TransformerActor(
            dim_state = state_dim,
            num_cont_actions = num_actions,
            dim_hidden = actor_hidden_dim,
        )

        # critic

        self.critic = TransformerCritic(
            dim_state = state_dim,
            num_cont_actions = num_actions,
            dim_hidden = critic_hidden_dim
        )

        self.ema_critic = EMA(self.critic, beta = ema_decay, include_online_model = False, update_every = 1, update_after_step = 0)

        # n-step returns via associative scan

        self.assoc_scan = AssocScan(reverse = True)

        # optimizers

        self.opt_bc_flow = Adam(self.bc_flow.parameters(), lr = lr, betas = betas)
        self.opt_one_step_actor = Adam(self.one_step_actor.parameters(), lr = lr, betas = betas)
        self.opt_critic = Adam(self.critic.parameters(), lr = lr, weight_decay = weight_decay, betas = betas)

        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

        self.bc_flow, self.opt_bc_flow = self.accelerator.prepare(self.bc_flow, self.opt_bc_flow)
        self.one_step_actor, self.opt_one_step_actor = self.accelerator.prepare(self.one_step_actor, self.opt_one_step_actor)
        self.critic, self.opt_critic = self.accelerator.prepare(self.critic, self.opt_critic)

        # hparams

        self.batch_size = batch_size
        self.action_chunk_size = action_chunk_size
        self.discount_factor = discount_factor
        self.max_grad_norm = max_grad_norm

        self.bc_distill_weight = bc_distill_weight
        self.pessimism_strength = pessimism_strength
        self.oob_penalty_weight = oob_penalty_weight
        self.actor_sample_steps_at_rollout = actor_sample_steps_at_rollout

    def learn(self, replay_buffer, num_updates = 1):

        dl = replay_buffer.dataloader(
            batch_size = self.batch_size,
            n_steps = self.action_chunk_size,
            next_fields = ('state',),
            sequence_fields = ('state', 'action', 'reward', 'terminal'),
            shuffle = True,
            device = self.accelerator.device,
            to_named_tuple = ('next_state', 'seq_state', 'seq_action', 'seq_reward', 'seq_terminal', 'n_step_lens')
        )

        dl_iter = iter(dl)

        with tqdm(range(num_updates), leave = False) as pbar:
            for i in pbar:
                try:
                    batch = next(dl_iter)
                except StopIteration:
                    dl_iter = iter(dl)
                    batch = next(dl_iter)

                states = batch.seq_state[:, 0]
                seq_states = batch.seq_state
                next_states = batch.next_state
                actions = batch.seq_action
                rewards = batch.seq_reward
                terminal = batch.seq_terminal
                n_step_lens = batch.n_step_lens

                b, seq_len = rewards.shape

                # loss mask from terminal + padding

                loss_mask = mask_after(terminal, True) & lens_to_mask(n_step_lens, seq_len)

                # critic loss - n-step returns via associative scan

                self.opt_critic.zero_grad()

                with torch.no_grad():
                    noise = torch.randn_like(actions[:, 0])
                    next_actions = self.one_step_actor(next_states, noise)

                    next_actions_seq = rearrange(next_actions, 'b d -> b 1 d')
                    target_q_ema = rearrange(self.ema_critic(next_states, next_actions_seq), 'b 1 -> b')
                    target_q_online = rearrange(self.critic(next_states, next_actions_seq), 'b 1 -> b')
                    target_q_all = torch.min(target_q_ema, target_q_online)

                    inputs = cat((rewards, torch.zeros_like(rewards[:, :1])), dim = 1)
                    batch_indices = torch.arange(b, device = rewards.device)

                    inputs[batch_indices, n_step_lens] = target_q_all

                    gates = torch.full_like(inputs, self.discount_factor)
                    gates[batch_indices, n_step_lens] = 0.
                    gates[:, :-1] = gates[:, :-1] * (~terminal).float()

                    target_q = self.assoc_scan(gates, inputs)[:, :-1]

                pred_q = self.critic(states, actions)

                diff = pred_q - target_q
                tau = 0.5 - self.pessimism_strength
                weight = torch.where(diff < 0, tau, 1. - tau)
                critic_loss = masked_mean(weight * diff.square(), loss_mask)

                self.accelerator.backward(critic_loss)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()

                # flatten loss mask for indexing

                loss_mask_flat = rearrange(loss_mask, 'b s -> (b s)')

                valid_states = rearrange(seq_states, 'b s d -> (b s) d')[loss_mask_flat]
                valid_actions = rearrange(actions, 'b s d -> (b s) d')[loss_mask_flat]

                # flow bc loss - filter to valid timesteps

                self.opt_bc_flow.zero_grad()

                noise = torch.randn_like(valid_actions)
                flow_loss = self.bc_flow(valid_actions, noise = noise, cond = valid_states)

                self.accelerator.backward(flow_loss)
                self.opt_bc_flow.step()

                # actor distillation + q loss (chunk-based)

                self.opt_one_step_actor.zero_grad()

                with torch.no_grad():
                    valid_bc_actions, valid_noise = self.bc_flow.sample(cond = valid_states, batch_size = valid_states.shape[0], steps = self.actor_sample_steps_at_rollout, return_noise = True)

                    bc_actions_seq = torch.zeros_like(actions)
                    bc_actions_seq_flat = rearrange(bc_actions_seq, 'b s d -> (b s) d')
                    bc_actions_seq_flat[loss_mask_flat] = valid_bc_actions
                    bc_actions_seq = rearrange(bc_actions_seq_flat, '(b s) d -> b s d', b = b)

                    noise_seq = torch.randn_like(actions)
                    noise_seq_flat = rearrange(noise_seq, 'b s d -> (b s) d')
                    noise_seq_flat[loss_mask_flat] = valid_noise
                    noise_seq = rearrange(noise_seq_flat, '(b s) d -> b s d', b = b)

                one_step_actions_seq = self.one_step_actor(states, noise_seq)

                q_value_seq = self.critic(states, one_step_actions_seq)
                q_values = q_value_seq[loss_mask]

                distill_loss = masked_mean((one_step_actions_seq - bc_actions_seq).pow(2).mean(dim = -1), loss_mask)
                oob_penalty = masked_mean(F.relu(one_step_actions_seq.abs() - 1.).pow(2).mean(dim = -1), loss_mask)

                actor_loss = -q_values.mean() + distill_loss * self.bc_distill_weight + oob_penalty * self.oob_penalty_weight

                self.accelerator.backward(actor_loss)
                nn.utils.clip_grad_norm_(self.one_step_actor.parameters(), self.max_grad_norm)
                self.opt_one_step_actor.step()

                if divisible_by(i, 10):
                    pbar.set_description(f'critic: {critic_loss.item():.3f} | flow: {flow_loss.item():.3f} | distill: {distill_loss.item():.3f} | q: {q_values.mean().item():.3f}')

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 5000,
    max_timesteps = 500,
    max_memory_episodes = 100,
    actor_hidden_dim = 64,
    critic_hidden_dim = 128,
    max_reward = 100,
    batch_size = 256,
    warmup_steps = 1000,
    prob_rand_action = 0.1,
    lr = 3e-4,
    weight_decay = 0.,
    betas = (0.9, 0.999),
    discount_factor = 0.99,
    bc_distill_weight = 1.,
    oob_penalty_weight = 1.,
    pessimism_strength = 0.05,
    ema_decay = 0.995,
    action_chunk_size = 4,
    dynamic_chunk = True,
    dynamic_chunk_eps = 0.05,
    dynamic_chunk_min_len = 1,
    update_every_episodes = 25,
    updates_per_train_step = 500,
    max_grad_norm = 1.,
    rollout_cpu = True,
    actor_sample_steps_at_rollout = 10,
    render = True,
    render_every_eps = 250,
    clear_videos = True,
    video_folder = './lunar-recording',
):
    accelerator = Accelerator()
    device = accelerator.device
    rollout_device = torch.device('cpu') if rollout_cpu else device

    if accelerator.is_main_process:
        wandb.init(project = 'flow-q-learning', mode = 'online')

    env = gym.make(
        env_name,
        render_mode = 'rgb_array',
        continuous = True
    )

    if render:
        if clear_videos:
            rmtree(video_folder, ignore_errors = True)

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = video_folder,
            name_prefix = 'lunar-video',
            episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
            disable_logger = True
        )

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    fields = dict(
        state = ('float', (state_dim,)),
        action = ('float', (num_actions,)),
        reward = 'float',
        terminal = 'bool',
        next_state = ('float', (state_dim,))
    )

    replay_buffer = ReplayBuffer(
        './replay_buffer',
        max_episodes = max_memory_episodes,
        max_timesteps = max_timesteps,
        fields = fields,
        flush_every_store_step = max_timesteps,
        circular = True
    )

    ep_rewards_deque = deque(maxlen = 20)
    ep_chunk_lengths_deque = deque(maxlen = 20)

    agent = Agent(
        accelerator,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        batch_size,
        lr,
        weight_decay,
        betas,
        discount_factor,
        ema_decay,
        action_chunk_size = action_chunk_size,
        bc_distill_weight = bc_distill_weight,
        pessimism_strength = pessimism_strength,
        oob_penalty_weight = oob_penalty_weight,
        actor_sample_steps_at_rollout = actor_sample_steps_at_rollout,
        max_grad_norm = max_grad_norm
    ).to(device)

    if rollout_cpu:
        agent.one_step_actor.to('cpu')

    time = 0
    action_queue = deque()

    pbar = tqdm(range(num_episodes), desc = 'episodes')

    for eps in pbar:

        ep_reward = 0.
        ep_sampled_chunk_lengths = []
        state, info = env.reset()
        state = torch.from_numpy(state).to(rollout_device)

        action_queue.clear()

        with replay_buffer.one_episode():
            for timestep in range(max_timesteps):
                time += 1

                if len(action_queue) > 0:
                    actions = action_queue.popleft()

                elif bernoulli(prob_rand_action):
                    actions = torch.rand((num_actions,), device = rollout_device) * 2 - 1.

                else:
                    with torch.no_grad():
                        state_b = rearrange(state, 'd -> 1 d')
                        noise = torch.randn((1, action_chunk_size, num_actions), device = rollout_device)
                        action_chunk = agent.one_step_actor(state_b, noise)

                        # adaptive action chunking - Shin et al. https://arxiv.org/abs/2605.10044

                        if dynamic_chunk:
                            q_values = agent.critic(state_b.to(device), action_chunk.to(device))
                            q_values = rearrange(q_values, '1 s -> s')

                            if bernoulli(dynamic_chunk_eps):
                                chunk_len = random.randint(dynamic_chunk_min_len, action_chunk_size)
                            else:
                                chunk_len = sample_categorical(q_values, low = dynamic_chunk_min_len)
                                ep_sampled_chunk_lengths.append(chunk_len)

                            action_chunk = action_chunk[:, :chunk_len]

                        action_chunk = rearrange(action_chunk, '1 s d -> s d').to(rollout_device)

                        for a in action_chunk:
                            action_queue.append(a)

                        actions = action_queue.popleft()

                actions.clamp_(-1., 1.)
                next_state, reward, terminated, truncated, _ = env.step(actions.tolist())

                ep_reward += reward
                reward /= max_reward

                next_state = torch.from_numpy(next_state).to(rollout_device)

                replay_buffer.store(
                    state = state,
                    action = actions,
                    reward = tensor(reward),
                    terminal = tensor(terminated),
                    next_state = next_state
                )

                state = next_state

                done = terminated or truncated

                if done:
                    break

        updating_agent = divisible_by(eps + 1, update_every_episodes) and replay_buffer.num_episodes > 0 and time > warmup_steps

        if updating_agent:
            if rollout_cpu:
                agent.one_step_actor.to(device)

            agent.learn(replay_buffer, num_updates = updates_per_train_step)

            if rollout_cpu:
                agent.one_step_actor.to('cpu')

        ep_rewards_deque.append(ep_reward)
        avg_reward = sum(ep_rewards_deque) / len(ep_rewards_deque)

        postfix_kwargs = dict(avg_reward = f"{avg_reward:.2f}")
        log_dict = dict(avg_reward = avg_reward, episode = eps)

        if len(ep_sampled_chunk_lengths) > 0:
            ep_chunk_lengths_deque.append(sum(ep_sampled_chunk_lengths) / len(ep_sampled_chunk_lengths))

        if dynamic_chunk and len(ep_chunk_lengths_deque) > 0:
            avg_chunk_len = sum(ep_chunk_lengths_deque) / len(ep_chunk_lengths_deque)
            postfix_kwargs.update(avg_chunk_len = f"{avg_chunk_len:.2f}")
            log_dict.update(avg_chunk_len = avg_chunk_len)

        pbar.set_postfix(**postfix_kwargs)

        if accelerator.is_main_process:
            wandb.log(log_dict)

# main

if __name__ == '__main__':
    fire.Fire(main)
