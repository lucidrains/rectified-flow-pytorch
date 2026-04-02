# along same veins as https://arxiv.org/abs/2502.02538
# but no more distillation and all that

# https://openreview.net/forum?id=mIeKe74W43

from __future__ import annotations

import fire
import random
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from functools import partial
from collections import namedtuple, deque

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, tensor, cat, stack
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Module, ModuleList
from torch.utils.data import TensorDataset, DataLoader

import gymnasium as gym

from einops import rearrange, repeat, reduce, einsum, pack

from ema_pytorch import EMA

from x_mlps_pytorch import Feedforwards

from rectified_flow_pytorch.mean_flow import MeanFlow

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# memory tuple

Memory = namedtuple('Memory', [
    'state',
    'action',
    'reward',
    'next_state',
    'done'
])

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# expectile regression
# for expectile bellman proposed by https://arxiv.org/abs/2406.04081v1, which obviates need for twin critic for alleviating overestimation bias

def expectile_l2_loss(
    x,
    target,
    tau = 0.5  # 0.5 would be the classic l2 loss - less would weigh negative higher, and more would weigh positive higher
):
    assert 0 <= tau <= 1.

    if tau == 0.5:
        return F.mse_loss(x, target)

    diff = x - target

    weight = torch.where(diff < 0, tau, 1. - tau)

    return (weight * diff.square()).mean()

# agent

class Actor(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ff = Feedforwards(**kwargs)

    def forward(
        self,
        noised_data,
        times,
        integral_start_times,
        states
    ):
        noise_and_cond = cat((noised_data, states), dim = -1)

        times = rearrange(times, 'b -> b 1')
        integral_start_times = rearrange(integral_start_times, 'b -> b 1')

        actor_input = cat((noise_and_cond, times, integral_start_times), dim = -1)

        return self.ff(actor_input)

class Critic(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ff = Feedforwards(**kwargs)

    def forward(self, states, actions):
        states_actions = cat((states, actions), dim = -1)
        q_values = self.ff(states_actions)
        return rearrange(q_values, '... 1 -> ...')

class Agent(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        batch_size,
        lr,
        weight_decay,
        betas,
        discount_factor,
        ema_decay,
        discount_factor_short = 0.9,
        consistency_weight = 0.1,
        flow_loss_weight = 0.0025,
        noise_std_dev = 2.,
        update_critic_with_ema_every = 100_000,
        pessimism_strength = 0.05,
        oob_penalty_weight = 50.
    ):
        super().__init__()

        # inspired by Farebrother et al. https://arxiv.org/abs/2602.19634

        self.use_short_horizon = consistency_weight > 0.

        self.actor = Actor(
            dim = actor_hidden_dim,
            dim_in = state_dim + num_actions + 2,
            depth = 2,
            dim_out = num_actions,
            final_norm = True
        ) # naively concat time and integral start time -> mlp

        self.mean_flow_actor = MeanFlow(
            self.actor,
            data_shape = (num_actions,),
            accept_cond = True,
            noise_std_dev = noise_std_dev
        )

        self.critic = Critic(
            dim = critic_hidden_dim,
            depth = 2,
            dim_in = state_dim + num_actions,
            dim_out = 1
        )

        if self.use_short_horizon:
            self.critic_short = Critic(
                dim = critic_hidden_dim,
                depth = 2,
                dim_in = state_dim + num_actions,
                dim_out = 1
            )

        self.ema_critic = EMA(self.critic, beta = ema_decay, include_online_model = False, update_model_with_ema_every = update_critic_with_ema_every)

        self.opt_actor = Adam(self.actor.parameters(), lr = lr, weight_decay = weight_decay, betas = betas)
        self.opt_critic = Adam(self.critic.parameters(), lr = lr, weight_decay = weight_decay, betas = betas)
        
        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

        if self.use_short_horizon:
            self.ema_critic_short = EMA(self.critic_short, beta = ema_decay, include_online_model = False, update_model_with_ema_every = update_critic_with_ema_every)
            self.opt_critic_short = Adam(self.critic_short.parameters(), lr = lr, weight_decay = weight_decay, betas = betas)
            self.ema_critic_short.add_to_optimizer_post_step_hook(self.opt_critic_short)

        # learning hparams

        self.batch_size = batch_size

        self.epochs = epochs

        self.discount_factor = discount_factor
        self.discount_factor_short = discount_factor_short
        self.consistency_weight = consistency_weight

        self.flow_loss_weight = flow_loss_weight

        # how much below `tau` for expectile regression

        self.pessimism_strength = pessimism_strength

        self.oob_penalty_weight = oob_penalty_weight

    def learn(self, memories):

        to_torch_tensor = lambda t: stack(t).to(device).detach()

        data = map(to_torch_tensor, zip(*memories))
        dataset = TensorDataset(*data)
        dl = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        for _ in range(self.epochs):
            with tqdm(dl, leave = False) as pbar:
                for i, (states, actions, rewards, next_states, terminal) in enumerate(pbar):

                    # updating actor / critic

                    self.opt_critic.zero_grad()
                    if self.use_short_horizon:
                        self.opt_critic_short.zero_grad()

                    with torch.no_grad():
                        noise = torch.randn_like(actions)
                        next_actions = self.mean_flow_actor.sample(noise = noise, cond = next_states)
                        
                        target_q_all       = self.ema_critic(next_states, next_actions)
                        target_q       = rewards.float() + (~terminal).float() * self.discount_factor * target_q_all

                        if self.use_short_horizon:
                            target_q_short_all = self.ema_critic_short(next_states, next_actions)
                            target_q_short = rewards.float() + (~terminal).float() * self.discount_factor_short * target_q_short_all
                            
                            q_beta_target_sa = self.ema_critic_short(states, actions)
                            y_cons = q_beta_target_sa + self.discount_factor * (~terminal).float() * target_q_all - self.discount_factor_short * (~terminal).float() * target_q_short_all

                    pred_q = self.critic(states, actions)
                    critic_loss = expectile_l2_loss(pred_q, target_q, tau = 0.5 - self.pessimism_strength)

                    if self.use_short_horizon:
                        pred_q_short = self.critic_short(states, actions)
                        critic_short_loss = expectile_l2_loss(pred_q_short, target_q_short, tau = 0.5 - self.pessimism_strength)
                        
                        cons_loss = F.mse_loss(pred_q, y_cons)
                        critic_loss = critic_loss + self.consistency_weight * cons_loss

                    critic_loss.backward()
                    self.opt_critic.step()
                    
                    if self.use_short_horizon:
                        critic_short_loss.backward()
                        self.opt_critic_short.step()

                    # flow and actor loss

                    noise = torch.randn_like(actions)
                    flow_loss = self.mean_flow_actor(actions, noise = noise, cond = states)

                    noise = torch.randn_like(actions)
                    sampled_actions = self.mean_flow_actor.sample(cond = states, noise = noise, requires_grad = True)

                    q_value = self.critic(states, sampled_actions)
                    oob_penalty = F.relu(sampled_actions.abs() - 1.0).pow(2).mean()

                    actor_loss = -q_value.mean() + flow_loss * self.flow_loss_weight + oob_penalty * self.oob_penalty_weight

                    actor_loss.backward()
                    self.opt_actor.step()
                    self.opt_actor.zero_grad()

                    if divisible_by(i, 10):
                        if self.use_short_horizon:
                            pbar.set_description(f'critic: {critic_loss.item():.3f} | cons: {cons_loss.item():.3f} | flow: {flow_loss.item():.3f} | q: {q_value.mean().item():.3f}')
                        else:
                            pbar.set_description(f'critic: {critic_loss.item():.3f} | flow: {flow_loss.item():.3f} | q: {q_value.mean().item():.3f}')

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50000,
    max_timesteps = 500,
    max_memory_timesteps = 20_000,
    actor_hidden_dim = 64,
    critic_hidden_dim = 128,
    max_reward = 100,
    batch_size = 64,
    prob_rand_action = 0.1,
    lr = 0.0008,
    weight_decay = 1e-3,
    betas = (0.9, 0.99),
    discount_factor = 0.99,
    discount_factor_short = 0.9,
    consistency_weight = 0.1,
    flow_loss_weight = 0.0025,
    ema_decay = 0.95,
    update_timesteps = 1000,
    epochs = 1,
    actor_sample_steps_at_rollout = 4,
    render = True,
    render_every_eps = 250,
    clear_videos = True,
    video_folder = './lunar-recording',
):
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

    memories = deque(maxlen = max_memory_timesteps)
    ep_rewards_deque = deque(maxlen = 20)

    agent = Agent(
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        batch_size,
        lr,
        weight_decay,
        betas,
        discount_factor,
        ema_decay,
        discount_factor_short = discount_factor_short,
        consistency_weight = consistency_weight,
        flow_loss_weight = flow_loss_weight,
    ).to(device)

    time = 0
    num_policy_updates = 0

    pbar = tqdm(range(num_episodes), desc = 'episodes')
    for eps in pbar:

        ep_reward = 0.
        state, info = env.reset()
        state = torch.from_numpy(state).to(device)

        for timestep in range(max_timesteps):
            time += 1

            noise = agent.mean_flow_actor.get_noise()

            if random.random() < prob_rand_action:
                actions = torch.rand((num_actions,), device = device) * 2 - 1.
            else:
                with torch.no_grad():
                    state_with_one_batch = rearrange(state, 'd -> 1 d')
                    actions = agent.mean_flow_actor.slow_sample(noise = noise, cond = state_with_one_batch, steps = actor_sample_steps_at_rollout)
                    actions = rearrange(actions, '1 ... -> ...')

            actions.clamp_(-1., 1.)
            next_state, reward, terminated, truncated, _ = env.step(actions.tolist())

            ep_reward += reward
            reward /= max_reward

            next_state = torch.from_numpy(next_state).to(device)

            memory = Memory(state, actions, tensor(reward), next_state, tensor(terminated))

            memories.append(memory)
            state = next_state

            updating_agent = divisible_by(time, update_timesteps) and len(memories) >= batch_size
            done = terminated or truncated

            if updating_agent:
                agent.learn(memories)
                num_policy_updates += 1

            if done:
                break

        ep_rewards_deque.append(ep_reward)
        avg_reward = sum(ep_rewards_deque) / len(ep_rewards_deque)
        pbar.set_postfix(avg_reward=f"{avg_reward:.2f}")

# main

if __name__ == '__main__':
    fire.Fire(main)
