# along same veins as https://arxiv.org/abs/2502.02538
# but no more distillation and all that

# https://openreview.net/forum?id=mIeKe74W43

from __future__ import annotations

import fire
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from functools import partial
from collections import namedtuple
from random import randrange, random

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

    loss = F.mse_loss(x, target, reduction = 'none')

    less_than_zero = loss < 0
    weight = (tau - less_than_zero.float()) 

    return (weight * loss).mean()

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
        lam,
        discount_factor,
        ema_decay,
        flow_loss_weight = 0.25,
        noise_std_dev = 2.,
        update_critic_with_ema_every = 100_000,
        pessimism_strength = 0.05
    ):
        super().__init__()

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

        self.ema_critic = EMA(self.critic, beta = ema_decay, include_online_model = False, update_model_with_ema_every = update_critic_with_ema_every)

        self.opt_actor = Adam(self.actor.parameters(), lr = lr, weight_decay = weight_decay, betas = betas)
        self.opt_critic = Adam(self.critic.parameters(), lr = lr, weight_decay = weight_decay, betas = betas)

        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

        # learning hparams

        self.batch_size = batch_size

        self.epochs = epochs

        self.discount_factor = discount_factor

        self.flow_loss_weight = flow_loss_weight

        # how much below `tau` for expectile regression

        self.pessimism_strength = pessimism_strength

    def learn(self, memories):

        # retrieve and prepare data from memory for training

        data = zip(*memories)

        # convert values to torch tensors

        to_torch_tensor = lambda t: stack(t).to(device).detach()

        data = map(to_torch_tensor, data)

        # prepare dataloader for policy phase training

        dataset = TensorDataset(*data)

        dl = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        # updating actor / critic

        with tqdm(range(self.epochs)) as pbar:
            for states, actions, rewards, next_states, terminal in dl:

                self.opt_critic.zero_grad()

                # the flow q-learning proposed here https://seohong.me/projects/fql/ is now simplified

                noise = torch.randn_like(actions)

                next_actions = self.mean_flow_actor.sample(noise = noise, cond = next_states)

                # learn critic

                pred_q = self.critic(states, actions)
                target_q = rewards.float() + (~terminal).float() * self.discount_factor * self.ema_critic(next_states, next_actions)

                critic_loss = expectile_l2_loss(pred_q, target_q, tau = 0.5 - self.pessimism_strength)
                critic_loss.backward()

                self.opt_critic.step()

                pbar.set_description(f'critic: {critic_loss.item():.3f}')

                pbar.update(1)

        with tqdm(range(self.epochs)) as pbar:
            for states, actions, rewards, next_states, terminal in dl:

                # flow loss

                noise = torch.randn_like(actions)

                flow_loss = self.mean_flow_actor(actions, noise = noise, cond = states)

                # actor learning to maximize q value

                noise = torch.randn_like(actions)

                sampled_actions = self.mean_flow_actor.sample(cond = states, noise = noise, requires_grad = True) # 1-step sample from mean flow paper, no more issue

                q_value = self.critic(states, sampled_actions)

                # total actor loss

                actor_loss = -q_value.mean() + (flow_loss * self.flow_loss_weight)
                actor_loss.backward()

                self.opt_actor.step()
                self.opt_actor.zero_grad()

                pbar.set_description(f'actor flow: {flow_loss.item():.3f} | actor q value: {q_value.mean().item():.3f}')

                pbar.update(1)

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50000,
    max_timesteps = 500,
    actor_hidden_dim = 64,
    critic_hidden_dim = 128,
    max_reward = 100,
    batch_size = 64,
    prob_rand_action = 0.1,
    lr = 0.0008,
    weight_decay = 1e-3,
    betas = (0.9, 0.99),
    lam = 0.95,
    discount_factor = 0.99,
    ema_decay = 0.95,
    update_timesteps = 10000,
    epochs = 5,
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

    memories = []

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
        lam,
        discount_factor,
        ema_decay,
    ).to(device)

    time = 0
    num_policy_updates = 0

    for eps in tqdm(range(num_episodes), desc = 'episodes'):

        state, info = env.reset()
        state = torch.from_numpy(state).to(device)

        for timestep in range(max_timesteps):
            time += 1

            noise = agent.mean_flow_actor.get_noise()

            if prob_rand_action < random():
                actions = torch.rand((num_actions,), device = device) * 2 - 1.
            else:
                with torch.no_grad():
                    state_with_one_batch = rearrange(state, 'd -> 1 d')
                    actions = agent.mean_flow_actor.slow_sample(noise = noise, cond = state_with_one_batch, steps = actor_sample_steps_at_rollout)
                    actions = rearrange(actions, '1 ... -> ...')

            actions.clamp_(-1., 1.)
            next_state, reward, terminated, truncated, _ = env.step(actions.tolist())

            reward /= max_reward

            next_state = torch.from_numpy(next_state).to(device)

            memory = Memory(state, actions, tensor(reward), next_state, tensor(terminated))

            memories.append(memory)

            state = next_state

            # determine if truncating, either from environment or learning phase of the agent

            updating_agent = divisible_by(time, update_timesteps)
            done = terminated or truncated or updating_agent

            # updating of the agent

            if updating_agent:
                agent.learn(memories)
                num_policy_updates += 1
                memories.clear()

            # break if done

            if done:
                break

# main

if __name__ == '__main__':
    fire.Fire(main)
