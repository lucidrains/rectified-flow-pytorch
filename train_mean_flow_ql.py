
# along same veins as https://arxiv.org/abs/2502.02538
# but no more distillation, BC, and all that

from __future__ import annotations

import fire
from pathlib import Path
from shutil import rmtree
from functools import partial
from collections import namedtuple
from random import randrange

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, tensor, cat, stack
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Module, ModuleList
from torch.utils.data import TensorDataset, DataLoader

import gymnasium as gym

from assoc_scan import AssocScan

from ema_pytorch import EMA

from x_mlps_pytorch import MLP

from rectified_flow_pytorch.mean_flow import MeanFlow

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# memory tuple

Memory = namedtuple('Memory', [
    'learnable',
    'state',
    'action',
    'reward',
    'is_boundary'
])

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# agent

class Agent(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        reward_range: tuple[float, float],
        epochs,
        batch_size,
        lr,
        betas,
        lam,
        discount_factor,
        ema_decay,
    ):
        super().__init__()

        self.actor = MLP(state_dim, actor_hidden_dim, num_actions)

        self.wrapped_actor = MeanFlow(self.actor)

        self.critic = MLP(state_dim, critic_hidden_dim, num_actions)

        self.ema_actor = EMA(self.actor, beta = ema_decay, include_online_model = False)
        self.ema_critic = EMA(self.critic, beta = ema_decay, include_online_model = False)

        self.opt_actor = Adam(self.actor.parameters(), lr = lr, betas = betas)
        self.opt_critic = Adam(self.critic.parameters(), lr = lr, betas = betas)

        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

        # learning hparams

        self.batch_size = batch_size

        self.epochs = epochs

        self.discount_factor = discount_factor

    def learn(self, memories):

        # retrieve and prepare data from memory for training

        (
            learnable,
            states,
            actions,
            rewards,
            is_boundaries,
        ) = zip(*memories)
        
        actions = [tensor(action) for action in actions]
        masks = [(1. - float(is_boundary)) for is_boundary in is_boundaries]

        # calculate generalized advantage estimate

        returns = calc_returns(
            rewards = tensor(rewards).to(device),
            masks = tensor(masks).to(device),
            discount_factor = self.discount_factor,
            use_accelerated = False
        )

        # convert values to torch tensors

        to_torch_tensor = lambda t: stack(t).to(device).detach()

        states = to_torch_tensor(states)
        actions = to_torch_tensor(actions)

        # prepare dataloader for policy phase training

        learnable = tensor(learnable).to(device)
        data = (states, actions, returns)
        data = tuple(t[learnable] for t in data)

        dataset = TensorDataset(*data)

        dl = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        # policy phase training, similar to original PPO

        for _ in range(self.epochs):
            for i, (states, actions, returns) in enumerate(dl):
                pass

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50000,
    max_timesteps = 500,
    actor_hidden_dim = 64,
    critic_hidden_dim = 256,
    reward_range = (-100, 100),
    batch_size = 64,
    lr = 0.0008,
    betas = (0.9, 0.99),
    lam = 0.95,
    discount_factor = 0.99,
    ema_decay = 0.9,
    update_timesteps = 5000,
    epochs = 2,
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
        reward_range,
        epochs,
        batch_size,
        lr,
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

            with torch.no_grad():
                action = agent.actor.forward(state)

            next_state, reward, terminated, truncated, _ = env.step(action.tolist())

            next_state = torch.from_numpy(next_state).to(device)

            reward = float(reward)

            memory = Memory(True, state, action, reward, terminated)

            memories.append(memory)

            state = next_state

            # determine if truncating, either from environment or learning phase of the agent

            updating_agent = divisible_by(time, update_timesteps)
            done = terminated or truncated or updating_agent

            # take care of truncated by adding a non-learnable memory storing the next value for GAE

            if done and not terminated:
                bootstrap_value_memory = memory._replace(
                    state = state,
                    learnable = False,
                    is_boundary = True,
                )

                memories.append(bootstrap_value_memory)

            # updating of the agent

            if updating_agent:
                continue
                agent.learn(memories)
                num_policy_updates += 1
                memories.clear()

            # break if done

            if done:
                break

if __name__ == '__main__':
    fire.Fire(main)
