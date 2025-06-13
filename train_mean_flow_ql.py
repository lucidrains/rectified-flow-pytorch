
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
    'pred_q_value',
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

        self.actor = MLP(state_dim + num_actions + 2, actor_hidden_dim, num_actions) # naively concat time and integral start time -> mlp

        self.mean_flow_actor = MeanFlow(
            self.actor,
            data_shape = (num_actions,),
            accept_cond = True
        )

        self.critic = MLP(state_dim + num_actions, critic_hidden_dim, 1)

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
            next_states,
            is_boundaries,
        ) = zip(*memories)
        
        actions = [tensor(action) for action in actions]

        # convert values to torch tensors

        to_torch_tensor = lambda t: stack(t).to(device).detach()

        states = to_torch_tensor(states)
        next_states = to_torch_tensor(next_states)
        actions = to_torch_tensor(actions)

        # prepare dataloader for policy phase training

        learnable = tensor(learnable).to(device)
        data = (states, actions, pred_q_value, returns)
        data = tuple(t[learnable] for t in data)

        dataset = TensorDataset(*data)

        dl = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        # updating actor / critic

        for _ in range(self.epochs):
            for states, actions, rewards, next_states in enumerate(dl):

                # the flow q-learning proposed here https://seohong.me/projects/fql/ is now simplified

                next_actions = self.mean_flow_actor(cond = next_states)

                # learn critic

                state_action = cat((states, actions), dim = -1)
                next_state_action = cat((next_states, next_actions), dim = -1)

                pred_q = self.critic(state_action)
                target_q = rewards + self.discount_factor * self.ema_critic(next_state_action)
                
                critic_loss = F.mse_loss(pred_q, target_q)
                critic_loss.backward()

                self.opt_critic.step()
                self.opt_critic.zero_grad()

                # learn mean flow actor

                flow_loss = self.mean_flow(actions, cond = states)

                sampled_action = self.mean_flow_actor.sample(cond = states, requires_grad = True) # 1-step sample from mean flow paper, no more issue

                with torch.no_grad():
                    state_action = cat((state, sampled_action), dim = -1)
                    critique = self.critic(state_action)

                actor_loss = critique + flow_loss
                actor_loss.backward()

                self.opt_actor.step()
                self.opt_actor.zero_grad()

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
                actions = agent.mean_flow_actor.sample(cond = state)
                actions = rearrange(actions, '1 ... -> ...')

            next_state, reward, terminated, truncated, _ = env.step(actions.tolist())

            next_state = torch.from_numpy(next_state).to(device)

            reward = float(reward)

            pred_q_value = self.ema_critic(cat(state, actions))

            memory = Memory(True, state, action, reward, next_state, terminated)

            memories.append(memory)

            state = next_state

            # determine if truncating, either from environment or learning phase of the agent

            updating_agent = divisible_by(time, update_timesteps)
            done = terminated or truncated or updating_agent

            # updating of the agent

            if updating_agent:
                continue
                agent.learn(memories)
                num_policy_updates += 1
                memories.clear()

            # break if done

            if done:
                break

# main

if __name__ == '__main__':
    fire.Fire(main)
