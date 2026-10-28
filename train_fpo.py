# /// script
# dependencies = [
#   "adam-atan2-pytorch",
#   "einops",
#   "einx",
#   "ema-pytorch",
#   "fire",
#   "gymnasium[box2d]",
#   "hl-gauss-pytorch",
#   "hyper-connections",
#   "moviepy",
#   "numpy",
#   "rectified-flow-pytorch",
#   "torch",
#   "tqdm",
#   "x-ppo"
# ]
# ///

# Flow Policy Optimization (FPO)
# McAllister et al. https://arxiv.org/abs/2507.21053

# Flow Policy Gradients for Legged Robots (FPO++)
# Yi et al. https://openreview.net/forum?id=BA6n0nmagi

# Flow-DPPO - Divergence Proximal Policy Optimization for Flow Matching Models
# Ping et al. https://arxiv.org/abs/2606.11025

# ReFPO - Reflow Regularization for Flow Matching Policy Gradients
# Wang et al. https://arxiv.org/abs/2606.21086

from __future__ import annotations

import fire
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from functools import partial
from collections import deque, namedtuple
from random import randrange

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, tensor, is_tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Categorical
from torch.utils._pytree import tree_map

import einx
from einops import reduce, repeat, einsum, rearrange, pack

from ema_pytorch import EMA

from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

from hl_gauss_pytorch import HLGaussLoss

from hyper_connections import HyperConnections

from x_ppo import calc_gae, hpo_actor_loss, ppo_actor_loss

from rectified_flow_pytorch.nano_flow import NanoFlow

import gymnasium as gym

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# memory tuple

Memory = namedtuple('Memory', [
    'learnable',
    'state',
    'action',
    'noise',
    'past_action',
    'reward',
    'is_boundary',
    'value',
])

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

def is_between(mid, lo, hi):
    return (lo < mid) & (mid < hi)

def add_batch(t):
    return rearrange(t, '... -> 1 ...')

def remove_batch(t):
    return rearrange(t, '1 ... -> ...')

# SimBa - Kaist + SonyAI

class ReluSquared(Module):
    def forward(self, x):
        return x.sign() * F.relu(x) ** 2

class SimBa(Module):

    def __init__(
        self,
        dim,
        dim_hidden = None,
        depth = 3,
        dropout = 0.,
        expansion_factor = 2,
        num_residual_streams = 4
    ):
        super().__init__()
        """
        following the design of SimBa https://arxiv.org/abs/2410.09754v1
        """

        self.num_residual_streams = num_residual_streams

        dim_hidden = default(dim_hidden, dim * expansion_factor)

        layers = []

        self.proj_in = nn.Linear(dim, dim_hidden)

        dim_inner = dim_hidden * expansion_factor

        # hyper connections

        init_hyper_conn, self.expand_stream, self.reduce_stream = HyperConnections.get_init_and_expand_reduce_stream_functions(1, num_fracs = num_residual_streams, disable = num_residual_streams == 1)

        for ind in range(depth):

            layer = nn.Sequential(
                nn.RMSNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_inner),
                ReluSquared(),
                nn.Linear(dim_inner, dim_hidden),
                nn.Dropout(dropout),
            )

            layer = init_hyper_conn(dim = dim_hidden, layer_index = ind, branch = layer)
            layers.append(layer)

        # final layer out

        self.layers = ModuleList(layers)

        self.final_norm = nn.RMSNorm(dim_hidden)

    def forward(self, x):
        no_batch = x.ndim == 1

        if no_batch:
            x = rearrange(x, '... -> 1 ...')

        x = self.proj_in(x)

        x = self.expand_stream(x)

        for layer in self.layers:
            x = layer(x)

        x = self.reduce_stream(x)

        out = self.final_norm(x)

        if no_batch:
            out = rearrange(out, '1 ... -> ...')

        return out

# networks

class RandomFourierEmbed(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        assert divisible_by(dim, 2)
        self.register_buffer('weights', torch.randn(dim // 2))

    def forward(self, x):
        freqs = einx.multiply('i, j -> i j', x, self.weights) * 2 * torch.pi
        fourier_embed, _ = pack((x, freqs.sin(), freqs.cos()), 'b *')
        return fourier_embed

class Actor(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        num_actions,
        dim_time = 16,
        mlp_depth = 3,
        dropout = 0.1,
    ):
        super().__init__()

        self.to_time_emb = nn.Sequential(
            RandomFourierEmbed(dim_time),
            nn.Linear(dim_time + 1, dim_time),
            nn.SiLU()
        )

        self.net = SimBa(
            state_dim + dim_time + num_actions,
            dim_hidden = hidden_dim * 2,
            depth = mlp_depth,
            dropout = dropout
        )

        self.to_flow = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            ReluSquared(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, noised_actions, *, state, time):
        time_emb = self.to_time_emb(time)

        inp = cat((noised_actions, state, time_emb), dim = -1)
        hidden = self.net(inp)
        return self.to_flow(hidden)

class Critic(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        hidden_dim,
        dim_pred = 1,
        mlp_depth = 6, # recent paper has findings that show scaling critic is more important than scaling actor
        dropout = 0.1,
    ):
        super().__init__()

        self.net = SimBa(
            state_dim + num_actions,
            dim_hidden = hidden_dim,
            depth = mlp_depth,
            dropout = dropout
        )

        self.value_head = nn.Linear(hidden_dim, dim_pred)

    def forward(self, x):
        hidden = self.net(x)
        value = self.value_head(hidden)
        return value

# agent

class PPO(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        critic_pred_num_bins,
        epochs,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        cautious_factor,
        eps_clip,
        ema_decay,
        advantage_offset_constant = 0.35,
        use_hpo = True,
        ema_kwargs: dict = dict(
            update_model_with_ema_every = 1000
        ),
        reward_range = (-300., 300.),
        num_noise_monte_carlo = 4,
        loss_diff_clamp_value = 4.,
        vel_kl_threshold = 0.08,
        reflow_weight = 0.08,
        save_path = './fpo.pt'
    ):
        super().__init__()

        actor_network = Actor(state_dim, actor_hidden_dim, num_actions)
        self.actor = NanoFlow(actor_network, data_shape = (num_actions,), times_cond_kwarg = 'time', loss_fn = F.huber_loss)

        self.critic = Critic(state_dim, num_actions, critic_hidden_dim, dim_pred = critic_pred_num_bins)

        # num noise monte carlo

        self.num_noise_monte_carlo = num_noise_monte_carlo

        # clamps and flow-dppo / refpo hparams

        self.use_hpo = use_hpo
        self.loss_diff_clamp_value = loss_diff_clamp_value
        self.vel_kl_threshold = vel_kl_threshold
        self.reflow_weight = reflow_weight

        # https://arxiv.org/abs/2403.03950

        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value = reward_range[0],
            max_value = reward_range[1],
            num_bins = critic_pred_num_bins,
            clamp_to_range = True
        )

        self.ema_actor = EMA(self.actor, beta = ema_decay, include_online_model = False, forward_method_names = ('sample',), **ema_kwargs)
        self.ema_critic = EMA(self.critic, beta = ema_decay, include_online_model = False, **ema_kwargs)

        self.opt_actor = AdoptAtan2(self.actor.parameters(), lr = lr, betas = betas, cautious_factor = cautious_factor)
        self.opt_critic = AdoptAtan2(self.critic.parameters(), lr = lr, betas = betas, cautious_factor = cautious_factor)

        self.ema_actor.add_to_optimizer_post_step_hook(self.opt_actor)
        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

        # learning hparams

        self.minibatch_size = minibatch_size

        self.epochs = epochs

        self.lam = lam
        self.gamma = gamma

        self.eps_clip = eps_clip

        self.advantage_offset_constant = advantage_offset_constant

        self.save_path = Path(save_path)

    def save(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, str(self.save_path))

    def load(self):
        if not self.save_path.exists():
            return

        data = torch.load(str(self.save_path), weights_only = True)

        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])

    def learn(self, memories):
        eps_clip = self.eps_clip
        hl_gauss = self.critic_hl_gauss_loss

        # retrieve and prepare data from memory for training

        (
            learnable,
            states,
            action,
            rollout_noise,
            past_action,
            rewards,
            is_boundaries,
            values,
        ) = zip(*memories)

        masks = [(1. - float(is_boundary)) for is_boundary in is_boundaries]

        # calculate generalized advantage estimate

        scalar_values = hl_gauss(stack(values))

        with torch.no_grad():
            calc_gae_from_values = partial(calc_gae,
                rewards = tensor(rewards).to(device),
                masks = tensor(masks).to(device),
                lam = self.lam,
                gamma = self.gamma,
                use_accelerated = False
            )

            returns = calc_gae_from_values(values = scalar_values)

        # convert values to torch tensors

        to_torch_tensor = lambda t: stack(t).to(device).detach()

        states = to_torch_tensor(states)
        action = to_torch_tensor(action)
        rollout_noise = to_torch_tensor(rollout_noise)
        past_action = to_torch_tensor(past_action)
        old_values = to_torch_tensor(values)

        # deepcopy the actor for the reference actor

        old_actor = deepcopy(self.actor)
        old_actor.eval()

        # prepare dataloader for policy phase training

        learnable = tensor(learnable).to(device)
        data = (states, action, rollout_noise, past_action, returns, old_values)
        data = tuple(t[learnable] for t in data)

        dataset = TensorDataset(*data)

        dl = DataLoader(dataset, batch_size = self.minibatch_size, shuffle = True)

        # policy phase training, similar to original PPO

        n_mc = self.num_noise_monte_carlo
        clamp_value = self.loss_diff_clamp_value

        for _ in tqdm(range(self.epochs)):
            for i, (states, action, rollout_noise, past_action, returns, old_values) in enumerate(dl):

                batch = action.shape[0]

                times = torch.rand((batch * n_mc,), device = action.device)

                expanded_states = repeat(states, 'b ... -> (b n) ...', n = n_mc)
                expanded_action = repeat(action, 'b ... -> (b n) ...', n = n_mc)

                noise = torch.randn_like(expanded_action)

                actor_forward_kwargs = dict(state = expanded_states, noise = noise, times = times, loss_reduction = 'none', return_pred_flow = True)

                loss, pred_flow = self.actor(expanded_action, **actor_forward_kwargs)

                with torch.no_grad():
                    old_loss, old_pred_flow = old_actor(expanded_action, **actor_forward_kwargs)

                scalar_old_values = hl_gauss(old_values)

                # calculate advantages

                raw_advantages = returns - scalar_old_values.detach()
                advantages = normalize(raw_advantages) + self.advantage_offset_constant
                expanded_advantages = repeat(advantages, 'b -> (b n)', n = n_mc)

                # flow-dppo - Ping et al. https://arxiv.org/abs/2606.11025

                vel_kl = 0.5 * (pred_flow - old_pred_flow.detach()).square().mean(dim = -1)
                log_diff = (old_loss.detach() - loss).sum(dim = -1)

                moving_away = ((expanded_advantages > 0.) & (log_diff > 0.)) | ((expanded_advantages < 0.) & (log_diff < 0.))
                keep_mask = ~(moving_away & (vel_kl > self.vel_kl_threshold))

                # actor loss - Yi et al. https://openreview.net/forum?id=BA6n0nmagi

                actor_loss_fn = hpo_actor_loss if self.use_hpo else partial(ppo_actor_loss, dual_clip = True)

                policy_loss = actor_loss_fn(
                    action_log_probs = -loss,
                    old_action_log_probs = -old_loss.detach(),
                    advantages = expanded_advantages,
                    mask = keep_mask,
                    eps_clip = eps_clip,
                    log_ratio_clamp = (-clamp_value, clamp_value)
                ).mean()

                # refpo - Wang et al. https://arxiv.org/abs/2606.21086

                if self.reflow_weight > 0.:
                    reflow_times = torch.rand((batch,), device = action.device)
                    reflow_loss = self.actor(action, state = states, noise = rollout_noise, times = reflow_times, loss_reduction = 'none')
                    pos_gate = F.softplus(advantages).detach()
                    policy_loss = policy_loss + self.reflow_weight * (reflow_loss.sum(dim = -1) * pos_gate).mean()

                policy_loss.backward()
                self.opt_actor.step()
                self.opt_actor.zero_grad()

                # calculate clipped value loss and update value network separate from policy network

                values = self.critic(cat((states, past_action), dim = -1))
                critic_loss = hl_gauss(values, returns).mean()

                critic_loss.backward()
                self.opt_critic.step()
                self.opt_critic.zero_grad()

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50000,
    max_timesteps = 500,
    actor_hidden_dim = 32,
    actor_flow_timesteps = 16,
    critic_hidden_dim = 64,
    critic_pred_num_bins = 500,
    minibatch_size = 64,
    lr = 0.0003,
    betas = (0.9, 0.99),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.05,
    cautious_factor = 0.1,
    ema_decay = 0.9,
    update_timesteps = 5000,
    memory_buffer_size = 10_000,
    advantage_offset_constant = 0.35,
    epochs = 4,
    seed = None,
    render = True,
    render_every_eps = 100,
    save_every = 1000,
    clear_videos = True,
    video_folder = './lunar-recording',
    load = False
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

    memories = deque([], memory_buffer_size)

    agent = PPO(
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        critic_pred_num_bins,
        epochs,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        cautious_factor,
        eps_clip,
        ema_decay,
        advantage_offset_constant
    ).to(device)

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    num_policy_updates = 0
    all_rewards = []

    for eps in tqdm(range(num_episodes), desc = 'episodes'):

        state, _ = env.reset(seed = seed)
        state = torch.from_numpy(state).to(device)

        cum_rewards = 0.
        past_action = torch.zeros((num_actions,), device = device)

        for i in range(max_timesteps):
            is_last = i == (max_timesteps - 1)
            time += 1

            actor_state = add_batch(state)
            action, noise = agent.ema_actor.sample(actor_flow_timesteps, state = actor_state, return_noise = True)
            action, noise = map(remove_batch, (action, noise))

            value = agent.ema_critic.forward_eval(cat((state, past_action)))

            action_to_env = action.cpu().numpy()

            next_state, reward, terminated, truncated, _ = env.step(action_to_env)

            action_out_of_bound_penalty = (action.abs() - 1.).relu().sum()

            reward = reward - action_out_of_bound_penalty

            cum_rewards += reward

            next_state = torch.from_numpy(next_state).to(device)

            reward = float(reward)

            # determine if truncating, either from environment or learning phase of the agent

            updating_agent = divisible_by(time, update_timesteps)
            done = terminated or truncated or is_last or updating_agent

            # take care of truncated by bootstrapping the next value for GAE

            if done and not terminated:
                next_value = agent.ema_critic.forward_eval(cat((next_state, action)))
                scalar_next_value = agent.critic_hl_gauss_loss(next_value).item()
                reward += agent.gamma * scalar_next_value

            memory = Memory(True, state, action, noise, past_action, reward, done, value)

            memories.append(memory)

            state = next_state
            past_action = action

            if done or is_last:
                all_rewards.append(cum_rewards)

            # updating of the agent

            if updating_agent:
                rewards_tensor = tensor(all_rewards)

                print(f'mean reward: {rewards_tensor.mean().item():.3f} | max reward: {rewards_tensor.amax().item():.3f}')

                agent.learn(memories)
                num_policy_updates += 1

                memories.clear()
                all_rewards.clear()

            if done:
                break

        if divisible_by(eps, save_every):
            agent.save()

if __name__ == '__main__':
    fire.Fire(main)
