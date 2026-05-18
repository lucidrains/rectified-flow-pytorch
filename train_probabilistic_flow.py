"""
Probabilistic Flow PPO

Adopts the 'Back to Basics' approach: Flow Matching actor with a 'predict clean' objective.
PPO optimizes log probs of the predicted x0 Beta manifold.
"""

from __future__ import annotations

import fire
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from functools import partial
from collections import deque, namedtuple
import math

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import TensorDataset, DataLoader

from discrete_continuous_embed_readout import Readout

import einx
from einops import repeat, rearrange, pack

from ema_pytorch import EMA

from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

from hl_gauss_pytorch import HLGaussLoss

from hyper_connections import HyperConnections

from assoc_scan import AssocScan

import gymnasium as gym

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# memory tuple

Memory = namedtuple('Memory', [
    'learnable',
    'state',
    'action',
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

        self.readout = Readout(
            dim = hidden_dim * 2,
            num_continuous = num_actions,
            continuous_dist_type = 'beta'
        )

    def forward(self, noised_actions, *, state, time):
        time_emb = self.to_time_emb(time)

        inp = cat((noised_actions, state, time_emb), dim = -1)
        hidden = self.net(inp)

        params = self.readout(hidden)
        dist = self.readout.continuous_dist.dist(params)

        return dist

# probabilistic nano flow

class ProbabilisticNanoFlow(Module):
    def __init__(self, model, eps = 1e-6):
        super().__init__()
        self.model = model
        self.eps = eps

    def clamp_to_beta_support(self, t):
        return t.clamp(self.eps, 1. - self.eps)

    def to_unit_interval(self, actions):
        """[-1, 1] -> [0, 1]"""
        return self.clamp_to_beta_support((actions + 1.) / 2.)

    def to_action_range(self, unit):
        """[0, 1] -> [-1, 1]"""
        return unit * 2. - 1.

    @torch.no_grad()
    def sample(self, steps = 4, batch_size = 1, data_shape = None, **kwargs):
        device = next(self.model.parameters()).device

        noise = self.clamp_to_beta_support(torch.rand((batch_size, *data_shape), device = device))

        times = torch.linspace(0., 1., steps + 1, device = device)[:-1]
        delta = 1. / steps
        denoised = noise

        for time in times:
            time = time.expand(batch_size)
            dist = self.model(denoised, time = time, **kwargs)

            predicted_clean = dist.sample()

            padded_time = rearrange(time, '... -> ... 1')
            flow = (predicted_clean - denoised) / (1. - padded_time)
            denoised = self.clamp_to_beta_support(denoised + delta * flow)

        return self.to_action_range(denoised)

    def forward(self, data, noise = None, times = None, return_entropy = False, **kwargs):
        batch, device = data.shape[0], data.device
        num_actions = data.shape[-1]

        unit_data = self.to_unit_interval(data)

        noise = default(noise, torch.rand_like(unit_data))
        times = default(times, torch.rand(batch, device = device))

        padded_times = rearrange(times, '... -> ... 1')
        noised_data = self.clamp_to_beta_support(noise.lerp(unit_data, padded_times))

        dist = self.model(noised_data, time = times, **kwargs)

        log_prob = dist.log_prob(unit_data).sum(dim = -1)

        # jacobian correction for bounded action space scaling
        jacobian_adjust = math.log(2) * num_actions
        log_prob = log_prob - jacobian_adjust

        if return_entropy:
            entropy = dist.entropy().sum(dim = -1) + jacobian_adjust
            return log_prob, entropy

        return log_prob


# critic

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

# GAE

def calc_gae(
    rewards,
    values,
    masks,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    assert values.shape[-1] == rewards.shape[-1]
    use_accelerated = default(use_accelerated, rewards.is_cuda)

    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[:-1], values[1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)

    returns = gae + values

    return returns

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
        advantage_offset_constant = 0.,
        num_noise_monte_carlo = 4,
        entropy_coef = 0.01,
        eps = 1e-6,
        ema_kwargs: dict = dict(
            update_model_with_ema_every = 1000
        ),
        reward_range = (-100., 100.),
        save_path = './prob_flow.pt'
    ):
        super().__init__()

        actor_network = Actor(state_dim, actor_hidden_dim, num_actions)
        self.actor = ProbabilisticNanoFlow(actor_network, eps = eps)

        self.critic = Critic(state_dim, num_actions, critic_hidden_dim, dim_pred = critic_pred_num_bins)

        self.num_noise_monte_carlo = num_noise_monte_carlo

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
        self.entropy_coef = entropy_coef

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
        past_action = to_torch_tensor(past_action)
        old_values = to_torch_tensor(values)

        # deepcopy the actor for the reference actor

        old_actor = deepcopy(self.ema_actor.ema_model)
        old_actor.eval()

        # prepare dataloader for policy phase training

        learnable = tensor(learnable).to(device)
        data = (states, action, past_action, returns, old_values)
        data = tuple(t[learnable] for t in data)

        dataset = TensorDataset(*data)

        dl = DataLoader(dataset, batch_size = self.minibatch_size, shuffle = True)

        # policy phase training, similar to original PPO

        n_mc = self.num_noise_monte_carlo

        with tqdm(range(self.epochs)) as pbar:
            for i, (states, action, past_action, returns, old_values) in enumerate(dl):
                batch = action.shape[0]

                times = torch.rand((batch * n_mc,), device = action.device)

                expanded_states = repeat(states, 'b ... -> (b n) ...', n = n_mc)
                expanded_action = repeat(action, 'b ... -> (b n) ...', n = n_mc)

                noise = torch.rand_like(expanded_action)

                actor_kwargs = dict(state = expanded_states, noise = noise, times = times)

                log_prob, entropy = self.actor(expanded_action, return_entropy = True, **actor_kwargs)

                with torch.no_grad():
                    old_log_prob = old_actor(expanded_action, **actor_kwargs)

                scalar_old_values = hl_gauss(old_values)

                # calculate clipped surrogate objective

                ratios = (log_prob - old_log_prob).exp()

                advantages = normalize(returns - scalar_old_values.detach()) + self.advantage_offset_constant
                advantages = repeat(advantages, 'b -> (b n)', n = n_mc)

                # SPO - Xie et al. https://arxiv.org/abs/2401.16025v9
                # Asymmetric SPO https://openreview.net/forum?id=BA6n0nmagi

                spo_policy_loss = ratios * advantages - (ratios - 1.).square() * advantages.abs() / (2 * self.eps_clip)

                ppo_policy_loss = torch.min(ratios * advantages, ratios.clamp(1. - eps_clip, 1. + eps_clip) * advantages)

                policy_loss = torch.where(advantages > 0., ppo_policy_loss, spo_policy_loss)

                policy_loss = -policy_loss.mean() - self.entropy_coef * entropy.mean()

                policy_loss.backward()
                self.opt_actor.step()
                self.opt_actor.zero_grad()

                # calculate clipped value loss and update value network separate from policy network

                critic_values = self.critic(cat((states, past_action), dim = -1))
                critic_loss = hl_gauss(critic_values, returns).mean()

                critic_loss.backward()
                self.opt_critic.step()
                self.opt_critic.zero_grad()

            pbar.set_description(f'actor loss: {policy_loss.item():.3f} | critic loss: {critic_loss.item():.3f}')
            pbar.update(1)

# main

def main(
    env_name = 'LunarLander-v3',
    num_episodes = 50000,
    max_timesteps = 500,
    actor_hidden_dim = 32,
    actor_flow_timesteps = 4,
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
    advantage_offset_constant = 0.,
    epochs = 4,
    eps = 1e-6,
    seed = None,
    render = True,
    render_every_eps = 100,
    save_every = 1000,
    clear_videos = True,
    video_folder = './lunar-recording',
    load = False,
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
        advantage_offset_constant,
        eps = eps,
    ).to(device)

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    num_policy_updates = 0
    all_rewards = []
    last_20_rewards = deque([], 20)

    pbar = tqdm(range(num_episodes), desc = 'episodes')
    for eps in pbar:

        state, _ = env.reset(seed = seed)
        state = torch.from_numpy(state).to(device)

        cum_rewards = 0.
        past_action = torch.zeros((num_actions,), device = device)

        for i in range(max_timesteps):
            is_last = i == (max_timesteps - 1)
            time += 1

            actor_state = add_batch(state)

            with torch.no_grad():
                action = agent.ema_actor.sample(steps = actor_flow_timesteps, batch_size = actor_state.shape[0], data_shape = (num_actions,), state = actor_state)

            action = remove_batch(action)

            value = agent.ema_critic.forward_eval(cat((state, past_action)))

            action_to_env = action.cpu().numpy()

            next_state, reward, terminated, truncated, _ = env.step(action_to_env)

            cum_rewards += reward

            next_state = torch.from_numpy(next_state).to(device)

            reward = float(reward)

            memory = Memory(True, state, action, past_action, reward, terminated, value)

            memories.append(memory)

            state = next_state
            past_action = action

            # determine if truncating, either from environment or learning phase of the agent

            updating_agent = divisible_by(time, update_timesteps)
            done = terminated or truncated or updating_agent

            # take care of truncated by adding a non-learnable memory storing the next value for GAE

            if done and not terminated:
                next_value = agent.ema_critic.forward_eval(cat((state, past_action)))

                bootstrap_value_memory = memory._replace(
                    state = state,
                    learnable = False,
                    is_boundary = True,
                    value = next_value,
                )

                memories.append(bootstrap_value_memory)

            # updating of the agent

            if updating_agent:
                rewards_tensor = tensor(all_rewards)

                print(f'mean reward: {rewards_tensor.mean().item():.3f} | max reward: {rewards_tensor.amax().item():.3f}')

                agent.learn(memories)
                num_policy_updates += 1

                memories.clear()
                all_rewards.clear()

            # break if done

            if done or is_last:
                all_rewards.append(cum_rewards)
                last_20_rewards.append(cum_rewards)

            if done:
                break

        if len(last_20_rewards) > 0:
            pbar.set_postfix(avg_reward = f'{sum(last_20_rewards) / len(last_20_rewards):.3f}')

        if divisible_by(eps, save_every):
            agent.save()

if __name__ == '__main__':
    fire.Fire(main)
