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

from mean_conc_beta import Beta

import einx
from einops import repeat, rearrange, pack

from ema_pytorch import EMA

from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

from hl_gauss_pytorch import HLGaussLoss

from hyper_connections import HyperConnections

from assoc_scan import AssocScan

from accelerate import Accelerator

import gymnasium as gym

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# memory

StepMemory = namedtuple('StepMemory', [
    'state',
    'past_action',
    'reward',
    'mask',
    'value',
    'next_value',
])

ChunkMemory = namedtuple('ChunkMemory', [
    'state',
    'action_chunk',
    'step_indices',
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

# simba - kaist + sonyai

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
        # simba - https://arxiv.org/abs/2410.09754v1

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

        # final norm

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

# actor

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
        chunk_size = 3,
        dim_time = 16,
        mlp_depth = 3,
        dropout = 0.,
        bounds = (-1., 1.),
        init_conc = 2.,
        unimodal = True,
        **beta_kwargs
    ):
        super().__init__()

        self.num_actions = num_actions
        self.chunk_size = chunk_size
        self.bounds = bounds

        self.to_time_emb = nn.Sequential(
            RandomFourierEmbed(dim_time),
            nn.Linear(dim_time + 1, dim_time),
            nn.SiLU()
        )

        dim_actions = num_actions * chunk_size

        self.net = SimBa(
            state_dim + dim_time + dim_actions,
            dim_hidden = hidden_dim * 2,
            depth = mlp_depth,
            dropout = dropout
        )

        self.to_params = nn.Linear(hidden_dim * 2, dim_actions * 2)
        nn.init.normal_(self.to_params.weight, std = 0.01)
        nn.init.zeros_(self.to_params.bias)

        self.distr = Beta(
            bounds = bounds,
            init_conc = init_conc,
            unimodal = unimodal,
            **beta_kwargs
        )

    def forward(self, noised_actions, *, state, time):
        time_emb = self.to_time_emb(time)

        if noised_actions.ndim > 2 and noised_actions.shape[-2:] == (self.chunk_size, self.num_actions):
            flat_noised_actions = rearrange(noised_actions, '... c a -> ... (c a)')
        else:
            flat_noised_actions = noised_actions

        inp = cat((flat_noised_actions, state, time_emb), dim = -1)
        hidden = self.net(inp)

        params = self.to_params(hidden)
        params = rearrange(params, '... (c a d) -> ... c a d', c = self.chunk_size, a = self.num_actions, d = 2)

        dist = self.distr(params)

        return dist

# probabilistic nano flow

class ProbabilisticNanoFlow(Module):
    def __init__(
        self,
        model,
        bounds = (-1., 1.),
        eps = 1e-5
    ):
        super().__init__()
        self.model = model
        self.bounds = bounds
        self.eps = eps

    @property
    def chunk_size(self):
        return getattr(self.model, 'chunk_size', 1)

    @property
    def num_actions(self):
        return self.model.num_actions

    @property
    def low(self):
        return self.bounds[0]

    @property
    def high(self):
        return self.bounds[1]

    def clamp_to_support(self, t):
        return t.clamp(self.low + self.eps, self.high - self.eps)

    def sample_noise(self, shape, device):
        return torch.randn(shape, device = device)

    @torch.no_grad()
    def sample(self, steps = 4, batch_size = 1, data_shape = None, **kwargs):
        device = next(self.model.parameters()).device
        data_shape = default(data_shape, (self.chunk_size, self.num_actions))

        noise = self.sample_noise((batch_size, *data_shape), device = device)

        times = torch.linspace(0., 1., steps + 1, device = device)[:-1]
        delta = 1. / steps
        denoised = noise

        for time in times:
            time = time.expand(batch_size)
            dist = self.model(denoised, time = time, **kwargs)

            predicted_clean = dist.sample()

            pad_dims = (1,) * (denoised.ndim - 1)
            padded_time = time.view(-1, *pad_dims)
            flow = (predicted_clean - denoised) / (1. - padded_time)
            denoised = denoised + delta * flow

        return self.clamp_to_support(denoised)

    def forward(self, data, noise = None, times = None, return_entropy = False, **kwargs):
        batch, device = data.shape[0], data.device

        target_data = self.clamp_to_support(data)

        if not exists(noise):
            noise = self.sample_noise(data.shape, device = device)

        if not exists(times):
            times = torch.rand(batch, device = device)

        pad_dims = (1,) * (data.ndim - 1)
        padded_times = times.view(-1, *pad_dims)
        noised_data = noise.lerp(target_data, padded_times)

        dist = self.model(noised_data, time = times, **kwargs)

        log_prob = dist.log_prob(target_data).sum(dim = -1)

        if return_entropy:
            entropy = dist.entropy().sum(dim = -1)
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
        mlp_depth = 6,
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

# gae

def calc_gae(
    rewards,
    values,
    next_values,
    masks,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    use_accelerated = default(use_accelerated, rewards.is_cuda)

    delta = rewards + gamma * next_values * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)

    returns = gae + values

    return returns, gae

# ppo

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
        xm_temperature = 0.05,
        eps = 1e-6,
        chunk_size = 3,
        ema_kwargs: dict = dict(
            update_model_with_ema_every = 1000
        ),
        reward_range = (-300., 300.),
        save_path = './prob_flow.pt',
        bounds = (-1., 1.),
        init_conc = 2.,
        unimodal = True,
        actor_kwargs: dict = dict(),
    ):
        super().__init__()

        self.chunk_size = chunk_size
        self.xm_temperature = xm_temperature

        actor_network = Actor(
            state_dim,
            actor_hidden_dim,
            num_actions,
            chunk_size = chunk_size,
            bounds = bounds,
            init_conc = init_conc,
            unimodal = unimodal,
            **actor_kwargs
        )
        self.actor = ProbabilisticNanoFlow(actor_network, bounds = bounds, eps = eps)

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
            'ema_actor': self.ema_actor.state_dict(),
            'ema_critic': self.ema_critic.state_dict(),
        }, str(self.save_path))

    def load(self):
        if not self.save_path.exists():
            return

        data = torch.load(str(self.save_path), weights_only = True)

        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])

        if 'ema_actor' in data:
            self.ema_actor.load_state_dict(data['ema_actor'])
        else:
            self.ema_actor.copy_params_from_model_to_ema()

        if 'ema_critic' in data:
            self.ema_critic.load_state_dict(data['ema_critic'])
        else:
            self.ema_critic.copy_params_from_model_to_ema()

    def learn(self, step_memories, chunk_memories):
        eps_clip = self.eps_clip
        hl_gauss = self.critic_hl_gauss_loss

        # gae

        rewards = tensor([m.reward for m in step_memories], device = device)
        masks = tensor([m.mask for m in step_memories], device = device)
        values = stack([m.value for m in step_memories])
        next_values = stack([m.next_value for m in step_memories])

        scalar_values = hl_gauss(values)
        scalar_next_values = hl_gauss(next_values)

        with torch.no_grad():
            returns, gae = calc_gae(
                rewards = rewards,
                values = scalar_values,
                next_values = scalar_next_values,
                masks = masks,
                gamma = self.gamma,
                lam = self.lam,
                use_accelerated = False
            )

        norm_advantages = normalize(gae) + self.advantage_offset_constant

        # actor dataset

        all_chunk_states = []
        all_chunk_actions = []
        all_chunk_advs = []
        all_chunk_masks = []

        for chunk in chunk_memories:
            indices = chunk.step_indices
            k = len(indices)

            chunk_adv = torch.zeros(self.chunk_size, device = device)
            chunk_mask = torch.zeros(self.chunk_size, dtype = torch.bool, device = device)

            chunk_adv[:k] = norm_advantages[indices]
            chunk_mask[:k] = True

            all_chunk_states.append(chunk.state)
            all_chunk_actions.append(chunk.action_chunk)
            all_chunk_advs.append(chunk_adv)
            all_chunk_masks.append(chunk_mask)

        actor_ds = TensorDataset(
            stack(all_chunk_states),
            stack(all_chunk_actions),
            stack(all_chunk_advs),
            stack(all_chunk_masks),
        )
        actor_dl = DataLoader(actor_ds, batch_size = self.minibatch_size, shuffle = True)

        # critic dataset

        critic_states = stack([m.state for m in step_memories])
        critic_past_actions = stack([m.past_action for m in step_memories])

        critic_ds = TensorDataset(critic_states, critic_past_actions, returns)
        critic_dl = DataLoader(critic_ds, batch_size = self.minibatch_size, shuffle = True)

        # sync online models to ema

        self.ema_actor.copy_params_from_ema_to_model()
        self.ema_critic.copy_params_from_ema_to_model()

        # reference actor

        old_actor = deepcopy(self.ema_actor.ema_model)
        old_actor.eval()

        n_mc = self.num_noise_monte_carlo

        with tqdm(range(self.epochs), desc = 'epochs', leave = False) as pbar:
            for epoch in range(self.epochs):

                # train actor

                for states, actions, advs, masks in actor_dl:
                    batch = actions.shape[0]

                    # share flow time across candidate noises

                    times = repeat(torch.rand(batch, device = device), 'b -> (b n)', n = n_mc)

                    expanded_states = repeat(states, 'b ... -> (b n) ...', n = n_mc)
                    expanded_actions = repeat(actions, 'b ... -> (b n) ...', n = n_mc)
                    expanded_advs = repeat(advs, 'b ... -> (b n) ...', n = n_mc)
                    expanded_masks = repeat(masks, 'b ... -> (b n) ...', n = n_mc)

                    noise = self.actor.sample_noise(expanded_actions.shape, device = device)

                    actor_kwargs = dict(state = expanded_states, noise = noise, times = times)

                    log_prob, entropy = self.actor(expanded_actions, return_entropy = True, **actor_kwargs)

                    with torch.no_grad():
                        old_log_prob = old_actor(expanded_actions, **actor_kwargs)

                    ratios = (log_prob - old_log_prob).exp()

                    surr1 = ratios * expanded_advs
                    surr2 = ratios.clamp(1. - eps_clip, 1. + eps_clip) * expanded_advs

                    ppo_policy_loss = torch.min(surr1, surr2)
                    spo_policy_loss = ratios * expanded_advs - (ratios - 1.).square() * expanded_advs.abs() / (2 * self.eps_clip)

                    policy_surr = torch.where(expanded_advs > 0., ppo_policy_loss, spo_policy_loss)

                    sub_loss = -policy_surr - self.entropy_coef * entropy
                    sub_loss = sub_loss.masked_fill(~expanded_masks, 0.)

                    # explorative modeling (xm) - alexi gladstone et al.

                    candidate_losses = sub_loss.sum(dim = -1) / expanded_masks.sum(dim = -1).clamp(min = 1)
                    candidate_losses = rearrange(candidate_losses, '(b n) -> b n', b = batch, n = n_mc)

                    if exists(self.xm_temperature) and self.xm_temperature > 0.:
                        weights = F.softmin(candidate_losses / self.xm_temperature, dim = -1)
                        policy_loss = (weights.detach() * candidate_losses).sum(dim = -1).mean()
                    else:
                        policy_loss = candidate_losses.amin(dim = -1).mean()

                    policy_loss.backward()
                    self.opt_actor.step()
                    self.opt_actor.zero_grad()

                # train critic

                for states, past_actions, rets in critic_dl:
                    critic_values = self.critic(cat((states, past_actions), dim = -1))
                    critic_loss = hl_gauss(critic_values, rets).mean()

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
    chunk_size = 3,
    minibatch_size = 64,
    lr = 0.0003,
    betas = (0.9, 0.99),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.05,
    cautious_factor = 0.1,
    ema_decay = 0.9,
    update_timesteps = 1500,
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
    save_path = './prob_flow.pt',
    load = False,
    use_wandb = False,
    cpu = True,
    recent_rewards_window = 20,
    init_conc = 2.,
    unimodal = True,
    reward_range = None,
    xm_temperature = 0.05,
):
    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    if use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project = 'prob-flow-ppo')

    env_kwargs = dict()
    if 'continuous' in env_name.lower() or 'lunar' in env_name.lower():
        env_kwargs['continuous'] = True

    env = gym.make(
        env_name,
        render_mode = 'rgb_array',
        **env_kwargs
    )

    reward_range = default(reward_range, (-300., 300.))

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
    action_space = env.action_space
    bounds = (float(action_space.low[0]), float(action_space.high[0]))

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
        chunk_size = chunk_size,
        reward_range = reward_range,
        bounds = bounds,
        init_conc = init_conc,
        unimodal = unimodal,
        save_path = save_path,
        xm_temperature = xm_temperature,
    ).to(device)

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    step_memories = []
    chunk_memories = []

    time = 0
    num_policy_updates = 0
    all_rewards = []
    recent_rewards = deque(maxlen = recent_rewards_window)

    pbar = tqdm(range(num_episodes), desc = 'episodes')
    for eps in pbar:

        state, _ = env.reset(seed = seed)
        state = torch.from_numpy(state).float().to(device)

        cum_rewards = 0.
        past_action = torch.zeros((num_actions,), device = device)
        timestep = 0

        while timestep < max_timesteps:
            chunk_start_state = state
            actor_state = add_batch(state)

            with torch.no_grad():
                action_chunk = agent.ema_actor.sample(
                    steps = actor_flow_timesteps,
                    batch_size = actor_state.shape[0],
                    data_shape = (chunk_size, num_actions),
                    state = actor_state
                )

            action_chunk = remove_batch(action_chunk)

            step_indices = []
            episode_done = False

            for a_idx in range(chunk_size):
                time += 1
                timestep += 1

                sub_state = state
                sub_past_action = past_action
                value = agent.ema_critic.forward_eval(cat((sub_state, sub_past_action)))

                action_to_env = action_chunk[a_idx].cpu().numpy()
                step_next_state, step_reward, terminated, truncated, _ = env.step(action_to_env)

                cum_rewards += step_reward
                next_state_tensor = torch.from_numpy(step_next_state).float().to(device)

                step_terminated = terminated
                step_truncated = truncated or (timestep >= max_timesteps)
                episode_done = step_terminated or step_truncated

                mask = 0. if step_terminated else 1.

                if step_terminated:
                    next_value = torch.zeros_like(value)
                else:
                    with torch.no_grad():
                        next_value = agent.ema_critic.forward_eval(cat((next_state_tensor, action_chunk[a_idx])))

                step_idx = len(step_memories)
                step_indices.append(step_idx)
                step_memories.append(StepMemory(
                    state = sub_state,
                    past_action = sub_past_action,
                    reward = float(step_reward),
                    mask = mask,
                    value = value,
                    next_value = next_value,
                ))

                state = next_state_tensor
                past_action = action_chunk[a_idx]

                if episode_done:
                    break

            chunk_memories.append(ChunkMemory(
                state = chunk_start_state,
                action_chunk = action_chunk,
                step_indices = step_indices,
            ))

            updating_agent = (time // update_timesteps) > ((time - len(step_indices)) // update_timesteps)

            if updating_agent:
                rewards_tensor = tensor(all_rewards)
                if len(all_rewards) > 0:
                    print(f'mean reward: {rewards_tensor.mean().item():.3f} | max reward: {rewards_tensor.amax().item():.3f}')

                agent.learn(step_memories, chunk_memories)
                num_policy_updates += 1

                step_memories.clear()
                chunk_memories.clear()
                all_rewards.clear()

            if episode_done:
                all_rewards.append(cum_rewards)
                recent_rewards.append(cum_rewards)
                break

        if len(recent_rewards) > 0:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            pbar.set_postfix(avg_reward = f'{avg_reward:.3f}')

            if use_wandb:
                wandb.log(dict(
                    episode = eps,
                    reward = avg_reward
                ))

        if divisible_by(eps, save_every):
            agent.save()

    agent.save()
    env.close()

if __name__ == '__main__':
    fire.Fire(main)
