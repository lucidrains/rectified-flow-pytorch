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
#   "mean-conc-beta",
#   "moviepy",
#   "numpy",
#   "torch",
#   "tqdm",
#   "x-ppo"
# ]
# ///

# Flow Policy Optimization (FPO)
# McAllister et al. https://arxiv.org/abs/2507.21053

# Flow Policy Gradients for Robot Control (FPO++)
# Yi et al. https://arxiv.org/abs/2602.02481

# Flow-DPPO - Divergence Proximal Policy Optimization for Flow Matching Models
# Ping et al. https://arxiv.org/abs/2606.11025

# ReFPO - Reflow Regularization for Flow Matching Policy Gradients
# Wang et al. https://arxiv.org/abs/2606.21086

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from shutil import rmtree

import einx
import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2
from einops import pack, rearrange, repeat
from ema_pytorch import EMA
from hl_gauss_pytorch import HLGaussLoss
from hyper_connections import HyperConnections
from mean_conc_beta import Beta
from torch import cat, nn, stack, tensor
from torch.distributions import Beta as TorchBeta, kl_divergence
from torch.nn import Module, ModuleList
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from x_ppo import calc_gae, ppo_actor_loss

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# memory

Memory = namedtuple('Memory', [
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

# SimBa - https://arxiv.org/abs/2410.09754

class ReluSquared(Module):
    def forward(self, x):
        return F.relu(x).square()

class SimBa(Module):
    def __init__(
        self,
        dim,
        dim_hidden = None,
        depth = 3,
        expansion_factor = 2,
        num_residual_streams = 4
    ):
        super().__init__()
        dim_hidden = default(dim_hidden, dim * expansion_factor)
        dim_inner = dim_hidden * expansion_factor

        self.proj_in = nn.Linear(dim, dim_hidden)

        init_hyper_conn, self.expand_stream, self.reduce_stream = HyperConnections.get_init_and_expand_reduce_stream_functions(
            1,
            num_fracs = num_residual_streams,
            disable = num_residual_streams == 1
        )

        self.layers = ModuleList([
            init_hyper_conn(
                dim = dim_hidden,
                layer_index = ind,
                branch = nn.Sequential(
                    nn.RMSNorm(dim_hidden),
                    nn.Linear(dim_hidden, dim_inner),
                    ReluSquared(),
                    nn.Linear(dim_inner, dim_hidden),
                )
            )
            for ind in range(depth)
        ])

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
    def __init__(self, dim):
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
        bounds = (-1., 1.),
        init_conc = 2.,
        unimodal = True,
        dim_time = 16,
        mlp_depth = 3,
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
        )

        self.to_params = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            ReluSquared(),
            nn.Linear(hidden_dim, num_actions * 2)
        )

        nn.init.normal_(self.to_params[-1].weight, std = 0.01)
        nn.init.zeros_(self.to_params[-1].bias)

        self.beta = Beta(bounds = bounds, init_conc = init_conc, unimodal = unimodal)

    def forward(self, noised_actions, *, state, time):
        time_emb = self.to_time_emb(time)
        inp = cat((noised_actions, state, time_emb), dim = -1)
        hidden = self.net(inp)

        params = rearrange(self.to_params(hidden), '... (a d) -> ... a d', d = 2)
        return self.beta(params)

class BetaFlow(Module):
    def __init__(
        self,
        model: Actor,
        num_actions: int,
        max_time = 0.75,
        eps = 1e-5,
    ):
        super().__init__()
        self.model = model
        self.num_actions = num_actions
        self.max_time = max_time
        self.eps = eps

    def clamp_to_support(self, x):
        return x.clamp(self.model.beta.low + self.eps, self.model.beta.high - self.eps)

    @torch.no_grad()
    def sample(
        self,
        steps = 4,
        batch_size = 1,
        noise = None,
        deterministic = False,
        **kwargs
    ):
        device = next(self.model.parameters()).device
        denoised = default(noise, torch.randn((batch_size, self.num_actions), device = device))

        times = torch.arange(steps, device = device) / steps
        delta = 1. / steps

        for time in times:
            dist = self.model(denoised, time = time.expand(batch_size), **kwargs)
            pred_clean = dist.mean if deterministic else dist.sample()

            flow = (pred_clean - denoised) / (1. - time).clamp(min = self.eps)
            denoised = denoised + delta * flow

        return self.clamp_to_support(denoised)

    def forward(self, action, *, noise = None, times = None, **kwargs):
        action = self.clamp_to_support(action)
        batch, device = action.shape[0], action.device

        times = default(times, torch.rand(batch, device = device) * self.max_time)
        noise = default(noise, torch.randn_like(action))

        padded_times = rearrange(times, 'b -> b 1')
        noised = noise.lerp(action, padded_times)

        dist = self.model(noised, time = times, **kwargs)
        log_prob = dist.log_prob(action)

        # unweighted clean-prediction reflow loss (ReFPO Eq. 226)
        reflow_loss = F.mse_loss(dist.mean, action, reduction = 'none').sum(dim = -1)

        return log_prob, dist, reflow_loss

class Critic(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        hidden_dim,
        dim_pred = 1,
        mlp_depth = 6,
    ):
        super().__init__()

        self.net = SimBa(
            state_dim + num_actions,
            dim_hidden = hidden_dim,
            depth = mlp_depth,
        )

        self.value_head = nn.Linear(hidden_dim, dim_pred)

    def forward(self, x):
        return self.value_head(self.net(x))

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
        bounds = (-1., 1.),
        init_conc = 2.,
        actor_flow_timesteps = 4,
        ema_kwargs: dict = dict(
            update_model_with_ema_every = 1000
        ),
        reward_range = (-300., 300.),
        num_noise_monte_carlo = 8,
        loss_diff_clamp_value = 4.,
        kl_threshold = 0.05,
        reflow_weight = 0.04,
        entropy_coef = 0.01,
        save_path = './fpo.pt'
    ):
        super().__init__()

        actor_network = Actor(
            state_dim,
            actor_hidden_dim,
            num_actions,
            bounds = bounds,
            init_conc = init_conc,
        )
        max_time = 1. - 1. / max(actor_flow_timesteps, 2)
        self.actor = BetaFlow(actor_network, num_actions = num_actions, max_time = max_time)
        self.critic = Critic(state_dim, num_actions, critic_hidden_dim, dim_pred = critic_pred_num_bins)

        self.num_noise_monte_carlo = num_noise_monte_carlo
        self.loss_diff_clamp_value = loss_diff_clamp_value
        self.kl_threshold = kl_threshold
        self.reflow_weight = reflow_weight
        self.entropy_coef = entropy_coef

        # HL-Gauss categorical critic - https://arxiv.org/abs/2403.03950

        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value = reward_range[0],
            max_value = reward_range[1],
            num_bins = critic_pred_num_bins,
            clamp_to_range = True
        )

        self.ema_actor = EMA(self.actor, beta = ema_decay, include_online_model = False, forward_method_names = ('sample',), **ema_kwargs)
        self.ema_critic = EMA(self.critic, beta = ema_decay, include_online_model = False, **ema_kwargs)

        self.ema_actor.eval()
        self.ema_critic.eval()

        self.opt_actor = AdoptAtan2(self.actor.parameters(), lr = lr, betas = betas, cautious_factor = cautious_factor)
        self.opt_critic = AdoptAtan2(self.critic.parameters(), lr = lr, betas = betas, cautious_factor = cautious_factor)

        self.ema_actor.add_to_optimizer_post_step_hook(self.opt_actor)
        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.lam = lam
        self.gamma = gamma
        self.eps_clip = eps_clip
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

        self.ema_actor.copy_params_from_model_to_ema()
        self.ema_critic.copy_params_from_model_to_ema()

    def learn(self, memories):
        eps_clip = self.eps_clip
        hl_gauss = self.critic_hl_gauss_loss

        states, actions, past_actions, rewards, is_boundaries, values = zip(*memories)
        masks = [(1. - float(b)) for b in is_boundaries]

        with torch.no_grad():
            scalar_values = hl_gauss(stack(values))

            returns, gae = calc_gae(
                rewards = tensor(rewards, dtype = torch.float32, device = device),
                values = scalar_values,
                masks = tensor(masks, dtype = torch.float32, device = device),
                lam = self.lam,
                gamma = self.gamma,
                use_accelerated = False,
                return_advantages = True
            )

            advantages = normalize(gae)

        states = stack(states).to(device).detach()
        actions = stack(actions).to(device).detach()
        past_actions = stack(past_actions).to(device).detach()

        # ReFPO Algorithm 1: sample and freeze (tau_i, eps_i) per action and precompute old policy once before epochs

        num_samples = actions.shape[0]
        n_mc = self.num_noise_monte_carlo
        clamp_value = self.loss_diff_clamp_value

        mc_times = torch.rand((num_samples, n_mc), device = device) * self.actor.max_time
        mc_noise = torch.randn((num_samples, n_mc, actions.shape[-1]), device = device)

        with torch.no_grad():
            old_log_probs, old_dist, _ = self.actor(
                repeat(actions, 'b ... -> (b n) ...', n = n_mc),
                state = repeat(states, 'b ... -> (b n) ...', n = n_mc),
                noise = rearrange(mc_noise, 'b n d -> (b n) d'),
                times = rearrange(mc_times, 'b n -> (b n)')
            )
            old_log_probs = rearrange(old_log_probs, '(b n) d -> b n d', n = n_mc)
            old_conc1 = rearrange(old_dist.base_dist.concentration1, '(b n) d -> b n d', n = n_mc)
            old_conc0 = rearrange(old_dist.base_dist.concentration0, '(b n) d -> b n d', n = n_mc)

        dataset = TensorDataset(states, actions, past_actions, returns, advantages, mc_times, mc_noise, old_log_probs, old_conc1, old_conc0)
        dl = DataLoader(dataset, batch_size = self.minibatch_size, shuffle = True)

        for _ in tqdm(range(self.epochs)):
            for states, actions, past_actions, returns, advantages, times, noise, old_log_prob, old_conc1, old_conc0 in dl:
                expanded_states = repeat(states, 'b ... -> (b n) ...', n = n_mc)
                expanded_actions = repeat(actions, 'b ... -> (b n) ...', n = n_mc)
                expanded_advantages = repeat(advantages, 'b -> (b n)', n = n_mc)

                times = rearrange(times, 'b n -> (b n)')
                noise = rearrange(noise, 'b n d -> (b n) d')
                old_log_prob = rearrange(old_log_prob, 'b n d -> (b n) d')
                old_dist = TorchBeta(
                    rearrange(old_conc1, 'b n d -> (b n) d'),
                    rearrange(old_conc0, 'b n d -> (b n) d')
                )

                log_prob, dist, reflow_loss = self.actor(
                    expanded_actions,
                    state = expanded_states,
                    noise = noise,
                    times = times
                )

                with torch.no_grad():
                    # flow-dppo - exact analytical KL divergence mask - Ping et al. https://arxiv.org/abs/2606.11025

                    step_kl = kl_divergence(old_dist, dist.base_dist).sum(dim = -1)
                    log_diff = (log_prob - old_log_prob).sum(dim = -1)

                    moving_away = (expanded_advantages * log_diff) > 0.
                    keep_mask = ~(moving_away & (step_kl > self.kl_threshold))

                # per-sample PPO actor loss - FPO++ (Yi et al. https://arxiv.org/abs/2602.02481)

                policy_loss = ppo_actor_loss(
                    action_log_probs = log_prob,
                    old_action_log_probs = old_log_prob,
                    advantages = expanded_advantages,
                    eps_clip = eps_clip,
                    dual_clip = True,
                    log_ratio_clamp = (-clamp_value, clamp_value)
                )

                policy_loss = torch.where(keep_mask, policy_loss, 0.).mean()

                # refpo - unweighted reflow regularization reusing the same forward pass - Wang et al. https://arxiv.org/abs/2606.21086

                if self.reflow_weight > 0.:
                    policy_loss = policy_loss + self.reflow_weight * reflow_loss.mean()

                if self.entropy_coef > 0.:
                    policy_loss = policy_loss - self.entropy_coef * dist.entropy().sum(dim = -1).mean()

                policy_loss.backward()
                self.opt_actor.step()
                self.opt_actor.zero_grad()

                # critic update

                values = self.critic(cat((states, past_actions), dim = -1))
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
    actor_flow_timesteps = 4,
    critic_hidden_dim = 64,
    critic_pred_num_bins = 500,
    minibatch_size = 128,
    lr = 0.0003,
    betas = (0.9, 0.99),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.1,
    cautious_factor = 0.1,
    ema_decay = 0.9,
    init_conc = 2.,
    update_timesteps = 2048,
    kl_threshold = 0.05,
    reflow_weight = 0.04,
    entropy_coef = 0.01,
    epochs = 4,
    seed = None,
    render = True,
    render_every_eps = 100,
    save_every = 1000,
    clear_videos = True,
    video_folder = './lunar-recording',
    save_path = './fpo.pt',
    load = False
):
    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

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
    bounds = (torch.from_numpy(env.action_space.low), torch.from_numpy(env.action_space.high))

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
        bounds = bounds,
        init_conc = init_conc,
        actor_flow_timesteps = actor_flow_timesteps,
        kl_threshold = kl_threshold,
        reflow_weight = reflow_weight,
        entropy_coef = entropy_coef,
        save_path = save_path
    ).to(device)

    if load:
        agent.load()

    time = 0
    memories = []
    all_rewards = []

    for eps in tqdm(range(num_episodes), desc = 'episodes'):
        state, _ = env.reset(seed = seed if eps == 0 else None)
        state = torch.from_numpy(state).float().to(device)

        cum_rewards = 0.
        past_action = torch.zeros((num_actions,), device = device)

        for i in range(max_timesteps):
            time += 1

            action = agent.ema_actor.sample(actor_flow_timesteps, state = rearrange(state, '... -> 1 ...'))
            action = rearrange(action, '1 ... -> ...')

            value = agent.ema_critic.forward_eval(cat((state, past_action)))

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())

            reward = float(reward)
            cum_rewards += reward

            next_state = torch.from_numpy(next_state).float().to(device)

            ep_done = terminated or truncated or (i == max_timesteps - 1)
            updating_agent = divisible_by(time, update_timesteps)
            is_boundary = ep_done or updating_agent

            if is_boundary and not terminated:
                next_value = agent.ema_critic.forward_eval(cat((next_state, action)))
                reward += agent.gamma * agent.critic_hl_gauss_loss(next_value).item()

            memories.append(Memory(state, action, past_action, reward, is_boundary, value))

            state = next_state
            past_action = action

            if ep_done:
                all_rewards.append(cum_rewards)

            if updating_agent:
                if len(all_rewards) > 0:
                    rewards_tensor = tensor(all_rewards)
                    print(f'mean reward: {rewards_tensor.mean().item():.3f} | max reward: {rewards_tensor.amax().item():.3f}')

                agent.learn(memories)
                memories.clear()
                all_rewards.clear()

            if ep_done:
                break

        if eps > 0 and divisible_by(eps, save_every):
            agent.save()

if __name__ == '__main__':
    fire.Fire(main)
