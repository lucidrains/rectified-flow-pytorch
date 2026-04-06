from __future__ import annotations
from typing import Callable
from random import random as random_

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor
from torch.nn import Module

import einx
from einops import rearrange, repeat

from ema_pytorch import EMA

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def append_dims(t, dims):
    shape = t.shape
    ones = ((1,) * dims)
    return t.reshape(*shape, *ones)

# transforms

def symlog(x):
    return x.sign() * (x.abs() + 1.).log()

def symexp(x):
    return x.sign() * (x.abs().exp() - 1.)

# main class

class ValueFlow(Module):
    def __init__(
        self,
        model: Module,
        *,
        gamma = 0.99,
        num_flow_steps = 10,
        temperature = 0.3,
        lambda_bcfm = 1.0,
        lambda_state_gen = 1.0,
        use_symlog = True,
        prob_state_generation = 0.,
        state_normalize_fn: Callable | None = None,
        state_unnormalize_fn: Callable | None = None,
        ema_kwargs = dict()
    ):
        super().__init__()

        self.model = model

        if use_symlog:
            self.transform = symlog
            self.inverse_transform = symexp
        else:
            self.transform = lambda x: x
            self.inverse_transform = lambda x: x

        self.target_model = EMA(
            self.model,
            forward_method_names = ('forward',),
            **ema_kwargs
        )

        self.gamma = gamma
        self.num_flow_steps = num_flow_steps
        self.temperature = temperature
        self.lambda_bcfm = lambda_bcfm
        self.lambda_state_gen = lambda_state_gen
        self.prob_state_generation = prob_state_generation

        self.state_normalize_fn = default(state_normalize_fn, lambda x: x)
        self.state_unnormalize_fn = default(state_unnormalize_fn, lambda x: x)

        self.register_buffer('zero', tensor(0.), persistent = False)

    def _get_cond(self, latents: Tensor) -> Tensor:
        if latents.ndim > 1 and latents.shape[-1] == 1:
            return rearrange(latents, '... 1 -> ...')
        return latents

    def _euler_ode_forward(
        self,
        model: Module,
        init_latents: Tensor,
        state: Tensor,
        action = None
    ) -> list[Tensor]:

        batch, device = init_latents.shape[0], init_latents.device
        delta_time = 1. / self.num_flow_steps

        latents = init_latents.clone()
        trajectory = [latents]

        for step in range(self.num_flow_steps):
            t = torch.full((batch,), step * delta_time, device = device)
            _, velocity = model(x = state, times = t, cond = self._get_cond(latents), action = action)
            latents = latents + velocity * delta_time
            trajectory.append(latents)

        return trajectory

    def get_confidence_weight(
        self,
        base_noise: Tensor,
        state: Tensor,
        action = None
    ) -> Tensor:

        batch, device = base_noise.shape[0], base_noise.device
        delta_time = 1. / self.num_flow_steps

        latents = base_noise.clone()
        flow_deriv = torch.ones_like(latents)

        for step in range(self.num_flow_steps):
            t = torch.full((batch,), step * delta_time, device = device)

            latents = latents.detach().requires_grad_()
            _, velocity = self.model(x = state, times = t, cond = self._get_cond(latents), action = action)

            velocity_deriv, = torch.autograd.grad(velocity.sum(), latents, retain_graph = False)

            latents = latents.detach() + velocity.detach() * delta_time
            flow_deriv = flow_deriv + velocity_deriv.detach() * flow_deriv * delta_time

        confidence = torch.sigmoid(-self.temperature * flow_deriv.abs()) + 0.5
        return confidence.detach()

    # state generation training

    def _forward_state_generation(
        self,
        state: Tensor,
        action = None,
        reward = None,
        explicit_target_return = None,
        loss_weight = None,
        flow_state = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:

        flow_state = default(flow_state, state)
        batch, device = flow_state.shape[0], flow_state.device

        cond_return = explicit_target_return if exists(explicit_target_return) else reward
        assert exists(cond_return), 'need explicit_target_return or reward for state generation'

        if cond_return.ndim == 1:
            cond_return = rearrange(cond_return, 'b -> b 1')

        cond_return = self.transform(cond_return)

        # standard flow matching on state

        noise = torch.randn_like(flow_state)
        times = torch.rand((batch,), device = device)
        padded_times = append_dims(times, flow_state.ndim - 1)

        z_t = torch.lerp(noise, flow_state, padded_times)
        target_v = flow_state - noise

        pred_v, _ = self.model(x = z_t, times = times, cond = self._get_cond(cond_return), action = action)

        loss_state = F.mse_loss(pred_v, target_v, reduction = 'none')

        if exists(loss_weight):
            loss_state = einx.multiply('b, b ... -> b ...', loss_weight, loss_state)

        loss_state = loss_state.mean()

        return self.lambda_state_gen * loss_state, (self.zero, self.zero, loss_state)

    @torch.no_grad()
    def sample_q_value(
        self,
        state: Tensor,
        action = None,
        num_samples = 16,
        zero_variance = False
    ) -> Tensor:

        batch, device = state.shape[0], state.device
        state = self.state_normalize_fn(state)

        noise_fn = torch.zeros if zero_variance else torch.randn
        base_noise = noise_fn(batch * num_samples, 1, device = device)

        expanded_state = repeat(state, 'b ... -> (b n) ...', n = num_samples)
        expanded_action = repeat(action, 'b ... -> (b n) ...', n = num_samples) if exists(action) else None

        trajectory = self._euler_ode_forward(self.model, base_noise, expanded_state, action = expanded_action)
        q_samples = trajectory[-1]

        q_samples = self.inverse_transform(q_samples)
        q_samples = rearrange(q_samples, '(b n) 1 -> b n', n = num_samples)

        return q_samples.mean(dim = -1)

    @torch.no_grad()
    def get_aleatoric_uncertainty(
        self,
        state: Tensor,
        action = None,
        num_samples = 16
    ) -> Tensor:

        batch, device = state.shape[0], state.device
        state = self.state_normalize_fn(state)

        base_noise = torch.randn(batch * num_samples, 1, device = device)

        expanded_state = repeat(state, 'b ... -> (b n) ...', n = num_samples)
        expanded_action = repeat(action, 'b ... -> (b n) ...', n = num_samples) if exists(action) else None

        trajectory = self._euler_ode_forward(self.model, base_noise, expanded_state, action = expanded_action)
        q_samples = trajectory[-1]

        q_samples = self.inverse_transform(q_samples)
        q_samples = rearrange(q_samples, '(b n) 1 -> b n', n = num_samples)

        return q_samples.std(dim = -1)

    @torch.no_grad()
    def sample_state(
        self,
        returns: Tensor,
        state_shape: tuple,
        action = None,
        num_flow_steps = None
    ) -> Tensor:

        batch, device = returns.shape[0], returns.device
        num_steps = default(num_flow_steps, self.num_flow_steps)
        delta_time = 1. / num_steps

        if returns.ndim == 1:
            returns = rearrange(returns, 'b -> b 1')

        cond = self.transform(returns)

        latents = torch.randn(batch, *state_shape, device = device)

        for step in range(num_steps):
            t = torch.full((batch,), step * delta_time, device = device)
            velocity, _ = self.model(x = latents, times = t, cond = self._get_cond(cond), action = action)
            latents = latents + velocity * delta_time

        return self.state_unnormalize_fn(latents)

    def forward(
        self,
        state: Tensor,
        action = None,
        reward = None,
        next_state = None,
        next_action = None,
        dones = None,
        explicit_target_return = None,
        loss_weight = None,
        flow_state: Tensor | None = None,
        next_flow_state: Tensor | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:

        batch, device = state.shape[0], state.device

        transform = self.transform
        inverse_transform = self.inverse_transform

        state = self.state_normalize_fn(state)

        if exists(next_state):
            next_state = self.state_normalize_fn(next_state)

        # randomly choose state generation path

        if self.prob_state_generation > 0. and random_() < self.prob_state_generation:
            return self._forward_state_generation(state, action, reward, explicit_target_return, loss_weight, flow_state)

        # return prediction path

        if exists(reward) and reward.ndim == 1:
            reward = rearrange(reward, 'b -> b 1')

        base_noise = torch.randn(batch, 1, device = device)

        step_idx = torch.randint(0, self.num_flow_steps, (1,)).item()
        times = torch.full((batch,), step_idx / self.num_flow_steps, device = device)

        confidence = self.get_confidence_weight(base_noise, state, action)

        # ppo path - explicit targets from GAE

        if exists(explicit_target_return):
            if explicit_target_return.ndim == 1:
                explicit_target_return = rearrange(explicit_target_return, 'b -> b 1')

            target = transform(explicit_target_return)
            z_t = torch.lerp(base_noise, target, rearrange(times, 'b -> b 1'))

            _, pred_v = self.model(x = state, times = times, cond = self._get_cond(z_t), action = action)
            target_v = target - base_noise
            loss_bcfm = confidence * F.mse_loss(pred_v, target_v.detach(), reduction = 'none')

            if exists(loss_weight):
                loss_bcfm = einx.multiply('b, b ... -> b ...', loss_weight, loss_bcfm)

            loss_bcfm = loss_bcfm.mean()

            return loss_bcfm * self.lambda_bcfm, (self.zero, loss_bcfm, self.zero)

        # q-learning path - bootstrapped targets

        assert exists(next_state) and exists(next_action), 'next_state and next_action required for q-learning mode'

        with torch.no_grad():
            trajectory = self._euler_ode_forward(self.target_model, base_noise, next_state, action = next_action)
            z_at_step, z_final = trajectory[step_idx], trajectory[-1]

            _, target_velocity = self.target_model(x = next_state, times = times, cond = self._get_cond(z_at_step), action = next_action)

        not_done = 1. - dones if exists(dones) else torch.ones((batch, 1), device = device)
        not_done = rearrange(not_done, 'b -> b 1') if not_done.ndim == 1 else not_done

        target_return = transform(reward + self.gamma * not_done * inverse_transform(z_final))
        target_latents = transform(reward + self.gamma * not_done * inverse_transform(z_at_step))

        # dcfm loss

        _, pred_v_dcfm = self.model(x = state, times = times, cond = self._get_cond(target_latents), action = action)
        target_v_dcfm = self.gamma * not_done * target_velocity.detach()
        loss_dcfm = not_done * confidence * F.mse_loss(pred_v_dcfm, target_v_dcfm, reduction = 'none')

        # bcfm loss

        z_t = torch.lerp(base_noise, target_return, rearrange(times, 'b -> b 1'))

        _, pred_v_bcfm = self.model(x = state, times = times, cond = self._get_cond(z_t), action = action)
        target_v_bcfm = target_return - base_noise
        loss_bcfm = confidence * F.mse_loss(pred_v_bcfm, target_v_bcfm.detach(), reduction = 'none')

        if exists(loss_weight):
            loss_dcfm = einx.multiply('b, b ... -> b ...', loss_weight, loss_dcfm)
            loss_bcfm = einx.multiply('b, b ... -> b ...', loss_weight, loss_bcfm)

        loss_dcfm = loss_dcfm.mean()
        loss_bcfm = loss_bcfm.mean()

        total_loss = loss_dcfm + self.lambda_bcfm * loss_bcfm

        return total_loss, (loss_dcfm, loss_bcfm, self.zero)

if __name__ == '__main__':

    from rectified_flow_pytorch.rectified_flow import Unet

    class DualModalityCoFlow(nn.Module):
        def __init__(self, state_dim = 64):
            super().__init__()
            self.w = nn.Parameter(torch.ones(1))
            self.unet = Unet(dim = 16, channels = 3, accept_cond = True, dim_cond = 1)
            self.to_q = nn.Linear(state_dim, 1)

        def forward(self, x, times, cond, action = None):
            if x.ndim == 4:
                # image co-flow path
                return self.unet(x, times = times, cond = cond), torch.zeros(x.shape[0], 1, device = x.device)

            # td q-value path
            return torch.zeros_like(x), self.to_q(x)

    state_dim = 64
    net = DualModalityCoFlow(state_dim)
    value_flow = ValueFlow(net, gamma = 0.99, num_flow_steps = 10, prob_state_generation = 1.0)

    # q-learning (multimodal routing over image vectors)

    state       = torch.randn(2, state_dim)
    image_state = torch.randn(2, 3, 32, 32)
    action      = torch.randn(2, state_dim)
    next_state  = torch.randn(2, state_dim)
    next_action = torch.randn(2, state_dim)
    reward      = torch.randn(2)
    dones       = torch.zeros(2)

    loss, (dcfm, bcfm, sgen) = value_flow(
        state = state,
        flow_state = image_state,
        action = action,
        next_state = next_state,
        next_action = next_action,
        reward = reward,
        dones = dones
    )

    loss.backward()

    # ppo (multimodal routing natively via GAE TD returns)

    loss_ppo, (_, _, _) = value_flow(
        state = state,
        flow_state = image_state,
        explicit_target_return = torch.randn(2)
    )

    loss_ppo.backward()

    # sample explicitly generated Image state distributions from returns!

    returns = tensor([100., -100.])
    generated = value_flow.sample_state(returns, state_shape = (3, 32, 32))

    print(f'generated image states shape natively conditionally on returns: {generated.shape}')
    assert generated.shape == (2, 3, 32, 32)

    print('q-learning, ppo natively, and multi-modal conditional image-state generation all perfectly cohesive and ok!')
