from __future__ import annotations
from math import log

import torch
from torch import tensor
from torch.nn import Module
import torch.nn.functional as F

from ema_pytorch import EMA
from rectified_flow_pytorch.nano_flow import NanoFlow, append_dims

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class TDFlow(Module):
    """
    Farebrother et al. https://arxiv.org/abs/2503.09817
    """

    def __init__(
        self,
        model: Module,
        ema_model: Module | None = None,
        long_horizon_discount_factor = 0.99,
        bootstrap_sampling_steps = 10, # they used 10 steps
        state_cond_kwarg_name = 'image_cond',
        time_cond_kwarg_name = 'times',
        discount_cond_kwarg_name = 'cond',
        condition_on_discount = False,
        ema_beta = 0.99,
        ema_kwargs: dict = dict()
    ):
        super().__init__()
        self.model = model

        if not exists(ema_model):
            ema_model = EMA(model, beta = ema_beta, **ema_kwargs)

        self.ema_model = ema_model

        self.model_flow = NanoFlow(model, times_cond_kwarg = time_cond_kwarg_name)
        self.ema_model_flow = NanoFlow(self.ema_model, times_cond_kwarg = time_cond_kwarg_name)

        self.time_cond_kwarg_name = time_cond_kwarg_name
        self.state_cond_kwarg_name = state_cond_kwarg_name
        self.discount_cond_kwarg_name = discount_cond_kwarg_name

        self.bootstrap_sampling_steps = bootstrap_sampling_steps

        # condition on discount

        self.condition_on_discount = condition_on_discount

        # td related

        self.long_horizon_discount_factor = long_horizon_discount_factor

        self.register_buffer('zero', tensor(0.), persistent = False)

    def update_ema(self):
        self.ema_model.update()

    def forward(
        self,
        state,
        next_state = None,
        return_loss_breakdown = False
    ):
        batch_size, device = state.shape[0], state.device

        # variables

        state_key, time_key, discount_key = self.state_cond_kwarg_name, self.time_cond_kwarg_name, self.discount_cond_kwarg_name

        # construct the discount conditioning, if needed

        discount_cond_kwarg = dict()
        if self.condition_on_discount:
            # Farebrother conditions using (γ), (1. - γ), and (-log(1. - γ))

            γ = self.long_horizon_discount_factor

            discount_cond = tensor([γ, 1. - γ, -log(1. - γ)], device = device)
            discount_cond_kwarg = {discount_key: discount_cond}

        # if not training, sample

        is_training = exists(next_state)

        if not is_training:
            data_shape = state.shape[1:]
            state_kwarg = {state_key: state}
            return self.model_flow.sample(batch_size = batch_size, data_shape = data_shape, **discount_cond_kwarg, **state_kwarg)

        # td flow

        long_gamma = self.long_horizon_discount_factor

        # (1. - γ) do usual t -> t+1 prediction

        state_kwarg = {state_key: state}

        next_state_flow_loss = self.model_flow(next_state, **state_kwarg, **discount_cond_kwarg)

        # (γ), predict the prediction of the next state
        # Farebrother proposes only matching the velocity instead

        data_shape = state.shape[1:]

        next_state_kwarg = {state_key: next_state}

        target = self.ema_model_flow.sample(batch_size = batch_size, steps = self.bootstrap_sampling_steps, data_shape = data_shape, **next_state_kwarg, **discount_cond_kwarg)

        times = torch.rand(batch_size, device = device)
        time_kwargs = {time_key: times}

        padded_times = append_dims(times, target.ndim - 1)

        noise = torch.randn_like(target)
        noised = noise.lerp(target, padded_times)

        pred_flow = self.model(noised, **state_kwarg, **time_kwargs, **discount_cond_kwarg)

        with torch.no_grad():
            self.ema_model.eval()
            target_flow = self.ema_model(noised, **next_state_kwarg, **time_kwargs, **discount_cond_kwarg)

        velocity_loss = F.mse_loss(pred_flow, target_flow)

        # total

        total_loss = (
            next_state_flow_loss * (1. - long_gamma) +
            velocity_loss * long_gamma
        )

        if not return_loss_breakdown:
            return total_loss

        loss_breakdown = (next_state_flow_loss, velocity_loss)
        return total_loss, loss_breakdown

# quick test

if __name__ == '__main__':
    from rectified_flow_pytorch.rectified_flow import Unet

    model = Unet(32, has_image_cond = True, accept_cond = True, dim_cond = 3)

    td_flow = TDFlow(model, long_horizon_discount_factor = 0.5, condition_on_discount = True)

    state = torch.randn(5, 3, 32, 32)
    next_state = torch.randn(5, 3, 32, 32)

    loss = td_flow(state, next_state)

    loss.backward()

    td_flow.update_ema()

    pred = td_flow(state)
    assert pred.shape == state.shape
