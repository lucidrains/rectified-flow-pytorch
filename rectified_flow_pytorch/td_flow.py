from __future__ import annotations
from math import log
from random import uniform

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

def identity(t, *args, **kwargs):
    return t

# classes

class TDFlow(Module):
    """
    Farebrother et al.
    https://arxiv.org/abs/2503.09817
    https://arxiv.org/abs/2602.19634
    """

    def __init__(
        self,
        model: Module,
        ema_model: Module | None = None,
        discount_factor = 0.996,
        min_discount_factor = 0.75,             # does not make sense for this to be too low
        sample_discount_factor = None,
        horizon_consistency = False,            # use horizon consistency, (td-hc), proposed by Farebrother in a follow up paper to his own work
        horizon_consistency_batch_frac = 0.25,  # Farebrother had to limit it to a fraction of the batch size
        bootstrap_sampling_steps = 10,          # they used 10 steps
        state_cond_kwarg_name = 'image_cond',
        time_cond_kwarg_name = 'times',
        discount_cond_kwarg_name = 'cond',
        condition_on_discount = None,
        ema_beta = 0.99,
        ema_kwargs: dict = dict(),
        flow_kwargs: dict = dict()
    ):
        super().__init__()
        self.model = model

        if not exists(ema_model):
            ema_model = EMA(model, beta = ema_beta, **ema_kwargs)

        self.ema_model = ema_model

        self.model_flow = NanoFlow(model, times_cond_kwarg = time_cond_kwarg_name, **flow_kwargs)
        self.ema_model_flow = NanoFlow(self.ema_model, times_cond_kwarg = time_cond_kwarg_name, **flow_kwargs)

        self.state_normalize_fn = default(self.model_flow.normalize_data_fn, identity)

        self.time_cond_kwarg_name = time_cond_kwarg_name
        self.state_cond_kwarg_name = state_cond_kwarg_name
        self.discount_cond_kwarg_name = discount_cond_kwarg_name

        self.bootstrap_sampling_steps = bootstrap_sampling_steps

        # condition on discount

        self.condition_on_discount = default(condition_on_discount, horizon_consistency)
        assert not (horizon_consistency and not self.condition_on_discount), 'must condition on discount when using horizon consistency'

        # td related

        assert 0. < discount_factor < 1.
        assert 0. <= min_discount_factor < discount_factor

        self.discount_factor = discount_factor
        self.min_discount_factor = min_discount_factor

        self.sample_discount_factor = default(sample_discount_factor, horizon_consistency) # for horizon consistency, this will be asserted to be True

        # td-hc from "jumpy" world models paper

        assert not (horizon_consistency and not self.sample_discount_factor), 'must be sampling discount factors during training if doing horizon consistency'

        self.horizon_consistency = horizon_consistency
        self.horizon_consistency_batch_frac = horizon_consistency_batch_frac

        self.register_buffer('zero', tensor(0.), persistent = False)

    def update_ema(self):
        self.ema_model.update()

    def forward(
        self,
        state,
        next_state = None,
        discount_factor = None,
        sample_discount_factor = None,
        return_loss_breakdown = False,
        is_training = None
    ):
        batch_size, device = state.shape[0], state.device

        maybe_state_normalize = self.state_normalize_fn

        horizon_consistency = self.horizon_consistency

        is_training = default(is_training, exists(next_state))

        # variables

        state_key, time_key, discount_key = self.state_cond_kwarg_name, self.time_cond_kwarg_name, self.discount_cond_kwarg_name

        # td variables

        sample_discount_factor = default(sample_discount_factor, self.sample_discount_factor)

        γ = default(discount_factor, self.discount_factor)
        assert 0. < γ < 1.

        if sample_discount_factor and is_training:
            # if sampling for td-hc or if researcher wishes it, then assume gamma is max gamma
            γ = uniform(self.min_discount_factor, γ)

        # function for constructing discount conditioning

        def to_discount_kwarg(discount_factor):
            # Farebrother conditions using (γ), (1. - γ), and (-log(1. - γ)) - will try this for other UVFA setups

            discount_cond = tensor([
                discount_factor,
                1. - discount_factor,
                -log(1. - discount_factor)
            ], device = device)

            return {discount_key: discount_cond}

        def to_state_kwarg(state):
            return {state_key: maybe_state_normalize(state)}

        # construct the discount conditioning, if needed

        discount_cond_kwarg = dict()
        if self.condition_on_discount:
            discount_cond_kwarg = to_discount_kwarg(γ)

        # if not training, sample and early return

        if not is_training:
            data_shape = state.shape[1:]
            state_kwarg = to_state_kwarg(state)
            return self.model_flow.sample(batch_size = batch_size, data_shape = data_shape, **discount_cond_kwarg, **state_kwarg)

        # if doing horizon consistency, construct short horizon related - beta is short horizon discount factor

        if horizon_consistency:
            β = uniform(self.min_discount_factor, γ)
            short_discount_cond_kwarg = to_discount_kwarg(β)

        # for horizon consistency, there will be a first short jump before the second long one

        first_discount_cond_kwarg = discount_cond_kwarg

        if horizon_consistency:
            first_discount_cond_kwarg = short_discount_cond_kwarg
            second_discount_cond_kwarg = discount_cond_kwarg

        # td flow

        # (1. - γ) do usual t -> t+1 prediction

        state_kwarg = to_state_kwarg(state)

        next_state_flow_loss = self.model_flow(next_state, **state_kwarg, **first_discount_cond_kwarg)

        # (γ), predict the prediction of the next state
        # Farebrother proposes only matching the velocity instead

        data_shape = state.shape[1:]

        next_state_kwarg = to_state_kwarg(next_state)

        target = self.ema_model_flow.sample(batch_size = batch_size, steps = self.bootstrap_sampling_steps, data_shape = data_shape, **next_state_kwarg, **first_discount_cond_kwarg)

        times = torch.rand(batch_size, device = device)
        time_kwargs = {time_key: times}

        padded_times = append_dims(times, target.ndim - 1)

        noise = torch.randn_like(target)
        noised = noise.lerp(target, padded_times)

        pred_flow = self.model(noised, **state_kwarg, **time_kwargs, **discount_cond_kwarg)

        with torch.no_grad():
            self.ema_model.eval()
            target_flow = self.ema_model(noised, **next_state_kwarg, **time_kwargs, **first_discount_cond_kwarg)

        velocity_loss = F.mse_loss(pred_flow, target_flow)

        second_term_weight = γ

        # handle maybe horizon consistency

        third_term_weight = 0.
        consistency_loss = self.zero

        if horizon_consistency:
            hc_batch_size = max(1, int(self.horizon_consistency_batch_frac * batch_size))

            # just redeclare for clarity

            state = state[:hc_batch_size]
            noise = noise[:hc_batch_size]
            target = target[:hc_batch_size]

            time_kwargs = {time_key: times[:hc_batch_size]}
            padded_times = padded_times[:hc_batch_size]

            # second jump

            short_state_kwarg = to_state_kwarg(target)
            second_target = self.ema_model_flow.sample(batch_size = hc_batch_size, steps = self.bootstrap_sampling_steps, data_shape = data_shape, **short_state_kwarg, **second_discount_cond_kwarg)

            second_noised = noise.lerp(second_target, padded_times)

            with torch.no_grad():
                self.ema_model.eval()
                second_target_flow = self.ema_model(second_noised, **short_state_kwarg, **time_kwargs, **second_discount_cond_kwarg)

            state_kwarg = to_state_kwarg(state)

            second_pred_flow = self.model(second_noised, **state_kwarg, **time_kwargs, **second_discount_cond_kwarg)

            consistency_loss = F.mse_loss(second_pred_flow, second_target_flow)

            # weights are now modified for the jump

            second_term_weight = (γ * (1. - γ)) / (1. - β)
            third_term_weight = (γ * (γ - β)) / (1. - β)

        # total

        total_loss = (
            next_state_flow_loss * (1. - γ) +
            velocity_loss * second_term_weight +
            consistency_loss * third_term_weight
        )

        if not return_loss_breakdown:
            return total_loss

        loss_breakdown = (next_state_flow_loss, velocity_loss, consistency_loss)
        return total_loss, loss_breakdown

# quick test

if __name__ == '__main__':
    from rectified_flow_pytorch.rectified_flow import Unet

    model = Unet(32, has_image_cond = True, accept_cond = True, dim_cond = 3)

    td_flow = TDFlow(model, horizon_consistency = True)

    state = torch.randn(5, 3, 32, 32)
    next_state = torch.randn(5, 3, 32, 32)

    loss = td_flow(state, next_state)

    loss.backward()

    td_flow.update_ema()

    pred = td_flow(state)
    assert pred.shape == state.shape
