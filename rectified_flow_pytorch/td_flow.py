from __future__ import annotations

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
        ema_beta = 0.99,
        ema_sampling_steps = 10, # they used 10 steps
        model_cond_kwarg_name = 'image_cond',
        time_cond_kwargs = 'times',
        recursive_loss_weight = 1.,
        ema_kwargs: dict = dict()
    ):
        super().__init__()
        self.model = model

        if not exists(ema_model):
            ema_model = EMA(model, beta = ema_beta, **ema_kwargs)

        self.ema_model = ema_model

        self.model_flow = NanoFlow(model)
        self.ema_model_flow = NanoFlow(self.ema_model)

        self.time_cond_kwargs = time_cond_kwargs
        self.model_cond_kwarg_name = model_cond_kwarg_name

        self.ema_sampling_steps = ema_sampling_steps

        # td related

        self.long_horizon_discount_factor = long_horizon_discount_factor

        # loss related

        self.recursive_loss_weight = recursive_loss_weight

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

        keyname, time_keyname = self.model_cond_kwarg_name, self.time_cond_kwargs

        # if not training, sample

        is_training = exists(next_state)

        if not is_training:
            data_shape = state.shape[1:]
            state_kwarg = {keyname: state}
            return self.model_flow.sample(batch_size = batch_size, data_shape = data_shape, **state_kwarg)

        # td flow

        long_gamma = self.long_horizon_discount_factor

        uniform = torch.rand(batch_size, device = device)

        recursive_pred = uniform < long_gamma
        base_pred = ~recursive_pred

        # (1. - gamma) probability of the time, do usual t -> t+1 prediction

        next_state_flow_loss = self.zero

        if base_pred.any():
            state_kwarg = {keyname: state[base_pred]}
            next_state_flow_loss = self.model_flow(next_state[base_pred], **state_kwarg)

        # the rest of the time, predict the prediction of the next state
        # Farebrother proposes only matching the velocity instead

        velocity_loss = self.zero

        if recursive_pred.any():

            data_shape = state.shape[1:]

            state_kwarg = {keyname: state[recursive_pred]}
            next_state_kwarg = {keyname: next_state[recursive_pred]}

            recursive_batch_size = recursive_pred.sum().item()

            target = self.ema_model_flow.sample(batch_size = recursive_batch_size, steps = self.ema_sampling_steps, data_shape = data_shape, **next_state_kwarg)

            times = torch.rand(recursive_batch_size, device = device)
            time_kwargs = {time_keyname: times}

            padded_times = append_dims(times, target.ndim - 1)

            noise = torch.randn_like(target)
            noised = noise.lerp(target, padded_times)

            pred_flow = self.model(noised, **state_kwarg, **time_kwargs)

            with torch.no_grad():
                self.ema_model.eval()
                target_flow = self.ema_model(noised, **next_state_kwarg, **time_kwargs)

            velocity_loss = F.mse_loss(pred_flow, target_flow)

        # total

        total_loss = (
            next_state_flow_loss * base_pred.float().mean() +
            velocity_loss * recursive_pred.float().mean() * self.recursive_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        loss_breakdown = (next_state_flow_loss, velocity_loss)
        return total_loss, loss_breakdown

# quick test

if __name__ == '__main__':
    from rectified_flow_pytorch.rectified_flow import Unet

    model = Unet(32, has_image_cond = True)

    td_flow = TDFlow(model, long_horizon_discount_factor = 0.5)

    state = torch.randn(5, 3, 32, 32)
    next_state = torch.randn(5, 3, 32, 32)

    loss = td_flow(state, next_state)

    loss.backward()

    td_flow.update_ema()

    pred = td_flow(state)
    assert pred.shape == state.shape
