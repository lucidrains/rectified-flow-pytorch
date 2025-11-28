from __future__ import annotations

import torch
from torch.nn import Module
import torch.nn.functional as F

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def append_dims(t, dims):
    shape = t.shape
    ones = ((1,) * dims)
    return t.reshape(*shape, *ones)

class NanoFlow(Module):
    def __init__(
        self,
        model: Module,
        times_cond_kwarg = None,
        data_shape = None,
        normalize_data_fn = identity,
        unnormalize_data_fn = identity,
        loss_fn = F.mse_loss,
        loss_clean_weight = 1.,
        loss_noise_weight = 1.,
        loss_flow_weight = 1.,
        eps = 5e-4
    ):
        super().__init__()
        self.model = model
        self.times_cond_kwarg = times_cond_kwarg
        self.data_shape = data_shape

        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn

        self.loss_clean_weight = loss_clean_weight
        self.loss_noise_weight = loss_noise_weight
        self.loss_flow_weight = loss_flow_weight

        self.loss_fn = loss_fn

        self.eps = eps

    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        *,
        steps_config: tuple[tuple[str, int], ...] = (('clean', 4), ('flow', 8), ('noise', 4)), # say 4 steps of clean, 8 velocity, then 4 noise at the end
        data_shape = None,
        return_noise = False,
        **kwargs
    ):

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'
        device = next(self.model.parameters()).device

        noise = torch.randn((batch_size, *data_shape), device = device)

        # validate step config

        assert isinstance(steps_config, tuple)

        total_steps = 0
        expanded_step_config = []

        for step_config in steps_config:
            assert isinstance(step_config, tuple)
            step_type, steps = step_config
            assert step_type in {'clean', 'flow', 'noise'}

            total_steps += steps

            expanded_step_config.extend([step_type] * steps)

        # usual

        times = torch.linspace(0., 1., int(total_steps) + 1, device = device)[:-1]

        delta = 1. / total_steps

        denoised = noise

        for time, use_pred_type in zip(times, expanded_step_config):

            time = time.expand(batch_size)
            time_kwarg = {self.times_cond_kwarg: time} if exists(self.times_cond_kwarg) else dict()

            model_output = self.model(denoised, **time_kwarg, **kwargs)

            pred_flow, pred_clean, pred_noise = model_output.unbind(dim = 1)

            padded_times = append_dims(time, denoised.ndim - 1)

            if use_pred_type == 'clean':
                flow = (pred_clean - denoised) / (1. - padded_times).clamp_min(self.eps)
            elif use_pred_type == 'flow':
                flow = pred_flow
            elif use_pred_type == 'noise':
                flow = (denoised - pred_noise) / padded_times.clamp_min(self.eps)

            denoised = denoised + delta * flow

        out = self.unnormalize_data_fn(denoised)

        if not return_noise:
            return out

        return out, noise

    def forward(self, data, noise = None, times = None, loss_reduction = 'mean', return_losses = False, **kwargs):
        data = self.normalize_data_fn(data)

        # shapes and variables

        batch, *data_shape, ndim, device = *data.shape, data.ndim, data.device
        self.data_shape = default(self.data_shape, data_shape) # store last data shape for inference

        # flow logic

        times = default(times, torch.rand(batch, device = device))

        noise = default(noise, torch.randn_like(data))
        flow = data - noise # flow is the velocity from noise to data, also what the model is trained to predict

        padded_times = append_dims(times, ndim - 1)
        noised_data = noise.lerp(data, padded_times) # noise the data with random amounts of noise (time) - lerp is read as noise -> data from 0. to 1.

        time_kwarg = {self.times_cond_kwarg: times} if exists(self.times_cond_kwarg) else dict() # maybe time conditioning, could work without it (https://arxiv.org/abs/2502.13129v1)
        model_output = self.model(noised_data, **time_kwarg, **kwargs)

        assert model_output.shape == (batch, 3, *data_shape), 'expect the output to be (batch, 3, *data)'

        pred_flow, pred_clean, pred_noise = model_output.unbind(dim = 1)

        pred_clean_flow = (pred_clean - noised_data) / (1. - padded_times).clamp_min(self.eps)

        loss_flow = self.loss_fn(pred_flow, flow, reduction = loss_reduction)
        loss_clean = self.loss_fn(pred_clean_flow, flow, reduction = loss_reduction)
        loss_noise = self.loss_fn(pred_noise, noise, reduction = loss_reduction)

        losses = (loss_flow, loss_clean, loss_noise)

        total_loss = (
            loss_flow * self.loss_flow_weight +
            loss_clean * self.loss_clean_weight +
            loss_noise * self.loss_noise_weight
        )

        if not return_losses:
            return total_loss

        return total_loss, losses

# quick test

if __name__ == '__main__':

    from torch import nn
    from einops.layers.torch import Rearrange

    model = nn.Sequential(nn.Conv2d(3, 3 * 3, 1), Rearrange('b (o c) ... -> b o c ...', o = 3))

    nano_flow = NanoFlow(model)
    data = torch.randn(16, 3, 16, 16)

    loss = nano_flow(data)
    loss.backward()

    sampled = nano_flow.sample(batch_size = 16)
    assert sampled.shape == data.shape
