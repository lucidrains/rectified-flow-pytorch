# Recursive Flow Matching
# Jiahe Huang et al. https://arxiv.org/abs/2605.26535

from functools import partial

import torch
from torch.nn import Module
import torch.nn.functional as F

import einx
from einops import rearrange, repeat

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

class RecursiveFlow(Module):
    def __init__(
        self,
        model: Module,
        times_cond_kwarg = 'times',
        alphas_cond_kwargs = 'cond',
        recursive_depth = 2,
        data_shape = None,
        normalize_data_fn = identity,
        unnormalize_data_fn = identity,
        predict_clean = False,
        max_timesteps = 100,
        loss_fn = F.mse_loss,
        consistency_loss_weight = 1.
    ):
        super().__init__()
        self.model = model

        self.times_cond_kwarg = times_cond_kwarg
        self.alphas_cond_kwargs = alphas_cond_kwargs

        self.recursive_depth = recursive_depth # D in paper, they used just 2

        self.data_shape = data_shape

        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn

        self.predict_clean = predict_clean # predicting x0
        self.max_timesteps = max_timesteps

        self.loss_fn = loss_fn
        self.consistency_loss_weight = consistency_loss_weight

    @torch.no_grad()
    def sample(
        self,
        steps = 16,
        batch_size = 1,
        data_shape = None,
        return_noise = False,
        **kwargs
    ):
        assert 1 <= steps <= self.max_timesteps

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'
        device = next(self.model.parameters()).device

        noise = torch.randn((batch_size, *data_shape), device = device)

        times = torch.linspace(0., 1., steps + 1, device = device)[:-1]
        delta = 1. / steps

        denoised = noise

        alphas = torch.ones((batch_size,), device = device)

        for time in times:
            time = time.expand(batch_size)
            time_and_alpha_kwargs = {self.times_cond_kwarg: time, self.alphas_cond_kwargs: alphas}

            model_output = self.model(denoised, **time_and_alpha_kwargs, **kwargs)

            if self.predict_clean:
                padded_time = append_dims(time, denoised.ndim - 1)
                pred_flow = (model_output - denoised) / (1. - padded_time)
            else:
                pred_flow = model_output

            denoised = denoised + delta * pred_flow

        out = self.unnormalize_data_fn(denoised)

        if not return_noise:
            return out

        return out, noise

    def forward(self, data, noise = None, times = None, loss_reduction = 'mean', return_loss_breakdown = False, **kwargs):
        data = self.normalize_data_fn(data)

        # shapes and variables

        shape, ndim = data.shape, data.ndim
        self.data_shape = default(self.data_shape, shape[1:]) # store last data shape for inference
        batch, device = shape[0], data.device

        # flow logic

        times = default(times, torch.rand(batch, device = device))
        times = times * (1. - self.max_timesteps ** -1)

        # noise and flow

        noise = default(noise, torch.randn_like(data))
        flow = data - noise # flow is the velocity from noise to data, also what the model is trained to predict

        padded_times = append_dims(times, ndim - 1)
        noised_data = noise.lerp(data, padded_times) # noise the data with random amounts of noise (time) - lerp is read as noise -> data from 0. to 1.

        # 'recursive' depth related

        D = self.recursive_depth
        has_recursive_depth = D > 1

        if has_recursive_depth:
            alphas = torch.rand(batch, device = device) * (1. - times) + times # chosen randomly from times to 1.

            times = repeat(times, 'b ... -> d b ...', d = D)
            padded_times = repeat(padded_times, 'b ... -> d b ...', d = D)
            noised_data = repeat(noised_data, 'b ... -> (d b) ...', d = D)

            powers = torch.arange(D, device = device)
            alphas = rearrange(alphas, 'b -> 1 b') ** rearrange(powers, 'd -> d 1')

            times = einx.divide('d b ..., d b -> (d b) ...', times, alphas)
            padded_times = einx.divide('d b ..., d b -> (d b) ...', padded_times, alphas)

            scaled_flows = einx.multiply('d b, b ... -> (d b) ...', alphas, flow)
        else:
            alphas = torch.ones_like(times)

        # readying for model predictions

        alphas_cond = rearrange(alphas, 'd b -> (d b)') if has_recursive_depth else alphas

        cond_kwarg = {self.times_cond_kwarg: times, self.alphas_cond_kwargs: alphas_cond}

        model_output = self.model(noised_data, **cond_kwarg, **kwargs)

        # convert depending on whether predicting x0

        if self.predict_clean:
            pred_flow = (model_output - noised_data) / (1. - padded_times)
        else:
            pred_flow = model_output

        loss_fn = partial(self.loss_fn, reduction = loss_reduction)

        # early return if D = 1

        if not has_recursive_depth:
            return loss_fn(flow, pred_flow)

        # trajectory and consistency losses

        trajectory_losses = loss_fn(scaled_flows, pred_flow)

        is_scalar = trajectory_losses.ndim == 0

        # consistency losses

        pred_flows = rearrange(pred_flow, '(d b) ... -> d b ...', d = D)

        scaled_first_pred_flows = einx.multiply('d b, b ... -> d b ...', alphas[1:], pred_flows[0])

        consistency_losses = loss_fn(scaled_first_pred_flows, pred_flows[1:])

        loss_breakdown = (trajectory_losses, consistency_losses)

        if not is_scalar:
            return loss_breakdown

        total_loss = (
            trajectory_losses +
            consistency_losses * self.consistency_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, loss_breakdown

# quick test

if __name__ == '__main__':

    class MockModel(Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Conv2d(3, 3, 1)

        def forward(self, data, times, alphas):
            return self.net(data)

    model = MockModel()

    recursive_flow = RecursiveFlow(model, alphas_cond_kwargs = 'alphas')
    data = torch.randn(16, 3, 16, 16)

    loss = recursive_flow(data)
    loss.backward()

    sampled = recursive_flow.sample(batch_size = 16)
    assert sampled.shape == data.shape
