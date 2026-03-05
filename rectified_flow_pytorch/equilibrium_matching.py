from __future__ import annotations

import torch
from torch import tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.optim import Optimizer, SGD

# functions

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

# truncated decay, best performing

def truncated_decay(gamma, a = 0.8):
    return torch.where(
        gamma <= a,
        torch.ones_like(gamma),
        (1. - gamma) / (1. - a)
    )

class EquilibriumMatching(Module):
    def __init__(
        self,
        model: Module,
        data_shape = None,
        normalize_data_fn = identity,
        unnormalize_data_fn = identity,
        decay_fn: Callable = truncated_decay,
        decay_kwargs: dict = dict(a = 0.8),
        lambda_multiplier = 4.0,
        loss_fn = F.mse_loss,
        sample_optim_klass: type[Optimizer] = SGD,
        sample_optim_kwargs: dict = dict(lr = 0.003, momentum = 0.35, nesterov = True), # their best performing used sgd with nesterov momentum on truncated decay
    ):
        super().__init__()
        self.model = model
        self.data_shape = data_shape

        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn

        self.decay_fn = decay_fn
        assert decay_fn(tensor(1.)).item() == 0., 'the decay function `c` must be 0 at 1'

        self.decay_kwargs = decay_kwargs
        self.lambda_multiplier = lambda_multiplier

        self.loss_fn = loss_fn

        self.sample_optim_klass = sample_optim_klass
        self.sample_optim_kwargs = sample_optim_kwargs

    @torch.no_grad()
    def sample(
        self,
        steps = 100,
        batch_size = 1,
        data_shape = None,
        return_noise = False,
        optim_klass = None ,
        optim_kwargs = None,
        **kwargs
    ):
        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'
        device = next(self.model.parameters()).device

        noise = torch.randn((batch_size, *data_shape), device = device)
        x = noise

        # optimizer

        optim_klass = default(optim_klass, self.sample_optim_klass)
        optim_kwargs = default(optim_kwargs, self.sample_optim_kwargs)

        optim = optim_klass([x], **optim_kwargs)

        # descend

        for _ in range(steps):
            optim.zero_grad()
            grad = self.model(x, **kwargs)
            x.grad = grad
            optim.step()

        out = self.unnormalize_data_fn(x)

        if not return_noise:
            return out

        return out, noise

    def forward(self, data, noise = None, times = None, loss_reduction = 'mean', **kwargs):
        data = self.normalize_data_fn(data)

        # shapes and variables

        shape, ndim = data.shape, data.ndim
        self.data_shape = default(self.data_shape, shape[1:])
        batch, device = shape[0], data.device

        # EqM logic - resembles flow matching but with decay of flow (viewed as gradient now) as it approaches data

        times = default(times, torch.rand(batch, device = device))

        noise = default(noise, torch.randn_like(data))

        padded_times = append_dims(times, ndim - 1)
        noised_data = noise.lerp(data, padded_times)

        # target gradient: (noise - data) * lambda * c(gamma)

        decay = self.decay_fn(times, **self.decay_kwargs)
        padded_decay = append_dims(decay, ndim - 1)

        target_grad = (noise - data) * self.lambda_multiplier * padded_decay

        model_output = self.model(noised_data, **kwargs)

        return self.loss_fn(model_output, target_grad, reduction = loss_reduction)

# quick test

if __name__ == '__main__':
    model = torch.nn.Conv2d(3, 3, 1)

    eq_matching = EquilibriumMatching(model)
    data = torch.randn(16, 3, 16, 16)

    loss = eq_matching(data)
    loss.backward()

    sampled = eq_matching.sample(batch_size = 16)
    assert sampled.shape == data.shape
