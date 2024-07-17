from __future__ import annotations

from typing import Tuple

import torch
from torch.nn import Module
import torch.nn.functional as F

from torchdiffeq import odeint

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

# main class

class RectifiedFlow(Module):
    def __init__(
        self,
        model: Module,
        time_cond_kwarg = 'times',
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        data_shape: Tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.model = model
        self.time_cond_kwarg = time_cond_kwarg # whether the model is to be conditioned on the times

        # sampling

        self.odeint_kwargs = odeint_kwargs
        self.data_shape = data_shape

    @property
    def device(self):
        return next(self.model.parameters()).device

    def sample(
        self,
        batch_size = 1,
        steps = 16,
        data_shape: Tuple[int, ...] | None = None,
        **model_kwargs
    ):
        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least with one forward'

        def ode_fn(t, x):
            time_kwarg = self.time_cond_kwarg

            if exists(time_kwarg):
                model_kwargs.update(**{time_kwarg: t})

            return self.model(x, **model_kwargs)

        # start with random gaussian noise - y0

        noise = torch.randn((batch_size, *data_shape))

        # time steps

        times = torch.linspace(0., 1., steps, device = self.device)

        # ode

        trajectory = odeint(ode_fn, noise, times, **self.odeint_kwargs)

        sampled_data = trajectory[-1]
        return sampled_data

    def forward(
        self,
        data,
        **model_kwargs
    ):
        batch, *data_shape = data.shape

        self.data_shape = default(self.data_shape, data_shape)

        # x0 - gaussian noise, x1 - data

        noise = torch.randn_like(data)

        # times, and times with dimension padding on right

        times = torch.rand(batch, device = self.device)
        padded_times = append_dims(times, data.ndim - 1)

        # Algorithm 2 in paper
        # linear interpolation of noise with data using random times
        # x1 * t + x0 * (1 - t) - so from noise (time = 0) to data (time = 1.)

        noised = padded_times * data + (1. - padded_times) * noise

        # prepare maybe time conditioning for model
        
        time_kwarg = self.time_cond_kwarg

        if exists(time_kwarg):
            model_kwargs.update(**{time_kwarg: times})

        # the model predicts the flow from the noised data

        flow = data - noise
        pred_flow = self.model(noised, **model_kwargs)

        loss = F.mse_loss(pred_flow, flow)

        return loss
