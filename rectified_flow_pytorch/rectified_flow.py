from __future__ import annotations
from typing import Tuple, Literal
from copy import deepcopy

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
        time_cond_kwarg: str | None = 'times',
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        loss_type: Literal[
            'mse',
            'pseudo_huber'
        ] = 'mse',
        data_shape: Tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.model = model
        self.time_cond_kwarg = time_cond_kwarg # whether the model is to be conditioned on the times

        # loss type

        self.loss_type = loss_type

        # sampling

        self.odeint_kwargs = odeint_kwargs
        self.data_shape = data_shape

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        steps = 16,
        noise = None,
        data_shape: Tuple[int, ...] | None = None,
        **model_kwargs
    ):
        was_training = self.training
        self.eval()

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least with one forward'

        def ode_fn(t, x):
            time_kwarg = self.time_cond_kwarg

            if exists(time_kwarg):
                model_kwargs.update(**{time_kwarg: t})

            return self.model(x, **model_kwargs)

        # start with random gaussian noise - y0

        noise = default(noise, torch.randn((batch_size, *data_shape)))

        # time steps

        times = torch.linspace(0., 1., steps, device = self.device)

        # ode

        trajectory = odeint(ode_fn, noise, times, **self.odeint_kwargs)

        sampled_data = trajectory[-1]

        self.train(was_training)
        return sampled_data

    def forward(
        self,
        data,
        noise = None,
        **model_kwargs
    ):
        batch, *data_shape = data.shape

        self.data_shape = default(self.data_shape, data_shape)

        # x0 - gaussian noise, x1 - data

        noise = default(noise, torch.randn_like(data))

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

        # loss
        # section 4.2 of https://arxiv.org/abs/2405.20320v1

        if self.loss_type == 'mse':
            loss = F.mse_loss(pred_flow, flow)

        elif self.loss_type == 'pseudo_huber':
            c = .00054 * data_shape[0]
            loss = (F.mse_loss(pred_flow, flow) + c ** 2).sqrt() - c

        else:
            raise ValueError(f'unrecognized loss type {self.loss_type}')

        return loss

# reflow wrapper

class Reflow(Module):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        frozen_model: RectifiedFlow | None = None,
        *,
        batch_size = 16,

    ):
        super().__init__()
        model, data_shape = rectified_flow.model, rectified_flow.data_shape
        assert exists(data_shape), '`data_shape` must be defined in RectifiedFlow'

        self.batch_size = batch_size
        self.data_shape = data_shape

        self.model = rectified_flow

        if not exists(frozen_model):
            # make a frozen copy of the model and set requires grad to be False for all parameters for safe measure

            frozen_model = deepcopy(rectified_flow)

            for p in frozen_model.parameters():
                p.detach_()

        self.frozen_model = frozen_model

    def parameters(self):
        return self.model.parameters() # omit frozen model

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)

    def forward(self):

        noise = torch.randn((self.batch_size, *self.data_shape))
        sampled_output = self.frozen_model.sample(noise = noise)

        # the coupling in the paper is (noise, sampled_output)

        loss = self.model(sampled_output, noise = noise)

        return loss
