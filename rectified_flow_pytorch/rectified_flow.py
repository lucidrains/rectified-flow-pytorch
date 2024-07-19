from __future__ import annotations
from typing import Tuple, Literal
from copy import deepcopy

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from torchdiffeq import odeint

import torchvision
from torchvision.models import VGG16_Weights

from einops import reduce, rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

# losses

class LPIPSLoss(Module):
    def __init__(
        self,
        vgg: Module | None = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
    ):
        super().__init__()

        if not exists(vgg):
            vgg = torchvision.models.vgg16(weights = vgg_weights)
            vgg.classifier = nn.Sequential(*vgg.classifier[:-2])

        self.vgg = vgg

    def forward(self, pred_data, data, reduction = 'mean'):
        embed = self.vgg(data)
        pred_embed = self.vgg(pred_data)
        loss = F.mse_loss(embed, pred_embed, reduction = reduction)

        if reduction == 'none':
            loss = reduce(loss, 'b ... -> b', 'mean')

        return loss

class PseudoHuberLoss(Module):
    def __init__(self, data_dim: int):
        super().__init__()
        self.data_dim = data_dim

    def forward(self, pred, target, reduction = 'mean', **kwargs):
        c = .00054 * self.data_dim
        loss = (F.mse_loss(pred, target, reduction = reduction) + c * c).sqrt() - c
        return loss

class PseudoHuberLossWithLPIPS(Module):
    def __init__(self, data_dim: int, lpips_kwargs: dict = dict()):
        super().__init__()
        self.pseudo_huber = PseudoHuberLoss(data_dim)
        self.lpips = LPIPSLoss(**lpips_kwargs)

    def forward(self, pred_flow, target_flow, *, times, data):
        huber_loss = self.pseudo_huber(pred_flow, target_flow, reduction = 'none')

        pred_data = pred_flow * times
        lpips_loss = self.lpips(data, pred_data, reduction = 'none')

        time_weighted_loss = huber_loss * (1 - times) + lpips_loss * (1. / times.clamp(min = 1e-2))
        return time_weighted_loss.mean()

class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)

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
        loss_fn: Literal['mse', 'pseudo_huber'] | Module = 'mse',
        loss_fn_kwargs: dict = dict(),
        data_shape: Tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.model = model
        self.time_cond_kwarg = time_cond_kwarg # whether the model is to be conditioned on the times

        # loss fn

        if isinstance(loss_fn, Module):
            loss_fn = loss_fn

        elif loss_fn == 'mse':
            loss_fn = MSELoss()

        elif loss_fn == 'pseudo_huber':
            # section 4.2 of https://arxiv.org/abs/2405.20320v1
            loss_fn = PseudoHuberLoss(**loss_fn_kwargs)

        elif loss_fn == 'pseudo_huber_with_lpips':
            loss_fn = PseudoHuberLossWithLPIPS(**loss_fn_kwargs)

        else:
            raise ValueError(f'unkwown loss function {loss_fn}')

        self.loss_fn = loss_fn

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

        loss = self.loss_fn(pred_flow, flow, times = times, data = data)

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
