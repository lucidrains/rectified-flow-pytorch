from __future__ import annotations

import math
from copy import deepcopy
from collections import namedtuple
from typing import Literal, Callable

import torch
from torch import Tensor
from torch import nn, pi, cat, stack, from_numpy
from torch.nn import Module, ModuleList
from torch.distributions import Normal
import torch.nn.functional as F

from torchdiffeq import odeint
from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

# normalizing helpers

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# noise schedules

def cosmap(t):
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))


LossBreakdown = namedtuple('LossBreakdown', ['total', 'main', 'guidance'])



class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)



class RectifiedFlow(Module):
    def __init__(
        self,
        model: dict | Module,
        time_cond_kwarg: str | None = 'times',
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        noise_schedule: Literal[
            'cosmap'
        ] | Callable = identity,
        data_shape: tuple[int, ...] | None = None,

        max_timesteps = 100,
        data_normalize_fn = normalize_to_neg_one_to_one,
        data_unnormalize_fn = unnormalize_to_zero_to_one,
        clip_during_sampling = False,
        clip_values: tuple[float, float] = (-1., 1.),
        clip_flow_during_sampling = None, 
        clip_flow_values: tuple[float, float] = (-3., 3),
        eps = 5e-3
    ):
        super().__init__()


        self.model = model
        self.time_cond_kwarg = time_cond_kwarg # whether the model is to be conditioned on the times

        # allow for mean variance output prediction

        self.max_timesteps = max_timesteps # when predicting clean, just make sure time never more than 1. - (1. / max_timesteps)

        loss_fn = MSELoss()

        self.loss_fn = loss_fn

        # noise schedule
      
        if noise_schedule == 'cosmap':
            noise_schedule = cosmap
        self.noise_schedule = noise_schedule

        # sampling

        self.odeint_kwargs = odeint_kwargs
        self.data_shape = data_shape

        # clipping for epsilon prediction

        self.clip_during_sampling = clip_during_sampling
        self.clip_flow_during_sampling = clip_flow_during_sampling

        self.clip_values = clip_values
        self.clip_flow_values = clip_flow_values

        # normalizing fn

        self.data_normalize_fn = default(data_normalize_fn, identity)
        self.data_unnormalize_fn = default(data_unnormalize_fn, identity)

        # epsilon for noise objective

        self.eps = eps

        self.z_net = deepcopy(self.model)
        self.guidance_net = deepcopy(self.model)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict_flow(self, model: Module, noised, *, times, **model_kwargs):
        """
        returns the model output as well as the derived flow, depending on the `predict` objective
        """

        batch = noised.shape[0]

        # prepare maybe time conditioning for model

        time_kwarg = self.time_cond_kwarg

        if exists(time_kwarg):
            times = rearrange(times, '... -> (...)')

            if times.numel() == 1:
                times = repeat(times, '1 -> b', b = batch)

            model_kwargs.update(**{time_kwarg: times})


        output = self.model(noised, **model_kwargs)

        return output

    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        steps = 16,
        noise = None,
        data_shape: tuple[int, ...] | None = None,
        temperature: float = 1.,
        guidance_scale: float = 1.0,  # New parameter for guidance strength
        **model_kwargs
    ):
   
        was_training = self.training
        self.eval()

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least with one forward'

        # clipping still helps for predict noise objective
        # much like original ddpm paper trick

        maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_during_sampling else identity

        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity

        # ode step function

        def ode_fn(t, x):
            x = maybe_clip(x)

            pred_flow = self.predict_flow(self.model, x, times = t, **model_kwargs)

            if guidance_scale > 0:

                # Calculate Guidance Direction (G_psi)
                g_pred = self.predict_flow(self.guidance_net, x, times=t, **model_kwargs)

                # Modified flow = Original Flow + Scale * Guidance Vector
                flow = pred_flow + guidance_scale * g_pred
            else:
                flow = pred_flow

            flow = maybe_clip_flow(flow)

            return flow

        # start with random gaussian noise - y0

        noise = default(noise, torch.randn((batch_size, *data_shape), device = self.device))

        # time steps

        times = torch.linspace(0., 1., steps, device = self.device)

        # ode

        trajectory = odeint(ode_fn, noise, times, **self.odeint_kwargs)

        sampled_data = trajectory[-1]

        self.train(was_training)

        return self.data_unnormalize_fn(sampled_data)

    def forward(
        self,
        data,
        noise: Tensor | None = None,
        rewards: Tensor | None = None, # (batch,) - The discounted cumulative returns 
        return_loss_breakdown = False,
        **model_kwargs
    ):
        batch, *data_shape = data.shape

        data = self.data_normalize_fn(data)

        self.data_shape = default(self.data_shape, data_shape)

        noise = default(noise, torch.randn_like(data))

        times = torch.rand(batch, device = self.device)

        padded_times = append_dims(times, data.ndim - 1)


        def get_noised_and_flows(model, t):

            t = self.noise_schedule(t)

            noised = noise.lerp(data, t) # noise -> data from 0. to 1.

            # the model predicts the flow from the noised data

            flow = data - noise

            pred_flow  = self.predict_flow(model, noised, times = t, **model_kwargs)

            return pred_flow, flow, noised, t


        pred_flow, target, x_t, padded_times = get_noised_and_flows(self.model, padded_times)


        main_loss = self.loss_fn(pred_flow, target, times = times)

        # guidance loss 

        if not exists(rewards):
            raise ValueError("You must provide 'rewards' for guidance training.")
        
    
        target_energy = append_dims(torch.exp(rewards), data.ndim - 1)
        
        z_output = self.predict_flow(self.z_net, x_t, times=padded_times, **model_kwargs)

        z_pred_flat = reduce(z_output, 'b ... -> b', 'mean')
        z_pred = append_dims(z_pred_flat, z_output.ndim - 1)
        
        weight = target_energy / (z_pred.detach().clamp(min=1e-6))
        
        g_pred = self.predict_flow(self.guidance_net, x_t, times=padded_times, **model_kwargs)
        
        target_guidance = (weight - 1) * pred_flow.detach()

        loss_z = F.mse_loss(z_pred, target_energy)
        loss_g = F.mse_loss(g_pred, target_guidance)

        total_loss = main_loss + loss_z + loss_g 

        if not return_loss_breakdown:
            return total_loss

        return total_loss, LossBreakdown(total_loss, main_loss, loss_z + loss_g)