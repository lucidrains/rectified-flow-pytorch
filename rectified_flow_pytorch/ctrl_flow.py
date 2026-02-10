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

import torchvision
from torchvision.utils import save_image

import einx
from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange
from torch.func import vmap, jacrev



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


LossBreakdown = namedtuple('LossBreakdown', ['total', 'main']) #, 'data_match', 'velocity_match'])



class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)



class RectifiedFlow(Module):
    def __init__(
        self,
        model: dict | Module,
        mean_variance_net: bool | None = None,
        time_cond_kwarg: str | None = 'times',
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        noise_schedule: Literal[
            'cosmap'
        ] | Callable = identity,
        loss_fn_kwargs: dict = dict(),
        ema_update_after_step: int = 100,
        ema_kwargs: dict = dict(),
        data_shape: tuple[int, ...] | None = None,
        immiscible = False,
        use_consistency = False,
        max_timesteps = 100,
        consistency_decay = 0.9999,
        consistency_velocity_match_alpha = 1e-5,
        consistency_delta_time = 1e-3,
        consistency_loss_weight = 1.,
        data_normalize_fn = normalize_to_neg_one_to_one,
        data_unnormalize_fn = unnormalize_to_zero_to_one,
        clip_during_sampling = False,
        clip_values: tuple[float, float] = (-1., 1.),
        clip_flow_during_sampling = None, # this seems to help a lot when training with predict epsilon, at least for me
        clip_flow_values: tuple[float, float] = (-3., 3),
        eps = 5e-3
    ):
        super().__init__()


        self.model = model
        self.time_cond_kwarg = time_cond_kwarg # whether the model is to be conditioned on the times

        # allow for mean variance output prediction

        self.mean_variance_net = self.model.mean_variance_net

        # objective - either flow or noise (proposed by Esser / Rombach et al in SD3)

        self.max_timesteps = max_timesteps # when predicting clean, just make sure time never more than 1. - (1. / max_timesteps)

        # automatically default to a working setting for predict epsilon

        # clip_flow_during_sampling = default(clip_flow_during_sampling, predict == 'noise')

        # loss fn

   
        loss_fn = MSELoss()

        self.loss_fn = loss_fn

        # noise schedule
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

        # consistency flow matching

        self.use_consistency = use_consistency
        self.consistency_decay = consistency_decay
        self.consistency_velocity_match_alpha = consistency_velocity_match_alpha
        self.consistency_delta_time = consistency_delta_time
        self.consistency_loss_weight = consistency_loss_weight

        self.immiscible = immiscible

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
        use_ema: bool = False,
        **model_kwargs
    ):
        use_ema = default(use_ema, self.use_consistency)
        assert not (use_ema and not self.use_consistency), 'in order to sample from an ema model, you must have `use_consistency` turned on'

        model = self.ema_model if use_ema else self.model

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

            _, output = self.predict_flow(model, x, times = t, **model_kwargs)

            flow = output

            if self.mean_variance_net:
                mean, std = output

                flow = torch.normal(mean, std * temperature)

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
        return_loss_breakdown = False,
        **model_kwargs
    ):
        batch, *data_shape = data.shape

        data = self.data_normalize_fn(data)

        self.data_shape = default(self.data_shape, data_shape)


        # x0 - gaussian noise, x1 - data

        noise = default(noise, torch.randn_like(data))

        # times, and times with dimension padding on right

        times = torch.rand(batch, device = self.device)

        # maybe cap times when predicting clean


        padded_times = append_dims(times, data.ndim - 1)


        def get_noised_and_flows(model, t):

            t = self.noise_schedule(t)

            noised = noise.lerp(data, t) # noise -> data from 0. to 1.

            # the model predicts the flow from the noised data

            flow = data - noise

            pred_flow  = self.predict_flow(model, noised, times = t, **model_kwargs)

            return pred_flow, flow, noised

        # getting flow and pred flow for main model

        pred_flow, target, x_t = get_noised_and_flows(self.model, padded_times)
        # determine target, depending on objective

        # losses

        main_loss = self.loss_fn(pred_flow, target, times = times)

        # guidance loss 
        
        # dummy reward for ctrl flow, just to test out the loss calculation
        rewards = torch.randn(batch, device = self.device)

        J_tau = -rewards 
        
        # exp(-J(tau)) which is exp(rewards)
        target_energy = append_dims(torch.exp(-J_tau), data.ndim - 1)
        
        z_output = self.z_net(x_t, times=times)

        z_pred_flat = reduce(z_output, 'b ... -> b', 'mean')
        z_pred = append_dims(z_pred_flat, z_output.ndim - 1)
        
        weight = target_energy / (z_pred.detach().clamp(min=1e-6))
        
        g_pred = self.guidance_net(x_t, times=times)
        
        target_guidance = (weight - 1) * pred_flow.detach()

        loss_z = F.mse_loss(z_pred, target_energy)
        loss_g = F.mse_loss(g_pred, target_guidance)

       # control loss

        projected_terminal_data = x_t + (1.0 - padded_times) * pred_flow

        loss_control_numerator = reduce((data - projected_terminal_data) ** 2, 'b ... -> b', 'sum')

        def get_velocity(x, t):
            t = self.noise_schedule(t)
            
            x_in = x.unsqueeze(0) 
            t_in = t.unsqueeze(0)
    
            flow = self.model(x_in, times=t_in)
            
            return flow.flatten()

        # calculate dv/dx
        jacobian_fn = vmap(jacrev(get_velocity, argnums=0))

        J_raw = jacobian_fn(x_t, padded_times)

        J_v = rearrange(J_raw, 'b d ... -> b d (...)')
        
        batch_size = J_v.shape[0]
        dim = J_v.shape[-1]
    
        I = repeat(torch.eye(dim, device=self.device), 'i j -> b i j', b=batch_size)
        
        t_broadcast = append_dims(times, 2)
    
        D_Phi = I + (1.0 - t_broadcast) * J_v

        Gramian = einsum(D_Phi, D_Phi, 'b i k, b j k -> b i j')

        eigvals = torch.linalg.eigvalsh(Gramian)
        
        lambda_min = eigvals[:, 0]

        loss_control = (loss_control_numerator / (lambda_min + 1e-6)).mean()

        total_loss = main_loss + loss_z + loss_g + loss_control

        if not return_loss_breakdown:
            return total_loss

        return total_loss, LossBreakdown(total_loss, main_loss)