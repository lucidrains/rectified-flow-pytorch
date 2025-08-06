# https://arxiv.org/abs/2507.16884

from random import random
from contextlib import nullcontext

import torch
from torch import tensor, ones, zeros
from torch.nn import Module
import torch.nn.functional as F

from einops import reduce

def exists(v):
    return v is not None

def xnor(x, y):
    return not (x ^ y)

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def append_dims(t, dims):
    shape = t.shape
    ones = ((1,) * dims)
    return t.reshape(*shape, *ones)

class SplitMeanFlow(Module):
    def __init__(
        self,
        model: Module,
        data_shape = None,
        normalize_data_fn = identity,
        unnormalize_data_fn = identity,
        use_adaptive_loss_weight = True,
        prob_default_flow_obj = 0.5,
        add_recon_loss = False,
        recon_loss_weight = 1.,
        accept_cond = False,
    ):
        super().__init__()
        self.model = model # model must accept three arguments in the order of (<noised data>, <times>, <integral start times>, <maybe condition?>)
        self.data_shape = data_shape

        # norm / unnorm data

        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn

        # they do 25-50% normal flow matching obj for boundary condition

        assert 0. <= prob_default_flow_obj <= 1.
        self.prob_default_flow_obj = prob_default_flow_obj

        # recon loss

        self.add_recon_loss = add_recon_loss and recon_loss_weight > 0
        self.recon_loss_weight = recon_loss_weight

        # accepting conditioning

        self.accept_cond = accept_cond

        self.register_buffer('dummy', tensor(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def get_noise(
        self,
        batch_size = 1,
        data_shape = None
    ):
        device = self.device

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'

        noise = torch.randn((batch_size, *data_shape), device = device)
        return noise

    @torch.no_grad()
    def slow_sample(
        self,
        steps = 16,
        batch_size = 1,
        noise = None,
        cond = None,
        data_shape = None,
        **kwargs
    ):
        assert steps >= 1

        device = self.device

        if not exists(noise):
            noise = self.get_noise(batch_size, data_shape = data_shape)

        times = torch.linspace(0., 1., steps + 1, device = device)[:-1]
        delta = 1. / steps

        denoised = noise

        maybe_cond = (cond,) if self.accept_cond else ()

        for time in times:
            time = time.expand(batch_size)
            pred_flow = self.model(denoised, time, time, *maybe_cond)
            denoised = denoised + delta * pred_flow

        return self.unnormalize_data_fn(denoised)

    def sample(
        self,
        batch_size = None,
        data_shape = None,
        requires_grad = False,
        cond = None,
        noise = None,
        steps = 2
    ):
        data_shape = default(data_shape, self.data_shape)

        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'

        # maybe condition

        maybe_cond = ()

        if self.accept_cond:
            batch_size = cond.shape[0]
            maybe_cond = (cond,)

        batch_size = default(batch_size, 1)

        # maybe slow sample

        assert steps >= 1

        if steps > 1:
            return self.slow_sample(
                batch_size = batch_size,
                data_shape = data_shape,
                cond = cond,
                steps = steps
            )

        assert xnor(self.accept_cond, exists(cond))
        device = next(self.model.parameters()).device

        context = nullcontext if requires_grad else torch.no_grad

        # Algorithm 2

        if not exists(noise):
            noise = self.get_noise(batch_size, data_shape = data_shape)

        with context():
            denoised = noise - self.model(noise, ones(batch_size, device = device), ones(batch_size, device = device), *maybe_cond)

        return self.unnormalize_data_fn(denoised)

    def forward(
        self,
        data,
        return_loss_breakdown = False,
        cond = None,
        noise = None
    ):
        assert xnor(self.accept_cond, exists(cond))

        data = self.normalize_data_fn(data)

        # shapes and variables

        shape, ndim = data.shape, data.ndim

        prob_time_end_start_same = self.prob_default_flow_obj

        self.data_shape = default(self.data_shape, shape[1:]) # store last data shape for inference
        batch, device = shape[0], data.device

        # flow logic

        times = torch.rand(batch, device = device)

        # some set prob of the time, normal flow matching training (times == start integral times)
        # this enforces the boundary condition u(zt, t, t) = v(zt, t)

        normal_flow_match_obj = prob_time_end_start_same > 0. and random() < prob_time_end_start_same

        if normal_flow_match_obj:
            integral_start_times = times # r = t for boundary condition
        else:
            integral_start_times = torch.rand(batch, device = device) * times # restrict range to [0, times]

        # derive flows

        if not exists(noise):
            noise = torch.randn_like(data)

        flow = data - noise # flow is the velocity from noise to data

        padded_times = append_dims(times, ndim - 1)
        noised_data = noise.lerp(data, padded_times) # noise the data with random amounts of noise (time) - lerp is read as noise -> data from 0. to 1.

        # condition the network on the delta times

        delta_times = times - integral_start_times # (t - r)

        # maybe condition

        maybe_cond = (cond,) if self.accept_cond else ()

        if normal_flow_match_obj:
            # normal flow matching for boundary condition u(zt, t, t) = v(zt, t)

            pred = self.model(noised_data, times, times, *maybe_cond)
            target = flow
        else:
            # algorithm 1 - interval splitting consistency

            # sample lambda uniformly from [0, 1]
            lambda_split = torch.rand(batch, device = device)
            
            # compute s = (1 - lambda) * t + lambda * r
            split_times = (1 - lambda_split) * times + lambda_split * integral_start_times
            
            # compute u(zt, s, t) - velocity from s to t
            delta_s_to_t = times - split_times

            with torch.no_grad():
                u2 = self.model(noised_data, times, delta_s_to_t, *maybe_cond).detach()
            
            # compute zs = zt - (t - s) * u2
            padded_delta_s_to_t = append_dims(delta_s_to_t, ndim - 1)
            noised_data_s = noised_data - padded_delta_s_to_t * u2 # detach for stop gradient
            
            # compute u(zs, r, s) - velocity from r to s
            delta_r_to_s = split_times - integral_start_times

            with torch.no_grad():
                u1 = self.model(noised_data_s, split_times, delta_r_to_s, *maybe_cond).detach()
            
            # the algebraic consistency target: u(zt, r, t) = (1 - lambda) * u1 + lambda * u2
            lambda_split_padded = append_dims(lambda_split, ndim - 1)
            target = (1 - lambda_split_padded) * u1 + lambda_split_padded * u2
            
            # predict u(zt, r, t)
            pred = self.model(noised_data, times, delta_times, *maybe_cond)

        flow_loss = F.mse_loss(pred, target)

        if not self.add_recon_loss:
            if not return_loss_breakdown:
                return flow_loss

            return flow_loss, (flow_loss,)

        # add predicted data recon loss, maybe adds stability

        if normal_flow_match_obj:
            pred_data = noised_data - pred * padded_times
        else:
            padded_delta_times = append_dims(delta_times, ndim - 1)
            pred_data = noised_data - pred * padded_delta_times
            
        recon_loss = F.mse_loss(pred_data, data)

        total_loss = (
            flow_loss +
            recon_loss * self.recon_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        loss_breakdown = (flow_loss, recon_loss)

        return total_loss, loss_breakdown
