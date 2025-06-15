# https://arxiv.org/abs/2505.13447

from random import random
from contextlib import nullcontext

import torch
from torch import tensor, ones, zeros
from torch.nn import Module
import torch.nn.functional as F
from torch.func import jvp

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

class MeanFlow(Module):
    def __init__(
        self,
        model: Module,
        data_shape = None,
        normalize_data_fn = identity,
        unnormalize_data_fn = identity,
        use_adaptive_loss_weight = True,
        adaptive_loss_weight_p = 0.5, # 0.5 is approximately pseudo huber loss
        prob_default_flow_obj = 0.5,
        add_recon_loss = False,
        recon_loss_weight = 1.,
        accept_cond = False,
        noise_std_dev = 1.,
        eps = 1e-3
    ):
        super().__init__()
        self.model = model # model must accept three arguments in the order of (<noised data>, <times>, <integral start times>, <maybe condition?>)
        self.data_shape = data_shape

        self.use_adaptive_loss_weight = use_adaptive_loss_weight
        self.adaptive_loss_weight_p = adaptive_loss_weight_p
        self.eps = eps

        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn

        # they do 25-50% normal flow matching obj

        assert 0. <= prob_default_flow_obj <= 1.
        self.prob_default_flow_obj = prob_default_flow_obj

        # recon loss

        self.add_recon_loss = add_recon_loss and recon_loss_weight > 0
        self.recon_loss_weight = recon_loss_weight

        # accepting conditioning

        self.accept_cond = accept_cond

        self.noise_std_dev = noise_std_dev

    def sample(
        self,
        batch_size = None,
        data_shape = None,
        requires_grad = False,
        cond = None,
        noise = None
    ):
        assert xnor(self.accept_cond, exists(cond))

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'
        device = next(self.model.parameters()).device

        context = nullcontext if requires_grad else torch.no_grad

        # maybe condition

        maybe_cond = ()

        if self.accept_cond:
            batch_size = cond.shape[0]
            maybe_cond = (cond,)

        batch_size = default(batch_size, 1)

        # Algorithm 2

        if not exists(noise):
            noise = torch.randn((batch_size, *data_shape), device = device) * self.noise_std_dev

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

        normal_flow_match_obj = prob_time_end_start_same > 0. and prob_time_end_start_same < random()

        if normal_flow_match_obj:
            integral_start_times = times
        else:
            integral_start_times = torch.rand(batch, device = device) * times # restrict range to [0, times]

        # derive flows

        if not exists(noise):
            noise = torch.randn_like(data) * self.noise_std_dev

        flow = noise - data # flow is the velocity from data to noise, also what the model is trained to predict

        padded_times = append_dims(times, ndim - 1)
        noised_data = data.lerp(noise, padded_times) # noise the data with random amounts of noise (time) - from data -> noise from 0. to 1.

        # they condition the network on the delta time instead

        delta_times = times - integral_start_times
        padded_delta_times = append_dims(delta_times, ndim - 1)

        # model forward with maybe jvp

        inputs = (noised_data, times, delta_times)
        tangents = (flow, ones(batch, device = device), ones(batch, device = device))

        if self.accept_cond:
            inputs = (*inputs, cond)
            tangents = (*tangents, cond)

        if normal_flow_match_obj:
            # Normal flow matching without jvp 25-50% of the time

            delta_times.zero_()

            pred, rate_avg_vel_change = (
                self.model(*inputs),
                tensor(0., device = device)
            )
        else:
            # Algorithm 1

            pred, rate_avg_vel_change = jvp(
                self.model,
                inputs,
                tangents
            )

        # the new proposed target

        integral = padded_delta_times * rate_avg_vel_change.detach()

        target = flow - integral

        flow_loss = F.mse_loss(pred, target, reduction = 'none')

        # section 4.3

        if self.use_adaptive_loss_weight:
            flow_loss = reduce(flow_loss, 'b ... -> b', 'mean')

            p = self.adaptive_loss_weight_p
            loss_weight = 1. / (flow_loss + self.eps).pow(p)

            flow_loss = flow_loss * loss_weight.detach()

        flow_loss = flow_loss.mean()

        if not self.add_recon_loss:
            if not return_loss_breakdown:
                return flow_loss

            return flow_loss, (flow_loss,)

        # add predicted data recon loss, maybe adds stability, not sure

        pred_data = noised_data - (pred + integral) * padded_times
        recon_loss = F.mse_loss(pred_data, data)

        total_loss = (
            flow_loss +
            recon_loss * self.recon_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        loss_breakdown = (flow_loss, recon_loss)

        return total_loss, loss_breakdown
