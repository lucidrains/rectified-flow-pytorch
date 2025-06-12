# https://arxiv.org/abs/2505.13447

import torch
from torch import ones, zeros
from torch.nn import Module
import torch.nn.functional as F
from contextlib import nullcontext
from torch.autograd.functional import jvp

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

class MeanFlow(Module):
    def __init__(
        self,
        model: Module,
        data_shape = None,
        normalize_data_fn = identity,
        unnormalize_data_fn = identity,
        use_huber_loss = True,
        prob_default_flow_obj = 0.5
    ):
        super().__init__()
        self.model = model # model must accept three arguments in the order of (<noised data>, <times>, <integral start times>)
        self.data_shape = None

        self.use_huber_loss = use_huber_loss
        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn

        # they do 25-50% normal flow matching obj

        assert 0. <= prob_default_flow_obj <= 1.
        self.prob_default_flow_obj = prob_default_flow_obj

    def sample(
        self,
        batch_size = 1,
        data_shape = None,
        requires_grad = False
    ):
        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'
        device = next(self.model.parameters()).device

        context = nullcontext if not requires_grad else torch.no_grad

        # Algorithm 2

        with context():
            noise = torch.randn((batch_size, *self.data_shape), device = device)

            denoised = noise - self.model(noise, ones(batch_size, device = device), zeros(batch_size, device = device))

        return self.unnormalize_data_fn(denoised)

    def forward(self, data):
        data = self.normalize_data_fn(data)

        # shapes and variables

        shape, ndim = data.shape, data.ndim

        prob_time_end_start_same = self.prob_default_flow_obj

        self.data_shape = default(self.data_shape, shape[1:]) # store last data shape for inference
        batch, device = shape[0], data.device

        # flow logic

        times = torch.rand(batch, device = device)
        integral_start_times = torch.rand(batch, device = device) * times # restrict range to [0, times]

        # some set prob of the time, normal flow matching training (times == start integral times)

        if prob_time_end_start_same > 0.:
            prob_same = torch.rand(batch, device = device) < prob_time_end_start_same
            integral_start_times = torch.where(prob_same, times, integral_start_times)

        # derive flows

        noise = torch.randn_like(data)
        flow = noise - data # flow is the velocity from data to noise, also what the model is trained to predict

        padded_times, padded_start_times = tuple(append_dims(t, ndim - 1) for t in (times, integral_start_times))
        noised_data = data.lerp(noise, padded_times) # noise the data with random amounts of noise (time) - from data -> noise from 0. to 1.

        # Algorithm 1

        pred, rate_avg_vel_change = jvp(
            self.model,
            (noised_data, times, integral_start_times),  # inputs
            (flow, ones(batch, device = device), zeros(batch, device = device)), # tangents
            create_graph = True
        )

        # the new proposed target

        target = flow - (padded_times - padded_start_times) * rate_avg_vel_change.detach()

        loss_fn = F.mse_loss if not self.use_huber_loss else F.huber_loss

        return loss_fn(pred, target)
