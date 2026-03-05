from __future__ import annotations

import math
from collections import namedtuple

import torch
from torch import pi, nn, tensor
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, repeat

from ema_pytorch import EMA
from rectified_flow_pytorch.fit import FiT

# constants

LossBreakdown = namedtuple('LossBreakdown', ['total', 'flow_loss', 'repr_loss'])

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def append_dims(tensor, num_dims):
    return tensor.reshape(*tensor.shape, *((1,) * num_dims))

# logit normal for sampling timesteps

def logit_normal_schedule(t, loc = 0.0, scale = 1.0):
    logits = torch.logit(t, eps = 1e-5)
    return 1. - torch.sigmoid(logits * scale + loc) # sticking with 0 -> 1 convention of noise to data

# default representation loss

def cosine_sim_loss(x, y):
    return 1. - F.cosine_similarity(x, y, dim = -1).mean()

# main class

class SelfFlow(Module):
    def __init__(
        self,
        model: Module,
        teacher_model: Module | None = None,
        times_cond_kwarg = 'times',
        data_shape: tuple[int, ...] | None = None,
        normalize_data_fn = identity,
        unnormalize_data_fn = identity,
        predict_clean = False,
        max_timesteps = 100,
        loss_fn = F.mse_loss,
        repr_loss_fn = cosine_sim_loss,
        repr_loss_weight = 1.0,
        alpha_shift = 0.5,
        mask_ratio = 0.5,
        num_patches: int | None = None,
        student_align_layer = -2,
        teacher_align_layer = -1,
        schedule_fn = logit_normal_schedule,
        eps = 1e-5,
        ema_kwargs = dict(
            beta = 0.999,
            update_every = 1
        )
    ):
        super().__init__()
        assert isinstance(model, FiT), "SelfFlow must receive a FiT instance for now."
        self.model = model

        self.teacher_model = default(teacher_model, EMA(model, **ema_kwargs))
        assert isinstance(self.teacher_model, EMA), 'teacher model must be an instance of EMA'

        self.times_cond_kwarg = times_cond_kwarg
        self.data_shape = data_shape

        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn

        self.predict_clean = predict_clean
        self.max_timesteps = max_timesteps

        self.loss_fn = loss_fn
        self.repr_loss_fn = repr_loss_fn
        self.repr_loss_weight = repr_loss_weight
        self.alpha_shift = alpha_shift
        self.mask_ratio = mask_ratio
        self.num_patches = num_patches
        self.schedule_fn = schedule_fn
        self.eps = eps

        self.student_align_layer = student_align_layer
        self.teacher_align_layer = teacher_align_layer

        # prediction head for representation alignment

        dim = model.dim
        self.projector = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.has_repr_loss = repr_loss_weight > 0.
        self.register_buffer('zero', tensor(0.), persistent = False)

    def post_training_step_update(self):
        self.teacher_model.update()

    @torch.no_grad()
    def sample(
        self,
        steps = 16,
        batch_size = 1,
        data_shape = None,
        return_noise = False,
        model = None,
        **kwargs
    ):
        model = default(model, 'teacher')
        assert model in ('student', 'teacher')

        selected_model = self.teacher_model if model == 'teacher' else self.model

        assert 1 <= steps <= self.max_timesteps

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'

        device = next(self.model.parameters()).device

        noise = torch.randn((batch_size, *data_shape), device = device)
        times = torch.linspace(0., 1., steps + 1, device = device)[:-1]

        delta_time = 1. / steps
        x = noise

        for t in times:
            t = repeat(t, '-> b', b = batch_size)
            padded_t = append_dims(t, x.ndim - 1)
            time_kwarg = {self.times_cond_kwarg: t} if exists(self.times_cond_kwarg) else dict()

            model_output = selected_model(x, **time_kwarg, **kwargs)
            pred = model_output[0] if isinstance(model_output, tuple) else model_output

            if self.predict_clean:
                pred = (pred - x) / (1. - padded_t).clamp(min = self.eps)

            x = x + delta_time * pred

        out = self.unnormalize_data_fn(x)
        return out if not return_noise else (out, noise)

    def forward(
        self,
        data,
        noise = None,
        return_loss_breakdown = False,
        **kwargs
    ):

        # variables

        shape, ndim, device = data.shape, data.ndim, data.device
        self.data_shape = default(self.data_shape, shape[1:])

        patch_size = self.model.patch_size
        batch, *_, height, width = shape
        grid_height, grid_width = height // patch_size, width // patch_size

        # normalize

        data = self.normalize_data_fn(data)

        # times

        teacher_time = self.schedule_fn(torch.rand(batch, device = device))
        student_time = self.schedule_fn(torch.rand(batch, device = device))

        times_clean_teacher = torch.maximum(teacher_time, student_time)

        # noise and derive flow

        noise = default(noise, torch.randn_like(data))
        flow = data - noise

        # Dual-Timestep Scheduling (Eq. 6 & 7) adapted for 0=noise, 1=data.

        teacher_time_patch = repeat(teacher_time, 'b -> b h w', h = grid_height, w = grid_width)
        student_time_patch = repeat(student_time, 'b -> b h w', h = grid_height, w = grid_width)

        # random mask for student

        mask = torch.rand((batch, grid_height, grid_width), device = device) < self.mask_ratio

        times_clean_student_patch = torch.where(mask, student_time_patch, teacher_time_patch)

        # times for the teacher

        if self.predict_clean:
            times_clean_teacher = times_clean_teacher * (1. - self.max_timesteps ** -1)

        # upsample times for pixel-wise lerp

        times_student_pixel = repeat(times_clean_student_patch, 'b h w -> b 1 (h p1) (w p2)', p1 = patch_size, p2 = patch_size)

        student_input = noise.lerp(data, times_student_pixel)

        # pass per-patch times to student model

        times_student_model = rearrange(times_clean_student_patch, 'b h w -> b (h w)')

        time_kwarg = {self.times_cond_kwarg: times_student_model} if exists(self.times_cond_kwarg) else dict()

        pred, student_hiddens = self.model(
            student_input,
            return_hiddens = True,
            **time_kwarg,
            **kwargs
        )

        if self.predict_clean:
            pred = (pred - student_input) / (1. - times_student_pixel).clamp(min = self.eps)

        # main loss

        flow_loss = self.loss_fn(flow, pred)

        # representation alignment loss

        repr_loss = self.zero

        if self.has_repr_loss:
            self.teacher_model.eval()

            with torch.no_grad():

                time_kwarg_teacher = {self.times_cond_kwarg: times_clean_teacher} if exists(self.times_cond_kwarg) else dict()

                times_clean_teacher_padded = append_dims(times_clean_teacher, data.ndim - 1)

                teacher_input = noise.lerp(data, times_clean_teacher_padded)

                _, teacher_hiddens = self.teacher_model(
                    teacher_input,
                    return_hiddens = True,
                    **time_kwarg_teacher,
                    **kwargs
                )

            student_repr = student_hiddens[self.student_align_layer]
            student_pred_teacher = self.projector(student_repr)

            teacher_repr = teacher_hiddens[self.teacher_align_layer]

            # repr align loss

            repr_loss = self.repr_loss_fn(student_pred_teacher, teacher_repr)

        total_loss = (
            flow_loss +
            repr_loss * self.repr_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, LossBreakdown(total_loss, flow_loss, repr_loss)
