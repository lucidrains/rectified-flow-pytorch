from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, einsum
from einops.layers.torch import Rearrange
from torch_einops_utils import pad_right_ndim_to
from torch_einops_utils.shape import shape, size

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def ensure_tuple(t):
    return (t,) if not isinstance(t, (tuple, list)) else tuple(t)

def resolve_layer(model: Module, layer: Module | str | None, default_path: str) -> Module:
    layer = default(layer, default_path)
    return model.get_submodule(layer) if isinstance(layer, str) else layer

def resolve_dim(layer: Module, explicit_dim: int | None = None) -> int:
    if exists(explicit_dim):
        return explicit_dim

    # last conv or linear is the output projection of the layer, even when wrapped, as in Residual(branch = ResnetBlock)

    for module in reversed(list(layer.modules())):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            return module.weight.shape[0]

    raise ValueError(f'could not infer feature dimension of {type(layer).__name__}, please pass read_dim / write_dim')

# n-adic recurrent state memory
#
# the state is order n + 1 per head, for n key / query projections
#   d            ->  state (heads, d, value_dim)          dyadic, a matrix state
#   d1, d2       ->  state (heads, d1, d2, value_dim)     triadic [default]
#   d1, d2, d3.. ->  quadradic and beyond
#
# read takes a feature and the state, returning the feature with the memory read added as a residual
# write takes a feature, returning the next state, with state dropout / jitter applied while training
#
# references
#   Recurrent Interface Networks - https://arxiv.org/abs/2212.11972
#   Triadic Linear Attention - https://arxiv.org/abs/2609.36529
#   Full-Bandwidth Transformer - https://arxiv.org/abs/2608.08888

class RecurrentStateMemory(Module):
    def __init__(
        self,
        read_dim,
        write_dim = None,
        *,
        num_heads = 4,
        key_dims: tuple[int, ...] | int = (16, 4),
        value_dim = None,
        state_dropout = 0.5,
        jitter_noise_std = 0.,
        detach_state = False,
        channels_first: bool | tuple[bool, bool] = True,
        spatial_dims = 2
    ):
        super().__init__()

        write_dim = default(write_dim, read_dim)

        key_dims = ensure_tuple(key_dims)
        value_dim = default(value_dim, key_dims[0])

        self.num_heads = num_heads
        self.key_dims = key_dims
        self.value_dim = value_dim

        self.query_split_sizes = tuple(num_heads * d for d in key_dims)
        self.write_split_sizes = (*self.query_split_sizes, num_heads * value_dim)

        total_query_dim = sum(self.query_split_sizes)
        total_out_dim = num_heads * value_dim

        # one key axis per key dim for the n-adic state einsums, e.g. for key_dims (d0, d1)
        # read  - 'b h ... d0, b h ... d1, b h d0 d1 v -> b h ... v'
        # write - 'b h ... d0, b h ... d1, b h ... v -> b h d0 d1 v'

        key_operands = ', '.join(f'b h ... d{i}' for i in range(len(key_dims)))
        key_axes = ' '.join(f'd{i}' for i in range(len(key_dims)))

        self.read_pattern = f'{key_operands}, b h {key_axes} v -> b h ... v'
        self.write_pattern = f'{key_operands}, b h ... v -> b h {key_axes} v'

        # the head split pattern depends on where the projection axis sits, first for conv (channels first), last for linear (channels last)

        self.read_channels_first, self.write_channels_first = (channels_first, channels_first) if isinstance(channels_first, bool) else channels_first

        self.read_head_pattern = 'b (h d) ... -> b h ... d' if self.read_channels_first else 'b ... (h d) -> b h ... d'
        self.write_head_pattern = 'b (h d) ... -> b h ... d' if self.write_channels_first else 'b ... (h d) -> b h ... d'

        # projections

        conv_klass = nn.Conv1d if spatial_dims == 1 else nn.Conv2d

        def create_proj(dim_in, dim_out, channels_first):
            return conv_klass(dim_in, dim_out, 1, bias = False) if channels_first else nn.Linear(dim_in, dim_out, bias = False)

        self.mat_q = create_proj(read_dim, total_query_dim, self.read_channels_first)
        self.mat_out = create_proj(total_out_dim, read_dim, self.read_channels_first)
        self.mat_write = create_proj(write_dim, total_query_dim + total_out_dim, self.write_channels_first)

        self.merge_heads = Rearrange('b h ... v -> b (h v) ...' if self.read_channels_first else 'b h ... v -> b ... (h v)')

        for mat in (self.mat_q, self.mat_out, self.mat_write):
            nn.init.normal_(mat.weight, std = 0.02)

        # carried state regularization

        self.state_dropout = state_dropout
        self.jitter_noise_std = jitter_noise_std
        self.detach_state = detach_state

    def init_state(self, batch_size, device = None):
        return torch.zeros(batch_size, self.num_heads, *self.key_dims, self.value_dim, device = device)

    def read(self, x, state):
        # x - (b read_dim ...) channels first or (b ... read_dim) channels last, state - (b heads *key_dims value_dim)

        split_dim = 1 if self.read_channels_first else -1

        queries = self.mat_q(x).split(self.query_split_sizes, dim = split_dim)
        queries = [F.normalize(rearrange(t, self.read_head_pattern, h = self.num_heads), dim = -1) for t in queries]

        read_out = einsum(*queries, state, self.read_pattern)

        return x + self.mat_out(self.merge_heads(read_out))

    def write(self, x):
        split_dim = 1 if self.write_channels_first else -1

        normed = F.group_norm(x, 1) if self.write_channels_first else F.layer_norm(x, (x.shape[-1],))
        *keys, values = self.mat_write(normed).split(self.write_split_sizes, dim = split_dim)

        keys = [F.normalize(rearrange(k, self.write_head_pattern, h = self.num_heads), dim = -1) for k in keys]
        values = rearrange(values, self.write_head_pattern, h = self.num_heads)

        n = math.prod(shape(values, 'b h ... v').ellipsis)  # number of positions

        state = einsum(*keys, values, self.write_pattern) / math.sqrt(n)

        # state dropout / jitter, as in FBT eq. 11, only while training

        if not self.training:
            return state

        if self.detach_state:
            state = state.detach()

        if self.state_dropout > 0.:
            keep = torch.rand(size(state, '[b] ...'), device = state.device) >= self.state_dropout
            state = state * pad_right_ndim_to(keep, state.ndim)

        if self.jitter_noise_std > 0.:
            state = state + torch.randn_like(state) * self.jitter_noise_std

        return state

# recurrent state nano flow
#
# nano flow wrapping a RecurrentStateMemory, read from and written to any two layers of the network through forward hooks, so any architecture works

class RecurrentStateNanoFlow(Module):
    def __init__(
        self,
        model: Module,
        *,
        # memory config
        num_heads: int = 4,
        key_dims: tuple[int, ...] | int = (16, 4),
        value_dim: int | None = None,
        state_dropout: float = 0.5,
        ablate: bool = False,
        # which two network layers to read from and write to
        read_layer: Module | str | None = None,
        write_layer: Module | str | None = None,
        read_dim: int | None = None,
        write_dim: int | None = None,
        channels_first: bool | tuple[bool, bool] = True,
        # recurrence training, full-bandwidth transformer multi-pass style
        num_train_steps: int = 1,
        rollout_mode: str = 'flow',
        detach_state: bool = False,
        jitter_noise_std: float = 0.,
        # flow config
        data_shape = None,
        predict_clean: bool = True,
        loss_fn: Callable = F.mse_loss,
        normalize_data_fn: Callable = identity,
        unnormalize_data_fn: Callable = identity,
        times_cond_kwarg: str = 'times'
    ):
        super().__init__()

        self.model = model
        self.ablate = ablate

        # recurrence training
        # num_train_steps 1 is a same-time dual pass, 2 and beyond is an ordered rollout t1 < t2 < ... carrying the state forward
        # rollout_mode 'flow' teacher forces the flow path at each step, 'euler' rolls out from the previous prediction

        self.num_train_steps = num_train_steps
        assert rollout_mode in ('flow', 'euler')
        self.rollout_mode = rollout_mode

        # flow config

        self.data_shape = data_shape
        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn
        self.predict_clean = predict_clean
        self.loss_fn = loss_fn
        self.times_cond_kwarg = times_cond_kwarg

        # memory, and the transient state for the forward hooks

        self.memory = None
        self.state = None
        self.next_state = None

        if not ablate:
            read_layer = resolve_layer(model, read_layer, 'init_conv')
            write_layer = resolve_layer(model, write_layer, 'mid_block2')

            # 1d data uses conv1d projections, otherwise assume 2d

            spatial_dims = len(self.data_shape) - 1 if exists(self.data_shape) else 2

            self.memory = RecurrentStateMemory(
                resolve_dim(read_layer, read_dim),
                resolve_dim(write_layer, write_dim),
                num_heads = num_heads,
                key_dims = key_dims,
                value_dim = value_dim,
                state_dropout = state_dropout,
                detach_state = detach_state,
                jitter_noise_std = jitter_noise_std,
                channels_first = channels_first,
                spatial_dims = spatial_dims
            )

            read_layer.register_forward_hook(self.memory_read_hook)
            write_layer.register_forward_hook(self.memory_write_hook)

    @property
    def device(self):
        return next(self.parameters()).device

    def init_state(self, batch_size, device = None):
        if not exists(self.memory):
            return None
        return self.memory.init_state(batch_size, device)

    # forward hooks

    def memory_read_hook(self, _, __, output):
        if not exists(self.state):
            return output

        is_tuple = isinstance(output, tuple)
        feat = output[0] if is_tuple else output

        out = self.memory.read(feat, self.state)
        return (out, *output[1:]) if is_tuple else out

    def memory_write_hook(self, _, __, output):
        feat = output[0] if isinstance(output, tuple) else output
        self.next_state = self.memory.write(feat)

    # flow

    def flow_from_clean(self, pred_clean, noised, time, eps = 1e-5):
        # rectified flow velocity from a clean prediction - (x0 - xt) / (1 - t)
        padded_time = pad_right_ndim_to(time, noised.ndim)
        return (pred_clean - noised) / (1. - padded_time).clamp(min = eps)

    def forward_model(self, x, times = None, state = None, **kwargs):
        self.state = state
        self.next_state = None

        time_kwarg = {self.times_cond_kwarg: times} if exists(self.times_cond_kwarg) and exists(times) else dict()
        out = self.model(x, **time_kwarg, **kwargs)

        next_state, self.state, self.next_state = self.next_state, None, None
        return out, next_state

    def predict_flow(self, x, time, state = None, eps = 1e-5, **kwargs):
        model_output, next_state = self.forward_model(x, times = time, state = state, **kwargs)

        if not self.predict_clean:
            return model_output, next_state

        return self.flow_from_clean(model_output, x, time, eps = eps), next_state

    @torch.no_grad()
    def sample(
        self,
        steps = 16,
        batch_size = 16,
        data_shape = None,
        noise = None,
        use_state = None,
        ablate: bool | None = None,
        eps = 1e-5,
        **kwargs
    ):
        self.eval()

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'data shape must be passed in, or known from training'

        ablate = default(ablate, self.ablate)
        use_state = default(use_state, not ablate) and not ablate
        device = self.device

        x = default(noise, torch.randn((batch_size, *data_shape), device = device))
        state = self.init_state(batch_size, device) if use_state else None

        times = torch.linspace(0., 1., steps + 1, device = device)[:-1]
        step_dt = 1. / steps

        for time in times:
            flow, next_state = self.predict_flow(x, time.expand(batch_size), state = state, eps = eps, **kwargs)
            x = x + step_dt * flow

            if use_state:
                state = default(next_state, state)

        return self.unnormalize_data_fn(x)

    def forward(
        self,
        data,
        noise = None,
        times = None,
        num_train_steps = None,
        ablate: bool | None = None,
        **kwargs
    ):
        if isinstance(data, (tuple, list)):
            data = data[0]

        data = self.normalize_data_fn(data)
        b = size(data, '[b] ...')
        device = data.device

        self.data_shape = default(self.data_shape, data.shape[1:])

        ablate = default(ablate, self.ablate)
        num_train_steps = default(num_train_steps, self.num_train_steps)
        noise = default(noise, torch.randn_like(data))

        # single step - same-time dual pass, base pass then memory feedback pass

        if num_train_steps == 1:
            times = default(times, torch.rand(b, device = device))
            times = times if times.ndim == 1 else times[:, 0]

            xt = noise.lerp(data, pad_right_ndim_to(times, data.ndim))

            pred, state = self.forward_model(xt, times = times, state = None, **kwargs)
            loss = self.loss_fn(pred, data)

            if ablate:
                return loss

            pred, _ = self.forward_model(xt, times = times, state = state, **kwargs)

            return 0.5 * (loss + self.loss_fn(pred, data))

        # multiple steps - state carried across ordered times t1 < t2 < ...

        if not exists(times):
            times = torch.rand(b, num_train_steps, device = device).sort(dim = -1).values
        elif times.ndim == 1:
            dt = (1. - times) / num_train_steps
            times = torch.stack([times + step * dt for step in range(num_train_steps)], dim = -1).clamp(0., 1.)
        else:
            times = times.sort(dim = -1).values

        total_loss = 0.
        state, xt, pred = None, None, None

        for step in range(num_train_steps):
            t = times[:, step]

            # teacher forced flow interpolation, or euler rollout from the previous prediction

            if step == 0 or self.rollout_mode == 'flow':
                xt = noise.lerp(data, pad_right_ndim_to(t, data.ndim))
            else:
                t_prev = times[:, step - 1]
                flow = self.flow_from_clean(pred, xt, t_prev)
                xt = (xt + pad_right_ndim_to(t - t_prev, data.ndim) * flow).detach()

            pred, next_state = self.forward_model(xt, times = t, state = state, **kwargs)
            total_loss = total_loss + self.loss_fn(pred, data)

            state = None if ablate else next_state

        return total_loss / num_train_steps

# quick test

if __name__ == '__main__':
    # recurrent state memory standalone (triadic: dual queries and keys -> order 3 tensor state per head)

    mem = RecurrentStateMemory(
        read_dim = 32,
        write_dim = 64,
        num_heads = 4,
        key_dims = (7, 3),
        value_dim = 11
    )

    feat = torch.randn(2, 32, 8, 8)
    state = mem.init_state(2)

    read_feat = mem.read(feat, state)
    next_state = mem.write(torch.randn(2, 64, 8, 8))

    assert next_state.shape == (2, 4, 7, 3, 11) # (batch, heads, key1, key2, value)

    # recurrent state nano flow

    model = nn.Conv2d(3, 3, 1)

    flow = RecurrentStateNanoFlow(
        model,
        times_cond_kwarg = None,
        data_shape = (3, 16, 16),
        read_layer = model,
        write_layer = model
    )

    data = torch.randn(16, 3, 16, 16)

    loss = flow(data)
    loss.backward()

    loss_ablated = flow(data, ablate = True)
    assert exists(loss_ablated)

    sampled = flow.sample(batch_size = 16)
    assert sampled.shape == data.shape
