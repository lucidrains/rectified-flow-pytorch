from inspect import signature

import torch
from torch import cat, is_tensor
from torch.nn import Module
from torch.utils._pytree import tree_map, tree_flatten
from einops import repeat, reduce, rearrange

# explorative modeling (forward xm)
# Alexi Gladstone et al. https://arxiv.org/abs/2607.27372
# antithetic noise pairing - Jia et al. https://arxiv.org/abs/2506.06185

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_batch_tensor(t, b):
    return is_tensor(t) and t.ndim > 0 and t.shape[0] == b

def batch_and_first_tensor(args, kwargs):
    leaves, _ = tree_flatten((args, kwargs))
    first_tensor = next(t for t in leaves if is_tensor(t))
    return first_tensor.shape[0], first_tensor

def sample_antithetic_noise(batch, candidates, shape, *, device, dtype = torch.float32):
    half = (candidates + 1) // 2
    noise = torch.randn((batch, half, *shape), device = device, dtype = dtype)
    return cat((noise, -noise), dim = 1)[:, :candidates]

# xm wrapper

class XMWrapper(Module):
    def __init__(
        self,
        flow_model: Module,
        candidates = 1,
        max_batch_size = None,
        antithetic_noise = True,
        noise_kwarg = 'noise'
    ):
        super().__init__()
        self.flow_model = flow_model

        assert candidates >= 1, 'candidates must be at least 1'
        self.candidates = candidates
        self.max_batch_size = max_batch_size
        self.antithetic_noise = antithetic_noise
        self.noise_kwarg = noise_kwarg

        self.forward_params = signature(flow_model.forward).parameters
        self.has_loss_reduction = 'loss_reduction' in self.forward_params
        self.accepts_noise = exists(noise_kwarg) and noise_kwarg in self.forward_params

    @property
    def data_shape(self):
        return getattr(self.flow_model, 'data_shape', None)

    @data_shape.setter
    def data_shape(self, val):
        if hasattr(self.flow_model, 'data_shape'):
            self.flow_model.data_shape = val

    def sample(self, *args, **kwargs):
        return self.flow_model.sample(*args, **kwargs)

    def sample_noise(self, batch, candidates, ref_tensor, antithetic_noise = None):
        antithetic_noise = default(antithetic_noise, self.antithetic_noise)
        shape = default(self.data_shape, ref_tensor.shape[1:])
        dtype = ref_tensor.dtype if ref_tensor.is_floating_point() else torch.float32
        device = ref_tensor.device

        if not antithetic_noise or candidates <= 1:
            return torch.randn((batch, candidates, *shape), device = device, dtype = dtype)

        return sample_antithetic_noise(batch, candidates, shape, device = device, dtype = dtype)

    def best_of_k_losses(self, args, kwargs, batch, candidates, max_batch_size):
        total = batch * candidates
        chunk_size = default(max_batch_size, total) if self.has_loss_reduction else 1
        extra_kwargs = {'loss_reduction': 'none'} if self.has_loss_reduction else dict()

        losses = []

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)

            chunk_args, chunk_kwargs = tree_map(
                lambda t: t[start:end] if is_batch_tensor(t, total) else t,
                (args, kwargs)
            )
            chunk_kwargs.update(extra_kwargs)

            loss = self.flow_model(*chunk_args, **chunk_kwargs)
            loss = rearrange(loss, '-> 1 1') if loss.ndim == 0 else rearrange(loss, 'b ... -> b (...)')
            losses.append(loss)

        raw_loss = cat(losses, dim = 0)
        return reduce(raw_loss, '(b k) ... -> b k', 'mean', b = batch, k = candidates)

    def forward(
        self,
        *args,
        candidates = None,
        max_batch_size = None,
        antithetic_noise = None,
        **kwargs
    ):
        candidates = default(candidates, self.candidates)
        max_batch_size = default(max_batch_size, self.max_batch_size)
        antithetic_noise = default(antithetic_noise, self.antithetic_noise)

        if candidates == 1:
            return self.flow_model(*args, **kwargs)

        # find batch size from first tensor in inputs

        batch, first_tensor = batch_and_first_tensor(args, kwargs)

        if 'times' in self.forward_params and not exists(kwargs.get('times')):
            kwargs['times'] = torch.rand(batch, device = first_tensor.device)

        # repeat inputs K candidates times

        args_K, kwargs_K = tree_map(
            lambda t: repeat(t, 'b ... -> (b k) ...', k = candidates) if is_batch_tensor(t, batch) else t,
            (args, kwargs)
        )

        # maybe sample antithetic noise pairs (+z, -z) across candidates

        if antithetic_noise and self.accepts_noise and not exists(kwargs.get(self.noise_kwarg)):
            noise = self.sample_noise(batch, candidates, first_tensor, antithetic_noise = True)
            kwargs_K[self.noise_kwarg] = rearrange(noise, 'b k ... -> (b k) ...')

        # select candidate with minimum loss for each sample in batch

        candidate_losses = self.best_of_k_losses(args_K, kwargs_K, batch, candidates, max_batch_size)

        return candidate_losses.amin(dim = -1).mean()
