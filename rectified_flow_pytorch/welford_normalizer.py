from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def welford_combine(count, mean, m2, other_count, other_mean, other_m2):
    new_count = count + other_count
    delta = other_mean - mean

    new_mean = mean + delta * (other_count / new_count)
    new_m2 = m2 + other_m2 + delta.square() * (count * other_count / new_count)

    return new_count, new_mean, new_m2

def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

# running standardization with welford
#
# stats shape follows `dim`
#   images  (b c ...) -> per-channel stats, dim = 1
#   vectors (b d)     -> per-feature stats, dim = 1
#   dim = None        -> a single global mean / variance

class WelfordNormalizer(Module):
    def __init__(
        self,
        data_shape: tuple[int, ...] | None = None,
        dim: int | None = 1,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.stats_dtype = dtype
        self.frozen = False

        # identity until the first update

        self.register_buffer('mean', torch.zeros(1, dtype = dtype))
        self.register_buffer('m2', torch.zeros(1, dtype = dtype))
        self.register_buffer('count', torch.zeros((), dtype = dtype))

        if exists(data_shape):
            self._init_stats((1, *tuple(data_shape)))

    # stats shape

    def _resolve_dim(self, ndim):
        if not exists(self.dim):
            return None

        dim = self.dim + ndim if self.dim < 0 else self.dim
        assert 0 <= dim < ndim, f'dim {self.dim} is out of range for {ndim} dims'
        return dim

    def _stats_shape(self, shape):
        dim = self._resolve_dim(len(shape))

        if not exists(dim):
            return (1,) * len(shape)

        return tuple(size if i == dim else 1 for i, size in enumerate(shape))

    @torch.no_grad()
    def _init_stats(self, shape):
        self.mean = self.mean.new_zeros(self._stats_shape(shape))
        self.m2 = torch.zeros_like(self.mean)
        self.count = self.mean.new_zeros(())
        return self

    @torch.no_grad()
    def reset(self):
        self.mean.zero_()
        self.m2.zero_()
        self.count.zero_()
        return self

    # freeze helpers

    def freeze(self):
        self.frozen = True
        return self

    def unfreeze(self):
        self.frozen = False
        return self

    # welford update

    @torch.no_grad()
    def update(self, data: Tensor):
        if self.frozen or data.numel() == 0:
            return self

        # half precision stats are not worth the trouble

        if data.dtype in (torch.float16, torch.bfloat16):
            data = data.float()

        if self.mean.shape != self._stats_shape(data.shape):
            self._init_stats(data.shape)

        reduce_dims = tuple(i for i in range(data.ndim) if i != self._resolve_dim(data.ndim))

        batch_mean = data.mean(dim = reduce_dims, keepdim = True)
        batch_m2 = (data - batch_mean).square().sum(dim = reduce_dims, keepdim = True)
        batch_count = data.new_tensor(data.numel() / batch_mean.numel())

        # combine local batch moments across ranks

        if is_distributed():
            num_stats = batch_mean.numel()
            local_stats = torch.cat((batch_count.reshape(1), batch_mean.reshape(-1), batch_m2.reshape(-1)))

            if dist.get_backend() == 'nccl' and not local_stats.is_cuda:
                local_stats = local_stats.cuda()

            gathered = [torch.empty_like(local_stats) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, local_stats)

            total_count = data.new_tensor(0.)
            total_mean = torch.zeros_like(batch_mean)
            total_m2 = torch.zeros_like(batch_m2)

            for stats in gathered:
                other_count = stats[0].to(data.device)
                other_mean = stats[1:1 + num_stats].reshape_as(batch_mean).to(data.device)
                other_m2 = stats[1 + num_stats:].reshape_as(batch_m2).to(data.device)

                total_count, total_mean, total_m2 = welford_combine(
                    total_count, total_mean, total_m2,
                    other_count, other_mean, other_m2
                )

            batch_count, batch_mean, batch_m2 = total_count, total_mean, total_m2

        # merge into running stats

        count = self.count.to(data.device, data.dtype)
        mean = self.mean.to(data.device, data.dtype)
        m2 = self.m2.to(data.device, data.dtype)

        count, mean, m2 = welford_combine(count, mean, m2, batch_count, batch_mean, batch_m2)

        self.count = count.to(self.stats_dtype)
        self.mean = mean.to(self.stats_dtype)
        self.m2 = m2.to(self.stats_dtype)
        return self

    # normalization

    def _stats(self, device, dtype):
        mean = self.mean.to(device = device, dtype = dtype)
        m2 = self.m2.to(device = device, dtype = dtype)
        count = self.count.to(device = device, dtype = dtype)

        var = m2 / count.clamp_min(1.)
        std = (var + self.eps).sqrt()

        # identity until the first update

        std = torch.where(count > 0, std, torch.ones_like(std))

        return mean, std

    def normalize(self, data: Tensor):
        mean, std = self._stats(data.device, data.dtype)
        return (data - mean) / std

    def denormalize(self, data: Tensor):
        mean, std = self._stats(data.device, data.dtype)
        return data * std + mean

    def forward(self, data: Tensor, update: bool | None = None):
        if default(update, self.training):
            self.update(data)

        return self.normalize(data)
