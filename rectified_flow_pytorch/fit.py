from __future__ import annotations

import math

import torch
from torch.nn import Module, ModuleList
from torch import nn, cat, zeros

import einx
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from x_transformers import Encoder

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# time conditioning

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        emb = rearrange(x, '... -> ... 1') * emb
        return cat((emb.sin(), emb.cos()), dim = -1)

class DepthWiseConv2d(Module):
    def __init__(
        self,
        dim,
        *,
        depth = 2,
        kernel_size = 3
    ):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, padding = kernel_size // 2, groups = dim),
                nn.SiLU()
            ))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

# main class

class FiT(Module):

    def __init__(
        self,
        *,
        dim,
        patch_size,
        channels = 3,
        depth = 12,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        conv_pos_emb_depth = 2
    ):
        super().__init__()
        self.dim = dim
        self.channels = channels
        self.patch_size = patch_size

        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # positional embeddings via depthwise convolutions

        self.pre_conv = DepthWiseConv2d(dim, depth = conv_pos_emb_depth)
        self.post_conv = DepthWiseConv2d(dim, depth = conv_pos_emb_depth)

        time_dim = dim * 4

        self.time_cond_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.transformer = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            ff_mult = ff_mult,
            use_adaptive_rmsnorm = True,
            dim_condition = time_dim
        )

        self.to_pixels = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim)
        )

    def forward(
        self,
        x,
        times = None,
        return_hiddens = False
    ):
        patch_size = self.patch_size
        batch, c, height, width = x.shape
        grid_height, grid_width = height // patch_size, width // patch_size

        assert divisible_by(height, patch_size) and divisible_by(width, patch_size), f'height {height} and width {width} must be divisible by patch size {patch_size}'

        if not exists(times):
            times = torch.zeros((batch,), device = x.device)

        if times.ndim == 0:
            # times: ()
            times = repeat(times, '-> b', b = batch)

        if times.ndim == 2:
            # times: (batch, num_patches)
            assert times.shape == (batch, grid_height * grid_width), f'times shape {times.shape} must match (batch, {grid_height * grid_width})'

        if times.ndim == 3:
            # times: (batch, grid_height, grid_width)
            assert times.shape == (batch, grid_height, grid_width), f'times shape {times.shape} must match (batch, {grid_height}, {grid_width})'
            times = rearrange(times, 'b h w -> b (h w)')

        tokens = self.to_patch_embedding(x)

        # pre-transformer depthwise convs for positional embeddings

        tokens_spatial = rearrange(tokens, 'b (h w) d -> b d h w', h = grid_height, w = grid_width)
        tokens_spatial = self.pre_conv(tokens_spatial)
        tokens = rearrange(tokens_spatial, 'b d h w -> b (h w) d')

        # embed time

        cond_emb = self.time_cond_mlp(times)

        out = self.transformer(
            tokens,
            condition = cond_emb,
            return_hiddens = return_hiddens
        )

        if return_hiddens:
            out, intermediates = out

        # post-transformer depthwise convs for positional embeddings

        out_spatial = rearrange(out, 'b (h w) d -> b d h w', h = grid_height, w = grid_width)
        out_spatial = self.post_conv(out_spatial)
        out = rearrange(out_spatial, 'b d h w -> b (h w) d')

        # to pixels

        out = self.to_pixels(out)

        out = rearrange(out, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = grid_height, w = grid_width, p1 = patch_size, p2 = patch_size)

        if not return_hiddens:
            return out

        return out, intermediates.hiddens
