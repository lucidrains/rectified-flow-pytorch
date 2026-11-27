# FlowOptimizer - Learning-to-Optimize via Deep Unfolded Flows
# Saravanos, So, Ahmad, Fan (ICML 2026 Spotlight)
# https://openreview.net/forum?id=ZOtOq7hxJP

from __future__ import annotations

import math
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
from torch import nn, cat, stack, zeros, linspace, rand, randn_like
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

from einops import rearrange, repeat, reduce, pack, unpack
from scipy.optimize import linear_sum_assignment
from x_transformers import Encoder

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def symlog(t):
    return t.sign() * torch.log1p(t.abs())

def softmin(f_vals, temp = 0.1, dim = -1):
    # normalized smooth minimum: min(f) <= softmin(f) <= mean(f), strictly monotone in each entry
    n = f_vals.shape[dim]
    return -temp * (torch.logsumexp(-f_vals / temp, dim = dim) - math.log(n))

def compute_population_stats(pop, eps = 1e-6):
    # pop: (b, p, d)
    mean = pop.mean(dim = 1)
    std = pop.std(dim = 1, unbiased = False).clamp(min = eps) if pop.shape[1] > 1 else torch.ones_like(mean) * eps
    return mean, std

def compute_obj_and_grad(
    pop,
    obj_fn: Callable,
    use_grad = True,
    backprop_through_grad = False
):
    # evaluates f(X) and optionally grad f(X) even inside torch.no_grad() inference blocks

    if not use_grad:
        with torch.no_grad():
            f_vals = obj_fn(pop)
        return f_vals.detach(), None

    with torch.enable_grad():
        if backprop_through_grad and pop.requires_grad:
            x_req = pop
            create_graph = True
        else:
            x_req = pop.detach().requires_grad_(True)
            create_graph = False

        f_vals = obj_fn(x_req)

        grads, = torch_grad(
            outputs = f_vals.sum(),
            inputs = x_req,
            create_graph = create_graph,
            retain_graph = create_graph,
            only_inputs = True
        )

    if not create_graph:
        f_vals = f_vals.detach()
        grads = grads.detach()

    grads = torch.nan_to_num(grads, nan = 0., posinf = 1e4, neginf = -1e4)
    return f_vals, grads

def clamp_bounds(x, bounds, straight_through = True):
    if not exists(bounds):
        return x

    low, high = bounds

    if isinstance(low, torch.Tensor):
        low = low.to(device = x.device, dtype = x.dtype)
    if isinstance(high, torch.Tensor):
        high = high.to(device = x.device, dtype = x.dtype)

    clamped = x.clamp(min = low, max = high)

    if straight_through and x.requires_grad:
        return x + (clamped - x).detach()

    return clamped

@torch.no_grad()
def improve_population(
    population,
    obj_fn: Callable,
    gamma = 50,
    std_scale = math.sqrt(3.0),
    min_std = 1e-2,
    keep_current = True,
    bounds = None,
    chunk_size = None
):
    """
    Construct improved target population X+ = Improve(X, f) (Section 3.5 & Appendix B.3).
    Samples gamma * P candidates from a Gaussian fitted to X with scaled std (default sqrt(3)
    corresponding to 3x covariance), evaluates f, and retains the top-P lowest-cost points.
    Supports shapes (b, p, d) or (p, d).
    """
    pop, ps = pack([population], '* p d')
    batch, pop_size, dim = pop.shape

    mean = pop.mean(dim = 1, keepdim = True)
    std = pop.std(dim = 1, unbiased = False, keepdim = True) if pop_size > 1 else torch.ones_like(mean)
    std = std.clamp(min = min_std) * std_scale

    num_candidates = max(1, int(gamma * pop_size))
    noise = torch.randn((batch, num_candidates, dim), device = pop.device, dtype = pop.dtype)
    candidates = mean + std * noise
    candidates = clamp_bounds(candidates, bounds, straight_through = False)

    if keep_current:
        candidates = cat((pop, candidates), dim = 1)

    total_cands = candidates.shape[1]

    if exists(chunk_size) and total_cands > chunk_size:
        scores = []
        for start in range(0, total_cands, chunk_size):
            chunk = candidates[:, start:start + chunk_size]
            scores.append(obj_fn(chunk))
        f_cands = cat(scores, dim = 1)
    else:
        f_cands = obj_fn(candidates)

    f_cands = torch.where(torch.isfinite(f_cands), f_cands, torch.full_like(f_cands, float('inf')))

    _, top_indices = torch.topk(f_cands, k = pop_size, dim = 1, largest = False)
    top_indices_exp = repeat(top_indices, 'b p -> b p d', d = dim)
    improved = candidates.gather(1, top_indices_exp)

    improved, = unpack(improved, ps, '* p d')
    return improved

@torch.no_grad()
def pair_populations(z0, z1, use_ot = True):
    """
    Pairs source population z0 (b, p, d) with improved target population z1 (b, p, d).
    Uses exact mini-batch 2-Wasserstein assignment when use_ot=True, or random permutation otherwise.
    """
    batch, pop_size, _ = z0.shape

    if not use_ot:
        perm = torch.randn((batch, pop_size), device = z0.device).argsort(dim = -1)
        perm_exp = repeat(perm, 'b p -> b p d', d = z1.shape[-1])
        return z1.gather(1, perm_exp)

    cost = torch.cdist(z0.float(), z1.float()).pow(2).cpu().numpy()
    paired_indices = []

    for b in range(batch):
        _, col_ind = linear_sum_assignment(cost[b])
        paired_indices.append(col_ind)

    col_idx = torch.from_numpy(np.asarray(paired_indices)).to(device = z1.device, dtype = torch.long)
    col_idx_exp = repeat(col_idx, 'b p -> b p d', d = z1.shape[-1])
    return z1.gather(1, col_idx_exp)

def build_local_obj_features(f_vals, eps = 1e-6):
    # f_vals: (b, p) -> (b, p, 4) scale-friendly local objective features across orders of magnitude
    f_clean = torch.nan_to_num(f_vals, nan = 0., posinf = 1e6, neginf = -1e6)
    f_symlog = symlog(f_clean)
    f_logmag = torch.log(f_clean.abs().clamp(min = eps)) * 0.1

    f_mean = f_clean.mean(dim = 1, keepdim = True)
    f_std = f_clean.std(dim = 1, unbiased = False, keepdim = True).clamp(min = eps) if f_clean.shape[1] > 1 else torch.ones_like(f_mean)
    f_zscore = ((f_clean - f_mean) / f_std).clamp(-10., 10.)

    f_min = f_clean.min(dim = 1, keepdim = True).values
    f_rel = symlog((f_clean - f_min).clamp(min = 0.))

    return stack((f_symlog, f_logmag, f_zscore, f_rel), dim = -1)

def build_grad_features(grads, eps = 1e-6):
    # grads: (b, p, d) -> (symlog_grad, unit_grad, symlog_norm, log_norm)
    g_symlog = symlog(grads)
    g_norm = grads.norm(dim = -1, keepdim = True)
    g_unit = grads / (g_norm + eps)
    g_norm_log = symlog(g_norm)
    g_logmag = torch.log(g_norm.clamp(min = eps)) * 0.1
    return g_symlog, g_unit, g_norm_log, g_logmag

# time conditioning

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device, dtype = x.dtype) * -emb)
        emb = rearrange(x, '... -> ... 1') * emb
        return cat((emb.sin(), emb.cos()), dim = -1)

# velocity field architectures

class PopulationVelocityField(Module):
    """
    Permutation-equivariant Transformer velocity field over a population of candidate solutions
    (Section 3.2 & Appendix B.2 of FlowOptimizer).
    """
    def __init__(
        self,
        dim_in: int,
        dim: int = 128,
        depth: int = 1,
        heads: int = 4,
        dim_head: int = 64,
        mlp_depth: int = 3,
        use_grad: bool = True,
        use_base_sampler: bool = False,
        history_len: int = 3,
        dim_cond: int = 0,
        grad_residual: bool = True
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim = dim
        self.use_grad = use_grad
        self.use_base_sampler = use_base_sampler
        self.history_len = history_len
        self.dim_cond = dim_cond
        self.grad_residual = grad_residual and use_grad

        # local per-member embeddings: state z_t -> R^dim, local context c_j -> R^dim

        local_ctx_dim = 4                                   # symlog(f), log(|f|), zscore(f), symlog(f - f_min)
        if use_grad:
            local_ctx_dim += 2 * dim_in + 2                 # symlog(grad), unit_grad, symlog(||grad||), log(||grad||)
        if use_base_sampler:
            local_ctx_dim += dim_in                         # original population X^k

        self.state_embed = nn.Linear(dim_in, dim)
        self.context_embed = nn.Linear(local_ctx_dim, dim)

        attn_dim = dim * 2

        # permutation-equivariant self-attention over population members (no positional embeddings)

        self.transformer = Encoder(
            dim = attn_dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head
        )

        # global conditioning: flow time t, mean gradient, history buffer (H * 2 * dim_in), optional cond

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.ELU(),
            nn.Linear(time_dim, time_dim)
        )

        global_in_dim = time_dim + history_len * 2 * dim_in
        if use_grad:
            global_in_dim += dim_in
        if dim_cond > 0:
            global_in_dim += dim_cond

        self.global_embed = nn.Sequential(
            nn.Linear(global_in_dim, dim),
            nn.LayerNorm(dim),
            nn.ELU()
        )

        # velocity readout MLP (Appendix B.2: 3 hidden layers of width dim, LayerNorm + ELU)

        mlp_layers = []
        in_d = attn_dim + dim
        for _ in range(mlp_depth):
            mlp_layers.extend([
                nn.Linear(in_d, dim),
                nn.LayerNorm(dim),
                nn.ELU()
            ])
            in_d = dim

        out_dim = dim_in * 2 if self.grad_residual else dim_in
        out_proj = nn.Linear(in_d, out_dim)
        nn.init.normal_(out_proj.weight, std = 1e-2)
        nn.init.zeros_(out_proj.bias)
        mlp_layers.append(out_proj)

        self.readout_mlp = nn.Sequential(*mlp_layers)

    def forward(
        self,
        z_t,
        times,
        obj_vals,
        grads = None,
        source_pop = None,
        history = None,
        cond = None
    ):
        batch, pop_size, dim_in = z_t.shape
        device, dtype = z_t.device, z_t.dtype

        if times.ndim == 0:
            times = repeat(times, '-> b', b = batch)

        # per-member local context

        local_feats = [build_local_obj_features(obj_vals)]

        if self.use_grad:
            assert exists(grads), 'grads must be provided when use_grad=True'
            g_symlog, g_unit, g_norm_log, g_logmag = build_grad_features(grads)
            local_feats.extend([g_symlog, g_unit, g_norm_log, g_logmag])

        if self.use_base_sampler:
            src = default(source_pop, z_t)
            local_feats.append(src)

        c_loc = cat(local_feats, dim = -1)

        h_z = self.state_embed(z_t)
        h_c = self.context_embed(c_loc)
        tokens = cat((h_z, h_c), dim = -1)

        # permutation-equivariant attention across population

        tokens = self.transformer(tokens)

        # global conditioning features

        time_emb = self.time_mlp(times.to(dtype = dtype))
        global_feats = [time_emb]

        if self.use_grad:
            mean_grad = symlog(grads.mean(dim = 1))
            global_feats.append(mean_grad)

        if not exists(history):
            mean, std = compute_population_stats(default(source_pop, z_t))
            stats = cat((mean, std), dim = -1)
            history = repeat(stats, 'b f -> b h f', h = self.history_len)

        history_flat = rearrange(history, 'b h f -> b (h f)')
        global_feats.append(history_flat)

        if self.dim_cond > 0:
            assert exists(cond), f'cond of dim {self.dim_cond} must be provided'
            global_feats.append(cond)

        global_emb = self.global_embed(cat(global_feats, dim = -1))
        global_emb = repeat(global_emb, 'b d -> b p d', p = pop_size)

        out = self.readout_mlp(cat((tokens, global_emb), dim = -1))

        if self.grad_residual:
            v_explore, precond = out.chunk(2, dim = -1)
            # learned per-member per-coordinate gradient preconditioner + population exploration velocity
            step_scale = F.softplus(precond - 3.0)
            return v_explore - step_scale * grads.clamp(-20., 20.)

        return out


class DimensionAgnosticVelocityField(Module):
    """
    Dimension-agnostic & population-size-agnostic Transformer velocity field with optional
    low-rank cross-dimensional mixing (Section 3.3 of FlowOptimizer).
    Parameter count does not depend on decision dimension n or population size P.
    """
    def __init__(
        self,
        dim: int = 128,
        depth: int = 1,
        heads: int = 4,
        dim_head: int = 32,
        mlp_depth: int = 3,
        use_grad: bool = True,
        use_base_sampler: bool = False,
        history_len: int = 3,
        low_rank_dim: int = 16,
        dim_cond: int = 0,
        grad_residual: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.use_grad = use_grad
        self.use_base_sampler = use_base_sampler
        self.history_len = history_len
        self.low_rank_dim = low_rank_dim
        self.dim_cond = dim_cond
        self.grad_residual = grad_residual and use_grad

        # time embedding

        time_dim = dim // 2
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ELU(),
            nn.Linear(time_dim, time_dim)
        )

        # fixed-size per-member pooled token s_j(t) (Eq. 7):
        # Stat(z_t) [4] + obj_feats [4] + Stat(grad) [4 if use_grad] + Stat(source) [4 if base_sampler]
        # + pooled history [history_len * 4] + time_dim + dim_cond

        token_in_dim = 4 + 4 + history_len * 4 + time_dim + dim_cond
        if use_grad:
            token_in_dim += 4 + 2               # Stat(symlog(grad)) + symlog(||grad||) + log(||grad||)
        if use_base_sampler:
            token_in_dim += 4

        self.token_embed = nn.Sequential(
            nn.Linear(token_in_dim, dim),
            nn.LayerNorm(dim),
            nn.ELU()
        )

        # permutation-equivariant self-attention over population tokens {h_j}

        self.transformer = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head
        )

        # shared per-coordinate decoder psi_theta (Eq. 8) + optional low-rank mixing (a_{j,i}, b_{j,i})

        coord_in_dim = 1 + dim + time_dim + history_len * 2
        if use_grad:
            coord_in_dim += 2                   # symlog(grad_{j,i}), unit_grad_{j,i}
        if use_base_sampler:
            coord_in_dim += 1

        out_dim = 1 + (1 if self.grad_residual else 0) + (2 * low_rank_dim if low_rank_dim > 0 else 0)

        decoder_layers = []
        in_d = coord_in_dim
        for _ in range(mlp_depth):
            decoder_layers.extend([
                nn.Linear(in_d, dim),
                nn.LayerNorm(dim),
                nn.ELU()
            ])
            in_d = dim

        out_proj = nn.Linear(in_d, out_dim)
        nn.init.normal_(out_proj.weight, std = 1e-2)
        nn.init.zeros_(out_proj.bias)
        decoder_layers.append(out_proj)

        self.coord_decoder = nn.Sequential(*decoder_layers)

    @staticmethod
    def _coord_stats(x, eps = 1e-6):
        # x: (..., n) -> (..., 4) [mean, std, max, min]
        mean = x.mean(dim = -1)
        std = x.std(dim = -1, unbiased = False).clamp(min = eps) if x.shape[-1] > 1 else torch.zeros_like(mean)
        mx = x.max(dim = -1).values
        mn = x.min(dim = -1).values
        return stack((mean, std, mx, mn), dim = -1)

    def forward(
        self,
        z_t,
        times,
        obj_vals,
        grads = None,
        source_pop = None,
        history = None,
        cond = None
    ):
        batch, pop_size, dim_in = z_t.shape
        dtype = z_t.dtype

        if times.ndim == 0:
            times = repeat(times, '-> b', b = batch)

        time_emb = self.time_mlp(times.to(dtype = dtype))

        # prepare history buffer (b, h, 2, n)

        if not exists(history):
            mean, std = compute_population_stats(default(source_pop, z_t))
            stats = cat((mean, std), dim = -1)
            history = repeat(stats, 'b f -> b h f', h = self.history_len)

        hist_mean, hist_std = history.chunk(2, dim = -1)                    # each (b, h, n)

        # pool history stats across coordinate dimension -> (b, h * 4)
        hist_pooled = cat((
            hist_mean.mean(dim = -1),
            hist_mean.std(dim = -1, unbiased = False) if dim_in > 1 else torch.zeros_like(hist_mean.mean(dim = -1)),
            hist_std.mean(dim = -1),
            hist_std.max(dim = -1).values
        ), dim = -1)

        # per-member pooled tokens s_j(t)

        token_feats = [
            self._coord_stats(z_t),
            build_local_obj_features(obj_vals),
            repeat(hist_pooled, 'b f -> b p f', p = pop_size),
            repeat(time_emb, 'b d -> b p d', p = pop_size)
        ]

        if self.use_grad:
            assert exists(grads), 'grads must be provided when use_grad=True'
            g_symlog, g_unit, g_norm_log, g_logmag = build_grad_features(grads)
            token_feats.extend([self._coord_stats(g_symlog), g_norm_log, g_logmag])

        if self.use_base_sampler:
            src = default(source_pop, z_t)
            token_feats.append(self._coord_stats(src))

        if self.dim_cond > 0:
            assert exists(cond), f'cond of dim {self.dim_cond} must be provided'
            token_feats.append(repeat(cond, 'b d -> b p d', p = pop_size))

        h = self.token_embed(cat(token_feats, dim = -1))
        h = self.transformer(h)                                             # (b, p, dim)

        # per-coordinate decoding across i = 1..n

        hist_coord = cat((
            rearrange(hist_mean, 'b h n -> b n h'),
            rearrange(hist_std, 'b h n -> b n h')
        ), dim = -1)                                                        # (b, n, 2h)

        coord_feats = [
            rearrange(z_t, 'b p n -> b p n 1'),
            repeat(h, 'b p d -> b p n d', n = dim_in),
            repeat(time_emb, 'b d -> b p n d', p = pop_size, n = dim_in),
            repeat(hist_coord, 'b n f -> b p n f', p = pop_size)
        ]

        if self.use_grad:
            coord_feats.extend([
                rearrange(g_symlog, 'b p n -> b p n 1'),
                rearrange(g_unit, 'b p n -> b p n 1')
            ])

        if self.use_base_sampler:
            coord_feats.append(rearrange(src, 'b p n -> b p n 1'))

        decoded = self.coord_decoder(cat(coord_feats, dim = -1))            # (b, p, n, out_dim)

        v = decoded[..., 0]                                                 # (b, p, n)
        cursor = 1

        if self.grad_residual:
            step_scale = F.softplus(decoded[..., cursor] - 3.0)
            v = v - step_scale * grads.clamp(-20., 20.)
            cursor += 1

        if self.low_rank_dim > 0:
            a, b_coord = decoded[..., cursor:].chunk(2, dim = -1)           # each (b, p, n, r)
            b_pop = b_coord.mean(dim = 2, keepdim = True)                   # (b, p, 1, r)
            v_mix = (a * b_pop).sum(dim = -1)                               # (b, p, n)
            v = v + v_mix

        return v

# main class

class FlowOptimizer(Module):
    """
    Deep Unfolded FlowOptimizer (Saravanos, So, Ahmad, Fan - ICML 2026 Spotlight).

    Models iterative optimization as continuous-time transport of a population of candidate
    solutions via permutation-equivariant Transformer velocity fields, trained in two phases:
      1. Simulation-free Flow Matching pre-training (`flow_matching_loss`) matching source
         populations X to improved populations X+ = Improve(X, f).
      2. End-to-end Deep Unfolded fine-tuning (`unfold_loss`) directly minimizing objective
         values across K unfolded iterations (with shared terminal steady-state model over
         the last K_final iterations).
    """
    def __init__(
        self,
        dim_in: int | None = None,
        dim: int = 128,
        depth: int = 1,
        heads: int = 4,
        dim_head: int | None = None,
        mlp_depth: int = 3,
        num_unfold_iters: int = 10,
        num_shared_final_iters: int = 3,
        share_weights: bool = False,
        ode_steps: int = 5,
        history_len: int = 3,
        use_grad: bool = True,
        use_base_sampler: bool = False,
        dimension_agnostic: bool = False,
        low_rank_dim: int = 16,
        dim_cond: int = 0,
        alpha: float = 0.8,
        softmin_temp: float = 0.1,
        bounds: tuple | None = None,
        backprop_through_context_grad: bool = False,
        velocity_clip: float | None = None,
        velocity_field_factory: Callable[[], Module] | None = None
    ):
        super().__init__()
        assert num_unfold_iters >= 1
        assert 1 <= num_shared_final_iters <= num_unfold_iters
        assert ode_steps >= 1
        assert history_len >= 1

        if not dimension_agnostic and not exists(velocity_field_factory):
            assert exists(dim_in), 'dim_in must be specified unless dimension_agnostic=True'

        self.dim_in = dim_in
        self.dim = dim
        self.num_unfold_iters = num_unfold_iters
        self.num_shared_final_iters = num_shared_final_iters
        self.share_weights = share_weights
        self.ode_steps = ode_steps
        self.history_len = history_len
        self.use_grad = use_grad
        self.use_base_sampler = use_base_sampler
        self.dimension_agnostic = dimension_agnostic
        self.dim_cond = dim_cond
        self.alpha = alpha
        self.softmin_temp = softmin_temp
        self.backprop_through_context_grad = backprop_through_context_grad
        self.velocity_clip = velocity_clip

        if exists(bounds):
            low, high = bounds
            low_t = torch.as_tensor(low, dtype = torch.float32)
            high_t = torch.as_tensor(high, dtype = torch.float32)
            self.register_buffer('bounds_low', low_t)
            self.register_buffer('bounds_high', high_t)
        else:
            self.bounds_low = None
            self.bounds_high = None

        # construct iteration-specific velocity fields (sharing the final K_final iterations)

        num_unique_models = 1 if share_weights else (num_unfold_iters - num_shared_final_iters + 1)
        self.num_unique_models = num_unique_models

        def make_field():
            if exists(velocity_field_factory):
                return velocity_field_factory()

            if dimension_agnostic:
                return DimensionAgnosticVelocityField(
                    dim = dim,
                    depth = depth,
                    heads = heads,
                    dim_head = default(dim_head, max(dim // heads, 16)),
                    mlp_depth = mlp_depth,
                    use_grad = use_grad,
                    use_base_sampler = use_base_sampler,
                    history_len = history_len,
                    low_rank_dim = low_rank_dim,
                    dim_cond = dim_cond
                )

            return PopulationVelocityField(
                dim_in = dim_in,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = default(dim_head, max((dim * 2) // heads, 16)),
                mlp_depth = mlp_depth,
                use_grad = use_grad,
                use_base_sampler = use_base_sampler,
                history_len = history_len,
                dim_cond = dim_cond
            )

        self.velocity_fields = ModuleList([make_field() for _ in range(num_unique_models)])

    @property
    def model(self) -> Module:
        """Base velocity field (used during Phase 1 flow matching pre-training)."""
        return self.velocity_fields[0]

    @property
    def bounds(self):
        if exists(self.bounds_low) and exists(self.bounds_high):
            return (self.bounds_low, self.bounds_high)
        return None

    def get_velocity_field(self, iter_idx: int) -> Module:
        return self.velocity_fields[min(iter_idx, self.num_unique_models - 1)]

    @torch.no_grad()
    def init_unfolded_from_base(self):
        """Copy pre-trained base velocity field weights into all unfolded iteration models."""
        if self.num_unique_models <= 1:
            return

        base_state = self.velocity_fields[0].state_dict()
        for field in self.velocity_fields[1:]:
            field.load_state_dict(deepcopy(base_state))

    def init_history(self, pop):
        mean, std = compute_population_stats(pop)
        stats = cat((mean, std), dim = -1)
        return repeat(stats, 'b f -> b h f', h = self.history_len)

    def update_history(self, history, pop):
        mean, std = compute_population_stats(pop)
        stats = rearrange(cat((mean, std), dim = -1), 'b f -> b 1 f')

        if not exists(history):
            return repeat(stats, 'b 1 f -> b h f', h = self.history_len)

        if self.history_len == 1:
            return stats

        return cat((history[:, 1:], stats), dim = 1)

    def apply_base_sampler(self, pop, min_std = 1e-3):
        mean, std = compute_population_stats(pop, eps = min_std)
        mean = rearrange(mean, 'b d -> b 1 d')
        std = rearrange(std, 'b d -> b 1 d')
        sampled = mean + std * randn_like(pop)
        return clamp_bounds(sampled, self.bounds, straight_through = True)

    def flow_matching_loss(
        self,
        population,
        obj_fn: Callable,
        *,
        target_population = None,
        gamma = 50,
        std_scale = math.sqrt(3.0),
        min_std = 1e-2,
        use_ot_pairing = True,
        cond = None,
        return_info = False
    ):
        """
        Phase 1: Simulation-free pre-training via conditional flow matching (Algorithm 1).
        Matches displacements from source population X to improved target population X+.
        """
        pop, ps = pack([population], '* p d')
        batch = pop.shape[0]
        device, dtype = pop.device, pop.dtype

        # evaluate context on source population X

        f_vals, grads = compute_obj_and_grad(
            pop,
            obj_fn,
            use_grad = self.use_grad,
            backprop_through_grad = False
        )
        history = self.init_history(pop)

        # construct or accept improved target population X+

        if exists(target_population):
            target_pop, _ = pack([target_population], '* p d')
        else:
            target_pop = improve_population(
                pop,
                obj_fn,
                gamma = gamma,
                std_scale = std_scale,
                min_std = min_std,
                bounds = self.bounds
            )

        # source state z_0 (optionally via Gaussian base sampler)

        z0 = self.apply_base_sampler(pop) if self.use_base_sampler else pop
        z1 = pair_populations(z0, target_pop, use_ot = use_ot_pairing)

        times = rand((batch,), device = device, dtype = dtype)
        t_expand = rearrange(times, 'b -> b 1 1')
        z_t = z0.lerp(z1, t_expand)
        target_vel = z1 - z0

        pred_vel = self.model(
            z_t,
            times,
            f_vals,
            grads = grads,
            source_pop = pop if self.use_base_sampler else None,
            history = history,
            cond = cond
        )

        loss = F.mse_loss(pred_vel, target_vel)

        if not return_info:
            return loss

        return loss, {
            'source_best': f_vals.min(dim = 1).values.mean().item(),
            'target_best': obj_fn(target_pop).min(dim = 1).values.mean().item()
        }

    def step(
        self,
        population,
        obj_fn: Callable,
        *,
        iter_idx: int = 0,
        history = None,
        cond = None,
        ode_steps: int | None = None,
        return_history: bool = False
    ):
        """
        Single FlowOptimizer iteration k: extract context c^k, integrate velocity field ODE
        from t=0 to t=1 via Euler integration, and return updated population X^{k+1}.
        """
        pop, ps = pack([population], '* p d')
        batch = pop.shape[0]
        device, dtype = pop.device, pop.dtype

        steps = default(ode_steps, self.ode_steps)
        dt = 1.0 / steps

        f_vals, grads = compute_obj_and_grad(
            pop,
            obj_fn,
            use_grad = self.use_grad,
            backprop_through_grad = self.backprop_through_context_grad
        )

        history = self.update_history(history, pop)
        z = self.apply_base_sampler(pop) if self.use_base_sampler else pop

        vel_field = self.get_velocity_field(iter_idx)
        time_grid = linspace(0., 1., steps + 1, device = device, dtype = dtype)[:-1]

        for t_scalar in time_grid:
            t_batch = repeat(t_scalar, '-> b', b = batch)
            vel = vel_field(
                z,
                t_batch,
                f_vals,
                grads = grads,
                source_pop = pop if self.use_base_sampler else None,
                history = history,
                cond = cond
            )

            if exists(self.velocity_clip):
                vel = vel.clamp(-self.velocity_clip, self.velocity_clip)

            z = z + dt * vel

        z = clamp_bounds(z, self.bounds, straight_through = True)
        z_out, = unpack(z, ps, '* p d')

        if return_history:
            return z_out, history

        return z_out

    def optimize(
        self,
        population,
        obj_fn: Callable,
        *,
        num_iters: int | None = None,
        cond = None,
        ode_steps: int | None = None,
        return_trajectory: bool = False,
        return_best: bool = False
    ):
        """
        Run deep-unfolded optimization for `num_iters` iterations (defaults to `num_unfold_iters`).
        Can run beyond `num_unfold_iters` at inference time by reusing the terminal velocity field.
        """
        total_iters = default(num_iters, self.num_unfold_iters)
        pop, ps = pack([population], '* p d')

        history = None
        trajectory = [unpack(pop, ps, '* p d')[0]] if return_trajectory else None

        for k in range(total_iters):
            pop, history = self.step(
                pop,
                obj_fn,
                iter_idx = k,
                history = history,
                cond = cond,
                ode_steps = ode_steps,
                return_history = True
            )

            if return_trajectory:
                trajectory.append(unpack(pop, ps, '* p d')[0])

        final_pop, = unpack(pop, ps, '* p d')

        if return_best:
            with torch.no_grad():
                f_final = obj_fn(pop)
                best_idx = f_final.argmin(dim = 1)
                best_x = pop[torch.arange(pop.shape[0], device = pop.device), best_idx]
                best_f = f_final[torch.arange(pop.shape[0], device = pop.device), best_idx]
            best_x, = unpack(best_x, ps, '* d')
            best_f, = unpack(best_f, ps, '*')
            if return_trajectory:
                return final_pop, best_x, best_f, trajectory
            return final_pop, best_x, best_f

        if return_trajectory:
            return final_pop, trajectory

        return final_pop

    def unfold_loss(
        self,
        population,
        obj_fn: Callable,
        *,
        num_iters: int | None = None,
        active_model_idx: int | None = None,
        iter_weights = None,
        alpha: float | None = None,
        softmin_temp: float | None = None,
        loss_transform: Callable | None = None,
        cond = None,
        ode_steps: int | None = None,
        return_trajectory: bool = False
    ):
        """
        Phase 2: End-to-end deep unfolding fine-tuning loss (Algorithm 2 & Eq. 12-13).
          L_FT = sum_{k=1}^K w_k * [alpha * SoftMin(f(X_k)) + (1 - alpha) * Mean(f(X_k))]

        When `active_model_idx` is set (progressive training, Appendix B.3), earlier iteration
        models run under stop-gradient and optimization runs up to the iterations covered by
        `active_model_idx`.
        """
        alpha = default(alpha, self.alpha)
        softmin_temp = default(softmin_temp, self.softmin_temp)

        if exists(active_model_idx):
            if active_model_idx < self.num_unique_models - 1:
                total_iters = active_model_idx + 1
            else:
                total_iters = self.num_unfold_iters
        else:
            total_iters = default(num_iters, self.num_unfold_iters)

        if not exists(iter_weights):
            weights = [1.0 / total_iters] * total_iters
        else:
            weights = iter_weights
            assert len(weights) == total_iters

        pop, ps = pack([population], '* p d')
        history = None
        total_loss = pop.new_tensor(0.)
        trajectory = [unpack(pop, ps, '* p d')[0]] if return_trajectory else None

        for k in range(total_iters):
            model_idx = min(k, self.num_unique_models - 1)
            freeze_step = exists(active_model_idx) and model_idx < active_model_idx

            if freeze_step:
                with torch.no_grad():
                    pop, history = self.step(
                        pop,
                        obj_fn,
                        iter_idx = k,
                        history = history,
                        cond = cond,
                        ode_steps = ode_steps,
                        return_history = True
                    )
                pop = pop.detach()
                history = history.detach()
            else:
                pop, history = self.step(
                    pop,
                    obj_fn,
                    iter_idx = k,
                    history = history,
                    cond = cond,
                    ode_steps = ode_steps,
                    return_history = True
                )

                f_vals = obj_fn(pop)
                if exists(loss_transform):
                    f_vals = loss_transform(f_vals)
                best_f = softmin(f_vals, temp = softmin_temp, dim = 1)
                mean_f = f_vals.mean(dim = 1)
                iter_loss = (alpha * best_f + (1.0 - alpha) * mean_f).mean()
                total_loss = total_loss + weights[k] * iter_loss

            if return_trajectory:
                trajectory.append(unpack(pop, ps, '* p d')[0])

        if return_trajectory:
            return total_loss, trajectory

        return total_loss

    def forward(
        self,
        population,
        obj_fn: Callable,
        **kwargs
    ):
        return self.optimize(population, obj_fn, **kwargs)
