
import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList
from torchdiffeq import odeint

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from huggingface_hub.inference._generated.types import zero_shot_image_classification


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def down(x): return F.avg_pool2d(x, kernel_size=2, stride=2)

def up(x): return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


# https://arxiv.org/abs/2212.09748

class TimestepEmbedder(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        
        args = (t[:, None].float() * 1000.0) * freqs[None]
        
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
    
    pos_h = grid_h.reshape(-1)
    pos_w = grid_w.reshape(-1)
    
    dim_half = embed_dim // 2
  
    omega = torch.exp(-math.log(10000) * torch.arange(dim_half // 2, dtype=torch.float32) / (dim_half // 2))
    
    out_h = pos_h[:, None] * omega[None, :]
    out_w = pos_w[:, None] * omega[None, :]
    
    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)
    
    return torch.cat([emb_h, emb_w], dim=1)
    

class AdaLNModulation(Module):
    def __init__(self, dim, out_multiplier=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, out_multiplier * dim, bias=True)
        )
       
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, c):
        return self.net(c)

        
class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_scales, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.num_scales = num_scales
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 3 * inner_dim, bias=False),
                Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
            )
            for _ in range(num_scales)
        ])

        self.to_out = nn.ModuleList([
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            for _ in range(num_scales)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, xs):

        device = xs[0].device
        
        lens = [x.shape[1] for x in xs]

        qkvs = [qkv_layer(x) for qkv_layer, x in zip(self.to_qkv, xs)]

        qs, ks, vs = zip(*qkvs)

        q = torch.cat(qs, dim=2)
        k = torch.cat(ks, dim=2)
        v = torch.cat(vs, dim=2)

        scale_indices = torch.arange(self.num_scales, device=device) 
        lens_t = torch.tensor(lens, device=device)
        
        token_scales = torch.repeat_interleave(scale_indices, lens_t)
        q_scales = rearrange(token_scales, 'q -> q 1')
        k_scales = rearrange(token_scales, 'k -> 1 k')
        
        causal_mask = k_scales > q_scales
        
        sim = torch.einsum('b h q d, b h k d -> b h q k', q, k) * self.scale
        sim.masked_fill_(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out_global = torch.einsum('b h q k, b h k d -> b h q d', attn, v)
        out_global = rearrange(out_global, 'b h n d -> b n (h d)')

        outs = torch.split(out_global, lens, dim=1)

        outs = [self.to_out[i](outs[i]) for i in range(self.num_scales)]

        return outs

    
class DiTBlock(Module):
    def __init__(self, dim, num_scales, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.num_scales = num_scales
        
        self.norm1 = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(num_scales)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(num_scales)])
        self.ff = nn.ModuleList([FeedForward(dim, mlp_dim, dropout) for _ in range(num_scales)])
        self.adaln = nn.ModuleList([AdaLNModulation(dim, out_multiplier=6) for _ in range(num_scales)])
        
        self.attn = Attention(dim, num_scales, heads, dim_head, dropout)

    def forward(self, xs, c):
        chunks = [adaln(c).chunk(6, dim=-1) for adaln in self.adaln]
    
        msa_inputs = [modulate(norm(x), ch[0], ch[1]) for x, norm, ch in zip(xs, self.norm1, chunks)]

        # global attention across scales
        attn_outs = self.attn(msa_inputs)

        outs = []
        for x, attn_out, norm2, ff, (_, _, gate_msa, shift_mlp, scale_mlp, gate_mlp) in zip(
            xs, attn_outs, self.norm2, self.ff, chunks
        ):

            x = x + gate_msa.unsqueeze(1) * attn_out
        
            x = x + gate_mlp.unsqueeze(1) * ff(modulate(norm2(x), shift_mlp, scale_mlp))
            
            outs.append(x)

        return outs



class LapFlowDiT(Module):
    def __init__(
        self, 
        base_image_size, 
        patch_size, 
        dim, 
        depth, 
        heads, 
        mlp_dim, 
        channels = 3, 
        dim_head = 64, 
        num_scales = 3,
        dropout = 0.
    ):
        super().__init__()
        self.num_scales = num_scales
        patch_dim = channels * patch_size * patch_size
        
        grids = [(base_image_size // (2 ** i)) // patch_size for i in reversed(range(num_scales))]

        def linear():
            l = nn.Linear(dim, patch_dim, bias=True)
            nn.init.constant_(l.weight, 0)
            nn.init.constant_(l.bias, 0)
            return l

        self.patch_embeds = nn.ModuleList([
            nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            ) for _ in range(num_scales)
        ])
        
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(get_2d_sincos_pos_embed(dim, g).clone().detach(), requires_grad=False) 
            for g in grids
        ])

        self.final_adalns = nn.ModuleList([AdaLNModulation(dim, out_multiplier=2) for _ in range(num_scales)])
        self.final_norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(num_scales)])
        self.final_linears = nn.ModuleList([linear() for _ in range(num_scales)])
        
        self.unpatchifys = nn.ModuleList([
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=g, w=g, p1=patch_size, p2=patch_size)
            for g in grids
        ])

        self.dropout = nn.Dropout(dropout)
        self.t_embedder = TimestepEmbedder(dim)
        
        self.y_embedder = None

        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_scales, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, imgs_list, times, cond=None):
     
        xs = [
            self.dropout(patch_embed(img) + pos_embed)
            for img, patch_embed, pos_embed in zip(imgs_list, self.patch_embeds, self.pos_embeds)
        ]

        assert exists(times), "Time embedding 't' or 'times' must be provided to LapFlowDiT"
        c = self.t_embedder(times)
        if exists(self.y_embedder) and exists(cond):
            c = c + self.y_embedder(cond)

        for block in self.blocks:
            xs = block(xs, c)
        
        outs = []
        for x, adaln, norm, linear, unpatch in zip(
            xs, self.final_adalns, self.final_norms, self.final_linears, self.unpatchifys
        ):
          
            shift, scale = adaln(c).chunk(2, dim=-1)
            
            x = modulate(norm(x), shift, scale)
            
            x = linear(x)
            
            outs.append(unpatch(x))
                
        return outs



class LapFlow(Module):
    def __init__(
        self,
        model: Module,
        num_scales=3,
        critical_times=None,
        loss_weights=None,
        times_cond_kwarg='times',
        data_shape=None,
        normalize_data_fn=lambda t: t,
        unnormalize_data_fn=lambda t: t
    ):
        super().__init__()

        assert num_scales in [2, 3]
        
        self.model = model
        self.num_scales = num_scales
        self.times_cond_kwarg = times_cond_kwarg
        self.data_shape = data_shape
        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn
        
        if critical_times is None:
            if num_scales == 2:
                critical_times = [0.0, 0.5]
            elif num_scales == 3:
                critical_times = [0.0, 0.33, 0.67]
            
        self.register_buffer('critical_times', torch.tensor(critical_times, dtype=torch.float32))
        
        weights = default(loss_weights, [1.0] * num_scales)
        self.register_buffer('loss_weights', torch.tensor(weights, dtype=torch.float32))
        

    def get_laplacian_pyramid(self, x):
        
        if self.num_scales == 2:
            coarse = down(x)
            return [coarse, x - up(coarse)]

        elif self.num_scales == 3:
            low = down(x)
            coarse = down(low)
            return [coarse, low - up(coarse), x - up(low)]

    @torch.no_grad()
    def sample(self, batch_size=1, data_shape=None, steps=30, **kwargs):
        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape)

        device = next(self.model.parameters()).device
        
        noise = torch.randn((batch_size, *data_shape), device=device)
        noise_pyramid = self.get_laplacian_pyramid(noise)
        
        time_points = torch.cat([self.critical_times, torch.tensor([1.0], device=device)])
        
        t_starts = time_points[:-1]
        t_ends = time_points[1:]
        durations = t_ends - t_starts

        is_active_matrix = torch.tril(torch.ones([self.num_scales] * 2, dtype=torch.bool, device=device))
        
        steps = torch.clamp((steps * durations).int(), min=1)

        pyd_states = [
            noise * (1.0 - timer_th.item()) 
            for noise, timer_th in zip(noise_pyramid, self.critical_times)
        ]

        for i in range(self.num_scales):
            t_start = t_starts[i]
            t_end = t_ends[i]

            is_active = is_active_matrix[i]
            step_count = steps[i]
            
            dt = (t_end - t_start) / step_count
            
            times = torch.linspace(t_start, t_end, step_count + 1, device=device)
            
            for time in times[:-1]:
                time = time.item()
                sigma_t = 1 - time
                
                time = repeat(torch.tensor([time], device=device), '1 -> b', b=batch_size)
                
                model_inputs = [
                    state if active else (noise * sigma_t)
                    for state, noise, active in zip(pyd_states, noise_pyramid, is_active)
                ]
                
                time_kwarg = {self.times_cond_kwarg: time} if exists(self.times_cond_kwarg) else dict()
                preds = self.model(model_inputs, **time_kwarg, **kwargs)
                
                pyd_states = [
                    (state + pred * dt) if active else state
                    for state, pred, active in zip(pyd_states, preds, is_active)
                ]

        curr = pyd_states[0] 
        for i in range(1, self.num_scales):
            curr = up(curr) + pyd_states[i]
            
        curr = self.unnormalize_data_fn(curr)
        return curr.clamp(0., 1.)


    def forward(self, data, **kwargs):
        if isinstance(data, (tuple, list)):
            actual_image, cond = data[0], data[1]
            cond = cond.flatten()
                
            kwargs['cond'] = cond
            data = actual_image


        data = self.normalize_data_fn(data)

        shape, ndim = data.shape, data.ndim
        
        self.data_shape = default(self.data_shape, shape[1:]) 
        batch, device = shape[0], data.device
       
        data_list = self.get_laplacian_pyramid(data)
        noise_list = self.get_laplacian_pyramid(torch.randn_like(data))
   
        active_scale = torch.randint(0, self.num_scales, (1,)).item()
     
        start_time = self.critical_times[active_scale]

        times = torch.lerp(start_time, torch.tensor(1.0, device=device), torch.rand(batch, device=device))
        
        alphas = torch.clamp((rearrange(times, 'b -> b 1') - self.critical_times) / (1 - self.critical_times), min=0.0)
        sigma = rearrange(1.0 - times, 'b -> b 1 1 1')
        
        noised_list = [
            (rearrange(alpha, 'b -> b 1 1 1') * data) + (sigma * noise)
            for alpha, data, noise in zip(alphas.unbind(dim=1), data_list, noise_list)
        ]
        
        target_velocities = [
            (1.0 / (1 - timer_th)) * data - noise
            for timer_th, data, noise in zip(self.critical_times, data_list, noise_list)
        ]

        time_kwarg = {self.times_cond_kwarg: times} if exists(self.times_cond_kwarg) else dict() 

        preds_list = self.model(noised_list, **time_kwarg, **kwargs)
        
        total_loss = 0.0
        
        s = active_scale + 1
        for pred, target, weight in zip(preds_list[:s], target_velocities[:s], self.loss_weights[:s]):
            total_loss += F.mse_loss(pred, target) * weight
        
        return total_loss