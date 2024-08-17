from __future__ import annotations
import math
from copy import deepcopy
from typing import Tuple, List, Literal, Callable

import torch
from torch import nn, pi, from_numpy
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from torchdiffeq import odeint

import torchvision
from torchvision.utils import save_image
from torchvision.models import VGG16_Weights

import einx
from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

# normalizing helpers

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# noise schedules

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))

# losses

class LPIPSLoss(Module):
    def __init__(
        self,
        vgg: Module | None = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
    ):
        super().__init__()

        if not exists(vgg):
            vgg = torchvision.models.vgg16(weights = vgg_weights)
            vgg.classifier = nn.Sequential(*vgg.classifier[:-2])

        self.vgg = [vgg]

    def forward(self, pred_data, data, reduction = 'mean'):
        vgg, = self.vgg
        vgg = vgg.to(data.device)

        pred_embed, embed = map(vgg, (pred_data, data))

        loss = F.mse_loss(embed, pred_embed, reduction = reduction)

        if reduction == 'none':
            loss = reduce(loss, 'b ... -> b', 'mean')

        return loss

class PseudoHuberLoss(Module):
    def __init__(self, data_dim: int):
        super().__init__()
        self.data_dim = data_dim

    def forward(self, pred, target, reduction = 'mean', **kwargs):
        c = .00054 * self.data_dim
        loss = (F.mse_loss(pred, target, reduction = reduction) + c * c).sqrt() - c
        return loss

class PseudoHuberLossWithLPIPS(Module):
    def __init__(self, data_dim: int, lpips_kwargs: dict = dict()):
        super().__init__()
        self.pseudo_huber = PseudoHuberLoss(data_dim)
        self.lpips = LPIPSLoss(**lpips_kwargs)

    def forward(self, pred_flow, target_flow, *, noised, times, data):
        huber_loss = self.pseudo_huber(pred_flow, target_flow, reduction = 'none')

        pred_data = noised + pred_flow * (1. - times)
        lpips_loss = self.lpips(data, pred_data, reduction = 'none')

        time_weighted_loss = huber_loss * (1 - times) + lpips_loss * (1. / times.clamp(min = 1e-2))
        return time_weighted_loss.mean()

class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)

# main class

class RectifiedFlow(Module):
    def __init__(
        self,
        model: Module,
        time_cond_kwarg: str | None = 'times',
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        loss_fn: Literal[
            'mse',
            'pseudo_huber',
            'pseudo_huber_with_lpips'
        ] | Module = 'mse',
        noise_schedule: Literal[
            'cosmap'
        ] | Callable = identity,
        loss_fn_kwargs: dict = dict(),
        data_shape: Tuple[int, ...] | None = None,
        immiscible = False,
        data_normalize_fn = normalize_to_neg_one_to_one,
        data_unnormalize_fn = unnormalize_to_zero_to_one
    ):
        super().__init__()
        self.model = model
        self.time_cond_kwarg = time_cond_kwarg # whether the model is to be conditioned on the times

        # loss fn

        if loss_fn == 'mse':
            loss_fn = MSELoss()

        elif loss_fn == 'pseudo_huber':
            # section 4.2 of https://arxiv.org/abs/2405.20320v1
            loss_fn = PseudoHuberLoss(**loss_fn_kwargs)

        elif loss_fn == 'pseudo_huber_with_lpips':
            loss_fn = PseudoHuberLossWithLPIPS(**loss_fn_kwargs)

        elif not isinstance(loss_fn, Module):
            raise ValueError(f'unknown loss function {loss_fn}')

        self.loss_fn = loss_fn

        # noise schedules

        if noise_schedule == 'cosmap':
            noise_schedule = cosmap

        elif not callable(noise_schedule):
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        self.noise_schedule = noise_schedule

        # sampling

        self.odeint_kwargs = odeint_kwargs
        self.data_shape = data_shape

        # immiscible diffusion paper, will be removed if does not work

        self.immiscible = immiscible

        # normalizing fn

        self.data_normalize_fn = default(data_normalize_fn, identity)
        self.data_unnormalize_fn = default(data_unnormalize_fn, identity)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        steps = 16,
        noise = None,
        data_shape: Tuple[int, ...] | None = None,
        **model_kwargs
    ):
        was_training = self.training
        self.eval()

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least with one forward'

        def ode_fn(t, x):
            time_kwarg = self.time_cond_kwarg

            if exists(time_kwarg):
                t = repeat(t, '-> b', b = x.shape[0])
                model_kwargs.update(**{time_kwarg: t})

            return self.model(x, **model_kwargs)

        # start with random gaussian noise - y0

        noise = default(noise, torch.randn((batch_size, *data_shape), device = self.device))

        # time steps

        times = torch.linspace(0., 1., steps, device = self.device)

        # ode

        trajectory = odeint(ode_fn, noise, times, **self.odeint_kwargs)

        sampled_data = trajectory[-1]

        self.train(was_training)

        return self.data_unnormalize_fn(sampled_data)

    def forward(
        self,
        data,
        noise = None,
        **model_kwargs
    ):
        batch, *data_shape = data.shape

        data = self.data_normalize_fn(data)

        self.data_shape = default(self.data_shape, data_shape)

        # x0 - gaussian noise, x1 - data

        noise = default(noise, torch.randn_like(data))

        # maybe immiscible flow

        if self.immiscible:
            cost = torch.cdist(data.flatten(1), noise.flatten(1))
            _, reorder_indices = linear_sum_assignment(cost.cpu())
            noise = noise[from_numpy(reorder_indices).to(cost.device)]

        # times, and times with dimension padding on right

        times = torch.rand(batch, device = self.device)
        padded_times = append_dims(times, data.ndim - 1)

        # maybe noise schedule

        t = self.noise_schedule(padded_times)

        # Algorithm 2 in paper
        # linear interpolation of noise with data using random times
        # x1 * t + x0 * (1 - t) - so from noise (time = 0) to data (time = 1.)

        noised = t * data + (1. - t) * noise

        # prepare maybe time conditioning for model
        
        time_kwarg = self.time_cond_kwarg

        if exists(time_kwarg):
            model_kwargs.update(**{time_kwarg: times})

        # the model predicts the flow from the noised data

        flow = data - noise

        pred_flow = self.model(noised, **model_kwargs)

        # loss

        loss = self.loss_fn(pred_flow, flow, noised = noised, times = times, data = data)

        return loss

# reflow wrapper

class Reflow(Module):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        frozen_model: RectifiedFlow | None = None,
        *,
        batch_size = 16,

    ):
        super().__init__()
        model, data_shape = rectified_flow.model, rectified_flow.data_shape
        assert exists(data_shape), '`data_shape` must be defined in RectifiedFlow'

        self.batch_size = batch_size
        self.data_shape = data_shape

        self.model = rectified_flow

        if not exists(frozen_model):
            # make a frozen copy of the model and set requires grad to be False for all parameters for safe measure

            frozen_model = deepcopy(rectified_flow)

            for p in frozen_model.parameters():
                p.detach_()

        self.frozen_model = frozen_model

    def device(self):
        return next(self.parameters()).device

    def parameters(self):
        return self.model.parameters() # omit frozen model

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)

    def forward(self):

        noise = torch.randn((self.batch_size, *self.data_shape), device = self.device)
        sampled_output = self.frozen_model.sample(noise = noise)

        # the coupling in the paper is (noise, sampled_output)

        loss = self.model(sampled_output, noise = noise)

        return loss

# unet

from functools import partial

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def divisible_by(num, den):
    return (num % den) == 0

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * (self.gamma + 1) * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = einx.multiply('i, j -> i j', x, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = tuple(rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads) for t in qkv)

        mk, mv = tuple(repeat(t, 'h c n -> b h c n', b = b) for t in self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = einsum(k, v, 'b h d n, b h e n -> b h d e')

        out = einsum(context, q, 'b h d e, b h d n -> b h e n')
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1, bias = False)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        q = q * self.scale
        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        channels = 3,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, times):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(times)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# dataset classes

from pathlib import Path

from torch.utils.data import Dataset
import torchvision.transforms as T

from PIL import Image

class ImageDataset(Dataset):
    def __init__(
        self,
        folder: str | Path,
        image_size: int,
        exts: List[str] = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        if isinstance(folder, str):
            folder = Path(folder)

        assert folder.is_dir()

        self.folder = folder
        self.image_size = image_size

        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        def convert_image_to_fn(img_type, image):
            if image.mode == img_type:
                return image

            return image.convert(img_type)

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer

from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import DataLoader
from ema_pytorch import EMA

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

class Trainer(Module):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        *,
        dataset: Dataset,
        num_train_steps = 70_000,
        learning_rate = 3e-4,
        batch_size = 16,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './results',
        save_results_every: int = 100,
        checkpoint_every: int = 1000,
        num_samples: int = 16,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.model = rectified_flow

        if self.is_main:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                **ema_kwargs
            )

            self.ema_model.to(self.accelerator.device)

        self.optimizer = Adam(rectified_flow.parameters(), lr = learning_rate, **adam_kwargs)
        self.dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)

        self.num_train_steps = num_train_steps

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(),
        )

        torch.save(save_package, str(self.checkpoints_folder / path))

    def forward(self):

        dl = cycle(self.dl)

        for ind in range(self.num_train_steps):
            step = ind + 1

            self.model.train()

            data = next(dl)
            loss = self.model(data)

            self.accelerator.print(f'[{step}] loss: {loss.item():.3f}')
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.is_main:
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main:
                if divisible_by(step, self.save_results_every):
                    self.ema_model.ema_model.data_shape = self.model.data_shape

                    with torch.no_grad():
                        sampled = self.ema_model.sample(batch_size = self.num_samples)

                    sampled.clamp_(0., 1.)
                    save_image(sampled, str(self.results_folder / f'results.{step}.png'), nrow = self.num_sample_rows)

                if divisible_by(step, self.checkpoint_every):
                    self.save(f'checkpoint.{step}.pt')

            self.accelerator.wait_for_everyone()

        print('training complete')

# reflow trainer

class ReflowTrainer(Module):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        *,
        num_train_steps = 70_000,
        learning_rate = 3e-4,
        batch_size = 16,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './results',
        save_results_every: int = 100,
        checkpoint_every: int = 1000,
        num_samples: int = 16,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict()
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.model = Reflow(rectified_flow)

        if self.is_main:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                **ema_kwargs
            )

            self.ema_model.to(self.accelerator.device)

        self.optimizer = Adam(rectified_flow.parameters(), lr = learning_rate, **adam_kwargs)

        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        self.batch_size = batch_size
        self.num_train_steps = num_train_steps

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(),
        )

        torch.save(save_package, str(self.checkpoints_folder / path))

    def forward(self):

        for ind in range(self.num_train_steps):
            step = ind + 1

            self.model.train()

            loss = self.model(batch_size = self.batch_size)

            self.accelerator.print(f'[{step}] reflow loss: {loss.item():.3f}')
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.is_main:
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main:
                if divisible_by(step, self.save_results_every):
                    self.ema_model.ema_model.data_shape = self.model.data_shape

                    with torch.no_grad():
                        sampled = self.ema_model.sample(batch_size = self.num_samples)

                    sampled.clamp_(0., 1.)
                    save_image(sampled, str(self.results_folder / f'results.{step}.png'), nrow = self.num_sample_rows)

                if divisible_by(step, self.checkpoint_every):
                    self.save(f'checkpoint.{step}.pt')

            self.accelerator.wait_for_everyone()

        print('reflow training complete')