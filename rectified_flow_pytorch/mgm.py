# Midpoint Generative Models - https://arxiv.org/abs/2605.29920

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from shutil import rmtree
from typing import Callable

import torch
from torch import rand, randn, randn_like, randint, where, zeros
from torch.nn import Module
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from torchvision.utils import save_image
from einops import rearrange
from accelerate import Accelerator
from ema_pytorch import EMA
from torch_einops_utils import frozen, temp_eval, align_dims_left, pad_right_ndim_to, device_dtype as dd
from torch_einops_utils.shape import shape, size

from rectified_flow_pytorch.rectified_flow import (
    Unet,
    ImageDataset,
    cycle,
    divisible_by
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

# functions

def zero_centered_gradient_penalty(inputs, outputs, weight = 1.):
    # zero-centered R1/R2 penalty (Huang et al., 2025); Rademacher Hutchinson via Gemini Neon

    rand_signs = randint(0, 2, outputs.shape, **dd(outputs)) * 2. - 1.

    grads, = torch_grad(
        outputs = outputs,
        inputs = inputs,
        grad_outputs = rand_signs,
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )

    return weight * grads.pow(2).mean()

def flipped_interpolant(x0, x1, sigma = 0.):
    # symmetric stochastic flipped interpolant (Eq. 7)

    batch = size(x0, '[b] ...')

    times = rand(batch, **dd(x0)) * 0.5                              # t ~ U[0, 1/2]
    flip = rand(batch, **dd(x0)) < 0.5                               # hidden flip B ~ Bernoulli(1/2)

    padded_times, padded_flip, _ = align_dims_left((times, flip, x0))
    interp_times = where(padded_flip, 1. - padded_times, padded_times)

    interpolated = x0.lerp(x1.to(x0.dtype), interp_times)

    if sigma > 0.:
        sigma_t = (sigma * padded_times * (1. - padded_times)).sqrt()
        interpolated = interpolated + sigma_t * randn_like(interpolated)

    return interpolated, times

# main class

class MGM(Module):
    def __init__(
        self,
        generator: Unet | dict | Module,
        critic: Unet | dict | Module | None = None,
        data_shape: tuple[int, ...] | None = None,
        sigma: float = 0.01,
        critic_only_noise: bool = True,
        grad_penalty_weight: float = 0.,
        normalize_data_fn: Callable = identity,
        unnormalize_data_fn: Callable = identity,
    ):
        super().__init__()
        assert sigma >= 0. and grad_penalty_weight >= 0.

        if isinstance(generator, dict):
            generator = Unet(**generator)

        if not exists(critic):
            critic = deepcopy(generator)
        elif isinstance(critic, dict):
            critic = Unet(**critic)

        assert generator is not critic, 'generator and critic must be separate modules'
        assert getattr(critic, 'accept_time', True), 'critic Unet must accept time conditioning'

        self.generator = generator
        self.critic = critic

        self.data_shape = data_shape
        self.sigma = sigma
        self.critic_only_noise = critic_only_noise
        self.grad_penalty_weight = grad_penalty_weight

        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn

    @property
    def device(self):
        return dd(self.generator)['device']

    def prepare_data(self, data, noise = None):
        data = self.normalize_data_fn(data)
        self.data_shape = default(self.data_shape, shape(data, 'b [...]').shape)
        noise = default(noise, randn_like(data))
        return data, noise

    def generate(self, noise, times = None, generator = None, **kwargs):
        generator = default(generator, self.generator)

        if getattr(generator, 'accept_time', False):
            times = default(times, zeros(size(noise, '[b] ...'), **dd(noise)))
            return generator(noise, times = times, **kwargs)

        return generator(noise, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        data_shape = None,
        noise = None,
        generator = None,
        **kwargs
    ):
        generator = default(generator, self.generator)

        if not exists(noise):
            data_shape = default(data_shape, self.data_shape)
            assert exists(data_shape), 'data_shape must be set at init or inferred during training'
            noise = randn((batch_size, *data_shape), **dd(generator))

        with temp_eval(generator):
            samples = self.generate(noise, generator = generator, **kwargs)

        return self.unnormalize_data_fn(samples)

    def denoising_warmup_loss(self, data, noise = None, **kwargs):
        data, noise = self.prepare_data(data, noise)
        times = rand(size(data, '[b] ...'), **dd(data))
        noised = noise.lerp(data, pad_right_ndim_to(times, data.ndim))
        return F.mse_loss(self.generate(noised, times = times, **kwargs), data)

    def critic_loss(self, data, noise = None, apply_grad_penalty = True, **kwargs):
        # regress (data - generated) from flipped interpolant

        data, noise = self.prepare_data(data, noise)

        with torch.no_grad():
            generated = self.generate(noise, **kwargs)

        interpolated, times = flipped_interpolant(generated, data, sigma = self.sigma)
        use_grad_penalty = apply_grad_penalty and self.grad_penalty_weight > 0.

        if use_grad_penalty:
            interpolated = interpolated.detach().requires_grad_(True)

        pred = self.critic(interpolated, times = times, **kwargs)
        loss = F.mse_loss(pred, data - generated)

        if use_grad_penalty:
            loss = loss + zero_centered_gradient_penalty(interpolated, pred, weight = self.grad_penalty_weight)

        return loss

    discriminator_loss = critic_loss

    def generator_loss(self, data, noise = None, **kwargs):
        # variational Midpoint Divergence (Eq. 10)

        data, noise = self.prepare_data(data, noise)
        generated = self.generate(noise, **kwargs)

        sigma = 0. if self.critic_only_noise else self.sigma
        interpolated, times = flipped_interpolant(generated, data, sigma = sigma)

        with frozen(self.critic):
            pred = self.critic(interpolated, times = times, **kwargs)

        target = data - generated
        return (pred * (2. * target - pred)).mean()

    def forward(self, data, return_critic_loss = False, **kwargs):
        loss_fn = self.critic_loss if return_critic_loss else self.generator_loss
        return loss_fn(data, **kwargs)

# trainer

class MGMTrainer(Module):
    def __init__(
        self,
        mgm: dict | MGM,
        *,
        dataset: dict | Dataset,
        num_train_steps = 70_000,
        learning_rate = 5e-5,
        critic_learning_rate = None,
        critic_steps = 1,
        warmup_steps = 0,
        warmup_learning_rate = 3e-4,
        critic_warmup_steps = 0,
        grad_penalty_every: int | None = 4,
        batch_size = 16,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './results',
        save_results_every: int | None = 100,
        checkpoint_every: int | None = 1000,
        num_samples: int = 16,
        adam_kwargs: dict = dict(betas = (0., 0.999)),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(beta = 0.999),
        use_ema = True,
        max_grad_norm = 1.0,
        clear_results_folder = False
    ):
        super().__init__()
        assert not exists(grad_penalty_every) or grad_penalty_every > 0

        self.accelerator = Accelerator(**accelerate_kwargs)
        acc = self.accelerator

        if isinstance(dataset, dict):
            dataset = ImageDataset(**dataset)

        if isinstance(mgm, dict):
            mgm = MGM(**mgm)

        self.model = mgm

        self.ema_generator = (
            EMA(mgm.generator, include_online_model = False, **ema_kwargs).to(acc.device)
            if self.is_main and use_ema else None
        )

        critic_learning_rate = default(critic_learning_rate, learning_rate)

        self.gen_optimizer = Adam(mgm.generator.parameters(), lr = learning_rate, **adam_kwargs)
        self.critic_optimizer = Adam(mgm.critic.parameters(), lr = critic_learning_rate, **adam_kwargs)
        self.dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        self.model, self.gen_optimizer, self.critic_optimizer, self.dl = acc.prepare(
            self.model, self.gen_optimizer, self.critic_optimizer, self.dl
        )

        self.num_train_steps = num_train_steps
        self.critic_steps = critic_steps
        self.warmup_steps = warmup_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.critic_warmup_steps = critic_warmup_steps
        self.grad_penalty_every = grad_penalty_every
        self.max_grad_norm = max_grad_norm

        # folders

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        if self.is_main and clear_results_folder and self.results_folder.exists():
            rmtree(str(self.results_folder))

        self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert self.num_sample_rows ** 2 == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples
        self.fixed_noise = None

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    def step_optimizer(self, optimizer, params, loss):
        acc = self.accelerator
        optimizer.zero_grad()
        acc.backward(loss)

        if acc.sync_gradients:
            acc.clip_grad_norm_(params, self.max_grad_norm)

        optimizer.step()

    def save(self, path):
        if not self.is_main:
            return

        torch.save(dict(
            model = self.unwrapped_model.state_dict(),
            ema_generator = self.ema_generator.state_dict() if exists(self.ema_generator) else None,
            gen_optimizer = self.gen_optimizer.state_dict(),
            critic_optimizer = self.critic_optimizer.state_dict()
        ), str(self.checkpoints_folder / path))

    def load(self, path):
        if not self.is_main:
            return

        checkpoint = torch.load(self.checkpoints_folder / path, weights_only = True)
        self.unwrapped_model.load_state_dict(checkpoint['model'])

        if exists(self.ema_generator) and exists(checkpoint['ema_generator']):
            self.ema_generator.load_state_dict(checkpoint['ema_generator'])

        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    def sample(self, fname):
        model = self.unwrapped_model
        eval_generator = self.ema_generator.ema_model if exists(self.ema_generator) else model.generator

        if not exists(self.fixed_noise):
            data_shape = default(model.data_shape, shape(next(iter(self.dl)), 'b [...]').shape)
            self.fixed_noise = randn((self.num_samples, *data_shape), **dd(eval_generator))

        sampled = model.sample(noise = self.fixed_noise, generator = eval_generator).clamp(0., 1.)
        grid = rearrange(sampled, '(row col) c h w -> c (row h) (col w)', row = self.num_sample_rows)
        save_image(grid, str(fname))
        return sampled

    def forward(self):
        acc = self.accelerator
        acc.print(f'saving results to {str(self.results_folder)}')

        dl = cycle(self.dl)
        model = self.unwrapped_model
        self.model.train()

        # denoising warmup

        if self.warmup_steps > 0:
            warmup_optimizer = acc.prepare(Adam(model.generator.parameters(), lr = self.warmup_learning_rate))

            for step in range(1, self.warmup_steps + 1):
                warmup_loss = model.denoising_warmup_loss(next(dl))
                self.step_optimizer(warmup_optimizer, model.generator.parameters(), warmup_loss)

                if step == 1 or divisible_by(step, 100):
                    acc.print(f'[warmup {step}/{self.warmup_steps}] loss: {warmup_loss.item():.4f}')

            model.critic.load_state_dict(model.generator.state_dict())

            if exists(self.ema_generator):
                self.ema_generator.copy_params_from_model_to_ema()

        # critic warmup

        for _ in range(self.critic_warmup_steps):
            critic_loss = self.model(next(dl), return_critic_loss = True, apply_grad_penalty = False)
            self.step_optimizer(self.critic_optimizer, model.critic.parameters(), critic_loss)

        # alternating critic and generator updates

        for step in range(1, self.num_train_steps + 1):
            apply_grad_penalty = exists(self.grad_penalty_every) and divisible_by(step, self.grad_penalty_every)

            for critic_step in range(self.critic_steps):
                critic_loss = self.model(next(dl), return_critic_loss = True, apply_grad_penalty = apply_grad_penalty and critic_step == 0)
                self.step_optimizer(self.critic_optimizer, model.critic.parameters(), critic_loss)

            gen_loss = self.model(next(dl), return_critic_loss = False)
            self.step_optimizer(self.gen_optimizer, model.generator.parameters(), gen_loss)

            if exists(self.ema_generator):
                self.ema_generator.update()

            acc.print(f'[{step}] critic: {critic_loss.item():.4f} | gen: {gen_loss.item():.4f}')

            acc.wait_for_everyone()

            if self.is_main:
                if exists(self.save_results_every) and divisible_by(step, self.save_results_every):
                    self.sample(self.results_folder / f'results.{step}.png')

                if exists(self.checkpoint_every) and divisible_by(step, self.checkpoint_every):
                    self.save(f'checkpoint.{step}.pt')

            acc.wait_for_everyone()

        acc.print('training complete')

# quick test

if __name__ == '__main__':
    generator = Unet(dim = 32, dim_mults = (1, 2))
    critic = Unet(dim = 32, dim_mults = (1, 2))

    mgm = MGM(generator, critic, grad_penalty_weight = 1.)
    data = randn(4, 3, 16, 16)

    critic_loss = mgm.critic_loss(data)
    critic_loss.backward()

    gen_loss = mgm.generator_loss(data)
    gen_loss.backward()

    sampled = mgm.sample(batch_size = 4)
    assert sampled.shape == data.shape
