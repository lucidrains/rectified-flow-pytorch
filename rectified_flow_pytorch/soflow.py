from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import tensor, ones
from torch.nn import Module
import torch.nn.functional as F

from einops import reduce

# functions

def exists(v):
    return v is not None

def xnor(x, y):
    return not (x ^ y)

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def append_dims(t, dims):
    shape = t.shape
    ones_shape = (1,) * dims
    return t.reshape(*shape, *ones_shape)

# soflow

# Luo et al of Princeton - https://arxiv.org/abs/2512.15657
# https://github.com/zlab-princeton/SoFlow

class SoFlow(Module):
    def __init__(
        self,
        model: Module,
        *,
        data_shape = None,
        normalize_data_fn = identity,
        unnormalize_data_fn = identity,
        use_adaptive_loss_weight = True,
        adaptive_loss_weight_p = 0.5,
        use_logit_normal_sampler = True,
        logit_normal_mean_fm = 0.2,
        logit_normal_std_fm = 0.8,
        logit_normal_mean_t = 0.2,
        logit_normal_std_t = 0.8,
        logit_normal_mean_s = -1.0,
        logit_normal_std_s = 0.8,
        model_output_clean = False, # Back to Basics paper
        r_schedule: str = "exponential",
        r_init: float = 0.1,
        r_end: float = 0.002,
        r_total_steps: int = 100_000,
        lambda_flow_matching: float = 0.75,
        eps: float = 1e-2,
        noise_std_dev: float = 1.0,
        accept_cond: bool = False,
    ):
        super().__init__()
        self.model = model
        self.data_shape = data_shape

        # normalization
        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn

        # weighting
        self.use_adaptive_loss_weight = use_adaptive_loss_weight
        self.adaptive_loss_weight_p = adaptive_loss_weight_p
        self.eps = eps

        # time sampling
        self.logit_normal_mean_fm = logit_normal_mean_fm
        self.logit_normal_std_fm = logit_normal_std_fm
        self.use_logit_normal_sampler = use_logit_normal_sampler
        self.logit_normal_mean_t = logit_normal_mean_t
        self.logit_normal_std_t = logit_normal_std_t
        self.logit_normal_mean_s = logit_normal_mean_s
        self.logit_normal_std_s = logit_normal_std_s

        # l -> t schedule
        self.r_schedule = r_schedule
        self.r_init = r_init
        self.r_end = r_end
        self.r_total_steps = r_total_steps
        self.step = 0

        # loss blend
        self.lambda_flow_matching = lambda_flow_matching

        self.noise_std_dev = noise_std_dev
        self.accept_cond = accept_cond

        self.model_output_clean = model_output_clean

        self.register_buffer("dummy", tensor(0), persistent=False)

    @property
    def device(self):
        return self.dummy.device

    def sample_times(self, batch, *, mean, std):
        shape, device = (batch,), self.device
        if not self.use_logit_normal_sampler:
            return torch.rand(shape, device=device)

        mean_t = torch.full(shape, mean, device=device)
        std_t = torch.full(shape, std, device=device)
        t = torch.normal(mean_t, std_t).sigmoid()
        return t.clamp(min=self.eps, max=1.0 - self.eps)

    def get_noise(self, batch_size=1, data_shape=None):
        device = self.device

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), "shape of the data must be passed in, or set at init or during training"

        noise = torch.randn((batch_size, *data_shape), device=device) * self.noise_std_dev
        return noise

    def r_value(self) -> float:
        step = min(self.step, self.r_total_steps)
        denom = max(self.r_total_steps, 1)
        ratio = step / denom

        if self.r_schedule == "constant":
            return float(self.r_end)

        if self.r_schedule == "linear":
            return float(self.r_init + (self.r_end - self.r_init) * ratio)

        if self.r_schedule == "cosine":
            return float(self.r_end + (self.r_init - self.r_end) * 0.5 * (1.0 + math.cos(math.pi * ratio)))

        # exponential (best performing in paper)
        if self.r_init <= 0 or self.r_end <= 0:
            return float(self.r_end)

        log_start = math.log(self.r_init)
        log_end = math.log(self.r_end)
        return float(math.exp(log_start + ratio * (log_end - log_start)))

    def predict_velocity(self, x, times, target_times, cond=None):
        cond_kwargs = dict()
        if self.accept_cond:
            cond_kwargs.update(cond=cond)

        model_output = self.model(x, times, s=target_times, **cond_kwargs)

        # if model outputs clean, derive velocity

        if not self.model_output_clean:
            velocity = model_output
        else:
            times = append_dims(times, model_output.ndim - 1)
            velocity = (model_output - x) / times.clamp_min(self.eps)

        return velocity

    def solution(self, x, times, target_times, cond=None, detach_model: bool = False):
        """Compute the Euler-parameterized solution f_θ(x_t, t, s) = x_t + (s - t) * F_θ(x_t, t, s)."""
        velocity = self.predict_velocity(x, times, target_times, cond)
        if detach_model:
            velocity = velocity.detach()
        delta = append_dims(target_times - times, x.ndim - 1)
        return x + delta * velocity

    def flow_matching_loss(self, xt, times, flow_target, cond):
        pred = self.predict_velocity(xt, times, times, cond)

        mse = (pred - flow_target).pow(2).flatten(1).mean(dim=1)

        if self.use_adaptive_loss_weight:
            w = 1.0 / (mse + self.eps).pow(self.adaptive_loss_weight_p)
            mse = mse * w.detach()

        return mse.mean()

    def solution_consistency_loss(self, xt, t, s, l, flow_target, cond):
        pred = self.solution(xt, t, s, cond)

        xl = xt + append_dims(l - t, xt.ndim - 1) * flow_target

        with torch.no_grad():
            target = self.solution(xl, l, s, cond)

        mse = (pred - target).pow(2).flatten(1).mean(dim=1)

        if self.use_adaptive_loss_weight:
            dt = (t - l).clamp_min(1e-4)
            b = (t - s).clamp_min(1e-4)
            scaled = mse / (dt * dt)
            w = (1.0 / (dt * b)) * (1.0 / (scaled + self.eps).pow(self.adaptive_loss_weight_p))
            mse = mse * w.detach()

        return mse.mean()

    def forward(self, data, *, return_loss_breakdown=False, cond=None, noise=None):
        assert xnor(self.accept_cond, exists(cond))

        data = self.normalize_data_fn(data)
        b, device = data.shape[0], data.device
        self.data_shape = default(self.data_shape, data.shape[1:])
        ndim = data.ndim

        if not exists(noise):
            noise = torch.randn_like(data) * self.noise_std_dev

        flow = noise - data

        t_fm_all = self.sample_times(b, mean=self.logit_normal_mean_fm, std=self.logit_normal_std_fm)
        t_scm_all = self.sample_times(b, mean=self.logit_normal_mean_t, std=self.logit_normal_std_t)
        s_scm_all = self.sample_times(b, mean=self.logit_normal_mean_s, std=self.logit_normal_std_s)

        fm_mask = torch.rand(b, device=device) < self.lambda_flow_matching
        scm_mask = ~fm_mask

        def maybe_masked_cond(mask):
            if not (self.accept_cond and exists(cond)):
                return None
            return cond[mask]

        # flow matching loss
        flow_loss = torch.zeros((), device=device)
        if fm_mask.any():
            t_fm = t_fm_all[fm_mask]
            x_t_fm = data[fm_mask].lerp(noise[fm_mask], append_dims(t_fm, ndim - 1))

            flow_loss = self.flow_matching_loss(
                x_t_fm,
                t_fm,
                flow[fm_mask],
                maybe_masked_cond(fm_mask),
            )

        # solution consistency loss
        scm_loss = torch.zeros((), device=device)
        if scm_mask.any():
            t_scm = t_scm_all[scm_mask]
            s_scm = s_scm_all[scm_mask]

            # enforce 0 <= s < t
            s_scm = torch.minimum(s_scm, t_scm - 1e-4)
            s_scm = torch.clamp(s_scm, min=0.0)

            # l = t + (s - t) * r(k)
            r_value = torch.tensor(self.r_value(), device=device, dtype=t_scm.dtype)

            l_scm = t_scm + (s_scm - t_scm) * r_value
            # enforce s < l < t
            l_scm = torch.maximum(l_scm, s_scm + 1e-4)
            l_scm = torch.minimum(l_scm, t_scm - 1e-4)

            x_t_scm = data[scm_mask].lerp(noise[scm_mask], append_dims(t_scm, ndim - 1))
            flow_scm = flow[scm_mask]

            scm_loss = self.solution_consistency_loss(
                x_t_scm,
                t_scm,
                s_scm,
                l_scm,
                flow_scm,
                maybe_masked_cond(scm_mask),
            )

        total_loss = flow_loss + scm_loss
        self.step += 1

        if return_loss_breakdown:
            return total_loss, (flow_loss.detach(), scm_loss.detach())
        return total_loss

    @torch.no_grad()
    def slow_sample(self, steps=4, batch_size=1, noise=None, data_shape=None, cond=None):
        assert steps >= 1

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), "shape of the data must be passed in, or set at init or during training"

        device = self.device
        maybe_cond = (cond,) if self.accept_cond else ()

        if not exists(noise):
            noise = self.get_noise(batch_size=batch_size, data_shape=data_shape)

        times = torch.linspace(1.0, 0.0, steps + 1, device=device)
        current = noise

        for idx in range(steps):
            t_curr, t_next = times[idx], times[idx + 1]
            t_batch = t_curr.expand(batch_size)
            s_batch = t_next.expand(batch_size)
            velocity = self.predict_velocity(current, t_batch, s_batch, *maybe_cond)
            delta = append_dims(s_batch - t_batch, current.ndim - 1)
            current = current + delta * velocity

        return self.unnormalize_data_fn(current)

    def sample(self, batch_size=None, data_shape=None, requires_grad=False, cond=None, noise=None, steps=1):
        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), "shape of the data must be passed in, or set at init or during training"

        maybe_cond = ()
        if self.accept_cond:
            batch_size = cond.shape[0]
            maybe_cond = (cond,)

        batch_size = default(batch_size, 1)

        if steps > 1:
            return self.slow_sample(steps=steps, batch_size=batch_size, data_shape=data_shape, cond=cond, noise=noise)

        assert xnor(self.accept_cond, exists(cond))

        device = self.device
        context = nullcontext if requires_grad else torch.no_grad

        if not exists(noise):
            noise = self.get_noise(batch_size=batch_size, data_shape=data_shape)

        times = ones(batch_size, device=device)
        target_times = torch.zeros_like(times)

        with context():
            velocity = self.predict_velocity(noise, times, target_times, *maybe_cond)
            delta = append_dims(target_times - times, noise.ndim - 1)
            data = noise + delta * velocity

        return self.unnormalize_data_fn(data)


# trainer

import math
from pathlib import Path
from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from ema_pytorch import EMA
from einops import rearrange
from tqdm import tqdm
from .rectified_flow import ImageDataset


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


class SoFlowTrainer(Module):
    def __init__(
        self,
        flow_model: dict | SoFlow,
        *,
        dataset: dict | Dataset,
        num_train_steps = 70_000,
        learning_rate = 3e-4,
        batch_size =16,
        checkpoints_folder: str = "./checkpoints",
        results_folder: str = "./results",
        save_results_every: int = 100,
        checkpoint_every: int = 1000,
        num_samples: int = 16,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        dl_kwargs: dict = dict(),
        use_ema = True,
        max_grad_norm = 0.5,
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        if isinstance(dataset, dict):
            dataset = ImageDataset(**dataset)

        if isinstance(flow_model, dict):
            flow_model = SoFlow(**flow_model)

        self.model = flow_model
        self.use_ema = use_ema
        self.ema_model = None

        if self.is_main and use_ema:
            self.ema_model = EMA(self.model, forward_method_names=("sample",), **ema_kwargs)
            self.ema_model.to(self.accelerator.device)
            self.ema_model.eval()

        self.optimizer = AdamW(flow_model.parameters(), lr=learning_rate, **adam_kwargs)
        self.dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, **dl_kwargs)

        self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)

        self.num_train_steps = num_train_steps
        self.max_grad_norm = max_grad_norm

        # folders

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows**2) == num_samples, f"{num_samples} must be a square"
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
            model=self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model=self.ema_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

        torch.save(save_package, str(self.checkpoints_folder / path))

    def load(self, path):
        if not self.is_main:
            return

        load_package = torch.load(path)

        self.model.load_state_dict(load_package["model"])
        self.ema_model.load_state_dict(load_package["ema_model"])
        self.optimizer.load_state_dict(load_package["optimizer"])

    def log(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def log_images(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def sample(self, fname):
        eval_model = default(self.ema_model, self.model)
        dl = cycle(self.dl)
        mock_data = next(dl)
        data_shape = mock_data.shape[1:]

        with torch.no_grad():
            sampled = eval_model.sample(batch_size=self.num_samples, data_shape=data_shape)

        sampled = rearrange(sampled, "(row col) c h w -> c (row h) (col w)", row=self.num_sample_rows)
        sampled.clamp_(0.0, 1.0)

        save_image(sampled, fname)
        return sampled

    def forward(self):
        dl = cycle(self.dl)
        pbar = tqdm(range(self.num_train_steps))

        for ind in pbar:
            step = ind + 1

            self.model.train()

            data = next(dl)

            loss, (flow_loss, scm_loss) = self.model(data, return_loss_breakdown=True)

            if divisible_by(step, 10):
                pbar.set_postfix({"loss": f"{loss.item():.3f}, flow: {flow_loss.item():.3f}, scm: {scm_loss.item():.3f}"})

            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.is_main and self.use_ema:
                self.ema_model.ema_model.data_shape = self.model.data_shape
                self.ema_model.update()

            if self.is_main:
                if divisible_by(step, self.save_results_every):
                    self.accelerator.wait_for_everyone()

                    sampled = self.sample(fname=str(self.results_folder / f"results.{step}.png"))

                    self.log_images(sampled, step=step)

                if divisible_by(step, self.checkpoint_every):
                    self.save(f"checkpoint.{step}.pt")

        print("training complete")
