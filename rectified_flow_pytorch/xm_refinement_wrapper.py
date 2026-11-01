from inspect import signature
from collections import namedtuple
from copy import deepcopy

import torch
from torch import cat, stack, is_tensor
from torch.optim import Optimizer, SGD
from torch.func import functional_call
from torch.utils._pytree import tree_map, tree_flatten
from einops import repeat, reduce, rearrange

from rectified_flow_pytorch.xm_wrapper import (
    XMWrapper,
    exists,
    default,
    is_batch_tensor
)

# explorative modeling (forward xm) with candidate refinement
# Alexi Gladstone et al. https://arxiv.org/abs/2607.27372

# helpers

def batch_and_first_tensor(args, kwargs):
    leaves, _ = tree_flatten((args, kwargs))
    first_tensor = next(t for t in leaves if is_tensor(t))
    return first_tensor.shape[0], first_tensor

# refinement info

XMRefinementInfo = namedtuple('XMRefinementInfo', 'raw_candidate_losses refined_candidate_losses best_candidates')

# a batch used only for scoring refined candidates

XMBatch = namedtuple('XMBatch', 'args kwargs batch')

# xm refinement wrapper

class XMRefinementWrapper(XMWrapper):
    def __init__(
        self,
        flow_model,
        candidates = 1,
        max_batch_size = None,
        refinement_steps = 0,
        refinement_lr = 1e-3,
        refinement_optimizer = None,
        refinement_optimizer_kwargs = dict(),
        times_kwarg = 'times',
        noise_kwarg = 'noise'
    ):
        super().__init__(flow_model, candidates = candidates, max_batch_size = max_batch_size)

        assert refinement_steps >= 0, 'refinement_steps must be non-negative'

        # inner sgd steps rank candidates only; fast weights are discarded after selection
        # pass refinement_steps = 0 on forward to skip refinement for a batch
        self.refinement_steps = refinement_steps
        self.refinement_lr = refinement_lr
        self.refinement_optimizer = refinement_optimizer
        self.refinement_optimizer_kwargs = refinement_optimizer_kwargs

        self.times_kwarg = times_kwarg
        self.noise_kwarg = noise_kwarg

        self.forward_params = signature(flow_model.forward).parameters
        self.stochastic_kwargs = {times_kwarg, noise_kwarg} - {None}

    # refinement helpers

    def make_functional_state(self):
        # per candidate copies of params and buffers, so the live model is never touched

        param_map = {}
        params = {}

        for name, param in self.flow_model.named_parameters(remove_duplicate = False):
            if id(param) not in param_map:
                param_map[id(param)] = param.detach().clone().requires_grad_(param.requires_grad)

            params[name] = param_map[id(param)]

        buffers = {
            name: buffer.detach().clone()
            for name, buffer in self.flow_model.named_buffers(remove_duplicate = False)
        }

        trainable = [clone for clone in param_map.values() if clone.requires_grad]
        return params, buffers, trainable, param_map

    def clone_refinement_optimizer(self, optimizer, param_map, optimizer_kwargs):
        # rebuild on the functional params, then copy the live optimizer state by position

        param_groups = []

        for group in optimizer.param_groups:
            cloned_params = [param_map.get(id(param)) for param in group['params']]
            assert all(map(exists, cloned_params)), 'refinement_optimizer must be built from the flow model parameters'

            param_groups.append({**group, 'params': cloned_params})

        cloned = type(optimizer)(param_groups)
        cloned.load_state_dict(deepcopy(optimizer.state_dict()))
        cloned.defaults = dict(optimizer.defaults)

        for group in cloned.param_groups:
            group.update(optimizer_kwargs)

        return cloned

    def make_refinement_optimizer(self, params, refinement_lr, refinement_optimizer, refinement_optimizer_kwargs, param_map = None):
        if not exists(refinement_optimizer):
            return SGD(params, lr = refinement_lr)

        optimizer_kwargs = dict(refinement_optimizer_kwargs or {})

        if isinstance(refinement_optimizer, Optimizer):
            assert exists(param_map), 'param_map is required to copy an optimizer instance'
            return self.clone_refinement_optimizer(refinement_optimizer, param_map, optimizer_kwargs)

        if isinstance(refinement_optimizer, type) and issubclass(refinement_optimizer, Optimizer):
            optimizer_kwargs.setdefault('lr', refinement_lr)
            return refinement_optimizer(params, **optimizer_kwargs)

        if callable(refinement_optimizer):
            return refinement_optimizer(params)

        raise TypeError('refinement_optimizer must be an optimizer class, instance, or factory')

    def functional_forward(self, params, buffers, args, kwargs, loss_reduction = None):
        kwargs = dict(kwargs)

        if exists(loss_reduction) and self.has_loss_reduction:
            kwargs['loss_reduction'] = loss_reduction

        return functional_call(self.flow_model, (params, buffers), args, kwargs)

    def scalar_loss(self, params, buffers, args, kwargs):
        loss = self.functional_forward(params, buffers, args, kwargs, 'none')
        return loss.mean() if loss.ndim == 0 else reduce(loss, 'b ... -> b', 'mean').mean()

    @torch.no_grad()
    def per_sample_loss(self, params, buffers, args, kwargs, batch):
        if self.has_loss_reduction:
            loss = self.functional_forward(params, buffers, args, kwargs, 'none')
            return loss.expand(batch) if loss.ndim == 0 else reduce(loss, 'b ... -> b', 'mean')

        losses = []

        for ind in range(batch):
            sample_args, sample_kwargs = tree_map(
                lambda t: t[ind:ind + 1] if is_batch_tensor(t, batch) else t,
                (args, kwargs)
            )

            loss = self.functional_forward(params, buffers, sample_args, sample_kwargs)
            losses.append(loss.reshape(1))

        return cat(losses, dim = 0)

    # input helpers

    def add_stochastic_kwargs(self, kwargs, batch, first_tensor, candidates = None):
        kwargs = dict(kwargs)
        device = first_tensor.device

        if exists(self.times_kwarg) and self.times_kwarg not in kwargs and self.times_kwarg in self.forward_params:
            kwargs[self.times_kwarg] = torch.rand(batch, device = device)

        if exists(self.noise_kwarg) and self.noise_kwarg not in kwargs and self.noise_kwarg in self.forward_params:
            noise_shape = default(self.data_shape, first_tensor.shape[1:])
            noise_dtype = first_tensor.dtype if first_tensor.is_floating_point() else torch.float32
            noise_batch_shape = (batch,) if not exists(candidates) else (batch, candidates)

            kwargs[self.noise_kwarg] = torch.randn(
                (*noise_batch_shape, *noise_shape),
                device = device,
                dtype = noise_dtype
            )

        return kwargs

    def unpack_next_batch(self, next_batch):
        if isinstance(next_batch, dict):
            return (), dict(next_batch)

        if isinstance(next_batch, (tuple, list)):
            is_args_kwargs_pair = (
                len(next_batch) == 2 and
                isinstance(next_batch[0], (tuple, list)) and
                isinstance(next_batch[1], dict)
            )

            if is_args_kwargs_pair:
                return tuple(next_batch[0]), dict(next_batch[1])

            return tuple(next_batch), {}

        return (next_batch,), {}

    def expand_inputs(self, args, kwargs, batch, candidates):
        total = batch * candidates

        def expand_tensor(t, candidate_specific = False):
            if not is_tensor(t) or t.ndim == 0:
                return t

            if t.shape[0] == total:
                return t

            if t.shape[0] != batch:
                return t

            if candidate_specific and t.ndim > 1 and t.shape[1] == candidates:
                return rearrange(t, 'b k ... -> (b k) ...')

            return repeat(t, 'b ... -> (b k) ...', k = candidates)

        args_K = tree_map(expand_tensor, args)
        kwargs_K = {
            key: tree_map(lambda t: expand_tensor(t, key in self.stochastic_kwargs), value)
            for key, value in kwargs.items()
        }

        return args_K, kwargs_K

    # forward paths

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

    def refined_loss(
        self,
        args,
        kwargs,
        batch,
        candidates,
        max_batch_size,
        refinement_steps,
        refinement_lr,
        refinement_optimizer,
        refinement_optimizer_kwargs,
        score = None
    ):
        total = batch * candidates
        _, first_tensor = batch_and_first_tensor(args, kwargs)
        device = first_tensor.device
        base_indices = torch.arange(batch, device = device) * candidates

        def select_candidate(t, indices):
            return t[indices] if is_batch_tensor(t, total) else t

        # keep the initial candidate losses so the winner does not need a final forward
        raw_losses = self.best_of_k_losses(args, kwargs, batch, candidates, max_batch_size)

        refined_losses = []

        for cand_ind in range(candidates):
            candidate_indices = base_indices + cand_ind

            candidate_args, candidate_kwargs = tree_map(
                lambda t: select_candidate(t, candidate_indices),
                (args, kwargs)
            )

            params, buffers, trainable, param_map = self.make_functional_state()
            assert trainable, 'refinement requires trainable parameters'

            optimizer = self.make_refinement_optimizer(
                trainable,
                refinement_lr,
                refinement_optimizer,
                refinement_optimizer_kwargs,
                param_map
            )

            for _ in range(refinement_steps):
                optimizer.zero_grad()
                loss = self.scalar_loss(params, buffers, candidate_args, candidate_kwargs)
                loss.backward()
                optimizer.step()

            if exists(score):
                refined_loss = self.per_sample_loss(params, buffers, score.args, score.kwargs, score.batch)
            else:
                refined_loss = self.per_sample_loss(params, buffers, candidate_args, candidate_kwargs, batch)

            refined_losses.append(refined_loss)

        refined_losses = stack(refined_losses, dim = -1)

        # consecutive batch scoring picks one candidate for the whole batch
        if exists(score):
            best_candidate = refined_losses.mean(dim = 0).argmin()
            loss = raw_losses[:, best_candidate].mean()

            info = XMRefinementInfo(raw_losses.detach(), refined_losses.detach(), best_candidate)
            return loss, info

        best_candidates = refined_losses.argmin(dim = -1)
        loss = raw_losses.gather(1, best_candidates[:, None]).squeeze(1).mean()

        info = XMRefinementInfo(raw_losses.detach(), refined_losses.detach(), best_candidates)
        return loss, info

    # score on the same batch by default; pass next_batch to score on a consecutive batch

    def forward(
        self,
        *args,
        candidates = None,
        max_batch_size = None,
        refinement_steps = None,
        refinement_lr = None,
        refinement_optimizer = None,
        refinement_optimizer_kwargs = None,
        next_batch = None,
        return_refinement_info = False,
        **kwargs
    ):
        candidates = default(candidates, self.candidates)
        max_batch_size = default(max_batch_size, self.max_batch_size)

        refinement_steps = default(refinement_steps, self.refinement_steps)
        refinement_lr = default(refinement_lr, self.refinement_lr)
        refinement_optimizer = default(refinement_optimizer, self.refinement_optimizer)
        refinement_optimizer_kwargs = default(refinement_optimizer_kwargs, self.refinement_optimizer_kwargs)

        assert refinement_steps >= 0, 'refinement_steps must be non-negative'

        if candidates == 1:
            loss = self.flow_model(*args, **kwargs)
            return (loss, None) if return_refinement_info else loss

        batch, first_tensor = batch_and_first_tensor(args, kwargs)

        # maybe sample shared times and candidate specific noises

        kwargs = self.add_stochastic_kwargs(kwargs, batch, first_tensor, candidates)

        use_refinement = refinement_steps > 0 and torch.is_grad_enabled()

        # maybe prepare a consecutive batch for held-out candidate scoring

        score = None

        if exists(next_batch) and use_refinement:
            score_args, score_kwargs = self.unpack_next_batch(next_batch)
            score_batch, score_first_tensor = batch_and_first_tensor(score_args, score_kwargs)
            score_kwargs = self.add_stochastic_kwargs(score_kwargs, score_batch, score_first_tensor)

            score = XMBatch(score_args, score_kwargs, score_batch)

        args_K, kwargs_K = self.expand_inputs(args, kwargs, batch, candidates)

        if not use_refinement:
            candidate_losses = self.best_of_k_losses(
                args_K,
                kwargs_K,
                batch = batch,
                candidates = candidates,
                max_batch_size = max_batch_size
            )

            loss = candidate_losses.amin(dim = -1).mean()

            if not return_refinement_info:
                return loss

            info = XMRefinementInfo(
                candidate_losses.detach(),
                candidate_losses.detach(),
                candidate_losses.argmin(dim = -1)
            )
            return loss, info

        loss, info = self.refined_loss(
            args_K,
            kwargs_K,
            batch = batch,
            candidates = candidates,
            max_batch_size = max_batch_size,
            refinement_steps = refinement_steps,
            refinement_lr = refinement_lr,
            refinement_optimizer = refinement_optimizer,
            refinement_optimizer_kwargs = refinement_optimizer_kwargs,
            score = score
        )

        return (loss, info) if return_refinement_info else loss
