import pytest
import torch
from torch import nn
from einops import rearrange

from rectified_flow_pytorch import (
    NanoFlow,
    RectifiedFlow,
    Unet,
    LapFlow,
    LapFlowDiT,
    XMWrapper,
    XMRefinementWrapper
)
from rectified_flow_pytorch.xm_wrapper import exists, default

def test_xm_wrapper_nanoflow():
    torch.manual_seed(42)
    conv_model = nn.Conv2d(3, 3, 1)
    nano_flow = NanoFlow(conv_model, data_shape = (3, 16, 16))

    wrapped_nano_flow = XMWrapper(nano_flow, candidates = 4)

    data = torch.randn(8, 3, 16, 16)
    loss = wrapped_nano_flow(data)
    assert loss.ndim == 0
    loss.backward()
    assert exists(conv_model.weight.grad)

    wrapped_nano_flow_chunked = XMWrapper(nano_flow, candidates = 4, max_batch_size = 8)
    conv_model.zero_grad()
    loss_chunked = wrapped_nano_flow_chunked(data)
    loss_chunked.backward()
    assert exists(conv_model.weight.grad)

    sampled = wrapped_nano_flow.sample(batch_size = 4)
    assert sampled.shape == (4, 3, 16, 16)

def test_xm_wrapper_rectified_flow():
    torch.manual_seed(42)
    unet = Unet(dim = 32)
    rf = RectifiedFlow(unet, data_shape = (3, 16, 16))

    wrapped_rf = XMWrapper(rf, candidates = 4)

    data = torch.randn(4, 3, 16, 16)
    loss = wrapped_rf(data)
    assert loss.ndim == 0
    loss.backward()
    assert exists(next(unet.parameters()).grad)

    wrapped_rf_chunked = XMWrapper(rf, candidates = 4, max_batch_size = 4)
    unet.zero_grad()
    loss_chunked = wrapped_rf_chunked(data)
    loss_chunked.backward()
    assert exists(next(unet.parameters()).grad)

    sampled = wrapped_rf.sample(batch_size = 4)
    assert sampled.shape == (4, 3, 16, 16)

def test_xm_refinement_wrapper_lapflow():
    torch.manual_seed(42)
    dit_model = LapFlowDiT(
        base_image_size = 16,
        patch_size = 2,
        dim = 64,
        depth = 2,
        heads = 4,
        mlp_dim = 128,
        num_scales = 2
    )

    lap_flow = LapFlow(
        model = dit_model,
        normalize_data_fn = lambda t: (t * 2) - 1,
        unnormalize_data_fn = lambda t: (t + 1) * 0.5
    )

    wrapped_lap_flow = XMRefinementWrapper(lap_flow, candidates = 4, times_kwarg = None, noise_kwarg = None)

    data = torch.rand(4, 3, 16, 16)
    loss = wrapped_lap_flow(data)
    assert loss.ndim == 0
    loss.backward()
    assert exists(next(dit_model.parameters()).grad)

    wrapped_lap_flow_chunked = XMRefinementWrapper(lap_flow, candidates = 4, max_batch_size = 4, times_kwarg = None, noise_kwarg = None)
    dit_model.zero_grad()
    loss_chunked = wrapped_lap_flow_chunked(data)
    loss_chunked.backward()
    assert exists(next(dit_model.parameters()).grad)

def test_xm_refinement_wrapper_noise_and_times_handling():
    torch.manual_seed(42)

    # 1. NanoFlow
    conv_model = nn.Conv2d(3, 3, 1)
    nano_flow = NanoFlow(conv_model, data_shape = (3, 16, 16))
    wrapped_nano_flow = XMRefinementWrapper(nano_flow, candidates = 4, times_kwarg = 'times', noise_kwarg = 'noise')

    data_nano = torch.randn(4, 3, 16, 16)
    loss_nano_auto = wrapped_nano_flow(data_nano)
    assert loss_nano_auto.ndim == 0

    custom_noise_nano = torch.randn(4, 3, 16, 16)
    custom_times_nano = torch.rand(4)
    loss_nano_custom = wrapped_nano_flow(data_nano, noise = custom_noise_nano, times = custom_times_nano)
    assert loss_nano_custom.ndim == 0

    # 2. RectifiedFlow
    unet = Unet(dim = 32)
    rf = RectifiedFlow(unet, data_shape = (3, 16, 16))
    wrapped_rf = XMRefinementWrapper(rf, candidates = 4, times_kwarg = 'times', noise_kwarg = 'noise')

    data_rf = torch.randn(4, 3, 16, 16)
    loss_rf_auto = wrapped_rf(data_rf)
    assert loss_rf_auto.ndim == 0

    custom_noise_rf_bk = torch.randn(4, 4, 3, 16, 16) # (batch, candidates, ...)
    custom_times_rf = torch.rand(4)
    loss_rf_custom = wrapped_rf(data_rf, noise = custom_noise_rf_bk, times = custom_times_rf)
    assert loss_rf_custom.ndim == 0

def make_refinement_test_setup(seed = 0, batch = 4):
    torch.manual_seed(seed)
    conv = nn.Conv2d(3, 3, 1)
    flow = NanoFlow(conv, data_shape = (3, 16, 16))
    return conv, flow, torch.randn(batch, 3, 16, 16)

@pytest.mark.parametrize(
    'optimizer_kind',
    ['default', 'class', 'instance'],
    ids = ['default-sgd', 'optimizer-class', 'optimizer-instance']
)
def test_xm_refinement_wrapper_three_steps(optimizer_kind):
    conv, flow, data = make_refinement_test_setup(seed = 42, batch = 8)

    candidates, refinement_steps = 4, 3

    if optimizer_kind == 'default':
        refinement_optimizer = None
    elif optimizer_kind == 'class':
        refinement_optimizer = torch.optim.AdamW
    else:
        refinement_optimizer = torch.optim.AdamW(conv.parameters(), lr = 0.01)

    wrapped = XMRefinementWrapper(
        flow,
        candidates = candidates,
        refinement_steps = refinement_steps,
        refinement_lr = 0.01
    )

    params_before = {name: value.clone() for name, value in conv.state_dict().items()}
    loss, info = wrapped(
        data,
        refinement_optimizer = refinement_optimizer,
        return_refinement_info = True
    )

    assert info.raw_candidate_losses.shape == (8, candidates)
    assert info.refined_candidate_losses.shape == (8, candidates)

    # 3 steps reduce every candidate's batch loss
    assert (info.refined_candidate_losses.mean(0) < info.raw_candidate_losses.mean(0)).all()

    # winner is selected by refined loss, trained on its original loss
    assert torch.equal(info.best_candidates, info.refined_candidate_losses.argmin(-1))

    selected_raw = info.raw_candidate_losses.gather(1, info.best_candidates[:, None]).squeeze(1)
    assert torch.allclose(loss.detach(), selected_raw.mean(), atol = 1e-5)

    # inner steps are a probe and must not leak into live weights
    for name, value in conv.state_dict().items():
        assert torch.equal(value, params_before[name])

    loss.backward()
    assert exists(conv.weight.grad)

def test_xm_refinement_wrapper_same_and_next_batch_scoring():
    conv, flow, data = make_refinement_test_setup(seed = 42, batch = 8)
    next_data = torch.randn(6, 3, 16, 16)

    wrapped = XMRefinementWrapper(
        flow,
        candidates = 4,
        refinement_steps = 3,
        refinement_lr = 0.01
    )

    # same batch: per sample winner
    loss_same, info_same = wrapped(data, return_refinement_info = True)
    assert info_same.raw_candidate_losses.shape == (8, 4)
    assert info_same.refined_candidate_losses.shape == (8, 4)
    assert info_same.best_candidates.shape == (8,)
    assert loss_same.ndim == 0

    # next batch: one winner for the whole batch
    loss, info = wrapped(data, next_batch = next_data, return_refinement_info = True)

    assert info.raw_candidate_losses.shape == (8, 4)
    assert info.refined_candidate_losses.shape == (6, 4)

    # one winner for the whole batch, chosen by mean held-out loss
    assert info.best_candidates.ndim == 0
    assert info.best_candidates == info.refined_candidate_losses.mean(0).argmin()

    selected_raw = info.raw_candidate_losses.mean(0)[info.best_candidates]
    assert torch.allclose(loss.detach(), selected_raw, atol = 1e-5)

    loss.backward()
    assert exists(conv.weight.grad)

def test_xm_refinement_wrapper_copies_optimizer_state():
    conv, flow, data = make_refinement_test_setup()

    class RecordingSGD(torch.optim.SGD):
        momentum_seen = []

        def step(self, *args, **kwargs):
            has_state = all(len(self.state[param]) > 0 for param in self.param_groups[0]['params'])
            type(self).momentum_seen.append(has_state)
            return super().step(*args, **kwargs)

    optimizer = RecordingSGD(conv.parameters(), lr = 0.1, momentum = 0.9)

    # prime the live optimizer so every param has momentum state
    loss = flow(data)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    RecordingSGD.momentum_seen.clear()

    wrapped = XMRefinementWrapper(
        flow,
        candidates = 4,
        refinement_steps = 3,
        refinement_optimizer = optimizer
    )

    assert wrapped(data).ndim == 0

    # every inner step starts from a copy of the primed state
    assert RecordingSGD.momentum_seen
    assert all(RecordingSGD.momentum_seen)

    # refinement can be skipped per forward
    RecordingSGD.momentum_seen.clear()
    assert wrapped(data, refinement_steps = 0).ndim == 0
    assert RecordingSGD.momentum_seen == []

def test_xm_antithetic_noise():
    seen_noise = []

    class CaptureFlow(nn.Module):
        def forward(self, data, noise = None, times = None, loss_reduction = 'mean'):
            seen_noise.append(noise)
            noise = default(noise, torch.randn_like(data))
            loss = (data - noise).square()
            return loss if loss_reduction == 'none' else loss.mean()

    data = torch.randn(3, 4)

    # default antithetic_noise = True pairs candidates as (+z, -z)

    wrapped = XMWrapper(CaptureFlow(), candidates = 4)
    assert wrapped(data).ndim == 0

    paired = rearrange(seen_noise.pop(), '(b k) d -> b k d', k = 4)
    assert torch.allclose(paired[:, :2], -paired[:, 2:])

    # explicit noise = None still triggers antithetic sampling; custom noise is preserved

    assert wrapped(data, noise = None).ndim == 0
    explicit_none = rearrange(seen_noise.pop(), '(b k) d -> b k d', k = 4)
    assert torch.allclose(explicit_none[:, :2], -explicit_none[:, 2:])

    custom_noise = torch.randn_like(data)
    assert wrapped(data, noise = custom_noise).ndim == 0
    passed = rearrange(seen_noise.pop(), '(b k) d -> b k d', k = 4)
    assert torch.allclose(passed[:, 0], custom_noise)

    # odd candidate count keeps exact antipodal prefix

    assert wrapped(data, candidates = 3).ndim == 0
    odd = rearrange(seen_noise.pop(), '(b k) d -> b k d', k = 3)
    assert torch.allclose(odd[:, 0], -odd[:, 2])

    # can be disabled at init or per forward

    wrapped_off = XMWrapper(CaptureFlow(), candidates = 4, antithetic_noise = False)
    assert wrapped_off(data).ndim == 0
    assert not exists(seen_noise.pop())

    assert wrapped(data, antithetic_noise = False).ndim == 0
    assert not exists(seen_noise.pop())

    # refinement wrapper uses the same antithetic pairing and supports per-call toggle

    ref = XMRefinementWrapper(CaptureFlow(), candidates = 4, refinement_steps = 0)
    assert ref(data).ndim == 0
    ref_paired = rearrange(seen_noise.pop(), '(b k) d -> b k d', k = 4)
    assert torch.allclose(ref_paired[:, :2], -ref_paired[:, 2:])

    assert ref(data, antithetic_noise = False).ndim == 0
    ref_iid = rearrange(seen_noise.pop(), '(b k) d -> b k d', k = 4)
    assert not torch.allclose(ref_iid[:, :2], -ref_iid[:, 2:])

@pytest.mark.parametrize('antithetic_noise', (True, False))
def test_xm_antithetic_e2e(antithetic_noise):
    torch.manual_seed(42)

    # NanoFlow + XMWrapper: forward, backward, optimizer step, sample

    conv = nn.Conv2d(3, 3, 1)
    wrapped_nano = XMWrapper(NanoFlow(conv), candidates = 2, antithetic_noise = antithetic_noise)
    opt = torch.optim.Adam(conv.parameters(), lr = 1e-2)

    weight_before = conv.weight.detach().clone()
    loss = wrapped_nano(torch.randn(4, 3, 8, 8))
    loss.backward()
    assert exists(conv.weight.grad)
    opt.step()
    assert not torch.equal(conv.weight.detach(), weight_before)
    assert wrapped_nano.sample(batch_size = 2, steps = 2).shape == (2, 3, 8, 8)

    # RectifiedFlow + XMWrapper: forward, backward, optimizer step, sample

    unet = Unet(dim = 16, channels = 3, dim_mults = (1, 2))
    wrapped_rf = XMWrapper(RectifiedFlow(unet), candidates = 2, antithetic_noise = antithetic_noise)
    opt_rf = torch.optim.Adam(unet.parameters(), lr = 1e-3)

    loss_rf = wrapped_rf(torch.randn(2, 3, 16, 16))
    loss_rf.backward()
    assert exists(next(unet.parameters()).grad)
    opt_rf.step()
    assert wrapped_rf.sample(batch_size = 2, steps = 2).shape == (2, 3, 16, 16)

    # NanoFlow + XMRefinementWrapper: refinement step, backward, optimizer step

    conv_ref = nn.Conv2d(3, 3, 1)
    wrapped_ref = XMRefinementWrapper(
        NanoFlow(conv_ref),
        candidates = 2,
        refinement_steps = 2,
        refinement_lr = 1e-2,
        antithetic_noise = antithetic_noise
    )
    opt_ref = torch.optim.Adam(conv_ref.parameters(), lr = 1e-2)

    loss_ref = wrapped_ref(torch.randn(4, 3, 8, 8))
    loss_ref.backward()
    assert exists(conv_ref.weight.grad)
    opt_ref.step()

if __name__ == '__main__':
    test_xm_wrapper_nanoflow()
    test_xm_wrapper_rectified_flow()
    test_xm_refinement_wrapper_lapflow()
    test_xm_refinement_wrapper_noise_and_times_handling()

    for optimizer_kind in ('default', 'class', 'instance'):
        test_xm_refinement_wrapper_three_steps(optimizer_kind)

    test_xm_refinement_wrapper_same_and_next_batch_scoring()
    test_xm_refinement_wrapper_copies_optimizer_state()
    test_xm_antithetic_noise()

    for antithetic_noise in (True, False):
        test_xm_antithetic_e2e(antithetic_noise)

    print('All XM wrapper tests passed successfully!')
