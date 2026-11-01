import pytest
import torch
from rectified_flow_pytorch import (
    NanoFlow,
    RectifiedFlow,
    Unet,
    LapFlow,
    LapFlowDiT,
    XMWrapper,
    XMRefinementWrapper
)

def test_xm_wrapper_nanoflow():
    print("Testing XMWrapper with NanoFlow (candidates=4)...")
    torch.manual_seed(42)
    conv_model = torch.nn.Conv2d(3, 3, 1)
    nano_flow = NanoFlow(conv_model, data_shape=(3, 16, 16))

    wrapped_nano_flow = XMWrapper(nano_flow, candidates=4)

    data = torch.randn(8, 3, 16, 16)
    loss = wrapped_nano_flow(data)
    assert loss.ndim == 0
    loss.backward()
    assert conv_model.weight.grad is not None
    print(f"NanoFlow candidates=4 loss: {loss.item():.4f}")

    wrapped_nano_flow_chunked = XMWrapper(nano_flow, candidates=4, max_batch_size=8)
    conv_model.zero_grad()
    loss_chunked = wrapped_nano_flow_chunked(data)
    loss_chunked.backward()
    assert conv_model.weight.grad is not None
    print(f"NanoFlow candidates=4 chunked (max_batch_size=8) loss: {loss_chunked.item():.4f}")

    sampled = wrapped_nano_flow.sample(batch_size=4)
    assert sampled.shape == (4, 3, 16, 16)
    print("NanoFlow sample shape:", sampled.shape)

def test_xm_wrapper_rectified_flow():
    print("Testing XMWrapper with RectifiedFlow (candidates=4)...")
    torch.manual_seed(42)
    unet = Unet(dim=32)
    rf = RectifiedFlow(unet, data_shape=(3, 16, 16))

    wrapped_rf = XMWrapper(rf, candidates=4)

    data = torch.randn(4, 3, 16, 16)
    loss = wrapped_rf(data)
    assert loss.ndim == 0
    loss.backward()
    assert next(unet.parameters()).grad is not None
    print(f"RectifiedFlow candidates=4 loss: {loss.item():.4f}")

    wrapped_rf_chunked = XMWrapper(rf, candidates=4, max_batch_size=4)
    unet.zero_grad()
    loss_chunked = wrapped_rf_chunked(data)
    loss_chunked.backward()
    assert next(unet.parameters()).grad is not None
    print(f"RectifiedFlow candidates=4 chunked (max_batch_size=4) loss: {loss_chunked.item():.4f}")

    sampled = wrapped_rf.sample(batch_size=4)
    assert sampled.shape == (4, 3, 16, 16)
    print("RectifiedFlow sample shape:", sampled.shape)

def test_xm_refinement_wrapper_lapflow():
    print("Testing XMRefinementWrapper with LapFlow (candidates=4)...")
    torch.manual_seed(42)
    dit_model = LapFlowDiT(
        base_image_size=16,
        patch_size=2,
        dim=64,
        depth=2,
        heads=4,
        mlp_dim=128,
        num_scales=2
    )

    lap_flow = LapFlow(
        model=dit_model,
        normalize_data_fn=lambda t: (t * 2) - 1,
        unnormalize_data_fn=lambda t: (t + 1) * 0.5
    )

    wrapped_lap_flow = XMRefinementWrapper(lap_flow, candidates=4, times_kwarg = None, noise_kwarg = None)

    data = torch.rand(4, 3, 16, 16)
    loss = wrapped_lap_flow(data)
    assert loss.ndim == 0
    loss.backward()
    assert next(dit_model.parameters()).grad is not None
    print(f"LapFlow candidates=4 loss: {loss.item():.4f}")

    wrapped_lap_flow_chunked = XMRefinementWrapper(lap_flow, candidates=4, max_batch_size=4, times_kwarg = None, noise_kwarg = None)
    dit_model.zero_grad()
    loss_chunked = wrapped_lap_flow_chunked(data)
    loss_chunked.backward()
    assert next(dit_model.parameters()).grad is not None
    print(f"LapFlow candidates=4 chunked (max_batch_size=4) loss: {loss_chunked.item():.4f}")

def test_xm_refinement_wrapper_noise_and_times_handling():
    print("Testing XMRefinementWrapper noise and times parameter handling for NanoFlow and RectifiedFlow...")
    torch.manual_seed(42)

    # 1. NanoFlow
    conv_model = torch.nn.Conv2d(3, 3, 1)
    nano_flow = NanoFlow(conv_model, data_shape=(3, 16, 16))
    wrapped_nano_flow = XMRefinementWrapper(nano_flow, candidates=4, times_kwarg='times', noise_kwarg='noise')

    data_nano = torch.randn(4, 3, 16, 16)
    loss_nano_auto = wrapped_nano_flow(data_nano)
    assert loss_nano_auto.ndim == 0

    custom_noise_nano = torch.randn(4, 3, 16, 16)
    custom_times_nano = torch.rand(4)
    loss_nano_custom = wrapped_nano_flow(data_nano, noise=custom_noise_nano, times=custom_times_nano)
    assert loss_nano_custom.ndim == 0

    # 2. RectifiedFlow
    unet = Unet(dim=32)
    rf = RectifiedFlow(unet, data_shape=(3, 16, 16))
    wrapped_rf = XMRefinementWrapper(rf, candidates=4, times_kwarg='times', noise_kwarg='noise')

    data_rf = torch.randn(4, 3, 16, 16)
    loss_rf_auto = wrapped_rf(data_rf)
    assert loss_rf_auto.ndim == 0

    custom_noise_rf_bk = torch.randn(4, 4, 3, 16, 16) # (batch, candidates, ...)
    custom_times_rf = torch.rand(4)
    loss_rf_custom = wrapped_rf(data_rf, noise=custom_noise_rf_bk, times=custom_times_rf)
    assert loss_rf_custom.ndim == 0

    print("Noise and times handling test passed for both NanoFlow and RectifiedFlow!")

def make_refinement_test_setup(seed = 0, batch = 4):
    torch.manual_seed(seed)
    conv = torch.nn.Conv2d(3, 3, 1)
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
    assert conv.weight.grad is not None

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
    assert conv.weight.grad is not None

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

if __name__ == '__main__':
    test_xm_wrapper_nanoflow()
    test_xm_wrapper_rectified_flow()
    test_xm_refinement_wrapper_lapflow()
    test_xm_refinement_wrapper_noise_and_times_handling()

    for optimizer_kind in ('default', 'class', 'instance'):
        test_xm_refinement_wrapper_three_steps(optimizer_kind)

    test_xm_refinement_wrapper_same_and_next_batch_scoring()
    test_xm_refinement_wrapper_copies_optimizer_state()
    print("All XM wrapper tests passed successfully!")

