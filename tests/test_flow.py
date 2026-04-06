import torch
from torch import nn
from torch.nn import Module

import pytest
param = pytest.mark.parametrize

def test_nano_flow():
    from rectified_flow_pytorch.nano_flow import NanoFlow

    model = torch.nn.Conv2d(3, 3, 1)

    nano_flow = NanoFlow(model)
    data = torch.randn(16, 3, 16, 16)

    loss = nano_flow(data)
    loss.backward()

    sampled = nano_flow.sample(batch_size = 16)
    assert sampled.shape == data.shape


@param('add_recon_loss', (False, True))
@param('accept_cond', (False, True))
@param('use_adaptive_loss_weight', (False, True))
def test_mean_flow(
    add_recon_loss,
    accept_cond,
    use_adaptive_loss_weight
):

    from einx import add
    from rectified_flow_pytorch.mean_flow import MeanFlow

    class Unet(Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Conv2d(3, 3, 1)

        def forward(self, x, t, r, cond = None):
            return add('b ..., b, b', self.net(x), t, r)

    model = Unet()

    mean_flow = MeanFlow(model, add_recon_loss = add_recon_loss, accept_cond = accept_cond, use_adaptive_loss_weight = use_adaptive_loss_weight)
    data = torch.randn(16, 3, 16, 16)
    cond = data.clone() if accept_cond else None

    loss = mean_flow(data, cond = cond)
    loss.backward()

    sampled = mean_flow.sample(batch_size = 16, cond = cond)
    assert sampled.shape == data.shape


@param('add_recon_loss', (False, True))
@param('accept_cond', (False, True))
@param('use_adaptive_loss_weight', (False, True))
@param('prob_default_flow_obj', (0.0, 0.5, 1.0))  # different boundary condition probabilities
def test_split_mean_flow(
    add_recon_loss,
    accept_cond,
    use_adaptive_loss_weight,
    prob_default_flow_obj
):
    from einx import add
    from rectified_flow_pytorch.split_mean_flow import SplitMeanFlow

    class Unet(Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Conv2d(3, 3, 1)

        def forward(self, x, t, r, cond = None):
            # r represents delta_time (t - r_start) in the implementation
            return add('b ..., b, b', self.net(x), t, r)

    model = Unet()

    split_mean_flow = SplitMeanFlow(
        model,
        add_recon_loss = add_recon_loss,
        accept_cond = accept_cond,
        use_adaptive_loss_weight = use_adaptive_loss_weight,
        prob_default_flow_obj = prob_default_flow_obj
    )

    data = torch.randn(16, 3, 16, 16)
    cond = data.clone() if accept_cond else None

    loss = split_mean_flow(data, cond = cond)
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()

    for steps in [1, 2, 8]:
        sampled = split_mean_flow.sample(
            batch_size = 16,
            cond = cond,
            steps = steps
        )
        assert sampled.shape == data.shape
        assert not torch.isnan(sampled).any()

    noise = torch.randn_like(data)
    sampled_with_noise = split_mean_flow.sample(
        batch_size = 16,
        cond = cond,
        noise = noise
    )
    assert sampled_with_noise.shape == data.shape

    if add_recon_loss:
        total_loss, (flow_loss, recon_loss) = split_mean_flow(
            data,
            return_loss_breakdown = True,
            cond = cond
        )
        assert isinstance(flow_loss, torch.Tensor)
        assert isinstance(recon_loss, torch.Tensor)
        assert torch.allclose(total_loss, flow_loss + recon_loss * split_mean_flow.recon_loss_weight)

@param('horizon_consistency', (False, True))
@param('action_conditioning', (False, True))
def test_td_flow(
    horizon_consistency,
    action_conditioning
):
    from rectified_flow_pytorch.td_flow import TDFlow
    from rectified_flow_pytorch.rectified_flow import Unet

    policy = None
    action_embedder = None

    if action_conditioning:
        from discrete_continuous_embed_readout import Embed

        action_embedder = Embed(
            num_discrete = 10,
            num_continuous = 4,
            dim = 128
        )

        class Policy(Module):
            def forward(self, state):
                batch = state.shape[0]
                discrete = torch.randint(0, 10, (batch,))
                continuous = torch.randn(batch, 4)
                return (discrete, continuous)

        policy = Policy()

    model = Unet(
        32,
        has_image_cond = True,
        accept_cond = horizon_consistency,
        dim_cond = 3,
        action_embedder = action_embedder
    )

    td_flow = TDFlow(
        model,
        discount_factor = 0.996,
        horizon_consistency = horizon_consistency,
        policy = policy
    )

    state = torch.randn(5, 3, 32, 32)
    next_state = torch.randn(5, 3, 32, 32)
    is_terminal = torch.tensor([False, True, False, False, True])

    action = policy(state) if action_conditioning else None

    loss = td_flow(state, next_state, action = action, is_terminal = is_terminal)

    loss.backward()

    td_flow.update_ema()

    pred = td_flow(state, action = action)
    assert pred.shape == state.shape

@param('use_symlog', (False, True))
@param('prob_state_generation', (0.0, 1.0))
@param('use_state_normalization', (False, True))
def test_value_flow(
    use_symlog,
    prob_state_generation,
    use_state_normalization
):
    from rectified_flow_pytorch.value_flow import ValueFlow
    from einops import rearrange
    from torch import nn

    class DummyNet(nn.Module):
        def __init__(self, state_dim = 64):
            super().__init__()
            self.w = nn.Parameter(torch.ones(1))
            self.to_state = nn.Linear(1, state_dim)

        def forward(self, x, times, cond, action = None):
            t = rearrange(times, 'b -> b 1')
            c = rearrange(cond, 'b -> b 1') if cond.ndim == 1 else cond
            state_vel = (x + t + self.to_state(c)) * self.w
            return_vel = (x.mean(dim = -1, keepdim = True) + t + c) * self.w
            return state_vel, return_vel

    state_dim = 64
    dummy_net = DummyNet(state_dim)
    value_flow = ValueFlow(
        dummy_net,
        gamma = 0.99,
        num_flow_steps = 10,
        prob_state_generation = prob_state_generation,
        use_symlog = use_symlog,
        state_normalize_fn = (lambda x: x * 2.) if use_state_normalization else None,
        state_unnormalize_fn = (lambda x: x / 2.) if use_state_normalization else None
    )

    state       = torch.randn(2, state_dim)
    action      = torch.randn(2, state_dim)
    next_state  = torch.randn(2, state_dim)
    next_action = torch.randn(2, state_dim)
    reward      = torch.randn(2)
    dones       = torch.zeros(2)

    # q-learning path

    loss, (loss_dcfm, loss_bcfm, loss_state_gen) = value_flow(
        state = state,
        action = action,
        next_state = next_state,
        next_action = next_action,
        reward = reward,
        dones = dones
    )
    loss.backward()

    # ppo path

    value_flow.zero_grad()
    loss_ppo, _ = value_flow(
        state = state,
        explicit_target_return = torch.randn(2)
    )
    loss_ppo.backward()

    # state → return

    q_values = value_flow.sample_q_value(state, action)
    assert q_values.shape == (2,)

    # return → state

    target_returns = torch.tensor([100., -100.])
    generated_states = value_flow.sample_state(target_returns, state_shape = (state_dim,))
    assert generated_states.shape == (2, state_dim)


def test_value_flow_co_flow_matching():
    from rectified_flow_pytorch.value_flow import ValueFlow
    from einops import rearrange
    from torch import nn

    class DualDummyNet(nn.Module):
        def __init__(self, state_dim=10, channels=3):
            super().__init__()
            self.w = nn.Parameter(torch.ones(1))
            self.to_image_vel = nn.Conv2d(channels, channels, 1)
            self.to_vector_q = nn.Linear(state_dim, 1)

        def forward(self, x, times, cond, action=None):
            t = rearrange(times, 'b -> b 1')
            c = rearrange(cond, 'b -> b 1') if cond.ndim == 1 else cond

            if x.ndim == 4:
                # image co-flow path
                state_vel = self.to_image_vel(x) * self.w
                return state_vel, torch.zeros(x.shape[0], 1, device=x.device)
            else:
                # numeric TD q-value path
                q_value_vel = self.to_vector_q(x) * self.w
                return torch.zeros_like(x), q_value_vel

    dummy_net = DualDummyNet()
    value_flow = ValueFlow(
        dummy_net,
        gamma = 0.99,
        num_flow_steps = 10,
        prob_state_generation = 1.0  # Force state generation explicitly
    )

    vector_state = torch.randn(2, 10)
    image_state = torch.randn(2, 3, 64, 64)
    action = torch.randn(2, 10)
    reward = torch.randn(2)
    explicit_target_return = torch.randn(2)

    loss_generation, _ = value_flow(
        state = vector_state,
        action = action,
        reward = reward,
        explicit_target_return = explicit_target_return,
        flow_state = image_state
    )
    loss_generation.backward()

    assert dummy_net.w.grad is not None, "gradients should flow through the image co-flow path natively"

    # testing pure image-state sampling
    generated_images = value_flow.sample_state(explicit_target_return, state_shape=(3, 64, 64))
    assert generated_images.shape == (2, 3, 64, 64), "co-flow generator failed to yield the targeted RGB visual synthesis shapes"
