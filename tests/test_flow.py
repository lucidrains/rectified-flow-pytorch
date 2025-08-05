import torch
from torch import nn
from torch.nn import Module

import pytest

def test_nano_flow():
    from rectified_flow_pytorch.nano_flow import NanoFlow

    model = torch.nn.Conv2d(3, 3, 1)

    nano_flow = NanoFlow(model)
    data = torch.randn(16, 3, 16, 16)

    loss = nano_flow(data)
    loss.backward()

    sampled = nano_flow.sample(batch_size = 16)
    assert sampled.shape == data.shape


@pytest.mark.parametrize('add_recon_loss', (False, True))
@pytest.mark.parametrize('accept_cond', (False, True))
@pytest.mark.parametrize('use_adaptive_loss_weight', (False, True))
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


@pytest.mark.parametrize('add_recon_loss', (False, True))
@pytest.mark.parametrize('accept_cond', (False, True))
@pytest.mark.parametrize('use_adaptive_loss_weight', (False, True))
@pytest.mark.parametrize('prob_default_flow_obj', (0.0, 0.5, 1.0))  # different boundary condition probabilities
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
