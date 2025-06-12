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


def test_mean_flow():
    from einx import add
    from rectified_flow_pytorch.mean_flow import MeanFlow

    class Unet(Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Conv2d(3, 3, 1)

        def forward(self, x, t, r):
            return add('b ..., b, b', self.net(x), t, r)

    model = Unet()

    mean_flow = MeanFlow(model)
    data = torch.randn(16, 3, 16, 16)

    loss = mean_flow(data)
    loss.backward()

    sampled = mean_flow.sample(batch_size = 16)
    assert sampled.shape == data.shape
