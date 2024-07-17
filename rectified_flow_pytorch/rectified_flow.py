import torch
from torch.nn import Module

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class RectifiedFlow(Module):
    def __init__(
        self,
        model: Module
    ):
        super().__init__()
        self.model = model

    def sample(self):
        raise NotImplementedError

    def forward(
        self,
        data
    ):
        return data
