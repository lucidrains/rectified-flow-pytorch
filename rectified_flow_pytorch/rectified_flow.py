import torch
from torch.nn import Module
import torch.nn.functional as F

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

# main class

class RectifiedFlow(Module):
    def __init__(
        self,
        model: Module,
        time_cond_kwarg = 'times'
    ):
        super().__init__()
        self.model = model
        self.time_cond_kwarg = time_cond_kwarg # whether the model is to be conditioned on the times

    def sample(self):
        raise NotImplementedError

    def forward(
        self,
        data
    ):
        batch, device = data.shape[0], data.device

        # x0 - gaussian noise, x1 - data

        noise = torch.randn_like(data)

        # times, and times with dimension padding on right

        times = torch.rand(batch, device = device)
        padded_times = append_dims(times, data.ndim - 1)

        # Algorithm 2 in paper
        # linear interpolation of noise with data using random times
        # x1 * t + x0 * (1 - t) - so from noise (time = 0) to data (time = 1.)

        noised = padded_times * data + (1. - padded_times) * noise

        # prepare maybe time conditioning for model

        model_kwargs = dict()
        time_kwarg = self.time_cond_kwarg

        if exists(time_kwarg):
            model_kwargs.update(**{time_kwarg: times})

        # the model predicts the flow from the noised

        flow = data - noise
        pred_flow = self.model(noised, **model_kwargs)

        loss = F.mse_loss(pred_flow, flow)

        return loss
