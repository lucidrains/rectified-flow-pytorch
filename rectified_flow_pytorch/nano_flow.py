import torch
from torch.nn import Module
import torch.nn.functional as F

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def append_dims(t, dims):
    shape = t.shape
    ones = ((1,) * dims)
    return t.reshape(*shape, *ones)

class NanoFlow(Module):
    def __init__(
        self,
        model: Module,
        times_cond_kwarg = None,
        data_shape = None
    ):
        super().__init__()
        self.model = model
        self.times_cond_kwarg = times_cond_kwarg
        self.data_shape = None

    @torch.no_grad()
    def sample(
        self,
        steps = 16,
        batch_size = 1,
        data_shape = None,
        **kwargs
    ):
        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'
        device = next(self.model.parameters()).device

        noise = torch.randn((batch_size, *self.data_shape), device = device)
        times = torch.linspace(0., 1., steps, device = device)
        delta = 1. / steps

        denoised = noise

        for time in times:
            time = time.expand(batch_size)
            time_kwarg = {self.times_cond_kwarg: time} if exists(self.times_cond_kwarg) else dict()

            pred_flow = self.model(noise, **time_kwarg, **kwargs)
            denoised = denoised + delta * pred_flow

        return denoised

    def forward(self, data, **kwargs):
        # shapes and variables

        shape, ndim = data.shape, data.ndim
        self.data_shape = default(self.data_shape, shape[1:]) # store last data shape for inference
        batch, device = shape[0], data.device

        # flow logic

        times = torch.rand(batch, device = device)
        noise = torch.randn_like(data)
        flow = data - noise # flow is the velocity from noise to data, also what the model is trained to predict

        padded_times = append_dims(times, ndim - 1)
        noised_data = noise * padded_times + data * (1. - padded_times) # noise the data with random amounts of noise (time)

        time_kwarg = {self.times_cond_kwarg: times} if exists(self.times_cond_kwarg) else dict() # maybe time conditioning, could work without it
        pred_flow = self.model(noised_data, **time_kwarg, **kwargs)

        return F.mse_loss(flow, pred_flow)
