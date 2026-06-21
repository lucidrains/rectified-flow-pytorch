import torch
from torch.nn import Module
import torch.nn.functional as F

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def append_dims(t, dims):
    shape = t.shape
    ones = ((1,) * dims)
    return t.reshape(*shape, *ones)

class NanoFlow(Module):
    def __init__(
        self,
        model: Module,
        times_cond_kwarg = None,
        data_shape = None,
        normalize_data_fn = identity,
        unnormalize_data_fn = identity,
        predict_clean = False,
        max_timesteps = 100,
        loss_fn = F.mse_loss
    ):
        super().__init__()
        self.model = model
        self.times_cond_kwarg = times_cond_kwarg
        self.data_shape = data_shape

        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn

        self.predict_clean = predict_clean # predicting x0
        self.max_timesteps = max_timesteps

        self.loss_fn = loss_fn

    @torch.no_grad()
    def sample(
        self,
        steps = 16,
        batch_size = 1,
        data_shape = None,
        return_noise = False,
        noise = None,
        image = None,
        reverse = False,
        eps = 1e-5,
        **kwargs
    ):
        assert exists(image) == reverse
        assert not (exists(image) and exists(noise))

        assert 1 <= steps <= self.max_timesteps

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'
        device = next(self.model.parameters()).device

        init = image if reverse else default(noise, torch.randn((batch_size, *data_shape), device = device))

        times = torch.linspace(0., 1., steps + 1, device = device)

        if reverse:
            times = times.flip(0)

        times = times[:-1]

        delta = (1. / steps) * (-1. if reverse else 1.)

        state = init

        for time in times:
            time = time.expand(batch_size)
            time_kwarg = {self.times_cond_kwarg: time} if exists(self.times_cond_kwarg) else dict()

            model_output = self.model(state, **time_kwarg, **kwargs)

            if self.predict_clean:
                padded_time = append_dims(time, state.ndim - 1)
                pred_flow = (model_output - state) / (1. - padded_time).clamp(min = eps)
            else:
                pred_flow = model_output

            state = state + delta * pred_flow

        out = self.unnormalize_data_fn(state)

        if not return_noise:
            return out

        return out, init

    def forward(self, data, noise = None, times = None, loss_reduction = 'mean', **kwargs):
        data = self.normalize_data_fn(data)

        # shapes and variables

        shape, ndim = data.shape, data.ndim
        self.data_shape = default(self.data_shape, shape[1:]) # store last data shape for inference
        batch, device = shape[0], data.device

        # flow logic

        times = default(times, torch.rand(batch, device = device))
        times = times * (1. - self.max_timesteps ** -1)

        noise = default(noise, torch.randn_like(data))
        flow = data - noise # flow is the velocity from noise to data, also what the model is trained to predict

        padded_times = append_dims(times, ndim - 1)
        noised_data = noise.lerp(data, padded_times) # noise the data with random amounts of noise (time) - lerp is read as noise -> data from 0. to 1.

        time_kwarg = {self.times_cond_kwarg: times} if exists(self.times_cond_kwarg) else dict() # maybe time conditioning, could work without it (https://arxiv.org/abs/2502.13129v1)
        model_output = self.model(noised_data, **time_kwarg, **kwargs)

        if self.predict_clean:
            pred_flow = (model_output - noised_data) / (1. - padded_times)
        else:
            pred_flow = model_output

        return self.loss_fn(flow, pred_flow, reduction = loss_reduction)

# quick test

if __name__ == '__main__':
    model = torch.nn.Conv2d(3, 3, 1)

    nano_flow = NanoFlow(model)
    data = torch.randn(16, 3, 16, 16)

    loss = nano_flow(data)
    loss.backward()

    sampled = nano_flow.sample(batch_size = 16)
    assert sampled.shape == data.shape

    reversed_noise = nano_flow.sample(batch_size = 16, image = sampled, reverse = True)
    assert reversed_noise.shape == data.shape
