import torch
from torch.nn import Module
import torch.nn.functional as F
from einops import reduce, rearrange, einsum
import einx


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


class UAFlow(Module):
    def __init__(
        self,
        model: Module,
        times_cond_kwarg = None,
        data_shape = None,
        normalize_data_fn = identity,
        unnormalize_data_fn = identity,
        max_timesteps = 100,
        ucg_scale = 0.0, # Uncertainty-Aware Classifier Guidance scale (w)
    ):
        super().__init__()
        self.model = model
        self.times_cond_kwarg = times_cond_kwarg
        self.data_shape = data_shape

        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn
        self.max_timesteps = max_timesteps
        
        self.ucg_scale = ucg_scale

    @torch.no_grad()
    def sample(
        self,
        steps = 16,
        batch_size = 1,
        data_shape = None,
        **kwargs
    ):
        assert 1 <= steps <= self.max_timesteps

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'
        device = next(self.model.parameters()).device

        noise = torch.randn((batch_size, *data_shape), device = device)
        # TODO - implement variance propagation 
        state_variance = torch.zeros_like(noise)

        times = torch.linspace(0., 1., steps + 1, device = device)[:-1]
        delta = 1. / steps

        denoised = noise

        for time in times:
            time_b = time.expand(batch_size)
            time_kwarg = {self.times_cond_kwarg: time_b} if exists(self.times_cond_kwarg) else dict()

            def wrapped_model(x_in):
                return self.model(x_in, **time_kwarg, **kwargs)

            with torch.set_grad_enabled(self.ucg_scale > 0):
                x_in = denoised.detach()
                if self.ucg_scale > 0:
                    x_in.requires_grad_(True)
                
                pred_mean, pred_log_var = wrapped_model(x_in)
                pred_var = torch.exp(pred_log_var)
                
                # Apply Uncertainty-Aware Classifier Guidance (U-CG)
                
                if self.ucg_scale > 0:
                    # get gradient of uncertainty w.r.t. input
                    f_unc = - (reduce(pred_var, 'b c h w -> b', 'mean')) ** 2
                    grad_x = torch.autograd.grad(f_unc.sum(), x_in)[0]

                    b_t = (1.0 - time) / (time + 1e-5)
                    
                    velocity = pred_mean + b_t * self.ucg_scale * grad_x
                else:
                    velocity = pred_mean

            denoised = denoised + delta * velocity

        return denoised

            
    def forward(self, data, noise=None, times=None, **kwargs):
        data = self.normalize_data_fn(data)

        shape, ndim = data.shape, data.ndim
        self.data_shape = default(self.data_shape, shape[1:]) 
        batch, device = shape[0], data.device

        times = default(times, torch.rand(batch, device=device))
        times = times * (1. - self.max_timesteps ** -1)

        noise = default(noise, torch.randn_like(data))
        padded_times = append_dims(times, ndim - 1)
        noised_data = noise.lerp(data, padded_times) 

        time_kwarg = {self.times_cond_kwarg: times} if exists(self.times_cond_kwarg) else dict() 
        
        pred_mean, pred_log_var = self.model(noised_data, **time_kwarg, **kwargs)

        # uncertainty-aware loss computation

        padded_times = rearrange(times, 'b -> b 1 1 1 1')

        exact_mean = einx.multiply("b, j c h w -> b j c h w", times, data)
        exact_variance = (1.0 - padded_times) ** 2 + 1e-8
        
        squared_err = (einx.subtract("b c h w, b j c h w -> b j c h w", noised_data, exact_mean) ** 2) / (2 * exact_variance)
        dist = reduce(squared_err, 'b j c h w -> b j', 'sum')

        weights = torch.softmax(-dist, dim=1)

        ut_cond_all = einx.subtract("j c h w, b c h w -> b j c h w", data, noised_data) / (1.0 - padded_times + 1e-8)

        u_hat = einsum(ut_cond_all, weights, "b j c h w, b j -> b c h w",)

        flow_cond = data - noise 

        correction = (u_hat ** 2 - flow_cond ** 2).detach()

        pred_var = torch.exp(pred_log_var)

        loss = ((pred_mean - flow_cond) ** 2) / (2 * pred_var) +  0.5 *  pred_log_var + (correction / (2 * pred_var))

        loss = pred_var.detach() * loss

        return loss.mean()

if __name__ == '__main__':
    
    class DummyUAModel(Module):
        def __init__(self, in_channels):
            super().__init__()
 
            self.net = torch.nn.Conv2d(in_channels, in_channels * 2, 1)

        def forward(self, x, **kwargs):
            out = self.net(x)
        
            mean, log_var = out.chunk(2, dim=1)
            return mean, log_var

    batch_size = 4
    channels = 3
    img_size = 16
    
    dummy_model = DummyUAModel(in_channels=channels)

    ua_flow = UAFlow(
        model=dummy_model, 
        ucg_scale=10.0, 
    )
    
    data = torch.randn(batch_size, channels, img_size, img_size)

    loss = ua_flow(data)

    loss.backward()
 
    denoised = ua_flow.sample(
        batch_size=batch_size, 
        steps=5,
    )