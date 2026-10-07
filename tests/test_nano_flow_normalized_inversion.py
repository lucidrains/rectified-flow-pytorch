import pytest
import torch
from torch import nn

from rectified_flow_pytorch.nano_flow import NanoFlow


class ConstantVelocity(nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.velocity = nn.Parameter(velocity.clone())

    def forward(self, state, times=None):
        return self.velocity.expand_as(state)


class CleanEndpoint(nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.velocity = nn.Parameter(velocity.clone())

    def forward(self, state, times):
        return state + (1 - times[:, None]) * self.velocity


def affine_flow(dtype, *, predict_clean=False):
    velocity = torch.tensor([0.375, -0.25], dtype=dtype)
    noise = torch.tensor([[-0.75, 0.125], [0.5, -0.5], [1.0, 0.75]], dtype=dtype)
    scale = torch.tensor([2.5, 0.4], dtype=dtype)
    center = torch.tensor([-1.5, 0.75], dtype=dtype)
    model = CleanEndpoint(velocity) if predict_clean else ConstantVelocity(velocity)
    flow = NanoFlow(
        model,
        times_cond_kwarg="times",
        data_shape=(2,),
        normalize_data_fn=lambda data: (data - center) / scale,
        unnormalize_data_fn=lambda state: state * scale + center,
        predict_clean=predict_clean,
    )
    image = (noise + velocity) * scale + center
    return flow, noise, image


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("steps", (1, 2, 8))
@pytest.mark.parametrize("fixed_point_steps", (1, 5))
def test_reverse_affine_normalization_recovers_analytical_latent(
    dtype, steps, fixed_point_steps
):
    flow, noise, image = affine_flow(dtype)
    actual = flow.sample(
        image=image,
        batch_size=len(image),
        steps=steps,
        reverse=True,
        reverse_fixed_point_steps=fixed_point_steps,
    )
    torch.testing.assert_close(actual, noise, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("steps", (1, 2, 8))
def test_forward_affine_output_stays_in_data_coordinates(dtype, steps):
    flow, noise, image = affine_flow(dtype)
    actual, initial_noise = flow.sample(
        noise=noise, batch_size=len(noise), steps=steps, return_noise=True
    )
    torch.testing.assert_close(actual, image, rtol=1e-5, atol=1e-6)
    assert initial_noise is noise


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("steps", (1, 4))
def test_nonlinear_data_transform_is_inverted_only_at_image_boundary(dtype, steps):
    velocity = torch.tensor([0.25, -0.375], dtype=dtype)
    noise = torch.tensor([[-0.5, 0.75], [0.875, -0.25]], dtype=dtype)
    flow = NanoFlow(
        ConstantVelocity(velocity),
        data_shape=(2,),
        normalize_data_fn=torch.log,
        unnormalize_data_fn=torch.exp,
    )
    image = torch.exp(noise + velocity)
    actual = flow.sample(
        image=image,
        batch_size=len(image),
        steps=steps,
        reverse=True,
        reverse_fixed_point_steps=5,
    )
    torch.testing.assert_close(actual, noise, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_reverse_return_noise_preserves_original_input_image(dtype):
    flow, noise, image = affine_flow(dtype)
    original_image = image.clone()
    actual, initial_image = flow.sample(
        image=image, batch_size=len(image), reverse=True, return_noise=True
    )
    torch.testing.assert_close(actual, noise, rtol=1e-5, atol=1e-6)
    assert initial_image is image
    torch.testing.assert_close(image, original_image, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("steps", (1, 8))
def test_identity_normalization_keeps_analytical_inverse(dtype, steps):
    velocity = torch.tensor([0.375, -0.25], dtype=dtype)
    noise = torch.tensor([[-0.75, 0.125], [0.5, -0.5]], dtype=dtype)
    flow = NanoFlow(ConstantVelocity(velocity), data_shape=(2,))
    actual = flow.sample(
        image=noise + velocity, batch_size=len(noise), steps=steps, reverse=True
    )
    torch.testing.assert_close(actual, noise, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("steps", (1, 2, 8))
def test_clean_prediction_uses_the_same_internal_inverse_coordinates(dtype, steps):
    flow, noise, image = affine_flow(dtype, predict_clean=True)
    actual = flow.sample(
        image=image,
        batch_size=len(image),
        steps=steps,
        reverse=True,
        reverse_fixed_point_steps=5,
    )
    torch.testing.assert_close(actual, noise, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_training_value_and_gradient_use_internal_flow_coordinates(dtype):
    target_velocity = torch.tensor([0.375, -0.25], dtype=dtype)
    residual = torch.tensor([0.1, -0.2], dtype=dtype)
    noise = torch.tensor([[-0.75, 0.125], [0.5, -0.5]], dtype=dtype)
    scale = torch.tensor([2.5, 0.4], dtype=dtype)
    center = torch.tensor([-1.5, 0.75], dtype=dtype)
    model = ConstantVelocity(target_velocity + residual)
    flow = NanoFlow(
        model,
        data_shape=(2,),
        normalize_data_fn=lambda data: (data - center) / scale,
        unnormalize_data_fn=lambda state: state * scale + center,
    )
    data = (noise + target_velocity) * scale + center
    loss = flow(data, noise=noise, times=torch.tensor([0.3, 0.8], dtype=dtype))
    torch.testing.assert_close(loss, residual.square().mean(), rtol=1e-5, atol=1e-6)
    loss.backward()
    torch.testing.assert_close(
        model.velocity.grad, 2 * residual / residual.numel(), rtol=1e-5, atol=1e-6
    )
