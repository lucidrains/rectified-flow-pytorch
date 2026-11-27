import pytest
import torch

from rectified_flow_pytorch.flow_optimizer import (
    FlowOptimizer,
    PopulationVelocityField,
    DimensionAgnosticVelocityField,
    improve_population,
    softmin
)

def quadratic_obj(shift):
    def fn(x):
        return (x - shift).pow(2).sum(dim = -1)
    return fn

def test_softmin_properties():
    vals = torch.tensor([[1.0, 2.0, 5.0, 10.0], [3.0, 3.0, 3.0, 3.0]])
    sm = softmin(vals, temp = 0.2, dim = -1)
    assert torch.all(sm >= vals.min(dim = -1).values - 1e-5)
    assert torch.all(sm <= vals.mean(dim = -1) + 1e-5)
    assert torch.allclose(sm[1], torch.tensor(3.0), atol = 1e-5)

def test_improve_population_monotone_and_bounded():
    torch.manual_seed(0)
    shift = torch.zeros(2, 1, 5)
    obj_fn = quadratic_obj(shift)
    pop = torch.randn(2, 12, 5) * 3.0 + 2.0

    bounds = (-2.0, 2.0)
    improved = improve_population(pop, obj_fn, gamma = 20, bounds = bounds)
    assert improved.shape == pop.shape
    assert improved.min() >= -2.0 - 1e-6 and improved.max() <= 2.0 + 1e-6
    assert obj_fn(improved).mean() <= obj_fn(pop).mean()

    # also works on unbatched (p, d)
    pop_unbatched = torch.randn(10, 4)
    imp_unbatched = improve_population(pop_unbatched, lambda x: x.pow(2).sum(dim = -1), gamma = 10)
    assert imp_unbatched.shape == pop_unbatched.shape

@pytest.mark.parametrize('dimension_agnostic', [False, True])
@pytest.mark.parametrize('use_grad', [True, False])
def test_permutation_equivariance(dimension_agnostic, use_grad):
    torch.manual_seed(42)
    batch, pop_size, dim_in = 2, 8, 6

    if dimension_agnostic:
        field = DimensionAgnosticVelocityField(dim = 32, depth = 1, heads = 4, use_grad = use_grad)
    else:
        field = PopulationVelocityField(dim_in = dim_in, dim = 32, depth = 1, heads = 4, use_grad = use_grad)
    field.eval()

    z = torch.randn(batch, pop_size, dim_in)
    times = torch.rand(batch)
    obj_vals = torch.rand(batch, pop_size)
    grads = torch.randn(batch, pop_size, dim_in) if use_grad else None

    perm = torch.randperm(pop_size)

    with torch.no_grad():
        v = field(z, times, obj_vals, grads = grads)
        v_perm = field(
            z[:, perm],
            times,
            obj_vals[:, perm],
            grads = grads[:, perm] if use_grad else None
        )

    assert torch.allclose(v[:, perm], v_perm, atol = 1e-5)

def test_dimension_agnostic_generalization():
    torch.manual_seed(7)
    opt = FlowOptimizer(
        dimension_agnostic = True,
        dim = 32,
        depth = 1,
        heads = 4,
        num_unfold_iters = 4,
        num_shared_final_iters = 2,
        ode_steps = 3,
        low_rank_dim = 8
    )

    for dim_in, pop_size in [(5, 8), (15, 12), (40, 20)]:
        pop = torch.randn(2, pop_size, dim_in)
        obj_fn = quadratic_obj(torch.zeros(2, 1, dim_in))

        fm_loss = opt.flow_matching_loss(pop, obj_fn, gamma = 5)
        assert torch.isfinite(fm_loss)

        with torch.no_grad():
            final_pop, best_x, best_f, traj = opt.optimize(
                pop, obj_fn, num_iters = 6, return_trajectory = True, return_best = True
            )

        assert final_pop.shape == (2, pop_size, dim_in)
        assert best_x.shape == (2, dim_in)
        assert best_f.shape == (2,)
        assert len(traj) == 7

@pytest.mark.parametrize('use_grad', [True, False])
@pytest.mark.parametrize('use_base_sampler', [False, True])
def test_flow_optimizer_two_phase_training_and_modes(use_grad, use_base_sampler):
    torch.manual_seed(1)
    dim_in = 5
    bounds = (torch.full((dim_in,), -3.0), torch.full((dim_in,), 3.0))

    opt = FlowOptimizer(
        dim_in = dim_in,
        dim = 32,
        depth = 1,
        heads = 4,
        mlp_depth = 2,
        num_unfold_iters = 4,
        num_shared_final_iters = 2,
        ode_steps = 3,
        use_grad = use_grad,
        use_base_sampler = use_base_sampler,
        bounds = bounds
    )
    assert opt.num_unique_models == 3

    shift = torch.randn(3, 1, dim_in) * 0.5
    obj_fn = quadratic_obj(shift)
    pop = torch.randn(3, 10, dim_in)

    # Phase 1: flow matching pre-training
    fm_loss = opt.flow_matching_loss(pop, obj_fn, gamma = 10)
    fm_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in opt.model.parameters())
    opt.zero_grad()

    # Copy pretrained weights into unfolded stages
    opt.init_unfolded_from_base()
    for p0, p1 in zip(opt.velocity_fields[0].parameters(), opt.velocity_fields[1].parameters()):
        assert torch.equal(p0, p1)

    # Phase 2: progressive and full deep unfolding
    prog_loss = opt.unfold_loss(pop, obj_fn, active_model_idx = 1)
    prog_loss.backward()
    assert all(p.grad is None for p in opt.velocity_fields[0].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in opt.velocity_fields[1].parameters())
    opt.zero_grad()

    full_loss, traj = opt.unfold_loss(pop, obj_fn, return_trajectory = True)
    full_loss.backward()
    assert len(traj) == 5
    assert all(
        any(p.grad is not None and p.grad.abs().sum() > 0 for p in field.parameters())
        for field in opt.velocity_fields
    )

    # Inference inside torch.no_grad (must still compute grads for use_grad=True)
    with torch.no_grad():
        out = opt(pop, obj_fn, num_iters = 6)
    assert out.shape == pop.shape
    assert out.min() >= -3.0 - 1e-6 and out.max() <= 3.0 + 1e-6

def test_flow_optimizer_learns_quadratic():
    torch.manual_seed(123)
    dim_in = 4
    opt = FlowOptimizer(
        dim_in = dim_in,
        dim = 48,
        depth = 1,
        heads = 4,
        mlp_depth = 2,
        num_unfold_iters = 4,
        num_shared_final_iters = 2,
        ode_steps = 3,
        use_grad = True
    )
    adam = torch.optim.Adam(opt.parameters(), lr = 3e-3)

    # Quick pre-train
    for _ in range(25):
        adam.zero_grad()
        shift = torch.empty(16, 1, dim_in).uniform_(-2.0, 2.0)
        pop = torch.randn(16, 12, dim_in) * 2.0
        loss = opt.flow_matching_loss(pop, quadratic_obj(shift), gamma = 15)
        loss.backward()
        adam.step()

    opt.init_unfolded_from_base()

    # Quick fine-tune
    for _ in range(35):
        adam.zero_grad()
        shift = torch.empty(16, 1, dim_in).uniform_(-2.0, 2.0)
        pop = torch.randn(16, 12, dim_in) * 2.0
        loss = opt.unfold_loss(pop, quadratic_obj(shift))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt.parameters(), 1.0)
        adam.step()

    opt.eval()
    test_shift = torch.tensor([[[1.5, -1.2, 0.8, -1.4]]])
    test_obj = quadratic_obj(test_shift)
    test_pop = torch.randn(1, 12, dim_in) * 2.0
    init_best = test_obj(test_pop).min().item()
    with torch.no_grad():
        _, _, final_best = opt.optimize(test_pop, test_obj, return_best = True)
    assert final_best.item() < init_best * 0.1
