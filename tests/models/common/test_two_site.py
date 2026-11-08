"""Two-site mechanics (ADR-0075 clause 3): projection parity with the loader,
quad kinematics, the J2 radial return and its by-construction properties."""

import math

import numpy as np
import pytest
import torch

from structbench.datasets.canonical import _cell_weights, _elements_to_nodes
from structbench.models.common.two_site import (
    LAW_DECODERS,
    RETURN_MAP_WIDTH,
    TWO_SITE_DECODERS,
    LawRows,
    admissible_decode,
    admissible_parts,
    cell_weights,
    corner_mean,
    deviator_from_direction,
    law_rows,
    project_elements_to_nodes,
    project_trial,
    project_trial_parts,
    quad_increment,
    radial_return,
    return_map_decode,
    sigma_y_rows,
    split_stress,
    von_mises4,
)

E_MOD, NU = 124_000.0, 0.34  # MPa, a copper-like elastic pair
SY0, H = 200.0, 1_000.0  # MPa; linear hardening, second knot far out


def _law(n: int, sy0: float = SY0, h: float = H) -> LawRows:
    return law_rows(
        torch.full((n,), E_MOD, dtype=torch.float64),
        torch.full((n,), NU, dtype=torch.float64),
        torch.tensor([[0.0, 10.0]] * n, dtype=torch.float64),
        torch.tensor([[sy0, sy0 + 10.0 * h]] * n, dtype=torch.float64),
    )


def _grid(nx: int, ny: int, dtype=torch.float64):
    """A unit-spaced structured quad mesh, corners counter-clockwise."""
    xs, ys = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float) + 0.5
    cells = []
    for i in range(nx):
        for j in range(ny):
            n0 = i * (ny + 1) + j
            cells.append([n0, n0 + (ny + 1), n0 + (ny + 1) + 1, n0 + 1])
    return torch.tensor(coords, dtype=dtype), torch.tensor(cells, dtype=torch.long)


def test_projection_matches_the_loader():
    coords, cells = _grid(3, 2)
    rng = np.random.default_rng(1)
    values = rng.normal(size=(cells.shape[0], 3))
    w_np = _cell_weights(coords.numpy(), cells.numpy())
    expected = _elements_to_nodes(values[None], cells.numpy(), w_np, coords.shape[0])[0]
    w = cell_weights(coords, cells)
    got = project_elements_to_nodes(torch.tensor(values), cells, w, coords.shape[0])
    assert torch.allclose(got, torch.tensor(expected))
    # A node outside every cell reads 0 (the loader's rule).
    got2 = project_elements_to_nodes(
        torch.tensor(values), cells, w, coords.shape[0] + 1
    )
    assert torch.equal(got2[-1], torch.zeros(3, dtype=torch.float64))


def test_corner_mean_is_the_mean_of_corner_rows():
    coords, cells = _grid(2, 1)
    node_values = torch.arange(coords.shape[0], dtype=torch.float64)[:, None]
    assert torch.allclose(corner_mean(node_values, cells), node_values[cells].mean(1))


def test_affine_increment_gives_its_symmetric_and_skew_parts():
    coords, cells = _grid(2, 2)
    a = torch.tensor([[1e-3, 4e-4], [-2e-4, -5e-4]], dtype=torch.float64)
    x_query = coords + coords @ a.T  # du = A x
    deps, rot = quad_increment(coords, x_query, cells, "plane_strain")
    sym = 0.5 * (a + a.T)
    assert torch.allclose(deps[:, :2, :2], sym.expand_as(deps[:, :2, :2]), atol=1e-6)
    assert torch.allclose(
        deps[:, 2, 2], torch.zeros(cells.shape[0], dtype=torch.float64)
    )
    skew = 0.5 * (a - a.T)
    assert torch.allclose(
        rot, (torch.eye(2, dtype=torch.float64) + skew).expand_as(rot), atol=1e-6
    )


def test_axisymmetric_hoop_strain_is_radial_displacement_over_radius():
    coords, cells = _grid(2, 2)  # first coordinate is the radius (>= 0.5)
    x_query = coords.clone()
    x_query[:, 0] *= 1.001  # u_r = 1e-3 r -> hoop strain 1e-3
    deps, _ = quad_increment(coords, x_query, cells, "axisymmetric")
    # Mid-configuration increments are the second-order (logarithmic) strain:
    # 1e-3 / (1 + 5e-4) against ln(1.001), equal to 1e-7 relative.
    log_strain = torch.full((cells.shape[0],), math.log(1.001), dtype=torch.float64)
    assert torch.allclose(deps[:, 2, 2], log_strain, rtol=1e-5)
    assert torch.allclose(deps[:, 0, 0], log_strain, rtol=1e-5)


def test_rigid_rotation_has_no_strain_and_rotates_the_stress():
    coords, cells = _grid(2, 2)
    theta = 0.01
    c, s = math.cos(theta), math.sin(theta)
    r = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    x_query = coords @ r.T
    deps, rot = quad_increment(coords, x_query, cells, "plane_strain")
    assert torch.allclose(deps, torch.zeros_like(deps), atol=1e-12)
    assert torch.allclose(rot, r.expand_as(rot), atol=1e-6)
    n = cells.shape[0]
    stress = torch.tensor([[50.0, -20.0, 10.0, 15.0]] * n, dtype=torch.float64)
    peeq = torch.full((n,), 0.02, dtype=torch.float64)
    new_stress, new_peeq, dg = radial_return(stress, peeq, deps, rot, _law(n))
    sig = torch.tensor([[50.0, 15.0], [15.0, -20.0]], dtype=torch.float64)
    rotated = r @ sig @ r.T
    assert torch.allclose(new_stress[:, 0], rotated[0, 0].expand(n), atol=1e-6)
    assert torch.allclose(new_stress[:, 1], rotated[1, 1].expand(n), atol=1e-6)
    assert torch.allclose(new_stress[:, 3], rotated[0, 1].expand(n), atol=1e-6)
    assert torch.allclose(new_stress[:, 2], stress[:, 2])
    assert torch.equal(new_peeq, peeq) and torch.equal(dg, torch.zeros_like(dg))


def test_uniaxial_strain_is_elastic_below_yield_and_hardens_above():
    coords, cells = _grid(1, 1)
    n = 1
    law = _law(n)
    g, k = float(law.shear[0]), float(law.bulk[0])
    zero_state = torch.zeros((n, RETURN_MAP_WIDTH), dtype=torch.float64)

    def step(eps: float):
        x_query = coords.clone()
        x_query[:, 0] = coords[:, 0] * (1.0 + eps)
        deps, rot = quad_increment(coords, x_query, cells, "plane_strain")
        return radial_return(zero_state[:, :4], zero_state[:, 4], deps, rot, law)

    # Elastic: uniaxial strain e_xx with e_yy = e_zz = 0. The mid-
    # configuration gradient reads the stretch as eps / (1 + eps / 2).
    eps = 5e-4
    eps_eff = eps / (1.0 + 0.5 * eps)
    stress, peeq, dg = step(eps)
    t = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
    assert torch.allclose(stress[0, 0], t((k + 4.0 * g / 3.0) * eps_eff), rtol=1e-8)
    assert torch.allclose(stress[0, 1], t((k - 2.0 * g / 3.0) * eps_eff), rtol=1e-8)
    assert float(peeq) == 0.0 and float(dg) == 0.0
    assert float(von_mises4(stress)) < SY0
    # Plastic: the trial exceeds sigma_y0; the return lands on the hardened
    # surface with the closed-form multiplier of linear hardening.
    eps = 2e-2
    eps_eff = eps / (1.0 + 0.5 * eps)
    stress, peeq, dg = step(eps)
    q_trial = 2.0 * g * eps_eff  # dev strain diag (2/3, -1/3, -1/3) eps
    expected_dg = (q_trial - SY0) / (3.0 * g + H)
    assert torch.allclose(dg[0], t(expected_dg), rtol=1e-8)
    assert torch.allclose(peeq[0], t(expected_dg), rtol=1e-8)
    assert torch.allclose(von_mises4(stress)[0], t(SY0 + H * expected_dg), rtol=1e-8)
    # The mean stress is the elastic one: the return is deviatoric.
    assert torch.allclose(stress[0, :3].sum() / 3.0, t(k * eps_eff), rtol=1e-8)


def test_sigma_y_rows_follows_np_interp_with_zero_slope_outside():
    knots_x = torch.tensor([[0.0, 0.1, 0.5]], dtype=torch.float64).expand(5, 3)
    knots_y = torch.tensor([[100.0, 150.0, 170.0]], dtype=torch.float64).expand(5, 3)
    peeq = torch.tensor([-0.1, 0.0, 0.05, 0.3, 0.9], dtype=torch.float64)
    sy, slope = sigma_y_rows(peeq, knots_x, knots_y)
    expected = np.interp(peeq.numpy(), knots_x[0].numpy(), knots_y[0].numpy())
    assert np.allclose(sy.numpy(), expected)
    assert np.allclose(slope.numpy(), [0.0, 500.0, 500.0, 50.0, 0.0])


def test_random_states_stay_admissible_after_the_return():
    coords, cells = _grid(4, 3)
    n = cells.shape[0]
    g = torch.Generator().manual_seed(3)
    stress = torch.randn((n, 4), generator=g, dtype=torch.float64) * 300.0
    peeq = torch.rand((n,), generator=g, dtype=torch.float64) * 0.5
    x_query = (
        coords + torch.randn(coords.shape, generator=g, dtype=torch.float64) * 0.02
    )
    state = torch.cat([stress, peeq[:, None]], dim=-1)
    law = _law(n)
    out = return_map_decode(state, coords, x_query, cells, law, "axisymmetric")
    vm = von_mises4(out[:, :4])
    sy, _ = sigma_y_rows(out[:, 4], law.knots_peeq, law.knots_sy)
    assert bool((vm <= sy * (1.0 + 1e-9) + 1e-9).all())
    assert bool((out[:, 4] >= peeq).all())
    # A random trial is almost surely outside: some rows yielded.
    assert bool((out[:, 4] > peeq).any())


def test_gradients_reach_the_query_positions_and_the_carried_state():
    coords, cells = _grid(2, 2)
    n = cells.shape[0]
    state = torch.zeros((n, RETURN_MAP_WIDTH), dtype=torch.float64, requires_grad=True)
    x_query = (coords * 1.01).clone().requires_grad_(True)
    out = return_map_decode(state, coords, x_query, cells, _law(n), "axisymmetric")
    out.square().sum().backward()
    assert x_query.grad is not None and torch.isfinite(x_query.grad).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert float(x_query.grad.abs().sum()) > 0


def test_decode_rejects_a_wrong_channel_width():
    coords, cells = _grid(1, 1)
    with pytest.raises(ValueError, match="5 state channels"):
        return_map_decode(
            torch.zeros((1, 4), dtype=torch.float64),
            coords,
            coords,
            cells,
            _law(1),
            "plane_strain",
        )
    with pytest.raises(ValueError, match="formulation"):
        quad_increment(coords, coords, cells, "shell")


# --- The admissible arm in two forms (ADR-0075 clause 3 note, 2026-11-08) ---


def _exact_trace(dev: torch.Tensor) -> torch.Tensor:
    """The deviator trace summed in the layout's order: exactly zero means
    zero in floating point, not within a tolerance."""
    return (dev[:, 0] + dev[:, 1]) + dev[:, 2]


def _random_law_rows(n: int, g: torch.Generator) -> LawRows:
    """Per-row laws with a three-knot concave hardening table."""
    sy0 = 150.0 + 200.0 * torch.rand((n,), generator=g, dtype=torch.float64)
    knots_x = torch.tensor([[0.0, 0.2, 10.0]] * n, dtype=torch.float64)
    knots_y = torch.stack([sy0, sy0 + 100.0, sy0 + 300.0], dim=-1)
    return law_rows(
        torch.full((n,), E_MOD, dtype=torch.float64),
        torch.full((n,), NU, dtype=torch.float64),
        knots_x,
        knots_y,
    )


def test_decoder_vocabularies_name_the_two_admissible_forms():
    assert set(LAW_DECODERS) <= set(TWO_SITE_DECODERS)
    assert {"admissible_coords", "projected_trial"} <= set(LAW_DECODERS)
    assert "corner_mean" not in LAW_DECODERS


def test_deviator_from_direction_has_an_exactly_zero_trace():
    g = torch.Generator().manual_seed(11)
    v = torch.randn((500, 3), generator=g, dtype=torch.float64) * 1e3
    dev = deviator_from_direction(v)
    assert bool((_exact_trace(dev) == 0.0).all())
    dev32 = deviator_from_direction(v.float())
    assert bool((_exact_trace(dev32) == 0.0).all())
    # split_stress recovers a deviator with the same exactness and the mean.
    stress = torch.randn((500, 4), generator=g, dtype=torch.float64) * 1e3
    d, p = split_stress(stress)
    assert bool((_exact_trace(d) == 0.0).all())
    assert torch.allclose(p, stress[:, :3].mean(-1))


@pytest.mark.parametrize("scale", [0.0, 1.0, 50.0])
def test_admissible_coordinates_hold_the_invariants_for_any_head(scale):
    """For ANY head value (zero, order one, saturating): peeq never below the
    carried anchor, von Mises never above sigma_y at the updated peeq, the
    deviator trace exactly zero, the mean stress the free channel."""
    n = 64
    g = torch.Generator().manual_seed(5)
    head = torch.randn((n, 5), generator=g, dtype=torch.float64) * scale
    anchor = torch.rand((n,), generator=g, dtype=torch.float64) * 0.4
    law = _random_law_rows(n, g)
    loc, sc = torch.tensor(-25.0, dtype=torch.float64), torch.tensor(180.0)
    dev, p, peeq = admissible_parts(head, anchor, law, loc, sc)
    assert dev.dtype == torch.float64 and peeq.dtype == torch.float64
    assert bool((peeq >= anchor).all())
    assert torch.allclose(peeq - anchor, torch.nn.functional.softplus(head[:, 3]))
    sy, _ = sigma_y_rows(peeq, law.knots_peeq, law.knots_sy)
    assert bool((von_mises4(dev) <= sy * (1.0 + 1e-12)).all())
    assert bool((_exact_trace(dev) == 0.0).all())
    assert torch.allclose(p, loc + sc.double() * head[:, 4])
    # The assembled five-channel state carries the same quantities.
    state = admissible_decode(head, anchor, law, loc, sc)
    assert state.shape == (n, RETURN_MAP_WIDTH)
    d2, p2 = split_stress(state[:, :4])
    assert torch.allclose(d2, dev, atol=1e-9) and torch.allclose(p2, p, atol=1e-9)
    assert torch.equal(state[:, 4], peeq)
    if scale == 0.0:
        assert bool((dev == 0.0).all())  # a zero direction is a zero deviator
    if scale == 50.0:
        # Saturated radius: the state sits on the surface (tanh -> 1).
        assert torch.allclose(von_mises4(dev), sy, rtol=1e-9)


def test_admissible_coordinates_are_differentiable_in_every_channel():
    n = 6
    g = torch.Generator().manual_seed(6)
    head = (torch.randn((n, 5), generator=g, dtype=torch.float64)).requires_grad_(True)
    anchor = torch.zeros((n,), dtype=torch.float64)
    state = admissible_decode(head, anchor, _law(n), 0.0, 100.0)
    state.square().sum().backward()
    assert head.grad is not None and torch.isfinite(head.grad).all()
    assert bool((head.grad.abs().sum(0) > 0).all())  # all five channels reached
    # And finite at the zero point of the direction.
    head0 = torch.zeros((n, 5), dtype=torch.float64, requires_grad=True)
    admissible_decode(head0, anchor, _law(n), 0.0, 100.0).sum().backward()
    assert torch.isfinite(head0.grad).all()


def test_projected_trial_holds_the_invariants_for_any_trial():
    """For ANY trial (stress far outside, peeq below the anchor): peeq never
    below the anchor nor the trial, von Mises never above sigma_y at the
    updated peeq, the deviator trace exactly zero, the mean stress untouched;
    and the identity on an admissible trial."""
    n = 64
    g = torch.Generator().manual_seed(7)
    trial = torch.cat(
        [
            torch.randn((n, 4), generator=g, dtype=torch.float64) * 600.0,
            torch.rand((n, 1), generator=g, dtype=torch.float64) * 1.2 - 0.4,
        ],
        dim=-1,
    )
    anchor = torch.rand((n,), generator=g, dtype=torch.float64) * 0.4
    law = _random_law_rows(n, g)
    dev, p, peeq = project_trial_parts(trial, anchor, law)
    assert bool((peeq >= anchor).all()) and bool((peeq >= trial[:, 4]).all())
    sy, _ = sigma_y_rows(peeq, law.knots_peeq, law.knots_sy)
    assert bool((von_mises4(dev) <= sy * (1.0 + 1e-12)).all())
    assert bool((_exact_trace(dev) == 0.0).all())
    assert torch.allclose(p, trial[:, :3].mean(-1))
    # Some rows yielded (a random trial of this size is almost surely outside).
    assert bool((peeq > torch.maximum(anchor, trial[:, 4])).any())
    # A plastic row sits ON the surface; an elastic one keeps its deviator.
    plastic = peeq > torch.maximum(anchor, trial[:, 4])
    assert torch.allclose(von_mises4(dev)[plastic], sy[plastic], rtol=1e-9)
    d_trial, _ = split_stress(trial[:, :4])
    assert torch.allclose(dev[~plastic], d_trial[~plastic], atol=1e-9)
    # Identity on an admissible trial.
    inside = torch.cat(
        [torch.randn((n, 4), generator=g, dtype=torch.float64) * 20.0, anchor[:, None]],
        dim=-1,
    )
    inside[:, 4] += 0.05
    out = project_trial(inside, anchor, law)
    assert torch.allclose(out, inside, atol=1e-9)


def test_projected_trial_is_non_expansive_in_the_law_metric():
    """The projection's Jacobian has operator norm at most one in the metric
    ``|dev|_F^2 / (2 G) + H peeq^2`` (the generalised-stress metric of linear
    hardening, in which the radial return is a closest-point projection), on
    random inputs that cover the elastic and the plastic branch; an elastic
    row's Jacobian is the identity, a plastic row's annihilates the normal to
    the yield surface in that space (one zero singular value) and contracts
    the tangential directions. In raw units the norm is NOT bounded by one:
    the hardening couples the stress to the trial plastic strain at ``~H``
    MPa per unit, which is the law, not a defect."""
    n = 12
    g = torch.Generator().manual_seed(8)
    trial = torch.cat(
        [
            torch.randn((n, 4), generator=g, dtype=torch.float64) * 400.0,
            torch.rand((n, 1), generator=g, dtype=torch.float64) - 0.2,
        ],
        dim=-1,
    )
    trial[0, :4] *= 0.01  # an elastic row for sure
    anchor = torch.rand((n,), generator=g, dtype=torch.float64) * 0.3
    law = _law(n)
    shear = float(law.shear[0])

    def f(x: torch.Tensor) -> torch.Tensor:
        return project_trial(x.view(n, 5), anchor, law).reshape(-1)

    jac = torch.autograd.functional.jacobian(f, trial.reshape(-1).clone())
    # Frobenius weights on (s11, s22, s33, s12): the shear enters twice.
    d_row = torch.tensor([1.0, 1.0, 1.0, math.sqrt(2.0), 0.0]) / math.sqrt(2 * shear)
    d_row[4] = math.sqrt(H)
    metric = torch.block_diag(*[torch.diag(d_row.double())] * n)
    jac_m = metric @ jac @ torch.linalg.inv(metric)
    assert float(torch.linalg.matrix_norm(jac_m, ord=2)) <= 1.0 + 1e-9
    _, _, peeq = project_trial_parts(trial, anchor, law)
    plastic = peeq > torch.maximum(anchor, trial[:, 4])
    assert bool(plastic.any()) and not bool(plastic.all())
    for i in range(n):
        blk = jac_m[5 * i : 5 * i + 5, 5 * i : 5 * i + 5]
        assert float(torch.linalg.matrix_norm(blk, ord=2)) <= 1.0 + 1e-9
        if plastic[i]:
            sv = torch.linalg.svdvals(blk)
            assert float(sv.min()) < 1e-8  # the surface normal is annihilated
            assert float(sv.max()) <= 1.0 + 1e-9
        else:
            assert torch.allclose(blk, torch.eye(5, dtype=blk.dtype), atol=1e-9)
    # The raw-units norm exceeds one through the hardening coupling.
    assert float(torch.linalg.matrix_norm(jac, ord=2)) > 1.0


def test_projected_trial_gradients_reach_the_trial_and_not_the_anchor_through_noise():
    """Gradients reach every trial channel; the anchor plastic strain enters
    only through the clamp (the lower bound), so a trial above the anchor is
    insensitive to it: noise on the anchor cannot ratchet the output."""
    n = 5
    trial = torch.tensor(
        [[300.0, -100.0, 50.0, 80.0, 0.3]] * n, dtype=torch.float64, requires_grad=True
    )
    anchor = torch.full((n,), 0.1, dtype=torch.float64, requires_grad=True)
    out = project_trial(trial, anchor, _law(n))
    out.square().sum().backward()
    assert trial.grad is not None and torch.isfinite(trial.grad).all()
    assert bool((trial.grad.abs() > 0).all())
    assert anchor.grad is not None and bool((anchor.grad == 0.0).all())


@pytest.mark.parametrize("fn", ["admissible", "projected"])
def test_admissible_forms_reject_a_wrong_channel_width(fn):
    n = 3
    bad = torch.zeros((n, 4), dtype=torch.float64)
    anchor = torch.zeros((n,), dtype=torch.float64)
    with pytest.raises(ValueError, match="5"):
        if fn == "admissible":
            admissible_decode(bad, anchor, _law(n), 0.0, 1.0)
        else:
            project_trial(bad, anchor, _law(n))
