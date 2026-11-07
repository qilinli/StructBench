"""Two-site mechanics (ADR-0075 clause 3): projection parity with the loader,
quad kinematics, the J2 radial return and its by-construction properties."""

import math

import numpy as np
import pytest
import torch

from structbench.datasets.canonical import _cell_weights, _elements_to_nodes
from structbench.models.common.two_site import (
    RETURN_MAP_WIDTH,
    LawRows,
    cell_weights,
    corner_mean,
    law_rows,
    project_elements_to_nodes,
    quad_increment,
    radial_return,
    return_map_decode,
    sigma_y_rows,
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
