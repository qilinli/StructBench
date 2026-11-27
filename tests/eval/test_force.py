"""Force read from the state (``structbench.eval.force``): the internal-force
operator against closed forms, the quantities built on it, and the two
reported properties."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from structbench.datasets.canonical import CaseTrajectory
from structbench.eval.force import (
    body_force,
    element_mean,
    force_balance_residual,
    internal_force,
    lumped_mass,
    plane_penetration,
    reaction_force,
    section_force,
    series_impulse,
    series_mean,
    series_peak,
)
from structbench.eval.metrics import QoiInputs
from structbench.eval.rollout import QoiState, rollout


def _grid(x0, x1, y0, y1, nx, ny, *, distort=0.0):
    """A structured mesh of ``nx`` by ``ny`` quadrilaterals; ``distort``
    moves the interior nodes off the lattice."""
    xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    if distort:
        rng = np.random.default_rng(7)
        inner = (
            (nodes[:, 0] > x0)
            & (nodes[:, 0] < x1)
            & (nodes[:, 1] > y0)
            & (nodes[:, 1] < y1)
        )
        nodes[inner] += distort * rng.uniform(-1, 1, (int(inner.sum()), 2))
    cells = np.array(
        [
            [
                j * (nx + 1) + i,
                j * (nx + 1) + i + 1,
                (j + 1) * (nx + 1) + i + 1,
                (j + 1) * (nx + 1) + i,
            ]
            for j in range(ny)
            for i in range(nx)
        ]
    )
    return nodes, cells


def _uniform(cells, s11=0.0, s22=0.0, s33=0.0, s12=0.0):
    return np.tile(np.array([s11, s22, s33, s12]), (len(cells), 1))


def test_a_uniform_stress_loads_only_the_boundary_of_a_distorted_patch():
    """Plane strain: the interior nodes carry nothing, each face carries its
    traction, and the body as a whole carries nothing."""
    nodes, cells = _grid(0.0, 3.0, 0.0, 2.0, 3, 2, distort=0.15)
    stress = _uniform(cells, s11=50.0, s22=-20.0, s33=7.0, s12=12.0)
    force = internal_force(nodes, stress, cells, "plane_strain")
    interior = (
        (nodes[:, 0] > 0) & (nodes[:, 0] < 3) & (nodes[:, 1] > 0) & (nodes[:, 1] < 2)
    )
    assert interior.sum() == 2
    np.testing.assert_allclose(force[interior], 0.0, atol=1e-10)
    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-10)
    # the face x = 3 (normal +x, length 2): traction (s11, s12) per unit area
    right = nodes[:, 0] == 3.0
    np.testing.assert_allclose(force[right].sum(axis=0), [100.0, 24.0], atol=1e-10)
    top = nodes[:, 1] == 2.0
    np.testing.assert_allclose(force[top].sum(axis=0), [36.0, -60.0], atol=1e-10)
    # node order around the element does not matter
    flipped = internal_force(nodes, stress, cells[:, ::-1], "plane_strain")
    np.testing.assert_allclose(flipped, force, atol=1e-10)


def test_an_axial_stress_on_a_tube_is_its_stress_times_its_section():
    nodes, cells = _grid(2.0, 5.0, 0.0, 3.0, 3, 4)
    force = internal_force(nodes, _uniform(cells, s22=-100.0), cells, "axisymmetric")
    section = np.pi * (5.0**2 - 2.0**2)
    top, base = nodes[:, 1] == 3.0, nodes[:, 1] == 0.0
    assert force[top, 1].sum() == pytest.approx(-100.0 * section)
    assert force[base, 1].sum() == pytest.approx(100.0 * section)
    assert force[:, 1].sum() == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(force[~top & ~base], 0.0, atol=1e-9)


def test_a_hydrostatic_stress_on_a_ring_balances_at_its_interior_nodes():
    """Uniform pressure is in equilibrium in a body of revolution: the hoop
    term cancels the radial gradient at every interior node."""
    nodes, cells = _grid(2.0, 5.0, 0.0, 2.0, 3, 2)
    force = internal_force(
        nodes, _uniform(cells, s11=-30.0, s22=-30.0, s33=-30.0), cells, "axisymmetric"
    )
    interior = (
        (nodes[:, 0] > 2) & (nodes[:, 0] < 5) & (nodes[:, 1] > 0) & (nodes[:, 1] < 2)
    )
    np.testing.assert_allclose(force[interior], 0.0, atol=1e-9)
    # the hoop stress alone pushes outward with 2 pi s_tt times the section
    hoop = internal_force(nodes, _uniform(cells, s33=4.0), cells, "axisymmetric")
    assert hoop[:, 0].sum() == pytest.approx(2 * np.pi * 4.0 * 3.0 * 2.0)


def test_frames_are_independent_and_shapes_are_checked():
    nodes, cells = _grid(2.0, 5.0, 0.0, 2.0, 3, 2)
    stress = np.stack([_uniform(cells, s22=1.0), _uniform(cells, s22=3.0)])
    force = internal_force(np.stack([nodes, nodes]), stress, cells, "axisymmetric")
    np.testing.assert_allclose(force[1], 3.0 * force[0])
    with pytest.raises(ValueError, match="not implemented"):
        internal_force(nodes, stress[0], cells, "shell")
    with pytest.raises(ValueError, match="four-node"):
        internal_force(nodes, stress[0], cells[:, :3], "plane_strain")
    with pytest.raises(ValueError, match="does not match"):
        internal_force(nodes, stress[0, :2], cells, "plane_strain")


def test_the_lumped_mass_sums_to_density_times_volume():
    nodes, cells = _grid(2.0, 5.0, 0.0, 3.0, 3, 4, distort=0.1)
    plane = lumped_mass(nodes, cells, "plane_strain", 2.0)
    assert plane.sum() == pytest.approx(2.0 * 9.0)
    ring, ring_cells = _grid(2.0, 5.0, 0.0, 3.0, 3, 4)
    solid = lumped_mass(ring, ring_cells, "axisymmetric", 2.0)
    assert solid.sum() == pytest.approx(2.0 * np.pi * (25.0 - 4.0) * 3.0)


@dataclass(frozen=True)
class _Plane:
    normal: tuple[float, ...]
    offset: float | None = None
    node_type: int | None = None


def _tube_inputs(n_frames=4):
    """A tube compressed between a floor and a plate carried by one node:
    axial stress growing linearly in time, nothing moving."""
    nodes, cells = _grid(2.0, 5.0, 0.0, 3.0, 3, 4)
    plate = np.array([[0.0, 3.0]])
    positions = np.tile(np.vstack([nodes, plate]), (n_frames, 1, 1))
    types = np.array([0] * len(nodes) + [2])
    stress = np.stack([_uniform(cells, s22=-100.0 * t) for t in range(n_frames)])
    return QoiInputs(
        time=np.arange(n_frames) * 0.5,
        positions=positions,
        aux=np.zeros((n_frames, len(types))),
        particle_type=types,
        init=1,
        element_stress=stress,
        stress_site="element",
        cells=cells,
        element_formulation="axisymmetric",
        node_sets={"base": np.append(nodes[:, 1] == 0.0, False)},
    )


SECTION = np.pi * (5.0**2 - 2.0**2)


def test_the_three_forces_of_a_compressed_tube_agree():
    inputs = _tube_inputs()
    growth = 100.0 * SECTION * np.arange(4)
    # the floor pushes the base up
    base = reaction_force("base", (0.0, 1.0))(inputs)
    np.testing.assert_allclose(base, growth, rtol=1e-12)
    # the plate pushes the top along its outward normal (down)
    plate = body_force(_Plane((0.0, -1.0), node_type=2), tolerance=1e-6)(inputs)
    np.testing.assert_allclose(plate, growth, rtol=1e-12)
    floor = body_force(_Plane((0.0, 1.0), offset=0.0), tolerance=1e-6)(inputs)
    np.testing.assert_allclose(floor, growth, rtol=1e-12)
    # through any section the upper part pushes the lower one down
    cut = section_force(1, 1.6, (0.0, 1.0))(inputs)
    np.testing.assert_allclose(cut, -growth, rtol=1e-12)
    with pytest.raises(KeyError, match="no node set 'top'"):
        reaction_force("top", (0.0, 1.0))(inputs)


def test_series_are_reduced_over_the_scored_span():
    inputs = _tube_inputs()
    plate = body_force(_Plane((0.0, -1.0), node_type=2), tolerance=1e-6)
    unit = 100.0 * SECTION
    # scored frames 1..3: values 1, 2, 3 units at times 0.5, 1.0, 1.5
    assert series_peak(plate)(inputs) == pytest.approx(3 * unit)
    assert series_mean(plate)(inputs) == pytest.approx(2 * unit)
    assert series_impulse(plate)(inputs) == pytest.approx(2 * unit * 1.0)


def test_a_force_is_undefined_without_the_stress():
    inputs = _tube_inputs()
    bare = QoiInputs(
        time=inputs.time,
        positions=inputs.positions,
        aux=inputs.aux,
        particle_type=inputs.particle_type,
        init=1,
    )
    plate = body_force(_Plane((0.0, -1.0), node_type=2), tolerance=1e-6)
    assert np.isnan(plate(bare)).all() and plate(bare).shape == (4,)
    assert np.isnan(series_peak(plate)(bare))
    assert np.isnan(section_force(1, 1.6, (0.0, 1.0))(bare)).all()


def test_the_balance_residual_is_zero_in_equilibrium_and_not_otherwise():
    nodes, cells = _grid(0.0, 3.0, 0.0, 2.0, 3, 2, distort=0.1)
    free = (nodes[:, 0] > 0) & (nodes[:, 0] < 3) & (nodes[:, 1] > 0) & (nodes[:, 1] < 2)
    x = nodes[None]
    uniform = _uniform(cells, s11=5.0, s12=2.0)[None]
    assert force_balance_residual(
        x, uniform, cells, "plane_strain", free
    ) == pytest.approx(0.0, abs=1e-12)
    rough = np.random.default_rng(3).normal(size=uniform.shape)
    residual = force_balance_residual(x, rough, cells, "plane_strain", free)
    assert 0.05 < residual < 1.0
    assert np.isnan(force_balance_residual(x, 0 * uniform, cells, "plane_strain", free))
    assert np.isnan(
        force_balance_residual(x, uniform * np.nan, cells, "plane_strain", free)
    )


def test_penetration_is_the_largest_depth_past_the_plane():
    inputs = _tube_inputs()
    positions = np.array(inputs.positions, dtype=float)
    plate = _Plane((0.0, -1.0), node_type=2)
    floor = _Plane((0.0, 1.0), offset=0.0)
    assert plane_penetration(plate, positions, inputs.particle_type) == 0.0
    positions[2, 17, 1] += 0.25  # a top node rises through the plate
    positions[3, 1, 1] -= 0.4  # a base node sinks through the floor
    assert plane_penetration(plate, positions, inputs.particle_type) == pytest.approx(
        0.25
    )
    assert plane_penetration(floor, positions, inputs.particle_type) == pytest.approx(
        0.4
    )
    # only the nodes asked about
    others = np.ones(len(inputs.particle_type), dtype=bool)
    others[1] = False
    assert plane_penetration(floor, positions, inputs.particle_type, others) == 0.0
    with pytest.raises(ValueError, match="exactly one node"):
        plane_penetration(
            _Plane((0.0, 1.0), node_type=9), positions, inputs.particle_type
        )


class _StillSim:
    """Nothing moves; the nodal stress channels double each step."""

    def __init__(self, aux0):
        self._aux = torch.as_tensor(aux0)

    def predict_positions(self, position_sequence, nparticles_per_example, types):
        self._aux = self._aux * 2.0
        return position_sequence[:, -1], self._aux


def test_a_rollout_hands_its_quantities_the_stress_of_both_sides():
    """A nodal run's stress is averaged to the elements and said to be; the
    reference side is the stored element stress; a series is kept."""
    nodes, cells = _grid(2.0, 5.0, 0.0, 3.0, 1, 2)
    n_frames, n_nodes = 5, len(nodes)
    positions = np.tile(nodes, (n_frames, 1, 1)).astype(np.float32)
    nodal = np.zeros((n_frames, n_nodes, 4), np.float32)
    nodal[..., 1] = -10.0  # s_zz, seeded frames
    traj = CaseTrajectory(
        "c",
        positions,
        np.zeros(n_nodes, np.int64),
        nodal,
        np.arange(n_frames, dtype=float),
        cells=cells,
        reference_coords=nodes.astype(np.float32),
    )
    stored = np.zeros((n_frames, len(cells), 4), np.float32)
    stored[..., 1] = -10.0 * np.arange(1, n_frames + 1)[:, None]
    base = reaction_force("base", (0.0, 1.0))
    seen = []

    def peak(inputs):
        seen.append(inputs.stress_site)
        return series_peak(base)(inputs)

    state = QoiState(
        formulation="axisymmetric",
        pred_stress=("aux", (0, 1, 2, 3)),
        true_stress=stored,
        node_sets={"base": nodes[:, 1] == 0.0},
        series={"base_force": base},
    )
    result = rollout(
        _StillSim(nodal[1]), traj, input_frames=2, qois={"peak": peak}, qoi_state=state
    )
    assert seen == ["node_mean", "element"]
    assert result.stress_site == "node_mean"
    np.testing.assert_allclose(
        result.element_stress, element_mean(result.predicted_aux, cells)
    )
    section = np.pi * 21.0
    # predicted: 10 on the two seeded frames, then doubling each step
    np.testing.assert_allclose(
        result.series_pred["base_force"], section * np.array([10, 10, 20, 40, 80.0])
    )
    np.testing.assert_allclose(
        result.series_true["base_force"], section * 10.0 * np.arange(1, 6)
    )
    assert result.qoi_pred["peak"] == pytest.approx(80 * section)
    assert result.qoi_true["peak"] == pytest.approx(50 * section)
    # without the state a quantity that needs it is undefined, not wrong
    plain = rollout(_StillSim(nodal[1]), traj, input_frames=2, qois={"peak": peak})
    assert np.isnan(plain.qoi_pred["peak"]) and plain.series_pred == {}
