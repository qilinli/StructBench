"""Taylor mesh-native wiring: transform, kinematic wall, bar-only QoIs (ADR-0047)."""

import numpy as np

from structbench.benchmarks.taylor_impact_2d import (
    QOIS,
    SPEC,
    WALL_NODE_TYPE,
    WALL_SPAN_MM,
    WALL_X_MM,
    native_mesh_transform,
)
from structbench.datasets.canonical import CaseTrajectory
from structbench.eval.metrics import QoiInputs, mushroom_width


def _sph_traj(nx=6, ny=4, T=8, spacing=0.5):
    xs, ys = np.meshgrid(
        np.arange(nx) * spacing, np.arange(ny) * spacing, indexing="xy"
    )
    p0 = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    pos = np.repeat(p0[None], T, axis=0).copy()
    pos[1:, :, 0] -= 0.01 * np.arange(1, T)[:, None]  # drifts toward the wall
    aux = np.full((T, len(p0)), 2.0, dtype=np.float32)
    return CaseTrajectory(
        "T-syn",
        pos,
        np.ones(len(p0), np.int64),
        aux,
        np.arange(T, dtype=np.float64),
    )


def test_spec_declares_native_wiring():
    assert SPEC.kinematic_types == (WALL_NODE_TYPE,)
    assert SPEC.mesh_transform is native_mesh_transform
    # The wall is scripted (the DP OBSTACLE analog; its GT velocity input is
    # identically zero) — required, or the simulators' scripted-subset check
    # rejects the (2,) kinematic set at construction.
    assert SPEC.scripted_types == (WALL_NODE_TYPE,)


def test_native_mesh_transform_meshes_and_appends_wall():
    traj = _sph_traj(nx=6, ny=4)
    out = native_mesh_transform(traj)
    n_bar = traj.positions.shape[1]
    assert out.cells is not None and out.reference_coords is not None
    # Cell indices reference bar rows only (wall rows are appended after).
    assert out.cells.max() < n_bar
    wall = out.particle_type == WALL_NODE_TYPE
    assert wall.sum() == int(round(2 * WALL_SPAN_MM / 0.5)) + 1
    np.testing.assert_allclose(out.positions[0, wall, 0], WALL_X_MM)
    np.testing.assert_allclose(out.aux[:, wall], 0.0)
    # Bar rows unchanged.
    np.testing.assert_array_equal(out.positions[:, :n_bar], traj.positions)


def test_qois_ignore_wall_rows():
    traj = native_mesh_transform(_sph_traj())
    inputs = QoiInputs(
        time=traj.time,
        positions=traj.positions,
        aux=traj.aux,
        particle_type=traj.particle_type,
        init=2,
    )
    bar = traj.particle_type != WALL_NODE_TYPE
    bar_inputs = QoiInputs(
        time=traj.time,
        positions=traj.positions[:, bar],
        aux=traj.aux[:, bar],
        particle_type=traj.particle_type[bar],
        init=2,
    )
    for name, fn in QOIS.items():
        assert fn(inputs) == fn(bar_inputs), name
    # The guard is load-bearing: the raw helper DOES read the wall span.
    assert mushroom_width(inputs) != mushroom_width(bar_inputs)
    assert QOIS["mushroom_width"](inputs) == mushroom_width(bar_inputs)


def test_qois_are_identity_without_wall_rows():
    # The cgn path never sees wall rows; wrapped QoIs must equal the raw
    # helpers there (cross-family QoI comparability).
    traj = _sph_traj()
    inputs = QoiInputs(
        time=traj.time,
        positions=traj.positions,
        aux=traj.aux,
        particle_type=traj.particle_type,
        init=2,
    )
    assert QOIS["mushroom_width"](inputs) == mushroom_width(inputs)
