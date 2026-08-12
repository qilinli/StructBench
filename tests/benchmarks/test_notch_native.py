"""Notch-impact mesh-native wiring: beam-only mesh, scripted pin (ADR-0048)."""

import numpy as np

from structbench.benchmarks.notch_beam_2d_impact import (
    CONCRETE_TYPE,
    PIN_TYPE,
    SUPPORT_TYPE,
    native_mesh_transform,
)
from structbench.benchmarks.registry import get_benchmark
from structbench.datasets.canonical import CaseTrajectory


def test_spec_declares_native_wiring():
    spec = get_benchmark("notch_beam_2d_impact")
    assert spec.mesh_transform is native_mesh_transform
    assert spec.kinematic_types == (PIN_TYPE, SUPPORT_TYPE)
    # Pin AND supports are scripted (the DP OBSTACLE analog): both move in
    # the data (dynamic rigid pin; constrained-but-displacing supports), and
    # both feed their real GT next-step velocity as the scripted input.
    assert spec.scripted_types == (PIN_TYPE, SUPPORT_TYPE)


def test_transform_meshes_beam_only_and_appends_nothing():
    spacing = 2.5
    xs, ys = np.meshgrid(np.arange(8) * spacing, np.arange(4) * spacing, indexing="xy")
    beam = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    beam = np.delete(beam, [3, 4], axis=0)  # notch: two vacant sites
    pin = np.array([[8.0, 12.0], [10.5, 12.0]], dtype=np.float32)
    sup = np.array([[0.0, -2.5], [17.5, -2.5]], dtype=np.float32)
    p0 = np.concatenate([beam, pin, sup])
    ptype = np.concatenate(
        [
            np.full(len(beam), CONCRETE_TYPE, np.int64),
            np.full(len(pin), PIN_TYPE, np.int64),
            np.full(len(sup), SUPPORT_TYPE, np.int64),
        ]
    )
    pos = np.repeat(p0[None], 3, axis=0)
    traj = CaseTrajectory(
        "nb-syn",
        pos.copy(),
        ptype,
        np.zeros((3, len(p0)), np.float32),
        np.arange(3, dtype=np.float64),
    )
    out = native_mesh_transform(traj)
    # No nodes appended: particle set (and QoI inputs) identical to cgn's.
    assert out.positions.shape == traj.positions.shape
    np.testing.assert_array_equal(out.particle_type, traj.particle_type)
    # Cells cover beam rows only; pin/support rows are unmeshed.
    assert out.cells.max() < len(beam)
    assert out.reference_coords.shape == (len(p0), 2)
