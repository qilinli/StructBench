"""Wave mesh-native wiring: lattice transform, empty scripted set (E-B)."""

import numpy as np

from structbench.benchmarks.wave_propagation_1d import SPEC, native_mesh_transform
from structbench.datasets.canonical import CaseTrajectory


def _sph_traj(nx=10, ny=5, T=4, spacing=2.0):
    """Synthetic wave-like strip: a complete nx x ny lattice of part-1 SPH."""
    xs, ys = np.meshgrid(
        np.arange(nx) * spacing, np.arange(ny) * spacing, indexing="xy"
    )
    p0 = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    pos = np.repeat(p0[None], T, axis=0).copy()
    pos[1:, :, 0] += 0.01 * np.arange(1, T)[:, None]  # small axial drift
    aux = np.zeros((T, len(p0)), dtype=np.float32)
    return CaseTrajectory(
        "W1D-syn",
        pos,
        np.ones(len(p0), np.int64),
        aux,
        np.arange(T, dtype=np.float64),
    )


def test_spec_declares_native_wiring():
    assert SPEC.mesh_transform is native_mesh_transform
    assert SPEC.kinematic_types == ()
    # No kinematic parts exist, so the scripted set MUST be pinned empty: the
    # simulators' family default (1,) would script every bar particle and
    # fail the scripted-subset check against the empty kinematic set.
    assert SPEC.scripted_types == ()
    assert SPEC.loading_scalar is not None


def test_native_mesh_transform_meshes_the_full_lattice():
    traj = _sph_traj(nx=10, ny=5)
    out = native_mesh_transform(traj)
    n = traj.positions.shape[1]
    assert out.cells is not None and out.reference_coords is not None
    # Complete lattice: 2 triangles per quad, no appended nodes.
    assert out.cells.shape == ((10 - 1) * (5 - 1) * 2, 3)
    assert out.cells.min() >= 0 and out.cells.max() < n
    assert out.reference_coords.shape == (n, 2)
    np.testing.assert_array_equal(out.reference_coords, traj.positions[0])
    # Particle set unchanged (positions/aux/types identical to the cgn path).
    np.testing.assert_array_equal(out.positions, traj.positions)
    np.testing.assert_array_equal(out.aux, traj.aux)
    np.testing.assert_array_equal(out.particle_type, traj.particle_type)
