"""Lattice mesh synthesis + wall-node injection (ADR-0047)."""

import numpy as np
import pytest

from structbench.datasets.canonical import CaseTrajectory
from structbench.datasets.sph_mesh import append_wall_nodes, synthesize_lattice_mesh


def _lattice_traj(nx=4, ny=3, T=5, spacing=0.5, permute=True, seed=0):
    """SPH-like trajectory whose frame 0 is an nx*ny lattice, then deforms."""
    xs, ys = np.meshgrid(
        np.arange(nx) * spacing, np.arange(ny) * spacing, indexing="xy"
    )
    p0 = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    if permute:
        # Particle order must not matter: the loader gives arbitrary order.
        p0 = p0[np.random.default_rng(seed).permutation(len(p0))]
    pos = np.repeat(p0[None], T, axis=0).copy()
    # Later frames deform arbitrarily; only frame 0 must be a lattice.
    pos[1:] += np.linspace(0.1, 0.3, T - 1)[:, None, None]
    aux = np.zeros((T, len(p0)), dtype=np.float32)
    return CaseTrajectory(
        "synthetic",
        pos,
        np.ones(len(p0), np.int64),
        aux,
        np.arange(T, dtype=np.float64),
    )


def test_synthesize_recovers_lattice_triangles():
    nx, ny = 4, 3
    traj = synthesize_lattice_mesh(_lattice_traj(nx=nx, ny=ny))
    assert traj.cells is not None and traj.reference_coords is not None
    assert traj.cells.shape == (2 * (nx - 1) * (ny - 1), 3)
    assert traj.cells.dtype == np.int64
    np.testing.assert_array_equal(traj.reference_coords, traj.positions[0])
    # Every vertex index valid, every triangle non-degenerate.
    assert traj.cells.min() >= 0 and traj.cells.max() < nx * ny
    for tri in traj.cells:
        assert len(set(tri.tolist())) == 3
    # The undirected edge set is the lattice: horizontals + verticals + one
    # diagonal per quad (a simplicial mesh, not the quad's double diagonal).
    edges = set()
    for a, b, c in traj.cells:
        for u, v in ((a, b), (b, c), (a, c)):
            edges.add((min(u, v), max(u, v)))
    n_expected = ny * (nx - 1) + nx * (ny - 1) + (nx - 1) * (ny - 1)
    assert len(edges) == n_expected
    # Geometric check: every edge spans at most one lattice quad's diagonal.
    p = traj.reference_coords
    for u, v in edges:
        assert np.linalg.norm(p[u] - p[v]) <= 0.5 * np.sqrt(2) + 1e-6


def test_synthesize_is_order_invariant():
    a = synthesize_lattice_mesh(_lattice_traj(permute=False))
    b = synthesize_lattice_mesh(_lattice_traj(permute=True))

    def edge_coords(traj):
        out = set()
        for tri in traj.cells:
            for i in range(3):
                for j in range(i + 1, 3):
                    pa = tuple(np.round(traj.reference_coords[tri[i]], 6))
                    pb = tuple(np.round(traj.reference_coords[tri[j]], 6))
                    out.add(tuple(sorted((pa, pb))))
        return out

    assert edge_coords(a) == edge_coords(b)


def test_synthesize_rejects_jittered_positions():
    traj = _lattice_traj()
    jitter = np.random.default_rng(1).normal(0.0, 0.05, traj.positions[0].shape)
    positions = traj.positions.copy()
    positions[0] += jitter.astype(np.float32)
    from dataclasses import replace

    with pytest.raises(ValueError, match="lattice"):
        synthesize_lattice_mesh(replace(traj, positions=positions))


def test_synthesize_rejects_incomplete_lattice():
    traj = _lattice_traj()
    from dataclasses import replace

    with pytest.raises(ValueError, match="lattice"):
        synthesize_lattice_mesh(
            replace(
                traj,
                positions=traj.positions[:, 1:],
                aux=traj.aux[:, 1:],
                particle_type=traj.particle_type[1:],
            )
        )


def test_synthesize_rejects_mesh_trajectory():
    traj = synthesize_lattice_mesh(_lattice_traj())
    with pytest.raises(ValueError, match="already has mesh connectivity"):
        synthesize_lattice_mesh(traj)


def test_append_wall_nodes():
    traj = synthesize_lattice_mesh(_lattice_traj(nx=4, ny=3, T=5))
    n_bar = traj.positions.shape[1]
    n_cells = traj.cells.shape[0]
    out = append_wall_nodes(
        traj, plane_x=-1.0, y_min=-2.0, y_max=2.0, spacing=0.5, node_type=2
    )
    n_wall = 9  # (2 - (-2)) / 0.5 + 1
    assert out.positions.shape == (5, n_bar + n_wall, 2)
    assert out.aux.shape == (5, n_bar + n_wall)
    assert out.particle_type.shape == (n_bar + n_wall,)
    assert out.reference_coords.shape == (n_bar + n_wall, 2)
    # Wall rows: static across frames, on the plane, type 2, zero aux.
    wall_pos = out.positions[:, n_bar:]
    np.testing.assert_array_equal(wall_pos, np.repeat(wall_pos[:1], 5, axis=0))
    np.testing.assert_allclose(wall_pos[0, :, 0], -1.0)
    np.testing.assert_allclose(out.aux[:, n_bar:], 0.0)
    assert set(out.particle_type[n_bar:].tolist()) == {2}
    # Bar rows and cells untouched.
    np.testing.assert_array_equal(out.positions[:, :n_bar], traj.positions)
    np.testing.assert_array_equal(out.cells, traj.cells)
    assert out.cells.shape == (n_cells, 3)


def test_append_wall_nodes_requires_mesh():
    with pytest.raises(ValueError, match="no mesh connectivity"):
        append_wall_nodes(
            _lattice_traj(),
            plane_x=-1.0,
            y_min=-2.0,
            y_max=2.0,
            spacing=0.5,
            node_type=2,
        )


def _two_part_traj(nx=6, ny=4, T=4, spacing=0.5, notch=((2, 0), (3, 0))):
    """Beam lattice (type 1) with vacant notch sites + an off-lattice pin (2)."""
    xs, ys = np.meshgrid(
        np.arange(nx) * spacing, np.arange(ny) * spacing, indexing="xy"
    )
    p0 = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    keep = np.ones(len(p0), bool)
    for cx, cy in notch:
        keep[cy * nx + cx] = False
    beam = p0[keep]
    pin = np.array([[1.13, 3.7], [1.63, 3.9]], dtype=np.float32)
    p0 = np.concatenate([beam, pin])
    ptype = np.concatenate(
        [np.ones(len(beam), np.int64), np.full(len(pin), 2, np.int64)]
    )
    pos = np.repeat(p0[None], T, axis=0).copy()
    pos[1:] += 0.05
    aux = np.zeros((T, len(p0)), dtype=np.float32)
    return CaseTrajectory("notchy", pos, ptype, aux, np.arange(T, dtype=np.float64))


def test_partial_lattice_skips_vacant_quads():
    # ADR-0048: quads touching a vacant (notch) site contribute no triangles.
    traj = _two_part_traj()
    out = synthesize_lattice_mesh(traj, part=1, allow_missing=True)
    n_beam = int((traj.particle_type == 1).sum())
    # Cells reference beam rows only; reference_coords covers ALL particles.
    assert out.cells.max() < n_beam
    assert out.reference_coords.shape == (traj.positions.shape[1], 2)
    # 6x4 lattice has 5x3=15 quads; the two vacant sites at (2,0),(3,0) kill
    # quads with x-index {1,2,3} in the bottom row -> 12 remain -> 24 tris.
    assert out.cells.shape == (24, 3)
    # Vacant sites are truly unreferenced.
    for cx, cy in ((2, 0), (3, 0)):
        vac = np.array([cx * 0.5, cy * 0.5], dtype=np.float32)
        assert not np.any(
            np.all(np.isclose(out.reference_coords[out.cells.ravel()], vac), axis=1)
        )


def test_partial_lattice_strict_mode_still_rejects():
    traj = _two_part_traj()
    with pytest.raises(ValueError, match="lattice"):
        synthesize_lattice_mesh(traj, part=1)  # allow_missing=False


def test_part_restriction_requires_particles():
    traj = _lattice_traj()
    with pytest.raises(ValueError, match="no particles of type"):
        synthesize_lattice_mesh(traj, part=7, allow_missing=True)
