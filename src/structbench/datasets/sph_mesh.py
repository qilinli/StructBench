"""Lattice mesh synthesis for SPH benchmarks (ADR-0047).

SPH cases carry no element connectivity, so the mesh-native model families
(mgn, transolver, geoflare) cannot run on them: all three consume
``cells``/``reference_coords`` (mesh edges for MGN, reference-coordinate input
channels for all). For SPH datasets whose particles sit on a regular
generation lattice, this module recovers that lattice from the initial frame
and emits a simplicial mesh — each lattice quad split into two triangles —
plus static boundary nodes for analytic walls.

Synthesis is loader-level and opt-in per benchmark
(:attr:`~structbench.benchmarks.registry.BenchmarkSpec.mesh_transform`): the
canonical ``.h5`` files are untouched (ADR-0042 stands) and the cgn family
never applies it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .canonical import CaseTrajectory

__all__ = ["append_wall_nodes", "synthesize_lattice_mesh"]

#: Largest spread between neighbouring lattice-line spacings tolerated before
#: the point set is rejected as not-a-lattice. Exact lattices have zero
#: spread; jittered or deformed frames split ``np.unique`` into many nearby
#: values whose spacing ratio explodes past this immediately.
_SPACING_RATIO_MAX = 1.5


def synthesize_lattice_mesh(
    traj: CaseTrajectory,
    *,
    part: int | None = None,
    allow_missing: bool = False,
) -> CaseTrajectory:
    """Recover a triangle mesh from a 2D SPH trajectory's initial lattice.

    The meshed particles' initial positions must sit on a regular lattice
    (unique site per particle, uniform line spacing per axis). By default
    the lattice must be complete (every ``nx × ny`` site occupied — the
    Taylor bar, ADR-0047); with ``allow_missing=True`` sites may be vacant
    (the notched beam, ADR-0048) and a quad contributes triangles only when
    all four corners exist, stair-stepping the boundary. Each emitted quad
    is split into two triangles, so
    :func:`~structbench.models.mgn.mesh_ops.cells_to_edges`'s
    every-vertex-pair expansion yields the lattice edges plus one diagonal
    per quad (a standard simplicial mesh), not the double-diagonal a quad
    cell would emit.

    Parameters
    ----------
    traj:
        SPH trajectory (``cells``/``reference_coords`` unset), 2D.
    part:
        When given, only particles of this ``particle_type`` are meshed
        (e.g. the notch beam; the kinematic pin/support bodies stay
        unmeshed and interact through world edges, ADR-0048). ``cells``
        indices always refer to the FULL particle array, and
        ``reference_coords`` always covers every particle.
    allow_missing:
        Permit vacant lattice sites (see above). ``False`` preserves the
        strict ADR-0047 complete-lattice contract.

    Returns
    -------
    CaseTrajectory
        Copy with triangle ``cells`` (int64, indices into the full particle
        array) and ``reference_coords`` = the initial positions of ALL
        particles.

    Raises
    ------
    ValueError
        If the trajectory already carries a mesh, is not 2D, the meshed
        particles do not sit on a regular lattice (or, without
        ``allow_missing``, a complete one), or no quad has all four
        corners.
    """
    if traj.cells is not None or traj.reference_coords is not None:
        raise ValueError(
            f"case {traj.case_id!r} already has mesh connectivity; "
            "synthesize_lattice_mesh applies to SPH trajectories only"
        )
    p0 = traj.positions[0]
    if p0.shape[1] != 2:
        raise ValueError(
            f"case {traj.case_id!r} is {p0.shape[1]}D; lattice synthesis "
            "supports 2D only (ADR-0047)"
        )
    if part is None:
        index = np.arange(p0.shape[0])
    else:
        index = np.nonzero(traj.particle_type == part)[0]
        if index.size == 0:
            raise ValueError(
                f"case {traj.case_id!r}: no particles of type {part} to mesh"
            )
    pm = p0[index]
    xs, ix = np.unique(pm[:, 0], return_inverse=True)
    ys, iy = np.unique(pm[:, 1], return_inverse=True)
    nx, ny = len(xs), len(ys)
    n = pm.shape[0]
    if nx < 2 or ny < 2 or (not allow_missing and nx * ny != n):
        raise ValueError(
            f"case {traj.case_id!r}: initial positions do not form a complete "
            f"regular lattice ({n} particles vs {nx} x {ny} distinct "
            "coordinate lines)"
        )
    for axis, lines in (("x", xs), ("y", ys)):
        d = np.diff(lines)
        if d.max() > _SPACING_RATIO_MAX * d.min():
            raise ValueError(
                f"case {traj.case_id!r}: non-uniform {axis} spacing "
                f"[{d.min():.6g}, {d.max():.6g}]; initial positions do not "
                "form a regular lattice"
            )
    grid = np.full((ny, nx), -1, dtype=np.int64)
    grid[iy, ix] = index
    occupied = np.zeros((ny, nx), dtype=bool)
    occupied[iy, ix] = True
    if occupied.sum() != n:
        raise ValueError(
            f"case {traj.case_id!r}: lattice sites multiply occupied; "
            "initial positions do not form a regular lattice"
        )
    if not allow_missing and (grid < 0).any():
        raise ValueError(
            f"case {traj.case_id!r}: lattice sites multiply occupied or "
            "empty; initial positions do not form a complete lattice"
        )

    a = grid[:-1, :-1].ravel()
    b = grid[:-1, 1:].ravel()
    c = grid[1:, 1:].ravel()
    d_ = grid[1:, :-1].ravel()
    full = (a >= 0) & (b >= 0) & (c >= 0) & (d_ >= 0)
    a, b, c, d_ = a[full], b[full], c[full], d_[full]
    if a.size == 0:
        raise ValueError(
            f"case {traj.case_id!r}: no lattice quad has all four corners; "
            "nothing to mesh"
        )
    cells = np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([a, c, d_], axis=1)], axis=0
    )
    return replace(traj, cells=cells, reference_coords=p0.copy())


def append_wall_nodes(
    traj: CaseTrajectory,
    *,
    plane_x: float,
    y_min: float,
    y_max: float,
    spacing: float,
    node_type: int,
) -> CaseTrajectory:
    """Append static wall nodes on an ``x = plane_x`` plane to a mesh case.

    The wall enters the graph as unmeshed nodes of a kinematic particle type
    (ADR-0047): constant position across all frames, zero auxiliary value,
    no cells — contact is learned through world edges exactly as
    DeformingPlate's OBSTACLE nodes (ADR-0043). Call after
    :func:`synthesize_lattice_mesh` so cell indices are unaffected (wall rows
    are appended at the end).

    Parameters
    ----------
    traj:
        Mesh trajectory (``cells``/``reference_coords`` populated), 2D.
    plane_x:
        Wall plane position in the mm working frame.
    y_min, y_max:
        Lateral extent of the wall segment (mm); must generously cover the
        contact footprint (for Taylor, the measured worst-case mushroom
        spread — ADR-0047).
    spacing:
        Node spacing along the wall (mm); the SPH particle spacing.
    node_type:
        Particle-type code for the wall rows; must be in the benchmark
        spec's ``kinematic_types``.

    Returns
    -------
    CaseTrajectory
        Copy with wall rows appended to ``positions``, ``aux``,
        ``particle_type``, and ``reference_coords``; ``cells`` unchanged.

    Raises
    ------
    ValueError
        If the trajectory has no mesh yet, is not 2D, or the span/spacing
        yields no nodes.
    """
    if traj.cells is None or traj.reference_coords is None:
        raise ValueError(
            f"case {traj.case_id!r} has no mesh connectivity; call "
            "synthesize_lattice_mesh before append_wall_nodes"
        )
    if traj.positions.shape[2] != 2:
        raise ValueError(
            f"case {traj.case_id!r} is {traj.positions.shape[2]}D; "
            "append_wall_nodes supports 2D only (ADR-0047)"
        )
    n_wall = int(round((y_max - y_min) / spacing)) + 1
    if n_wall < 2 or y_max <= y_min:
        raise ValueError(
            f"wall span [{y_min}, {y_max}] at spacing {spacing} yields "
            f"{n_wall} node(s); need at least 2"
        )
    ys = np.linspace(y_min, y_max, n_wall)
    wall = np.stack([np.full(n_wall, plane_x), ys], axis=1).astype(np.float32)

    n_frames = traj.positions.shape[0]
    positions = np.concatenate(
        [traj.positions, np.broadcast_to(wall[None], (n_frames, n_wall, 2))],
        axis=1,
    )
    aux = np.concatenate(
        [
            traj.aux,
            np.zeros((n_frames, n_wall, traj.aux.shape[2]), dtype=np.float32),
        ],
        axis=1,
    )
    particle_type = np.concatenate(
        [traj.particle_type, np.full(n_wall, node_type, dtype=np.int64)]
    )
    reference_coords = np.concatenate([traj.reference_coords, wall], axis=0)
    return replace(
        traj,
        positions=positions,
        aux=aux,
        particle_type=particle_type,
        reference_coords=reference_coords,
    )
