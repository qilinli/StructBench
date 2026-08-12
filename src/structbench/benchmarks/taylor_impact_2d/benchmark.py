"""The v0.1 Taylor 2D impact benchmark: split, wall feature, QoIs (ADR-0019)."""

from __future__ import annotations

from dataclasses import replace

import torch

from ...datasets.canonical import CaseTrajectory
from ...datasets.sph_mesh import append_wall_nodes, synthesize_lattice_mesh
from ...eval import (
    QoiFn,
    QoiInputs,
    final_length,
    mushroom_width,
    peak_mean_aux,
    t_peak_mean_aux,
)

_GEOMS = (60, 80, 100)


def _cases(velocities: tuple[int, ...]) -> list[str]:
    return [f"T-20-{g}-{v}" for v in velocities for g in _GEOMS]


#: Fixed, immutable split (ADR-0019). Changing it is a new benchmark version.
TRAIN: list[str] = _cases((100, 110, 120, 140, 160, 180, 190))
VAL: list[str] = _cases((150,))
TEST_INTERP: list[str] = _cases((130, 170))
TEST_EXTRAP: list[str] = _cases((200,))
HELD_ASIDE: list[str] = ["T-20-80-Convergence"]
ALL_BENCHMARK_CASES: list[str] = TRAIN + VAL + TEST_INTERP + TEST_EXTRAP

#: Auxiliary per-particle target field (named correctly, not "strain").
AUX_FIELD = "von_mises_stress"

#: Rigidwall plane position in the model's mm working frame.
WALL_X_MM = -2.0

#: Particle-type code of the synthesized wall nodes (ADR-0047): the wall
#: shell's raw LS-DYNA part id (bar SPH particles are part id 1).
WALL_NODE_TYPE = 2

#: Half-span (mm) of the synthesized wall segment. Measured 2026-08-12 over
#: all 34 cases: worst-case lateral spread |y| = 47.2 mm (T-20-100-200, the
#: extrapolation velocity), so ±60 mm carries a >25% margin.
WALL_SPAN_MM = 60.0

#: Wall node spacing (mm) = the SPH generation-lattice spacing.
WALL_NODE_SPACING_MM = 0.5


def _bar_only(fn: QoiFn) -> QoiFn:
    """Evaluate ``fn`` on bar particles only, dropping wall rows (ADR-0047).

    The mesh-native families append static wall nodes (type
    :data:`WALL_NODE_TYPE`) to each trajectory; geometry QoIs like
    ``mushroom_width`` would otherwise read the wall span instead of the
    bar. A no-op when no wall rows are present (the cgn path), keeping QoI
    values comparable across families.
    """

    def wrapped(inputs: QoiInputs) -> float:
        ptype = inputs.particle_type
        if ptype is None:
            return fn(inputs)
        keep = ptype != WALL_NODE_TYPE
        if bool(keep.all()):
            return fn(inputs)
        return fn(
            replace(
                inputs,
                positions=inputs.positions[:, keep],
                aux=inputs.aux[:, keep],
                particle_type=ptype[keep],
            )
        )

    return wrapped


#: ADR-0019 §5 quantities of interest (extended by ADR-0032 §6): each maps a
#: full rollout (mm/MPa working frame) to one scalar. All are wrapped
#: bar-only (ADR-0047) so synthesized wall rows never enter the value.
QOIS: dict[str, QoiFn] = {
    "final_length": _bar_only(final_length),
    "mushroom_width": _bar_only(mushroom_width),
    # Temporal-fidelity pair (ADR-0032 §6): peak of the particle-mean von
    # Mises field (MPa) and its time (ms) — read mid-trajectory, not off the
    # final frame.
    "peak_von_mises": _bar_only(peak_mean_aux),
    "t_peak_von_mises": _bar_only(t_peak_mean_aux),
}


def native_mesh_transform(trajectory: CaseTrajectory) -> CaseTrajectory:
    """Lattice mesh + wall nodes for the mesh-native families (ADR-0047).

    Recovers the bar's 0.5 mm generation lattice as a triangle mesh
    (:func:`~structbench.datasets.sph_mesh.synthesize_lattice_mesh`) and
    appends static wall nodes of type :data:`WALL_NODE_TYPE` on the
    x = :data:`WALL_X_MM` plane spanning ±:data:`WALL_SPAN_MM`
    (:func:`~structbench.datasets.sph_mesh.append_wall_nodes`). Applied by
    the training pipeline to mgn/transolver/geoflare loads only — the cgn
    path never sees it.
    """
    return append_wall_nodes(
        synthesize_lattice_mesh(trajectory),
        plane_x=WALL_X_MM,
        y_min=-WALL_SPAN_MM,
        y_max=WALL_SPAN_MM,
        spacing=WALL_NODE_SPACING_MM,
        node_type=WALL_NODE_TYPE,
    )


def wall_distance_feature(positions_mm: torch.Tensor, radius: float) -> torch.Tensor:
    """Signed per-particle distance to the rigidwall plane, in radius units.

    Follows the canonical GNS boundary encoding: ``clamp(dist / radius, -1, 1)``.
    Negative values signal penetration *depth* (ADR-0028) — the unsigned
    ``clamp(dist, 0, radius)`` form used before v0.1 read identically zero for
    a particle resting on the wall and one driven through it, giving rollouts
    no restoring signal exactly where contact errors concentrate.

    Parameters
    ----------
    positions_mm:
        Current particle positions, shape ``(P, dim)``, in mm.
    radius:
        Connectivity radius (mm); sets the sensing horizon on both sides.

    Returns
    -------
    torch.Tensor
        Shape ``(P, 1)``: ``clamp((x - WALL_X_MM) / radius, -1, 1)``.
    """
    return torch.clamp((positions_mm[:, 0:1] - WALL_X_MM) / radius, min=-1.0, max=1.0)
