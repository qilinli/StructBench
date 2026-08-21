"""The wave-1d benchmark: split, aux field, QoIs (ADR-0025)."""

from __future__ import annotations

from ...datasets.canonical import CaseTrajectory
from ...datasets.sph_mesh import synthesize_lattice_mesh
from ...eval import QoiFn, arrival_time, peak_stress

_LENGTHS = (200, 300, 400, 500)
_VELOCITIES = (1, 2, 4, 8)


def _case(length: int, velocity: int) -> str:
    return f"W1D-{length}-{velocity}"


#: Fixed, immutable split (ADR-0025). Changing it is a new benchmark version.
VAL: list[str] = [_case(300, 2), _case(400, 4)]
TEST_INTERP: list[str] = [_case(300, 4), _case(400, 2)]
TRAIN: list[str] = [
    _case(length, velocity)
    for length in _LENGTHS
    for velocity in _VELOCITIES
    if _case(length, velocity) not in VAL + TEST_INTERP
]
ALL_BENCHMARK_CASES: list[str] = TRAIN + VAL + TEST_INTERP

#: Auxiliary per-particle target: the travelling stress wave IS the signal.
AUX_FIELD = "axial_stress"

#: ADR-0025 QoIs: gauge arrival times (ms) and global peak stress (MPa).
QOIS: dict[str, QoiFn] = {
    "arrival_time_25": arrival_time(0.25),
    "arrival_time_50": arrival_time(0.50),
    "arrival_time_75": arrival_time(0.75),
    "peak_stress": peak_stress,
}


def native_mesh_transform(trajectory: CaseTrajectory) -> CaseTrajectory:
    """Lattice mesh for the mesh-native families (the ADR-0047 mechanism).

    The wave bar's particles sit on an exact, complete 2.0 mm generation
    lattice (5 rows x {100, 150, 200, 250} columns by bar length; verified on
    all 16 cases, 2026-08-21), so
    :func:`~structbench.datasets.sph_mesh.synthesize_lattice_mesh` recovers it
    directly — the strict complete-lattice contract, no ``allow_missing``. No
    boundary nodes are appended: unlike Taylor's analytic rigid wall, the
    bar's arrest is realised inside the particle set itself (a constrained
    end column of ordinary part-1 particles; the benchmark declares no
    ``kinematic_types`` and no ``boundary_feature_fn``), so there is nothing
    kinematic to synthesize. Applied by the training pipeline to the
    mesh-native families only — the cgn path never sees it.
    """
    return synthesize_lattice_mesh(trajectory)


def _initial_velocity(case_id: str) -> float:
    """Initial axial velocity (mm/ms) from a wave case id ``W1D-<length>-<v>``.

    The scalar loading parameter of the ADR-0025 sweep (1-8 mm/ms): bar
    length is visible to a model through the geometry itself, so the initial
    velocity is the one case parameter a static input cannot reveal — the
    wave analog of Taylor/notch impact velocity for the Transolver
    ``impact_velocity_feature`` (ADR-0051 B / ADR-0054).
    """
    return float(case_id.rsplit("-", 1)[1])
