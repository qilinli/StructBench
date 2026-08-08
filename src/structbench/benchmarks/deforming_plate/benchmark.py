"""DeformingPlate task facts: frozen split, aux field, kinematics, QoIs (ADR-0043)."""

from __future__ import annotations

from ...eval import QoiFn, peak_nodal_aux, terminal_peak_displacement

#: Fixed, immutable split (ADR-0043 §2) — the published MeshGraphNets split
#: verbatim; ids follow the converter naming. Changing it is a new benchmark
#: version.
TRAIN: list[str] = [f"train_{i:04d}" for i in range(1000)]
VAL: list[str] = [f"val_{i:04d}" for i in range(100)]
TEST: list[str] = [f"test_{i:04d}" for i in range(100)]

#: Nodal auxiliary target (ADR-0043 §5): stored per-node von Mises stress.
AUX_FIELD = "von_mises_stress"

#: Node-type codes prescribed from ground truth and excluded from scoring
#: (ADR-0043 §4): OBSTACLE = 1 (scripted actuator), HANDLE = 3 (fixed).
KINEMATIC_TYPES = (1, 3)

QOIS: dict[str, QoiFn] = {
    "peak_vm_stress": peak_nodal_aux(exclude_types=KINEMATIC_TYPES),
    "terminal_peak_deflection": terminal_peak_displacement(
        exclude_types=KINEMATIC_TYPES
    ),
}
