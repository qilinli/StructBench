"""The wave-1d benchmark: split, aux field, QoIs (ADR-0025)."""

from __future__ import annotations

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
