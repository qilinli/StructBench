"""The v0.1 Taylor 2D impact benchmark: split, wall feature, QoIs (ADR-0019)."""

from __future__ import annotations

import torch

from ...eval import QoiFn, final_length, mushroom_width

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

#: ADR-0019 §5 quantities of interest: each maps a full ``(T, P, dim)``
#: trajectory (mm working frame) to a scalar read off the final frame.
QOIS: dict[str, QoiFn] = {
    "final_length": final_length,
    "mushroom_width": mushroom_width,
}

#: Rigidwall plane position in the model's mm working frame.
WALL_X_MM = -2.0


def wall_distance_feature(positions_mm: torch.Tensor, radius: float) -> torch.Tensor:
    """Per-particle distance to the rigidwall plane, clamped to ``[0, radius]``.

    Parameters
    ----------
    positions_mm:
        Current particle positions, shape ``(P, dim)``, in mm.
    radius:
        Connectivity radius (mm); distances are clamped to it.

    Returns
    -------
    torch.Tensor
        Shape ``(P, 1)``: ``clamp(x - WALL_X_MM, 0, radius)``.
    """
    return torch.clamp(positions_mm[:, 0:1] - WALL_X_MM, min=0.0, max=radius)
