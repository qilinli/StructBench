"""Array-level verification kernels (ADR-0066).

Pure numpy, frame-agnostic, unit-agnostic: callers keep units consistent.
They take plain arrays — for example ``(von_mises, state)``, not a stress
tensor — so that ``eval`` can later call the same functions on a model's
predicted fields. Reductions that may run over frame chunks return records
that merge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "RatioExtreme",
    "abs_max",
    "decrease_max",
    "energy_residual",
    "interp_yield_stress",
    "nearest_knot_distance",
    "nonfinite_count",
    "pressure_trace_residual",
    "relative_drift",
    "signed_distance_to_plane",
    "yield_ratio_max",
    "yielding_mask",
]


@dataclass(frozen=True)
class RatioExtreme:
    """The maximum of a ratio, where it occurred, and over how many samples."""

    value: float
    frame: int
    index: int
    n_samples: int

    def merge(self, other: RatioExtreme) -> RatioExtreme:
        """Combine two chunks; the earlier chunk wins a tie."""
        best = self if self.value >= other.value else other
        return RatioExtreme(
            best.value, best.frame, best.index, self.n_samples + other.n_samples
        )


def interp_yield_stress(
    state: ArrayLike, knots_state: Sequence[float], knots_sy: Sequence[float]
) -> NDArray[np.float64]:
    """Tabulated yield stress at ``state``: linear between knots, end-clamped.

    ``np.interp`` semantics, the contract ADR-0064 fixes for the hardening
    table. Knot values are used verbatim, so a non-monotone knot is honoured.
    """
    return np.interp(
        np.asarray(state, dtype=np.float64),
        np.asarray(knots_state, dtype=np.float64),
        np.asarray(knots_sy, dtype=np.float64),
    )


def yield_ratio_max(
    von_mises: ArrayLike,
    state: ArrayLike,
    knots_state: Sequence[float],
    knots_sy: Sequence[float],
    *,
    frame_offset: int = 0,
) -> RatioExtreme:
    """Largest ``von_mises / yield_stress(state)`` over a ``(T, P)`` block.

    Parameters
    ----------
    von_mises, state : array_like, shape (T, P)
        Equivalent stress and the hardening variable, same unit as the table.
    knots_state, knots_sy : sequence of float
        The hardening table.
    frame_offset : int
        Frame index of the first row of the block, for chunked evaluation.
    """
    vm = np.asarray(von_mises, dtype=np.float64)
    ratio = vm / interp_yield_stress(state, knots_state, knots_sy)
    frame, index = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    return RatioExtreme(
        float(ratio[frame, index]), int(frame) + frame_offset, int(index), ratio.size
    )


def yielding_mask(state: ArrayLike) -> NDArray[np.bool_]:
    """Samples mid-way through plastic loading, shape ``(T, P)``.

    True at frame ``t`` where the state grows both into and out of ``t``.
    A point that yielded between two stored frames need not sit on the
    surface at either; one that is loading on both sides is the best
    available witness. First and last frames are never marked.
    """
    s = np.asarray(state, dtype=np.float64)
    mask = np.zeros(s.shape, dtype=bool)
    if s.shape[0] >= 3:
        growing = np.diff(s, axis=0) > 0.0
        mask[1:-1] = growing[:-1] & growing[1:]
    return mask


def decrease_max(x: ArrayLike) -> tuple[float, int]:
    """Largest backward step along axis 0, and how many steps went backward."""
    steps = np.diff(np.asarray(x, dtype=np.float64), axis=0)
    backward = steps < 0.0
    largest = float(-steps[backward].min()) if backward.any() else 0.0
    return largest, int(backward.sum())


def nonfinite_count(a: ArrayLike) -> int:
    """Number of NaN or infinite entries."""
    return int((~np.isfinite(np.asarray(a, dtype=np.float64))).sum())


def abs_max(a: ArrayLike) -> float:
    """Largest magnitude."""
    return float(np.abs(np.asarray(a, dtype=np.float64)).max())


def relative_drift(series: ArrayLike) -> float:
    """``max |s(t) / s(0) - 1|`` of a 1-D series.

    Raises
    ------
    ValueError
        If the first sample is zero.
    """
    s = np.asarray(series, dtype=np.float64)
    if s[0] == 0.0:
        raise ValueError("relative_drift needs a non-zero first sample")
    return float(np.abs(s / s[0] - 1.0).max())


def nearest_knot_distance(value: float, knots: Sequence[float]) -> float:
    """Distance from ``value`` to the nearest table knot."""
    return float(np.abs(np.asarray(knots, dtype=np.float64) - value).min())


def pressure_trace_residual(stress_voigt: ArrayLike, pressure: ArrayLike) -> float:
    """``max |p + tr(sigma) / 3| / max |sigma|``; pressure positive in compression.

    Parameters
    ----------
    stress_voigt : array_like, shape (..., 6)
        Voigt order ``xx, yy, zz, xy, yz, zx``.
    pressure : array_like, shape (...)
    """
    sigma = np.asarray(stress_voigt, dtype=np.float64)
    trace = sigma[..., 0] + sigma[..., 1] + sigma[..., 2]
    residual = np.abs(np.asarray(pressure, dtype=np.float64) + trace / 3.0).max()
    return float(residual / np.abs(sigma).max())


def signed_distance_to_plane(
    positions: ArrayLike,
    point: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Signed distance of positions ``(..., dim)`` to a plane; positive = admissible.

    Uses the first ``dim`` components of ``point`` and ``normal``, so a
    two-dimensional case works against a three-component plane.
    """
    x = np.asarray(positions, dtype=np.float64)
    dim = x.shape[-1]
    p = np.asarray(point[:dim], dtype=np.float64)
    n = np.asarray(normal[:dim], dtype=np.float64)
    return np.asarray((x - p) @ n, dtype=np.float64)


def energy_residual(
    total: ArrayLike, external_work: ArrayLike, kinetic: ArrayLike
) -> NDArray[np.float64]:
    """The signed energy indicator ``r(t)`` of ADR-0066 clause 7.

    ::

        R(t) = [E_tot(t) - E_tot(t0)] - [W_ext(t) - W_ext(t0)]
        r(t) = R(t) / max(|E_tot(t0) + W_ext(t) - W_ext(t0)|,
                          E_kin(t), E_tot(t) - E_kin(t))

    ``r > 0`` is energy created, ``r < 0`` energy unaccounted for. Energy
    present at the first sample counts as input, so an initial-velocity
    impact reads zero, not one, at the start. Where every scale is zero
    nothing has happened and ``r`` is zero.

    Parameters
    ----------
    total, external_work, kinetic : array_like, shape (T,)
        The identity's total, the external work, and the kinetic term, in
        one energy unit.
    """
    e, w, k = (np.asarray(a, dtype=np.float64) for a in (total, external_work, kinetic))
    work = w - w[0]
    residual = (e - e[0]) - work
    scale = np.maximum.reduce([np.abs(e[0] + work), k, e - k])
    return np.divide(residual, scale, out=np.zeros_like(residual), where=scale > 0.0)
