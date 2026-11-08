"""Coverage of a scored case by the training design (ADR-0074 clause 2).

A scored case is read by where it lies relative to the training points, not
by its split's name: *within* the declared training box or *beyond* it, and
how far from the nearest training point in units of the training set's own
median nearest-neighbour spacing (on a full grid with equal normalised
steps, the step). An unseen categorical level or a changed constant is
*novelty* and is declared by hand, not computed here.

Pure functions over arrays; the plan of 2026-11-08 (open evaluation) fixes
the conventions: unit-cube coordinates from the box, fixed axes dropped from
the distance, inclusive edges, and the three distance bins.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["DISTANCE_BINS", "Coverage", "coverage", "coverage_bin", "median_spacing"]

#: Distance bins for rendering, in units of the training set's spacing:
#: ``(upper bound, label)``; the last bound is unbounded.
DISTANCE_BINS: tuple[tuple[float, str], ...] = (
    (1.0, "≤ 1"),
    (2.0, "1–2"),
    (float("inf"), "> 2"),
)

_EDGE_RTOL = 1e-9


@dataclass(frozen=True)
class Coverage:
    """One scored case's coverage.

    Parameters
    ----------
    within : bool
        Every numeric parameter lies in the training box, inclusive.
    distance : float
        Distance to the nearest training point in unit-cube coordinates,
        divided by the training set's median nearest-neighbour spacing.
    nearest : int
        Index of that training point.
    novelty : bool
        Declared by the benchmark: an unseen categorical level or a changed
        constant. A novel case is shown apart, never binned.
    """

    within: bool
    distance: float
    nearest: int
    novelty: bool = False


def _normalise(
    points: NDArray[np.float64], box: Sequence[tuple[float, float]]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Unit-cube coordinates over the varying axes; the mask of fixed axes."""
    lo = np.array([b[0] for b in box], dtype=float)
    hi = np.array([b[1] for b in box], dtype=float)
    if np.any(hi < lo):
        raise ValueError("box bounds must satisfy low <= high")
    width = hi - lo
    fixed = width == 0
    scaled = np.zeros_like(points, dtype=float)
    scaled[:, ~fixed] = (points[:, ~fixed] - lo[~fixed]) / width[~fixed]
    return scaled, fixed


def median_spacing(train_unit: NDArray[np.float64]) -> float:
    """The training set's median nearest-neighbour distance (unit cube)."""
    n = train_unit.shape[0]
    if n < 2:
        raise ValueError("a training design needs at least two points")
    d = np.linalg.norm(train_unit[:, None, :] - train_unit[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    unit = float(np.median(nearest))
    if not unit > 0:
        raise ValueError("training points coincide; the spacing is zero")
    return unit


def coverage(
    train: NDArray,
    scored: NDArray,
    box: Sequence[tuple[float, float]],
    *,
    novelty: Sequence[bool] | None = None,
) -> list[Coverage]:
    """Coverage of each scored case by the training design.

    Parameters
    ----------
    train : array, shape (n, k)
        The training points' numeric parameters.
    scored : array, shape (m, k)
        The scored cases' numeric parameters, same columns.
    box : sequence of (low, high)
        The declared training box per parameter. A zero-width entry is a
        parameter the design holds fixed: it is dropped from the distance,
        and a scored case that moves it is *beyond*.
    novelty : sequence of bool or None
        The declared novelty flag per scored case.

    Returns
    -------
    list of Coverage
        One per scored case, in order.

    Raises
    ------
    ValueError
        On mismatched shapes, a malformed box, fewer than two training
        points, or coincident training points.
    """
    tr = np.atleast_2d(np.asarray(train, dtype=float))
    sc = np.atleast_2d(np.asarray(scored, dtype=float))
    if tr.shape[1] != sc.shape[1] or tr.shape[1] != len(box):
        raise ValueError("train, scored and box must agree on the parameter count")
    if novelty is not None and len(novelty) != sc.shape[0]:
        raise ValueError("novelty flags must align with the scored cases")
    lo = np.array([b[0] for b in box], dtype=float)
    hi = np.array([b[1] for b in box], dtype=float)
    tr_u, fixed = _normalise(tr, box)
    sc_u, _ = _normalise(sc, box)
    unit = median_spacing(tr_u[:, ~fixed]) if (~fixed).any() else 1.0
    tol = _EDGE_RTOL * np.maximum(np.abs(lo), np.abs(hi), out=np.ones_like(lo)) + 1e-12
    out: list[Coverage] = []
    for i in range(sc.shape[0]):
        inside = bool(np.all(sc[i] >= lo - tol) and np.all(sc[i] <= hi + tol))
        if (~fixed).any():
            d = np.linalg.norm(tr_u[:, ~fixed] - sc_u[i, ~fixed], axis=1)
            j = int(np.argmin(d))
            distance = float(d[j] / unit)
        else:
            j, distance = 0, 0.0
        out.append(
            Coverage(
                within=inside,
                distance=distance,
                nearest=j,
                novelty=bool(novelty[i]) if novelty is not None else False,
            )
        )
    return out


def coverage_bin(c: Coverage) -> str:
    """The rendering bin: ``"novel"``, or ``"within, ≤ 1"`` and the like."""
    if c.novelty:
        return "novel"
    side = "within" if c.within else "beyond"
    for bound, label in DISTANCE_BINS:
        if c.distance <= bound:
            return f"{side}, {label}"
    raise AssertionError("unreachable: the last bin is unbounded")
