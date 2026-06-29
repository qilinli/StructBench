"""Rollout metrics for the Taylor benchmark (ADR-0019)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def position_rmse(pred: NDArray, true: NDArray) -> NDArray[np.float64]:
    """Per-frame position RMSE over particles and dimensions.

    Parameters
    ----------
    pred, true:
        Arrays of shape ``(T, P, dim)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(T,)``.
    """
    d = (np.asarray(pred, float) - np.asarray(true, float)) ** 2
    return np.sqrt(d.mean(axis=(1, 2)))


def field_rmse(pred: NDArray, true: NDArray) -> NDArray[np.float64]:
    """Per-frame RMSE of a scalar per-particle field, shapes ``(T, P)``."""
    d = (np.asarray(pred, float) - np.asarray(true, float)) ** 2
    return np.sqrt(d.mean(axis=1))


def final_length(positions: NDArray) -> float:
    """x-extent of the final frame. ``positions`` is ``(T, P, dim)``."""
    last = np.asarray(positions, float)[-1]
    x = last[:, 0]
    return float(x.max() - x.min())


def mushroom_width(positions: NDArray) -> float:
    """y-extent of the final frame. ``positions`` is ``(T, P, dim)``."""
    last = np.asarray(positions, float)[-1]
    y = last[:, 1]
    return float(y.max() - y.min())
