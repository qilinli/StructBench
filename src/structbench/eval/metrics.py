"""Rollout metrics and quantity-of-interest inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QoiInputs:
    """Arrays a quantity of interest may read (predicted or ground truth).

    Attributes
    ----------
    time:
        ``(T,)`` global time axis, seconds.
    positions:
        ``(T, P, dim)`` particle positions, working frame (mm).
    aux:
        ``(T, P)`` auxiliary field, working frame (the card's aux unit).
    """

    time: NDArray[np.float64]
    positions: NDArray[np.float32]
    aux: NDArray[np.float32]


#: A quantity of interest maps rollout arrays to one scalar.
QoiFn = Callable[[QoiInputs], float]


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


def final_length(inputs: QoiInputs) -> float:
    """x-extent of the final frame (ADR-0019 QoI; value unchanged).

    Parameters
    ----------
    inputs:
        Rollout inputs; only ``positions`` is read.

    Returns
    -------
    float
        ``x.max() - x.min()`` over particles in the final frame.
    """
    last = np.asarray(inputs.positions, float)[-1]
    x = last[:, 0]
    return float(x.max() - x.min())


def mushroom_width(inputs: QoiInputs) -> float:
    """y-extent of the final frame (ADR-0019 QoI; value unchanged).

    Parameters
    ----------
    inputs:
        Rollout inputs; only ``positions`` is read.

    Returns
    -------
    float
        ``y.max() - y.min()`` over particles in the final frame.
    """
    last = np.asarray(inputs.positions, float)[-1]
    y = last[:, 1]
    return float(y.max() - y.min())
