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
    particle_type:
        ``(P,)`` particle part-ids, when the caller provides them.
    """

    time: NDArray[np.float64]
    positions: NDArray[np.float32]
    aux: NDArray[np.float32]
    particle_type: NDArray[np.int64] | None = None


#: A quantity of interest maps rollout arrays to one scalar.
QoiFn = Callable[[QoiInputs], float]


def position_rmse(
    pred: NDArray, true: NDArray, keep: NDArray[np.bool_] | None = None
) -> NDArray[np.float64]:
    """Per-frame position RMSE over particles and dimensions.

    Parameters
    ----------
    pred, true:
        Arrays of shape ``(T, P, dim)``.
    keep:
        Optional boolean particle mask ``(P,)``; when given, the mean runs
        over kept particles only (e.g. excluding kinematically prescribed
        particles, ADR-0026).

    Returns
    -------
    numpy.ndarray
        Shape ``(T,)``.
    """
    d = (np.asarray(pred, float) - np.asarray(true, float)) ** 2
    if keep is not None:
        d = d[:, keep, :]
    return np.sqrt(d.mean(axis=(1, 2)))


def field_rmse(
    pred: NDArray, true: NDArray, keep: NDArray[np.bool_] | None = None
) -> NDArray[np.float64]:
    """Per-frame RMSE of a scalar per-particle field, shapes ``(T, P)``.

    Parameters
    ----------
    pred, true:
        Arrays of shape ``(T, P)``.
    keep:
        Optional boolean particle mask ``(P,)``; when given, the mean runs
        over kept particles only (e.g. excluding kinematically prescribed
        particles, ADR-0026).

    Returns
    -------
    numpy.ndarray
        Shape ``(T,)``.
    """
    d = (np.asarray(pred, float) - np.asarray(true, float)) ** 2
    if keep is not None:
        d = d[:, keep]
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


def arrival_time(station_frac: float, *, threshold_frac: float = 0.1) -> QoiFn:
    """QoI factory: wave-front arrival time at a gauge station, milliseconds.

    The gauge is the particle nearest to the fractional position
    ``station_frac`` along the frame-0 x-extent of the bar. Arrival is the
    first frame where the gauge's ``|aux|`` reaches ``threshold_frac`` of
    that trajectory's own peak ``|aux|`` (self-referenced so predicted and
    ground-truth trajectories are judged by the same rule). If the signal
    never crosses (e.g. an all-zero field), the final time is returned —
    a saturating "never arrived" value rather than NaN.

    Parameters
    ----------
    station_frac:
        Fractional gauge position along the bar, in ``[0, 1]``.
    threshold_frac:
        Arrival threshold as a fraction of the trajectory's peak ``|aux|``.

    Returns
    -------
    QoiFn
        Maps :class:`QoiInputs` to the arrival time in milliseconds.
    """

    def qoi(inputs: QoiInputs) -> float:
        x0 = np.asarray(inputs.positions, float)[0, :, 0]
        gauge_x = x0.min() + station_frac * (x0.max() - x0.min())
        gauge = int(np.argmin(np.abs(x0 - gauge_x)))
        signal = np.abs(np.asarray(inputs.aux, float)[:, gauge])
        peak = float(np.abs(np.asarray(inputs.aux, float)).max())
        if peak == 0.0:
            return float(inputs.time[-1] * 1e3)
        hits = np.nonzero(signal >= threshold_frac * peak)[0]
        frame = int(hits[0]) if hits.size else -1
        return float(inputs.time[frame] * 1e3)

    return qoi


def peak_stress(inputs: QoiInputs) -> float:
    """Peak ``|aux|`` over the second half of the trajectory (working unit).

    The late window is the reflection regime: it tests whether a surrogate
    sustains the correct wave amplitude through repeated traversals. The
    early-time global peak sits at excitation onset, inside the frames a
    rollout seeds with ground truth, and would be trivially matched by any
    model (maintainer decision, 2026-07-03).
    """
    aux = np.abs(np.asarray(inputs.aux, float))
    return float(aux[aux.shape[0] // 2 :].max())
