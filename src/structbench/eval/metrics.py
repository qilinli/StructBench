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


def midspan_deflection_peak(
    gauge_halfwidth: float = 5.0, concrete_type: int | None = None
) -> QoiFn:
    """QoI factory: peak downward mid-span deflection, mm (ADR-0026).

    The gauge is the set of particles within ``gauge_halfwidth`` of the
    frame-0 x-midspan (optionally restricted to ``concrete_type``
    particles). Deflection is the gauge's mean y-displacement from frame 0;
    the QoI is its peak downward excursion over the trajectory.

    Parameters
    ----------
    gauge_halfwidth:
        Half-width of the mid-span gauge window, mm.
    concrete_type:
        When given and ``inputs.particle_type`` is present, only particles
        of this part-id form the gauge.

    Returns
    -------
    QoiFn
        Maps :class:`QoiInputs` to the peak downward deflection (mm).
    """

    def qoi(inputs: QoiInputs) -> float:
        pos = np.asarray(inputs.positions, float)
        x0 = pos[0, :, 0]
        mid = 0.5 * (x0.min() + x0.max())
        gauge = np.abs(x0 - mid) <= gauge_halfwidth
        if concrete_type is not None and inputs.particle_type is not None:
            gauge &= inputs.particle_type == concrete_type
        y = pos[:, gauge, 1].mean(axis=1)
        return float(np.max(y[0] - y))

    return qoi


def cracked_fraction(
    threshold: float = 0.01, concrete_type: int | None = None
) -> QoiFn:
    """QoI factory: final-frame fraction of particles past the crack threshold.

    Operates on the max-principal-strain auxiliary field (ADR-0029). The
    default ``threshold=0.01`` (1% principal strain) is **provisional**
    (maintainer, 2026-07-04): it sits clearly beyond the elastic band in
    the ingested data (median ~3e-4, p90 ~1e-2 in a damaged bend case),
    but the crack criterion has not been validated against the prior
    study and may be revised before the first trained leaderboard
    entries. Changing it is a benchmark version change (ADR-0019
    precedent).

    Parameters
    ----------
    threshold:
        Principal-strain level counted as cracked.
    concrete_type:
        When given and ``inputs.particle_type`` is present, the fraction
        runs over that part-id's particles only.

    Returns
    -------
    QoiFn
        Maps :class:`QoiInputs` to a fraction in ``[0, 1]``.
    """

    def qoi(inputs: QoiInputs) -> float:
        strain = np.asarray(inputs.aux, float)[-1]
        if concrete_type is not None and inputs.particle_type is not None:
            strain = strain[inputs.particle_type == concrete_type]
        if strain.size == 0:
            return 0.0
        return float((strain >= threshold).mean())

    return qoi
