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
    init:
        First scored frame (the protocol's ``input_frames``, ADR-0035).
        Frames before it are the ground-truth-observed prefix; QoIs that scan
        over time should read ``[init:]``, while frame-0 geometry (gauge
        positions, reference spans) remains available.
    """

    time: NDArray[np.float64]
    positions: NDArray[np.float32]
    aux: NDArray[np.float32]
    particle_type: NDArray[np.int64] | None = None
    init: int = 0


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
        Arrays of shape ``(T, P)`` or ``(T, P, C)`` (ADR-0059 channel
        blocks); the mean pools every non-time axis, so ``C = 1`` reproduces
        the scalar-field value exactly. Pooling across ``C > 1`` channels
        mixes units — per-channel reads slice the channel axis first.
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
    return np.sqrt(d.mean(axis=tuple(range(1, d.ndim))))


def relative_l2(
    pred_field: NDArray,
    gt_field: NDArray,
    mask: NDArray[np.bool_] | None = None,
    eps: float = 1e-8,
) -> NDArray[np.float64]:
    """Per-frame relative L2 error of a per-particle field (ADR-0055).

    Relative L2 is ``‖û − u‖₂ / max(‖u‖₂, ε)`` over the scored particles, one
    value per frame. This **per-frame** form is a reference utility, NOT a
    reported metric: the reported relative L2 is the pooled space+time
    :func:`relative_l2_pooled` (ADR-0055 pooled follow-up, 2026-08-16). The
    per-frame form is retained because it makes the fragility that motivates
    pooling explicit — dividing each frame by its own reference norm blows up on
    a field that starts at ~0 (e.g. von Mises stress before impact), which the
    pooled denominator (the whole-trajectory norm) avoids. See the metrics test
    contrasting the two.

    Parameters
    ----------
    pred_field, gt_field:
        Arrays of shape ``(T, P)`` (a scalar field, e.g. the aux field) or
        ``(T, P, dim)`` (a vector field, e.g. displacement). The L2 norms run
        over every non-time axis, so a vector field is normed over particles
        and components jointly.
    mask:
        Optional boolean particle mask ``(P,)`` (``True`` keeps the particle);
        when given, both norms run over kept particles only — the same
        kinematic/scripted exclusion the RMSE metrics apply (ADR-0026), so the
        two read over an identical particle set.
    eps:
        Floor on the reference norm ``‖u‖₂`` (default ``1e-8``); guards a
        near-static frame (e.g. displacement ~0 just after seeding) against
        division by zero (ADR-0055 §4).

    Returns
    -------
    numpy.ndarray
        Shape ``(T,)``, dimensionless — one relative L2 value per frame.
    """
    pred = np.asarray(pred_field, float)
    gt = np.asarray(gt_field, float)
    if mask is not None:
        pred = pred[:, mask]
        gt = gt[:, mask]
    axes = tuple(range(1, gt.ndim))  # all but the time axis
    err = np.sqrt(((pred - gt) ** 2).sum(axis=axes))
    ref = np.sqrt((gt**2).sum(axis=axes))
    return err / np.maximum(ref, eps)


def relative_l2_pooled(
    pred_field: NDArray,
    gt_field: NDArray,
    mask: NDArray[np.bool_] | None = None,
    eps: float = 1e-12,
) -> float:
    """Pooled space+time relative L2 of a field over one trajectory (ADR-0055).

    One scalar ratio per trajectory per quantity: the *whole scored rollout* of
    the quantity is flattened into a single vector — pooling frames × particles ×
    (the quantity's commensurate vector components) — and

        ``‖pred − gt‖₂ / max(‖gt‖₂, eps)``

    is taken over that pooled vector. This is the aggregation the Transolver
    family actually reports (thuml ``TestLoss.rel`` and, for time-dependent
    benchmarks, ``exp_plas.py`` / ``exp_ns.py`` ``test_l2_full`` — the whole
    concatenated trajectory flattened per sample, ``/ntest``; GeoTransolver's
    ``ε_L2`` over the predicted spatiotemporal response). It is the **headline**
    relative-L2 aggregation (ADR-0055 follow-up amendment, 2026-08-16),
    superseding the per-frame mean of :func:`relative_l2`.

    Unlike the per-frame form, this is **robust to near-zero frames**: the
    denominator is the whole trajectory's field energy, so a field that starts at
    ~0 (e.g. von Mises stress before impact) cannot drive the reference norm to
    zero the way a single early frame does. The per-frame mean divides a real
    single-frame error by a near-zero single-frame reference and explodes (Taylor
    ``rollout_rel_l2_aux`` came out ~5.7×10⁸); pooling removes that pathology,
    which is why the per-frame form is retained only as a secondary metric.

    Quantities are **not** merged: displacement (mm) and aux (MPa / strain) are
    incommensurable, so each is its own pooled ratio (pooling mm with MPa would
    let the larger-magnitude field swamp the other). Within displacement the
    ``{x, y[, z]}`` components share a unit and *are* pooled.

    Parameters
    ----------
    pred_field, gt_field:
        Arrays of shape ``(T, P)`` (a scalar field, e.g. the aux field) or
        ``(T, P, dim)`` (a vector field, e.g. displacement). The L2 norms run
        over *every* axis (time, particles, and any components jointly).
    mask:
        Optional boolean particle mask ``(P,)`` (``True`` keeps the particle);
        when given, both norms run over kept particles only — the same
        kinematic/scripted exclusion the RMSE metrics apply (ADR-0026).
    eps:
        Pure exact-zero guard on the pooled reference norm (default ``1e-12``);
        it prevents division by zero on a degenerate all-zero fixture only and is
        **not** a scale knob (the pooled denominator is the whole trajectory's
        field energy and cannot be driven near zero by an early ~0 frame).

    Returns
    -------
    float
        One dimensionless pooled relative-L2 ratio for the trajectory.
    """
    pred = np.asarray(pred_field, float)
    gt = np.asarray(gt_field, float)
    if mask is not None:
        pred = pred[:, mask]
        gt = gt[:, mask]
    err = float(np.sqrt(((pred - gt) ** 2).sum()))
    ref = float(np.sqrt((gt**2).sum()))
    return err / max(ref, eps)


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


def peak_mean_aux(inputs: QoiInputs) -> float:
    """Peak of the particle-mean auxiliary field over the trajectory (ADR-0032).

    A temporal-fidelity QoI: the particle-mean aux (von Mises for Taylor)
    peaks mid-trajectory and relaxes, so a surrogate that only matches end
    states — or blurs through transients at coarse internal time steps —
    misses it. The mean over particles keeps the value robust against
    single-particle outliers.

    Parameters
    ----------
    inputs:
        Rollout inputs; ``aux`` is read.

    Returns
    -------
    float
        Maximum over scored frames (``inputs.init`` onward) of the per-frame
        particle-mean aux, in the card's working aux unit.
    """
    mean_aux = np.asarray(inputs.aux, float)[inputs.init :].mean(axis=1)
    return float(mean_aux.max())


def t_peak_mean_aux(inputs: QoiInputs) -> float:
    """Time of the particle-mean aux peak, milliseconds (ADR-0032).

    Companion of :func:`peak_mean_aux`: getting the peak value right at the
    wrong time is still a temporal-resolution failure.

    Parameters
    ----------
    inputs:
        Rollout inputs; ``aux`` and ``time`` are read.

    Returns
    -------
    float
        Time of the peak frame in milliseconds (the ``arrival_time``
        convention).
    """
    mean_aux = np.asarray(inputs.aux, float)[inputs.init :].mean(axis=1)
    return float(inputs.time[inputs.init + int(mean_aux.argmax())] * 1e3)


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
        x0 = np.asarray(inputs.positions, float)[0, :, 0]  # frame-0 gauge geometry
        gauge_x = x0.min() + station_frac * (x0.max() - x0.min())
        gauge = int(np.argmin(np.abs(x0 - gauge_x)))
        # Scan only the scored span; the ground-truth-seeded prefix must not
        # decide arrival, and the self-referenced peak is over the same span.
        scored = np.abs(np.asarray(inputs.aux, float)[inputs.init :])
        signal = scored[:, gauge]
        peak = float(scored.max())
        if peak == 0.0:
            return float(inputs.time[-1] * 1e3)
        hits = np.nonzero(signal >= threshold_frac * peak)[0]
        frame = inputs.init + int(hits[0]) if hits.size else -1
        return float(inputs.time[frame] * 1e3)

    return qoi


def peak_stress(inputs: QoiInputs) -> float:
    """Peak ``|aux|`` over the second half of the trajectory (working unit).

    The late window is the reflection regime: it tests whether a surrogate
    sustains the correct wave amplitude through repeated traversals. The
    second-half restriction was decided (maintainer, 2026-07-03) under the
    pre-ADR-0032 protocol, where the onset peak fell inside the
    ground-truth-seeded frames; under input_frames = 6 the onset is predicted
    too, but the late window remains the discriminative regime.
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
        # Reference is frame-0 geometry; the peak excursion is scored-span only
        # (the seeded prefix must not leak into the QoI, ADR-0032 §4).
        return float(np.max(y[0] - y[inputs.init :]))

    return qoi


def cracked_fraction(
    threshold: float = 0.01, concrete_type: int | None = None
) -> QoiFn:
    """QoI factory: final-frame fraction of particles past the crack threshold.

    Operates on the max-principal-strain auxiliary field (ADR-0029). The
    default ``threshold=0.01`` (1% principal strain) is a **declared
    protocol definition**, not an approximation of a solver constant: the
    source SPH simulations use no erosion and no crack criterion, so any
    crack count draws a line on a continuous field (ADR-0029 amendment,
    2026-08-06). A 221-case sweep found no empirical knee; within the
    factor-2 band [0.005, 0.02] the ground-truth fraction shifts by
    ~0.03-0.05 mean per case, second-order against baseline QoI error and
    identical for every model under the shared definition. Changing it is
    a benchmark version change (ADR-0019 precedent).

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


def peak_nodal_aux(*, exclude_types: tuple[int, ...] = ()) -> QoiFn:
    """QoI factory: peak nodal aux value over the scored span (ADR-0043).

    The maximum is taken pointwise over kept nodes and scored frames, so it
    reads the single hottest node at its hottest scored frame — a stricter
    read than the particle-mean peak (:func:`peak_mean_aux`). This is the
    ADR-0043 ``peak_vm_stress`` QoI.

    Parameters
    ----------
    exclude_types:
        Part-ids to drop from the node set (e.g. kinematically prescribed
        boundary nodes). Ignored when ``inputs.particle_type`` is ``None``.

    Returns
    -------
    QoiFn
        Maps :class:`QoiInputs` to the peak aux value, in the card's
        working aux unit.
    """

    def qoi(inputs: QoiInputs) -> float:
        aux = inputs.aux[inputs.init :]
        if exclude_types and inputs.particle_type is not None:
            aux = aux[:, ~np.isin(inputs.particle_type, exclude_types)]
        return float(aux.max())

    return qoi


def terminal_peak_displacement(*, exclude_types: tuple[int, ...] = ()) -> QoiFn:
    """QoI factory: peak final-frame displacement magnitude, mm (ADR-0043).

    Displacement is measured node-by-node from the ground-truth initial
    frame (frame 0) to the final frame; the QoI is the maximum L2 norm over
    kept nodes. This is the ADR-0043 ``terminal_peak_deflection`` QoI.

    Parameters
    ----------
    exclude_types:
        Part-ids to drop from the node set (e.g. kinematically prescribed
        boundary nodes). Ignored when ``inputs.particle_type`` is ``None``.

    Returns
    -------
    QoiFn
        Maps :class:`QoiInputs` to the peak displacement magnitude (mm).
    """

    def qoi(inputs: QoiInputs) -> float:
        disp = np.linalg.norm(inputs.positions[-1] - inputs.positions[0], axis=-1)
        if exclude_types and inputs.particle_type is not None:
            disp = disp[~np.isin(inputs.particle_type, exclude_types)]
        return float(disp.max())

    return qoi
