"""The properties block (ADR-0075 clause 6; plan 3c of the two-site design).

Every registered result reports, where the benchmark defines them, a group of
properties rendered before its accuracy: whether the prediction honours the
constraints the reference data honours (the yield bound and the monotone
plastic strain at the state site), how much of a matched pair's divergence
the stored state recovers, how the free rollout's error relates to the
teacher-forced one-step error (closure), and how fast the error grows along
the horizon. Each property has one kernel here, pure numpy over the rollout
record; where ``structbench.verification.kernels`` already judges the
reference data by the same quantity, the kernel is called so that the data
and a prediction meet the same function. No property gates anything.

A kernel returns ``nan`` where its property is undefined on the input (a
zero denominator, too few frames); the metrics writer records ``null``.
Rates are fractions in ``[0, 1]``; the closure ratio and the sufficiency are
dimensionless ratios; the error growth is per predicted frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from structbench.verification.kernels import decrease_max, interp_yield_stress

__all__ = [
    "PROPERTY_KEYS",
    "closure_ratio",
    "error_growth",
    "peeq_decrease_rate",
    "state_sufficiency",
    "von_mises4",
    "yield_violation_rate",
]

#: The block's metric keys in render order (the design's §7 table, minus the
#: deferred deviator trace and part 3b's semigroup row). ``prop_`` marks a key
#: as a property wherever metrics are partitioned.
PROPERTY_KEYS: tuple[str, ...] = (
    "prop_closure_ratio",
    "prop_error_growth",
    "prop_yield_violation_rate",
    "prop_peeq_decrease_rate",
    "prop_state_sufficiency",
)


def closure_ratio(rollout_err: ArrayLike, one_step_err: ArrayLike) -> float:
    """Free-rollout error over teacher-forced one-step error on one span.

    The ratio of the two sums over the scored span (not a mean of per-frame
    ratios, which a near-zero one-step frame would dominate). ``1`` means
    the rollout accumulates nothing beyond its one-step error; ``nan`` when
    the one-step sum is zero.

    Parameters
    ----------
    rollout_err, one_step_err : array_like, shape (n,)
        Per-frame errors over the same scored frames, same unit.
    """
    r = np.asarray(rollout_err, dtype=np.float64)
    s = np.asarray(one_step_err, dtype=np.float64)
    denominator = float(s.sum())
    return float(r.sum() / denominator) if denominator > 0.0 else float("nan")


def error_growth(err: ArrayLike) -> float:
    """Least-squares slope of ``log(err)`` against the frame index, per frame.

    Frames with a non-positive error (a frame seeded exactly) are dropped;
    fewer than two remaining frames give ``nan``.

    Parameters
    ----------
    err : array_like, shape (n,)
        Per-frame error over the scored span, in frame order.
    """
    e = np.asarray(err, dtype=np.float64)
    t = np.arange(e.shape[0], dtype=np.float64)
    keep = e > 0.0
    if keep.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(t[keep], np.log(e[keep]), 1)
    return float(slope)


def von_mises4(stress: ArrayLike) -> NDArray[np.float64]:
    """Von Mises stress of ``(..., 4)`` ``(s11, s22, s33, s12)`` components.

    The numpy twin of :func:`structbench.models.common.two_site.von_mises4`,
    for the element-site layout
    :data:`~structbench.benchmarks.constitutive.RETURN_MAP_STATE_CHANNELS`.
    """
    s = np.asarray(stress, dtype=np.float64)
    s11, s22, s33, s12 = (s[..., i] for i in range(4))
    return np.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * s12**2
    )


def yield_violation_rate(
    von_mises: ArrayLike,
    peeq: ArrayLike,
    knots_peeq: Sequence[float],
    knots_sy: Sequence[float],
    tol: float,
) -> float:
    """Fraction of ``(frame, element)`` samples with ``vm > (1 + tol) σ_y(peeq)``.

    The yield stress is the case's own table through
    :func:`~structbench.verification.kernels.interp_yield_stress`, the
    function the reference data's yield-ratio row uses. ``tol`` is the
    benchmark's declared tolerance, the reference's own measured excess.

    Parameters
    ----------
    von_mises, peeq : array_like, shape (T, E)
    knots_peeq, knots_sy : sequence of float
        The hardening table.
    tol : float
        Relative tolerance above the surface, non-negative.
    """
    vm = np.asarray(von_mises, dtype=np.float64)
    sy = interp_yield_stress(peeq, knots_peeq, knots_sy)
    return float((vm > (1.0 + tol) * sy).mean())


def peeq_decrease_rate(peeq: ArrayLike, tol: float) -> float:
    """Fraction of element steps whose plastic strain falls by more than ``tol``.

    Counts with :func:`~structbench.verification.kernels.decrease_max`'s
    backward-step count over the block shifted by ``tol``, so a drop within
    the tolerance is not a decrease.

    Parameters
    ----------
    peeq : array_like, shape (T, E)
    tol : float
        Absolute tolerance on a backward step, non-negative.
    """
    p = np.asarray(peeq, dtype=np.float64)
    if p.shape[0] < 2:
        return float("nan")
    # A step d counts when d < -tol, i.e. when the step of (p + tol·t) is
    # negative: shifting each frame by tol turns the tolerance into a sign.
    shifted = p + tol * np.arange(p.shape[0], dtype=np.float64)[:, None]
    _, count = decrease_max(shifted)
    n_steps = (p.shape[0] - 1) * p.shape[1]
    return float(count / n_steps)


def state_sufficiency(
    pred_a: ArrayLike, pred_b: ArrayLike, true_a: ArrayLike, true_b: ArrayLike
) -> float:
    """Fraction of a matched pair's true divergence the predictions recover.

    ``‖X̂_a − X̂_b‖₂ / ‖X_a − X_b‖₂`` pooled over the scored span: ``1`` when
    the two rollouts diverge as the reference pair does, ``0`` when a model
    that reads no state predicts the same trajectory for both. ``nan`` when
    the reference pair does not diverge.

    Parameters
    ----------
    pred_a, pred_b, true_a, true_b : array_like, shape (T, P, dim)
        Predicted and stored positions of the pair over the scored frames.
    """
    d_pred = np.asarray(pred_a, dtype=np.float64) - np.asarray(pred_b, dtype=np.float64)
    d_true = np.asarray(true_a, dtype=np.float64) - np.asarray(true_b, dtype=np.float64)
    denominator = float(np.linalg.norm(d_true.ravel()))
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(d_pred.ravel()) / denominator)
