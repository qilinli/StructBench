"""The restart task family (ADR-0075 clause 5; plan 3b of the two-site design).

A restart task supplies the stored two-site state at an anchor frame and
scores the prediction over a horizon. A restart is the existing rollout on a
*window* of the trajectory: the ``input_frames`` frames up to and including
the anchor are the seed, the ``horizon`` frames after it are predicted and
all of them are scored. The window carries the element-site state block, so
a two-site simulator restarts from the stored state; every other scheme
restarts from the kinematic anchor alone and its rows are marked
``restart_kind = "kinematic"``.

This module holds the pieces that need no simulator: the window, the metric
keys of the restart and composition rows, and the semigroup error between a
direct map and its composition. The rollouts themselves are
:func:`~structbench.eval.rollout.rollout` and
:func:`~structbench.eval.rollout.flow_map_rollout` on the window, driven by
``structbench-train --mode restart``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import ArrayLike

from ..datasets.canonical import CaseTrajectory
from .metrics import relative_l2_pooled

__all__ = [
    "RESTART_FIELD_METRICS",
    "RESTART_KINDS",
    "RESTART_STATE_METRICS",
    "compose_keys",
    "restart_keys",
    "restart_window",
    "semigroup_error",
]

#: The rollout's field metrics every restart row carries, and the two the
#: element-site state adds on a two-site run (``RolloutResult.mean_<name>``).
RESTART_FIELD_METRICS: tuple[str, ...] = (
    "rel_l2_displacement",
    "rel_l2_aux",
    "position_rmse",
    "aux_rmse",
)
RESTART_STATE_METRICS: tuple[str, ...] = ("rel_l2_state", "state_rmse")

#: What a result restarted from: nothing declared, the stored two-site state,
#: or the kinematic anchor alone (ADR-0075 clause 5).
RESTART_KINDS: tuple[str, ...] = ("", "state", "kinematic")


def restart_window(
    trajectory: CaseTrajectory, anchor: int, horizon: int, input_frames: int
) -> CaseTrajectory:
    """The frames ``[anchor − input_frames + 1, anchor + horizon]`` of a case.

    Frame ``input_frames − 1`` of the window is the anchor, so a rollout
    seeded with ``input_frames`` frames predicts exactly the ``horizon``
    frames after it. Per-frame arrays (positions, aux, time, state) are
    sliced; the static ones (particle types, cells, reference coordinates)
    are shared.

    Raises
    ------
    ValueError
        If the anchor has fewer than ``input_frames − 1`` frames before it,
        the horizon is not positive, or ``anchor + horizon`` runs past the
        trajectory's last frame.
    """
    n_frames = trajectory.positions.shape[0]
    start = anchor - input_frames + 1
    if start < 0:
        raise ValueError(
            f"anchor {anchor} has no history of {input_frames - 1} frames before it"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    stop = anchor + horizon + 1
    if stop > n_frames:
        raise ValueError(
            f"anchor {anchor} + horizon {horizon} runs past the trajectory's last "
            f"frame {n_frames - 1}"
        )
    return replace(
        trajectory,
        positions=trajectory.positions[start:stop],
        aux=trajectory.aux[start:stop],
        time=trajectory.time[start:stop],
        state=None if trajectory.state is None else trajectory.state[start:stop],
    )


def restart_keys(anchor: int, horizon: int, metrics: Sequence[str]) -> tuple[str, ...]:
    """``restart_a{anchor}_h{horizon}_<metric>`` for each metric name."""
    return tuple(f"restart_a{anchor}_h{horizon}_{m}" for m in metrics)


def compose_keys(horizon: int, k: int, metrics: Sequence[str]) -> tuple[str, ...]:
    """``compose_T{horizon}_k{k}_<metric>``: the interval ``horizon`` reached in
    ``k`` compositions."""
    return tuple(f"compose_T{horizon}_k{k}_{m}" for m in metrics)


def semigroup_error(
    direct: ArrayLike, composed: ArrayLike, reference: ArrayLike | None = None
) -> float:
    """``‖Φ_t(Φ_s(x)) − Φ_{s+t}(x)‖₂ / ‖Φ_{s+t}(x)‖₂`` over the composed frames.

    The pooled relative L2 (ADR-0055's form) of the composed map against the
    direct map, both ``(n, P, dim)`` positions over the same frames. With
    ``reference`` (``(P, dim)``, the case's frame-0 positions, the house
    convention) both are read as displacements from it, so the ratio is not
    origin noise (ADR-0055); the caller passes the frames after the first
    re-anchor only, where the two maps can differ (design §6). ``nan`` when
    the direct map's displacement is all zero.

    Parameters
    ----------
    direct, composed : array_like, shape (n, P, dim)
        Predicted positions of the one-step and the composed map.
    reference : array_like, shape (P, dim), optional
        The positions displacements are measured from.

    Returns
    -------
    float
    """
    d = np.asarray(direct, dtype=np.float64)
    c = np.asarray(composed, dtype=np.float64)
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)[None]
        d, c = d - ref, c - ref
    if not np.any(d):
        return float("nan")
    return float(relative_l2_pooled(c, d))
