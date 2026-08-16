"""Autoregressive rollout and one-step evaluation of a learned simulator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from ..datasets.canonical import CaseTrajectory
from .metrics import (
    QoiFn,
    QoiInputs,
    field_rmse,
    position_rmse,
    relative_l2,
    relative_l2_pooled,
)


class _SimulatorLike(Protocol):
    """Structural type for the simulator consumed by :func:`rollout`."""

    def predict_positions(
        self,
        position_sequence: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        particle_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(next_positions (P,dim), aux (P,n_aux))``.

        A k-frames-per-call simulator (ADR-0050/0051) instead returns a
        ``(P, k, dim)`` / ``(P, k, n_aux)`` bundle; :func:`rollout` and the
        one-step sweeps dispatch on the returned rank, so k=1 simulators (every
        non-Transolver family) need no change. A k>1 simulator additionally
        accepts a ``teacher_forced`` keyword the one-step sweeps pass it.
        """
        ...


@dataclass
class RolloutResult:
    """Predicted trajectory with per-step, cumulative, and QoI errors.

    Per-step arrays cover the ``T - input_frames`` predicted frames (the model
    observes the first ``input_frames`` ground-truth frames and predicts the
    rest, ADR-0035) and always run to the trajectory end as the long-horizon
    diagnostic. The cumulative means aggregate the *scored* span — every
    predicted frame, or ``[input_frames, scored_frames)`` when the benchmark
    pins a scored horizon (ADR-0039). QoI dicts are filled
    only when :func:`rollout` is given a ``qois`` mapping (``qoi_error`` is
    signed, predicted − true); QoIs see only the scored span's frames. Units
    follow the trajectory's working frame (mm and MPa for Taylor 2D). The
    seeded frames of ``predicted_aux`` carry
    ground-truth aux values, mirroring the seeded positions; kinematic
    particles carry zero aux on every frame.
    """

    predicted_positions: NDArray[np.float32]  # (T, P, dim)
    predicted_aux: NDArray[np.float32]  # (T, P)
    position_rmse: NDArray[np.float64]  # (nsteps,)
    aux_rmse: NDArray[np.float64]  # (nsteps,)
    mean_position_rmse: float = float("nan")
    mean_aux_rmse: float = float("nan")
    # Relative-L2 companions of the two RMSE means (ADR-0055). The HEADLINE
    # aggregation is pooled space+time per trajectory (ADR-0055 follow-up
    # amendment, 2026-08-16): the whole scored rollout of the quantity flattened
    # into one vector, ‖pred−gt‖₂/‖gt‖₂ — robust to near-zero frames, matching
    # the Transolver family's test_l2_full. Displacement relative L2 is on
    # u = pos - pos[0] (frame-0 reference, decision A); aux relative L2 is on the
    # raw aux field. Same scored horizon and kinematic mask as the RMSEs.
    mean_rel_l2_displacement: float = float("nan")
    mean_rel_l2_aux: float = float("nan")
    # Per-frame-mean relative L2 (mean over scored frames of the per-frame
    # ‖err‖/‖gt‖), retained as a SECONDARY metric for comparability with the
    # GeoFLARE / PhysicsNeMo crash line (per-timestep error). Never the headline:
    # it is the unguarded quantity that blows up on zero-start fields.
    mean_rel_l2_displacement_perframe: float = float("nan")
    mean_rel_l2_aux_perframe: float = float("nan")
    qoi_pred: dict[str, float] = field(default_factory=dict)
    qoi_true: dict[str, float] = field(default_factory=dict)
    qoi_error: dict[str, float] = field(default_factory=dict)


def rollout(
    simulator: _SimulatorLike,
    trajectory: CaseTrajectory,
    input_frames: int,
    device: str = "cpu",
    qois: Mapping[str, QoiFn] | None = None,
    kinematic_types: tuple[int, ...] = (),
    scored_frames: int | None = None,
) -> RolloutResult:
    """Seed with the observed prefix, then autoregress to the end.

    The model observes exactly ``input_frames`` ground-truth frames (its full
    history window) and predicts every frame from ``input_frames`` onward; the
    scored span is ``[input_frames, T)`` (ADR-0035), narrowed to
    ``[input_frames, scored_frames)`` when a benchmark pins a scored horizon
    (ADR-0039). There is no history
    backfill: the observed prefix *is* the model's input window, so no rollout
    step is fed a fabricated (constant-velocity) history.

    Parameters
    ----------
    simulator:
        Object with
        ``predict_positions(position_sequence, nparticles_per_example,
        particle_types) -> (next_positions (P,dim), aux (P,n_aux))``.
    trajectory:
        Ground-truth :class:`CaseTrajectory`.
    input_frames:
        The model's history length in frames, which is also the number of
        ground-truth frames observed to seed the rollout (they are the same
        quantity under ADR-0035). Equal to the benchmark card's protocol.
    device:
        Torch device string.
    qois:
        Optional mapping of quantity-of-interest name to a function of a
        :class:`~structbench.eval.metrics.QoiInputs` (e.g. the Taylor
        benchmark's ``QOIS``).  Each is evaluated on the predicted and the
        ground-truth inputs and recorded with its signed error.
    kinematic_types:
        Particle part-ids whose motion is prescribed (ADR-0026).  At every
        autoregressive step their predicted positions are overwritten with the
        ground-truth position at that frame, and their ``predicted_aux`` is
        zeroed over all frames (no aux training signal reaches them, so the
        decoder's output there is meaningless).  They are also excluded from
        the reported ``position_rmse`` and ``aux_rmse`` (``keep`` mask).
        Defaults to ``()`` (no prescribed particles).
    scored_frames:
        Exclusive upper frame bound of the scored span (ADR-0039), mirroring
        the full-trajectory bound ``T``: ``mean_position_rmse``,
        ``mean_aux_rmse``, and the QoIs are computed over
        ``[input_frames, min(scored_frames, T))``, while the per-step arrays
        and predicted trajectories still run to ``T`` as the long-horizon
        diagnostic. ``None`` (default) scores every predicted frame.

    Returns
    -------
    RolloutResult

    Raises
    ------
    ValueError
        If ``input_frames < 2`` (no velocity can be formed) or
        ``input_frames >= T``.
    """
    pos = torch.from_numpy(trajectory.positions).to(device)  # (T, P, dim)
    n_frames, n_particles, _ = pos.shape
    ptype = torch.from_numpy(trajectory.particle_type).to(device)
    npp = torch.tensor([n_particles], device=device)

    # Kinematic-particle book-keeping (ADR-0026).
    kin_mask_np = np.isin(trajectory.particle_type, np.asarray(kinematic_types))
    keep: np.ndarray | None = ~kin_mask_np if kin_mask_np.any() else None
    kin_idx = torch.from_numpy(np.nonzero(kin_mask_np)[0]).to(device)

    if input_frames < 2:
        raise ValueError(f"input_frames must be >= 2, got {input_frames}")
    if input_frames >= n_frames:
        raise ValueError(
            f"input_frames={input_frames} but trajectory has {n_frames} frames"
        )
    seq = pos[:input_frames].clone()  # (input_frames, P, dim)
    predicted = [pos[i] for i in range(input_frames)]
    aux_true = torch.from_numpy(trajectory.aux).to(device)
    aux_pred = [aux_true[i] for i in range(input_frames)]

    # Rank-dispatch on the returned tensor (ADR-0050/0051): a 2-D (P, dim)
    # return is the autoregressive k=1 path (byte-identical, and what every
    # non-Transolver family returns); a 3-D (P, k, dim) return is a k-frame
    # bundle. The predicted[] / aux_pred[] lists end at exactly n_frames
    # entries either way, so RolloutResult's (T, P, dim)/(T, P) contract and
    # everything downstream stay scheme-agnostic.
    with torch.no_grad():
        t = input_frames
        while t < n_frames:
            seq_pw = seq.permute(1, 0, 2).contiguous()  # (P, input_frames, dim)
            next_pos, aux = simulator.predict_positions(seq_pw, npp, ptype)
            if next_pos.ndim == 2:
                # k=1 autoregressive step (byte-identical to the pre-0051 loop).
                if kin_idx.numel():
                    next_pos = next_pos.clone()
                    next_pos[kin_idx] = pos[t][kin_idx]
                predicted.append(next_pos)
                aux_pred.append(aux[:, 0])
                seq = torch.cat([seq[1:], next_pos[None]], dim=0)
                t += 1
            else:
                # k>1 bundle: predict-k-and-truncate remainder (fixed head).
                k_eff = min(next_pos.shape[1], n_frames - t)
                bundle = next_pos[:, :k_eff]  # (P, k_eff, dim)
                bundle_aux = aux[:, :k_eff]  # (P, k_eff, 1)
                if kin_idx.numel():
                    # Override every bundle frame's kinematic rows with GT
                    # BEFORE the re-seed, so the fed-back window's last frame
                    # carries GT kinematic rows and the next bundle's tripwire
                    # stays green.
                    bundle = bundle.clone()
                    for j in range(k_eff):
                        bundle[kin_idx, j] = pos[t + j][kin_idx]
                for j in range(k_eff):
                    predicted.append(bundle[:, j])
                    aux_pred.append(bundle_aux[:, j, 0])
                seq = torch.stack(predicted[-input_frames:], dim=0)
                t += k_eff

    return _finalize_rollout(
        predicted,
        aux_pred,
        trajectory,
        input_frames,
        kin_mask_np,
        keep,
        scored_frames,
        qois,
    )


def _finalize_rollout(
    predicted: list[torch.Tensor],
    aux_pred: list[torch.Tensor],
    trajectory: CaseTrajectory,
    input_frames: int,
    kin_mask_np: np.ndarray,
    keep: np.ndarray | None,
    scored_frames: int | None,
    qois: Mapping[str, QoiFn] | None,
) -> RolloutResult:
    """Assemble per-frame arrays into a :class:`RolloutResult` (shared tail).

    Stacks the per-frame ``predicted``/``aux_pred`` lists (each already
    containing ``T`` entries: the ``input_frames`` seeded frames plus every
    predicted frame), zeroes kinematic aux, computes the ADR-0035/0039 scored
    per-frame RMSEs, their relative-L2 companions (ADR-0055), and the QoIs, and
    packages a :class:`RolloutResult`. Shared by
    the autoregressive :func:`rollout` and the time-conditioned
    :func:`time_conditioned_rollout` so both report identically defined
    metrics (registry comparability, ADR-0054 decision C).
    """
    pred_pos = torch.stack(predicted, dim=0).cpu().numpy().astype(np.float32)
    pred_aux = torch.stack(aux_pred, dim=0).cpu().numpy().astype(np.float32)
    n_frames = pred_pos.shape[0]
    if kin_mask_np.any():
        # Kinematic particles receive no aux training signal (their loss is
        # masked, ADR-0026), so the decoder's output there is meaningless;
        # zero it so saved rollouts and visualizations cannot mistake it for
        # a prediction.
        pred_aux[:, kin_mask_np] = 0.0
    pos_rmse = position_rmse(
        pred_pos[input_frames:], trajectory.positions[input_frames:], keep=keep
    )
    aux_rmse = field_rmse(
        pred_aux[input_frames:], trajectory.aux[input_frames:], keep=keep
    )
    # Relative-L2 companions (ADR-0055), same predicted span and kinematic mask
    # as the RMSEs. Displacement is referenced to the GT frame-0 initial
    # positions for BOTH predicted and ground truth (decision A: frame-0 is the
    # shared seeded prefix, and relative L2 must be on displacement, not on
    # origin-dominated absolute coordinates). The per-frame arrays feed the
    # SECONDARY per-frame-mean metric; the POOLED headline is computed below.
    ref0 = trajectory.positions[0]  # (P, dim) GT frame-0 geometry
    disp_pred_all = pred_pos[input_frames:] - ref0
    disp_gt_all = trajectory.positions[input_frames:] - ref0
    rel_l2_disp = relative_l2(disp_pred_all, disp_gt_all, mask=keep)
    rel_l2_aux = relative_l2(
        pred_aux[input_frames:], trajectory.aux[input_frames:], mask=keep
    )

    # Scored span (ADR-0039): scored_frames mirrors T as an exclusive bound,
    # clamped so short (e.g. fixture) trajectories keep their full span.
    scored_end = n_frames if scored_frames is None else min(scored_frames, n_frames)
    n_scored = scored_end - input_frames

    # Pooled space+time headline (ADR-0055 follow-up, 2026-08-16): flatten each
    # quantity's SCORED rollout into one vector, ‖pred−gt‖₂/‖gt‖₂. Robust to the
    # near-zero early frames that make the per-frame mean explode. Quantities stay
    # separate (mm vs aux unit are incommensurable).
    pooled_disp = relative_l2_pooled(
        disp_pred_all[:n_scored], disp_gt_all[:n_scored], mask=keep
    )
    pooled_aux = relative_l2_pooled(
        pred_aux[input_frames:][:n_scored],
        trajectory.aux[input_frames:][:n_scored],
        mask=keep,
    )

    pred_inputs = QoiInputs(
        time=trajectory.time[:scored_end],
        positions=pred_pos[:scored_end],
        aux=pred_aux[:scored_end],
        particle_type=trajectory.particle_type,
        init=input_frames,
    )
    true_inputs = QoiInputs(
        time=trajectory.time[:scored_end],
        positions=trajectory.positions[:scored_end],
        aux=trajectory.aux[:scored_end],
        particle_type=trajectory.particle_type,
        init=input_frames,
    )
    qoi_pred = {name: float(fn(pred_inputs)) for name, fn in (qois or {}).items()}
    qoi_true = {name: float(fn(true_inputs)) for name, fn in (qois or {}).items()}
    return RolloutResult(
        predicted_positions=pred_pos,
        predicted_aux=pred_aux,
        position_rmse=pos_rmse,
        aux_rmse=aux_rmse,
        mean_position_rmse=float(pos_rmse[:n_scored].mean()),
        mean_aux_rmse=float(aux_rmse[:n_scored].mean()),
        mean_rel_l2_displacement=pooled_disp,
        mean_rel_l2_aux=pooled_aux,
        mean_rel_l2_displacement_perframe=float(rel_l2_disp[:n_scored].mean()),
        mean_rel_l2_aux_perframe=float(rel_l2_aux[:n_scored].mean()),
        qoi_pred=qoi_pred,
        qoi_true=qoi_true,
        qoi_error={name: qoi_pred[name] - qoi_true[name] for name in qoi_pred},
    )


class _TimeConditionedSimulatorLike(Protocol):
    """Structural type for the time-conditioned simulator (ADR-0054)."""

    def predict_state_at(
        self, frame: int, t_norm: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(positions (P, dim), aux (P, 1))`` at a scored frame."""
        ...


def time_conditioned_rollout(
    simulator: _TimeConditionedSimulatorLike,
    trajectory: CaseTrajectory,
    input_frames: int,
    time_ref_frames: int,
    device: str = "cpu",
    qois: Mapping[str, QoiFn] | None = None,
    kinematic_types: tuple[int, ...] = (),
    scored_frames: int | None = None,
) -> RolloutResult:
    """Independent-query trajectory assembly for the time-conditioned scheme.

    The faithful thuml structural scheme (ADR-0054) is NOT autoregressive: each
    frame ``f`` is predicted independently from the static geometry and the
    normalized query time ``t = f / (time_ref_frames - 1)`` — there is no
    feedback and no error accumulation. This queries every predicted frame
    ``f ∈ [input_frames, T)`` via ``simulator.predict_state_at``, seeds the
    first ``input_frames`` frames with ground truth (mirroring
    :func:`rollout`'s contract so metrics stay comparable), and reports the
    same ADR-0035/0039 per-frame position/aux RMSEs and QoIs. The reported
    ``position_rmse`` therefore fills the ``rollout_position_rmse`` slot but is
    a genuine independent-query error, not a rollout-accumulated one.

    Frames past the scored horizon (``f >= time_ref_frames``) query ``t > 1``
    — a genuine time-extrapolation of the model, contributing only to the
    non-scored full-horizon diagnostic (``position_rmse[scored:]``).

    Parameters
    ----------
    simulator:
        Object exposing ``predict_state_at(frame, t_norm) -> (positions, aux)``
        and already bound to ``trajectory``'s case (rest coords, node types,
        ground-truth trajectory) via ``bind_case``.
    trajectory:
        Ground-truth :class:`CaseTrajectory`.
    input_frames:
        History length / seed count (kept for metric-span parity with the AR
        baseline; the TC model is history-free).
    time_ref_frames:
        Denominator of the normalized query time: ``t = f / (time_ref_frames
        - 1)``. Must match the value the checkpoint was trained with (the
        scored horizon in frames; ADR-0054).
    device:
        Torch device string.
    qois, kinematic_types, scored_frames:
        As in :func:`rollout`.

    Returns
    -------
    RolloutResult
    """
    pos = torch.from_numpy(trajectory.positions).to(device)  # (T, P, dim)
    n_frames = pos.shape[0]
    aux_true = torch.from_numpy(trajectory.aux).to(device)

    if input_frames < 2:
        raise ValueError(f"input_frames must be >= 2, got {input_frames}")
    if input_frames >= n_frames:
        raise ValueError(
            f"input_frames={input_frames} but trajectory has {n_frames} frames"
        )
    if time_ref_frames < 2:
        raise ValueError(f"time_ref_frames must be >= 2, got {time_ref_frames}")

    kin_mask_np = np.isin(trajectory.particle_type, np.asarray(kinematic_types))
    keep: np.ndarray | None = ~kin_mask_np if kin_mask_np.any() else None

    predicted = [pos[i] for i in range(input_frames)]
    aux_pred = [aux_true[i] for i in range(input_frames)]
    with torch.no_grad():
        for f in range(input_frames, n_frames):
            t_norm = f / (time_ref_frames - 1)
            next_pos, aux = simulator.predict_state_at(f, t_norm)
            predicted.append(next_pos)
            aux_pred.append(aux[:, 0])

    return _finalize_rollout(
        predicted,
        aux_pred,
        trajectory,
        input_frames,
        kin_mask_np,
        keep,
        scored_frames,
        qois,
    )


def one_step_position_rmse(
    simulator: _SimulatorLike,
    trajectory: CaseTrajectory,
    input_frames: int,
    device: str = "cpu",
    kinematic_types: tuple[int, ...] = (),
) -> NDArray[np.float64]:
    """Teacher-forced next-step position RMSE per predicted frame (ADR-0019 §5).

    Every frame from ``input_frames`` onward is predicted from its *ground-truth*
    history, so no rollout error accumulates: this isolates the model's
    single-step accuracy, complementing the full-rollout RMSE from
    :func:`rollout`.  It scores the same ``[input_frames, T)`` span as the
    rollout (ADR-0035).

    Parameters
    ----------
    simulator:
        Same structural interface as :func:`rollout`.
    trajectory:
        Ground-truth :class:`CaseTrajectory`; must have more than
        ``input_frames`` frames.
    input_frames:
        History length used for each prediction.
    device:
        Torch device string.
    kinematic_types:
        Particle part-ids excluded from the reported RMSE (ADR-0026).
        Teacher-forced evaluation already uses ground-truth history, so no
        position overwrite is needed; only the metric mask is applied.

    Returns
    -------
    numpy.ndarray
        Shape ``(T - input_frames,)``, in the trajectory's working length unit
        (mm for the Taylor benchmark).
    """
    pos = torch.from_numpy(trajectory.positions).to(device)
    n_frames, n_particles, _ = pos.shape
    if n_frames <= input_frames:
        raise ValueError(
            f"trajectory has {n_frames} frames; input_frames={input_frames}"
        )
    ptype = torch.from_numpy(trajectory.particle_type).to(device)
    npp = torch.tensor([n_particles], device=device)

    kin_mask_np = np.isin(trajectory.particle_type, np.asarray(kinematic_types))
    keep: np.ndarray | None = ~kin_mask_np if kin_mask_np.any() else None

    # A k>1 bundled simulator predicts k frames per call; the one-step metric
    # scores only the FIRST (genuine single-step-from-GT-history) frame, and
    # needs teacher-forced pointer mode (advance by 1, not k) since this sweep
    # slides the GT window by 1. k=1 simulators expose no frames_per_call and
    # take the unchanged, byte-identical path.
    tf_kw = (
        {"teacher_forced": True} if getattr(simulator, "frames_per_call", 1) > 1 else {}
    )
    predicted = []
    with torch.no_grad():
        for t in range(input_frames, n_frames):
            seq_pw = pos[t - input_frames : t].permute(1, 0, 2).contiguous()
            next_pos, _ = simulator.predict_positions(seq_pw, npp, ptype, **tf_kw)
            if next_pos.ndim == 3:  # (P, k, dim) bundle -> first frame
                next_pos = next_pos[:, 0]
            predicted.append(next_pos)

    pred = torch.stack(predicted, dim=0).cpu().numpy().astype(np.float32)
    return position_rmse(pred, trajectory.positions[input_frames:], keep=keep)


def one_step_aux_rmse(
    simulator: _SimulatorLike,
    trajectory: CaseTrajectory,
    input_frames: int,
    device: str = "cpu",
    kinematic_types: tuple[int, ...] = (),
) -> NDArray[np.float64]:
    """Teacher-forced next-step aux-field RMSE per predicted frame (ADR-0025).

    Same protocol as :func:`one_step_position_rmse`, reading the auxiliary
    prediction instead: each frame from ``input_frames`` onward is predicted
    from its ground-truth history, isolating single-step accuracy of the aux
    head.

    Parameters
    ----------
    simulator:
        Same structural interface as :func:`rollout`.
    trajectory:
        Ground-truth :class:`CaseTrajectory`; must have more than
        ``input_frames`` frames.
    input_frames:
        History length used for each prediction.
    device:
        Torch device string.
    kinematic_types:
        Particle part-ids excluded from the reported RMSE (ADR-0026).
        Teacher-forced evaluation already uses ground-truth history, so no
        position overwrite is needed; only the metric mask is applied.

    Returns
    -------
    numpy.ndarray
        Shape ``(T - input_frames,)``, in the trajectory's working aux unit.
    """
    pos = torch.from_numpy(trajectory.positions).to(device)
    n_frames, n_particles, _ = pos.shape
    if n_frames <= input_frames:
        raise ValueError(
            f"trajectory has {n_frames} frames; input_frames={input_frames}"
        )
    ptype = torch.from_numpy(trajectory.particle_type).to(device)
    npp = torch.tensor([n_particles], device=device)

    kin_mask_np = np.isin(trajectory.particle_type, np.asarray(kinematic_types))
    keep: np.ndarray | None = ~kin_mask_np if kin_mask_np.any() else None

    # See one_step_position_rmse: teacher-forced pointer mode + first-frame
    # scoring for a k>1 bundle; byte-identical unchanged path at k=1.
    tf_kw = (
        {"teacher_forced": True} if getattr(simulator, "frames_per_call", 1) > 1 else {}
    )
    predicted = []
    with torch.no_grad():
        for t in range(input_frames, n_frames):
            seq_pw = pos[t - input_frames : t].permute(1, 0, 2).contiguous()
            _, aux = simulator.predict_positions(seq_pw, npp, ptype, **tf_kw)
            if aux.ndim == 3:  # (P, k, 1) bundle -> first frame's aux
                aux = aux[:, 0]
            predicted.append(aux[:, 0])

    pred = torch.stack(predicted, dim=0).cpu().numpy().astype(np.float32)
    return field_rmse(pred, trajectory.aux[input_frames:], keep=keep)
