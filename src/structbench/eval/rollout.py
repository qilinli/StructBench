"""Autoregressive rollout and one-step evaluation of a learned simulator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from ..datasets.canonical import CaseTrajectory
from .metrics import QoiFn, QoiInputs, field_rmse, position_rmse


class _SimulatorLike(Protocol):
    """Structural type for the simulator consumed by :func:`rollout`."""

    def predict_positions(
        self,
        position_sequence: torch.Tensor,
        nparticles_per_example: torch.Tensor,
        particle_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(next_positions (P,dim), aux (P,n_aux))``."""
        ...


@dataclass
class RolloutResult:
    """Predicted trajectory with per-step, cumulative, and QoI errors.

    Per-step arrays cover the ``T - input_frames`` predicted frames (the model
    observes the first ``input_frames`` ground-truth frames and predicts the
    rest, ADR-0035). The cumulative means aggregate them; QoI dicts are filled
    only when :func:`rollout` is given a ``qois`` mapping (``qoi_error`` is
    signed, predicted − true). Units follow the trajectory's working frame (mm
    and MPa for Taylor 2D). The seeded frames of ``predicted_aux`` carry
    ground-truth aux values, mirroring the seeded positions.
    """

    predicted_positions: NDArray[np.float32]  # (T, P, dim)
    predicted_aux: NDArray[np.float32]  # (T, P)
    position_rmse: NDArray[np.float64]  # (nsteps,)
    aux_rmse: NDArray[np.float64]  # (nsteps,)
    mean_position_rmse: float = float("nan")
    mean_aux_rmse: float = float("nan")
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
) -> RolloutResult:
    """Seed with the observed prefix, then autoregress to the end.

    The model observes exactly ``input_frames`` ground-truth frames (its full
    history window) and predicts every frame from ``input_frames`` onward; the
    scored span is ``[input_frames, T)`` (ADR-0035). There is no history
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
        ground-truth position at that frame.  They are also excluded from the
        reported ``position_rmse`` and ``aux_rmse`` (``keep`` mask).  Defaults
        to ``()`` (no prescribed particles).

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

    with torch.no_grad():
        for t in range(input_frames, n_frames):
            seq_pw = seq.permute(1, 0, 2).contiguous()  # (P, input_frames, dim)
            next_pos, aux = simulator.predict_positions(seq_pw, npp, ptype)
            if kin_idx.numel():
                next_pos = next_pos.clone()
                next_pos[kin_idx] = pos[t][kin_idx]
            predicted.append(next_pos)
            aux_pred.append(aux[:, 0])
            seq = torch.cat([seq[1:], next_pos[None]], dim=0)

    pred_pos = torch.stack(predicted, dim=0).cpu().numpy().astype(np.float32)
    pred_aux = torch.stack(aux_pred, dim=0).cpu().numpy().astype(np.float32)
    pos_rmse = position_rmse(
        pred_pos[input_frames:], trajectory.positions[input_frames:], keep=keep
    )
    aux_rmse = field_rmse(
        pred_aux[input_frames:], trajectory.aux[input_frames:], keep=keep
    )

    pred_inputs = QoiInputs(
        time=trajectory.time,
        positions=pred_pos,
        aux=pred_aux,
        particle_type=trajectory.particle_type,
        init=input_frames,
    )
    true_inputs = QoiInputs(
        time=trajectory.time,
        positions=trajectory.positions,
        aux=trajectory.aux,
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
        mean_position_rmse=float(pos_rmse.mean()),
        mean_aux_rmse=float(aux_rmse.mean()),
        qoi_pred=qoi_pred,
        qoi_true=qoi_true,
        qoi_error={name: qoi_pred[name] - qoi_true[name] for name in qoi_pred},
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

    predicted = []
    with torch.no_grad():
        for t in range(input_frames, n_frames):
            seq_pw = pos[t - input_frames : t].permute(1, 0, 2).contiguous()
            next_pos, _ = simulator.predict_positions(seq_pw, npp, ptype)
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

    predicted = []
    with torch.no_grad():
        for t in range(input_frames, n_frames):
            seq_pw = pos[t - input_frames : t].permute(1, 0, 2).contiguous()
            _, aux = simulator.predict_positions(seq_pw, npp, ptype)
            predicted.append(aux[:, 0])

    pred = torch.stack(predicted, dim=0).cpu().numpy().astype(np.float32)
    return field_rmse(pred, trajectory.aux[input_frames:], keep=keep)
