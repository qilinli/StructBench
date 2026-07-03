"""Autoregressive rollout and one-step evaluation of a learned simulator."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from ..datasets.canonical import CaseTrajectory
from .metrics import field_rmse, position_rmse

#: A quantity of interest maps a full ``(T, P, dim)`` trajectory to a scalar.
QoiFn = Callable[[NDArray], float]


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

    Per-step arrays cover the ``T - window`` predicted frames. The cumulative
    means aggregate them; QoI dicts are filled only when :func:`rollout` is
    given a ``qois`` mapping (``qoi_error`` is signed, predicted − true).
    Units follow the trajectory's working frame (mm and MPa for Taylor 2D).
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
    window: int,
    device: str = "cpu",
    qois: Mapping[str, QoiFn] | None = None,
) -> RolloutResult:
    """Seed with ``window`` ground-truth frames, then autoregress to the end.

    Parameters
    ----------
    simulator:
        Object with
        ``predict_positions(position_sequence, nparticles_per_example,
        particle_types) -> (next_positions (P,dim), aux (P,n_aux))``.
    trajectory:
        Ground-truth :class:`CaseTrajectory`.
    window:
        History length (frames used to seed and to predict each step).
    device:
        Torch device string.
    qois:
        Optional mapping of quantity-of-interest name to a function of a full
        ``(T, P, dim)`` trajectory (e.g. the Taylor benchmark's ``QOIS``).
        Each is evaluated on the predicted and the ground-truth trajectory and
        recorded with its signed error.

    Returns
    -------
    RolloutResult
    """
    pos = torch.from_numpy(trajectory.positions).to(device)  # (T, P, dim)
    n_frames, n_particles, _ = pos.shape
    ptype = torch.from_numpy(trajectory.particle_type).to(device)
    npp = torch.tensor([n_particles], device=device)

    seq = pos[:window].clone()  # (window, P, dim)
    predicted = [pos[i] for i in range(window)]
    aux_pred = [torch.zeros(n_particles, device=device) for _ in range(window)]

    with torch.no_grad():
        for _ in range(window, n_frames):
            seq_pw = seq.permute(1, 0, 2).contiguous()  # (P, window, dim)
            next_pos, aux = simulator.predict_positions(seq_pw, npp, ptype)
            predicted.append(next_pos)
            aux_pred.append(aux[:, 0])
            seq = torch.cat([seq[1:], next_pos[None]], dim=0)

    pred_pos = torch.stack(predicted, dim=0).cpu().numpy().astype(np.float32)
    pred_aux = torch.stack(aux_pred, dim=0).cpu().numpy().astype(np.float32)
    pos_rmse = position_rmse(pred_pos[window:], trajectory.positions[window:])
    aux_rmse = field_rmse(pred_aux[window:], trajectory.aux[window:])

    qoi_pred = {name: float(fn(pred_pos)) for name, fn in (qois or {}).items()}
    qoi_true = {
        name: float(fn(trajectory.positions)) for name, fn in (qois or {}).items()
    }
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
    window: int,
    device: str = "cpu",
) -> NDArray[np.float64]:
    """Teacher-forced next-step position RMSE per predicted frame (ADR-0019 §5).

    Every frame from ``window`` onward is predicted from its *ground-truth*
    history, so no rollout error accumulates: this isolates the model's
    single-step accuracy, complementing the full-rollout RMSE from
    :func:`rollout`.

    Parameters
    ----------
    simulator:
        Same structural interface as :func:`rollout`.
    trajectory:
        Ground-truth :class:`CaseTrajectory`; must have more than ``window``
        frames.
    window:
        History length used for each prediction.
    device:
        Torch device string.

    Returns
    -------
    numpy.ndarray
        Shape ``(T - window,)``, in the trajectory's working length unit
        (mm for the Taylor benchmark).
    """
    pos = torch.from_numpy(trajectory.positions).to(device)
    n_frames, n_particles, _ = pos.shape
    if n_frames <= window:
        raise ValueError(f"trajectory has {n_frames} frames; window={window}")
    ptype = torch.from_numpy(trajectory.particle_type).to(device)
    npp = torch.tensor([n_particles], device=device)

    predicted = []
    with torch.no_grad():
        for t in range(window, n_frames):
            seq_pw = pos[t - window : t].permute(1, 0, 2).contiguous()
            next_pos, _ = simulator.predict_positions(seq_pw, npp, ptype)
            predicted.append(next_pos)

    pred = torch.stack(predicted, dim=0).cpu().numpy().astype(np.float32)
    return position_rmse(pred, trajectory.positions[window:])
