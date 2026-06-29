"""Autoregressive rollout of a learned simulator over a trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from ..datasets.canonical import CaseTrajectory
from .metrics import field_rmse, position_rmse


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
    """Predicted trajectory and per-step error against ground truth."""

    predicted_positions: NDArray[np.float32]  # (T, P, dim)
    predicted_aux: NDArray[np.float32]  # (T, P)
    position_rmse: NDArray[np.float64]  # (nsteps,)
    aux_rmse: NDArray[np.float64]  # (nsteps,)


def rollout(
    simulator: _SimulatorLike,
    trajectory: CaseTrajectory,
    window: int,
    device: str = "cpu",
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
    return RolloutResult(
        predicted_positions=pred_pos,
        predicted_aux=pred_aux,
        position_rmse=position_rmse(pred_pos[window:], trajectory.positions[window:]),
        aux_rmse=field_rmse(pred_aux[window:], trajectory.von_mises[window:]),
    )
