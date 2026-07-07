"""CPU-only end-to-end smoke test: datasets -> normalization -> rollout."""

import numpy as np
import torch

from structbench.datasets.canonical import CaseTrajectory
from structbench.datasets.normalization import compute_stats
from structbench.eval.rollout import rollout


def _tiny_traj(P: int = 5, T: int = 6) -> CaseTrajectory:
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pos[:, :, 0] = np.arange(T)[:, None]
    pos[:, :, 1] = np.linspace(0, 1, P)[None, :]
    return CaseTrajectory(
        "a",
        pos,
        np.ones(P, np.int64),
        np.zeros((T, P), np.float32),
        np.arange(T, dtype=float),
    )


def test_stats_and_rollout_shapes_compose() -> None:
    traj = _tiny_traj()
    stats = compute_stats([traj])
    assert stats.acceleration_mean.shape == (2,)

    # constant-velocity stub stands in for a trained model here
    class _Stub:
        def predict_positions(
            self,
            seq: torch.Tensor,
            npp: torch.Tensor,
            pt: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nxt = seq[:, -1] + (seq[:, -1] - seq[:, -2])
            return nxt, torch.zeros(seq.shape[0], 1)

    res = rollout(_Stub(), traj, input_frames=3)
    assert res.predicted_positions.shape == (6, 5, 2)
