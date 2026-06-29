import numpy as np
import torch

from structbench.datasets.canonical import CaseTrajectory
from structbench.eval.rollout import RolloutResult, rollout


class _ConstVelSim:
    """Predicts next = last + (last - prev): perfect constant-velocity motion."""

    def predict_positions(
        self, position_sequence, nparticles_per_example, particle_types
    ):
        last = position_sequence[:, -1]
        prev = position_sequence[:, -2]
        nxt = last + (last - prev)
        aux = torch.zeros(position_sequence.shape[0], 1)
        return nxt, aux


def test_rollout_is_exact_for_constant_velocity():
    T, P = 6, 4
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pos[:, :, 0] = np.arange(T)[:, None]  # const velocity +1 in x
    traj = CaseTrajectory("a", pos, np.ones(P, np.int64),
                          np.zeros((T, P), np.float32), np.arange(T, dtype=float))
    res = rollout(_ConstVelSim(), traj, window=3)
    assert isinstance(res, RolloutResult)
    assert res.predicted_positions.shape == (T, P, 2)
    np.testing.assert_allclose(res.predicted_positions, pos, atol=1e-5)
    np.testing.assert_allclose(res.position_rmse, 0.0, atol=1e-5)
