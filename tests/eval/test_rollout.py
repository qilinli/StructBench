import numpy as np
import torch

from structbench.datasets.canonical import CaseTrajectory
from structbench.eval.metrics import final_length, mushroom_width
from structbench.eval.rollout import RolloutResult, one_step_position_rmse, rollout


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


class _FrozenSim:
    """Predicts next = last (zero velocity): wrong for any moving trajectory."""

    def predict_positions(
        self, position_sequence, nparticles_per_example, particle_types
    ):
        last = position_sequence[:, -1]
        aux = torch.zeros(position_sequence.shape[0], 1)
        return last, aux


def _const_vel_traj(T: int = 6, P: int = 4) -> CaseTrajectory:
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pos[:, :, 0] = np.arange(T)[:, None]  # const velocity +1 in x
    return CaseTrajectory(
        "a",
        pos,
        np.ones(P, np.int64),
        np.zeros((T, P), np.float32),
        np.arange(T, dtype=float),
    )


def test_rollout_is_exact_for_constant_velocity():
    traj = _const_vel_traj()
    res = rollout(_ConstVelSim(), traj, window=3)
    assert isinstance(res, RolloutResult)
    assert res.predicted_positions.shape == (6, 4, 2)
    np.testing.assert_allclose(res.predicted_positions, traj.positions, atol=1e-5)
    np.testing.assert_allclose(res.position_rmse, 0.0, atol=1e-5)


def test_rollout_reports_cumulative_means():
    res = rollout(_ConstVelSim(), _const_vel_traj(), window=3)
    np.testing.assert_allclose(res.mean_position_rmse, 0.0, atol=1e-5)
    np.testing.assert_allclose(res.mean_aux_rmse, res.aux_rmse.mean(), atol=1e-12)


def test_rollout_computes_qois_when_given():
    traj = _const_vel_traj()
    qois = {"final_length": final_length, "mushroom_width": mushroom_width}
    res = rollout(_ConstVelSim(), traj, window=3, qois=qois)
    # Perfect prediction: predicted and true QoIs agree, errors vanish.
    assert res.qoi_true["final_length"] == final_length(traj.positions)
    np.testing.assert_allclose(
        res.qoi_pred["final_length"], res.qoi_true["final_length"], atol=1e-5
    )
    np.testing.assert_allclose(res.qoi_error["mushroom_width"], 0.0, atol=1e-5)


def test_rollout_qois_default_to_empty():
    res = rollout(_ConstVelSim(), _const_vel_traj(), window=3)
    assert res.qoi_pred == {} and res.qoi_true == {} and res.qoi_error == {}


def test_one_step_position_rmse_zero_for_perfect_simulator():
    out = one_step_position_rmse(_ConstVelSim(), _const_vel_traj(), window=3)
    assert out.shape == (3,)  # T - window predicted frames
    np.testing.assert_allclose(out, 0.0, atol=1e-5)


def test_one_step_position_rmse_is_teacher_forced():
    # A frozen simulator is off by exactly one frame of motion (1 mm in x) at
    # every step when fed ground-truth history: per-step RMSE stays constant at
    # sqrt(mean([1, 0])) = sqrt(0.5). Autoregressive error would grow instead.
    out = one_step_position_rmse(_FrozenSim(), _const_vel_traj(), window=3)
    np.testing.assert_allclose(out, np.sqrt(0.5), atol=1e-5)
