import dataclasses

import numpy as np
import torch

from structbench.datasets.canonical import CaseTrajectory
from structbench.eval.metrics import QoiInputs, final_length, mushroom_width
from structbench.eval.rollout import (
    RolloutResult,
    one_step_aux_rmse,
    one_step_position_rmse,
    rollout,
)


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


class _ZeroSim:
    """Predicts zeros for every particle position and aux."""

    def predict_positions(
        self, position_sequence, nparticles_per_example, particle_types
    ):
        n_particles = position_sequence.shape[0]
        dim = position_sequence.shape[2]
        return torch.zeros(n_particles, dim), torch.zeros(n_particles, 1)


class _PerfectSim:
    """Returns ground-truth next positions/aux by counting prediction steps.

    Single-rollout-use only: internal step counter is never reset.

    Parameters
    ----------
    traj:
        The ground-truth trajectory used by rollout; positions and aux are
        read at increasing frame indices starting from ``window``.
    window:
        History length passed to :func:`rollout`; sets the starting frame.
    """

    def __init__(self, traj: CaseTrajectory, window: int = 2) -> None:
        self._pos = traj.positions  # (T, P, dim)
        self._aux = traj.aux  # (T, P)
        self._step = window

    def predict_positions(
        self, position_sequence, nparticles_per_example, particle_types
    ):
        pos = torch.from_numpy(self._pos[self._step])  # (P, dim)
        aux = torch.from_numpy(self._aux[self._step]).unsqueeze(1)  # (P, 1)
        self._step += 1
        return pos, aux


def _const_vel_traj(T: int = 6, P: int = 4) -> CaseTrajectory:
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pos[:, :, 0] = np.arange(T)[:, None]  # const velocity +1 in x
    # Non-trivial aux so seed-with-ground-truth tests are meaningful.
    aux = np.arange(T * P, dtype=np.float32).reshape(T, P)
    return CaseTrajectory(
        "a",
        pos,
        np.ones(P, np.int64),
        aux,
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
    true_inputs = QoiInputs(
        time=traj.time,
        positions=traj.positions,
        aux=traj.aux,
    )
    assert res.qoi_true["final_length"] == final_length(true_inputs)
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


def test_rollout_qois_receive_aux_and_time():
    traj = _const_vel_traj()
    sim = _ConstVelSim()

    def aux_peak(inputs: QoiInputs) -> float:
        assert inputs.time.shape[0] == inputs.positions.shape[0]
        return float(np.abs(inputs.aux).max())

    result = rollout(sim, traj, window=2, qois={"aux_peak": aux_peak})
    assert np.isfinite(result.qoi_true["aux_peak"])
    assert result.qoi_true["aux_peak"] == float(np.abs(traj.aux).max())


def test_rollout_seeds_predicted_aux_with_ground_truth():
    traj = _const_vel_traj()
    sim = _ConstVelSim()
    result = rollout(sim, traj, window=2)
    np.testing.assert_allclose(result.predicted_aux[:2], traj.aux[:2])


def test_one_step_aux_rmse_shape_and_finiteness():
    traj = _const_vel_traj()
    sim = _ConstVelSim()
    per_frame = one_step_aux_rmse(sim, traj, window=2)
    assert per_frame.shape == (traj.positions.shape[0] - 2,)
    assert np.all(np.isfinite(per_frame))


def test_rollout_qoi_inputs_carry_particle_type():
    traj = _const_vel_traj()
    sim = _ConstVelSim()

    def type_checker(inputs: QoiInputs) -> float:
        assert inputs.particle_type is not None
        return float(inputs.particle_type.sum())

    result = rollout(sim, traj, window=2, qois={"tc": type_checker})
    assert result.qoi_true["tc"] == float(traj.particle_type.sum())


def test_rollout_prescribes_kinematic_particles():
    """Kinematic particle follows ground truth despite a zero predictor."""
    traj = _const_vel_traj()
    ptype = traj.particle_type.copy()
    ptype[0] = 7
    traj = dataclasses.replace(traj, particle_type=ptype)
    sim = _ZeroSim()
    result = rollout(sim, traj, window=2, kinematic_types=(7,))
    # particle 0 follows ground truth exactly despite the zero predictor
    np.testing.assert_allclose(
        result.predicted_positions[:, 0, :], traj.positions[:, 0, :]
    )
    # RMSE has one entry per predicted time step
    assert result.position_rmse.shape[0] == traj.positions.shape[0] - 2


def test_rollout_metrics_exclude_kinematic_particles():
    """With a perfect predictor, RMSE over free particles is zero."""
    traj = _const_vel_traj()
    ptype = traj.particle_type.copy()
    ptype[0] = 7
    traj = dataclasses.replace(traj, particle_type=ptype)
    sim = _PerfectSim(traj, window=2)
    result = rollout(sim, traj, window=2, kinematic_types=(7,))
    assert np.allclose(result.position_rmse, 0.0)
