import dataclasses

import numpy as np
import pytest
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
        read at increasing frame indices starting from ``input_frames``.
    input_frames:
        History length passed to :func:`rollout`; sets the starting frame.
    """

    def __init__(self, traj: CaseTrajectory, input_frames: int = 2) -> None:
        self._pos = traj.positions  # (T, P, dim)
        self._aux = traj.aux  # (T, P)
        self._step = input_frames

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
    res = rollout(_ConstVelSim(), traj, input_frames=3)
    assert isinstance(res, RolloutResult)
    assert res.predicted_positions.shape == (6, 4, 2)
    np.testing.assert_allclose(res.predicted_positions, traj.positions, atol=1e-5)
    np.testing.assert_allclose(res.position_rmse, 0.0, atol=1e-5)


def test_rollout_reports_cumulative_means():
    res = rollout(_ConstVelSim(), _const_vel_traj(), input_frames=3)
    np.testing.assert_allclose(res.mean_position_rmse, 0.0, atol=1e-5)
    np.testing.assert_allclose(res.mean_aux_rmse, res.aux_rmse.mean(), atol=1e-12)


def test_rollout_computes_qois_when_given():
    traj = _const_vel_traj()
    qois = {"final_length": final_length, "mushroom_width": mushroom_width}
    res = rollout(_ConstVelSim(), traj, input_frames=3, qois=qois)
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
    res = rollout(_ConstVelSim(), _const_vel_traj(), input_frames=3)
    assert res.qoi_pred == {} and res.qoi_true == {} and res.qoi_error == {}


def test_one_step_position_rmse_zero_for_perfect_simulator():
    out = one_step_position_rmse(_ConstVelSim(), _const_vel_traj(), input_frames=3)
    assert out.shape == (3,)  # T - input_frames predicted frames
    np.testing.assert_allclose(out, 0.0, atol=1e-5)


def test_one_step_position_rmse_is_teacher_forced():
    # A frozen simulator is off by exactly one frame of motion (1 mm in x) at
    # every step when fed ground-truth history: per-step RMSE stays constant at
    # sqrt(mean([1, 0])) = sqrt(0.5). Autoregressive error would grow instead.
    out = one_step_position_rmse(_FrozenSim(), _const_vel_traj(), input_frames=3)
    np.testing.assert_allclose(out, np.sqrt(0.5), atol=1e-5)


def test_rollout_qois_receive_aux_and_time():
    traj = _const_vel_traj()
    sim = _ConstVelSim()

    def aux_peak(inputs: QoiInputs) -> float:
        assert inputs.time.shape[0] == inputs.positions.shape[0]
        return float(np.abs(inputs.aux).max())

    result = rollout(sim, traj, input_frames=2, qois={"aux_peak": aux_peak})
    assert np.isfinite(result.qoi_true["aux_peak"])
    assert result.qoi_true["aux_peak"] == float(np.abs(traj.aux).max())


def test_rollout_seeds_predicted_aux_with_ground_truth():
    traj = _const_vel_traj()
    sim = _ConstVelSim()
    result = rollout(sim, traj, input_frames=2)
    np.testing.assert_allclose(result.predicted_aux[:2], traj.aux[:2])


def test_one_step_aux_rmse_shape_and_finiteness():
    traj = _const_vel_traj()
    sim = _ConstVelSim()
    per_frame = one_step_aux_rmse(sim, traj, input_frames=2)
    assert per_frame.shape == (traj.positions.shape[0] - 2,)
    assert np.all(np.isfinite(per_frame))


def test_rollout_qoi_inputs_carry_particle_type():
    traj = _const_vel_traj()
    sim = _ConstVelSim()

    def type_checker(inputs: QoiInputs) -> float:
        assert inputs.particle_type is not None
        return float(inputs.particle_type.sum())

    result = rollout(sim, traj, input_frames=2, qois={"tc": type_checker})
    assert result.qoi_true["tc"] == float(traj.particle_type.sum())


def test_rollout_prescribes_kinematic_particles():
    """Kinematic particle follows ground truth despite a zero predictor."""
    traj = _const_vel_traj()
    ptype = traj.particle_type.copy()
    ptype[0] = 7
    traj = dataclasses.replace(traj, particle_type=ptype)
    sim = _ZeroSim()
    result = rollout(sim, traj, input_frames=2, kinematic_types=(7,))
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
    sim = _PerfectSim(traj, input_frames=2)
    result = rollout(sim, traj, input_frames=2, kinematic_types=(7,))
    assert np.allclose(result.position_rmse, 0.0)


def test_rollout_zeroes_kinematic_aux():
    """Kinematic particles carry zero aux on every frame; free aux is kept."""
    traj = _const_vel_traj()  # aux = arange: particle 0 is nonzero from frame 1
    ptype = traj.particle_type.copy()
    ptype[0] = 7
    traj = dataclasses.replace(traj, particle_type=ptype)
    sim = _PerfectSim(traj, input_frames=2)
    result = rollout(sim, traj, input_frames=2, kinematic_types=(7,))
    # The predictor emits ground-truth (nonzero) aux for particle 0 too, and
    # its seeded frames carry nonzero ground truth — both must be zeroed.
    np.testing.assert_allclose(result.predicted_aux[:, 0], 0.0)
    # Free particles keep the predictor's aux (here: exact ground truth).
    np.testing.assert_allclose(result.predicted_aux[:, 1:], traj.aux[:, 1:])
    # The masked metric is untouched by the zeroing.
    assert np.allclose(result.aux_rmse, 0.0)


class _RecordSim(_ConstVelSim):
    """Constant-velocity stub that records the first input history it sees."""

    def __init__(self):
        self.first_seq = None

    def predict_positions(
        self, position_sequence, nparticles_per_example, particle_types
    ):
        if self.first_seq is None:
            self.first_seq = position_sequence.clone()
        return super().predict_positions(
            position_sequence, nparticles_per_example, particle_types
        )


def test_rollout_first_input_is_observed_prefix_no_backfill():
    """The model's first input is the first input_frames real frames (ADR-0035).

    There is no constant-velocity backfill: the observed prefix IS the input
    window, so the first history the model sees is exactly frames 0..input_frames-1.
    """
    traj = _const_vel_traj(T=8)
    sim = _RecordSim()
    rollout(sim, traj, input_frames=4)
    seq = sim.first_seq.numpy()  # (P, input_frames, dim)
    assert seq.shape == (4, 4, 2)
    # x-history is exactly frames 0,1,2,3 — not a backfilled negative-index past.
    np.testing.assert_allclose(seq[:, :, 0], np.array([[0.0, 1.0, 2.0, 3.0]] * 4))


def test_rollout_scored_span_starts_at_input_frames():
    """Scored span is [input_frames, T); the observed prefix is ground truth."""
    traj = _const_vel_traj(T=10)
    res = rollout(_ConstVelSim(), traj, input_frames=5)
    assert res.position_rmse.shape == (5,)  # 10 - 5 predicted frames
    assert res.predicted_positions.shape == (10, 4, 2)
    np.testing.assert_array_equal(res.predicted_positions[:5], traj.positions[:5])
    np.testing.assert_array_equal(res.predicted_aux[:5], traj.aux[:5])


def test_rollout_input_frames_validation():
    traj = _const_vel_traj(T=6)
    import pytest

    with pytest.raises(ValueError, match="input_frames must be >= 2"):
        rollout(_ConstVelSim(), traj, input_frames=1)
    with pytest.raises(ValueError, match="trajectory has 6 frames"):
        rollout(_ConstVelSim(), traj, input_frames=6)


def test_rollout_scored_frames_truncates_means_keeps_full_arrays():
    # Frozen sim on constant-velocity motion: error at frame t is t-1, so the
    # scored mean is sensitive to exactly which frames are aggregated.
    traj = _const_vel_traj(T=8)
    res = rollout(_FrozenSim(), traj, input_frames=2, scored_frames=5)
    # Per-frame diagnostics still cover every predicted frame [2, 8).
    assert res.position_rmse.shape == (6,)
    assert res.predicted_positions.shape == (8, 4, 2)
    # Aggregates cover the scored span [2, 5) only: the mean equals the mean
    # of the first three per-frame values and sits below the full-span mean
    # (per-frame error grows with t under a frozen prediction).
    np.testing.assert_allclose(
        res.mean_position_rmse, res.position_rmse[:3].mean(), atol=1e-12
    )
    assert res.mean_position_rmse < res.position_rmse.mean()
    np.testing.assert_allclose(res.mean_aux_rmse, res.aux_rmse[:3].mean(), atol=1e-12)


def test_rollout_scored_frames_clamped_to_trajectory_end():
    traj = _const_vel_traj(T=6)
    full = rollout(_FrozenSim(), traj, input_frames=2)
    clamped = rollout(_FrozenSim(), traj, input_frames=2, scored_frames=100)
    np.testing.assert_allclose(
        clamped.mean_position_rmse, full.mean_position_rmse, atol=1e-12
    )
    np.testing.assert_allclose(clamped.mean_aux_rmse, full.mean_aux_rmse, atol=1e-12)


def test_rollout_scored_frames_windows_qois():
    # QoIs must see only the scored span's frames (ADR-0039): a final-frame
    # QoI evaluated under scored_frames=5 reads frame 4, not frame 7.
    traj = _const_vel_traj(T=8)
    qois = {
        "last_x": lambda inp: float(inp.positions[-1, 0, 0]),
        "last_t": lambda inp: float(inp.time[-1]),
    }
    res = rollout(_ConstVelSim(), traj, input_frames=2, qois=qois, scored_frames=5)
    assert res.qoi_true["last_x"] == 4.0
    assert res.qoi_true["last_t"] == 4.0
    np.testing.assert_allclose(res.qoi_error["last_x"], 0.0, atol=1e-5)


# --- ADR-0050/0051 k-frames-per-call: bundled rollout (rank-dispatch) ---


class _KFrameConstVelSim:
    """Predicts ``k`` constant-velocity frames as a ``(P, k, dim)`` bundle.

    Exposes ``frames_per_call`` so the one-step sweep switches to teacher-forced
    mode, and counts ``predict_positions`` calls so a one-shot (single-call)
    rollout can be asserted. Accepts (and ignores) the ``teacher_forced``
    keyword the one-step path passes to a k>1 simulator.
    """

    def __init__(self, k: int) -> None:
        self.frames_per_call = k
        self.calls = 0

    def predict_positions(
        self,
        position_sequence,
        nparticles_per_example,
        particle_types,
        *,
        teacher_forced: bool = False,
    ):
        self.calls += 1
        last = position_sequence[:, -1]  # (P, dim)
        vel = last - position_sequence[:, -2]  # (P, dim)
        steps = torch.arange(1, self.frames_per_call + 1).view(1, -1, 1)
        nxt = last.unsqueeze(1) + steps * vel.unsqueeze(1)  # (P, k, dim)
        aux = torch.zeros(position_sequence.shape[0], self.frames_per_call, 1)
        return nxt, aux


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_kframe_rollout_exact_for_constant_velocity(k):
    # A k-frame bundle of the true constant velocity reproduces GT for any k,
    # including k values that leave a ragged remainder (predict-k-and-truncate):
    # 10 predicted frames, k=3 -> 3*3+1, k=4 -> 4+4+2, k=5 -> 5+5.
    traj = _const_vel_traj(T=12, P=4)
    res = rollout(_KFrameConstVelSim(k), traj, input_frames=2)
    assert res.predicted_positions.shape == (12, 4, 2)
    assert res.mean_position_rmse == pytest.approx(0.0, abs=1e-5)


def test_kframe_oneshot_is_a_single_forward_call():
    # k = T - input_frames covers the whole horizon: one bundle, no re-seed.
    traj = _const_vel_traj(T=12, P=4)
    sim = _KFrameConstVelSim(k=10)
    res = rollout(sim, traj, input_frames=2)
    assert sim.calls == 1
    assert res.mean_position_rmse == pytest.approx(0.0, abs=1e-5)


def _moving_kin_traj(T: int = 10, P: int = 3) -> CaseTrajectory:
    # Particle 0 is kinematic (type 2) with QUADRATIC motion, so the const-vel
    # sim predicts it WRONG -- unlike Taylor's static wall, this actually
    # exercises the all-k-frame kinematic override.
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pos[:, 0, 0] = (np.arange(T) ** 2).astype(np.float32)  # kinematic, quadratic
    pos[:, 1:, 0] = np.arange(T)[:, None]  # free, constant velocity
    aux = np.zeros((T, P), dtype=np.float32)
    ptype = np.array([2, 1, 1], dtype=np.int64)
    return CaseTrajectory("m", pos, ptype, aux, np.arange(T, dtype=float))


def test_kframe_rollout_overrides_kinematic_across_whole_bundle():
    # The kinematic particle's predicted positions must equal GT at EVERY
    # frame, including mid-bundle frames the const-vel sim gets wrong.
    traj = _moving_kin_traj(T=10)
    res = rollout(_KFrameConstVelSim(3), traj, input_frames=2, kinematic_types=(2,))
    np.testing.assert_allclose(
        res.predicted_positions[:, 0], traj.positions[:, 0], atol=1e-5
    )


def test_kframe_one_step_scores_first_bundle_frame():
    # One-step is teacher-forced and scores only the first bundle frame; the
    # const-vel bundle's frame 0 is exact, so the one-step RMSE is ~0 and has
    # the usual (T - input_frames,) shape.
    traj = _const_vel_traj(T=12, P=4)
    rmse = one_step_position_rmse(_KFrameConstVelSim(4), traj, input_frames=2)
    assert rmse.shape == (10,)
    np.testing.assert_allclose(rmse, 0.0, atol=1e-5)
