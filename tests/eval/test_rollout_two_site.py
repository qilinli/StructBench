"""``flow_map_rollout`` carries the element state of a two-site simulator
across the anchors (ADR-0075) and scores it beside the nodal view."""

import numpy as np
import pytest
import torch

from structbench.datasets import CaseTrajectory
from structbench.eval import flow_map_rollout

T, P, E, S, DIM = 8, 4, 2, 3, 2


def _trajectory():
    rng = np.random.default_rng(0)
    positions = rng.random((T, P, DIM)).astype(np.float32)
    aux = rng.random((T, P, S)).astype(np.float32)
    state = rng.random((T, E, S)).astype(np.float32)
    return CaseTrajectory(
        case_id="two-site",
        positions=positions,
        particle_type=np.zeros(P, dtype=np.int64),
        aux=aux,
        time=np.arange(T, dtype=np.float64),
        cells=np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64),
        reference_coords=positions[0].copy(),
        state=state,
        state_channels=("a", "b", "c"),
    )


class _FakeTwoSite:
    """Duck-typed two-site simulator: each query returns the anchor's state
    plus one, so the carried state is checkable per segment."""

    two_site = "corner_mean"

    def __init__(self):
        self.anchor_states: list[torch.Tensor] = []
        self._state = None
        self._pos = None
        self._aux = None

    def set_anchor(
        self, frame, prev_positions, positions, aux, anchor_t_norm=None, state=None
    ):
        assert state is not None, "the rollout must pass the element state"
        self.anchor_states.append(state.clone())
        self._state, self._pos, self._aux = state, positions, aux

    def predict_two_site_at(self, frame, t_norm):
        return self._pos + 0.01, self._aux, self._state + 1.0

    def predict_state_at(self, frame, t_norm):  # pragma: no cover - not used
        raise AssertionError(
            "a two-site simulator is queried through predict_two_site_at"
        )


def test_self_anchored_rollout_carries_the_predicted_state():
    tr = _trajectory()
    sim = _FakeTwoSite()
    result = flow_map_rollout(
        sim, tr, input_frames=2, time_ref_frames=T, interval=2, anchor_mode="self"
    )
    assert result.predicted_state is not None and result.predicted_state.shape == (
        T,
        E,
        S,
    )
    # Seeded frames carry ground truth; the first segment is anchor + 1.
    np.testing.assert_array_equal(result.predicted_state[:2], tr.state[:2])
    np.testing.assert_allclose(result.predicted_state[2], tr.state[1] + 1.0)
    # The re-anchor at frame 3 fed the model's OWN state (anchor + 1), so the
    # next segment reads anchor + 2.
    np.testing.assert_allclose(sim.anchor_states[1].numpy(), tr.state[1] + 1.0)
    np.testing.assert_allclose(result.predicted_state[4], tr.state[1] + 2.0)
    assert np.isfinite(result.mean_rel_l2_state) and np.isfinite(result.mean_state_rmse)
    assert result.state_rmse is not None and result.state_rmse.shape == (T - 2,)
    assert len(result.mean_rel_l2_state_channels) == S
    assert len(result.mean_state_rmse_channels) == S


def test_oracle_anchored_rollout_feeds_the_stored_state():
    tr = _trajectory()
    sim = _FakeTwoSite()
    flow_map_rollout(
        sim, tr, input_frames=2, time_ref_frames=T, interval=2, anchor_mode="oracle"
    )
    np.testing.assert_array_equal(sim.anchor_states[1].numpy(), tr.state[3])


def test_two_site_simulator_needs_a_state_block():
    tr = _trajectory()
    bare = CaseTrajectory(
        case_id=tr.case_id,
        positions=tr.positions,
        particle_type=tr.particle_type,
        aux=tr.aux,
        time=tr.time,
        cells=tr.cells,
        reference_coords=tr.reference_coords,
    )
    with pytest.raises(ValueError, match="element-site state"):
        flow_map_rollout(_FakeTwoSite(), bare, 2, T, 2, "self")


class _AnchorRecorder(_FakeTwoSite):
    """Records the anchor times the rollout passes (plan 3b ``frame_offset``)."""

    def __init__(self):
        super().__init__()
        self.anchor_times: list[float] = []

    def set_anchor(
        self, frame, prev_positions, positions, aux, anchor_t_norm=None, state=None
    ):
        super().set_anchor(frame, prev_positions, positions, aux, anchor_t_norm, state)
        self.anchor_times.append(anchor_t_norm)


def test_frame_offset_shifts_the_anchor_time_and_nothing_else():
    # ADR-0075 clause 5 (plan 3b): a restart rolls the flow map out on a
    # window, so the absolute anchor-time feature must see the true anchor.
    tr = _trajectory()
    plain, shifted = _AnchorRecorder(), _AnchorRecorder()
    r0 = flow_map_rollout(plain, tr, 2, 8, 3, "self")
    r5 = flow_map_rollout(shifted, tr, 2, 8, 3, "self", frame_offset=5)
    # anchors at window frames 1 and 4 -> times 1/7 and 4/7, shifted by 5/7
    assert plain.anchor_times == pytest.approx([1 / 7, 4 / 7])
    assert shifted.anchor_times == pytest.approx([6 / 7, 9 / 7])
    np.testing.assert_array_equal(r0.predicted_positions, r5.predicted_positions)
    np.testing.assert_array_equal(r0.predicted_state, r5.predicted_state)
