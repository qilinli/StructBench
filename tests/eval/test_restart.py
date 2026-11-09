"""The restart task family's helpers (ADR-0075 clause 5; plan 3b): the window
a restart rolls out on, its metric keys and the semigroup error."""

import numpy as np
import pytest

from structbench.datasets import CaseTrajectory
from structbench.eval import (
    compose_keys,
    restart_keys,
    restart_window,
    semigroup_error,
)

T, P, E, S, DIM = 10, 4, 2, 3, 2


def _trajectory(with_state: bool = True) -> CaseTrajectory:
    rng = np.random.default_rng(0)
    positions = rng.random((T, P, DIM)).astype(np.float32)
    return CaseTrajectory(
        case_id="restart",
        positions=positions,
        particle_type=np.zeros(P, dtype=np.int64),
        aux=rng.random((T, P, S)).astype(np.float32),
        time=np.arange(T, dtype=np.float64) * 0.5,
        cells=np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64),
        reference_coords=positions[0].copy(),
        state=rng.random((T, E, S)).astype(np.float32) if with_state else None,
        state_channels=("a", "b", "c") if with_state else (),
    )


def test_window_has_the_history_then_the_horizon():
    tr = _trajectory()
    w = restart_window(tr, anchor=4, horizon=3, input_frames=2)
    # frames [3, 8): two seed frames (3, 4) and three predicted (5, 6, 7)
    assert w.positions.shape == (5, P, DIM)
    np.testing.assert_array_equal(w.positions[1], tr.positions[4])  # the anchor
    np.testing.assert_array_equal(w.positions[0], tr.positions[3])
    np.testing.assert_array_equal(w.positions[-1], tr.positions[7])
    np.testing.assert_array_equal(w.aux, tr.aux[3:8])
    np.testing.assert_array_equal(w.time, tr.time[3:8])
    assert w.state is not None
    np.testing.assert_array_equal(w.state, tr.state[3:8])
    assert w.state_channels == tr.state_channels
    # the static arrays are shared, not copied
    assert w.cells is tr.cells and w.reference_coords is tr.reference_coords
    assert w.case_id == tr.case_id


def test_window_without_a_state_block_keeps_none():
    w = restart_window(_trajectory(with_state=False), 4, 3, 2)
    assert w.state is None and w.state_channels == ()


def test_window_refuses_an_anchor_without_history_or_a_horizon_past_the_end():
    tr = _trajectory()
    with pytest.raises(ValueError, match="history"):
        restart_window(tr, anchor=0, horizon=2, input_frames=2)
    restart_window(tr, anchor=1, horizon=2, input_frames=2)  # the first legal one
    with pytest.raises(ValueError, match="past the trajectory"):
        restart_window(tr, anchor=8, horizon=2, input_frames=2)
    restart_window(tr, anchor=7, horizon=2, input_frames=2)  # ends at the last frame
    with pytest.raises(ValueError, match="horizon"):
        restart_window(tr, anchor=4, horizon=0, input_frames=2)


def test_keys_name_the_anchor_the_horizon_and_the_composition():
    assert restart_keys(10, 20, ("rel_l2_displacement", "aux_rmse")) == (
        "restart_a10_h20_rel_l2_displacement",
        "restart_a10_h20_aux_rmse",
    )
    assert compose_keys(20, 2, ("position_rmse",)) == ("compose_T20_k2_position_rmse",)


def test_semigroup_error_is_zero_for_identical_maps_and_pooled_otherwise():
    rng = np.random.default_rng(1)
    direct = rng.random((4, P, DIM))
    assert semigroup_error(direct, direct) == 0.0
    composed = direct * 1.5  # every entry 50 % larger: pooled ratio 0.5
    assert semigroup_error(direct, composed) == pytest.approx(0.5)
    assert np.isnan(semigroup_error(np.zeros((2, P, DIM)), np.ones((2, P, DIM))))
