"""Tests for the array-level verification kernels (ADR-0066)."""

from __future__ import annotations

import numpy as np
import pytest

from structbench.verification.kernels import (
    RatioExtreme,
    abs_max,
    decrease_max,
    interp_yield_stress,
    nearest_knot_distance,
    nonfinite_count,
    pressure_trace_residual,
    relative_drift,
    signed_distance_to_plane,
    yield_ratio_max,
    yielding_mask,
)

# A table with a non-monotone knot, as real inputs can have; np.interp semantics.
_EPS = (0.0, 0.5, 1.0, 2.0)
_SY = (200.0e6, 250.0e6, 249.0e6, 300.0e6)


def test_yield_stress_interpolates_and_clamps_at_both_ends() -> None:
    got = interp_yield_stress(np.array([-1.0, 0.0, 0.25, 0.75, 5.0]), _EPS, _SY)
    np.testing.assert_allclose(got, [200e6, 200e6, 225e6, 249.5e6, 300e6])


def test_yield_ratio_max_reports_where_the_maximum_sits() -> None:
    state = np.array([[0.0, 0.0], [0.5, 0.25], [0.5, 1.0]])
    vm = np.array([[100e6, 50e6], [250e6, 225e6], [100e6, 249e6 * 1.01]])
    got = yield_ratio_max(vm, state, _EPS, _SY)
    assert got.value == pytest.approx(1.01)
    assert (got.frame, got.index, got.n_samples) == (2, 1, 6)


def test_ratio_extremes_merge_like_one_array() -> None:
    rng = np.random.default_rng(1)
    state = np.sort(rng.random((6, 4)), axis=0)
    vm = rng.random((6, 4)) * 300e6
    whole = yield_ratio_max(vm, state, _EPS, _SY)
    head = yield_ratio_max(vm[:3], state[:3], _EPS, _SY)
    tail = yield_ratio_max(vm[3:], state[3:], _EPS, _SY, frame_offset=3)
    assert head.merge(tail) == whole
    assert isinstance(whole, RatioExtreme)


def test_yielding_mask_marks_samples_mid_way_through_plastic_loading() -> None:
    state = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.2, 0.3], [0.2, 0.4]])
    mask = yielding_mask(state)
    assert mask.shape == state.shape
    # particle 0 grows across frames 0->1->2: frame 1 is mid-loading
    # particle 1 grows across 2->3->4: frame 3 is mid-loading
    assert mask.tolist() == [
        [False, False],
        [True, False],
        [False, False],
        [False, True],
        [False, False],
    ]
    assert not yielding_mask(np.zeros((2, 3))).any()  # too short for an interior


def test_decrease_max_finds_backward_steps_along_frames() -> None:
    x = np.array([[0.0, 1.0], [0.5, 0.75], [0.4, 0.75]])
    largest, count = decrease_max(x)
    assert largest == pytest.approx(0.25)
    assert count == 2
    assert decrease_max(np.array([[0.0], [0.0], [1.0]])) == (0.0, 0)


def test_small_reductions() -> None:
    assert nonfinite_count(np.array([1.0, np.nan, np.inf, -np.inf])) == 3
    assert abs_max(np.array([[-3.0, 2.0]])) == 3.0
    assert relative_drift(np.array([2.0, 2.0, 1.5])) == pytest.approx(0.25)
    assert nearest_knot_distance(0.6, _EPS) == pytest.approx(0.1)


def test_relative_drift_needs_a_nonzero_start() -> None:
    with pytest.raises(ValueError, match="first sample"):
        relative_drift(np.array([0.0, 1.0]))


def test_pressure_is_minus_a_third_of_the_stress_trace() -> None:
    stress = np.zeros((2, 3, 6))
    stress[..., 0], stress[..., 1], stress[..., 2] = -300.0, -150.0, -150.0
    pressure = np.full((2, 3), 200.0)
    assert pressure_trace_residual(stress, pressure) == pytest.approx(0.0)
    assert pressure_trace_residual(stress, pressure + 30.0) == pytest.approx(0.1)


def test_signed_distance_uses_as_many_components_as_the_positions_have() -> None:
    positions = np.array([[[0.5, 9.0], [-0.25, 9.0]]])  # (T=1, P=2, dim=2)
    got = signed_distance_to_plane(positions, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    np.testing.assert_allclose(got, [[0.5, -0.25]])
