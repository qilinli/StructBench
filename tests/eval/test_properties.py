"""The properties block's kernels (ADR-0075 clause 6; plan 3c): pure numpy
over the rollout record, judged by the verification kernels where one exists.
"""

import math

import numpy as np
import pytest
import torch

from structbench.eval import (
    PROPERTY_KEYS,
    closure_ratio,
    error_growth,
    peeq_decrease_rate,
    state_sufficiency,
    yield_violation_rate,
)
from structbench.eval.properties import von_mises4
from structbench.models.common.two_site import von_mises4 as von_mises4_torch

KNOTS_PEEQ = (0.0, 10.0)
KNOTS_SY = (200.0, 10_200.0)  # linear hardening, slope 1000 MPa


def test_property_keys_are_the_block_in_render_order():
    assert PROPERTY_KEYS == (
        "prop_closure_ratio",
        "prop_error_growth",
        "prop_yield_violation_rate",
        "prop_peeq_decrease_rate",
        "prop_state_sufficiency",
    )


# --- closure -----------------------------------------------------------------


def test_closure_of_equal_series_is_one():
    err = np.array([0.1, 0.2, 0.3])
    assert closure_ratio(err, err) == pytest.approx(1.0)


def test_closure_is_the_ratio_of_scored_span_sums():
    one_step = np.array([0.1, 0.2, 0.3])
    assert closure_ratio(2.0 * one_step, one_step) == pytest.approx(2.0)
    # Not a mean of per-frame ratios: a zero one-step frame does not blow up
    # while the sum is positive.
    assert closure_ratio(np.array([1.0, 1.0]), np.array([0.0, 2.0])) == pytest.approx(
        1.0
    )


def test_closure_with_zero_one_step_error_is_nan():
    assert math.isnan(closure_ratio(np.array([0.1, 0.1]), np.zeros(2)))


# --- error growth ------------------------------------------------------------


def test_error_growth_recovers_an_exponential_rate():
    t = np.arange(20)
    assert error_growth(np.exp(0.1 * t)) == pytest.approx(0.1, abs=1e-9)


def test_error_growth_of_a_constant_is_zero():
    assert error_growth(np.full(7, 0.3)) == pytest.approx(0.0, abs=1e-12)


def test_error_growth_drops_zero_frames_and_needs_two():
    # A zero first frame (seeded exactly) is dropped, not log(0).
    t = np.arange(6)
    err = np.exp(0.2 * t)
    err[0] = 0.0
    assert error_growth(err) == pytest.approx(0.2, abs=1e-9)
    assert math.isnan(error_growth(np.array([0.5])))
    assert math.isnan(error_growth(np.array([0.0, 0.5])))


# --- yield bound -------------------------------------------------------------


def _stress_at(ratio: float, peeq: np.ndarray) -> np.ndarray:
    """Uniaxial s11 at ``ratio`` times the yield stress of ``peeq``."""
    sy = np.interp(peeq, KNOTS_PEEQ, KNOTS_SY)
    stress = np.zeros(peeq.shape + (4,))
    stress[..., 0] = ratio * sy
    return stress


def test_yield_rate_is_zero_inside_the_surface():
    peeq = np.linspace(0.0, 0.5, 12).reshape(3, 4)
    vm = von_mises4(_stress_at(0.9, peeq))
    assert yield_violation_rate(vm, peeq, KNOTS_PEEQ, KNOTS_SY, 0.0) == 0.0


def test_yield_rate_reads_the_tolerance():
    peeq = np.linspace(0.0, 0.5, 12).reshape(3, 4)
    vm = von_mises4(_stress_at(1.01, peeq))
    assert yield_violation_rate(vm, peeq, KNOTS_PEEQ, KNOTS_SY, 0.02) == 0.0
    assert yield_violation_rate(vm, peeq, KNOTS_PEEQ, KNOTS_SY, 0.0) == 1.0


def test_yield_rate_is_a_fraction_of_samples():
    peeq = np.zeros((2, 4))
    stress = _stress_at(0.5, peeq)
    stress[1, 0, 0] = 2.0 * KNOTS_SY[0]  # one of eight samples outside
    vm = von_mises4(stress)
    assert yield_violation_rate(vm, peeq, KNOTS_PEEQ, KNOTS_SY, 0.0) == pytest.approx(
        1.0 / 8.0
    )


# --- plastic-strain monotonicity ---------------------------------------------


def test_peeq_rate_is_zero_on_a_monotone_block():
    peeq = np.cumsum(np.random.default_rng(0).random((6, 5)), axis=0)
    assert peeq_decrease_rate(peeq, 0.0) == 0.0


def test_peeq_rate_counts_drops_beyond_the_tolerance():
    peeq = np.cumsum(np.ones((6, 5)), axis=0)
    peeq[3, 2] -= 1.5  # one drop of 0.5 among 25 steps; recovers at frame 4
    assert peeq_decrease_rate(peeq, 1.0) == 0.0
    assert peeq_decrease_rate(peeq, 0.25) == pytest.approx(1.0 / 25.0)


# --- state sufficiency -------------------------------------------------------


def test_sufficiency_is_one_when_the_predictions_are_exact():
    rng = np.random.default_rng(1)
    true_a = rng.normal(size=(5, 7, 2))
    true_b = true_a + rng.normal(size=(5, 7, 2))
    assert state_sufficiency(true_a, true_b, true_a, true_b) == pytest.approx(1.0)


def test_sufficiency_is_zero_when_the_model_reads_no_state():
    rng = np.random.default_rng(2)
    true_a = rng.normal(size=(5, 7, 2))
    true_b = true_a + rng.normal(size=(5, 7, 2))
    pred = rng.normal(size=(5, 7, 2))
    assert state_sufficiency(pred, pred, true_a, true_b) == 0.0


def test_sufficiency_is_nan_for_a_pair_that_does_not_diverge():
    x = np.ones((3, 4, 2))
    assert math.isnan(state_sufficiency(x, x, x, x))


# --- von Mises ---------------------------------------------------------------


def test_numpy_von_mises4_matches_the_torch_helper():
    stress = np.random.default_rng(3).normal(size=(4, 6, 4)) * 100.0
    expected = von_mises4_torch(torch.from_numpy(stress)).numpy()
    np.testing.assert_allclose(von_mises4(stress), expected, rtol=1e-12)
