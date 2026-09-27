"""Edges of the convergence engine found by the plan 2a review: zeros, NaN, a
flat first step, and elements the restriction cannot own."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from test_convergence import grid  # noqa: E402

from structbench.verification import convergence as cv  # noqa: E402


def test_a_zero_finest_value_gives_no_ratio_and_no_crash():
    x = cv.richardson({"1": 0.4, "2": 0.1, "4": 0.0}, ratio=2.0)
    assert x.status == "monotone" and x.coarsest_vs_finest is None
    assert x.gci_fine is None  # divides by the finest value too


def test_all_zero_values_are_flat_with_no_ratios():
    x = cv.richardson({"1": 0.0, "2": 0.0, "4": 0.0}, ratio=2.0)
    assert x.status == "flat" and x.coarsest_vs_finest is None


def test_an_extrapolated_zero_gives_no_error_against_it():
    # 4, 2, 1 at ratio 2: order 1, extrapolated 0
    x = cv.richardson({"1": 4.0, "2": 2.0, "4": 1.0}, ratio=2.0)
    assert x.status == "monotone" and x.extrapolated == pytest.approx(0.0)
    assert x.error_vs_extrapolated is None and x.coarsest_vs_finest == pytest.approx(
        3.0
    )


def test_non_finite_values_are_refused_by_name():
    with pytest.raises(ValueError, match="finite"):
        cv.richardson({"1": 1.0, "2": float("nan"), "4": 1.0}, ratio=2.0)
    with pytest.raises(ValueError, match="finite"):
        cv.richardson({"1": 1.0, "2": 1.0, "4": float("inf")}, ratio=2.0)


def test_a_flat_first_step_with_a_growing_second_is_diverging():
    x = cv.richardson({"1": 1.0, "2": 1.0, "4": 1.05}, ratio=2.0)
    assert x.status == "diverging"


def test_restriction_names_its_limit_on_elements_that_are_not_rectangles():
    cxy, cconn = grid(2, 2, 1.0, 1.0)
    fxy, fconn = grid(4, 4, 1.0, 1.0)
    shear = np.array([[1.0, 0.1], [0.0, 1.0]])  # the same taper on both: still nested
    with pytest.raises(cv.NestingError, match="axis-aligned"):
        cv.Restriction(cxy @ shear.T, cconn, fxy @ shear.T, fconn)
