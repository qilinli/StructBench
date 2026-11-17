"""Edges of the convergence engine found by the plan 2a review: zeros, NaN, a
flat first step, and elements the restriction cannot own."""

import math

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


# --- a ladder whose two ratios differ -------------------------------------------


def _ladder(order, spacings, limit=1.0, c=0.01):
    return {f"{1 / h:g}": limit + c * h**order for h in spacings}


@pytest.mark.parametrize("order", [0.5, 1.0, 2.0, 3.7])
@pytest.mark.parametrize("ratios", [(3.0, 2.0), (2.0, 3.0), (1.5, 2.0), (4.0, 1.25)])
def test_unequal_ratios_give_back_the_manufactured_order_and_limit(order, ratios):
    coarse, fine = ratios
    values = _ladder(order, [coarse * fine, fine, 1.0], limit=3.0)
    x = cv.richardson(values, ratio=fine, coarse_ratio=coarse)
    assert x.status == "monotone"
    assert x.order == pytest.approx(order, abs=1e-9)
    assert x.extrapolated == pytest.approx(3.0, rel=1e-12)
    finest = list(values)[-1]
    assert x.error_vs_extrapolated[finest] == pytest.approx(0.01 / 3.0, rel=1e-9)
    assert x.gci_fine == pytest.approx(
        1.25 * (0.01 * (fine**order - 1) / values[finest]) / (fine**order - 1)
    )


def test_equal_ratios_are_the_closed_form_bit_for_bit():
    values = {"1": 1.04, "2": 1.0113, "4": 1.0025}
    plain = cv.richardson(values, ratio=2.0)
    assert cv.richardson(values, ratio=2.0, coarse_ratio=2.0) == plain
    assert plain.order == math.log((1.04 - 1.0113) / (1.0113 - 1.0025)) / math.log(2.0)


def test_unequal_ratios_keep_the_guards():
    def status(values):
        return cv.richardson(values, ratio=2.0, coarse_ratio=3.0).status

    assert status({"1": 1.0, "3": 1.1, "6": 1.05}) == "oscillatory"
    assert status({"1": 1.1, "3": 1.0, "6": 1.0}) == "flat"
    assert status({"1": 1.0, "3": 1.0, "6": 1.05}) == "diverging"
    # At order zero the coarse change is ln 3 / ln 2 of the fine one: a change
    # that is larger than the fine one but below that limit has no order.
    limit = math.log(3.0) / math.log(2.0)
    assert status({"1": 1.0 + 0.01 * (1 + limit * 0.99), "3": 1.01, "6": 1.0}) == (
        "diverging"
    )
    just = cv.richardson(
        {"1": 1.0 + 0.01 * (1 + limit * 1.01), "3": 1.01, "6": 1.0},
        ratio=2.0,
        coarse_ratio=3.0,
    )
    assert just.status == "monotone" and 0.0 < just.order < 0.05
    with pytest.raises(ValueError, match="exceed 1"):
        cv.richardson({"1": 1.2, "2": 1.1, "3": 1.0}, ratio=1.0, coarse_ratio=2.0)
