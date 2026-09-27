"""Temporal measures (plan 2b): settling, separation, rise time, the frame probe."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("scipy")

from test_convergence import case, grid  # noqa: E402

from structbench.verification import temporal  # noqa: E402


def _case_with_displacement(fn, frames=5, *, force=None):
    """A 2x2 grid whose x-displacement is fn(t) on every node, t in [0, 1]."""
    xy, conn = grid(2, 2)
    t = np.linspace(0.0, 1.0, frames)
    u = np.zeros((frames, len(xy), 2), np.float32)
    u[:, :, 0] = fn(t)[:, None]
    c = case(xy, conn, node={"displacement": u}, frames=frames)
    globals_ = {} if force is None else {"reaction_force": np.asarray(force, float)}
    return dataclasses.replace(
        c, response=dataclasses.replace(c.response, globals_=globals_)
    )


def test_response_until_keeps_frames_up_to_and_including_the_one_named():
    c = _case_with_displacement(lambda t: t, frames=5, force=np.arange(5.0))
    t = temporal.response_until(c, 2)
    assert t.response.time.shape == (3,)
    assert t.response.node["displacement"].shape[0] == 3
    assert t.response.element["solid"]["stress"].shape[0] == 3
    assert t.response.globals_["reaction_force"].tolist() == [0.0, 1.0, 2.0]
    assert c.response.time.shape == (5,)  # the original is untouched
    with pytest.raises(ValueError):
        temporal.response_until(c, 5)


def test_qoi_history_evaluates_the_hook_frame_by_frame():
    c = _case_with_displacement(lambda t: 4.0 * t, frames=5)
    hist = temporal.qoi_history(
        c, lambda k: {"ux": float(k.response.node["displacement"][-1, 0, 0])}
    )
    assert hist["ux"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_settling_frame_is_the_last_frame_outside_tolerance():
    hist = {"q": np.array([0.0, 0.5, 0.9, 0.99, 0.999, 1.0])}
    assert temporal.settling_frame(hist, {"q": 0.05}) == 2  # 0.9 is 10 % off
    assert temporal.settling_frame(hist, {"q": 0.4}) == 1
    assert temporal.settling_frame({"q": np.ones(4)}, {"q": 0.01}) == 0
    two = {"a": np.array([0.0, 1.0, 1.0]), "b": np.array([0.0, 0.0, 1.0])}
    assert temporal.settling_frame(two, {"a": 0.01, "b": 0.01}) == 1  # b settles last


def test_settling_frame_of_a_quantity_ending_at_zero_uses_an_absolute_floor():
    hist = {"q": np.array([1.0, 0.5, 0.0, 0.0])}
    assert temporal.settling_frame(hist, {"q": 0.01}) == 1


def test_separation_frame_and_rise_time():
    force = np.array([0.0, 4.0, 10.0, 6.0, 0.005, 0.0, 0.0])
    # threshold 1e-3 * 10 = 0.01; frames above it are 1, 2, 3; 0.005 is below
    assert temporal.separation_frame(force) == 3
    assert temporal.separation_frame(-force) == 3
    assert temporal.separation_frame(np.zeros(3)) is None
    # 10 % of the peak (1.0) is first crossed at frame 2, 90 % (9.0) at frame 3
    assert temporal.rise_time_frames(np.array([0.0, 0.5, 2.0, 9.5, 10.0])) == 1
    assert temporal.rise_time_frames(np.array([10.0, 5.0, 0.0])) is None
    assert temporal.rise_time_frames(np.zeros(4)) is None


def test_midpoint_interpolation_error_is_zero_for_linear_and_positive_for_curved():
    linear = _case_with_displacement(lambda t: t, force=np.linspace(0.0, 2.0, 5))
    curved = _case_with_displacement(
        lambda t: t * t, force=np.linspace(0.0, 2.0, 5) ** 2
    )
    zero = temporal.midpoint_interpolation_errors(linear)
    assert set(zero) == {"node/displacement", "solid/stress", "global/reaction_force"}
    assert zero["node/displacement"] == pytest.approx(0.0, abs=1e-6)
    assert zero["global/reaction_force"] == pytest.approx(0.0, abs=1e-12)
    bent = temporal.midpoint_interpolation_errors(curved)
    # midpoints of t^2 at t = .25, .75 read .125 and .625 against .0625 and .5625
    assert bent["node/displacement"] == pytest.approx(
        0.0625 * np.sqrt(2) / np.hypot(0.0625, 0.5625), rel=1e-3
    )
    assert bent["global/reaction_force"] > 0.1
    with pytest.raises(ValueError, match="stride"):
        temporal.midpoint_interpolation_errors(
            _case_with_displacement(lambda t: t, frames=4)
        )


def test_common_instant_errors_align_by_stride_and_refuse_misaligned_clocks():
    coarse = _case_with_displacement(lambda t: t, frames=3, force=[0.0, 1.0, 2.0])
    fine = _case_with_displacement(
        lambda t: t, frames=5, force=[0.0, 0.5, 1.0, 1.5, 2.0]
    )
    errors = temporal.common_instant_errors(coarse, fine)
    assert errors["node/displacement"] == pytest.approx(0.0, abs=1e-6)
    assert errors["global/reaction_force"] == pytest.approx(0.0, abs=1e-12)
    shifted = _case_with_displacement(
        lambda t: t + 0.1, frames=5, force=[0.0, 0.5, 1.0, 1.5, 2.0]
    )
    assert temporal.common_instant_errors(coarse, shifted)["node/displacement"] > 0.05
    same = temporal.common_instant_errors(coarse, coarse)  # stride 1
    assert same["node/displacement"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="stride"):
        temporal.common_instant_errors(
            coarse, _case_with_displacement(lambda t: t, frames=4)
        )


def test_common_instant_errors_skip_fields_one_side_lacks():
    coarse = _case_with_displacement(lambda t: t, frames=3)
    fine = _case_with_displacement(
        lambda t: t, frames=5, force=[0.0, 0.5, 1.0, 1.5, 2.0]
    )
    assert "global/reaction_force" not in temporal.common_instant_errors(coarse, fine)


def test_interpolation_errors_take_a_stride_and_nan_is_never_a_rise():
    fine = _case_with_displacement(lambda t: t * t, frames=9)  # t = 0, .125, ..., 1
    by_two = temporal.interpolation_errors(fine, 2)
    by_four = temporal.interpolation_errors(fine, 4)
    assert by_four["node/displacement"] > by_two["node/displacement"] > 0.0
    assert temporal.midpoint_interpolation_errors(fine) == by_two
    with pytest.raises(ValueError, match="stride"):
        temporal.interpolation_errors(fine, 3)  # 8 intervals are not strides of 3
    with pytest.raises(ValueError, match="stride"):
        temporal.interpolation_errors(fine, 1)
    assert temporal.rise_time_frames(np.array([0.0, np.nan, 1.0])) is None
