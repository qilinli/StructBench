"""The gradient task family's estimators and record (ADR-0076)."""

import json
import math

import numpy as np
import pytest

from structbench.datasets.canonical import CaseTrajectory
from structbench.eval.sensitivity import (
    GradientFamily,
    QoiNoise,
    SensitivityPoint,
    free_flight_seed,
    gradient_floor,
    gradient_metrics,
    model_sensitivity,
    reference_record,
    reference_sensitivities,
    render_reference_markdown,
    sensitivities_from_record,
)

V = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]


def test_central_difference_is_exact_on_a_quadratic():
    # d/dv (a v² + b v + c) = 2 a v + b; the central difference is exact and
    # the 2h estimate agrees, so the truncation estimate is rounding only.
    q = [0.01 * v * v - 2.0 * v + 5.0 for v in V]
    pts = reference_sensitivities(V, q)
    for p in pts[1:-1]:
        assert p.sensitivity == pytest.approx(0.02 * p.value - 2.0, abs=1e-9)
        assert p.truncation_error < 1e-9
        assert not p.one_sided
    # ends are one-sided: first-order, so not exact, and flagged
    assert pts[0].one_sided and pts[-1].one_sided
    assert pts[0].truncation_error > 1e-6


def test_truncation_estimate_recovers_the_cubic_error():
    # For q = v³ the central-difference error at step h is exactly h², and
    # |s_h − s_2h| = 3 h² (three times the leading term, as documented).
    h = 10.0
    q = [v**3 for v in V]
    pts = reference_sensitivities(V, q)
    interior_full = [p for i, p in enumerate(pts) if 2 <= i <= len(pts) - 3]
    for p in interior_full:
        assert p.sensitivity - 3 * p.value**2 == pytest.approx(h * h, rel=1e-9)
        assert p.truncation_error == pytest.approx(3 * h * h, rel=1e-9)


def test_order_of_input_does_not_matter_and_grid_must_be_uniform():
    q = [v**2 for v in V]
    shuffled = list(zip(V, q, strict=True))[::-1]
    pts = reference_sensitivities([v for v, _ in shuffled], [x for _, x in shuffled])
    assert [p.value for p in pts] == V
    with pytest.raises(ValueError, match="uniform"):
        reference_sensitivities([1.0, 2.0, 4.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="three"):
        reference_sensitivities([1.0, 2.0], [1.0, 2.0])


def test_measurement_error_propagates_the_declared_noise():
    pts = reference_sensitivities(V, [1.0] * len(V), noise=QoiNoise(absolute=0.5))
    # central: √2·σ/2h ; one-sided: √2·σ/h
    assert pts[3].measurement_error == pytest.approx(math.sqrt(2) * 0.5 / 20.0)
    assert pts[0].measurement_error == pytest.approx(math.sqrt(2) * 0.5 / 10.0)
    rel = reference_sensitivities(V, [200.0] * len(V), noise=QoiNoise(relative=0.01))
    assert rel[3].measurement_error == pytest.approx(math.sqrt(2) * 2.0 / 20.0)


def test_judged_rule_is_half_the_sensitivity():
    judged = SensitivityPoint(1.0, 2.0, 1.0, 0.5, False)
    reported = SensitivityPoint(1.0, 2.0, 1.01, 0.0, False)
    assert judged.judged and judged.error == 1.0
    assert not reported.judged


def test_model_sensitivity_and_floor():
    assert model_sensitivity(q_minus=1.0, q_plus=3.0, delta=1.0) == 1.0
    assert gradient_floor(1.0, 1.25) == 0.25
    with pytest.raises(ValueError):
        model_sensitivity(1.0, 2.0, 0.0)


def test_gradient_metrics_pool_judged_points_only():
    ref = (
        SensitivityPoint(1.0, 2.0, 0.1, 0.0, False),  # judged
        SensitivityPoint(2.0, -2.0, 0.1, 0.0, False),  # judged
        SensitivityPoint(3.0, 0.1, 5.0, 0.0, False),  # reported
    )
    model = [2.0, 2.0, 100.0]  # second sign wrong; third ignored
    m = gradient_metrics(ref, model, floor=[0.2, 0.2, 50.0])
    assert m["n_judged"] == 2 and m["n_reported"] == 1
    # ‖(0, 4)‖ / ‖(2, −2)‖ = 4 / √8
    assert m["rel_l2"] == pytest.approx(4 / math.sqrt(8))
    assert m["sign"] == 0.5
    assert m["floor"] == pytest.approx(math.sqrt(0.08) / math.sqrt(8))
    none = gradient_metrics((ref[2],), [1.0])
    assert math.isnan(none["rel_l2"]) and none["n_judged"] == 0
    with pytest.raises(ValueError):
        gradient_metrics(ref, [1.0])


def _free_flight_case(
    v_m_s: float, *, frames: int = 10, wall: bool = False
) -> CaseTrajectory:
    t = np.arange(frames, dtype=float) * 2e-6
    x0 = np.stack([np.linspace(0.0, 10.0, 6), np.zeros(6)], axis=1)  # mm
    pos = np.repeat(x0[None], frames, axis=0)
    pos[..., 0] -= v_m_s * 1e3 * t[:, None]  # fired toward −x, mm
    ptype = np.ones(6, dtype=np.int64)
    if wall:
        pos = np.concatenate([pos, np.tile([[-2.0, 0.0]], (frames, 1, 1))], axis=1)
        ptype = np.concatenate([ptype, np.array([2])])
    return CaseTrajectory(
        case_id=f"T-{v_m_s:g}",
        positions=pos.astype(np.float32),
        particle_type=ptype,
        aux=np.zeros((frames, pos.shape[1]), dtype=np.float32),
        time=t,
    )


def test_free_flight_seed_translates_the_lattice_at_the_new_velocity():
    case = _free_flight_case(150.0, wall=True)
    new = free_flight_seed(
        case,
        velocity_from=150.0,
        velocity_to=152.0,
        seed_frames=6,
        kinematic_types=(2,),
    )
    target = _free_flight_case(152.0, wall=True)
    np.testing.assert_allclose(new.positions[:6], target.positions[:6], atol=1e-4)
    # frames beyond the seed are the stored ones, untouched
    np.testing.assert_array_equal(new.positions[6:], case.positions[6:])
    # the wall never moves
    np.testing.assert_array_equal(new.positions[:, -1], case.positions[:, -1])
    assert new.case_id == case.case_id


def test_free_flight_seed_refuses_a_seed_that_is_not_free_flight():
    case = _free_flight_case(150.0)
    bent = case.positions.copy()
    bent[3, 0, 1] += 0.5  # a particle moves sideways inside the seed
    case = CaseTrajectory(case.case_id, bent, case.particle_type, case.aux, case.time)
    with pytest.raises(ValueError, match="free flight"):
        free_flight_seed(case, velocity_from=150.0, velocity_to=151.0, seed_frames=6)
    with pytest.raises(ValueError, match="free flight"):
        free_flight_seed(
            _free_flight_case(150.0),
            velocity_from=100.0,
            velocity_to=101.0,
            seed_frames=6,
        )


def _family() -> GradientFamily:
    return GradientFamily(
        parameter="v_m_s",
        parameter_of=lambda cid: float(cid.split("-")[2]),
        line_of=lambda cid: f"L{cid.split('-')[1]}",
        noise={
            "length": QoiNoise(absolute=0.5, basis="a spacing"),
            "peak": QoiNoise(relative=0.01, basis="1 %"),
        },
    )


def test_family_validates_its_deltas():
    with pytest.raises(ValueError, match="differ"):
        GradientFamily("p", float, str, {}, deltas=(1.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        GradientFamily("p", float, str, {}, deltas=(0.0, 1.0))


def test_reference_record_round_trips_and_recomputes():
    fam = _family()
    cases = {}
    for L in (60, 80):
        for v in (100, 110, 120, 130):
            cases[f"T-{L}-{v}"] = {"length": L - 0.2 * v, "peak": 180.0 + 0.1 * v}
    rec = reference_record(
        fam, benchmark="demo", quantities=cases, source={"archive": "x"}
    )
    payload = json.loads(json.dumps(rec))
    assert list(payload["lines"]) == ["L60", "L80"]
    line = payload["lines"]["L60"]
    assert line["values"] == [100.0, 110.0, 120.0, 130.0]
    assert line["cases"][0] == "T-60-100"
    # d(length)/dv = −0.2 exactly; judged everywhere (noise 0.5 → meas 0.035)
    for p in line["sensitivities"]["length"]:
        assert p["sensitivity"] == pytest.approx(-0.2)
        assert p["judged"] is True
    # recomputation from the stored quantities matches the stored sensitivities
    again = sensitivities_from_record(payload)
    for name, pts in line["sensitivities"].items():
        for stored, re in zip(pts, again["L60"][name], strict=True):
            assert stored["sensitivity"] == pytest.approx(re.sensitivity)
            assert stored["judged"] == re.judged
    md = render_reference_markdown(payload)
    assert "do not edit by hand" in md
    assert "## Line L60" in md and "## Line L80" in md
    assert "| `length` |" in md and "a spacing" in md
