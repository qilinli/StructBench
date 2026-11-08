"""BenchmarkCard invariants (ADR-0027)."""

import json

import pytest

from structbench.benchmarks.card import (
    COMPLIANCE_STATUSES,
    STANDARD_ITEMS,
    STRUCTURAL_SCALES,
    BenchmarkCard,
    BenchmarkFigure,
    ComplianceRow,
    GroundTruth,
    ParameterSpace,
)


def _kwargs(**overrides):
    base = dict(
        name="Demo-Bench",
        version="0.1",
        description="A demo benchmark.",
        provenance="Synthetic, for tests.",
        data_license="CC BY 4.0",
        solver="LS-DYNA",
        discretisation="SPH",
        materials=("*MAT_ELASTIC",),
        erosion=False,
        loading="rigid-wall impact",
        source_units="g-mm-ms",
        geometry="2D bar",
        n_cases=3,
        splits={"train": 2, "val": 1},
        task="autoregressive transition",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=("final_length",),
        fields=("positions", "stress"),
        particles_per_case="100",
        n_frames=10,
        output_dt_ms=0.1,
        input_frames=6,
        protocol_rationale="test-only rationale",
    )
    base.update(overrides)
    return base


def test_card_accepts_consistent_splits():
    card = BenchmarkCard(**_kwargs())
    assert card.n_cases == 3
    assert card.size_gb is None


def test_card_rejects_split_sum_mismatch():
    with pytest.raises(ValueError, match="n_cases"):
        BenchmarkCard(**_kwargs(n_cases=99))


def test_card_rejects_input_frames_below_two():
    # Needs >= 2 frames to form one velocity (ADR-0035).
    with pytest.raises(ValueError, match="input_frames must be >= 2"):
        BenchmarkCard(**_kwargs(input_frames=1))


def test_card_json_dict_serializes():
    card = BenchmarkCard(**_kwargs())
    payload = json.dumps(card.to_json_dict())
    assert "Demo-Bench" in payload


def test_card_defaults_to_no_overview_or_figures():
    card = BenchmarkCard(**_kwargs())
    assert card.overview == ""
    assert card.figures == ()


def test_card_json_dict_serializes_figures():
    # asdict recurses into nested BenchmarkFigure records (ADR-0036).
    card = BenchmarkCard(
        **_kwargs(
            overview="## Demo\n\nText.",
            figures=(BenchmarkFigure(path="assets/x.png", caption="a plot"),),
        )
    )
    payload = json.loads(json.dumps(card.to_json_dict()))
    assert payload["figures"][0]["path"] == "assets/x.png"
    assert payload["overview"].startswith("## Demo")


@pytest.mark.parametrize("bad", [{"path": "  "}, {"caption": " "}])
def test_figure_rejects_blank_path_or_caption(bad):
    kwargs = {"path": "assets/x.png", "caption": "ok"}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        BenchmarkFigure(**kwargs)


def test_figure_alt_defaults_blank():
    assert BenchmarkFigure(path="assets/x.png", caption="c").alt == ""


# --- V&V identity fields and the compliance table (ADR-0065 follow-up 2,
# plan of 2026-11-08) ---


def _full_compliance(status="missing"):
    return tuple(ComplianceRow(item, status, "a reason") for item in STANDARD_ITEMS)


def test_vv_fields_default_to_undeclared():
    card = BenchmarkCard(**_kwargs())
    assert card.structural_scale == ""
    assert card.load_type == ""
    assert card.autonomy == ""
    assert card.parameter_space is None
    assert card.ground_truth is None
    assert card.compliance == ()
    assert card.compliance_covers_standard() is False


def test_structural_scale_is_a_fixed_vocabulary():
    for scale in STRUCTURAL_SCALES:
        assert BenchmarkCard(**_kwargs(structural_scale=scale)).structural_scale
    with pytest.raises(ValueError, match="structural_scale"):
        BenchmarkCard(**_kwargs(structural_scale="bridge"))


def test_standard_items_are_the_d14_items_plus_the_blind_row():
    # ADR-0073 D14 names four required items and two recorded-where-they-exist;
    # ADR-0074 adds the blind set. Order is render order.
    assert tuple(STANDARD_ITEMS) == (
        "convergence_run",
        "noise_floor",
        "survivor_log",
        "constitutive_sensitivity",
        "physical_test_anchor",
        "second_solver_reproduction",
        "blind_evaluation",
    )
    assert COMPLIANCE_STATUSES == ("pass", "missing", "not_applicable", "never")


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"item": "unit_test"}, "unknown standard item"),
        ({"status": "maybe"}, "unknown compliance status"),
        ({"reason": "   "}, "needs a reason"),
    ],
)
def test_compliance_row_rejects_unknown_item_status_or_blank_reason(bad, match):
    kwargs = {"item": "noise_floor", "status": "pass", "reason": "measured"}
    kwargs.update(bad)
    with pytest.raises(ValueError, match=match):
        ComplianceRow(**kwargs)


def test_card_rejects_a_repeated_compliance_item():
    rows = (
        ComplianceRow("noise_floor", "pass", "measured"),
        ComplianceRow("noise_floor", "missing", "not measured"),
    )
    with pytest.raises(ValueError, match="repeats"):
        BenchmarkCard(**_kwargs(compliance=rows))


def test_compliance_covers_standard_needs_every_item():
    full = BenchmarkCard(**_kwargs(compliance=_full_compliance()))
    assert full.compliance_covers_standard() is True
    partial = BenchmarkCard(**_kwargs(compliance=_full_compliance()[:-1]))
    assert partial.compliance_covers_standard() is False


def test_parameter_space_validates_design_and_box():
    with pytest.raises(ValueError, match="design"):
        ParameterSpace(design=" ")
    with pytest.raises(ValueError, match="exceeds"):
        ParameterSpace(design="grid", box={"v_m_s": (200.0, 100.0)})
    space = ParameterSpace(design="grid", box={"v_m_s": (100.0, 100.0)})
    assert space.margin == "" and space.categorical == {}


def test_ground_truth_states_regime_and_uncertainty():
    with pytest.raises(ValueError, match="regime"):
        GroundTruth(regime="", uncertainty="not measured")
    with pytest.raises(ValueError, match="uncertainty"):
        GroundTruth(regime="SPH", uncertainty=" ")


def test_card_json_dict_serializes_vv_records():
    card = BenchmarkCard(
        **_kwargs(
            structural_scale="specimen",
            parameter_space=ParameterSpace(
                design="grid 3 x 7",
                box={"v_m_s": (100.0, 190.0)},
                categorical={"shape": ("plate", "rod")},
            ),
            ground_truth=GroundTruth(regime="SPH, one run", uncertainty="none"),
            compliance=_full_compliance(),
        )
    )
    payload = json.loads(json.dumps(card.to_json_dict()))
    assert payload["structural_scale"] == "specimen"
    assert payload["parameter_space"]["box"]["v_m_s"] == [100.0, 190.0]
    assert payload["parameter_space"]["categorical"]["shape"] == ["plate", "rod"]
    assert payload["ground_truth"]["uncertainty"] == "none"
    assert [r["item"] for r in payload["compliance"]] == list(STANDARD_ITEMS)
