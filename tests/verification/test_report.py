"""Tests for the JSON record and the generated Markdown report (ADR-0066 cl. 8)."""

from __future__ import annotations

import json
import re

import pytest

from structbench.core import Absence, AbsenceReason, EvidenceItem
from structbench.verification.criteria import judge
from structbench.verification.quantities import get_quantity
from structbench.verification.report import (
    SCHEMA_ID,
    from_json,
    render_markdown,
    to_json,
)
from structbench.verification.results import (
    CaseMeasurements,
    DatasetMeasurements,
    Location,
    Measurement,
)


def _value(name: str, number: float, **kw: object) -> Measurement:
    return Measurement(name, number, get_quantity(name).unit, **kw)  # type: ignore[arg-type]


def _absent(name: str, reason: AbsenceReason, *missing: EvidenceItem) -> Measurement:
    return Measurement(
        name, None, get_quantity(name).unit, absence=Absence(reason, frozenset(missing))
    )


def _case(case_id: str, table_dips: float) -> CaseMeasurements:
    rows = [
        _absent("eos_closure", AbsenceReason.UNSUPPORTED),
        _absent("energy_gain_max", AbsenceReason.SOURCE_MISSING, EvidenceItem.E5),
        Measurement("implicit_convergence", None, "1", not_applicable=True),
        _value("input_density_plausible", 8.9e-9),
        _value("nonfinite_count", 0.0, n_samples=23355104),
        _value(
            "yield_ratio_max",
            1.0003531234567 + 0.001 * (1.0 - table_dips),  # differs by case
            locations=frozenset({Location.CASE, Location.DECLARED}),
            detail={"frame": 33, "state_at_max": 0.500852123456},
        ),
        _value("yield_table_monotone", table_dips),
    ]
    rows.sort(key=lambda m: m.quantity)
    return CaseMeasurements(
        case_id, "ab" * 32, frozenset({"explicit", "dim2"}), frozenset(), tuple(rows)
    )


def _dataset() -> DatasetMeasurements:
    versions = {"nonfinite_count": 1, "yield_ratio_max": 1}
    cases = (_case("T-1", 1.0), _case("T-2", 0.0))
    return DatasetMeasurements("taylor_impact_2d", "v0.1.0", "0.3.0", versions, cases)


def test_the_record_round_trips_byte_for_byte() -> None:
    text = to_json(_dataset())
    assert to_json(from_json(text)) == text
    assert text.endswith("\n") and "\r" not in text
    assert json.loads(text)["schema"] == SCHEMA_ID


def test_values_are_stored_to_nine_significant_digits() -> None:
    raw = json.loads(to_json(_dataset()))
    row = next(
        m for m in raw["cases"][0]["measurements"] if m["quantity"] == "yield_ratio_max"
    )
    assert row["value"] == 1.00035312
    assert row["detail"] == {"frame": 33, "state_at_max": 0.500852123}
    assert row["locations"] == ["case", "declared"]


def test_empty_fields_are_left_out_of_the_record() -> None:
    raw = json.loads(to_json(_dataset()))
    row = next(
        m
        for m in raw["cases"][0]["measurements"]
        if m["quantity"] == "implicit_convergence"
    )
    assert row == {
        "quantity": "implicit_convergence",
        "unit": "1",
        "not_applicable": True,
    }


def test_another_schema_is_refused() -> None:
    text = to_json(_dataset()).replace(SCHEMA_ID, "structbench.verification/0")
    with pytest.raises(ValueError, match="schema"):
        from_json(text)


def test_the_report_is_regenerated_from_the_record_alone() -> None:
    direct = render_markdown(judge(_dataset()))
    from_record = render_markdown(judge(from_json(to_json(_dataset()))))
    assert direct == from_record
    assert direct.endswith("\n") and not direct.endswith("\n\n")


def _sections(text: str) -> dict[str, str]:
    parts = text.split("\n## ")
    return {part.split("\n", 1)[0]: part for part in parts[1:]}


def test_the_report_opens_with_a_bottom_line() -> None:
    text = render_markdown(judge(_dataset()))
    assert text.startswith("# taylor_impact_2d — reference-data verification\n")
    summary = _sections(text)["Summary"]
    assert "**1 check passes** wherever it applies." in summary
    assert "**1 finding**: dips in the input's hardening table." in summary
    assert "**2 quantities are measured but not judged**" in summary
    assert "**2 checks could not be made**" in summary
    assert "1 check does not apply to these runs." in summary
    # the scorecard counts quantities by category
    assert "| Data integrity | 1 | 1 |  |  |  |" in summary
    assert "| Energy and mass conservation |  |  |  | 2 |  |" in summary


def test_a_finding_says_what_was_found_what_was_required_and_what_it_means() -> None:
    findings = _sections(render_markdown(judge(_dataset())))["Findings"]
    assert "### Dips in the input's hardening table — fail" in findings
    assert "- Found: 1 in `T-1`." in findings
    assert "- Required: must be zero." in findings
    assert "- What it means: the input's hardening table is not monotone" in findings
    assert "review" not in findings  # no sourced level is ratified: none judges


def test_results_are_grouped_by_category_in_plain_words() -> None:
    results = _sections(render_markdown(judge(_dataset())))["Results by category"]
    integrity = results.split("### Data integrity")[1].split("###")[0]
    assert (
        "| Non-finite values (NaN, infinity) | 0 | all cases | pass | must be zero |"
        in integrity
    )
    assert (
        "| Dips in the input's hardening table | 0 to 1 | `T-1` | **fail** (1 of 2) |"
        in integrity
    )
    assert "nonfinite_count" not in results  # names a reader never sees
    health = results.split("### Numerical health of the runs")[1].split("###")[0]
    assert "Does not apply to these runs: implicit increments accepted" in health


def test_unjudged_numbers_carry_their_context_but_no_verdict() -> None:
    text = render_markdown(judge(_dataset()))
    measured = _sections(text)["Measured, not judged"]
    assert "**Most extreme input density**: 8.9e-09 kg/m^3." in measured
    shown = (
        "not confirmed by this platform: between 16 kg/m^3 and 22590 kg/m^3"
        " [M-D6, M-D5, M-D10]"
    )
    assert shown in measured
    assert (
        "**Largest stress relative to the yield surface**: 1.00035 to 1.00135"
        in measured
    )
    assert "(worst: `T-2`)" in measured
    assert "not this platform's standard" in measured


def test_what_could_not_be_checked_says_why_in_plain_words() -> None:
    gaps = _sections(render_markdown(judge(_dataset())))["What could not be checked"]
    assert (
        "- Because the runs did not supply the global energy ledger: largest energy"
        " gain during the run." in gaps
    )
    assert (
        "- Because this instrument cannot measure it yet: pressure against the"
        " equation of state." in gaps
    )


def test_per_case_values_list_only_the_unjudged_quantities_that_vary() -> None:
    table = _sections(render_markdown(judge(_dataset())))["Per-case values"]
    assert "| Case | Largest stress relative to the yield surface |" in table
    assert "| `T-1` | 1.00035 |" in table and "| `T-2` | 1.00135 |" in table
    assert "input density" not in table  # identical in every case


def test_the_reading_guide_gives_each_applied_bound_its_rationale() -> None:
    guide = _sections(render_markdown(judge(_dataset())))["How to read this report"]
    assert (
        "*Non-finite values (NaN, infinity)* — must be zero. A stored response" in guide
    )
    assert "Published levels shown for context (not applied)" in guide
    assert "*Most extreme input density* — between 16 kg/m^3 and 22590 kg/m^3" in guide
    assert "Largest energy gain" not in guide  # never measured here: nothing to show


def test_no_artefact_carries_a_path_a_host_or_a_licence() -> None:
    data = _dataset()
    for text in (to_json(data), render_markdown(judge(data))):
        assert not re.search(r"[A-Za-z]:\\|/home/|/Users/|\\\\|OneDrive", text)
        assert not re.search(r"(?i)licen[cs]e|hostname|@[a-z0-9-]+\.", text)
