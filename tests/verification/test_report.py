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
            1.0003531234567,
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


def test_the_report_separates_findings_data_gaps_and_platform_gaps() -> None:
    text = render_markdown(judge(_dataset()))
    verdicts, rest = text.split("## Findings")
    findings, rest = rest.split("## Not applicable to these runs")
    skipped, rest = rest.split("## Evidence the runs did not supply")
    data_gaps, rest = rest.split("## Not yet checked by this instrument")
    platform_gaps, criteria = rest.split("## Criteria")

    assert "| `yield_table_monotone` | 1 | 0 … 1 | 1 | 1 |" in verdicts
    # only measured quantities are table rows
    for name in ("eos_closure", "energy_gain_max", "implicit_convergence"):
        assert f"`{name}`" not in verdicts
    assert skipped.strip() == "`implicit_convergence`"
    # measured with no criterion: published with its value and no verdict
    assert "| `yield_ratio_max` | 1 | 1.00035 |  |  |  |  | 2 | none |" in verdicts
    # an indicator with no level for this scope shows the levels that do exist
    assert (
        "out of scope"
        not in verdicts.split("`input_density_plausible`")[1].split("\n")[0]
    )

    assert "- **fail** `yield_table_monotone` = 1 — `T-1`" in findings
    assert "- **review** `input_density_plausible` = 8.9e-09 — `T-1`, `T-2`" in findings
    assert "- `source_missing` (E5), 2 cases: `energy_gain_max`" in data_gaps
    assert "- `eos_closure` — unsupported" in platform_gaps
    assert "`energy_gain_max`" not in platform_gaps
    assert "`yield_table_monotone` <= 0 any run (requirement)." in criteria
    assert "Source: M-D6, M-D5, M-D10." in criteria
    assert "energy_gain_max" not in criteria  # unused criteria are not listed


def test_no_artefact_carries_a_path_a_host_or_a_licence() -> None:
    data = _dataset()
    for text in (to_json(data), render_markdown(judge(data))):
        assert not re.search(r"[A-Za-z]:\\|/home/|/Users/|\\\\|OneDrive", text)
        assert not re.search(r"(?i)licen[cs]e|hostname|@[a-z0-9-]+\.", text)
