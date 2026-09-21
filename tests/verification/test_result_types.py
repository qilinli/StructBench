"""Tests for the verification result types (ADR-0066)."""

from __future__ import annotations

import math

import pytest

from structbench.core import Absence, AbsenceReason, EvidenceItem
from structbench.verification import (
    CaseMeasurements,
    Category,
    DatasetMeasurements,
    Location,
    Measurement,
    Status,
    Verdict,
)


def _measured(quantity: str = "nonfinite_count", value: float = 0.0) -> Measurement:
    return Measurement(quantity, value, "1", frozenset({Location.CASE}), n_samples=12)


def test_vocabularies() -> None:
    assert {v.value for v in Verdict} == {
        "pass",
        "fail",
        "review",
        "not_applicable",
        "not_assessable",
    }
    assert Category.UNITS.value == "units"
    assert {s.value for s in Status} == {"specified", "implemented"}
    assert {loc.value for loc in Location} == {"case", "input", "run", "declared"}


def test_a_measurement_has_a_value_or_an_absence_never_both() -> None:
    absent = Measurement(
        "energy_gain_max",
        None,
        "1",
        absence=Absence(AbsenceReason.SOURCE_MISSING, frozenset({EvidenceItem.E5})),
    )
    assert absent.value is None
    with pytest.raises(ValueError, match="absence"):
        Measurement("energy_gain_max", None, "1")
    with pytest.raises(ValueError, match="absence"):
        Measurement(
            "energy_gain_max", 0.1, "1", absence=Absence(AbsenceReason.UNSUPPORTED)
        )


def test_not_applicable_carries_neither_value_nor_absence() -> None:
    row = Measurement(
        "zero_energy_mode_peak_over_internal_peak", None, "1", not_applicable=True
    )
    assert row.not_applicable
    with pytest.raises(ValueError, match="not_applicable"):
        Measurement("x", 1.0, "1", not_applicable=True)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_measured_value_is_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Measurement("nonfinite_count", bad, "1")


def test_detail_strings_are_tokens_not_free_text() -> None:
    ok = Measurement("yield_ratio_max", 1.0, "1", detail={"frame": 7, "kind": "sph"})
    assert ok.detail["kind"] == "sph"
    with pytest.raises(ValueError, match="detail"):
        Measurement("yield_ratio_max", 1.0, "1", detail={"where": "C:\runs\a b"})


def test_case_measurements_are_sorted_and_unique() -> None:
    rows = (_measured("a_quantity"), _measured("b_quantity"))
    case = CaseMeasurements("T-1", None, frozenset({"explicit"}), frozenset(), rows)
    assert [m.quantity for m in case.measurements] == ["a_quantity", "b_quantity"]
    with pytest.raises(ValueError, match="sorted"):
        CaseMeasurements("T-1", None, frozenset(), frozenset(), rows[::-1])
    with pytest.raises(ValueError, match="sorted"):
        CaseMeasurements("T-1", None, frozenset(), frozenset(), (rows[0], rows[0]))


def test_dataset_measurements_sort_cases_by_id() -> None:
    a = CaseMeasurements("A", None, frozenset(), frozenset(), ())
    b = CaseMeasurements("B", None, frozenset(), frozenset(), ())
    data = DatasetMeasurements("bench", "v0", "0.3.0", {"nonfinite_count": 1}, (a, b))
    assert [c.case_id for c in data.cases] == ["A", "B"]
    with pytest.raises(ValueError, match="sorted"):
        DatasetMeasurements("bench", "v0", "0.3.0", {}, (b, a))
