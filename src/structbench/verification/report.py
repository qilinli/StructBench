"""The committed record (JSON) — ADR-0066 clause 8. The report is ``markdown.py``.

Measurements are what is committed; verdicts are generated from them and the
criteria, so ``render_markdown(judge(from_json(text)))`` needs no data. Both
outputs are deterministic and carry no time, host, or path.
"""

from __future__ import annotations

import json
from typing import Any

from ..core import Absence, AbsenceReason, EvidenceItem
from .markdown import render_markdown
from .results import (
    CaseMeasurements,
    DatasetMeasurements,
    Location,
    Measurement,
)

__all__ = ["SCHEMA_ID", "from_json", "render_markdown", "to_json"]

SCHEMA_ID = "structbench.verification/1"


def _round(x: float | int | str) -> float | int | str:
    """Nine significant digits: past float32, short of platform noise."""
    return float(f"{x:.9g}") if isinstance(x, float) else x


def _measurement(m: Measurement) -> dict[str, Any]:
    out: dict[str, Any] = {"quantity": m.quantity, "unit": m.unit}
    if m.value is not None:
        out["value"] = _round(m.value)
    if m.locations:
        out["locations"] = sorted(str(loc) for loc in m.locations)
    if m.n_samples is not None:
        out["n_samples"] = m.n_samples
    if m.absence is not None:
        out["absence"] = {
            "reason": str(m.absence.reason),
            "missing": sorted(str(item) for item in m.absence.missing),
        }
    if m.not_applicable:
        out["not_applicable"] = True
    if m.detail:
        out["detail"] = {k: _round(v) for k, v in m.detail.items()}
    return out


def to_json(measurements: DatasetMeasurements) -> str:
    """Serialise the record; byte-identical for identical measurements."""
    payload = {
        "schema": SCHEMA_ID,
        "benchmark": measurements.benchmark,
        "dataset_revision": measurements.dataset_revision,
        "structbench_version": measurements.structbench_version,
        "definition_versions": dict(measurements.definition_versions),
        "cases": [
            {
                "case_id": case.case_id,
                "file_sha256": case.file_sha256,
                "run_traits": sorted(case.run_traits),
                "declared_intent": sorted(case.declared_intent),
                "measurements": [_measurement(m) for m in case.measurements],
            }
            for case in measurements.cases
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _read_measurement(raw: dict[str, Any]) -> Measurement:
    absence = raw.get("absence")
    return Measurement(
        raw["quantity"],
        raw.get("value"),
        raw["unit"],
        frozenset(Location(loc) for loc in raw.get("locations", ())),
        raw.get("n_samples"),
        None
        if absence is None
        else Absence(
            AbsenceReason(absence["reason"]),
            frozenset(EvidenceItem(item) for item in absence["missing"]),
        ),
        raw.get("not_applicable", False),
        raw.get("detail", {}),
    )


def from_json(text: str) -> DatasetMeasurements:
    """Read a record written by :func:`to_json`.

    Raises
    ------
    ValueError
        If the schema id is not this module's.
    """
    raw = json.loads(text)
    if raw.get("schema") != SCHEMA_ID:
        raise ValueError(f"expected schema {SCHEMA_ID!r}, got {raw.get('schema')!r}")
    return DatasetMeasurements(
        raw["benchmark"],
        raw["dataset_revision"],
        raw["structbench_version"],
        raw["definition_versions"],
        tuple(
            CaseMeasurements(
                case["case_id"],
                case["file_sha256"],
                frozenset(case["run_traits"]),
                frozenset(case["declared_intent"]),
                tuple(_read_measurement(m) for m in case["measurements"]),
            )
            for case in raw["cases"]
        ),
    )
