"""The committed record (JSON) and the generated report (Markdown) — ADR-0066 cl. 8.

Measurements are what is committed; verdicts are generated from them and the
criteria, so ``render_markdown(judge(from_json(text)))`` needs no data. Both
outputs are deterministic and carry no time, host, or path.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from ..core import PLATFORM_REASONS, Absence, AbsenceReason, EvidenceItem
from .criteria import CheckResult, DatasetReport
from .results import (
    CaseMeasurements,
    DatasetMeasurements,
    Location,
    Measurement,
    Verdict,
)

__all__ = ["SCHEMA_ID", "from_json", "render_markdown", "to_json"]

SCHEMA_ID = "structbench.verification/1"

_PLATFORM = {str(r) for r in PLATFORM_REASONS}
_JUDGED = (Verdict.PASS, Verdict.FAIL, Verdict.REVIEW)


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


def _cases(ids: list[str], limit: int = 6) -> str:
    shown = ", ".join(f"`{i}`" for i in ids[:limit])
    return shown if len(ids) <= limit else f"{shown}, … ({len(ids)} cases)"


def _span(values: list[float]) -> str:
    if not values:
        return "—"
    lo, hi = min(values), max(values)
    return f"{lo:.6g}" if lo == hi else f"{lo:.6g} … {hi:.6g}"


def render_markdown(report: DatasetReport) -> str:
    """The human-readable report: what was judged, what was not, and why."""
    data = report.measurements
    rows: dict[str, list[tuple[str, CheckResult]]] = defaultdict(list)
    for case in report.cases:
        for result in case.results:
            rows[result.quantity].append((case.case_id, result))

    def is_platform_gap(results: list[tuple[str, CheckResult]]) -> bool:
        return all(
            r.verdict is Verdict.NOT_ASSESSABLE
            and r.reason in _PLATFORM
            and r.value is None
            for _, r in results
        )

    checked = {q: rs for q, rs in rows.items() if not is_platform_gap(rs)}
    out = [
        f"# Reference-data verification: {data.benchmark or 'unregistered runs'}",
        "",
        f"- Dataset revision: {data.dataset_revision or 'not recorded'}",
        f"- Cases: {len(report.cases)}",
        f"- Instrument: structbench {data.structbench_version}, `{SCHEMA_ID}`",
        "",
        "Generated from the committed measurements and the platform criteria; no"
        " data was read to produce it. A `pass` on numerical health or"
        " conservation is a necessary solution-verification indicator, not"
        " evidence of accuracy. A `review` is an indicator above its reference"
        " level: the call is a person's, not this instrument's.",
        "",
        "## Verdicts",
        "",
        "Quantities measured on at least one case.",
        "",
        "| Quantity | Unit | Measured | pass | fail | review | n/a | not assessable"
        " | Criterion |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for quantity in sorted(checked):
        results = [r for _, r in checked[quantity]]
        count = Counter(r.verdict for r in results)
        values = [r.value for r in results if r.value is not None]
        if not values:
            continue  # listed below: not applicable, or evidence not supplied
        labels = sorted({r.criterion for r in results if r.criterion})
        if any(r.provisional for r in results):
            labels = [f"{label} (provisional)" for label in labels]
        levels = sorted({lv for r in results for lv in r.out_of_scope_levels})
        criterion = "; ".join(labels) or "none"
        if levels:
            criterion += " — out of scope: " + "; ".join(levels)
        out.append(
            f"| `{quantity}` | {results[0].unit} | {_span(values)} | "
            + " | ".join(str(count[v] or "") for v in _JUDGED)
            + f" | {count[Verdict.NOT_APPLICABLE] or ''}"
            + f" | {count[Verdict.NOT_ASSESSABLE] or ''} | {criterion} |"
        )

    out += ["", "## Findings", ""]
    findings = [
        (quantity, verdict, [cid for cid, r in rs if r.verdict is verdict])
        for quantity, rs in sorted(checked.items())
        for verdict in (Verdict.FAIL, Verdict.REVIEW)
    ]
    findings = [f for f in findings if f[2]]
    for quantity, verdict, ids in findings:
        values = [
            r.value
            for cid, r in checked[quantity]
            if cid in ids and r.value is not None
        ]
        out.append(f"- **{verdict}** `{quantity}` = {_span(values)} — {_cases(ids)}")
    if not findings:
        out.append("None.")

    out += ["", "## Not applicable to these runs", ""]
    skipped = [
        f"`{q}`"
        for q, rs in sorted(checked.items())
        if all(r.verdict is Verdict.NOT_APPLICABLE for _, r in rs)
    ]
    out.append(", ".join(skipped) if skipped else "None.")

    out += ["", "## Evidence the runs did not supply", ""]
    gaps: dict[tuple[str, tuple[str, ...]], dict[str, list[str]]] = defaultdict(dict)
    for quantity, rs in sorted(checked.items()):
        for cid, r in rs:
            if r.verdict is Verdict.NOT_ASSESSABLE and r.reason not in _PLATFORM:
                gaps[(r.reason, r.missing)].setdefault(quantity, []).append(cid)
    for (reason, missing), quantities in sorted(gaps.items()):
        items = f" ({', '.join(missing)})" if missing else ""
        names = ", ".join(f"`{q}`" for q in quantities)
        n_cases = len({cid for ids in quantities.values() for cid in ids})
        out.append(f"- `{reason}`{items}, {n_cases} cases: {names}")
    if not gaps:
        out.append("None.")

    out += [
        "",
        "## Not yet checked by this instrument",
        "",
        "Gaps in the platform, not in the data: a quantity the instrument cannot"
        " measure yet, or a declaration it has no home for.",
        "",
    ]
    for quantity in sorted(set(rows) - set(checked)):
        reasons = sorted({r.reason for _, r in rows[quantity]})
        out.append(f"- `{quantity}` — {', '.join(reasons)}")

    out += ["", "## Criteria", ""]
    used = {q for q, rs in checked.items() if any(r.criterion for _, r in rs)}
    for c in sorted(report.criteria, key=lambda c: (c.quantity, c.label())):
        if c.quantity in used:
            tags = [str(c.kind)] + (["provisional"] if c.provisional else [])
            source = f" Source: {c.source}." if c.source else ""
            head = f"- `{c.quantity}` {c.label()} ({', '.join(tags)})."
            out.append(f"{head} {c.rationale}{source}")
    return "\n".join(out) + "\n"
