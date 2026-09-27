"""The validation record (``validation-record/1``) and its Markdown rendering.

The JSON is byte-stable: sorted keys, no timestamps, no absolute paths, LF,
UTF-8; regenerating it from the same cases gives the same bytes anywhere. The
Markdown is rendered from the record in a fixed order (variants and setup
rows sorted by key), never edited by hand, and states deviations without
judging them. Every result carries the measured values it was compared with,
so the record stands on its own; rendering refuses a reference set other than
the one the record was built from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from structbench import __version__
from structbench.validation.compare import MEASURES, Comparison
from structbench.validation.reference import REFERENCE_FORMAT, ReferenceSet

RECORD_FORMAT = "validation-record/1"
_MEASURE_TITLES = {
    "length": "Final length L_f / L₀",
    "largest_radius": "Largest radius R_f / R₀",
    "lateral_rms": "Lateral profile W_f (RMS deviation over the fractions)",
}


@dataclass(frozen=True)
class Record:
    format: str
    reference: dict[str, str]
    setup: dict[str, str]
    caveats: tuple[str, ...]
    variants: dict[str, str]
    results: tuple[dict[str, Any], ...]
    summary: dict[str, dict[str, dict[str, float | int]]]
    structbench_version: str


def build_record(
    ref: ReferenceSet,
    comparison: Comparison,
    variants: dict[str, str],
    setup: dict[str, str],
    caveats: tuple[str, ...],
) -> Record:
    results = []
    for r in comparison.results:
        d = asdict(r)
        for key in ("lateral_radii_mm", "measured_lateral_mm"):
            if d[key] is not None:
                d[key] = list(d[key])
        results.append(d)
    return Record(
        format=RECORD_FORMAT,
        reference={
            "name": ref.name,
            "format": REFERENCE_FORMAT,
            "title": ref.title,
            "sha256": ref.sha256,
        },
        setup=dict(setup),
        caveats=tuple(caveats),
        variants=dict(variants),
        results=tuple(results),
        summary=comparison.summary,
        structbench_version=__version__,
    )


def to_json(record: Record) -> bytes:
    text = json.dumps(asdict(record), indent=2, sort_keys=True, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def from_json(data: bytes) -> Record:
    raw = json.loads(data)
    return Record(
        format=str(raw["format"]),
        reference=dict(raw["reference"]),
        setup=dict(raw["setup"]),
        caveats=tuple(raw["caveats"]),
        variants=dict(raw["variants"]),
        results=tuple(dict(r) for r in raw["results"]),
        summary=raw["summary"],
        structbench_version=str(raw["structbench_version"]),
    )


def _pct(x: float) -> str:
    return f"{100 * x:+.1f} %"


def _cell(measure: str, r: dict[str, Any], l0_mm: float, d0_mm: float) -> str:
    if r["status"] != "completed":
        return r["status"]
    if measure == "length":
        cell = f"{r['length_mm'] / l0_mm:.3f} ({_pct(r['dev_length'])})"
        band = r["length_band_mm"] / l0_mm
        return cell + (f" ± {band:.3f}" if band >= 0.0005 else "")
    if measure == "largest_radius":
        return f"{2 * r['largest_radius_mm'] / d0_mm:.3f} ({_pct(r['dev_radius'])})"
    return f"RMS {100 * r['dev_lateral_rms']:.1f} %"


def _measured(measure: str, r: dict[str, Any], l0_mm: float, d0_mm: float) -> str:
    if measure == "length":
        return f"{r['measured_length_mm'] / l0_mm:.3f}"
    if measure == "largest_radius":
        return f"{2 * r['measured_radius_mm'] / d0_mm:.3f}"
    return "—"


def render_markdown(record: Record, ref: ReferenceSet) -> str:
    """The reader's document from the record; ``ref`` gives the tests' conditions."""
    if ref.name != record.reference["name"] or ref.sha256 != record.reference["sha256"]:
        raise ValueError(
            f"the record was built from reference set {record.reference['name']!r} "
            f"(sha256 {record.reference['sha256'][:12]}…), not the one given "
            f"({ref.name!r}, sha256 {ref.sha256[:12]}…)"
        )
    used = [t for t in ref.tests if any(r["test"] == t.id for r in record.results)]
    variants = sorted(record.variants)
    by_key = {(r["variant"], r["test"]): r for r in record.results}
    out: list[str] = []
    out.append(f"# Validation against {ref.title}")
    out.append("")
    out.append(
        f"*Generated from the `{record.format}` record by StructBench "
        f"{record.structbench_version}; reference set `{ref.name}` "
        f"(sha256 {ref.sha256[:12]}…). Deviations are reported, not judged.*"
    )
    out.append("")
    out.append("## Sources")
    out.append("")
    out.append("| Id | Citation | DOI | Consulted directly | Licence |")
    out.append("|---|---|---|---|---|")
    for s in ref.sources:
        doi = s.get("doi")
        link = f"[{doi}](https://doi.org/{doi})" if doi else "—"
        out.append(
            f"| {s['id']} | {s['citation']} | {link} | "
            f"{'yes' if s.get('consulted') else 'no'} | {s.get('licence') or '—'} |"
        )
    out.append("")
    out.append("## Tests used")
    out.append("")
    out.append("| Test | Material | L₀ (mm) | D₀ (mm) | v₀ (m/s) | T₀ (K) | Source |")
    out.append("|---|---|---|---|---|---|---|")
    for t in used:
        out.append(
            f"| {t.id} | {t.material} | {t.L0_mm:g} | {t.D0_mm:g} | {t.v0_ms:g} | "
            f"{t.T0_K:g} | {t.source} |"
        )
    out.append("")
    out.append("## Results")
    out.append("")
    out.append(
        "Measured value first where the measure is a ratio; the simulated value with "
        "its deviation from the measurement in brackets; a status word where the run "
        "did not complete or the case could not be measured."
    )
    for measure, _attr in MEASURES:
        out.append("")
        out.append(f"### {_MEASURE_TITLES[measure]}")
        out.append("")
        labels = " | ".join(record.variants[v] for v in variants)
        out.append(f"| Test | Measured | {labels} |")
        out.append("|---|---|" + "---|" * len(variants))
        for t in used:
            rows = [by_key[(v, t.id)] for v in variants if (v, t.id) in by_key]
            measured = _measured(measure, rows[0], t.L0_mm, t.D0_mm) if rows else "—"
            cells = []
            for v in variants:
                r = by_key.get((v, t.id))
                cells.append(_cell(measure, r, t.L0_mm, t.D0_mm) if r else "—")
            out.append(f"| {t.id} | {measured} | " + " | ".join(cells) + " |")
    gaps = [r for r in record.results if r["status"] != "completed"]
    if gaps:
        out.append("")
        out.append("Gaps:")
        out.append("")
        for r in sorted(gaps, key=lambda r: (r["variant"], r["test"])):
            reason = f": {r['reason']}" if r.get("reason") else ""
            out.append(
                f"- {record.variants[r['variant']]}, test {r['test']}: "
                f"{r['status']}{reason}"
            )
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append("| Variant | Measure | n | min | median | max |")
    out.append("|---|---|---|---|---|---|")
    for v in variants:
        for measure, _attr in MEASURES:
            s = record.summary.get(v, {}).get(measure, {"n": 0})
            if s.get("n", 0):
                spread = [_pct(float(s[k])) for k in ("min", "median", "max")]
                row = " | ".join([str(s["n"]), *spread])
            else:
                row = "0 | — | — | —"
            out.append(f"| {record.variants[v]} | {measure} | {row} |")
    out.append("")
    out.append("## Setup")
    out.append("")
    if record.setup:
        out.append("| Item | Setting |")
        out.append("|---|---|")
        for k in sorted(record.setup):
            out.append(f"| {k} | {record.setup[k]} |")
    else:
        out.append("(none stated)")
    out.append("")
    out.append("## Caveats")
    out.append("")
    for c in (*record.caveats, *ref.caveats):
        out.append(f"- {c}")
    out.append("")
    return "\n".join(out)
