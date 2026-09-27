"""Convergence of a sweep across mesh levels: pairing, QoIs, fields, the record.

    structbench-datagen converge --dataset <dir> --sweep <runs>/<name>
        [--root <other run root> ...] [--out DIR]

Runs are paired across the run roots by their parameters (without the refine
key and the ``[limits]`` values) and their variant; a production case is a run
of a split that is not a probe, and the probe runs at other levels are its
levels. At each level the first run with a canonical file counts — the
production run first, then root order, then case id — and duplicates are
named. Levels are ordered as ``[levels].pilot`` lists them (coarse to fine),
then numerically; a label is a refinement factor, larger meaning finer. Each
quantity of interest (the dataset's ``qoi()``) is extrapolated with the engine
of ``structbench.verification.convergence`` when the three finest levels have
positive numeric labels in a constant ratio, and every stored field of each
coarser level is measured against the finest. The record
(``convergence-record/1``) is byte-stable and names cases, never paths; its
Markdown is rendered from it. Exit 0 clean, 1 when a case carries a finding
(a refusal, a gap, no pairs at all), 2 when refused (definition, paths).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from structbench import __version__
from structbench.core import Case
from structbench.core.io import read_case
from structbench.datagen.definition import (
    Definition,
    DefinitionError,
    load_definition,
    load_problem,
)
from structbench.verification import convergence as engine

RECORD_FORMAT = "convergence-record/1"


@dataclass(frozen=True)
class LevelRun:
    case_id: str
    split: str
    level: str
    root_index: int
    production: bool
    canonical: Path | None


def _numeric(label: str) -> float | None:
    """The label as a positive finite number, or None."""
    try:
        value = float(label)
    except ValueError:
        return None
    return value if 0.0 < value < float("inf") else None


def _level_key(label: str) -> tuple[int, float, str]:
    value = _numeric(label)
    return (0, value, label) if value is not None else (1, 0.0, label)


def level_order(defn: Definition, labels: set[str]) -> list[str]:
    """Labels as ``[levels].pilot`` lists them, then the rest by number, then name."""
    listed = list(defn.levels.pilot)

    def key(label: str) -> tuple[int, int, float, str]:
        if label in listed:
            return (0, listed.index(label), 0.0, label)
        kind, value, name = _level_key(label)
        return (1, kind, value, name)

    return sorted(labels, key=key)


def pair_levels(
    defn: Definition, roots: Sequence[Path]
) -> tuple[dict[str, dict[str, LevelRun]], list[str]]:
    """Production case id -> level label -> the run that stands for it, plus notes."""
    probe = {s.name: s.probe for s in defn.splits}
    key_of = defn.levels.refine_key
    candidates: dict[str, list[LevelRun]] = {}
    notes: list[str] = []
    unknown_splits: set[str] = set()
    unlevelled: dict[str, list[str]] = {}
    for index, root in enumerate(roots):
        found = sorted(root.glob("*/provenance.json"), key=lambda p: p.parent.name)
        for prov_path in found:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
            case_id = str(prov.get("case_id", prov_path.parent.name))
            split = str(prov.get("split", ""))
            if split not in probe:
                unknown_splits.add(split)
                continue
            params = dict(prov.get("params", {}))
            if key_of not in params:
                unlevelled.setdefault(split, []).append(case_id)
                continue
            level = str(params.pop(key_of))
            for limit in defn.limits:
                params.pop(limit, None)
            key = json.dumps([params, prov.get("variant")], sort_keys=True)
            canonical = root / "canonical" / f"{case_id}.h5"
            candidates.setdefault(key, []).append(
                LevelRun(
                    case_id,
                    split,
                    level,
                    index,
                    not probe[split],
                    canonical if canonical.is_file() else None,
                )
            )
    for split, ids in sorted(unlevelled.items()):
        notes.append(
            f"split {split!r}: {len(ids)} runs without {key_of!r} in their parameters, "
            f"skipped: {', '.join(ids)}"
        )
    for split in sorted(unknown_splits):
        notes.append(f"split {split!r} is not in the definition; its runs were skipped")
    paired: dict[str, dict[str, LevelRun]] = {}
    for runs in candidates.values():
        production = sorted(
            (r for r in runs if r.production), key=lambda r: (r.root_index, r.case_id)
        )
        if not production:
            continue
        case_id = production[0].case_id
        by_level: dict[str, LevelRun] = {}
        for level in level_order(defn, {r.level for r in runs}):
            same = sorted(
                (r for r in runs if r.level == level and r.canonical is not None),
                key=lambda r: (not r.production, r.root_index, r.case_id),
            )
            if not same:
                continue
            if len(same) > 1:
                others = ", ".join(
                    f"{r.case_id} (root {r.root_index})" for r in same[1:]
                )
                notes.append(
                    f"{case_id}: level {level} has {len(same)} runs with a canonical "
                    f"file; {same[0].case_id} (root {same[0].root_index}) stands for "
                    f"it, not {others}"
                )
            by_level[level] = same[0]
        if len(by_level) >= 2:
            paired[case_id] = by_level
    return dict(sorted(paired.items())), notes


def _spread(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "n": len(values),
    }


def _stored_fields(case: Case) -> set[str]:
    assert case.response is not None
    return set(case.response.node) | set(case.response.element.get("solid", {}))


def _one_case(
    defn: Definition,
    problem: ModuleType,
    case_id: str,
    by_level: dict[str, LevelRun],
) -> dict[str, Any]:
    production_level = defn.levels.production
    entry: dict[str, Any] = {
        "case_id": case_id,
        "levels": {label: run.case_id for label, run in by_level.items()},
        "production_level": production_level,
        "qoi": {},
        "extrapolation": {},
        "fields": {},
        "notes": [],
    }
    standing = by_level.get(production_level)
    if standing is not None and not standing.production:
        entry["notes"].append(
            f"level {production_level} (the production level) is stood for by a probe "
            f"run, {standing.case_id}"
        )
    elsewhere = [
        lv for lv, run in by_level.items() if run.production and lv != production_level
    ]
    if elsewhere:
        entry["notes"].append(
            "production runs also stand at levels "
            + ", ".join(elsewhere)
            + f" (older roots); the errors below are read at level {production_level}"
        )
    loaded: dict[str, Case] = {}
    for label, run in by_level.items():
        assert run.canonical is not None
        try:
            loaded[label] = read_case(run.canonical)
        except Exception as exc:  # a broken file is a note, not an abort
            entry["notes"].append(
                f"level {label} ({run.case_id}) could not be read: "
                f"{type(exc).__name__}: {exc}"
            )
    labels = level_order(defn, set(loaded))
    for label in labels:
        try:
            values = problem.qoi(loaded[label])
            found = {
                name: float(values[name]) for name in defn.qoi.names if name in values
            }
        except Exception as exc:
            entry["notes"].append(
                f"level {label}: qoi() raised {type(exc).__name__}: {exc}"
            )
            continue
        for name, value in found.items():
            entry["qoi"].setdefault(name, {})[label] = value
    finest_three = labels[-3:]
    ratio = None
    if len(finest_three) == 3:
        h = [_numeric(label) for label in finest_three]
        if h[0] is None or h[1] is None or h[2] is None:
            entry["notes"].append(
                f"levels {finest_three} are not positive numeric labels: "
                "no extrapolation"
            )
        else:
            ratios = [h[1] / h[0], h[2] / h[1]]
            if abs(ratios[1] - ratios[0]) <= 1e-9 * ratios[0]:
                ratio = ratios[0]
            else:
                entry["notes"].append(
                    f"levels {finest_three} are not in a constant ratio: "
                    "no extrapolation"
                )
    for name in defn.qoi.names:
        per_level = entry["qoi"].get(name, {})
        if ratio is None or any(label not in per_level for label in finest_three):
            entry["extrapolation"][name] = None
            continue
        try:
            x = engine.richardson(
                {label: per_level[label] for label in finest_three}, ratio=ratio
            )
        except (ValueError, ZeroDivisionError) as exc:
            entry["extrapolation"][name] = None
            entry["notes"].append(f"{name}: no extrapolation: {exc}")
            continue
        entry["extrapolation"][name] = {
            **asdict(x),
            "ratio": ratio,
            "levels": finest_three,
        }
    if len(labels) >= 2:
        finest = labels[-1]
        fine = loaded[finest]
        fine_fields = _stored_fields(fine)
        for label in labels[:-1]:
            coarse = loaded[label]
            missing = sorted(_stored_fields(coarse) ^ fine_fields)
            if missing:
                entry["notes"].append(
                    f"fields stored at level {label} or {finest} but not both, "
                    f"left out: {', '.join(missing)}"
                )
            try:
                errors = engine.field_errors(
                    coarse,
                    fine,
                    axisymmetric=defn.levels.symmetry == "axisymmetric",
                )
            except (engine.NestingError, ValueError) as exc:
                entry["notes"].append(f"level {label} against {finest}: {exc}")
                continue
            for field, value in errors.items():
                entry["fields"].setdefault(field, {})[label] = value
    return entry


def converge(
    defn: Definition, problem: ModuleType, roots: Sequence[Path]
) -> dict[str, Any]:
    """The record of every production case that has runs at two or more levels."""
    paired, notes = pair_levels(defn, roots)
    cases = [
        _one_case(defn, problem, case_id, by_level)
        for case_id, by_level in paired.items()
    ]
    level_counts = Counter(label for c in cases for label in c["levels"])
    order = level_order(defn, set(level_counts))
    summary: dict[str, Any] = {"qoi": {}, "fields": {}}
    for name in defn.qoi.names:
        statuses = Counter(
            c["extrapolation"][name]["status"]
            for c in cases
            if c["extrapolation"].get(name)
        )
        at_production = []
        for c in cases:
            x = c["extrapolation"].get(name)
            errors = (x or {}).get("error_vs_extrapolated") or {}
            if c["production_level"] in errors:
                at_production.append(errors[c["production_level"]])
        summary["qoi"][name] = {
            "statuses": dict(sorted(statuses.items())),
            "error_at_production_level": _spread(at_production),
        }
    fields = sorted({f for c in cases for f in c["fields"]})
    for field in fields:
        summary["fields"][field] = {}
        for label in order:
            values = [
                c["fields"][field][label]
                for c in cases
                if label in c["fields"].get(field, {})
            ]
            if values:
                summary["fields"][field][label] = _spread(values)
    return {
        "format": RECORD_FORMAT,
        "dataset": defn.name,
        "refine_key": defn.levels.refine_key,
        "symmetry": defn.levels.symmetry,
        "production_level": defn.levels.production,
        "level_order": order,
        "levels": {label: level_counts[label] for label in order},
        "cases": cases,
        "summary": summary,
        "notes": notes,
        "structbench_version": __version__,
    }


def to_json(record: dict[str, Any]) -> bytes:
    text = json.dumps(
        record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    )
    return (text + "\n").encode("utf-8")


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.2f} %"


def _spread_row(s: dict[str, Any]) -> str:
    if not s.get("n"):
        return "0 | — | — | —"
    return f"{s['n']} | {_pct(s['min'])} | {_pct(s['median'])} | {_pct(s['max'])}"


def render_markdown(record: dict[str, Any]) -> str:
    """The reader's document, from the record alone."""
    levels = list(record.get("level_order") or sorted(record["levels"], key=_level_key))
    out = [f"# Convergence of {record['dataset']} across mesh levels", ""]
    out.append(
        f"*From the `{record['format']}` record (StructBench "
        f"{record['structbench_version']}); refine key `{record['refine_key']}`, "
        f"{record['symmetry']} volume weights, production level "
        f"{record.get('production_level', '?')}. Levels: "
        + ", ".join(f"{lv} ({record['levels'][lv]} cases)" for lv in levels)
        + ".*"
    )
    cases = record["cases"]
    qoi_names = sorted({name for c in cases for name in c["qoi"]})
    for name in qoi_names:
        out += ["", f"## {name}", ""]
        header = "| Case | " + " | ".join(f"level {lv}" for lv in levels)
        out.append(
            header
            + " | Status | Order | Extrapolated | Error at production level"
            + " | Coarsest vs finest | GCI (finest) |"
        )
        out.append("|---|" + "---|" * (len(levels) + 6))
        for c in cases:
            per = c["qoi"].get(name, {})
            values = " | ".join(f"{per[lv]:.6g}" if lv in per else "—" for lv in levels)
            x = c["extrapolation"].get(name)
            if x and x["status"] == "monotone":
                prod = (x["error_vs_extrapolated"] or {}).get(c["production_level"])
                order = f"{x['order']:.3g}" if x["order"] is not None else "—"
                tail = (
                    f"{x['status']} | {order} | {x['extrapolated']:.6g} | {_pct(prod)}"
                    f" | {_pct(x['coarsest_vs_finest'])} | {_pct(x['gci_fine'])}"
                )
            elif x:
                tail = (
                    f"{x['status']} | — | — | — | {_pct(x['coarsest_vs_finest'])} | —"
                )
            else:
                tail = "— | — | — | — | — | —"
            out.append(f"| {c['case_id']} | {values} | {tail} |")
        s = record["summary"]["qoi"].get(name, {})
        out += [
            "",
            "Statuses: "
            + (
                ", ".join(f"{k} {v}" for k, v in s.get("statuses", {}).items())
                or "none"
            )
            + ". Error at the production level (n | min | median | max): "
            + _spread_row(s.get("error_at_production_level", {"n": 0}))
            + ".",
        ]
    fields = sorted({f for c in cases for f in c["fields"]})
    if fields:
        out += ["", "## Field errors against the finest level (pooled relative L2)", ""]
        coarser = levels[:-1]
        columns = " | ".join(f"level {lv}" for lv in coarser)
        out.append(f"| Case | Field | {columns} |")
        out.append("|---|---|" + "---|" * len(coarser))
        for c in cases:
            for field in fields:
                if field not in c["fields"]:
                    continue
                cells = " | ".join(_pct(c["fields"][field].get(lv)) for lv in coarser)
                out.append(f"| {c['case_id']} | {field} | {cells} |")
        out += [
            "",
            "| Field | Level | n | min | median | max |",
            "|---|---|---|---|---|---|",
        ]
        for field in fields:
            for lv in levels:
                s = record["summary"]["fields"].get(field, {}).get(lv)
                if s:
                    out.append(f"| {field} | {lv} | {_spread_row(s)} |")
    notes = list(record["notes"]) + [
        f"{c['case_id']}: {n}" for c in cases for n in c["notes"]
    ]
    if notes:
        out += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="structbench-datagen converge", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, help="default: <sweep>/converge")
    args = parser.parse_args(argv)
    try:
        defn = load_definition(args.dataset)
        problem = load_problem(args.dataset)
    except DefinitionError as exc:
        print(exc, file=sys.stderr)
        return 2
    roots = [args.sweep, *args.root]
    for root in roots:
        if not root.is_dir():
            print(f"{root} is not a directory", file=sys.stderr)
            return 2
    record = converge(defn, problem, roots)
    out = args.out or args.sweep / "converge"
    out.mkdir(parents=True, exist_ok=True)
    (out / "convergence.json").write_bytes(to_json(record))
    (out / "convergence.md").write_bytes(render_markdown(record).encode("utf-8"))
    n = len(record["cases"])
    findings = sum(len(c["notes"]) for c in record["cases"])
    print(
        f"{n} case{'s' if n != 1 else ''} across levels, "
        f"{findings} finding{'s' if findings != 1 else ''} -> {out}"
    )
    for note in record["notes"]:
        print(f"  {note}")
    return 0 if n and not findings else 1


if __name__ == "__main__":
    sys.exit(main())
