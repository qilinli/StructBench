"""Convergence of a sweep across mesh levels: pairing, QoIs, fields, the record.

    structbench-datagen converge --dataset <dir> --sweep <runs>/<name>
        [--root <other run root> ...] [--out DIR]

Runs are paired across the run roots by their parameters (without the refine
key and the ``[limits]`` values) and their variant; a production case is a run
of a split that is not a probe, and the probe runs at other levels are its
levels. At each level the first run with a canonical file counts — the
production run first, then root order, then case id — and duplicates are
named. Each quantity of interest (the dataset's ``qoi()``) is extrapolated
with the engine of ``structbench.verification.convergence`` when three levels
stand in a constant ratio, and every stored field of each coarser level is
measured against the finest. The record (``convergence-record/1``) is
byte-stable and names cases, never paths; its Markdown is rendered from it.
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


def _level_key(label: str) -> tuple[int, float, str]:
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, 0.0, label)


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
        for prov_path in sorted(root.glob("*/provenance.json")):
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
        for level in sorted({r.level for r in runs}, key=_level_key):
            same = sorted(
                (r for r in runs if r.level == level and r.canonical is not None),
                key=lambda r: (not r.production, r.root_index, r.case_id),
            )
            if not same:
                continue
            if len(same) > 1:
                others = ", ".join(r.case_id for r in same[1:])
                notes.append(
                    f"{case_id}: level {level} has {len(same)} runs with a canonical "
                    f"file; {same[0].case_id} stands for it, not {others}"
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
    entry: dict[str, Any] = {
        "case_id": case_id,
        "levels": {label: run.case_id for label, run in by_level.items()},
        "production_level": next(
            (label for label, run in by_level.items() if run.production), None
        ),
        "qoi": {},
        "extrapolation": {},
        "fields": {},
        "notes": [],
    }
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
    labels = sorted(loaded, key=_level_key)
    for label in labels:
        try:
            values = problem.qoi(loaded[label])
        except Exception as exc:
            entry["notes"].append(
                f"level {label}: qoi() raised {type(exc).__name__}: {exc}"
            )
            continue
        for name in defn.qoi.names:
            if name in values:
                entry["qoi"].setdefault(name, {})[label] = float(values[name])
    ratio = None
    numeric = all(_level_key(label)[0] == 0 for label in labels)
    if numeric and len(labels) >= 3:
        h = [float(label) for label in labels]
        ratios = [h[i + 1] / h[i] for i in range(len(h) - 1)]
        if all(abs(r - ratios[0]) <= 1e-9 * ratios[0] for r in ratios):
            ratio = ratios[0]
        else:
            entry["notes"].append(
                f"levels {labels} are not in a constant ratio: no extrapolation"
            )
    elif len(labels) >= 3:
        entry["notes"].append(
            f"levels {labels} are not numeric labels: no extrapolation"
        )
    finest_three = labels[-3:]
    for name in defn.qoi.names:
        per_level = entry["qoi"].get(name, {})
        if ratio is None or any(label not in per_level for label in finest_three):
            entry["extrapolation"][name] = None
            continue
        x = engine.richardson(
            {label: per_level[label] for label in finest_three}, ratio=ratio
        )
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
    summary: dict[str, Any] = {"qoi": {}, "fields": {}}
    for name in defn.qoi.names:
        statuses = Counter(
            c["extrapolation"][name]["status"]
            for c in cases
            if c["extrapolation"].get(name)
        )
        at_production = [
            c["extrapolation"][name]["error_vs_extrapolated"][c["production_level"]]
            for c in cases
            if c["extrapolation"].get(name)
            and c["extrapolation"][name]["error_vs_extrapolated"]
            and c["production_level"]
            in c["extrapolation"][name]["error_vs_extrapolated"]
        ]
        summary["qoi"][name] = {
            "statuses": dict(sorted(statuses.items())),
            "error_at_production_level": _spread(at_production),
        }
    fields = sorted({f for c in cases for f in c["fields"]})
    for field in fields:
        summary["fields"][field] = {}
        for label in sorted(
            {lv for c in cases for lv in c["fields"].get(field, {})}, key=_level_key
        ):
            summary["fields"][field][label] = _spread(
                [
                    c["fields"][field][label]
                    for c in cases
                    if label in c["fields"].get(field, {})
                ]
            )
    return {
        "format": RECORD_FORMAT,
        "dataset": defn.name,
        "refine_key": defn.levels.refine_key,
        "symmetry": defn.levels.symmetry,
        "levels": {
            label: level_counts[label] for label in sorted(level_counts, key=_level_key)
        },
        "cases": cases,
        "summary": summary,
        "notes": notes,
        "structbench_version": __version__,
    }


def to_json(record: dict[str, Any]) -> bytes:
    return (
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.2f} %"


def _spread_row(s: dict[str, Any]) -> str:
    if not s.get("n"):
        return "0 | — | — | —"
    return f"{s['n']} | {_pct(s['min'])} | {_pct(s['median'])} | {_pct(s['max'])}"


def render_markdown(record: dict[str, Any]) -> str:
    """The reader's document, from the record alone."""
    levels = list(record["levels"])
    out = [f"# Convergence of {record['dataset']} across mesh levels", ""]
    out.append(
        f"*From the `{record['format']}` record (StructBench "
        f"{record['structbench_version']}); refine key `{record['refine_key']}`, "
        f"{record['symmetry']} volume weights. Levels: "
        + ", ".join(f"{lv} ({n} cases)" for lv, n in record["levels"].items())
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
            + " | GCI (finest) |"
        )
        out.append("|---|" + "---|" * (len(levels) + 5))
        for c in cases:
            values = " | ".join(
                f"{c['qoi'].get(name, {}).get(lv):.6g}"
                if lv in c["qoi"].get(name, {})
                else "—"
                for lv in levels
            )
            x = c["extrapolation"].get(name)
            if x:
                prod = (x["error_vs_extrapolated"] or {}).get(c["production_level"])
                tail = (
                    f"{x['status']} | {x['order']:.3g} | {x['extrapolated']:.6g} | "
                    f"{_pct(prod)} | {_pct(x['gci_fine'])}"
                    if x["status"] == "monotone"
                    else f"{x['status']} | — | — | — | —"
                )
            else:
                tail = "— | — | — | — | —"
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
        out.append(
            "| Case | Field | " + " | ".join(f"level {lv}" for lv in coarser) + " |"
        )
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
            for lv, s in record["summary"]["fields"].get(field, {}).items():
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
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        defn = load_definition(args.dataset)
        problem = load_problem(args.dataset)
    except DefinitionError as exc:
        print(exc, file=sys.stderr)
        return 2
    record = converge(defn, problem, [args.sweep, *args.root])
    out = args.out or args.sweep / "converge"
    out.mkdir(parents=True, exist_ok=True)
    (out / "convergence.json").write_bytes(to_json(record))
    (out / "convergence.md").write_bytes(render_markdown(record).encode("utf-8"))
    n = len(record["cases"])
    print(f"{n} case{'s' if n != 1 else ''} across levels -> {out}")
    for note in record["notes"]:
        print(f"  {note}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
