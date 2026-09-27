"""Generate a sweep's Abaqus case folders from a dataset's dataset.toml and problem.py.

    structbench-datagen generate --dataset <dir> --work-root <root>
        [--split NAME ...] [--force] [--dry-run] [--no-preflight]

Each case gets ``<work-root>/<name>/<case_id>/<case_id>.inp`` and a
``provenance.json``; the sweep gets a ``manifest.csv`` listing every case. An
existing identical deck is skipped. A different one is a conflict unless
``--force``, and a deck whose case already ran (``run.json``) is never replaced
(ADR-0069).

A split that is not a probe is generated only against a passing preflight
stamp (``<work-root>/<name>/preflight/stamp.json``) for the current
``dataset.toml`` and ``problem.py``; ``--no-preflight`` generates anyway and
records the omission in every such case's provenance (ADR-0071).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np
import scipy

from structbench import __version__
from structbench.core.io import unit_factors
from structbench.datagen import sampling
from structbench.datagen.definition import (
    Definition,
    file_sha256,
    load_definition,
    load_problem,
    problem_sha256,
)

#: The checkout this module runs from, when it is one (src/structbench/datagen
#: -> repo root); an installed package has no commit to record, only a version.
_REPO = Path(__file__).resolve().parents[3]
#: Version 3 (plan 2b) adds ``preflight`` (the stamp a guarded case was
#: generated against, ``{"skipped": true}``, or null for a probe split) and
#: ``probe`` (a preflight case's role); readers accept version 2 as well.
PROVENANCE_FORMAT = "abaqus-provenance/3"
#: The preflight's stamp, relative to the sweep directory.
STAMP_FILE = "preflight/stamp.json"
#: A conservative Abaqus job-name rule; the conformance run confirms or relaxes it.
_CASE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,37}$")

Outcome = Literal["written", "unchanged", "conflict", "locked"]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    split: str
    index: int
    variant: str | None
    seed: int | None
    params: dict[str, float | str]
    #: A preflight case's role: {"role": level|increment|frame|conformance,
    #: "level": ..., "factor": ...}; None for an ordinary case.
    probe: dict[str, Any] | None = None


def case_id_for(
    defn: Definition, split: str, index: int, variant: str | None, suffix: str = ""
) -> str:
    """``<prefix>-<split>-<index>[-<variant>]<suffix>``, checked as a job name."""
    case_id = f"{defn.case_prefix}-{split}-{index:04d}"
    case_id += f"-{variant}" if variant else ""
    case_id += suffix
    if not _CASE_ID.fullmatch(case_id):
        raise ValueError(f"case id {case_id!r} is not a safe Abaqus job name")
    return case_id


def plan_cases(
    defn: Definition, feasible: Callable[[dict[str, Any]], bool] | None = None
) -> list[CaseSpec]:
    """Every case of every split, in file then Sobol order.

    ``feasible`` is the problem's optional hook, its declared feasibility or
    severity limit; it sees the fixed values and the ``[limits]`` too, and it
    filters sampled points only (explicit points are deliberate). A split that
    is not a probe runs at ``[levels].production``: its cases carry the refine
    key at that level, and a split naming another level is refused.
    """
    unit_factors(defn.units)
    seeds = Counter(s.seed for s in defn.splits if s.seed is not None)
    shared = sorted(seed for seed, count in seeds.items() if count > 1)
    if shared:
        # Same seed, same engine: two splits would draw the same points.
        raise ValueError(f"sampled splits share seed {shared[0]}; give each its own")
    check = None
    if feasible is not None:
        model_feasible = feasible

        def check(params: dict[str, float | str]) -> bool:
            return bool(model_feasible({**defn.fixed, **defn.limits, **params}))

    specs = []
    for split in defn.splits:
        sampled = set(defn.variables) | set(split.extra) | set(split.categorical)
        clash = set(defn.fixed) & sampled
        if clash:
            raise ValueError(f"{sorted(clash)} are both fixed and sampled")
        for point in sampling.sample_split(defn.variables, defn.regions, split, check):
            for variant in split.variants or (None,):
                case_id = case_id_for(defn, split.name, point.index, variant)
                params = {**defn.fixed, **point.params}
                if not split.probe:
                    key, production = defn.levels.refine_key, defn.levels.production
                    named = params.get(key)
                    if named is not None and str(named) != production:
                        raise ValueError(
                            f"splits.{split.name}: {key}={named!r}, but a split that "
                            f"is not a probe runs at levels.production={production!r}"
                        )
                    params[key] = production
                specs.append(
                    CaseSpec(
                        case_id, split.name, point.index, variant, split.seed, params
                    )
                )
    return specs


def git_state(path: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"{path.name} is not a git repository with a commit"
        ) from exc


def package_state() -> dict[str, Any]:
    """StructBench's identity for the provenance: the version, and the commit
    when the package runs from a checkout (an installed copy records none)."""
    state: dict[str, Any] = {"version": __version__, "commit": None, "dirty": None}
    if (_REPO / ".git").exists():
        state.update(git_state(_REPO))
    return state


def write_case(
    case_dir: Path, deck_text: str, provenance: dict[str, Any], *, force: bool
) -> Outcome:
    inp = case_dir / f"{case_dir.name}.inp"
    data = deck_text.encode("utf-8")
    if inp.exists():
        if inp.read_bytes() == data:
            return "unchanged"
        if (case_dir / "run.json").exists():
            return "locked"
        if not force:
            return "conflict"
    case_dir.mkdir(parents=True, exist_ok=True)
    inp.write_bytes(data)
    (case_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return "written"


def read_stamp(sweep_dir: Path) -> dict[str, Any] | None:
    """The sweep's preflight stamp, or None when there is none to read."""
    path = sweep_dir / STAMP_FILE
    if not path.is_file():
        return None
    try:
        stamp = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return stamp if isinstance(stamp, dict) else None


def stamp_refusal(
    stamp: dict[str, Any] | None, definition_sha: str, problem_sha: str
) -> str | None:
    """Why the stamp does not open the gate for this definition, or None."""
    if stamp is None:
        return "no preflight stamp"
    if not stamp.get("passed"):
        return "the preflight stamp did not pass"
    if stamp.get("definition_sha256") != definition_sha:
        return "the preflight stamp is for another dataset.toml"
    if stamp.get("problem_sha256") != problem_sha:
        return "the preflight stamp is for another problem.py"
    return None


def materialise(
    sweep_dir: Path,
    specs: Sequence[CaseSpec],
    *,
    defn: Definition,
    problem: ModuleType,
    dataset_dir: Path,
    force: bool = False,
    preflight: dict[str, Any] | None = None,
    decks: Mapping[str, str] | None = None,
) -> tuple[Counter[str], list[str]]:
    """Write each spec's deck and provenance into ``sweep_dir``.

    Returns the outcome counts and the problems (``"<id>: conflict|locked"``).
    ``decks`` hands in deck text by case id (the preflight's transformed
    conformance deck); every other deck is ``problem.input_deck``'s.
    ``preflight`` is recorded on the cases of splits that are not probes.
    """
    sweep_dir.mkdir(parents=True, exist_ok=True)
    probe_of = {s.name: s.probe for s in defn.splits}
    dataset_repo = git_state(dataset_dir)
    definition_sha, problem_sha = defn.sha256(), problem_sha256(dataset_dir)
    repo = package_state()
    counts: Counter[str] = Counter()
    problems: list[str] = []
    for spec in specs:
        if decks is not None and spec.case_id in decks:
            text = decks[spec.case_id]
        else:
            text = problem.input_deck(dict(spec.params), spec.variant)
        provenance = {
            "format": PROVENANCE_FORMAT,
            "case_id": spec.case_id,
            "dataset": defn.name,
            "units": defn.units,
            "split": spec.split,
            "index": spec.index,
            "variant": spec.variant,
            "seed": spec.seed,
            "params": spec.params,
            "probe": spec.probe,
            "preflight": None if probe_of.get(spec.split, True) else preflight,
            "definition_sha256": definition_sha,
            "problem_sha256": problem_sha,
            "inp_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "repository": repo,
            "dataset_repository": dataset_repo,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        outcome = write_case(sweep_dir / spec.case_id, text, provenance, force=force)
        counts[outcome] += 1
        if outcome in ("conflict", "locked"):
            problems.append(f"{spec.case_id}: {outcome}")
    return counts, problems


def write_manifest(sweep_dir: Path, specs: Sequence[CaseSpec]) -> None:
    names = sorted({k for s in specs for k in s.params})
    with (sweep_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case_id", "split", "index", "variant", *names])
        for s in specs:
            row = [s.params.get(k, "") for k in names]
            writer.writerow([s.case_id, s.split, s.index, s.variant or "", *row])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="structbench-datagen generate", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--split", action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="generate splits that are not probes without a passing preflight "
        "stamp; the omission is recorded in every such case's provenance",
    )
    args = parser.parse_args(argv)

    dataset_dir = args.dataset.resolve()
    try:
        defn = load_definition(dataset_dir)
        problem = load_problem(dataset_dir)
        specs = plan_cases(defn, getattr(problem, "feasible", None))
    except (ValueError, KeyError) as exc:  # DefinitionError is a ValueError
        print(
            f"{dataset_dir.name}: {exc.args[0] if exc.args else exc}", file=sys.stderr
        )
        return 2
    selected = [s for s in specs if not args.split or s.split in args.split]
    if args.dry_run:
        for split, count in Counter(s.split for s in selected).items():
            print(f"{split}: {count} cases")
        return 0

    try:
        dataset_repo = git_state(dataset_dir)
    except RuntimeError as exc:
        print(
            f"{dataset_dir.name}: {exc}; commit the definition first", file=sys.stderr
        )
        return 2
    sweep_dir = args.work_root / defn.name
    probe_of = {s.name: s.probe for s in defn.splits}
    guarded = sorted({s.split for s in selected if not probe_of[s.split]})
    preflight: dict[str, Any] | None = None
    if guarded:
        if args.no_preflight:
            preflight = {"skipped": True}
        else:
            stamp = read_stamp(sweep_dir)
            why = stamp_refusal(stamp, defn.sha256(), problem_sha256(dataset_dir))
            if why or stamp is None:
                print(
                    f"{', '.join(guarded)}: {why}; run `structbench-datagen preflight "
                    f"--dataset {dataset_dir} --work-root {args.work_root}`, or pass "
                    "--no-preflight to record the omission",
                    file=sys.stderr,
                )
                return 2
            preflight = {
                "stamp_sha256": file_sha256(sweep_dir / STAMP_FILE),
                "created_utc": stamp.get("created_utc"),
            }
    if dataset_repo["dirty"]:
        print(
            "warning: the dataset repository has uncommitted changes", file=sys.stderr
        )
    counts, problems = materialise(
        sweep_dir,
        selected,
        defn=defn,
        problem=problem,
        dataset_dir=dataset_dir,
        force=args.force,
        preflight=preflight,
    )
    write_manifest(sweep_dir, specs)
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
