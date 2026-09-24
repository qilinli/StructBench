"""Generate a sweep's Abaqus case folders from a dataset's sweep.toml and model.py.

    python data_generation/abaqus/generate.py --dataset <dir> --work-root <root>
        [--split NAME ...] [--force] [--dry-run]

Each case gets ``<work-root>/<name>/<case_id>/<case_id>.inp`` and a
``provenance.json``; the sweep gets a ``manifest.csv`` listing every case. An
existing identical deck is skipped. A different one is a conflict unless
``--force``, and a deck whose case already ran (``run.json``) is never replaced
(ADR-0069).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np
import sampling
import scipy

from structbench.core.io import unit_factors

_REPO = Path(__file__).resolve().parents[2]
PROVENANCE_FORMAT = "abaqus-provenance/1"
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


def load_sweep(dataset_dir: Path) -> dict[str, Any]:
    with (dataset_dir / "sweep.toml").open("rb") as handle:
        return tomllib.load(handle)


def load_model(dataset_dir: Path) -> ModuleType:
    """Import the dataset's model.py by path; it must define ``build``.

    No bytecode is written: a ``__pycache__`` inside the dataset's repository
    would make its provenance read as uncommitted work.
    """
    path = dataset_dir / "model.py"
    spec = importlib.util.spec_from_file_location(
        f"dataset_model_{dataset_dir.name}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.name} from {dataset_dir.name}")
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    if not callable(getattr(module, "build", None)):
        raise AttributeError(f"{dataset_dir.name}/model.py defines no build()")
    return module


def plan_cases(sweep: dict[str, Any]) -> list[CaseSpec]:
    """Every case of every split, in file then Sobol order."""
    prefix = sweep["dataset"]["case_prefix"]
    # An unknown unit label fails here, before any deck exists, not after a solve.
    unit_factors(sweep["dataset"]["units"])
    fixed = dict(sweep.get("fixed", {}))
    variables = sampling.parse_bounds(sweep["variables"], "variables")
    regions = {
        name: sampling.parse_bounds(box, f"regions.{name}")
        for name, box in sweep.get("regions", {}).items()
    }
    splits = sampling.parse_splits(sweep)
    seeds = Counter(s.seed for s in splits if s.seed is not None)
    shared = sorted(seed for seed, count in seeds.items() if count > 1)
    if shared:
        # Same seed, same engine: two splits would draw the same points.
        raise ValueError(f"sampled splits share seed {shared[0]}; give each its own")
    specs = []
    for split in splits:
        sampled = set(variables) | set(split.extra) | set(split.categorical)
        clash = set(fixed) & sampled
        if clash:
            raise ValueError(f"{sorted(clash)} are both fixed and sampled")
        for point in sampling.sample_split(variables, regions, split):
            for variant in split.variants or (None,):
                case_id = f"{prefix}-{split.name}-{point.index:04d}"
                case_id += f"-{variant}" if variant else ""
                if not _CASE_ID.fullmatch(case_id):
                    raise ValueError(
                        f"case id {case_id!r} is not a safe Abaqus job name"
                    )
                params = {**fixed, **point.params}
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


def write_manifest(sweep_dir: Path, specs: list[CaseSpec]) -> None:
    names = sorted({k for s in specs for k in s.params})
    with (sweep_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case_id", "split", "index", "variant", *names])
        for s in specs:
            row = [s.params.get(k, "") for k in names]
            writer.writerow([s.case_id, s.split, s.index, s.variant or "", *row])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--split", action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    dataset_dir = args.dataset.resolve()
    sweep = load_sweep(dataset_dir)
    specs = plan_cases(sweep)
    selected = [s for s in specs if not args.split or s.split in args.split]
    if args.dry_run:
        for split, count in Counter(s.split for s in selected).items():
            print(f"{split}: {count} cases")
        return 0

    model = load_model(dataset_dir)
    sweep_dir = args.work_root / sweep["dataset"]["name"]
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_sha = hashlib.sha256((dataset_dir / "sweep.toml").read_bytes()).hexdigest()
    repo, dataset_repo = git_state(_REPO), git_state(dataset_dir)
    if dataset_repo["dirty"]:
        print(
            "warning: the dataset repository has uncommitted changes", file=sys.stderr
        )
    counts: Counter[str] = Counter()
    problems = []
    for spec in selected:
        text = model.build(dict(spec.params), spec.variant)
        provenance = {
            "format": PROVENANCE_FORMAT,
            "case_id": spec.case_id,
            "dataset": sweep["dataset"]["name"],
            "units": sweep["dataset"]["units"],
            "split": spec.split,
            "index": spec.index,
            "variant": spec.variant,
            "seed": spec.seed,
            "params": spec.params,
            "sweep_sha256": sweep_sha,
            "inp_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "repository": repo,
            "dataset_repository": dataset_repo,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        outcome = write_case(
            sweep_dir / spec.case_id, text, provenance, force=args.force
        )
        counts[outcome] += 1
        if outcome in ("conflict", "locked"):
            problems.append(f"{spec.case_id}: {outcome}")
    write_manifest(sweep_dir, specs)
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
