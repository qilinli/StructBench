"""Validate a sweep: run evidence, measurements and verdicts in one pass.

    python data_generation/abaqus/validate.py --sweep <work-root>/<name>
        --dataset <dataset-dir> [--split NAME ...] [--data-root DIR]

1. Collects the run evidence of every case that ran (``collect_run_evidence``).
2. Measures the canonical cases under ``--data-root`` (default
   ``<sweep>/canonical``, what ``convert.py`` writes) against the
   ``[declaration]`` of ``<dataset-dir>/sweep.toml``. A case that ran but has
   no canonical file -- an aborted run, or one that failed to convert -- is
   measured from its deck and run record alone, so it is reported, not lost.
3. Judges the record and writes ``run_evidence.json``, ``measurements.json``
   and ``report.md`` into ``<sweep>/datacheck/``.
4. Prints the cases with no canonical file, the failing rows by case, and
   the lowest, median and highest value of each measured-only row. Exits 1
   on any ``fail`` or any case without a canonical file, else 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from collect_run_evidence import collect_sweep

from structbench import __version__
from structbench.cli.datacheck import declared_from_toml, measure_cases
from structbench.core.io import dump_run_evidence
from structbench.core.io.abaqus_run import read_abaqus_input_facts
from structbench.verification import Verdict
from structbench.verification.criteria import judge
from structbench.verification.measures import measure_case
from structbench.verification.quantities import CATALOGUE
from structbench.verification.report import render_markdown, to_json
from structbench.verification.results import DatasetMeasurements


def _without_case(sweep: Path, case_id: str, declared, run):  # noqa: ANN001
    """A run with no canonical file, measured on its deck and run record."""
    folder = sweep / case_id
    prov = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    deck = folder / f"{case_id}.inp"
    facts = (
        read_abaqus_input_facts(
            deck.read_text(encoding="utf-8"), source_units=prov["units"]
        )
        if deck.is_file()
        else None
    )
    return measure_case(None, facts, declared, case_id=case_id, run=run)


def validate_sweep(
    sweep: Path,
    dataset: Path,
    *,
    splits: list[str] | None = None,
    data_root: Path | None = None,
) -> int:
    """Run the pass described in the module docstring; returns the exit code."""
    data_root = data_root or sweep / "canonical"
    out = sweep / "datacheck"
    out.mkdir(parents=True, exist_ok=True)
    runs = collect_sweep(sweep, splits=splits)
    (out / "run_evidence.json").write_text(dump_run_evidence(runs), encoding="utf-8")
    _, declared = declared_from_toml(dataset / "sweep.toml")
    stored = sorted(cid for cid in runs if (data_root / f"{cid}.h5").is_file())
    record = measure_cases(declared, None, data_root, stored, run_evidence=runs)
    extra = [
        _without_case(sweep, cid, declared, runs[cid])
        for cid in sorted(set(runs) - set(stored))
    ]
    record = DatasetMeasurements(
        None,
        None,
        __version__,
        record.definition_versions,
        tuple(sorted((*record.cases, *extra), key=lambda c: c.case_id)),
    )
    (out / "measurements.json").write_text(to_json(record), encoding="utf-8")
    report = judge(record)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")

    failing = [
        (case.case_id, r)
        for case in report.cases
        for r in case.results
        if r.verdict is Verdict.FAIL
    ]
    print(f"cases={len(record.cases)} fail_rows={len(failing)} -> {out}")
    missing = sorted(set(runs) - set(stored))
    if missing:
        # Measured on the deck and run record alone: their response rows read
        # as absences, which a summary of fails would never show.
        print(f"  no canonical file: {', '.join(missing)}")
    by_case: dict[str, list[str]] = defaultdict(list)
    for case_id, r in failing:
        by_case[case_id].append(f"{r.quantity}={r.value}")
    for case_id in sorted(by_case):
        print(f"  fail {case_id}: {', '.join(by_case[case_id])}")
    judged = {c.quantity for c in report.criteria}
    print("measured only (lowest / median / highest over cases):")
    for q in CATALOGUE:
        if q.name in judged:
            continue
        values = [
            m.value
            for case in record.cases
            for m in case.measurements
            if m.quantity == q.name and m.value is not None
        ]
        if values:
            lo, mid, hi = np.min(values), np.median(values), np.max(values)
            print(f"  {q.name}: {lo:.4g} / {mid:.4g} / {hi:.4g} (n={len(values)})")
    return 1 if failing or missing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", action="append")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    return validate_sweep(
        args.sweep, args.dataset, splits=args.split, data_root=args.data_root
    )


if __name__ == "__main__":
    sys.exit(main())
