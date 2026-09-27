"""Collect a sweep's run evidence into one whitelisted JSON record.

    python data_generation/abaqus/collect_run_evidence.py --sweep <work-root>/<name>
        [--split NAME ...] --out <evidence.json>

For every case folder with a ``run.json`` -- completed or not, so a failed run
is reported rather than lost -- the ``.sta``, ``.msg`` and ``.dat`` go through
``read_abaqus_run_evidence`` and, when the export exists, its Assembly history
through ``abaqus_ledger``. The source units come from ``provenance.json``.

The printed file carries a licence line naming the seat and host. None of it
leaves the readers: they return numbers, enum values and version tokens, and
the file written is built from those records alone (ADR-0066 clause 3). Pass
it to ``python -m structbench.cli.datacheck measure --run-evidence``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from structbench.core import RunEvidence
from structbench.core.io import dump_run_evidence
from structbench.core.io.abaqus import abaqus_ledger
from structbench.core.io.abaqus_run import read_abaqus_run_evidence


def _text(path: Path) -> str | None:
    return (
        path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
    )


def collect_case(case_dir: Path, source_units: str) -> RunEvidence:
    """One case folder's evidence, with its ledger when the export exists."""
    stem = case_dir / case_dir.name
    evidence = read_abaqus_run_evidence(
        status_text=_text(stem.with_suffix(".sta")),
        messages_text=_text(stem.with_suffix(".msg")),
        printed_text=_text(stem.with_suffix(".dat")),
        source_units=source_units,
    )
    export = stem.with_suffix(".npz")
    if not export.is_file():
        return evidence
    try:
        ledger = abaqus_ledger(export, source_units=source_units)
    except (ValueError, NotImplementedError):
        # An export the adapter refuses has no ledger to offer, and saying so
        # is the record's job; its reason is convert.py's to report.
        unparsable = evidence.unparsable | {"energy_ledger"}
        return dataclasses.replace(evidence, unparsable=unparsable)
    return dataclasses.replace(evidence, ledger=ledger)


def collect_sweep(
    sweep: Path, *, splits: list[str] | None = None
) -> dict[str, RunEvidence]:
    """Evidence of every case of ``sweep`` (in ``splits``, if given) that ran."""
    records = {}
    for case_dir in sorted(p for p in sweep.iterdir() if (p / "run.json").is_file()):
        prov = json.loads((case_dir / "provenance.json").read_text(encoding="utf-8"))
        if splits and prov.get("split") not in splits:
            continue
        records[case_dir.name] = collect_case(case_dir, prov["units"])
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--split", action="append")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    records = collect_sweep(args.sweep, splits=args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump_run_evidence(records), encoding="utf-8")
    with_ledger = sum(r.ledger is not None for r in records.values())
    print(f"runs={len(records)} with_ledger={with_ledger} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
