"""Convert a sweep's ``abaqus-npz/1`` exports into canonical HDF5 cases.

    python data_generation/abaqus/convert.py --sweep <work-root>/<name>
        [--split NAME ...] [--out DIR]

A case is converted when its ``run.json`` says completed, its ``<id>.npz``
exists, and ``<out>/<id>.h5`` does not; so a re-run writes only what is new.
``--out`` defaults to ``<sweep>/canonical``. The dataset id is the sweep
folder's name, the source units come from each case's ``provenance.json``, and
the generation date is the run's end date. A case that fails to convert is
reported with its reason and the rest carry on; it is never half-written.

Solver-generic within Abaqus: everything dataset-specific is already in the
deck and the export (ADR-0069).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from structbench.core.exceptions import StructBenchError
from structbench.core.io import abaqus_export_to_case, write_case


@dataclass
class ConvertReport:
    written: list[str] = field(default_factory=list)
    skipped: int = 0
    failed: dict[str, str] = field(default_factory=dict)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def convert_sweep(
    sweep: Path, *, splits: list[str] | None = None, out: Path | None = None
) -> ConvertReport:
    """Convert every eligible case of ``sweep``; see the module docstring."""
    out = out or sweep / "canonical"
    report = ConvertReport()
    for case_dir in sorted(p for p in sweep.iterdir() if (p / "run.json").is_file()):
        if splits and _load(case_dir / "provenance.json").get("split") not in splits:
            continue
        target = out / f"{case_dir.name}.h5"
        partial = target.with_suffix(".partial.h5")
        completed = _load(case_dir / "run.json").get("status") == "completed"
        exported = (case_dir / f"{case_dir.name}.npz").is_file()
        if not (completed and exported) or target.exists():
            report.skipped += 1
            continue
        try:
            prov = _load(case_dir / "provenance.json")
            run = _load(case_dir / "run.json")
            case = abaqus_export_to_case(
                case_dir / f"{case_dir.name}.npz",
                (case_dir / f"{case_dir.name}.inp").read_text(encoding="utf-8"),
                source_units=prov["units"],
                dimension=2,
                case_id=case_dir.name,
                dataset_id=sweep.name,
                generation_date=str(run.get("end_utc", "unknown"))[:10],
            )
            out.mkdir(parents=True, exist_ok=True)
            write_case(case, partial)
            os.replace(partial, target)
        except (
            OSError,
            KeyError,
            ValueError,
            NotImplementedError,
            StructBenchError,
        ) as exc:
            partial.unlink(missing_ok=True)
            report.failed[case_dir.name] = f"{type(exc).__name__}: {exc}"
            continue
        report.written.append(case_dir.name)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--split", action="append")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = convert_sweep(args.sweep, splits=args.split, out=args.out)
    print(
        f"written={len(report.written)} skipped={report.skipped} "
        f"failed={len(report.failed)}"
    )
    for name, reason in sorted(report.failed.items()):
        print(f"  {name}: {reason}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
