"""Follow a running sweep: export and convert finished cases while it proceeds.

    structbench-datagen follow --sweep <work-root>/<name>
        [--abaqus EXE] [--interval 60] [--once] [--split NAME ...]

Each round hands the cases whose run completed and that have no export to the
exporter (one ``abaqus python`` call), then converts every case that has an
export and no canonical file. It stops after one round with ``--once``, and
otherwise when the runner's lock is gone and nothing is left to export -- or
when the runner is gone and the exporter made no progress, so a broken
exporter cannot keep it spinning. Exit codes: 0 clean, 1 an export or a
conversion failed, 2 no solver (ADR-0071).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from structbench.datagen.convert import convert_sweep
from structbench.datagen.export import export_cases
from structbench.datagen.run import LOCK_NAME


@dataclass
class FollowReport:
    rounds: int = 0
    exported: list[str] = field(default_factory=list)  # ids handed to the exporter
    converted: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # case id, or "export"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_export(sweep: Path, splits: Sequence[str] | None = None) -> list[str]:
    """Cases whose run completed and that have no ``<id>.npz`` yet."""
    out = []
    for case_dir in sorted(p for p in sweep.iterdir() if (p / "run.json").is_file()):
        if (case_dir / f"{case_dir.name}.npz").is_file():
            continue
        if _load(case_dir / "run.json").get("status") != "completed":
            continue
        if splits and _load(case_dir / "provenance.json").get("split") not in splits:
            continue
        out.append(case_dir.name)
    return out


def follow(
    sweep: Path,
    abaqus: str,
    *,
    interval: float = 60.0,
    once: bool = False,
    splits: Sequence[str] | None = None,
    exporter_args: Sequence[str] | None = None,
    echo: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> FollowReport:
    """Export and convert in rounds until the runner is gone; see the module."""
    report = FollowReport()
    lock = sweep / LOCK_NAME
    while True:
        report.rounds += 1
        pending = to_export(sweep, splits)
        exported_ok = True
        if pending:
            rc = export_cases(sweep, abaqus, cases=pending, exporter_args=exporter_args)
            if rc == 0:
                report.exported += pending
            else:
                exported_ok = False
                report.failed["export"] = (
                    f"round {report.rounds}: the exporter returned {rc}"
                )
        conversion = convert_sweep(sweep, splits=list(splits) if splits else None)
        report.converted += conversion.written
        report.failed.update(conversion.failed)
        echo(
            f"round {report.rounds}: exported {len(pending) if exported_ok else 0}, "
            f"converted {len(conversion.written)}, failed {len(conversion.failed)}"
        )
        if once:
            return report
        remaining = to_export(sweep, splits)
        if not lock.exists():
            if not remaining:
                return report
            if not exported_ok or remaining == pending:
                echo(
                    f"the runner is gone and {len(remaining)} case(s) stay unexported; "
                    "stopping"
                )
                return report
        sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="structbench-datagen follow", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--abaqus", default="abaqus")
    parser.add_argument("--interval", type=float, default=60.0, help="seconds")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--split", action="append")
    parser.add_argument(
        "--exporter-args", nargs="*", default=None, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if shutil.which(args.abaqus) is None:
        print(f"abaqus executable {args.abaqus!r} not found", file=sys.stderr)
        return 2
    report = follow(
        args.sweep,
        args.abaqus,
        interval=args.interval,
        once=args.once,
        splits=args.split,
        exporter_args=args.exporter_args,
    )
    for name, reason in sorted(report.failed.items()):
        print(f"  {name}: {reason}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
