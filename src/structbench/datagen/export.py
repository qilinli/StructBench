"""Export a sweep's ODBs to ``abaqus-npz/1`` under Abaqus's own Python.

    structbench-datagen export --sweep <work-root>/<name>
        [--cases ID ...] [--abaqus EXE]

The exporter itself is package data (``datagen/abaqus/odb_export.py``: Python
3.10, ``odbAccess``, no imports from ``structbench``); this module only builds
and runs ``abaqus python <exporter> --sweep ...``. The ``preflight`` and
``follow`` stages call :func:`export_cases` directly (ADR-0069, ADR-0071).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

EXPORTER = resources.files("structbench.datagen.abaqus") / "odb_export.py"


def export_command(abaqus: str, sweep: Path, cases: Sequence[str] | None) -> list[str]:
    """The ``abaqus python odb_export.py ...`` command line for a sweep."""
    with resources.as_file(EXPORTER) as path:
        command = [abaqus, "python", str(path), "--sweep", str(sweep)]
    if cases:
        command += ["--cases", *cases]
    return command


def export_cases(
    sweep: Path,
    abaqus: str,
    *,
    cases: Sequence[str] | None = None,
    exporter_args: Sequence[str] | None = None,
) -> int:
    """Run the exporter over ``sweep`` (or ``cases`` of it).

    Returns the exporter's return code, or 2 when ``abaqus`` is not on the
    path. ``exporter_args`` replaces ``python <exporter>`` on the command line
    (tests: a stub script in place of the solver's interpreter).
    """
    exe = shutil.which(abaqus)
    if exe is None:
        print(f"abaqus executable {abaqus!r} not found", file=sys.stderr)
        return 2
    command = export_command(exe, sweep, cases)
    if exporter_args is not None:
        command = [exe, *exporter_args, "--sweep", str(sweep)]
        if cases:
            command += ["--cases", *cases]
    return subprocess.run(command).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="structbench-datagen export", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--abaqus", default="abaqus")
    parser.add_argument(
        "--exporter-args", nargs="*", default=None, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    return export_cases(
        args.sweep, args.abaqus, cases=args.cases, exporter_args=args.exporter_args
    )


if __name__ == "__main__":
    sys.exit(main())
