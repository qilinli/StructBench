"""``structbench-datagen``: the data-generation stages behind one command (ADR-0071).

    structbench-datagen new      <dir>                       scaffold a definition
    structbench-datagen check    <dir>                       validate it, no solver
    structbench-datagen generate --dataset <dir> --work-root <runs> [...]
    structbench-datagen run      --sweep <runs>/<name> [...]
    structbench-datagen export   --sweep <runs>/<name> [--cases ID ...] [--abaqus EXE]
    structbench-datagen convert  --sweep <runs>/<name> [...]
    structbench-datagen validate --sweep <runs>/<name> --dataset <dir> [...]
    structbench-datagen archive  --sweep <runs>/<name> --dataset <dir>
                                 --data-root <tree> [...]

Each stage's own options are its module's; ``export`` hands the packaged
exporter to ``abaqus python`` because it must run under the solver's Python.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from importlib import resources
from pathlib import Path

from structbench.datagen import archive, convert, generate, run, template, validate

EXPORTER = resources.files("structbench.datagen.abaqus") / "odb_export.py"

STAGES: dict[str, Callable[[list[str] | None], int]] = {
    "new": template.main_new,
    "check": template.main_check,
    "generate": generate.main,
    "run": run.main,
    "convert": convert.main,
    "validate": validate.main,
    "archive": archive.main,
}


def export_command(abaqus: str, sweep: Path, cases: Sequence[str] | None) -> list[str]:
    """The ``abaqus python odb_export.py ...`` command line for a sweep."""
    with resources.as_file(EXPORTER) as path:
        command = [abaqus, "python", str(path), "--sweep", str(sweep)]
    if cases:
        command += ["--cases", *cases]
    return command


def _export(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(
        prog="structbench-datagen export",
        description="export ODBs to abaqus-npz/1 under Abaqus's Python",
    )
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--abaqus", default="abaqus")
    # tests: a stub script in place of "python <exporter>"
    parser.add_argument(
        "--exporter-args", nargs="*", default=None, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    exe = shutil.which(args.abaqus)
    if exe is None:
        print(f"abaqus executable {args.abaqus!r} not found", file=sys.stderr)
        return 2
    command = export_command(exe, args.sweep, args.cases)
    if args.exporter_args is not None:
        command = [exe, *args.exporter_args, "--sweep", str(args.sweep)]
    return subprocess.run(command).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="structbench-datagen", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("stage", choices=[*STAGES, "export"])
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.stage == "export":
        return _export(args.rest)
    return STAGES[args.stage](args.rest)


if __name__ == "__main__":
    sys.exit(main())
