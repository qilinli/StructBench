"""``structbench-datagen``: the data-generation stages behind one command (ADR-0071).

    structbench-datagen new      <dir>                       scaffold a definition
    structbench-datagen check    <dir>                       check it, no solver
    structbench-datagen generate --dataset <dir> --work-root <runs> [...]
    structbench-datagen run      --sweep <runs>/<name> [...]
    structbench-datagen export   --sweep <runs>/<name> [--cases ID ...] [--abaqus EXE]
    structbench-datagen convert  --sweep <runs>/<name> [...]
    structbench-datagen verify   --sweep <runs>/<name> --dataset <dir> [...]
    structbench-datagen converge --dataset <dir> --sweep <runs>/<name>
                                 [--root DIR ...] [--out DIR]
    structbench-datagen archive  --sweep <runs>/<name> --dataset <dir>
                                 --data-root <tree> [...]

Each stage's own options are its module's; ``export`` hands the packaged
exporter to ``abaqus python`` because it must run under the solver's Python.
``verify`` runs the ADR-0066 instrument; validation against experiments is
``structbench-validate`` (ADR-0072), and the old stage name is refused.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from structbench.datagen import (
    archive,
    converge,
    convert,
    export,
    generate,
    run,
    template,
    verify,
)
from structbench.datagen.export import EXPORTER as EXPORTER
from structbench.datagen.export import export_command as export_command

STAGES: dict[str, Callable[[list[str] | None], int]] = {
    "new": template.main_new,
    "check": template.main_check,
    "generate": generate.main,
    "run": run.main,
    "export": export.main,
    "convert": convert.main,
    "verify": verify.main,
    "converge": converge.main,
    "archive": archive.main,
}
#: Old names, refused with a pointer (ADR-0072: validate now means
#: comparison with experiment).
RENAMED = {"validate": "verify"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="structbench-datagen", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("stage", choices=[*STAGES, *RENAMED])
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.stage in RENAMED:
        new = RENAMED[args.stage]
        print(
            f'the stage is now "{new}" (ADR-0072): structbench-datagen {new} ...',
            file=sys.stderr,
        )
        return 2
    return STAGES[args.stage](args.rest)


if __name__ == "__main__":
    sys.exit(main())
