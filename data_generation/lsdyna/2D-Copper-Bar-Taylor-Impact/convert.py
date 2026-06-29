"""Batch-convert the 2D Copper Bar Taylor Impact LS-DYNA sweep to canonical HDF5.

Per-dataset glue (ADR-0016 §6). It knows only this dataset's specifics:

- where the runs live: ``<data-root>/lsdyna/20<geom>/<vel>/`` (one LS-DYNA run
  each: a ``d3plot`` family plus a paired ``Taylor.k`` deck);
- the source unit convention is ``g-mm-ms`` (the deck has no ``*CONTROL_UNITS``
  card, so the convention is supplied here per ADR-0016 §5);
- the model is 2D;
- the case-id naming is ``T-20-<geom>-<vel>``.

All extraction is delegated to ``structbench.core.io.lsdyna.lsdyna_to_case``;
this script never touches response data. It is not part of the importable
package (ADR-0010).

Run with the project venv from the repo root. ``SCRIPT`` below stands for this
file, ``data_generation/lsdyna/2D-Copper-Bar-Taylor-Impact/convert.py``::

    python SCRIPT --dry-run      # list discovered cases, read nothing
    python SCRIPT --case 60/100  # convert one case
    python SCRIPT --out D:/out   # choose the output directory

``--dry-run`` lists discovered cases without reading any d3plot, so it triggers
no OneDrive hydration. A full batch reads every d3plot family (~340 MB each).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from structbench.core import write_case
from structbench.core.io.lsdyna import lsdyna_to_case

DATASET_ID = "2D-Copper-Bar-Taylor-Impact"
SOURCE_UNITS = "g-mm-ms"  # no *CONTROL_UNITS in the deck (ADR-0016 §5)
DIMENSION = 2
DECK_NAME = "Taylor.k"

#: <repo>/data_generation/lsdyna/<dataset>/convert.py -> repo root is parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_ROOT = _REPO_ROOT.parent / "data" / DATASET_ID

_LOG = logging.getLogger("convert")


@dataclass(frozen=True)
class Run:
    """One LS-DYNA run in the sweep."""

    geom: str
    vel: str
    run_dir: Path

    @property
    def case_id(self) -> str:
        return f"T-20-{self.geom}-{self.vel}"

    @property
    def d3plot(self) -> Path:
        return self.run_dir / "d3plot"

    @property
    def deck(self) -> Path:
        return self.run_dir / DECK_NAME


def discover_runs(data_root: Path) -> list[Run]:
    """Find ``lsdyna/20<geom>/<vel>/`` runs that hold a d3plot and a deck.

    Directory enumeration and existence checks only -- it does not read d3plot
    contents, so it does not trigger OneDrive hydration.
    """
    runs: list[Run] = []
    for geom_dir in sorted((data_root / "lsdyna").glob("20*")):
        if not geom_dir.is_dir():
            continue
        geom = geom_dir.name[2:]  # "2060" -> "60"
        for run_dir in sorted(p for p in geom_dir.glob("*") if p.is_dir()):
            run = Run(geom=geom, vel=run_dir.name, run_dir=run_dir)
            if run.d3plot.exists() and run.deck.exists():
                runs.append(run)
    return runs


def convert_run(run: Run, out_dir: Path) -> Path:
    """Convert one run to canonical HDF5 and return the output path."""
    case = lsdyna_to_case(
        run.d3plot,
        run.deck,
        source_units=SOURCE_UNITS,
        dimension=DIMENSION,
        case_id=run.case_id,
        dataset_id=DATASET_ID,
    )
    out_path = out_dir / f"{run.case_id}.h5"
    write_case(case, out_path)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-convert the Taylor 2D SPH sweep to canonical HDF5."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_DATA_ROOT,
        help=f"dataset root (default: {_DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory for .h5 (default: <data-root>/h5_canonical)",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="convert only one case, given as GEOM/VEL (e.g. 60/100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list discovered cases and exit; reads no d3plot (no hydration)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reconvert cases whose output .h5 already exists",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = args.out or (args.data_root / "h5_canonical")

    runs = discover_runs(args.data_root)
    if args.case:
        runs = [r for r in runs if f"{r.geom}/{r.vel}" == args.case]
    if not runs:
        _LOG.error("no runs found under %s (case filter %r)", args.data_root, args.case)
        return 1

    print(f"data root : {args.data_root}")
    print(f"output    : {out_dir}")
    print(f"{len(runs)} case(s) discovered:")
    for run in runs:
        print(f"  {run.case_id:16s} <- lsdyna/20{run.geom}/{run.vel}/d3plot")
    if args.dry_run:
        print("\n(dry run -- nothing converted, no d3plot read or hydrated)")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    failures: list[tuple[str, str]] = []
    for run in runs:
        out_path = out_dir / f"{run.case_id}.h5"
        if out_path.exists() and not args.overwrite:
            print(f"  SKIP {run.case_id:16s} (exists; --overwrite to redo)")
            done += 1
            continue
        try:
            written = convert_run(run, out_dir)
            size_mib = written.stat().st_size / 1024 / 1024
            print(f"  OK   {run.case_id:16s} {size_mib:7.1f} MiB")
            done += 1
        except Exception as exc:  # noqa: BLE001 - batch driver keeps going
            failures.append((run.case_id, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL {run.case_id:16s} {type(exc).__name__}: {exc}")

    print(f"\n{done}/{len(runs)} done, {len(failures)} failed")
    for case_id, err in failures:
        print(f"  {case_id}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
