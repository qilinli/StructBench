"""Batch-convert the 1D Wave Propagation LS-DYNA sweep to canonical HDF5.

Per-dataset glue (ADR-0016 §6). It knows only this dataset's specifics:

- where the runs live: ``<data-root>/<length>_<velocity>/`` (one LS-DYNA run
  each: a ``d3plot`` family plus a paired ``WavePropagation.k`` deck);
- the source unit convention is ``g-mm-ms`` (the deck has no ``*CONTROL_UNITS``
  card, so the convention is supplied here per ADR-0016 §5);
- the model is 2D (thin strip / bar along x, ``*CONTROL_SPH IDIM=2``);
- the case-id naming is ``W1D-<length>-<velocity>``.

The sweep covers 4 bar lengths (200, 300, 400, 500 mm) × 4 impact velocities
(1, 2, 4, 8 mm/ms) = 16 runs total.  Run-dir names follow the ``<L>_<v>``
convention (e.g. ``200_1``, ``500_8``).

All extraction is delegated to ``structbench.core.io.lsdyna.lsdyna_to_case``;
this script never touches response data.  It is not part of the importable
package (ADR-0010).

Run with the project venv from the repo root.  ``SCRIPT`` below stands for
``data_generation/lsdyna/1DWavePropagation/convert.py``::

    python SCRIPT --dry-run          # list discovered cases, read nothing
    python SCRIPT --case 200/1       # convert one case
    python SCRIPT --out D:/out       # choose the output directory

``--dry-run`` lists discovered cases without reading any d3plot, so it triggers
no OneDrive hydration.  A full batch hydrates 16 d3plot families.
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

DATASET_ID = "1D-Wave-Propagation"
SOURCE_UNITS = "g-mm-ms"  # no *CONTROL_UNITS in the deck (ADR-0016 §5)
DIMENSION = 2  # *CONTROL_SPH IDIM=2; thin strip, bar along x
DECK_NAME = "WavePropagation.k"

#: <repo>/data_generation/lsdyna/<dataset>/convert.py -> repo root is parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_ROOT = _REPO_ROOT.parent / "data" / "Concrete-Beam" / "1DWavePropagation"

_LOG = logging.getLogger("convert")


@dataclass(frozen=True)
class Run:
    """One LS-DYNA run: <data-root>/<length>_<velocity>/."""

    length: str
    velocity: str
    run_dir: Path

    @property
    def case_id(self) -> str:
        return f"W1D-{self.length}-{self.velocity}"

    @property
    def d3plot(self) -> Path:
        return self.run_dir / "d3plot"

    @property
    def deck(self) -> Path:
        return self.run_dir / DECK_NAME


def discover_runs(data_root: Path) -> list[Run]:
    """Enumerate <L>_<v> run dirs holding a d3plot + deck (no hydration).

    Directory enumeration and existence checks only -- it does not read d3plot
    contents, so it does not trigger OneDrive hydration.
    """
    runs: list[Run] = []
    for run_dir in sorted(p for p in data_root.glob("[0-9]*_[0-9]*") if p.is_dir()):
        length, _, velocity = run_dir.name.partition("_")
        run = Run(length=length, velocity=velocity, run_dir=run_dir)
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
        description="Batch-convert the 1D Wave Propagation SPH sweep to canonical HDF5."
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
        help="convert only one case, given as LENGTH/VELOCITY (e.g. 200/1)",
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
        runs = [r for r in runs if f"{r.length}/{r.velocity}" == args.case]
    if not runs:
        _LOG.error("no runs found under %s (case filter %r)", args.data_root, args.case)
        return 1

    print(f"data root : {args.data_root}")
    print(f"output    : {out_dir}")
    print(f"{len(runs)} case(s) discovered:")
    for run in runs:
        print(f"  {run.case_id:16s} <- {run.length}_{run.velocity}/d3plot")
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
