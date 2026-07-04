"""Batch-convert the 2D Notched-Beam LS-DYNA sweep to canonical HDF5.

Per-dataset glue (ADR-0016 §6). It knows only this dataset's specifics:

- where the runs live: two families (ConstantVelocity / InitialVelocity) plus
  a generalisation-probe set (2DGeneralizibility);
- the source unit convention is ``g-mm-ms`` (the deck has no ``*CONTROL_UNITS``
  card, so the convention is supplied here per ADR-0016 §5);
- the model is 2D (``DIMENSION = 2``);
- the deck name is ``Beam1.k``;
- case-id naming:

  * Bend family (ConstantVelocity):
    ``NB-B-<span>-<Ln>-<v>`` where span ∈ {320, 480, 640},
    L ∈ {A, B, C}, n ∈ {a, b, c}, v ∈ {8, 12, 16, 20};
    run dir = ``ConstantVelocity/80<span>/<Ln><v>/``.

  * Impact family (InitialVelocity):
    ``NB-I-<span>-<Shape>-<n>-<v>`` where span ∈ {320, 480, 640},
    Shape ∈ {Bullet, Rectangular, Sphere}, n ∈ {a, b, c},
    v ∈ {40, 80, 120, 160};
    run dir = ``InitialVelocity/<Shape>/80<span>/A<n><v>/``.

  * Probes (2DGeneralizibility):
    case-id = folder name verbatim (e.g. ``C_60_240_V22_extrapolation``);
    run dir = ``2DGeneralizibility/<folder>/``.

The sweep covers 108 Bend + 108 Impact + 5 probes = 221 cases total.

Known data quirk: ConstantVelocity/80320/Aa12/Beam1.k is an LS-PrePost state
export (no material cards); NB-B-320-Aa-12 was converted with the sibling Aa8
deck — the adapter reads the deck only for materials, which are identical across
the family (2026-07-04).

All extraction is delegated to ``structbench.core.io.lsdyna.lsdyna_to_case``;
this script never touches response data. It is not part of the importable
package (ADR-0010).

Run with the project venv from the repo root. ``SCRIPT`` below stands for
``data_generation/lsdyna/2DNotchBeam/convert.py``::

    uv run python SCRIPT --dry-run          # list 221 cases, read nothing
    uv run python SCRIPT --case NB-B-320-Aa-8   # convert one bend case
    uv run python SCRIPT --case NB-I-320-Bullet-a-40  # convert one impact case
    uv run python SCRIPT --out D:/out       # choose the output directory

``--dry-run`` lists discovered cases without reading any d3plot, so it triggers
no OneDrive hydration. A full batch hydrates 221 d3plot families (tens of GB)
from OneDrive; use ``--case`` for spot conversions; re-runs skip existing outputs.
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

DATASET_ID = "2D-Notched-Beam"
SOURCE_UNITS = "g-mm-ms"  # no *CONTROL_UNITS in the deck (ADR-0016 §5)
DIMENSION = 2
DECK_NAME = "Beam1.k"

#: <repo>/data_generation/lsdyna/<dataset>/convert.py -> repo root is parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_ROOT = _REPO_ROOT.parent / "data" / "Concrete-Beam" / "2DNotchBeam"

_LOG = logging.getLogger("convert")

# --- Enumeration grid constants ---

_SPANS = ("320", "480", "640")
_BEND_L = ("A", "B", "C")
_BEND_N = ("a", "b", "c")
_BEND_V = ("8", "12", "16", "20")
_IMPACT_SHAPES = ("Bullet", "Rectangular", "Sphere")
_IMPACT_N = ("a", "b", "c")
_IMPACT_V = ("40", "80", "120", "160")


@dataclass(frozen=True)
class Run:
    """One LS-DYNA run: a named case id plus its directory on disk."""

    case_id: str
    run_dir: Path

    @property
    def d3plot(self) -> Path:
        return self.run_dir / "d3plot"

    @property
    def deck(self) -> Path:
        return self.run_dir / DECK_NAME


def discover_runs(data_root: Path) -> list[Run]:
    """Enumerate all 221 expected runs across three families (no hydration).

    Enumerates the spec grid explicitly (loops over value tuples, checks
    ``run_dir.exists()``) rather than globbing letters — anything on disk
    beyond the grid is thereby ignored. Logs a count of grid cells whose
    directory is missing (expect 0).

    Directory enumeration and existence checks only -- this function does not
    read d3plot contents, so it does not trigger OneDrive hydration.
    """
    runs: list[Run] = []
    missing_grid_cells = 0

    # --- Bend family (ConstantVelocity) ---
    for span in _SPANS:
        span_dir = data_root / "ConstantVelocity" / f"80{span}"
        for L in _BEND_L:
            for n in _BEND_N:
                for v in _BEND_V:
                    run_dir = span_dir / f"{L}{n}{v}"
                    if not run_dir.exists():
                        _LOG.warning("MISSING grid cell (Bend): %s", run_dir)
                        missing_grid_cells += 1
                        continue
                    case_id = f"NB-B-{span}-{L}{n}-{v}"
                    if (run_dir / "d3plot").exists() and (run_dir / DECK_NAME).exists():
                        runs.append(Run(case_id=case_id, run_dir=run_dir))
                    else:
                        _LOG.warning(
                            "SKIP %s: d3plot or deck missing in %s", case_id, run_dir
                        )

    # --- Impact family (InitialVelocity) ---
    for shape in _IMPACT_SHAPES:
        for span in _SPANS:
            span_dir = data_root / "InitialVelocity" / shape / f"80{span}"
            for n in _IMPACT_N:
                for v in _IMPACT_V:
                    run_dir = span_dir / f"A{n}{v}"
                    if not run_dir.exists():
                        _LOG.warning("MISSING grid cell (Impact): %s", run_dir)
                        missing_grid_cells += 1
                        continue
                    case_id = f"NB-I-{span}-{shape}-{n}-{v}"
                    if (run_dir / "d3plot").exists() and (run_dir / DECK_NAME).exists():
                        runs.append(Run(case_id=case_id, run_dir=run_dir))
                    else:
                        _LOG.warning(
                            "SKIP %s: d3plot or deck missing in %s", case_id, run_dir
                        )

    # --- Probe family (2DGeneralizibility) ---
    gen_root = data_root / "2DGeneralizibility"
    if gen_root.exists():
        for probe_dir in sorted(p for p in gen_root.iterdir() if p.is_dir()):
            if (probe_dir / DECK_NAME).exists():
                case_id = probe_dir.name
                if (probe_dir / "d3plot").exists():
                    runs.append(Run(case_id=case_id, run_dir=probe_dir))
                else:
                    _LOG.warning(
                        "SKIP probe %s: d3plot missing in %s", case_id, probe_dir
                    )
    else:
        _LOG.warning("2DGeneralizibility directory not found under %s", data_root)

    if missing_grid_cells:
        _LOG.warning("%d grid cell(s) missing from disk", missing_grid_cells)
    else:
        _LOG.info("0 missing grid cells (all expected directories present)")

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
        description="Batch-convert the 2D Notched-Beam SPH sweep to canonical HDF5."
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
        help=(
            "convert only one case, given as its case id "
            "(e.g. NB-B-320-Aa-8, NB-I-320-Bullet-a-40, C_60_240_V22_extrapolation)"
        ),
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
        runs = [r for r in runs if r.case_id == args.case]
    if not runs:
        _LOG.error("no runs found under %s (case filter %r)", args.data_root, args.case)
        return 1

    print(f"data root : {args.data_root}")
    print(f"output    : {out_dir}")
    print(f"{len(runs)} case(s) discovered:")
    for run in runs:
        print(f"  {run.case_id:40s} <- {run.run_dir.relative_to(args.data_root)}")
    if args.dry_run:
        print("\n(dry run -- nothing converted, no d3plot read or hydrated)")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    failures: list[tuple[str, str]] = []
    for run in runs:
        out_path = out_dir / f"{run.case_id}.h5"
        if out_path.exists() and not args.overwrite:
            print(f"  SKIP {run.case_id:40s} (exists; --overwrite to redo)")
            done += 1
            continue
        try:
            written = convert_run(run, out_dir)
            size_mib = written.stat().st_size / 1024 / 1024
            print(f"  OK   {run.case_id:40s} {size_mib:7.1f} MiB")
            done += 1
        except Exception as exc:  # noqa: BLE001 - batch driver keeps going
            failures.append((run.case_id, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL {run.case_id:40s} {type(exc).__name__}: {exc}")

    print(f"\n{done}/{len(runs)} done, {len(failures)} failed")
    for case_id, err in failures:
        print(f"  {case_id}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
