"""Download-and-convert: MeshGraphNets ``deforming_plate`` tfrecords to canonical HDF5.

Per-dataset glue (ADR-0016 §6 pattern, extended by ADR-0042 for nodal-FE
ingestion). Unlike the LS-DYNA converters, this dataset is not held on
OneDrive: it carries **no redistribution licence** (only the DeepMind
*meshgraphnets* code is Apache-2.0; the hosted data itself states no terms).
StructBench therefore ships download-and-convert, not rehost (ADR-0042 §2a):
run this script in a throwaway environment with ``tensorflow`` +
``structbench`` installed, against a copy of the dataset downloaded from the
source GCS bucket yourself -- see ``README.md`` in this directory for the
download step.

All extraction is delegated to
``structbench.core.io.meshgraphnets.read_deforming_plate`` (lazy TF import)
and ``build_deforming_plate_case``; this script never touches response data
itself and is not part of the importable package (ADR-0010).

``SOURCE_UNITS`` below is a **placeholder** -- see the constant's comment.

Run with the throwaway TF env from the repo root. ``SCRIPT`` below stands for
``data_generation/meshgraphnets/deforming_plate/convert.py``::

    python SCRIPT --data-root <dir> --out <dir> --split valid --limit 2
    python SCRIPT --data-root <dir> --out <dir>              # full 1000/100/100

``--help`` needs no TensorFlow or data: TF is imported lazily inside
``read_deforming_plate`` and is only touched once a split actually starts
iterating.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from structbench.core import write_case
from structbench.core.io.meshgraphnets import (
    build_deforming_plate_case,
    read_deforming_plate,
)

DATASET_ID = "deforming_plate"

# PLACEHOLDER: world_pos/stress units are undocumented upstream (meta.json and
# the paper are silent). "kg-m-s" is provisional -- a later human task (Task 8
# of the ingestion plan) measures the true convention against the downloaded
# data and patches this constant before the archive is trusted (ADR-0042 §2b).
SOURCE_UNITS = "kg-m-s"

#: Source split (the on-disk <split>.tfrecord file) -> canonical split name
#: used in output filenames. StructBench's platform-wide convention is
#: train/val/test (see benchmarks.registry); MeshGraphNets' own file naming
#: is train/valid/test, hence the "valid" -> "val" rename here.
SPLITS = {"train": "train", "valid": "val", "test": "test"}

_LOG = logging.getLogger("convert")


def convert_split(
    data_dir: Path,
    source_split: str,
    canon_split: str,
    out_dir: Path,
    *,
    limit: int | None,
    overwrite: bool,
) -> tuple[int, list[tuple[str, str]]]:
    """Convert one source split's trajectories to canonical HDF5.

    Iterates ``read_deforming_plate(data_dir, source_split)``, writing each
    trajectory as ``<out_dir>/<canon_split>_<i>.h5``. A failure building or
    writing one case is caught and recorded so the batch keeps going past a
    bad case (mirrors the Taylor/Wave1D drivers); a failure decoding the
    tfrecord stream itself (bad ``data_dir``, missing file, TF not installed)
    is not caught here and propagates as a fatal error for the whole split.

    Parameters
    ----------
    data_dir:
        Directory holding ``meta.json`` and ``<source_split>.tfrecord``.
    source_split:
        Split name as it appears on disk, e.g. ``"valid"``.
    canon_split:
        Canonical split name used in output filenames, e.g. ``"val"``.
    out_dir:
        Directory to write ``<canon_split>_<i>.h5`` into (must already exist).
    limit:
        Convert only the first ``limit`` trajectories (``None`` = all).
    overwrite:
        Reconvert cases whose output ``.h5`` already exists.

    Returns
    -------
    tuple[int, list[tuple[str, str]]]
        Number of cases written, and ``(case_id, error)`` pairs for cases
        that failed to convert.
    """
    n_ok = 0
    failures: list[tuple[str, str]] = []
    for i, arrays in enumerate(read_deforming_plate(data_dir, source_split)):
        if limit is not None and i >= limit:
            break
        case_id = f"{canon_split}_{i:04d}"
        out_path = out_dir / f"{case_id}.h5"
        if out_path.exists() and not overwrite:
            print(f"  SKIP {case_id:16s} (exists; --overwrite to redo)")
            continue
        try:
            case = build_deforming_plate_case(
                arrays,
                source_units=SOURCE_UNITS,
                case_id=case_id,
                dataset_id=DATASET_ID,
            )
            write_case(case, out_path)
        except Exception as exc:  # noqa: BLE001 - batch driver keeps going
            failures.append((case_id, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL {case_id:16s} {type(exc).__name__}: {exc}")
            continue
        print(f"  OK   {case_id:16s} ({case.nodes.coords.shape[0]} nodes)")
        n_ok += 1
    return n_ok, failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert MeshGraphNets deforming_plate tfrecords to canonical "
        "HDF5 (ADR-0042). Data must already be downloaded locally -- see README.md."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="dir holding meta.json + {train,valid,test}.tfrecord "
        "(from download_dataset.sh or the GCS bucket)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory for canonical .h5 (e.g. canonical/deforming_plate)",
    )
    parser.add_argument(
        "--split",
        choices=list(SPLITS),
        default=None,
        help="convert only this source split (default: all of train/valid/test)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="convert only the first N trajectories per split (for a smoke run)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reconvert cases whose output .h5 already exists",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    splits = {args.split: SPLITS[args.split]} if args.split else SPLITS
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"data root : {args.data_root}")
    print(f"output    : {args.out}")
    print(f"split(s)  : {', '.join(splits)}")

    total_ok = 0
    all_failures: list[tuple[str, str]] = []
    for source_split, canon_split in splits.items():
        try:
            n_ok, failures = convert_split(
                args.data_root,
                source_split,
                canon_split,
                args.out,
                limit=args.limit,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001 - one bad split shouldn't stop the rest
            _LOG.error("split %r failed: %s: %s", source_split, type(exc).__name__, exc)
            all_failures.append((f"{canon_split}:*", f"{type(exc).__name__}: {exc}"))
            continue
        total_ok += n_ok
        all_failures.extend(failures)

    print(f"\n{total_ok} case(s) written to {args.out}, {len(all_failures)} failed")
    for case_id, err in all_failures:
        print(f"  {case_id}: {err}")
    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main())
