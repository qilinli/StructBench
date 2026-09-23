"""Collect the run-evidence record of the 2D Notched-Beam impact sweep.

Per-dataset glue (ADR-0066 clause 3, ADR-0016 §6). It knows only where this
dataset's runs live and what their text files are called:

- ``<data-root>/InitialVelocity/<Shape>/80<span>/A<n><v>/`` holds one run of
  the impact grid; its case id is ``NB-I-<span>-<Shape>-<n>-<v>``;
- ``<data-root>/2DGeneralizibility/<folder>/`` holds one probe run; its case
  id is the folder name verbatim (e.g. ``S_80_400_V140_intrapolation``);
- ``mes0000`` is the message file of rank 0;
- the unit convention is ``kg-mm-ms`` (see ``convert.py``, ADR-0030).

**These runs kept no global-statistics file.** The decks never requested
``*DATABASE_GLSTAT``, so there is no energy ledger and no time-step history to
read: every record comes out with ``ledger=None``, and the checks that rest on
evidence item E5 report ``not_assessable`` naming what is missing. That is the
honest reading of this archive, not a gap to paper over with a substitute
(CORRECTIONS 2026-09-21).

The sweep's bend family (``NB-B-…``, ``ConstantVelocity/``) is descoped
(ADR-0056) and is not mapped here.

Paths are *built* from case ids — nothing is globbed or walked — and only one
small text file per run is opened, so no d3plot is hydrated. The message file
contains a licence number, a host name and local paths; none of it is kept:
``structbench.core.read_run_evidence`` returns numbers, enum values and version
tokens only, and the one file this script writes is built from that record. It
is not part of the importable package (ADR-0010).

Run from the repo root with the project environment (``SCRIPT`` is this file)::

    python SCRIPT --out runs/datachecks/notch_run_evidence.json
    python SCRIPT --case NB-I-320-Bullet-a-40 --out runs/datachecks/one.json

then pass the file to ``python -m structbench.cli.datacheck measure
--run-evidence``.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from structbench.benchmarks import get_benchmark
from structbench.core import RunEvidence, read_run_evidence
from structbench.core.io import dump_run_evidence

BENCHMARK = "notch_beam_2d_impact"
SOURCE_UNITS = "kg-mm-ms"
MESSAGES_NAME = "mes0000"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_ROOT = (
    _REPO_ROOT.parent / "data" / "StructBench" / "raw" / "notch_beam_2d"
)
#: ``NB-I-<span>-<Shape>-<n>-<v>`` — the impact grid.
_GRID_ID = re.compile(r"^NB-I-(\d+)-([A-Za-z]+)-([a-z])-(\d+)$")
#: A probe case id is its folder name; the character class is what keeps a
#: separator (and so a path outside the data root) out of the built path.
_PROBE_ID = re.compile(r"^[A-Za-z0-9_]+$")
_LOG = logging.getLogger("collect_run_evidence")


def run_dir(data_root: Path, case_id: str) -> Path:
    """``NB-I-<span>-<Shape>-<n>-<v>`` or a probe folder name -> its run dir."""
    grid = _GRID_ID.fullmatch(case_id)
    if grid is not None:
        span, shape, notch, velocity = grid.groups()
        return (
            data_root / "InitialVelocity" / shape / f"80{span}" / f"A{notch}{velocity}"
        )
    if _PROBE_ID.fullmatch(case_id):
        return data_root / "2DGeneralizibility" / case_id
    raise ValueError(f"not a case id of this benchmark: {case_id!r}")


def _text(path: Path) -> str | None:
    return (
        path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
    )


def collect(data_root: Path, case_ids: Sequence[str]) -> dict[str, RunEvidence]:
    """Read each run's message file into a ``RunEvidence``."""
    records = {}
    for case_id in case_ids:
        messages = _text(run_dir(data_root, case_id) / MESSAGES_NAME)
        if messages is None:
            _LOG.warning("%s: no run record found", case_id)
            continue
        records[case_id] = read_run_evidence(
            messages_text=messages,
            global_statistics_text=None,  # these runs wrote none
            source_units=SOURCE_UNITS,
        )
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--case", action="append", help="case id; repeatable")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.data_root.is_dir():
        _LOG.error("data root is not a directory: %s", args.data_root)
        return 2
    splits = get_benchmark(BENCHMARK).splits
    case_ids = args.case or sorted({cid for ids in splits.values() for cid in ids})
    records = collect(args.data_root, case_ids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump_run_evidence(records), encoding="utf-8", newline="\n")
    _LOG.info("%d of %d runs -> %s", len(records), len(case_ids), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
