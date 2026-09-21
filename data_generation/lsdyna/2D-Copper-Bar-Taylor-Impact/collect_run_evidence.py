"""Collect the run-evidence record of the 2D Copper Bar Taylor Impact sweep.

Per-dataset glue (ADR-0066 clause 3, ADR-0016 §6). It knows only where this
dataset's runs live and what their text files are called:

- ``<data-root>/lsdyna/20<geom>/<vel>/`` holds one run; its case id is
  ``T-20-<geom>-<vel>``;
- ``mes0000`` is the message file of rank 0, ``glstat`` the ASCII
  global-statistics file;
- the unit convention is ``g-mm-ms`` (see ``convert.py``).

Paths are *built* from case ids — nothing is globbed or walked — and only
those two small text files are opened, so no d3plot is hydrated. The message
file contains a licence number, a host name and local paths; none of it is
kept: ``structbench.core.read_run_evidence`` returns numbers, enum values and
version tokens only, and the one file this script writes is built from that
record. It is not part of the importable package (ADR-0010).

Run from the repo root with the project environment (``SCRIPT`` is this file)::

    python SCRIPT --out runs/datachecks/taylor_run_evidence.json
    python SCRIPT --case T-20-100-100 --out runs/datachecks/one.json

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

SOURCE_UNITS = "g-mm-ms"
MESSAGES_NAME = "mes0000"
STATISTICS_NAME = "glstat"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DATA_ROOT = (
    _REPO_ROOT.parent / "data" / "StructBench" / "raw" / "taylor_impact_2d"
)
_CASE_ID = re.compile(r"^T-20-(\d+)-(\w+)$")
_LOG = logging.getLogger("collect_run_evidence")


def run_dir(data_root: Path, case_id: str) -> Path:
    """``T-20-<geom>-<vel>`` -> ``<data_root>/lsdyna/20<geom>/<vel>``."""
    found = _CASE_ID.fullmatch(case_id)
    if found is None:
        raise ValueError(f"not a case id of this dataset: {case_id!r}")
    return data_root / "lsdyna" / f"20{found.group(1)}" / found.group(2)


def _text(path: Path) -> str | None:
    return (
        path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
    )


def collect(data_root: Path, case_ids: Sequence[str]) -> dict[str, RunEvidence]:
    """Read each run's two text files into a ``RunEvidence``."""
    records = {}
    for case_id in case_ids:
        folder = run_dir(data_root, case_id)
        messages = _text(folder / MESSAGES_NAME)
        statistics = _text(folder / STATISTICS_NAME)
        if messages is None and statistics is None:
            _LOG.warning("%s: no run record found", case_id)
            continue
        records[case_id] = read_run_evidence(
            messages_text=messages,
            global_statistics_text=statistics,
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
    splits = get_benchmark("taylor_impact_2d").splits
    case_ids = args.case or sorted({cid for ids in splits.values() for cid in ids})
    records = collect(args.data_root, case_ids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump_run_evidence(records), encoding="utf-8", newline="\n")
    _LOG.info("%d of %d runs -> %s", len(records), len(case_ids), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
