"""Reference-data verification from the command line (ADR-0066).

Two verbs, kept apart::

    python -m structbench.cli.datacheck measure
        (--benchmark NAME | --declaration SWEEP.toml) --data-root DIR
        [--case ID ...] [--run-evidence FILE.json] --out FILE.json
    python -m structbench.cli.datacheck judge
        --measurements FILE.json [--report FILE.md]

``measure`` reads canonical case files — and, when given, the whitelisted
run-evidence record a dataset's glue wrote, never a run directory — and
writes the measurements record; ``judge`` reads only that record. A sweep
not yet in the registry (a private dataset before publication) states what a
benchmark card would in the ``[declaration]`` table of its ``dataset.toml``,
and is measured with ``--declaration``: every ``*.h5`` under ``--data-root``
unless ``--case`` names some. No flag
alters a criterion. The exit code is 0 when the command completed — failed
checks are in the record, not in the exit code — and 2 on a usage or I/O
error.
"""

from __future__ import annotations

import argparse
import logging
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..benchmarks import BenchmarkSpec, get_benchmark
from ..core import (
    DeclaredFacts,
    RunEvidence,
)
from ..core.io import load_run_evidence
from ..verification.criteria import judge
from ..verification.dataset import _INPUT_READERS as _INPUT_READERS
from ..verification.dataset import UNIT_SYSTEM_PATTERN as _UNIT_SYSTEM
from ..verification.dataset import declared_from_toml as declared_from_toml
from ..verification.dataset import input_facts_for as input_facts_for
from ..verification.dataset import measure_cases as measure_cases
from ..verification.report import from_json, render_markdown, to_json
from ..verification.results import DatasetMeasurements

logger = logging.getLogger(__name__)

_MPA = 1.0e6  # BenchmarkSpec.hardening_curve is in MPa (ADR-0064); criteria are SI


def declared_from_spec(spec: BenchmarkSpec) -> DeclaredFacts:
    """What a registered benchmark declares about its runs (ADR-0066).

    Units anchors come from ``card.units_anchors``; a benchmark that
    declares none leaves ``units_anchors_consistent`` without its E10b.
    """
    card = spec.card
    label = card.source_units.split()[0] if card.source_units.strip() else ""
    table = None
    if spec.hardening_curve is not None:
        knots, sigma_y = spec.hardening_curve
        table = (tuple(knots), tuple(s * _MPA for s in sigma_y))
    return DeclaredFacts(
        anchors=card.units_anchors,
        unit_system=label if _UNIT_SYSTEM.fullmatch(label) else None,
        fields=frozenset(card.fields),
        discretisation=card.discretisation,
        erosion=card.erosion,
        yield_table=table,
    )


def measure_dataset(
    spec: BenchmarkSpec,
    data_root: Path,
    case_ids: Sequence[str] | None = None,
    *,
    dataset_revision: str | None = None,
    run_evidence: Mapping[str, RunEvidence] | None = None,
) -> DatasetMeasurements:
    """Measure a registered benchmark; ``case_ids`` defaults to its splits."""
    if case_ids is None:
        case_ids = sorted({cid for ids in spec.splits.values() for cid in ids})
    return measure_cases(
        declared_from_spec(spec),
        spec.card.name,
        data_root,
        case_ids,
        dataset_revision=dataset_revision,
        run_evidence=run_evidence,
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _measure(args: argparse.Namespace) -> int:
    if not args.data_root.is_dir():
        print(f"error: data root is not a directory: {args.data_root}")
        return 2
    spec = None
    try:
        if args.benchmark is not None:
            spec = get_benchmark(args.benchmark)
        else:
            _, declared = declared_from_toml(args.declaration)
    except (KeyError, OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"error: {error}")
        return 2
    runs = None
    if args.run_evidence is not None:
        try:
            runs = load_run_evidence(args.run_evidence.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"error: cannot read run evidence: {error}")
            return 2
    if spec is not None:
        record = measure_dataset(
            spec,
            args.data_root,
            args.case or None,
            dataset_revision=args.dataset_revision,
            run_evidence=runs,
        )
    else:
        record = measure_cases(
            declared,
            None,  # not a registry name: the sweep is unregistered
            args.data_root,
            args.case or None,
            dataset_revision=args.dataset_revision,
            run_evidence=runs,
        )
    _write(args.out, to_json(record))
    print(f"measured {len(record.cases)} cases -> {args.out}")
    return 0


def _judge(args: argparse.Namespace) -> int:
    try:
        record = from_json(args.measurements.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as error:
        print(f"error: cannot read measurements: {error}")
        return 2
    report = judge(record)
    count = Counter(r.verdict for case in report.cases for r in case.results)
    print(", ".join(f"{verdict}: {n}" for verdict, n in sorted(count.items())))
    text = render_markdown(report)
    if args.report is None:
        print(text, end="")
    else:
        _write(args.report, text)
        print(f"report -> {args.report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command; 0 on completion, 2 on a usage or I/O error."""
    parser = argparse.ArgumentParser(
        prog="python -m structbench.cli.datacheck", description=__doc__.split("\n")[0]
    )
    verbs = parser.add_subparsers(dest="verb", required=True)
    measure = verbs.add_parser("measure", help="measure canonical case files")
    source = measure.add_mutually_exclusive_group(required=True)
    source.add_argument("--benchmark")
    source.add_argument("--declaration", type=Path, help="a dataset.toml")
    measure.add_argument("--data-root", required=True, type=Path)
    measure.add_argument("--case", action="append", help="case id; repeatable")
    measure.add_argument("--dataset-revision", default=None)
    measure.add_argument("--run-evidence", default=None, type=Path)
    measure.add_argument("--out", required=True, type=Path)
    measure.set_defaults(run=_measure)
    judged = verbs.add_parser("judge", help="judge a measurements record")
    judged.add_argument("--measurements", required=True, type=Path)
    judged.add_argument("--report", default=None, type=Path)
    judged.set_defaults(run=_judge)
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:
        return int(stop.code or 0)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
