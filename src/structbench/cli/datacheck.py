"""Reference-data verification from the command line (ADR-0066).

Two verbs, kept apart::

    python -m structbench.cli.datacheck measure --benchmark NAME \
        --data-root DIR [--case ID ...] --out FILE.json
    python -m structbench.cli.datacheck judge --measurements FILE.json \
        [--report FILE.md]

``measure`` reads canonical case files and writes the measurements record;
``judge`` reads only that record. No flag alters a criterion. The exit code is
0 when the command completed — failed checks are in the record, not in the
exit code — and 2 on a usage or I/O error.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .. import __version__
from ..benchmarks import BenchmarkSpec, get_benchmark
from ..core import (
    Absence,
    AbsenceReason,
    DeclaredFacts,
    read_case,
    read_input_facts,
)
from ..verification.criteria import judge
from ..verification.measures import measure_case
from ..verification.quantities import CATALOGUE
from ..verification.report import from_json, render_markdown, to_json
from ..verification.results import CaseMeasurements, DatasetMeasurements, Measurement

logger = logging.getLogger(__name__)

_UNIT_SYSTEM = re.compile(r"^[a-zA-Z]+(-[a-zA-Z]+){2}$")
_MPA = 1.0e6  # BenchmarkSpec.hardening_curve is in MPa (ADR-0064); criteria are SI


def declared_from_spec(spec: BenchmarkSpec) -> DeclaredFacts:
    """What a registered benchmark declares about its runs (ADR-0066).

    The card has no home for units anchors yet, so none are declared and
    ``units_anchors_consistent`` reads ``no_declaration_home``.
    """
    card = spec.card
    label = card.source_units.split()[0] if card.source_units.strip() else ""
    table = None
    if spec.hardening_curve is not None:
        knots, sigma_y = spec.hardening_curve
        table = (tuple(knots), tuple(s * _MPA for s in sigma_y))
    return DeclaredFacts(
        unit_system=label if _UNIT_SYSTEM.fullmatch(label) else None,
        fields=frozenset(card.fields),
        discretisation=card.discretisation,
        erosion=card.erosion,
        yield_table=table,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unreadable(case_id: str, file_sha256: str | None) -> CaseMeasurements:
    rows = tuple(
        Measurement(
            q.name,
            None,
            q.unit,
            absence=Absence(AbsenceReason.SOURCE_UNREADABLE, frozenset()),
        )
        for q in CATALOGUE
    )
    return CaseMeasurements(case_id, file_sha256, frozenset(), frozenset(), rows)


def measure_dataset(
    spec: BenchmarkSpec,
    data_root: Path,
    case_ids: Sequence[str] | None = None,
    *,
    dataset_revision: str | None = None,
) -> DatasetMeasurements:
    """Measure ``<data_root>/<case_id>.h5`` for every case (ADR-0066).

    The solver input is the text stored with each case
    (``metadata.source_deck``); a case without one is measured with no input
    facts. A file that is missing or cannot be read becomes a recorded case
    whose every row is ``source_unreadable`` — never an abort.

    Parameters
    ----------
    spec : BenchmarkSpec
    data_root : Path
        Directory of canonical case files.
    case_ids : sequence of str, optional
        Defaults to every case in the benchmark's splits.
    dataset_revision : str, optional
        The dataset tag the files came from.
    """
    if case_ids is None:
        case_ids = sorted({cid for ids in spec.splits.values() for cid in ids})
    declared = declared_from_spec(spec)
    cases = []
    for case_id in sorted(set(case_ids)):
        path = data_root / f"{case_id}.h5"
        digest = None
        try:
            digest = _sha256(path)
            case = read_case(path)
            deck, units = case.metadata.source_deck, case.metadata.source_units
            facts = (
                read_input_facts(deck, source_units=units) if deck and units else None
            )
            cases.append(
                measure_case(case, facts, declared, case_id=case_id, file_sha256=digest)
            )
        except Exception:  # noqa: BLE001 - recorded, never an abort
            logger.warning("case %s could not be read", case_id)
            cases.append(_unreadable(case_id, digest))
    versions = {
        q.name: q.definition_version
        for q in CATALOGUE
        if q.definition_version is not None
    }
    return DatasetMeasurements(
        spec.card.name, dataset_revision, __version__, versions, tuple(cases)
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _measure(args: argparse.Namespace) -> int:
    if not args.data_root.is_dir():
        print(f"error: data root is not a directory: {args.data_root}")
        return 2
    try:
        spec = get_benchmark(args.benchmark)
    except KeyError as error:
        print(f"error: {error}")
        return 2
    record = measure_dataset(
        spec, args.data_root, args.case or None, dataset_revision=args.dataset_revision
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
    measure.add_argument("--benchmark", required=True)
    measure.add_argument("--data-root", required=True, type=Path)
    measure.add_argument("--case", action="append", help="case id; repeatable")
    measure.add_argument("--dataset-revision", default=None)
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
