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
benchmark card would in the ``[declaration]`` table of its ``sweep.toml``,
and is measured with ``--declaration``: every ``*.h5`` under ``--data-root``
unless ``--case`` names some. No flag
alters a criterion. The exit code is 0 when the command completed — failed
checks are in the record, not in the exit code — and 2 on a usage or I/O
error.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import get_args

from .. import __version__
from ..benchmarks import BenchmarkSpec, get_benchmark
from ..benchmarks.card import Discretisation
from ..core import (
    Absence,
    AbsenceReason,
    Case,
    DeclaredFacts,
    InputFacts,
    RunEvidence,
    read_case,
    read_input_facts,
)
from ..core.io import load_run_evidence
from ..core.io.abaqus_run import read_abaqus_input_facts
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


#: ``BenchmarkCard.discretisation``'s vocabulary, which the traits row compares.
_DISCRETISATIONS = frozenset(get_args(Discretisation))
#: Keys a ``[declaration]`` table may hold; ``unit_system`` is required.
_DECLARATION_KEYS = frozenset(
    {"unit_system", "fields", "discretisation", "erosion", "material_family"}
)


def declared_from_toml(path: Path) -> tuple[str, DeclaredFacts]:
    """``([dataset].name, DeclaredFacts from [declaration])`` of a ``sweep.toml``.

    No yield table is declared: a sweep that varies the hardening has none,
    and the yield rows then read each case's own input table.

    Raises
    ------
    ValueError
        On a missing ``[declaration]`` or ``unit_system``, or an unknown key.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    block = raw.get("declaration")
    if not isinstance(block, dict):
        raise ValueError(f"{path.name}: no [declaration] table")
    unknown = sorted(set(block) - _DECLARATION_KEYS)
    if unknown:
        raise ValueError(f"{path.name}: unknown [declaration] keys {unknown}")
    unit = block.get("unit_system")
    if not isinstance(unit, str) or not _UNIT_SYSTEM.fullmatch(unit):
        raise ValueError(
            f"{path.name}: [declaration] needs unit_system 'mass-length-time'"
        )
    discretisation = block.get("discretisation")
    if discretisation is not None and discretisation not in _DISCRETISATIONS:
        raise ValueError(
            f"{path.name}: discretisation {discretisation!r} is not one of "
            f"{sorted(_DISCRETISATIONS)} (a benchmark card's words)"
        )
    fields = block.get("fields")
    name = str(raw.get("dataset", {}).get("name", path.parent.name))
    return name, DeclaredFacts(
        unit_system=unit,
        fields=None if fields is None else frozenset(str(f) for f in fields),
        discretisation=discretisation,
        erosion=block.get("erosion"),
        material_family=block.get("material_family"),
    )


#: Solver input readers by normalised solver name (``Provenance.solver_name``).
#: A deck whose solver is absent or unlisted is NOT parsed: guessing is how a
#: foreign deck gets reported as a defective one (ADR-0068).
_INPUT_READERS = {"lsdyna": read_input_facts, "abaqus": read_abaqus_input_facts}


def _normalise_solver(name: str | None) -> str:
    """``"LS-DYNA"``, ``"ls dyna"``, ``"LSDYNA"`` -> ``"lsdyna"``."""
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def input_facts_for(case: Case) -> tuple[InputFacts | None, AbsenceReason | None]:
    """Read the stored solver input with the reader for ITS solver.

    Returns ``(facts, reason)``. ``reason`` is ``None`` when the input was
    read, or when the case stores no input at all -- a case that carries no
    deck is missing evidence the dataset should have supplied, which is the
    dataset's gap and already reported as such. ``reason`` is
    ``UNSUPPORTED`` -- a platform reason -- when a deck IS stored and this
    instrument has no reader for the solver that wrote it. That is the
    platform's gap and must never be counted against the contributor.

    Fails closed: an unattributable deck is not parsed. ``Provenance`` is
    optional and ``solver_name`` is free text, so the alternative is handing
    an unknown input to whichever parser happens to be first.
    """
    deck, units = case.metadata.source_deck, case.metadata.source_units
    if not deck or not units:
        return None, None
    provenance = case.metadata.provenance
    reader = _INPUT_READERS.get(
        _normalise_solver(provenance.solver_name if provenance else None)
    )
    if reader is None:
        return None, AbsenceReason.UNSUPPORTED
    return reader(deck, source_units=units), None


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


def measure_cases(
    declared: DeclaredFacts,
    name: str | None,
    data_root: Path,
    case_ids: Sequence[str] | None = None,
    *,
    dataset_revision: str | None = None,
    run_evidence: Mapping[str, RunEvidence] | None = None,
) -> DatasetMeasurements:
    """Measure ``<data_root>/<case_id>.h5`` for every case (ADR-0066).

    The solver input is the text stored with each case
    (``metadata.source_deck``); a case without one is measured with no input
    facts. A file that is missing or cannot be read becomes a recorded case
    whose every row is ``source_unreadable`` — never an abort.

    Parameters
    ----------
    declared : DeclaredFacts
        What the benchmark or sweep declares about its runs.
    name : str or None
        The registry name, or ``None`` for an unregistered sweep.
    data_root : Path
        Directory of canonical case files.
    case_ids : sequence of str, optional
        Defaults to every ``*.h5`` under ``data_root`` (a ``.partial.h5`` is
        a write that never finished, and is left out).
    dataset_revision : str, optional
        The dataset tag the files came from.
    run_evidence : mapping of str to RunEvidence, optional
        The run record of each case, by case id (E2-E5).
    """
    if case_ids is None:
        case_ids = [
            p.stem for p in data_root.glob("*.h5") if not p.stem.endswith(".partial")
        ]
    cases = []
    runs = run_evidence or {}
    for case_id in sorted(set(case_ids)):
        path = data_root / f"{case_id}.h5"
        digest = None
        try:
            digest = _sha256(path)
            case = read_case(path)
            facts, input_reason = input_facts_for(case)
            cases.append(
                measure_case(
                    case,
                    facts,
                    declared,
                    case_id=case_id,
                    file_sha256=digest,
                    run=runs.get(case_id),
                    input_reason=input_reason,
                )
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
        name, dataset_revision, __version__, versions, tuple(cases)
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
    source.add_argument("--declaration", type=Path, help="a sweep.toml")
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
