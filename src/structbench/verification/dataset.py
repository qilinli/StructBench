"""Measuring a set of canonical cases against a declaration (ADR-0066).

What a benchmark's spec or a dataset's ``[declaration]`` says, applied to every
case file, with each case's solver input read by the reader for *its* solver.
This is the dataset-level entry the ``datacheck`` CLI and ``structbench-datagen
verify`` share; it lived in ``cli/datacheck.py`` until ADR-0071's part two
moved it here so that ``datagen`` need not import ``cli``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import get_args

from structbench import __version__
from structbench.core import (
    Absence,
    AbsenceReason,
    Case,
    DeclaredFacts,
    InputFacts,
    RunEvidence,
    read_case,
    read_input_facts,
)
from structbench.core.evidence import CardDiscretisation
from structbench.core.io.abaqus_run import read_abaqus_input_facts
from structbench.verification.measures import measure_case
from structbench.verification.quantities import CATALOGUE
from structbench.verification.results import (
    CaseMeasurements,
    DatasetMeasurements,
    Measurement,
)

logger = logging.getLogger(__name__)

#: ``"<mass>-<length>-<time>"``, the shape of a declared unit system (E10a).
UNIT_SYSTEM_PATTERN = re.compile(r"^[a-zA-Z]+(-[a-zA-Z]+){2}$")
#: ``BenchmarkCard.discretisation``'s vocabulary, which the traits row compares.
_DISCRETISATIONS = frozenset(get_args(CardDiscretisation))
#: Keys a ``[declaration]`` table may hold; ``unit_system`` is required.
_DECLARATION_KEYS = frozenset(
    {"unit_system", "fields", "discretisation", "erosion", "material_family"}
)


def declared_from_toml(path: Path) -> tuple[str, DeclaredFacts]:
    """``([dataset].name, DeclaredFacts from [declaration])`` of a ``dataset.toml``.

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
    if not isinstance(unit, str) or not UNIT_SYSTEM_PATTERN.fullmatch(unit):
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
