"""Measuring one case: every catalogue quantity answers, none is judged (ADR-0066).

``measure_case`` is the only entry point. For each catalogue row it asks the
gate first (traits, then evidence); a row that passes is handed to its
measure function. A measure that raises becomes a recorded
``source_unreadable`` row — one bad array never aborts a dataset.
"""

from __future__ import annotations

from ...core import (
    Absence,
    AbsenceReason,
    Case,
    DeclaredFacts,
    EvidenceItem,
    InputFacts,
    RunEvidence,
)
from ..quantities import CATALOGUE, gate
from ..results import CaseMeasurements, Measurement, Status
from ..traits import run_traits
from . import constitutive, health, integrity, units
from ._common import MeasureFn
from .run import RUN_MEASURES

__all__ = ["MEASURES", "RUN_MEASURES", "measure_case"]

MEASURES: dict[str, MeasureFn] = {
    **health.MEASURES,
    **integrity.MEASURES,
    **constitutive.MEASURES,
    **units.MEASURES,
}

_IMPLEMENTED = {q.name for q in CATALOGUE if q.status is Status.IMPLEMENTED}
_BUILT = set(MEASURES) | set(RUN_MEASURES)
if _BUILT != _IMPLEMENTED:  # the catalogue and the code must not drift apart
    raise RuntimeError(
        f"measures and catalogue disagree: {sorted(_BUILT ^ _IMPLEMENTED)}"
    )


_RUN_ITEMS = frozenset(
    {EvidenceItem.E2, EvidenceItem.E3, EvidenceItem.E4, EvidenceItem.E5}
)


def _supplied(
    case: Case | None,
    facts: InputFacts | None,
    declared: DeclaredFacts | None,
    run: RunEvidence | None,
) -> frozenset[EvidenceItem]:
    items = set()
    if run is not None:
        if run.identity is not None:
            items.add(EvidenceItem.E2)
        if run.termination is not None and run.n_errors is not None:
            items.add(EvidenceItem.E3)
        if run.timestep is not None:
            items.add(EvidenceItem.E4)
        if run.ledger is not None:
            items.add(EvidenceItem.E5)
    if facts is not None:
        items.add(EvidenceItem.E1)
    if case is not None and case.response is not None:
        items.add(EvidenceItem.E8)
    if declared is not None and declared.unit_system:
        items.add(EvidenceItem.E10A)
    if declared is not None and declared.anchors:
        items.add(EvidenceItem.E10B)
    return frozenset(items)


def measure_case(
    case: Case | None,
    facts: InputFacts | None,
    declared: DeclaredFacts | None,
    *,
    case_id: str,
    file_sha256: str | None = None,
    run: RunEvidence | None = None,
) -> CaseMeasurements:
    """Measure every catalogue quantity on one run.

    Parameters
    ----------
    case : Case or None
        ``None`` for a run that left no canonical case (a failed or
        discarded run); input-only quantities are still measured.
    facts : InputFacts or None
        What the solver input establishes; ``None`` when none was supplied.
    declared : DeclaredFacts or None
        What the benchmark declares about the run.
    case_id : str
    file_sha256 : str or None
        Digest of the case file the measurements were taken from.
    run : RunEvidence or None
        What the solver's run record establishes (E2-E5).

    Returns
    -------
    CaseMeasurements
        One measurement per catalogue row, sorted by quantity name.
    """
    supplied = _supplied(case, facts, declared, run)
    rows: list[Measurement] = []
    for q in CATALOGUE:
        row = gate(q, facts, declared, supplied)
        if row is None:
            try:
                if q.name in RUN_MEASURES:
                    assert run is not None  # the gate saw its evidence
                    row = RUN_MEASURES[q.name](run, facts)
                else:
                    row = MEASURES[q.name](case, facts, declared)
            except Exception:  # noqa: BLE001 - recorded, never an abort
                row = Measurement(
                    q.name,
                    None,
                    q.unit,
                    absence=Absence(AbsenceReason.SOURCE_UNREADABLE, frozenset()),
                )
        elif (
            run is not None
            and run.unparsable
            and row.absence is not None
            and row.absence.reason is AbsenceReason.SOURCE_MISSING
            and row.absence.missing
            and row.absence.missing <= _RUN_ITEMS
        ):
            # the run record was supplied, but the reader refused part of it
            row = Measurement(
                q.name,
                None,
                q.unit,
                absence=Absence(AbsenceReason.UNPARSABLE, row.absence.missing),
            )
        rows.append(row)
    intent = (
        frozenset({"quasi_static"})
        if declared and declared.quasi_static
        else frozenset()
    )
    return CaseMeasurements(
        case_id=case_id,
        file_sha256=file_sha256,
        run_traits=run_traits(facts),
        declared_intent=intent,
        measurements=tuple(sorted(rows, key=lambda m: m.quantity)),
    )
