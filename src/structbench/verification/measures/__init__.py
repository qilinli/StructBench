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
from ._common import PARTICLES, MeasureFn, absent
from .closure import CLOSURE_MEASURES, shared_samples
from .run import RUN_MEASURES

__all__ = ["CLOSURE_MEASURES", "MEASURES", "RUN_MEASURES", "measure_case"]

#: Measures that read the SPH block directly. On a case without one the
#: access is a ``KeyError``, which ``measure_case`` would stamp
#: ``source_unreadable`` -- a contributor-owned reason -- for a case that is
#: perfectly readable. Eleven of these sixteen carry no particle trait gate, so
#: the guard belongs here rather than on each row; a census that said nine of
#: fifteen is what let `rigid_surface_penetration_max` slip out of the set,
#: since its gate is `needs_rigid_plane`. The quantity exists for a
#: meshed element too; what is missing is a measure that reads one, which is
#: the platform's gap (``unsupported``), never the dataset's.
_PARTICLE_ONLY: frozenset[str] = frozenset(
    {
        "active_mass_drift",
        "density_slot_matches_input",
        "out_of_plane_shear_max",
        "particle_deactivated_count",
        "particle_neighbors_growth",
        "particle_neighbors_min",
        "plane_strain_ezz_max",
        "pressure_trace_residual",
        "rigid_surface_penetration_max",
        "smoothing_length_at_bound_fraction",
        "smoothing_length_within_input_bounds",
        "state_variable_decrease_max",
        "state_variable_min",
        "yield_ratio_max",
        "yield_saturation_min",
        "yield_table_covers_range",
    }
)


def _needs_particles(name: str, fn: MeasureFn) -> MeasureFn:
    """Report the platform's gap rather than crashing on a mesh-only case."""

    def wrapped(
        case: Case | None, facts: InputFacts | None, declared: DeclaredFacts | None
    ) -> Measurement:
        if case is not None and PARTICLES not in case.elements:
            return absent(name, AbsenceReason.UNSUPPORTED)
        return fn(case, facts, declared)

    return wrapped


_RAW_MEASURES: dict[str, MeasureFn] = {
    **health.MEASURES,
    **integrity.MEASURES,
    **constitutive.MEASURES,
    **units.MEASURES,
}

MEASURES: dict[str, MeasureFn] = {
    name: _needs_particles(name, fn) if name in _PARTICLE_ONLY else fn
    for name, fn in _RAW_MEASURES.items()
}

_IMPLEMENTED = {q.name for q in CATALOGUE if q.status is Status.IMPLEMENTED}
_BUILT = set(MEASURES) | set(RUN_MEASURES) | set(CLOSURE_MEASURES)
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
        if shared_samples(case, run)[0].size >= 2:
            items.add(EvidenceItem.E9)  # found in the files, not declared
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
    input_reason: AbsenceReason | None = None,
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
    input_reason : AbsenceReason or None
        Why ``facts`` is ``None`` when a solver input WAS stored -- the
        platform has no reader for the solver that wrote it (ADR-0068).
        ``None`` means the case simply stored no input, which is the
        dataset's gap and reported as such.
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
        row = gate(q, facts, declared, supplied, input_reason)
        if row is None:
            try:
                if q.name in CLOSURE_MEASURES:
                    assert case is not None and run is not None
                    row = CLOSURE_MEASURES[q.name](case, facts, run)
                elif q.name in RUN_MEASURES:
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
