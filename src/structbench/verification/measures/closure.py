"""Closures: a ledger term against the same quantity summed from the fields.

A closure compares two independent records of one run at the instants both
were sampled, so it sees what neither sees alone — most cheaply, particles
and nodes that are misaligned in the case file.

Shared instants are *found*, not declared: a stored frame and a ledger
sample coincide when they lie within ``_ALIGN`` of a stored interval of each
other (maintainer decision 2026-09-21: alignment visible in the files
suffices for this check). A frame written off the sampling interval, such as
the state a solver writes at the termination time, has no partner and is
left out.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ...core import AbsenceReason, Case, InputFacts, RunEvidence
from ...datasets import n_valid_frames
from ..results import Location, Measurement
from ._common import (
    PARTICLES,
    absent,
    field_gap,
    known_particles,
    not_applicable,
    particle_field,
    value,
)

ClosureFn = Callable[[Case, InputFacts | None, RunEvidence], Measurement]

#: Two samples coincide within this fraction of the median stored interval.
#: Ledger times are printed to six digits, about 5e-4 of an interval late in
#: a run; 1e-2 leaves a decade and a half, and a field changes negligibly
#: over a hundredth of an output interval.
_ALIGN = 1.0e-2
_MESHED = frozenset({"solid", "shell", "beam"})

CASE, RUN = Location.CASE, Location.RUN


def shared_samples(
    case: Case | None, run: RunEvidence | None
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Indices ``(frames, ledger_samples)`` of the instants both records hold."""
    none = np.empty(0, dtype=np.intp)
    if case is None or case.response is None or run is None or run.ledger is None:
        return none, none
    frames = np.asarray(case.response.time, dtype=np.float64)
    ledger = np.asarray(run.ledger.time, dtype=np.float64)
    if frames.size < 2 or ledger.size == 0:
        return none, none
    nearest = np.abs(frames[:, None] - ledger[None, :]).argmin(axis=1)
    close = np.abs(frames - ledger[nearest]) <= _ALIGN * np.median(np.diff(frames))
    return np.flatnonzero(close), nearest[close]


def _kinetic_energy_closure(
    case: Case, facts: InputFacts | None, run: RunEvidence
) -> Measurement:
    """Largest ``|KE_fields - KE_ledger|`` over the run's peak ledger value.

    Normalised by the peak so that a run coming to rest does not divide by
    nothing. Particle parts only: a meshed part's nodal masses are not
    stored with the case.
    """
    name = "kinetic_energy_closure"
    assert run.ledger is not None and case.response is not None
    meshed = facts is not None and any(p.discretisation in _MESHED for p in facts.parts)
    if meshed or PARTICLES not in case.elements:
        return absent(name, AbsenceReason.UNSUPPORTED)
    keep = known_particles(case, facts)
    mass = particle_field(case, "mass", keep)
    velocity = case.response.node.get("velocity")
    if mass is None or velocity is None:
        return field_gap(name)
    frames, samples = shared_samples(case, run)
    rows = case.elements[PARTICLES].connectivity[:, 0][keep]  # row indexes, not ids
    speed2 = (velocity[frames][:, rows].astype(np.float64) ** 2).sum(axis=-1)
    from_fields = 0.5 * (mass[frames].astype(np.float64) * speed2).sum(axis=1)
    ledger = np.asarray(run.ledger.terms["kinetic"], dtype=np.float64)
    peak = float(ledger.max())
    if peak <= 0.0:
        return not_applicable(name)  # nothing ever moved
    gap = np.abs(from_fields - ledger[samples]) / peak
    at = int(np.argmax(gap))
    return value(
        name,
        float(gap[at]),
        CASE,
        RUN,
        n=frames.size,
        detail={
            "frame": int(frames[at]),
            "first_sample": float(gap[0]),
            "frames_without_partner": int(case.response.time.shape[0] - frames.size),
        },
    )


#: Stored global channel -> the ledger term it must reproduce. Whatever the
#: adapter took off the solver's state output answers to what the solver
#: itself printed; a channel with no partner here cannot be compared and is
#: not a defect.
_GLOBAL_TERMS = {"kinetic_energy": "kinetic", "internal_energy": "internal"}


def _ledger_series(run: RunEvidence, channel: str) -> tuple[float, ...] | None:
    """The ledger series a stored channel answers to, if the ledger has one."""
    assert run.ledger is not None
    if channel == "total_energy":
        return run.ledger.solver_total
    term = _GLOBAL_TERMS.get(channel)
    return None if term is None else run.ledger.terms.get(term)


def _stored_globals_match_ledger(
    case: Case, facts: InputFacts | None, run: RunEvidence
) -> Measurement:
    """Largest gap between a stored global channel and the ledger's own series.

    Each channel is normalised by the peak of the series it answers to, so
    the number reads as a fraction of the energy that channel ever carried,
    and the worst channel over the run is the one reported. This is the one
    check that sees an *ingestion* error -- a channel dropped, misnamed or
    left in the solver's units -- because it is the only place the stored
    arrays meet the solver's own account of the same run.
    """
    name = "stored_globals_match_ledger"
    assert run.ledger is not None and case.response is not None
    stored = case.response.globals_
    if not stored:
        return field_gap(name)
    frames, samples = shared_samples(case, run)
    worst = -1.0
    detail: dict[str, float | int | str] = {}
    for channel, series in sorted(stored.items()):
        reference = _ledger_series(run, channel)
        if reference is None:
            continue
        ref = np.asarray(reference, dtype=np.float64)
        scale = float(np.abs(ref).max())
        if scale <= 0.0:
            continue  # a series that is zero throughout sets no scale
        gap = np.abs(np.asarray(series, np.float64)[frames] - ref[samples]) / scale
        at = int(np.argmax(gap))
        if float(gap[at]) > worst:
            worst = float(gap[at])
            detail = {"channel": channel, "frame": int(frames[at])}
    if worst < 0.0:
        return absent(name, AbsenceReason.UNSUPPORTED)  # no channel has a partner
    return value(name, worst, CASE, RUN, n=frames.size, detail=detail)


def _sampling_clock_consistent(
    case: Case, facts: InputFacts | None, run: RunEvidence
) -> Measurement:
    """Frames on the output grid with no ledger sample at the same instant.

    "The same instant" is ``shared_samples``'s alignment, the one every
    closure uses: a ledger printed to six digits sits within 5e-4 of an
    interval of its frame, while a missing sample is a whole interval away.
    A terminal frame off the grid (ADR-0028) is ``terminal_artifact_frames``'s
    to report; E9 asks only that the ledger divide the output interval.
    """
    name = "sampling_clock_consistent"
    assert case.response is not None
    on_grid = n_valid_frames(case.response.time)
    frames, _ = shared_samples(case, run)
    missing = on_grid - int((frames < on_grid).sum())
    return value(name, missing, CASE, RUN, n=on_grid)


CLOSURE_MEASURES: dict[str, ClosureFn] = {
    "sampling_clock_consistent": _sampling_clock_consistent,
    "kinetic_energy_closure": _kinetic_energy_closure,
    "stored_globals_match_ledger": _stored_globals_match_ledger,
}
