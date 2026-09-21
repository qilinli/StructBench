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


CLOSURE_MEASURES: dict[str, ClosureFn] = {
    "kinetic_energy_closure": _kinetic_energy_closure,
}
