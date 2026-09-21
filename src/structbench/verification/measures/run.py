"""Measures on the run-evidence record: identity, termination, time step, energy.

These need no ``Case``, so a run that was discarded before it became one can
still be measured (ADR-0066 clause 3). The energy rows have **one**
definition: every ledger term the run's traits make applicable must be
present, otherwise the row is absent — there is no partial-ledger check
(clause 2).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...core import AbsenceReason, EvidenceItem, InputFacts, RunEvidence
from ..kernels import energy_residual
from ..results import Location, Measurement
from ._common import absent, input_gap, not_applicable, value

RunMeasureFn = Callable[[RunEvidence, InputFacts | None], Measurement]

RUN, INPUT = Location.RUN, Location.INPUT
_MESHED = frozenset({"solid", "shell", "beam"})


def _terminated_normally(run: RunEvidence, facts: InputFacts | None) -> Measurement:
    assert run.termination is not None
    normal = all(segment.status == "normal" for segment in run.termination)
    last = run.termination[-1]
    detail: dict[str, float | int | str] = {"segments": len(run.termination)}
    if last.n_steps is not None:
        detail["steps"] = last.n_steps
    if last.criterion is not None:
        detail["ended_by"] = last.criterion
    return value("terminated_normally", float(normal), RUN, detail=detail)


def _solver_error_count(run: RunEvidence, facts: InputFacts | None) -> Measurement:
    assert run.n_errors is not None
    return value("solver_error_count", run.n_errors, RUN)


def _solver_warning_count(run: RunEvidence, facts: InputFacts | None) -> Measurement:
    assert run.n_warnings is not None
    return value("solver_warning_count", run.n_warnings, RUN)


def _solver_identity_complete(
    run: RunEvidence, facts: InputFacts | None
) -> Measurement:
    """Identity items the record leaves out. Output-stream precision is part
    of E2 but no run has supplied it yet, so it is not counted."""
    assert run.identity is not None
    items = ("version", "revision", "precision", "parallel_layout")
    missing = sum(getattr(run.identity, item) is None for item in items)
    return value("solver_identity_complete", missing, RUN, n=len(items))


def _timestep_min_ratio(run: RunEvidence, facts: InputFacts | None) -> Measurement:
    assert run.timestep is not None
    times, steps = (np.asarray(a, dtype=np.float64) for a in run.timestep)
    if steps.size == 0 or steps[0] <= 0.0:
        return absent("timestep_min_ratio", AbsenceReason.UNPARSABLE, EvidenceItem.E4)
    at = int(np.argmin(steps))
    return value(
        "timestep_min_ratio",
        float(steps[at] / steps[0]),
        RUN,
        n=steps.size,
        detail={"sample": at, "time": float(times[at])},
    )


def _required_terms(facts: InputFacts) -> set[str]:
    """Ledger terms this run's traits make applicable (ADR-0066 clause 2)."""
    required = {"kinetic", "internal", "external_work"}
    meshed = [p for p in facts.parts if p.discretisation in _MESHED]
    if any(p.under_integrated is not False for p in meshed):
        required.add("zero_energy_mode")  # unknown integration counts: fail closed
    if facts.contact_defined is not False:
        required.add("contact")
    if facts.rigid_planes:
        required.add("rigid_surface")
    if facts.erosion_enabled is not False:
        required |= {"eroded_kinetic", "eroded_internal"}
    return required


def _residual(
    name: str, run: RunEvidence, facts: InputFacts | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | Measurement:
    """``(time, E_tot, r)`` of the energy indicator, or the absence that blocks it."""
    assert run.ledger is not None
    if facts is None:
        return input_gap(name, facts)  # applicability of the terms is unknown
    ledger = run.ledger
    if _required_terms(facts) - set(ledger.terms):
        return absent(name, AbsenceReason.SOURCE_MISSING, EvidenceItem.E5)
    series = {k: np.asarray(v, dtype=np.float64) for k, v in ledger.terms.items()}
    total = sum(sign * series[k] for k, sign in ledger.identity.items())
    assert isinstance(total, np.ndarray)
    r = energy_residual(total, series["external_work"], series["kinetic"])
    return np.asarray(ledger.time), total, r


def _extreme(name: str, sign: float) -> RunMeasureFn:
    def measure(run: RunEvidence, facts: InputFacts | None) -> Measurement:
        got = _residual(name, run, facts)
        if isinstance(got, Measurement):
            return got
        time, _, r = got
        at = int(np.argmax(sign * r))
        return value(
            name,
            max(0.0, float(sign * r[at])),
            RUN,
            INPUT,
            n=r.size,
            detail={"sample": at, "time": float(time[at])},
        )

    return measure


def _energy_residual_final(run: RunEvidence, facts: InputFacts | None) -> Measurement:
    got = _residual("energy_residual_final", run, facts)
    if isinstance(got, Measurement):
        return got
    return value("energy_residual_final", float(got[2][-1]), RUN, INPUT, n=got[2].size)


def _total_energy_change_final(
    run: RunEvidence, facts: InputFacts | None
) -> Measurement:
    name = "total_energy_change_final"
    got = _residual(name, run, facts)
    if isinstance(got, Measurement):
        return got
    total = got[1]
    if total[0] == 0.0:
        return not_applicable(name)  # no energy at the first sample to compare with
    return value(name, float(total[-1] / total[0] - 1.0), RUN, INPUT, n=total.size)


RUN_MEASURES: dict[str, RunMeasureFn] = {
    "terminated_normally": _terminated_normally,
    "solver_error_count": _solver_error_count,
    "solver_warning_count": _solver_warning_count,
    "solver_identity_complete": _solver_identity_complete,
    "timestep_min_ratio": _timestep_min_ratio,
    "energy_gain_max": _extreme("energy_gain_max", 1.0),
    "energy_loss_max": _extreme("energy_loss_max", -1.0),
    "energy_residual_final": _energy_residual_final,
    "total_energy_change_final": _total_energy_change_final,
}
