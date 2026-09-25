"""Constitutive measures, dispatched on material class (ADR-0066 clause 6).

The operative yield table is the *declared* one when the benchmark declares
one (ADR-0064's single hardening curve, in Pa), and the input's own table is
then read only to check the two agree. A sweep that varies the hardening
declares none, so each case is judged against its own input table (plan 2,
Decision 4). The state rows read every block in ``STATE_BLOCKS``: particles
and meshed continuum elements alike (Decision 5). Every solver state is a
genuine state, so these use all stored frames.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...core import Case, DeclaredFacts, InputFacts
from ...datasets import von_mises_from_voigt
from ..kernels import (
    abs_max,
    decrease_max,
    interp_yield_stress,
    nearest_knot_distance,
    pressure_trace_residual,
    yield_ratio_max,
    yielding_mask,
)
from ..materials import MaterialClass
from ..results import Location, Measurement
from ._common import (
    STATE_BLOCKS,
    MeasureFn,
    class_mask,
    element_field,
    field_gap,
    input_gap,
    known_particles,
    needs_case,
    not_applicable,
    particle_field,
    value,
)

CASE, INPUT, DECLARED = Location.CASE, Location.INPUT, Location.DECLARED
_STATE = "effective_plastic_strain"  # the schema slot; its meaning is the class's


def _stacked(
    case: Case, facts: InputFacts, wanted: Callable[[MaterialClass], bool], *names: str
) -> tuple[np.ndarray, ...] | None:
    """``names`` over every state block's wanted elements, joined on the element axis.

    ``None`` when no block holds a wanted element, or one that does lacks a
    field -- a partial answer would judge some elements and not say so.
    """
    found: list[list[np.ndarray]] = []
    for block in STATE_BLOCKS:
        if block not in case.elements:
            continue
        keep = class_mask(case, facts, wanted, block)
        if not keep.any():
            continue
        fields = [element_field(case, block, name, keep) for name in names]
        if any(f is None for f in fields):
            return None
        found.append(fields)  # type: ignore[arg-type]
    if not found:
        return None
    return tuple(
        np.concatenate([f[i] for f in found], axis=1) for i in range(len(names))
    )


def _yield_inputs(
    case: Case, facts: InputFacts
) -> tuple[np.ndarray, np.ndarray] | None:
    got = _stacked(
        case, facts, lambda c: c.yield_law == "tabulated_j2", "stress", _STATE
    )
    if got is None:
        return None
    stress, state = got
    return von_mises_from_voigt(stress), state.astype(np.float64)


def _yield_table(
    declared: DeclaredFacts | None, facts: InputFacts
) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], Location] | None:
    """The operative table and where it was read, or ``None`` (Decision 4)."""
    if declared is not None and declared.yield_table:
        return declared.yield_table, DECLARED
    table = _input_table(facts)
    return None if table is None else (table, INPUT)


def _yield_ratio_max(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "yield_ratio_max"
    assert facts is not None
    source = _yield_table(declared, facts)
    if source is None:
        return input_gap(name, facts)
    (knots, sy), where = source
    got = _yield_inputs(case, facts)
    if got is None:
        return field_gap(name)
    vm, state = got
    extreme = yield_ratio_max(vm, state, knots, sy)
    ratio = vm / interp_yield_stress(state, knots, sy)
    at = float(state[extreme.frame, extreme.index])
    return value(
        name,
        extreme.value,
        CASE,
        where,
        n=extreme.n_samples,
        detail={
            "frame": extreme.frame,
            "index": extreme.index,
            "state_at_max": at,
            "nearest_knot_distance": nearest_knot_distance(at, knots),
            "fraction_above_one": float((ratio > 1.0).mean()),
        },
    )


def _yield_saturation_min(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "yield_saturation_min"
    assert facts is not None
    source = _yield_table(declared, facts)
    if source is None:
        return input_gap(name, facts)
    (knots, sy), where = source
    got = _yield_inputs(case, facts)
    if got is None:
        return field_gap(name)
    vm, state = got
    loading = yielding_mask(state)
    if not loading.any():
        return not_applicable(name)  # nothing is yielding across two stored intervals
    ratio = vm / interp_yield_stress(state, knots, sy)
    return value(name, float(ratio[loading].max()), CASE, where, n=int(loading.sum()))


def _state(case: Case, facts: InputFacts) -> np.ndarray | None:
    got = _stacked(case, facts, lambda c: c.monotone, _STATE)
    return None if got is None else got[0].astype(np.float64)


def _state_variable_decrease_max(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "state_variable_decrease_max"
    assert facts is not None
    state = _state(case, facts)
    if state is None:
        return field_gap(name)
    largest, count = decrease_max(state)
    transitions = (state.shape[0] - 1) * state.shape[1]
    return value(name, largest, CASE, n=transitions, detail={"decreasing_steps": count})


def _state_variable_min(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "state_variable_min"
    assert facts is not None
    state = _state(case, facts)
    if state is None:
        return field_gap(name)
    return value(name, float(state.min()), CASE, n=state.size)


def _plane_strain_ezz_max(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "plane_strain_ezz_max"
    strain = particle_field(case, "strain", known_particles(case, facts))
    if strain is None:
        return field_gap(name)
    return value(name, abs_max(strain[..., 2]), CASE, n=strain[..., 2].size)


def _out_of_plane_shear_max(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "out_of_plane_shear_max"
    stress = particle_field(case, "stress", known_particles(case, facts))
    if stress is None:
        return field_gap(name)
    return value(name, abs_max(stress[..., 4:6]), CASE, n=stress[..., 4:6].size)


def _pressure_trace_residual(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "pressure_trace_residual"
    keep = known_particles(case, facts)
    stress = particle_field(case, "stress", keep)
    pressure = particle_field(case, "pressure", keep)
    if stress is None or pressure is None or abs_max(stress) == 0.0:
        return field_gap(name)
    return value(name, pressure_trace_residual(stress, pressure), CASE, n=pressure.size)


def _input_table(
    facts: InputFacts,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    tables = [m.yield_table for m in facts.materials if m.yield_table is not None]
    return tables[0] if len(tables) == 1 else None


def _yield_table_matches_input(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "yield_table_matches_input"
    assert facts is not None and declared is not None and declared.yield_table
    table = _input_table(facts)
    if table is None:
        return input_gap(name, facts)
    mine = np.asarray(declared.yield_table, dtype=np.float64)
    theirs = np.asarray(table, dtype=np.float64)
    if mine.shape != theirs.shape:
        return value(
            name, 1.0, INPUT, DECLARED, detail={"knots_input": theirs.shape[1]}
        )
    scale = np.maximum(np.abs(theirs), np.finfo(np.float64).tiny)
    worst = float((np.abs(mine - theirs) / scale).max())
    return value(name, worst, INPUT, DECLARED, n=theirs.shape[1])


def _yield_table_monotone(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "yield_table_monotone"
    assert facts is not None
    table = _input_table(facts)
    if table is None:
        return input_gap(name, facts)
    largest, count = decrease_max(np.asarray(table[1]))
    return value(name, count, INPUT, n=len(table[1]), detail={"largest_drop": largest})


def _yield_table_covers_range(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "yield_table_covers_range"
    assert facts is not None and declared is not None and declared.yield_table
    state = _state(case, facts)
    if state is None:
        return field_gap(name)
    last_knot = declared.yield_table[0][-1]
    return value(name, float(state.max()) / last_knot, CASE, DECLARED, n=state.size)


MEASURES: dict[str, MeasureFn] = {
    "yield_ratio_max": needs_case(_yield_ratio_max),
    "yield_saturation_min": needs_case(_yield_saturation_min),
    "state_variable_decrease_max": needs_case(_state_variable_decrease_max),
    "state_variable_min": needs_case(_state_variable_min),
    "plane_strain_ezz_max": needs_case(_plane_strain_ezz_max),
    "out_of_plane_shear_max": needs_case(_out_of_plane_shear_max),
    "pressure_trace_residual": needs_case(_pressure_trace_residual),
    "yield_table_matches_input": needs_case(_yield_table_matches_input),
    "yield_table_monotone": needs_case(_yield_table_monotone),
    "yield_table_covers_range": needs_case(_yield_table_covers_range),
}
