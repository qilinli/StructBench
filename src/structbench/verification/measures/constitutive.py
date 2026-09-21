"""Constitutive measures, dispatched on material class (ADR-0066 clause 6).

The operative yield table is the *declared* one (ADR-0064's single hardening
curve, in Pa); the input's own table is read only to check the two agree.
Every solver state is a genuine state, so these use all stored frames.
"""

from __future__ import annotations

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
from ..results import Location, Measurement
from ._common import (
    MeasureFn,
    class_mask,
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


def _yield_inputs(
    case: Case, facts: InputFacts
) -> tuple[np.ndarray, np.ndarray] | None:
    keep = class_mask(case, facts, lambda c: c.yield_law == "tabulated_j2")
    stress = particle_field(case, "stress", keep)
    state = particle_field(case, _STATE, keep)
    if stress is None or state is None or not keep.any():
        return None
    return von_mises_from_voigt(stress), state.astype(np.float64)


def _yield_ratio_max(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "yield_ratio_max"
    assert facts is not None and declared is not None and declared.yield_table
    got = _yield_inputs(case, facts)
    if got is None:
        return field_gap(name)
    vm, state = got
    knots, sy = declared.yield_table
    extreme = yield_ratio_max(vm, state, knots, sy)
    ratio = vm / interp_yield_stress(state, knots, sy)
    at = float(state[extreme.frame, extreme.index])
    return value(
        name,
        extreme.value,
        CASE,
        DECLARED,
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
    assert facts is not None and declared is not None and declared.yield_table
    got = _yield_inputs(case, facts)
    if got is None:
        return field_gap(name)
    vm, state = got
    loading = yielding_mask(state)
    if not loading.any():
        return not_applicable(name)  # nothing is yielding across two stored intervals
    knots, sy = declared.yield_table
    ratio = vm / interp_yield_stress(state, knots, sy)
    return value(
        name, float(ratio[loading].max()), CASE, DECLARED, n=int(loading.sum())
    )


def _state(case: Case, facts: InputFacts) -> np.ndarray | None:
    keep = class_mask(case, facts, lambda c: c.monotone)
    state = particle_field(case, _STATE, keep)
    return None if state is None or not keep.any() else state.astype(np.float64)


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
