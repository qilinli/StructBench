"""Numerical-health and conservation measures that need only the case and its input.

Termination reads the *raw* time axis: the state a solver writes at the exact
termination time is evidence that the run reached it (ADR-0066, design note 5).
"""

from __future__ import annotations

import numpy as np

from ...core import Case, DeclaredFacts, InputFacts
from ..kernels import relative_drift, signed_distance_to_plane
from ..results import Location, Measurement
from ._common import (
    PARTICLES,
    MeasureFn,
    field_gap,
    input_gap,
    known_particles,
    needs_case,
    particle_field,
    value,
)

#: Relative width within which a smoothing-length scale counts as "at" a bound.
#: Two float32 values divide to about 1e-7; 1e-5 leaves two decades of margin.
_AT_BOUND_RTOL = 1.0e-5

CASE, INPUT = Location.CASE, Location.INPUT


def _reached_end_time(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "reached_end_time"
    if facts is None or not facts.end_time:
        return input_gap(name, facts)
    assert case.response is not None
    time = case.response.time
    return value(name, float(time[-1]) / facts.end_time, CASE, INPUT, n=time.shape[0])


def _particle_deactivated_count(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "particle_deactivated_count"
    flags = particle_field(case, "deletion", known_particles(case, facts))
    if flags is None:
        return field_gap(name)
    # the adapter stores the solver's "deleted" flag as 0.0 / 1.0
    return value(name, int((flags > 0.5).any(axis=0).sum()), CASE, n=flags.shape[1])


def _particle_neighbors_min(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "particle_neighbors_min"
    counts = particle_field(case, "n_neighbors", known_particles(case, facts))
    if counts is None:
        return field_gap(name)
    frame, index = np.unravel_index(int(np.argmin(counts)), counts.shape)
    return value(
        name,
        float(counts.min()),
        CASE,
        n=counts.size,
        detail={"frame": int(frame), "index": int(index)},
    )


def _particle_neighbors_growth(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "particle_neighbors_growth"
    counts = particle_field(case, "n_neighbors", known_particles(case, facts))
    if counts is None or counts[0].max() <= 0:
        return field_gap(name)
    per_frame = counts.max(axis=1)
    return value(
        name,
        float(per_frame.max() / per_frame[0]),
        CASE,
        n=counts.size,
        detail={"frame": int(np.argmax(per_frame)), "initial_max": float(per_frame[0])},
    )


def _smoothing_scale(
    case: Case, facts: InputFacts | None
) -> tuple[np.ndarray, tuple[float, float]] | None:
    radius = particle_field(case, "radius", known_particles(case, facts))
    if radius is None or facts is None or facts.smoothing_length_scale_bounds is None:
        return None
    scale = radius.astype(np.float64) / radius[0].astype(np.float64)
    return scale, facts.smoothing_length_scale_bounds


def _smoothing_length_within_input_bounds(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "smoothing_length_within_input_bounds"
    got = _smoothing_scale(case, facts)
    if got is None:
        return input_gap(name, facts)
    scale, (lo, hi) = got
    excess = np.maximum(np.maximum(lo - scale, scale - hi), 0.0)
    return value(name, float(excess.max()), CASE, INPUT, n=scale.size)


def _smoothing_length_at_bound_fraction(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "smoothing_length_at_bound_fraction"
    got = _smoothing_scale(case, facts)
    if got is None:
        return input_gap(name, facts)
    scale, (lo, hi) = got
    at_lo = float(np.isclose(scale, lo, rtol=_AT_BOUND_RTOL, atol=0.0).mean())
    at_hi = float(np.isclose(scale, hi, rtol=_AT_BOUND_RTOL, atol=0.0).mean())
    return value(
        name,
        max(at_lo, at_hi),
        CASE,
        INPUT,
        n=scale.size,
        detail={"at_lower": at_lo, "at_upper": at_hi},
    )


def _rigid_surface_penetration_max(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "rigid_surface_penetration_max"
    if facts is None or not facts.rigid_planes:
        return input_gap(name, facts)
    assert case.response is not None
    keep = known_particles(case, facts)
    rows = case.elements[PARTICLES].connectivity[:, 0][keep]  # row indexes, not ids
    displacement = case.response.node["displacement"].astype(np.float64)
    positions = case.nodes.coords[rows][None, :, :] + displacement[:, rows]
    deepest, where = 0.0, 0
    for plane in facts.rigid_planes:
        distance = signed_distance_to_plane(positions, plane.point, plane.normal)
        if -distance.min() > deepest:
            deepest = float(-distance.min())
            where = int(np.unravel_index(int(np.argmin(distance)), distance.shape)[0])
    return value(
        name,
        deepest,
        CASE,
        INPUT,
        n=positions.shape[0] * positions.shape[1],
        detail={"frame": where},
    )


def _active_mass_drift(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "active_mass_drift"
    mass = particle_field(case, "mass", known_particles(case, facts))
    if mass is None:
        return field_gap(name)
    total = mass.astype(np.float64).sum(axis=1)
    return value(name, relative_drift(total), CASE, n=total.shape[0])


MEASURES: dict[str, MeasureFn] = {
    "reached_end_time": needs_case(_reached_end_time),
    "particle_deactivated_count": needs_case(_particle_deactivated_count),
    "particle_neighbors_min": needs_case(_particle_neighbors_min),
    "particle_neighbors_growth": needs_case(_particle_neighbors_growth),
    "smoothing_length_within_input_bounds": needs_case(
        _smoothing_length_within_input_bounds
    ),
    "smoothing_length_at_bound_fraction": needs_case(
        _smoothing_length_at_bound_fraction
    ),
    "rigid_surface_penetration_max": needs_case(_rigid_surface_penetration_max),
    "active_mass_drift": needs_case(_active_mass_drift),
}
