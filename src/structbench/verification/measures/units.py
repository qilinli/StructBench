"""Units measures: checks against things outside the unit label (ADR-0066 clause 2).

A consistently mislabelled unit system survives every identity inside the
data, so nothing here compares the data with itself — except
``density_slot_matches_input``, which says in its name that it tests the
ingestion mapping and not the units.
"""

from __future__ import annotations

import math

import numpy as np

from ...core import (
    AbsenceReason,
    Case,
    DeclaredFacts,
    InputFacts,
    MaterialInput,
    UnitsAnchor,
)
from ..results import Location, Measurement
from ._common import (
    MeasureFn,
    absent,
    field_gap,
    input_gap,
    known_particles,
    needs_case,
    not_applicable,
    particle_field,
    value,
)

CASE, INPUT, DECLARED = Location.CASE, Location.INPUT, Location.DECLARED

# Selection rule only: with several materials, the one reported is the one
# farthest (in decades) from these centres. The bounds live in criteria.py.
_DENSITY_CENTRE = 600.0  # kg/m^3, log-centre of the condensed-matter range
_MODULUS_CENTRE = 5.5e8  # Pa, log-centre of the engineering-solid range
_STRENGTH_CENTRE = 8.2e6  # Pa, log-centre of the foam-to-carbide range

_MATERIAL_ATTRIBUTES = ("density", "shear_modulus", "youngs_modulus")
_AXES = {"extent_x": 0, "extent_y": 1, "extent_z": 2}


def _most_extreme(values: list[float], centre: float) -> float:
    return max(values, key=lambda v: abs(math.log10(v / centre)))


def _resolve(anchor: UnitsAnchor, facts: InputFacts, case: Case | None) -> float | None:
    """SI value of the input quantity an anchor names, or ``None`` if unknown."""
    kind, *rest = anchor.input_quantity.split(":")
    if kind == "material" and len(rest) == 2 and rest[1] in _MATERIAL_ATTRIBUTES:
        for material in facts.materials:
            if str(material.material_id) == rest[0]:
                found = getattr(material, rest[1])
                return None if found is None else float(found)
        return None
    if kind == "geometry" and len(rest) == 1 and rest[0] in _AXES and case is not None:
        axis = _AXES[rest[0]]
        if axis < case.nodes.coords.shape[1]:
            return float(np.ptp(case.nodes.coords[:, axis]))
    return None


def _same_to_digits(a: float, b: float, digits: int) -> bool:
    return float(f"{a:.{digits - 1}e}") == float(f"{b:.{digits - 1}e}")


def _units_anchors_consistent(
    case: Case | None, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "units_anchors_consistent"
    assert facts is not None and declared is not None
    mismatches = 0
    for anchor in declared.anchors:
        found = _resolve(anchor, facts, case)
        if found is None:
            return absent(name, AbsenceReason.UNSUPPORTED)  # locator not understood
        mismatches += not _same_to_digits(found, anchor.si_value, anchor.digits)
    synthetic = sum(a.source_kind == "synthetic" for a in declared.anchors)
    return value(
        name,
        mismatches,
        INPUT,
        DECLARED,
        n=len(declared.anchors),
        detail={"synthetic_anchors": synthetic},
    )


def _input_density_plausible(
    case: Case | None, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "input_density_plausible"
    assert facts is not None
    densities = [m.density for m in facts.materials if m.density]
    if not densities:
        return input_gap(name, facts)
    return value(
        name, _most_extreme(densities, _DENSITY_CENTRE), INPUT, n=len(densities)
    )


def _youngs_modulus(material: MaterialInput) -> float | None:
    """Young's modulus as the input gives it, or from ``E = 2 G (1 + nu)``."""
    if material.youngs_modulus:
        return material.youngs_modulus
    if material.shear_modulus and material.poisson_ratio is not None:
        return 2.0 * material.shear_modulus * (1.0 + material.poisson_ratio)
    return None


def _input_constants_plausible(
    case: Case | None, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    """The most extreme Young's modulus — the one modulus with a sourced range.

    A shear modulus alone is not screened: no surveyed source gives its
    range, and a level attaches only to the statistic its source states.
    """
    name = "input_constants_plausible"
    assert facts is not None
    moduli = [e for e in map(_youngs_modulus, facts.materials) if e]
    if not moduli:
        return not_applicable(name)  # no material here defines a Young's modulus
    return value(name, _most_extreme(moduli, _MODULUS_CENTRE), INPUT, n=len(moduli))


def _input_strength_plausible(
    case: Case | None, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    """The most extreme tabulated yield stress of any material in the input."""
    name = "input_strength_plausible"
    assert facts is not None
    strengths = [
        s for m in facts.materials if m.yield_table for s in m.yield_table[1] if s > 0
    ]
    if not strengths:
        return not_applicable(name)  # no material here states a strength
    return value(
        name, _most_extreme(strengths, _STRENGTH_CENTRE), INPUT, n=len(strengths)
    )


def _input_dimensionless_groups_plausible(
    case: Case | None, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    """Largest first-yield strain: initial yield stress over an elastic modulus.

    No unit enters a ratio of two stresses, so this sees units mixed *within*
    a material definition. With only a shear modulus, ``3 G`` stands in for
    Young's modulus (exact for an incompressible solid, within 20 % otherwise).
    """
    name = "input_dimensionless_groups_plausible"
    assert facts is not None
    strains = []
    for m in facts.materials:
        modulus = m.youngs_modulus or (
            3.0 * m.shear_modulus if m.shear_modulus else None
        )
        if m.yield_table is not None and modulus:
            strains.append(m.yield_table[1][0] / modulus)
    if not strains:
        return absent(name, AbsenceReason.UNSUPPORTED)  # no material offers both
    return value(name, max(strains), INPUT, n=len(strains))


def _response_magnitudes_plausible(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "response_magnitudes_plausible"
    assert case.response is not None
    velocity = case.response.node.get("velocity")
    if velocity is None:
        return field_gap(name)
    speed = np.linalg.norm(velocity.astype(np.float64), axis=-1)
    return value(name, float(speed.max()), CASE, n=speed.size)


def _density_slot_matches_input(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "density_slot_matches_input"
    assert facts is not None
    by_id = {m.material_id: m.density for m in facts.materials}
    block = case.elements["sph"]
    density = particle_field(case, "density", np.ones(block.part_id.shape, dtype=bool))
    if density is None:
        return field_gap(name)
    known = known_particles(case, facts)
    worst, compared = 0.0, 0
    for part in facts.parts:
        reference = by_id.get(part.material_id)
        rows = known & (block.part_id == part.part_id)
        if reference and rows.any():
            initial = float(np.median(density[0, rows].astype(np.float64)))
            worst = max(worst, abs(initial / reference - 1.0))
            compared += 1
    if not compared:
        return input_gap(name, facts)
    return value(name, worst, CASE, INPUT, n=compared)


MEASURES: dict[str, MeasureFn] = {
    "units_anchors_consistent": _units_anchors_consistent,
    "input_density_plausible": _input_density_plausible,
    "input_constants_plausible": _input_constants_plausible,
    "input_strength_plausible": _input_strength_plausible,
    "input_dimensionless_groups_plausible": _input_dimensionless_groups_plausible,
    "response_magnitudes_plausible": needs_case(_response_magnitudes_plausible),
    "density_slot_matches_input": needs_case(_density_slot_matches_input),
}
