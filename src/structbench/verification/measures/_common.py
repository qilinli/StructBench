"""Helpers shared by the measure modules (ADR-0066).

A measure is ``(case, facts, declared) -> Measurement``: a value, never a
verdict. ``measure_case`` calls one only after both gates passed, so a measure
may assume its catalogue evidence exists — but a *specific* fact or field can
still be missing, and then it answers with a typed absence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from ...core import (
    Absence,
    AbsenceReason,
    Case,
    DeclaredFacts,
    EvidenceItem,
    InputFacts,
)
from ..materials import MaterialClass, material_class
from ..quantities import get_quantity
from ..results import Location, Measurement

MeasureFn = Callable[
    [Case | None, InputFacts | None, DeclaredFacts | None], Measurement
]
CaseMeasureFn = Callable[[Case, InputFacts | None, DeclaredFacts | None], Measurement]

PARTICLES = "sph"  # element-block key of particle elements in the case schema
#: Element blocks whose integration points carry a constitutive state: SPH
#: particles, and meshed continuum elements (plan 2, Decision 5).
STATE_BLOCKS = (PARTICLES, "solid")


def needs_case(fn: CaseMeasureFn) -> MeasureFn:
    """Adapt a measure that reads the case; the evidence gate guarantees one."""

    def wrapped(
        case: Case | None, facts: InputFacts | None, declared: DeclaredFacts | None
    ) -> Measurement:
        if case is None or case.response is None:
            raise ValueError("this measure needs a case with a response")
        return fn(case, facts, declared)

    return wrapped


def value(
    name: str,
    number: float,
    *locations: Location,
    n: int | None = None,
    detail: Mapping[str, float | int | str] | None = None,
) -> Measurement:
    """A measured value, in the unit the catalogue gives the quantity."""
    return Measurement(
        name,
        float(number),
        get_quantity(name).unit,
        frozenset(locations),
        n_samples=n,
        detail=detail or {},
    )


def absent(name: str, reason: AbsenceReason, *missing: EvidenceItem) -> Measurement:
    """A typed absence for ``name``."""
    return Measurement(
        name, None, get_quantity(name).unit, absence=Absence(reason, frozenset(missing))
    )


def not_applicable(name: str) -> Measurement:
    """The quantity does not exist for this case."""
    return Measurement(name, None, get_quantity(name).unit, not_applicable=True)


def input_gap(name: str, facts: InputFacts | None) -> Measurement:
    """The input did not establish a fact this measure needs."""
    unparsable = facts is not None and bool(facts.unparsable)
    reason = AbsenceReason.UNPARSABLE if unparsable else AbsenceReason.SOURCE_MISSING
    return absent(name, reason, EvidenceItem.E1)


def unstated_or_unread(name: str, facts: InputFacts) -> Measurement:
    """No such quantity in the input -- because none is stated, or none was read.

    ``not_applicable`` is a claim about the input: it states no quantity of
    this kind. That claim can only be made about an input the reader got
    through. Where a card was left unparsed, one of the slots it holds might
    have stated one, so the honest answer is the typed absence.
    """
    return input_gap(name, facts) if facts.unparsable else not_applicable(name)


def field_gap(name: str) -> Measurement:
    """A response field this measure needs is not stored with the case."""
    return absent(name, AbsenceReason.SOURCE_MISSING, EvidenceItem.E8)


def known_particles(case: Case, facts: InputFacts | None) -> NDArray[np.bool_]:
    """Particles whose part the input defines; others are left out of checks."""
    block = case.elements[PARTICLES]
    if facts is None:
        return np.ones(block.part_id.shape[0], dtype=bool)
    return np.isin(block.part_id, [p.part_id for p in facts.parts])


def element_field(
    case: Case, block: str, name: str, keep: NDArray[np.bool_]
) -> NDArray[np.float32] | None:
    """One element-block response field ``(T, E_kept, ...)``, or ``None``."""
    if case.response is None:
        return None
    data = case.response.element.get(block, {}).get(name)
    return None if data is None else data[:, keep]


def particle_field(
    case: Case, name: str, keep: NDArray[np.bool_]
) -> NDArray[np.float32] | None:
    """One particle response field ``(T, P_kept, ...)``, or ``None`` if not stored."""
    return element_field(case, PARTICLES, name, keep)


def class_mask(
    case: Case,
    facts: InputFacts,
    wanted: Callable[[MaterialClass], bool],
    block: str = PARTICLES,
) -> NDArray[np.bool_]:
    """Elements of ``block`` whose part's material class satisfies ``wanted``."""
    by_id = {m.material_id: m for m in facts.materials}
    parts: list[int] = []
    for part in facts.parts:
        material = by_id.get(part.material_id)
        cls = material_class(material.canonical_model) if material else None
        if cls is not None and wanted(cls):
            parts.append(part.part_id)
    return np.isin(case.elements[block].part_id, parts)
