"""Run traits: the facts about a run that reference levels are scoped on (ADR-0066).

A trait is asserted only when the input establishes it. An unknown stays
unasserted, so a level scoped on it does not apply — it is never assumed.
"""

from __future__ import annotations

from enum import StrEnum

from ..core import InputFacts
from .materials import material_class


class RunTrait(StrEnum):
    """Tokens a criterion's scope is written in."""

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    LAGRANGIAN_MESH = "lagrangian_mesh"
    PARTICLE_CONSERVATIVE = "particle_conservative"
    PARTICLE_NONCONSERVATIVE = "particle_nonconservative"
    MASS_SCALED = "mass_scaled"
    FRICTIONLESS_CONTACT = "frictionless_contact"
    INITIAL_ENERGY_DRIVEN = "initial_energy_driven"
    EXTERNALLY_DRIVEN = "externally_driven"
    DIM2 = "dim2"
    DIM3 = "dim3"


_MESHED = frozenset({"solid", "shell", "beam"})


def _load_carrying_kinds(facts: InputFacts) -> set[str]:
    by_id = {m.material_id: m for m in facts.materials}
    kinds: set[str] = set()
    for part in facts.parts:
        material = by_id.get(part.material_id)
        cls = material_class(material.canonical_model) if material else None
        if cls is not None and not cls.structural:
            continue
        kinds.add(part.discretisation)
    return kinds


def run_traits(facts: InputFacts | None) -> frozenset[str]:
    """The run traits ``facts`` establishes, as tokens.

    The discretisation traits describe the *whole* load-carrying model: a
    coupled run gets neither, because the sourced energy levels were stated
    for one formulation at a time. ``frictionless_contact`` and
    ``initial_energy_driven`` need a load and friction inventory that
    ``InputFacts`` does not carry yet; they arrive with the energy measures.
    """
    if facts is None:
        return frozenset()
    traits: set[RunTrait] = set()
    if facts.time_integration == "explicit":
        traits.add(RunTrait.EXPLICIT)
    elif facts.time_integration == "implicit":
        traits.add(RunTrait.IMPLICIT)
    kinds = _load_carrying_kinds(facts)
    if kinds and kinds <= _MESHED:
        traits.add(RunTrait.LAGRANGIAN_MESH)
    elif kinds == {"particle"} and facts.particle_pairwise_conservative is not None:
        traits.add(
            RunTrait.PARTICLE_CONSERVATIVE
            if facts.particle_pairwise_conservative
            else RunTrait.PARTICLE_NONCONSERVATIVE
        )
    if facts.mass_scaling_enabled:
        traits.add(RunTrait.MASS_SCALED)
    if facts.prescribed_motion_defined:
        traits.add(RunTrait.EXTERNALLY_DRIVEN)
    if facts.dimension == 2:
        traits.add(RunTrait.DIM2)
    elif facts.dimension == 3:
        traits.add(RunTrait.DIM3)
    return frozenset(str(t) for t in traits)
