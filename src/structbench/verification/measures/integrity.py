"""Data-integrity measures (ADR-0066).

They read *all* stored frames: a non-finite value in the last frame is still
a non-finite value. Only ``terminal_artifact_frames`` looks at the interval
pattern, through ``n_valid_frames`` (ADR-0028).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ...core import Case, DeclaredFacts, InputFacts
from ...datasets import n_valid_frames
from ..kernels import nonfinite_count
from ..materials import material_class
from ..results import Location, Measurement
from ._common import MeasureFn, input_gap, needs_case, value

CASE, INPUT, DECLARED = Location.CASE, Location.INPUT, Location.DECLARED


def _nonfinite_count(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    response = case.response
    assert response is not None
    arrays: list[NDArray[np.generic]] = [
        response.time,
        *response.node.values(),
        *response.globals_.values(),
    ]
    for block in response.element.values():
        arrays.extend(block.values())
    total = sum(nonfinite_count(a) for a in arrays)
    return value("nonfinite_count", total, CASE, n=sum(a.size for a in arrays))


def _time_axis_monotone(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    assert case.response is not None
    time = case.response.time
    backward = int((np.diff(time) <= 0.0).sum())
    return value("time_axis_monotone", backward, CASE, n=time.shape[0])


def _terminal_artifact_frames(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    assert case.response is not None
    time = case.response.time
    dropped = time.shape[0] - n_valid_frames(time)
    return value("terminal_artifact_frames", dropped, CASE, n=time.shape[0])


def _elements_without_input_part(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "elements_without_input_part"
    if facts is None or not facts.parts:
        return input_gap(name, facts)
    known = [p.part_id for p in facts.parts]
    orphans = sum(
        int((~np.isin(block.part_id, known)).sum()) for block in case.elements.values()
    )
    total = sum(block.part_id.shape[0] for block in case.elements.values())
    return value(name, orphans, CASE, INPUT, n=total)


def _stored_fields(case: Case) -> frozenset[str]:
    assert case.response is not None
    names = {f"node/{k}" for k in case.response.node}
    names |= {f"global/{k}" for k in case.response.globals_}
    for etype, block in case.response.element.items():
        names |= {f"{etype}/{k}" for k in block}
    return frozenset(names)


def _fields_match_declaration(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    assert declared is not None and declared.fields is not None
    stored = _stored_fields(case)
    return value(
        "fields_match_declaration",
        len(stored ^ declared.fields),
        CASE,
        DECLARED,
        n=len(stored | declared.fields),
        detail={
            "undeclared": len(stored - declared.fields),
            "missing": len(declared.fields - stored),
        },
    )


def _derived_discretisation(facts: InputFacts) -> str | None:
    """``"SPH"`` / ``"FEM"`` / ``"coupled"`` from the load-carrying parts."""
    by_id = {m.material_id: m for m in facts.materials}
    kinds: set[str] = set()
    for part in facts.parts:
        material = by_id.get(part.material_id)
        cls = material_class(material.canonical_model) if material else None
        if cls is not None and not cls.structural:
            continue
        kinds.add("SPH" if part.discretisation == "particle" else "FEM")
    if not kinds:
        return None
    return kinds.pop() if len(kinds) == 1 else "coupled"


def _declared_traits_match_input(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    name = "declared_traits_match_input"
    assert declared is not None
    if facts is None:
        return input_gap(name, facts)
    pairs = [
        (declared.discretisation, _derived_discretisation(facts)),
        (declared.erosion, facts.erosion_enabled),
    ]
    comparable = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not comparable:
        return input_gap(name, facts)
    contradictions = sum(a != b for a, b in comparable)
    return value(name, contradictions, INPUT, DECLARED, n=len(comparable))


MEASURES: dict[str, MeasureFn] = {
    "nonfinite_count": needs_case(_nonfinite_count),
    "time_axis_monotone": needs_case(_time_axis_monotone),
    "terminal_artifact_frames": needs_case(_terminal_artifact_frames),
    "elements_without_input_part": needs_case(_elements_without_input_part),
    "fields_match_declaration": needs_case(_fields_match_declaration),
    "declared_traits_match_input": needs_case(_declared_traits_match_input),
}
