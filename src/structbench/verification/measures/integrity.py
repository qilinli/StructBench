"""Data-integrity measures (ADR-0066).

They read *all* stored frames: a non-finite value in the last frame is still
a non-finite value. Only ``terminal_artifact_frames`` looks at the interval
pattern, through ``n_valid_frames`` (ADR-0028).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ...core import AbsenceReason, Case, DeclaredFacts, EvidenceItem, InputFacts
from ...datasets import n_valid_frames
from ..kernels import nonfinite_count
from ..materials import material_class
from ..results import Location, Measurement
from ._common import (
    STATE_BLOCKS,
    MeasureFn,
    absent,
    field_gap,
    input_gap,
    needs_case,
    not_applicable,
    value,
)

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


#: Databases every run must request, whatever it models: the energy ledger
#: and time-step history (E5, E4) and the per-part account (E6).
_REQUIRED_DATABASES = ("DATABASE_GLSTAT", "DATABASE_MATSUM")
#: Databases a run must request when it has the feature they record (E7).
_REQUIRED_WITH_FEATURE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("contact", ("DATABASE_RCFORC", "DATABASE_SLEOUT")),
    ("rigid_plane", ("DATABASE_RWFORC",)),
    ("prescribed_motion", ("DATABASE_BNDOUT",)),
)
#: A ledger term is required of the runs that can produce it: left out, it is
#: missing from the solver's own total, where nothing downstream can tell it
#: from a term that is genuinely zero. A run with no contact is not asked to
#: compute contact energy.
_REQUIRED_ENERGY_TERMS: tuple[tuple[str, str], ...] = (
    ("zero_energy_mode", "under_integrated"),
    ("rigid_surface", "rigid_plane"),
    ("contact", "contact"),
    ("damping", "damping"),
)


def _input_requests_required_evidence(
    case: Case | None, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    """How many of the platform's required output requests the input omits.

    The standard input block states what a run must ask the solver to write
    for the evidence items to be readable at all. This is the one row that
    reads it back off the deck, so a dataset whose runs *could never* have
    supplied the evidence is distinguished from one whose files were lost.
    A violation is the input's, and it is fixable only before the run.
    """
    name = "input_requests_required_evidence"
    assert facts is not None
    if facts.solver != "lsdyna":
        # `_REQUIRED_DATABASES` holds LS-DYNA keyword names, so conformance
        # cannot be checked for another solver until its output-request
        # vocabulary is defined (ADR-0068 clause 8). That is the platform's
        # gap; `input_gap` here would blame a complete deck for it.
        return absent(name, AbsenceReason.UNSUPPORTED)
    if facts.databases_requested is None:
        return input_gap(name, facts)  # the deck hides what it asks for
    features = {
        "contact": bool(facts.contact_defined),
        "rigid_plane": bool(facts.rigid_planes),
        "prescribed_motion": bool(facts.prescribed_motion_defined),
        "damping": bool(facts.damping_defined),
        "under_integrated": any(p.under_integrated for p in facts.parts),
    }
    required = set(_REQUIRED_DATABASES)
    for feature, databases in _REQUIRED_WITH_FEATURE:
        if features[feature]:
            required.update(databases)
    terms = {term for term, feature in _REQUIRED_ENERGY_TERMS if features[feature]}
    missing = {d.lower() for d in required - facts.databases_requested}
    if facts.energy_terms_computed is None:
        # Not a default to assume, but a requirement in its own right: the
        # standard block asks the input to *state* which terms it computes.
        missing.add("control_energy:stated")
        n_energy = 1
    else:
        missing |= {f"control_energy:{t}" for t in terms - facts.energy_terms_computed}
        n_energy = len(terms)
    detail = {"first_missing": min(missing)} if missing else {}
    return value(
        name,
        len(missing),
        INPUT,
        n=len(required) + n_energy,
        detail=detail,
    )


#: Readers that parse the input's initial conditions into ``InputFacts``.
_READS_INITIAL_STATE = frozenset({"abaqus"})


def _initial_state_matches_input(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    """Worst deviation of frame 0 from the stated initial state, over its scale.

    Velocity rows compare the stated component on the listed nodes; hardening
    rows compare the stored plastic strain of the listed elements. Each kind
    is scaled by its largest stated magnitude (1 if all are zero).
    """
    name = "initial_state_matches_input"
    assert facts is not None and case.response is not None
    if facts.solver not in _READS_INITIAL_STATE:
        return absent(name, AbsenceReason.UNSUPPORTED)  # the reader's gap
    if facts.initial_velocity is None or facts.initial_hardening is None:
        return input_gap(name, facts)
    if not facts.initial_velocity and not facts.initial_hardening:
        return not_applicable(name)
    worst: dict[str, float] = {}
    checked = 0
    if facts.initial_velocity:
        velocity = case.response.node.get("velocity")
        if velocity is None:
            return field_gap(name)
        index = {int(label): k for k, label in enumerate(case.nodes.node_id)}
        scale = max(abs(v) for _, _, v in facts.initial_velocity) or 1.0
        for nodes, dof, stated in facts.initial_velocity:
            if not set(nodes) <= set(index) or not 1 <= dof <= velocity.shape[2]:
                return field_gap(name)
            rows = np.array([index[n] for n in sorted(nodes)], dtype=np.intp)
            got = velocity[0, rows, dof - 1].astype(np.float64)
            worst["velocity"] = max(
                worst.get("velocity", 0.0), float(np.abs(got - stated).max()) / scale
            )
            checked += len(rows)
    if facts.initial_hardening:
        stored: dict[int, float] = {}
        for block in STATE_BLOCKS:
            peeq = case.response.element.get(block, {}).get("effective_plastic_strain")
            if peeq is not None:
                for k, label in enumerate(case.elements[block].element_id):
                    stored[int(label)] = float(peeq[0, k])
        scale = max(abs(p) for _, p in facts.initial_hardening) or 1.0
        for element, stated in facts.initial_hardening:
            if element not in stored:
                return field_gap(name)
            deviation = abs(stored[element] - stated) / scale
            worst["hardening"] = max(worst.get("hardening", 0.0), deviation)
        checked += len(facts.initial_hardening)
    kind = max(worst, key=worst.__getitem__)
    return value(name, worst[kind], CASE, INPUT, n=checked, detail={"worst": kind})


def _plastic_dissipation_late_growth(
    case: Case, facts: InputFacts | None, declared: DeclaredFacts | None
) -> Measurement:
    """Share of the final plastic work done in the last tenth of the frames."""
    name = "plastic_dissipation_late_growth"
    assert case.response is not None
    stored = case.response.globals_.get("plastic_dissipation")
    if stored is None:
        # The series is the energy record's, stored with the case; a run that
        # kept no plastic-dissipation history lacks E5, not field output.
        return absent(name, AbsenceReason.SOURCE_MISSING, EvidenceItem.E5)
    work = np.asarray(stored, dtype=np.float64)
    if work[-1] == 0.0:
        return not_applicable(name)  # nothing yielded
    at = min(int(np.floor(0.9 * work.size)), work.size - 1)
    growth = float((work[-1] - work[at]) / work[-1])
    return value(name, growth, CASE, n=work.size, detail={"from_frame": at})


MEASURES: dict[str, MeasureFn] = {
    "initial_state_matches_input": needs_case(_initial_state_matches_input),
    "plastic_dissipation_late_growth": needs_case(_plastic_dissipation_late_growth),
    "input_requests_required_evidence": _input_requests_required_evidence,
    "nonfinite_count": needs_case(_nonfinite_count),
    "time_axis_monotone": needs_case(_time_axis_monotone),
    "terminal_artifact_frames": needs_case(_terminal_artifact_frames),
    "elements_without_input_part": needs_case(_elements_without_input_part),
    "fields_match_declaration": needs_case(_fields_match_declaration),
    "declared_traits_match_input": needs_case(_declared_traits_match_input),
}
