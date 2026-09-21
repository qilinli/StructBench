"""Tests for the criteria records and ``judge`` (ADR-0066 clauses 5 and 7).

Data-free: ``judge`` sees only measurements and criteria.
"""

from __future__ import annotations

import itertools

import pytest

from structbench.core import Absence, AbsenceReason, EvidenceItem
from structbench.verification import Verdict
from structbench.verification.criteria import (
    CRITERIA,
    CheckResult,
    Criterion,
    CriterionKind,
    Scope,
    judge,
)
from structbench.verification.quantities import CATALOGUE, get_quantity
from structbench.verification.results import (
    CaseMeasurements,
    DatasetMeasurements,
    Measurement,
    Status,
)
from structbench.verification.traits import RunTrait

# trait pairs no single run can hold at once
_EXCLUSIVE = [
    {RunTrait.EXPLICIT, RunTrait.IMPLICIT},
    {RunTrait.LAGRANGIAN_MESH, RunTrait.PARTICLE_CONSERVATIVE},
    {RunTrait.LAGRANGIAN_MESH, RunTrait.PARTICLE_NONCONSERVATIVE},
    {RunTrait.PARTICLE_CONSERVATIVE, RunTrait.PARTICLE_NONCONSERVATIVE},
    {RunTrait.INITIAL_ENERGY_DRIVEN, RunTrait.EXTERNALLY_DRIVEN},
    {RunTrait.DIM2, RunTrait.DIM3},
]


def _value(name: str, number: float) -> Measurement:
    return Measurement(name, number, get_quantity(name).unit)


def _judge_one(
    m: Measurement,
    traits: set[str] | None = None,
    versions: dict[str, int] | None = None,
) -> CheckResult:
    case = CaseMeasurements("c", None, frozenset(traits or set()), frozenset(), (m,))
    data = DatasetMeasurements(None, None, "0", versions or {}, (case,))
    return judge(data).cases[0].results[0]


# --- the records --------------------------------------------------------------


def test_a_criterion_needs_a_bound_a_rationale_and_a_known_quantity() -> None:
    def make(**kw: object) -> Criterion:
        base = {
            "quantity": "nonfinite_count",
            "lo": None,
            "hi": 0.0,
            "strict": False,
            "kind": CriterionKind.REQUIREMENT,
            "scope": Scope(),
            "source": "",
            "rationale": "why",
        }
        return Criterion(**{**base, **kw})  # type: ignore[arg-type]

    make()
    with pytest.raises(ValueError, match="lo or hi"):
        make(hi=None)
    with pytest.raises(ValueError, match="rationale"):
        make(rationale="  ")
    with pytest.raises(KeyError):
        make(quantity="not_a_quantity")
    with pytest.raises(ValueError, match="source"):
        make(kind=CriterionKind.INDICATOR)


def test_a_strict_bound_excludes_its_end_point() -> None:
    (strict,) = [c for c in CRITERIA if c.quantity == "added_mass_fraction"]
    assert strict.admits(0.049) and not strict.admits(0.05)
    (closed,) = [c for c in CRITERIA if c.quantity == "total_energy_change_final"]
    assert closed.admits(0.10) and closed.admits(-0.10) and not closed.admits(-0.11)


def test_levels_of_one_quantity_have_pairwise_disjoint_scopes() -> None:
    by_quantity = itertools.groupby(
        sorted(CRITERIA, key=lambda c: c.quantity), key=lambda c: c.quantity
    )
    for name, group in by_quantity:
        for a, b in itertools.combinations(list(group), 2):
            together = a.scope.run_traits | b.scope.run_traits
            assert any(pair <= together for pair in _EXCLUSIVE), name


def test_only_indicators_are_scoped_or_sourced() -> None:
    for c in CRITERIA:
        if c.kind is not CriterionKind.INDICATOR:
            assert c.scope == Scope() and c.source == "", c.quantity


def test_the_quantities_shipped_without_a_criterion_are_the_named_ones() -> None:
    implemented = {q.name for q in CATALOGUE if q.status is Status.IMPLEMENTED}
    assert implemented - {c.quantity for c in CRITERIA} == {
        "energy_residual_final",
        "kinetic_energy_closure",
        "solver_warning_count",
        "terminal_artifact_frames",  # a fact, reported and not judged
        "timestep_min_ratio",
        "input_dimensionless_groups_plausible",
        "particle_neighbors_growth",
        "particle_neighbors_min",
        "pressure_trace_residual",
        "rigid_surface_penetration_max",
        "smoothing_length_at_bound_fraction",
        "yield_ratio_max",
    }


# --- judge --------------------------------------------------------------------


def test_a_requirement_passes_or_fails() -> None:
    assert _judge_one(_value("nonfinite_count", 0.0)).verdict is Verdict.PASS
    failed = _judge_one(_value("nonfinite_count", 3.0))
    assert (failed.verdict, failed.reason) == (Verdict.FAIL, "outside_bound")
    assert failed.criterion == "<= 0 any run"


def test_an_instrument_tolerance_fails_and_says_it_is_provisional() -> None:
    assert _judge_one(_value("reached_end_time", 1.0001)).verdict is Verdict.PASS
    short = _judge_one(_value("reached_end_time", 0.68))
    assert (short.verdict, short.provisional) == (Verdict.FAIL, True)


def test_an_indicator_above_its_level_is_a_review_never_a_fail() -> None:
    result = _judge_one(_value("input_density_plausible", 8.9e-9))
    assert (result.verdict, result.reason) == (
        Verdict.REVIEW,
        "exceeds_reference_level",
    )
    assert result.criterion_source == "M-D6, M-D5, M-D10"
    assert _judge_one(_value("input_density_plausible", 8900.0)).verdict is Verdict.PASS


def test_a_level_applies_only_inside_its_scope() -> None:
    gain = _value("energy_gain_max", 0.07)
    inside = _judge_one(gain, {"explicit", "lagrangian_mesh", "dim3"})
    assert inside.verdict is Verdict.REVIEW
    outside = _judge_one(gain, {"explicit", "dim2"})  # a particle run of unknown class
    assert outside.verdict is Verdict.NOT_ASSESSABLE
    assert outside.reason == "no_ratified_criterion"
    assert outside.value == 0.07
    assert outside.out_of_scope_levels == (
        "<= 0.01 {explicit, lagrangian_mesh} [B-BLM-1, B-BLM-2]",
        "<= 0.01 {explicit, particle_conservative} [B-BLM-1, B-BLM-2]",
    )


def test_a_quantity_with_no_criterion_is_published_without_a_verdict() -> None:
    result = _judge_one(_value("yield_ratio_max", 1.000353))
    assert (result.verdict, result.reason) == (
        Verdict.NOT_ASSESSABLE,
        "no_ratified_criterion",
    )
    assert result.out_of_scope_levels == ()


def test_absence_and_not_applicable_pass_straight_through() -> None:
    gap = Measurement(
        "energy_gain_max",
        None,
        "1",
        absence=Absence(AbsenceReason.SOURCE_MISSING, frozenset({EvidenceItem.E5})),
    )
    result = _judge_one(gap)
    assert (result.verdict, result.reason, result.missing) == (
        Verdict.NOT_ASSESSABLE,
        "source_missing",
        ("E5",),
    )
    skipped = Measurement("implicit_convergence", None, "1", not_applicable=True)
    assert _judge_one(skipped).verdict is Verdict.NOT_APPLICABLE


def test_a_stale_definition_asks_to_be_remeasured() -> None:
    current = get_quantity("nonfinite_count").definition_version
    assert current is not None
    m = _value("nonfinite_count", 0.0)
    assert _judge_one(m, versions={"nonfinite_count": current}).verdict is Verdict.PASS
    stale = _judge_one(m, versions={"nonfinite_count": current + 1})
    assert (stale.verdict, stale.reason) == (Verdict.NOT_ASSESSABLE, "stale_definition")


def test_overlapping_scopes_are_a_construction_error() -> None:
    twice = (*CRITERIA, next(c for c in CRITERIA if c.quantity == "nonfinite_count"))
    case = CaseMeasurements(
        "c", None, frozenset(), frozenset(), (_value("nonfinite_count", 0.0),)
    )
    data = DatasetMeasurements(None, None, "0", {}, (case,))
    with pytest.raises(ValueError, match="overlapping"):
        judge(data, twice)
