"""Tests for the quantity catalogue and its gates (ADR-0066 clauses 2 and 5)."""

from __future__ import annotations

import pytest

from structbench.core import (
    AbsenceReason,
    DeclaredFacts,
    EvidenceItem,
    InputFacts,
    MaterialInput,
    PartTraits,
    RigidPlane,
    UnitsAnchor,
)
from structbench.verification import Status
from structbench.verification.quantities import CATALOGUE, gate, get_quantity

E = EvidenceItem
_STAGE1 = frozenset({E.E1, E.E8, E.E10A})

_IMPLEMENTED = {
    "active_mass_drift",
    "energy_gain_max",
    "energy_loss_max",
    "energy_residual_final",
    "solver_error_count",
    "solver_identity_complete",
    "solver_warning_count",
    "terminated_normally",
    "timestep_min_ratio",
    "total_energy_change_final",
    "declared_traits_match_input",
    "stored_globals_match_ledger",
    "input_requests_required_evidence",
    "density_slot_matches_input",
    "elements_without_input_part",
    "kinetic_energy_closure",
    "fields_match_declaration",
    "input_constants_plausible",
    "input_density_plausible",
    "input_strength_plausible",
    "input_dimensionless_groups_plausible",
    "nonfinite_count",
    "out_of_plane_shear_max",
    "particle_deactivated_count",
    "particle_neighbors_growth",
    "particle_neighbors_min",
    "plane_strain_ezz_max",
    "pressure_trace_residual",
    "reached_end_time",
    "response_magnitudes_plausible",
    "rigid_surface_penetration_max",
    "smoothing_length_at_bound_fraction",
    "smoothing_length_within_input_bounds",
    "state_variable_decrease_max",
    "state_variable_min",
    "terminal_artifact_frames",
    "time_axis_monotone",
    "units_anchors_consistent",
    "yield_ratio_max",
    "yield_saturation_min",
    "yield_table_covers_range",
    "yield_table_matches_input",
    "yield_table_monotone",
}


def _facts(**overrides: object) -> InputFacts:
    base: dict[str, object] = {
        "parts": (PartTraits(1, 2, "particle", None),),
        "materials": (
            MaterialInput(
                2, "elastic_plastic_hydro", 8900.0, 3.759e10, None, None, None
            ),
        ),
        "time_integration": "explicit",
        "dimension": 2,
        "plane_strain": True,
        "end_time": 3.0e-4,
        "other_termination_criteria": frozenset(),
        "mass_scaling_enabled": False,
        "erosion_enabled": False,
        "contact_defined": False,
        "prescribed_motion_defined": False,
        "damping_defined": False,
        "rigid_planes": (RigidPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        "particle_pairwise_conservative": False,
        "smoothing_length_scale_bounds": (1.0, 1.0),
        "energy_terms_computed": None,
        "databases_requested": None,
        "unparsable": frozenset(),
    }
    base.update(overrides)
    return InputFacts(**base)  # type: ignore[arg-type]


_DECLARED = DeclaredFacts(
    unit_system="g-mm-ms",
    fields=frozenset({"sph/stress"}),
    discretisation="SPH",
    erosion=False,
    yield_table=((0.0, 0.5), (2.0e8, 2.5e8)),
)


def test_names_are_unique_and_sorted() -> None:
    names = [q.name for q in CATALOGUE]
    assert names == sorted(set(names))


def test_the_requirement_is_the_union_of_the_rows() -> None:
    union = frozenset().union(*(q.requires for q in CATALOGUE))
    assert union == frozenset(EvidenceItem)


def test_stage_one_rows_are_implemented_and_versioned() -> None:
    implemented = {q.name for q in CATALOGUE if q.status is Status.IMPLEMENTED}
    assert implemented == _IMPLEMENTED
    for q in CATALOGUE:
        expected = 1 if q.status is Status.IMPLEMENTED else None
        assert q.definition_version == expected, q.name
        assert q.meaning.strip() and "\n" not in q.meaning, q.name


def test_implemented_rows_need_only_evidence_a_run_has_supplied() -> None:
    # E6 and E7 have no record field yet: no run has supplied them. E9 is
    # found in the files (shared sample instants), not declared.
    built = {E.E1, E.E2, E.E3, E.E4, E.E5, E.E8, E.E9, E.E10A, E.E10B}
    for q in CATALOGUE:
        if q.status is Status.IMPLEMENTED:
            assert q.requires <= built, q.name


def test_get_quantity_names_the_valid_choices() -> None:
    assert get_quantity("nonfinite_count").name == "nonfinite_count"
    with pytest.raises(KeyError, match="nonfinite_count"):
        get_quantity("no_such_quantity")


def test_an_open_row_is_measurable() -> None:
    assert gate(get_quantity("nonfinite_count"), _facts(), _DECLARED, _STAGE1) is None
    assert gate(get_quantity("yield_ratio_max"), _facts(), _DECLARED, _STAGE1) is None


def test_traits_own_not_applicable() -> None:
    # hourglass modes do not exist on a particle part, whatever was computed
    row = gate(
        get_quantity("zero_energy_mode_peak_over_internal_peak"),
        _facts(),
        _DECLARED,
        _STAGE1,
    )
    assert row is not None and row.not_applicable
    shear = gate(
        get_quantity("out_of_plane_shear_max"), _facts(dimension=3), None, _STAGE1
    )
    assert shear is not None and shear.not_applicable


def test_unknown_traits_are_never_not_applicable() -> None:
    row = gate(get_quantity("plane_strain_ezz_max"), None, None, frozenset({E.E8}))
    assert row is not None and not row.not_applicable
    assert row.absence is not None
    assert row.absence.reason is AbsenceReason.SOURCE_MISSING
    assert row.absence.missing == {E.E1}

    unknown = gate(
        get_quantity("timestep_min_ratio"),
        _facts(time_integration=None, unparsable=frozenset({"include"})),
        None,
        _STAGE1,
    )
    assert unknown is not None and unknown.absence is not None
    assert unknown.absence.reason is AbsenceReason.UNPARSABLE


def test_missing_evidence_is_named() -> None:
    row = gate(get_quantity("energy_gain_max"), _facts(), _DECLARED, _STAGE1)
    assert row is not None and row.absence is not None
    assert row.absence.reason is AbsenceReason.SOURCE_MISSING
    assert row.absence.missing == {E.E5}


def test_missing_anchors_have_no_home_yet() -> None:
    row = gate(get_quantity("units_anchors_consistent"), _facts(), _DECLARED, _STAGE1)
    assert row is not None and row.absence is not None
    assert row.absence.reason is AbsenceReason.NO_DECLARATION_HOME
    assert row.absence.missing == {E.E10B}

    anchored = DeclaredFacts(
        unit_system="g-mm-ms",
        anchors=(UnitsAnchor("density", 8900.0, 3, "material:2:density", "handbook"),),
    )
    supplied = _STAGE1 | {E.E10B}
    assert (
        gate(get_quantity("units_anchors_consistent"), _facts(), anchored, supplied)
        is None
    )


def test_a_missing_declaration_blocks_the_row() -> None:
    row = gate(
        get_quantity("yield_ratio_max"),
        _facts(),
        DeclaredFacts(unit_system="g-mm-ms"),
        _STAGE1,
    )
    assert row is not None and row.absence is not None
    assert row.absence.reason is AbsenceReason.SOURCE_MISSING


def test_an_unsupported_material_class_is_the_platforms_gap() -> None:
    facts = _facts(materials=(MaterialInput(2, None, 2400.0, None, None, None, None),))
    row = gate(get_quantity("yield_ratio_max"), facts, _DECLARED, _STAGE1)
    assert row is not None and row.absence is not None
    assert row.absence.reason is AbsenceReason.UNSUPPORTED


def test_a_specified_row_with_all_its_evidence_is_an_instrument_gap() -> None:
    everything = frozenset(EvidenceItem)
    row = gate(get_quantity("external_work_closure"), _facts(), _DECLARED, everything)
    assert row is not None and row.absence is not None
    assert row.absence.reason is AbsenceReason.UNSUPPORTED


def test_every_quantity_declares_where_a_violation_lands() -> None:
    """``bears_on`` lets a report say where a finding sits, from the definition."""
    assert {q.bears_on for q in CATALOGUE} == {"input", "response", "run", "declared"}
    where = {q.name: q.bears_on for q in CATALOGUE}
    assert where["yield_table_monotone"] == "input"  # a defect in the solver input
    assert where["elements_without_input_part"] == "response"  # in the stored fields
    assert where["terminated_normally"] == "run"  # in the solver's own record
    assert where["fields_match_declaration"] == "declared"  # in the benchmark's card
    assert where["nonfinite_count"] == "response"


def test_a_surface_that_exists_but_cannot_be_evaluated_is_not_not_applicable() -> None:
    """ADR-0067: `not_applicable` claims the quantity does not exist here.

    K&C concrete has a yield surface; its arguments are simply not exported,
    which is a fact about the solver's output, not about the material. The
    row must say so rather than read as though yield were irrelevant to a
    concrete impact benchmark.
    """
    concrete = MaterialInput(1, "concrete_damage", 2400.0, None, None, 0.2, None)
    facts = _facts(materials=(concrete,), parts=(PartTraits(1, 1, "particle", None),))
    row = gate(get_quantity("yield_ratio_max"), facts, _DECLARED, _STAGE1)
    assert row is not None
    assert not row.not_applicable
    assert row.absence is not None
    assert row.absence.reason is AbsenceReason.NOT_AVAILABLE_FROM_SOLVER


def test_a_material_with_no_surface_at_all_still_does_not_apply() -> None:
    """`rigid` carries no yield surface, so the row genuinely does not apply."""
    rigid = MaterialInput(1, "rigid", 7850.0, None, 2.1e11, 0.3, None)
    facts = _facts(materials=(rigid,), parts=(PartTraits(1, 1, "particle", None),))
    row = gate(get_quantity("yield_ratio_max"), facts, _DECLARED, _STAGE1)
    assert row is not None and row.not_applicable
