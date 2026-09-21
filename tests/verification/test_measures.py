"""Tests for the stage-1 measures and ``measure_case`` (ADR-0066).

One healthy synthetic particle run; every defect test breaks exactly one
thing and names the quantity that must see it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from structbench.core import (
    AbsenceReason,
    Case,
    DeclaredFacts,
    ElementBlock,
    EvidenceItem,
    InputFacts,
    MaterialInput,
    Metadata,
    Nodes,
    PartTraits,
    Response,
    RigidPlane,
    UnitsAnchor,
)
from structbench.verification import Status
from structbench.verification import measures as measures_module
from structbench.verification.measures import MEASURES, measure_case
from structbench.verification.quantities import CATALOGUE
from structbench.verification.results import CaseMeasurements, Measurement
from structbench.verification.traits import run_traits

E = EvidenceItem
T, P = 4, 4
_TABLE = ((0.0, 1.0), (100.0e6, 200.0e6))  # yield stress 100 MPa + 100 MPa per unit
_STATE = np.array([0.0, 0.1, 0.2, 0.3])  # every particle loads through all frames
_FIELDS = frozenset(
    {"node/displacement", "node/velocity"}
    | {
        f"sph/{k}"
        for k in (
            "stress",
            "strain",
            "effective_plastic_strain",
            "pressure",
            "density",
            "mass",
            "radius",
            "n_neighbors",
            "deletion",
        )
    }
)


def _sph_fields() -> dict[str, np.ndarray]:
    state = np.repeat(_STATE[:, None], P, axis=1)
    stress = np.zeros((T, P, 6), dtype=np.float32)
    stress[..., 0] = 100.0e6 + 100.0e6 * state  # uniaxial, so von Mises = sigma_xx
    neighbors = np.full((T, P), 24.0, dtype=np.float32)
    neighbors[0] = 20.0
    return {
        "stress": stress,
        "strain": np.zeros((T, P, 6), dtype=np.float32),
        "effective_plastic_strain": state.astype(np.float32),
        "pressure": (-stress[..., 0] / 3.0).astype(np.float32),
        "density": np.full((T, P), 8900.0, dtype=np.float32),
        "mass": np.full((T, P), 1.0e-6, dtype=np.float32),
        "radius": np.full((T, P), 1.0e-3, dtype=np.float32),
        "n_neighbors": neighbors,
        "deletion": np.zeros((T, P), dtype=np.float32),
    }


def _case(
    sph: dict[str, np.ndarray] | None = None,
    *,
    time: np.ndarray | None = None,
    displacement: np.ndarray | None = None,
) -> Case:
    """Four particles right of a wall at x = 0, plus a shell no input part owns."""
    coords = np.array(
        [[1e-3, 0.0], [2e-3, 0.0], [1e-3, 1e-3], [2e-3, 1e-3]]  # particles
        + [[5e-3, 0.0], [6e-3, 0.0], [6e-3, 1e-3], [5e-3, 1e-3]]  # the shell
    )
    velocity = np.zeros((T, 8, 2), dtype=np.float32)
    velocity[0, :4, 0] = -100.0
    return Case(
        metadata=Metadata(case_id="synthetic", dimension=2, source_units="SI"),
        nodes=Nodes(coords=coords, node_id=np.arange(1, 9, dtype=np.int64)),
        elements={
            "sph": ElementBlock(
                connectivity=np.arange(P, dtype=np.int64).reshape(P, 1),
                element_id=np.arange(1, P + 1, dtype=np.int64),
                part_id=np.ones(P, dtype=np.int64),
            ),
            "shell": ElementBlock(
                connectivity=np.array([[4, 5, 6, 7]], dtype=np.int64),
                element_id=np.array([99], dtype=np.int64),
                part_id=np.array([9], dtype=np.int64),
            ),
        },
        materials=[],
        response=Response(
            time=np.linspace(0.0, 3.0e-4, T) if time is None else time,
            node={
                "displacement": (
                    np.zeros((T, 8, 2), dtype=np.float32)
                    if displacement is None
                    else displacement
                ),
                "velocity": velocity,
            },
            element={"sph": _sph_fields() if sph is None else sph},
        ),
    )


def _facts(**overrides: object) -> InputFacts:
    base: dict[str, object] = {
        "parts": (PartTraits(1, 2, "particle", None),),
        "materials": (
            MaterialInput(
                2, "elastic_plastic_hydro", 8900.0, 4.0e10, 1.0e11, None, _TABLE
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
        "rigid_planes": (RigidPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        "particle_pairwise_conservative": None,
        "smoothing_length_scale_bounds": (0.2, 2.0),
        "unparsable": frozenset(),
    }
    return InputFacts(**{**base, **overrides})  # type: ignore[arg-type]


def _declared(**overrides: object) -> DeclaredFacts:
    base: dict[str, object] = {
        "unit_system": "kg-m-s",
        "anchors": (
            UnitsAnchor("density", 8900.0, 4, "material:2:density", "measured"),
            UnitsAnchor("stress", 4.0e10, 2, "material:2:shear_modulus", "measured"),
            UnitsAnchor("length", 5.0e-3, 2, "geometry:extent_x", "measured"),
        ),
        "fields": _FIELDS,
        "discretisation": "SPH",
        "erosion": False,
        "yield_table": _TABLE,
        "material_family": None,
        "quasi_static": False,
    }
    return DeclaredFacts(**{**base, **overrides})  # type: ignore[arg-type]


def _run(
    case: Case | None = None,
    facts: InputFacts | None = None,
    declared: DeclaredFacts | None = None,
    *,
    no_facts: bool = False,
    no_case: bool = False,
) -> CaseMeasurements:
    return measure_case(
        None if no_case else (case or _case()),
        None if no_facts else (facts or _facts()),
        declared or _declared(),
        case_id="synthetic",
    )


def _get(result: CaseMeasurements, name: str) -> Measurement:
    return next(m for m in result.measurements if m.quantity == name)


def _with(field: str, edit) -> Case:  # noqa: ANN001 - a one-line array mutation
    fields = _sph_fields()
    edit(fields[field])
    return _case(fields)


# --- the orchestrator ---------------------------------------------------------


def test_every_catalogue_row_answers_once_in_name_order() -> None:
    result = _run()
    names = [m.quantity for m in result.measurements]
    assert names == sorted(q.name for q in CATALOGUE)
    assert result.run_traits == {"explicit", "dim2"}
    assert result.declared_intent == frozenset()


def test_a_healthy_run_measures_every_implemented_quantity() -> None:
    result = _run()
    unmeasured = [name for name in MEASURES if _get(result, name).value is None]
    assert unmeasured == []


def test_a_specified_row_is_a_platform_gap_not_a_data_gap() -> None:
    result = _run()
    for q in CATALOGUE:
        row = _get(result, q.name)
        if q.status is Status.SPECIFIED and not row.not_applicable:
            assert row.absence is not None, q.name


def test_a_measure_that_raises_is_recorded_not_propagated(monkeypatch) -> None:  # noqa: ANN001
    def boom(*_: object) -> Measurement:
        raise OSError("corrupt chunk")

    monkeypatch.setitem(measures_module.MEASURES, "nonfinite_count", boom)
    row = _get(_run(), "nonfinite_count")
    assert row.absence is not None
    assert row.absence.reason is AbsenceReason.SOURCE_UNREADABLE


def test_without_an_input_the_trait_gated_rows_name_e1() -> None:
    result = _run(no_facts=True)
    row = _get(result, "state_variable_decrease_max")
    assert row.absence is not None
    assert row.absence.reason is AbsenceReason.SOURCE_MISSING
    assert row.absence.missing == {E.E1}
    assert _get(result, "nonfinite_count").value == 0.0  # needs no input
    assert result.run_traits == frozenset()


def test_a_run_with_no_case_is_still_measured_on_its_input() -> None:
    result = _run(no_case=True)
    assert _get(result, "input_density_plausible").value == 8900.0
    row = _get(result, "nonfinite_count")
    assert row.absence is not None and row.absence.missing == {E.E8}


def test_traits_decide_before_evidence() -> None:
    result = _run(facts=_facts(dimension=3, plane_strain=None), no_case=True)
    assert _get(result, "plane_strain_ezz_max").not_applicable


def test_a_response_field_that_was_never_stored_names_e8() -> None:
    fields = _sph_fields()
    del fields["n_neighbors"]
    row = _get(_run(_case(fields)), "particle_neighbors_min")
    assert row.absence is not None
    assert row.absence.reason is AbsenceReason.SOURCE_MISSING
    assert row.absence.missing == {E.E8}


# --- healthy values -----------------------------------------------------------

_HEALTHY = {
    "reached_end_time": 1.0,
    "yield_ratio_max": 1.0,
    "yield_saturation_min": 1.0,
    "state_variable_decrease_max": 0.0,
    "state_variable_min": 0.0,
    "plane_strain_ezz_max": 0.0,
    "out_of_plane_shear_max": 0.0,
    "pressure_trace_residual": 0.0,
    "nonfinite_count": 0.0,
    "time_axis_monotone": 0.0,
    "terminal_artifact_frames": 0.0,
    "elements_without_input_part": 1.0,  # the shell no input part owns
    "particle_deactivated_count": 0.0,
    "particle_neighbors_min": 20.0,
    "particle_neighbors_growth": 1.2,
    "smoothing_length_within_input_bounds": 0.0,
    "smoothing_length_at_bound_fraction": 0.0,
    "rigid_surface_penetration_max": 0.0,
    "active_mass_drift": 0.0,
    "units_anchors_consistent": 0.0,
    "input_density_plausible": 8900.0,
    "input_constants_plausible": 1.0e11,
    "input_dimensionless_groups_plausible": 1.0e-3,
    "response_magnitudes_plausible": 100.0,
    "density_slot_matches_input": 0.0,
    "yield_table_matches_input": 0.0,
    "yield_table_monotone": 0.0,
    "yield_table_covers_range": 0.3,
    "fields_match_declaration": 0.0,
    "declared_traits_match_input": 0.0,
}


def test_the_healthy_table_covers_every_measure() -> None:
    assert set(_HEALTHY) == set(MEASURES)


@pytest.mark.parametrize("name", sorted(_HEALTHY))
def test_healthy_value(name: str) -> None:
    assert _get(_run(), name).value == pytest.approx(_HEALTHY[name], abs=1e-6)


def test_the_yield_maximum_says_where_it_was_found() -> None:
    fields = _sph_fields()
    fields["stress"][2, 3, 0] *= 1.05
    detail = _get(_run(_case(fields)), "yield_ratio_max").detail
    assert (detail["frame"], detail["index"]) == (2, 3)
    assert detail["state_at_max"] == pytest.approx(0.2)
    assert detail["nearest_knot_distance"] == pytest.approx(0.2)
    assert detail["fraction_above_one"] >= 1 / (T * P)


# --- one seeded defect per measure --------------------------------------------


def test_stress_stored_a_thousand_times_too_small_fails_the_lower_side() -> None:
    def shrink(a: np.ndarray) -> None:
        a *= 1.0e-3

    result = _run(_with("stress", shrink))
    assert _get(result, "yield_ratio_max").value == pytest.approx(1.0e-3)
    assert _get(result, "yield_saturation_min").value == pytest.approx(1.0e-3)


def test_no_sample_mid_loading_makes_saturation_not_applicable() -> None:
    def freeze(a: np.ndarray) -> None:
        a[:] = 0.0

    assert _get(
        _run(_with("effective_plastic_strain", freeze)), "yield_saturation_min"
    ).not_applicable


def test_a_state_variable_that_goes_backward_is_measured() -> None:
    def rewind(a: np.ndarray) -> None:
        a[3, 1] = 0.15  # 0.2 -> 0.15

    row = _get(
        _run(_with("effective_plastic_strain", rewind)), "state_variable_decrease_max"
    )
    assert row.value == pytest.approx(0.05)
    assert row.detail["decreasing_steps"] == 1


def test_out_of_plane_components_are_measured() -> None:
    def strain_zz(a: np.ndarray) -> None:
        a[1, 0, 2] = -2.0e-3

    def shear_yz(a: np.ndarray) -> None:
        a[1, 0, 4] = 5.0e5

    assert _get(
        _run(_with("strain", strain_zz)), "plane_strain_ezz_max"
    ).value == pytest.approx(2.0e-3)
    assert _get(
        _run(_with("stress", shear_yz)), "out_of_plane_shear_max"
    ).value == pytest.approx(5.0e5)


def test_a_pressure_slot_with_the_wrong_sign_is_measured() -> None:
    def flip(a: np.ndarray) -> None:
        a *= -1.0

    # |p + tr/3| = 2 tr/3 at the largest stress, over max |sigma| = that stress
    assert _get(
        _run(_with("pressure", flip)), "pressure_trace_residual"
    ).value == pytest.approx(2 / 3)


def test_a_nan_in_the_last_frame_is_counted() -> None:
    def poison(a: np.ndarray) -> None:
        a[-1, 0] = np.nan

    assert _get(_run(_with("density", poison)), "nonfinite_count").value == 1.0


def test_time_axis_defects() -> None:
    backward = np.array([0.0, 2.0e-4, 1.0e-4, 3.0e-4])
    assert _get(_run(_case(time=backward)), "time_axis_monotone").value == 1.0
    stub = np.array([0.0, 1.0e-4, 2.0e-4, 2.04e-4])  # final interval << the others
    result = _run(_case(time=stub))
    assert _get(result, "terminal_artifact_frames").value == 1.0
    # termination reads the raw axis: the stub frame is evidence, not noise
    assert _get(result, "reached_end_time").value == pytest.approx(2.04 / 3.0)


def test_particle_health_defects() -> None:
    def delete(a: np.ndarray) -> None:
        a[2:, 1] = 1.0

    def starve(a: np.ndarray) -> None:
        a[3, 2] = 3.0

    def clump(a: np.ndarray) -> None:
        a[3, 2] = 80.0

    assert (
        _get(_run(_with("deletion", delete)), "particle_deactivated_count").value == 1.0
    )
    row = _get(_run(_with("n_neighbors", starve)), "particle_neighbors_min")
    assert (row.value, row.detail["frame"], row.detail["index"]) == (3.0, 3, 2)
    assert (
        _get(_run(_with("n_neighbors", clump)), "particle_neighbors_growth").value
        == 4.0
    )


def test_smoothing_length_against_the_input_bounds() -> None:
    def pin(a: np.ndarray) -> None:
        a[2:, :2] = 0.2e-3  # a quarter of all samples sit on the lower bound

    def escape(a: np.ndarray) -> None:
        a[3, 0] = 2.5e-3

    row = _get(_run(_with("radius", pin)), "smoothing_length_at_bound_fraction")
    assert row.value == pytest.approx(0.25)
    assert row.detail["at_lower"] == pytest.approx(0.25)
    excess = _get(_run(_with("radius", escape)), "smoothing_length_within_input_bounds")
    assert excess.value == pytest.approx(0.5)


def test_a_fixed_smoothing_length_has_no_bounds_to_check() -> None:
    result = _run(facts=_facts(smoothing_length_scale_bounds=(1.0, 1.0)))
    assert _get(result, "smoothing_length_at_bound_fraction").not_applicable


def test_a_particle_through_the_wall_is_measured() -> None:
    displacement = np.zeros((T, 8, 2), dtype=np.float32)
    displacement[2, 0, 0] = -1.4e-3  # starts at x = 1 mm
    row = _get(_run(_case(displacement=displacement)), "rigid_surface_penetration_max")
    assert row.value == pytest.approx(0.4e-3, rel=1e-4)
    assert row.detail["frame"] == 2


def test_mass_that_changes_without_erosion_is_measured() -> None:
    def leak(a: np.ndarray) -> None:
        a[3] *= 0.98

    assert _get(_run(_with("mass", leak)), "active_mass_drift").value == pytest.approx(
        0.02, rel=1e-4
    )


def test_orphan_elements_stay_out_of_the_per_part_measures() -> None:
    fields = _sph_fields()
    case = _case(fields)
    block = case.elements["sph"]
    orphaned = dataclasses.replace(
        block, part_id=np.array([1, 1, 1, 7], dtype=np.int64)
    )
    fields["effective_plastic_strain"][3, 3] = 0.0  # a backward step on the orphan only
    case = dataclasses.replace(case, elements={**case.elements, "sph": orphaned})
    result = _run(case)
    assert _get(result, "elements_without_input_part").value == 2.0
    assert _get(result, "state_variable_decrease_max").value == 0.0


# --- units --------------------------------------------------------------------


def test_an_anchor_the_input_contradicts_is_counted() -> None:
    wrong = _facts(
        materials=(
            MaterialInput(2, "elastic_plastic_hydro", 8.9, 4.0e10, None, None, _TABLE),
        )
    )
    row = _get(_run(facts=wrong), "units_anchors_consistent")
    assert row.value == 1.0
    assert row.n_samples == 3


def test_an_anchor_locator_the_instrument_cannot_resolve_is_a_platform_gap() -> None:
    odd = _declared(
        anchors=(
            UnitsAnchor("velocity", 100.0, 2, "initial_condition:velocity", "measured"),
        )
    )
    row = _get(_run(declared=odd), "units_anchors_consistent")
    assert row.absence is not None
    assert row.absence.reason is AbsenceReason.UNSUPPORTED


def test_no_anchors_means_no_home_for_the_declaration() -> None:
    row = _get(_run(declared=_declared(anchors=())), "units_anchors_consistent")
    assert row.absence is not None
    assert row.absence.reason is AbsenceReason.NO_DECLARATION_HOME


def test_the_most_extreme_input_constant_is_the_one_reported() -> None:
    materials = (
        MaterialInput(2, "elastic_plastic_hydro", 8900.0, 4.0e10, None, None, _TABLE),
        MaterialInput(
            3, "rigid", 7.85e-9, None, 2.1e5, 0.3, None
        ),  # tonne-mm-s left in
    )
    result = _run(facts=_facts(materials=materials))
    assert _get(result, "input_density_plausible").value == 7.85e-9
    assert _get(result, "input_constants_plausible").value == 2.1e5


def test_density_in_the_wrong_slot_is_an_ingestion_finding() -> None:
    def misplace(a: np.ndarray) -> None:
        a[:] = 0.4  # some other history variable landed in the density slot

    row = _get(_run(_with("density", misplace)), "density_slot_matches_input")
    assert row.value == pytest.approx(1.0, abs=1e-3)


# --- input and declaration cross-checks ----------------------------------------


def test_yield_table_defects() -> None:
    drifted = _declared(yield_table=((0.0, 1.0), (100.0e6, 210.0e6)))
    assert _get(
        _run(declared=drifted), "yield_table_matches_input"
    ).value == pytest.approx(0.05)
    shorter = _declared(yield_table=((0.0,), (100.0e6,)))
    assert _get(_run(declared=shorter), "yield_table_matches_input").value == 1.0
    sagging = ((0.0, 0.5, 1.0), (100.0e6, 99.0e6, 200.0e6))
    dented = _facts(
        materials=(
            MaterialInput(
                2, "elastic_plastic_hydro", 8900.0, 4.0e10, None, None, sagging
            ),
        )
    )
    row = _get(_run(facts=dented), "yield_table_monotone")
    assert row.value == 1.0
    assert row.detail["largest_drop"] == pytest.approx(1.0e6)


def test_declaration_mismatches_are_counted() -> None:
    fewer = _declared(fields=_FIELDS - {"sph/pressure"} | {"sph/temperature"})
    row = _get(_run(declared=fewer), "fields_match_declaration")
    assert (row.value, row.detail["undeclared"], row.detail["missing"]) == (2.0, 1, 1)
    claims = _declared(discretisation="FEM", erosion=True)
    assert _get(_run(declared=claims), "declared_traits_match_input").value == 2.0


# --- run traits ---------------------------------------------------------------


def test_run_traits_assert_only_what_the_input_establishes() -> None:
    assert run_traits(_facts(particle_pairwise_conservative=False)) == {
        "explicit",
        "dim2",
        "particle_nonconservative",
    }
    meshed = _facts(
        parts=(PartTraits(1, 2, "solid", True),),
        mass_scaling_enabled=True,
        prescribed_motion_defined=True,
        dimension=3,
    )
    assert run_traits(meshed) == {
        "explicit",
        "dim3",
        "lagrangian_mesh",
        "mass_scaled",
        "externally_driven",
    }
    coupled = _facts(
        parts=(PartTraits(1, 2, "particle", None), PartTraits(2, 2, "solid", True)),
        particle_pairwise_conservative=True,
    )
    assert run_traits(coupled) == {"explicit", "dim2"}
    assert run_traits(_facts(time_integration=None, dimension=None)) == frozenset()
