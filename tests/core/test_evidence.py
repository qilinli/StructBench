"""Tests for the solver-neutral evidence records (ADR-0066)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from structbench.core import (
    PLATFORM_REASONS,
    Absence,
    AbsenceReason,
    DeclaredFacts,
    EvidenceItem,
    InputFacts,
    MaterialInput,
    PartTraits,
    RigidPlane,
    UnitsAnchor,
)


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
        "rigid_planes": (RigidPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        "particle_pairwise_conservative": False,
        "smoothing_length_scale_bounds": (1.0, 1.0),
        "unparsable": frozenset(),
    }
    base.update(overrides)
    return InputFacts(**base)  # type: ignore[arg-type]


def test_evidence_items_cover_e1_to_e10() -> None:
    values = {item.value for item in EvidenceItem}
    assert values == {f"E{i}" for i in range(1, 10)} | {"E10a", "E10b"}


def test_platform_reasons_are_the_instrument_side_ones() -> None:
    assert PLATFORM_REASONS == {
        AbsenceReason.NOT_INGESTED,
        AbsenceReason.UNSUPPORTED,
        AbsenceReason.NO_RATIFIED_CRITERION,
        AbsenceReason.NO_DECLARATION_HOME,
        AbsenceReason.STALE_DEFINITION,
    }
    assert AbsenceReason.NOT_REQUESTED not in PLATFORM_REASONS


def test_absence_names_the_missing_items_and_is_frozen() -> None:
    absence = Absence(AbsenceReason.SOURCE_MISSING, frozenset({EvidenceItem.E5}))
    assert absence.missing == {EvidenceItem.E5}
    assert Absence(AbsenceReason.UNSUPPORTED).missing == frozenset()
    with pytest.raises(dataclasses.FrozenInstanceError):
        absence.reason = AbsenceReason.UNPARSABLE  # type: ignore[misc]


def test_absence_rejects_a_non_evidence_member() -> None:
    with pytest.raises(TypeError, match="EvidenceItem"):
        Absence(AbsenceReason.SOURCE_MISSING, frozenset({"E5"}))  # type: ignore[arg-type]


def test_input_facts_accepts_a_particle_impact_run() -> None:
    facts = _facts()
    assert facts.parts[0].discretisation == "particle"
    assert facts.rigid_planes[0].normal == (1.0, 0.0, 0.0)


@pytest.mark.parametrize("token", ["include", "unknown_card_layout:MAT_EXAMPLE"])
def test_unparsable_accepts_construct_tokens(token: str) -> None:
    assert token in _facts(unparsable=frozenset({token})).unparsable


@pytest.mark.parametrize("token", ["C:\\runs\\deck.k", "host name", ""])
def test_unparsable_rejects_free_text(token: str) -> None:
    with pytest.raises(ValueError, match="unparsable"):
        _facts(unparsable=frozenset({token}))


def test_rigid_plane_normal_must_be_a_unit_vector() -> None:
    with pytest.raises(ValueError, match="unit"):
        RigidPlane((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))


def test_yield_table_columns_must_match() -> None:
    with pytest.raises(ValueError, match="yield_table"):
        MaterialInput(1, None, None, None, None, None, ((0.0, 0.5), (2.0e8,)))


def test_anchor_is_a_positive_finite_si_value_with_digits() -> None:
    anchor = UnitsAnchor("density", 8900.0, 3, "material:2:density", "handbook")
    assert anchor.digits == 3
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="si_value"):
            UnitsAnchor("density", bad, 3, "material:2:density", "handbook")
    with pytest.raises(ValueError, match="digits"):
        UnitsAnchor("density", 8900.0, 0, "material:2:density", "handbook")
    with pytest.raises(ValueError, match="input_quantity"):
        UnitsAnchor("density", 8900.0, 3, "the copper bar", "handbook")


def test_declared_facts_defaults_declare_nothing() -> None:
    declared = DeclaredFacts(unit_system=None)
    assert declared.anchors == ()
    assert declared.fields is None
    assert declared.quasi_static is False


def test_declared_unit_system_is_a_three_part_token() -> None:
    assert DeclaredFacts(unit_system="g-mm-ms").unit_system == "g-mm-ms"
    with pytest.raises(ValueError, match="unit_system"):
        DeclaredFacts(unit_system="grams, millimetres")
