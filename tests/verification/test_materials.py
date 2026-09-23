"""Tests for material-class semantics (ADR-0066 clause 6)."""

from __future__ import annotations

from structbench.verification.materials import MATERIAL_CLASSES, material_class


def test_tabulated_j2_metal_has_a_monotone_plastic_strain_and_a_yield_law() -> None:
    cls = material_class("elastic_plastic_hydro")
    assert cls is not None
    assert cls.state_variable == "plastic_strain"
    assert cls.state_bounds == (0.0, None)
    assert cls.monotone
    assert cls.yield_law == "tabulated_j2"
    assert cls.has_equation_of_state


def test_non_structural_classes_have_no_state_and_no_yield_law() -> None:
    for name in ("null", "rigid"):
        cls = material_class(name)
        assert cls is not None
        assert cls.state_variable == "none"
        assert cls.yield_law == "none"
        assert not cls.structural


def test_an_unmapped_or_unknown_class_is_unsupported() -> None:
    assert material_class(None) is None
    assert material_class("some_future_model") is None


def test_keys_match_the_records() -> None:
    assert all(name == cls.canonical_model for name, cls in MATERIAL_CLASSES.items())


def test_concrete_damage_carries_a_bounded_monotone_damage_measure() -> None:
    """ADR-0067: the K&C slot holds a scaled damage measure on 0..2.

    Measured across 22 notch cases: the range is exactly [0, 2] and the
    number of decreasing samples is zero in ~66 million.
    """
    cls = material_class("concrete_damage")
    assert cls is not None
    assert cls.state_variable == "damage"
    assert cls.state_bounds == (0.0, 2.0)
    assert cls.monotone
    assert cls.structural
    assert not cls.has_equation_of_state


def test_neither_new_class_offers_a_yield_law_that_can_be_assessed() -> None:
    """ADR-0067: a surface exists for each, and its arguments are not exported.

    K&C's is pressure-dependent and generated internally from f'c, so its
    coefficients never appear in the input; the steel's is bilinear, stated
    by `sigy` and `etan` rather than by knots.
    """
    for name in ("concrete_damage", "elastic_plastic_kinematic"):
        cls = material_class(name)
        assert cls is not None, name
        assert cls.yield_law == "not_assessable", name


def test_bilinear_steel_keeps_an_unbounded_monotone_plastic_strain() -> None:
    cls = material_class("elastic_plastic_kinematic")
    assert cls is not None
    assert cls.state_variable == "plastic_strain"
    assert cls.state_bounds == (0.0, None)
    assert cls.monotone and cls.structural
    assert not cls.has_equation_of_state
