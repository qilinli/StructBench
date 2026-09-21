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
