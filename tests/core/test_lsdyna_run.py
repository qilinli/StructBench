"""Tests for the fail-closed solver-input reader (ADR-0066 clause 3).

Synthetic decks with invented numbers on the real fixed-width layout:
8 fields of 10 columns, ``$`` comments, a title line under ``*PART``.
"""

from __future__ import annotations

import pytest

from structbench.core import InputFacts, read_input_facts


def _row(*values: object) -> str:
    return "".join(f"{v!s:>10}" for v in values)


_MAT = "\n".join(
    [
        "*MAT_ELASTIC_PLASTIC_HYDRO",
        "$#     mid        ro         g      sigy        eh        pc        fs",
        _row(2, 0.0027, 26000.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        _row(0.0, 0.5, 1.0, 2.0, "", "", "", ""),
        _row("", "", "", "", "", "", "", ""),
        _row(100.0, 140.0, 139.0, 180.0, "", "", "", ""),
        _row("", "", "", "", "", "", "", ""),
    ]
)


def _deck(*extra: str, termination: str | None = None, idim: int = 2) -> str:
    return "\n".join(
        [
            "*KEYWORD",
            "*TITLE",
            "an invented particle impact",
            "*CONTROL_SPH",
            "$#    ncbs     boxid        dt      idim   nmneigh      form",
            # fields abut with no space, as real decks do
            f"         1         01.00000E20{idim:>10}       150         0",
            "*CONTROL_TERMINATION",
            "$#  endtim    endcyc     dtmin    endeng    endmas",
            termination or _row(0.25, 0, 0.0, 0.0, "1.000000E8"),
            "*PART",
            "the bar",
            "$#     pid     secid       mid     eosid      hgid",
            _row(1, 7, 2, 0, 0),
            "*SECTION_SPH",
            "$#   secid      cslh      hmin      hmax",
            _row(7, 1.2, 0.8, 1.2),
            _MAT,
            "*RIGIDWALL_PLANAR_ID",
            _row(1, "the wall"),
            _row(0, 0, 0, 0.0),
            "$#      xt        yt        zt        xh        yh        zh      fric",
            _row(-1.5, 0.0, 0.0, 3.5, 0.0, 0.0, 0.0),
            *extra,
            "*END",
        ]
    )


def _read(text: str) -> InputFacts:
    return read_input_facts(text, source_units="g-mm-ms")


def test_a_particle_impact_deck_in_si() -> None:
    facts = _read(_deck())
    assert facts.unparsable == frozenset()
    (part,) = facts.parts
    assert (part.part_id, part.material_id, part.discretisation) == (1, 2, "particle")
    assert part.under_integrated is None
    (material,) = facts.materials
    assert material.canonical_model == "elastic_plastic_hydro"
    assert material.density == pytest.approx(2700.0)  # g/mm^3 -> kg/m^3
    assert material.shear_modulus == pytest.approx(2.6e10)  # MPa -> Pa
    assert facts.end_time == pytest.approx(2.5e-4)  # ms -> s
    assert facts.time_integration == "explicit"
    assert (facts.dimension, facts.plane_strain) == (2, True)
    assert facts.smoothing_length_scale_bounds == (0.8, 1.2)
    (wall,) = facts.rigid_planes
    assert wall.point == pytest.approx((-1.5e-3, 0.0, 0.0))
    assert wall.normal == pytest.approx((1.0, 0.0, 0.0))
    assert facts.other_termination_criteria == frozenset()


def test_features_that_must_be_requested_are_off_when_nothing_asks_for_them() -> None:
    facts = _read(_deck())
    assert facts.mass_scaling_enabled is False
    assert facts.erosion_enabled is False
    assert facts.contact_defined is False
    assert facts.prescribed_motion_defined is False


def test_blank_cards_are_kept_so_the_hardening_table_is_not_shifted() -> None:
    (material,) = _read(_deck()).materials
    assert material.yield_table == (
        (0.0, 0.5, 1.0, 2.0),
        (100.0e6, 140.0e6, 139.0e6, 180.0e6),  # the non-monotone knot is kept
    )


def test_axisymmetric_is_not_plane_strain() -> None:
    facts = _read(_deck(idim=-2))
    assert (facts.dimension, facts.plane_strain) == (2, False)


def test_requests_are_read() -> None:
    facts = _read(
        _deck(
            "*CONTROL_TIMESTEP",
            _row(0.0, 0.9, 0, 0.0, -1.0e-6),
            "*MAT_ADD_EROSION",
            _row(2),
            "*CONTACT_AUTOMATIC_NODES_TO_SURFACE",
            _row(1, 2),
            "*BOUNDARY_PRESCRIBED_MOTION_SET",
            _row(1, 1, 0, 3),
            termination=_row(0.25, 5000, 0.01, 0.0, 0.0),
        )
    )
    assert facts.mass_scaling_enabled is True
    assert facts.erosion_enabled is True
    assert facts.contact_defined is True
    assert facts.prescribed_motion_defined is True
    assert facts.other_termination_criteria == {"step_limit", "min_timestep"}


def test_an_include_may_hide_a_request_so_absence_proves_nothing() -> None:
    facts = _read(_deck("*INCLUDE", "mesh.k"))
    assert "include" in facts.unparsable
    assert facts.mass_scaling_enabled is None
    assert facts.contact_defined is None
    assert facts.time_integration is None
    assert facts.end_time == pytest.approx(2.5e-4)  # what was read stays read


def test_a_parameter_reference_is_not_resolved() -> None:
    facts = _read(_deck(termination="      &end         0       0.0       0.0"))
    assert "parameter" in facts.unparsable
    assert facts.end_time is None


def test_a_free_format_row_is_refused() -> None:
    facts = _read(_deck(termination="0.25, 0, 0.0"))
    assert "free_format" in facts.unparsable
    assert facts.end_time is None


def test_two_parts_and_a_material_the_reader_does_not_know() -> None:
    facts = _read(
        _deck(
            "*PART",
            "the striker",
            _row(2, 7, 9, 0, 0),
            "*MAT_SOME_FUTURE_MODEL_TITLE",
            "striker steel",
            _row(9, 0.0078, 200000.0),
        )
    )
    assert [(p.part_id, p.material_id) for p in facts.parts] == [(1, 2), (2, 9)]
    unknown = next(m for m in facts.materials if m.material_id == 9)
    assert unknown.canonical_model is None
    assert unknown.density is None  # an unknown layout is not guessed at
    assert "unknown_card_layout:MAT_SOME_FUTURE_MODEL" in facts.unparsable


def test_an_implicit_request_is_read() -> None:
    facts = _read(_deck("*CONTROL_IMPLICIT_GENERAL", _row(1, 0.01)))
    assert facts.time_integration == "implicit"


def test_a_missing_card_with_a_default_stays_unknown() -> None:
    text = _deck().replace("*CONTROL_SPH", "*CONTROL_SOMETHING_ELSE")
    facts = _read(text)
    assert facts.dimension is None
    assert facts.plane_strain is None
