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


# --- what the input asks the solver to write (E5, E6, E7) ---------------------

_ENERGY = "\n".join(
    [
        "*CONTROL_ENERGY",
        "$#    hgen      rwen    slnten     rylen",
        _row(2, 2, 2, 2),
    ]
)


def test_the_energy_terms_the_input_switches_on_are_read() -> None:
    facts = _read(_deck(_ENERGY))
    assert facts.energy_terms_computed == frozenset(
        {"zero_energy_mode", "rigid_surface", "contact", "damping"}
    )


def test_a_term_left_at_one_is_not_computed_and_says_so() -> None:
    """The default HGEN = 1 computes no hourglass energy at all."""
    off = "\n".join(
        [
            "*CONTROL_ENERGY",
            "$#    hgen      rwen    slnten     rylen",
            _row(1, 2, 2, 1),
        ]
    )
    facts = _read(_deck(off))
    assert facts.energy_terms_computed == frozenset({"rigid_surface", "contact"})


def test_no_energy_card_establishes_nothing_rather_than_a_default() -> None:
    """A solver default is never assumed for an absent setting (ADR-0066)."""
    assert _read(_deck()).energy_terms_computed is None


def test_the_databases_the_input_requests_are_read() -> None:
    cards = "\n".join(
        [
            "*DATABASE_GLSTAT",
            "$#      dt",
            _row(0.002),
            "*DATABASE_MATSUM",
            _row(0.002),
            "*DATABASE_RWFORC",
            _row(0.0),  # a zero interval writes nothing
            "*DATABASE_BINARY_D3PLOT",
            _row(0.002),
        ]
    )
    requested = _read(_deck(cards)).databases_requested
    assert requested is not None
    assert "DATABASE_GLSTAT" in requested and "DATABASE_MATSUM" in requested
    assert "DATABASE_RWFORC" not in requested  # dt = 0 is no output
    assert "DATABASE_BINARY_D3PLOT" in requested


def test_a_settings_database_card_is_not_an_output_request() -> None:
    """``*DATABASE_EXTENT_BINARY`` configures output; it requests none."""
    cards = "\n".join(["*DATABASE_EXTENT_BINARY", _row(0, 0, 1, 0, 0, 0, 8)])
    requested = _read(_deck(cards)).databases_requested
    assert requested == frozenset()


def test_a_deck_that_hides_its_content_establishes_no_requests() -> None:
    hidden = _read(_deck("*INCLUDE", "other.k"))
    assert hidden.databases_requested is None
    assert hidden.energy_terms_computed is None


# --- card layouts for the notch sweep's materials ------------------------------

_CONCRETE = "\n".join(
    [
        "*MAT_CONCRETE_DAMAGE_REL3_TITLE",
        "concrete",
        "$#     mid        ro        pr  ",
        _row(11, "2.40000E-6", 0.2),
        "$#      ft        a0        a1        a2        b1     omega       a1f   ",
        _row(0.0, -0.05, 0.0, 0.0, 0.82375, 0.75, 0.0),
    ]
)
_STEEL = "\n".join(
    [
        "*MAT_PLASTIC_KINEMATIC",
        "$#     mid        ro         e        pr      sigy      etan      beta    ",
        _row(12, "7.85000E-6", 200.0, 0.3, 337.0, 1.2, 0.0),
        "$#     src       srp        fs        vp  ",
        _row(40.0, 5.0, "", 0.0),
    ]
)


def _by_id(text: str, mid: int):
    facts = read_input_facts(text, source_units="kg-mm-ms")
    return facts, next(m for m in facts.materials if m.material_id == mid)


def test_the_concrete_card_gives_up_its_density_and_poisson_ratio() -> None:
    facts, concrete = _by_id(_deck(_CONCRETE), 11)
    assert facts.unparsable == frozenset()
    assert concrete.density == pytest.approx(2400.0)  # kg/mm3 -> kg/m3
    assert concrete.poisson_ratio == pytest.approx(0.2)
    assert concrete.yield_table is None  # its yield surface is not tabulated


def test_the_steel_card_gives_up_its_density_and_modulus() -> None:
    facts, steel = _by_id(_deck(_STEEL), 12)
    assert facts.unparsable == frozenset()
    assert steel.density == pytest.approx(7850.0)
    assert steel.youngs_modulus == pytest.approx(200.0e9)  # GPa in kg-mm-ms
    assert steel.poisson_ratio == pytest.approx(0.3)
    assert steel.yield_table is None  # bilinear, not a table of deck knots


def test_naming_a_class_is_not_the_same_as_knowing_what_it_means() -> None:
    """The layout says what numbers a card holds, and `canonical_model` names
    the class; only `MATERIAL_CLASSES` says what its stored state *means*.

    The two notch materials got both in ADR-0067. `*MAT_ELASTIC` still shows
    the split: the adapter maps it to `linear_elastic`, and the platform has
    no semantics for that class, so its constitutive rows stay unsupported.
    """
    from structbench.verification.materials import material_class

    facts = read_input_facts(_deck(), source_units="g-mm-ms")
    assert any(m.canonical_model == "elastic_plastic_hydro" for m in facts.materials)
    assert material_class("linear_elastic") is None


def test_neither_card_is_reported_as_an_unknown_layout() -> None:
    facts = read_input_facts(_deck(_CONCRETE, _STEEL), source_units="kg-mm-ms")
    assert not any(t.startswith("unknown_card_layout") for t in facts.unparsable)


def test_the_two_notch_cards_now_resolve_a_canonical_class() -> None:
    """ADR-0067: the layout says what the numbers are, the class what they mean."""
    _, concrete = _by_id(_deck(_CONCRETE), 11)
    _, steel = _by_id(_deck(_STEEL), 12)
    assert concrete.canonical_model == "concrete_damage"
    assert steel.canonical_model == "elastic_plastic_kinematic"
