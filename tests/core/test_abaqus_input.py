"""Tests for the Abaqus input reader (ADR-0068 step 7).

Synthetic decks on the observed layout of an Abaqus/Standard `.inp`. The
reader implements only what a real deck was seen to contain and fails closed
on everything else, per ADR-0066 clause 3 ("a record field is implemented
when a run first supplies the evidence for it, never before").

The id contract under test: Abaqus names its parts and materials, while
`PartTraits.part_id` and `MaterialInput.material_id` are integers, so the
integers are MINTED here. This reader is the single authority for that
minting — the conversion glue maps response entities onto these ids by name,
and the `.odb` extractor emits names, never ids. Nothing has to agree across
two processes by luck.
"""

from __future__ import annotations

import pytest

from structbench.core.io.abaqus_run import mint_ids, read_abaqus_input_facts

_DECK = "\n".join(
    [
        "*Heading",
        "** Job name: Cantilever Model name: Model-1",
        "*Part, name=Beam",
        "*Node",
        "      1,           0.,           0.,           0.",
        "      2,         200.,           0.,           0.",
        "*Element, type=C3D8I",
        " 1,  1,  2,  3,  4,  5,  6,  7,  8",
        "*Solid Section, elset=Set-1, material=Steel",
        ",",
        "*End Part",
        "*Assembly, name=Assembly",
        "*Instance, name=Beam-1, part=Beam",
        "*End Instance",
        "*End Assembly",
        "*Material, name=Steel",
        "*Elastic",
        "210000., 0.3",
        "*Step, name=Load, nlgeom=NO",
        "*Static",
        "1., 1., 1e-05, 1.",
        "*Boundary",
        "FixedEnd, ENCASTRE",
        "*Dsload",
        "Top, P, 1.",
        "*Output, field, variable=PRESELECT",
        "*Output, history, variable=PRESELECT",
        "*End Step",
    ]
)


def _read(deck: str = _DECK):
    return read_abaqus_input_facts(deck, source_units="t-mm-s")


def test_the_reader_names_itself() -> None:
    assert _read().solver == "abaqus"


def test_dimension_comes_from_the_stored_coordinates_not_an_element_table() -> None:
    """Counting the columns of a `*Node` line needs no keyword knowledge."""
    assert _read().dimension == 3


def test_a_static_step_is_implicit() -> None:
    assert _read().time_integration == "implicit"


def test_an_explicit_dynamic_step_is_explicit() -> None:
    deck = _DECK.replace("*Static\n1., 1., 1e-05, 1.", "*Dynamic, Explicit\n, 2e-3")
    assert _read(deck).time_integration == "explicit"


def test_a_static_step_establishes_no_end_time() -> None:
    """A `*Static` step's time period is a dimensionless load parameter.

    Converting it to seconds would be a category error, so the field stays
    unset rather than carrying a number that means something else.
    """
    assert _read().end_time is None


def test_the_elastic_constants_are_read_in_si() -> None:
    (material,) = _read().materials
    assert material.youngs_modulus == pytest.approx(210.0e9)  # 210000 MPa
    assert material.poisson_ratio == pytest.approx(0.3)
    assert material.density is None  # the deck states none
    assert material.yield_table is None


def test_an_elastic_only_material_is_linear_elastic() -> None:
    (material,) = _read().materials
    assert material.canonical_model == "linear_elastic"


def test_adding_plasticity_withdraws_the_linear_elastic_claim() -> None:
    deck = _DECK.replace(
        "*Elastic\n210000., 0.3", "*Elastic\n210000., 0.3\n*Plastic\n250., 0."
    )
    (material,) = _read(deck).materials
    assert material.canonical_model == "elastic_plastic_isotropic"  # ADR-0070


def test_ids_are_minted_by_order_of_first_appearance() -> None:
    facts = _read()
    (part,) = facts.parts
    (material,) = facts.materials
    assert (part.part_id, part.material_id) == (1, 1)
    assert material.material_id == 1
    assert part.discretisation == "solid"
    assert part.under_integrated is None  # not established from the element code


def test_the_minting_rule_is_exported_for_the_conversion_glue() -> None:
    """One authority, so the response side never mints independently."""
    assert mint_ids(["Beam", "Plate"]) == {"Beam": 1, "Plate": 2}
    assert mint_ids(["beam"]) == {"beam": 1}


def test_an_unknown_element_type_is_refused_rather_than_classified() -> None:
    deck = _DECK.replace("type=C3D8I", "type=ZZ99")
    facts = _read(deck)
    assert "unknown_element_type:ZZ99" in facts.unparsable
    (part,) = facts.parts
    assert part.discretisation == "unknown"


def test_an_include_hides_the_deck_so_nothing_is_asserted_absent() -> None:
    deck = _DECK + "\n*Include, input=other.inp"
    facts = _read(deck)
    assert "include" in facts.unparsable
    assert facts.contact_defined is None  # not False: the deck hides content
    assert facts.damping_defined is None


def test_a_plain_boundary_is_not_prescribed_motion() -> None:
    assert _read().prescribed_motion_defined is False


def test_a_typed_boundary_is_prescribed_motion() -> None:
    deck = _DECK.replace(
        "*Boundary\nFixedEnd, ENCASTRE", "*Boundary, type=VELOCITY\nTip, 1, 1, 5."
    )
    assert _read(deck).prescribed_motion_defined is True


def test_the_output_vocabulary_is_not_yet_defined_for_abaqus() -> None:
    """ADR-0068 clause 8 defers it until the claim dossier exists."""
    facts = _read()
    assert facts.databases_requested is None
    assert facts.energy_terms_computed is None


# --- one deck, several blocks: each fact belongs to its own block -------------

_TWO_MATERIALS = "\n".join(
    [
        "*Heading",
        "*Part, name=Beam",
        "*Node",
        "      1,   0.,   0.,   0.",
        "*Element, type=C3D8I",
        " 1, 1,2,3,4,5,6,7,8",
        "*Solid Section, elset=S1, material=Steel",
        ",",
        "*End Part",
        "*Material, name=Steel",
        "*Elastic",
        "210000., 0.3",
        "*Material, name=Rubber",
        "*Elastic",
        "5., 0.49",
        "*Step, name=s",
        "*Static",
        "1., 1.",
        "*End Step",
    ]
)


def test_each_material_keeps_its_own_constants() -> None:
    """A deck-wide scalar reported Steel at Rubber's stiffness, silently."""
    by_id = {m.material_id: m for m in _read(_TWO_MATERIALS).materials}
    assert len(by_id) == 2
    assert by_id[1].youngs_modulus == pytest.approx(210.0e9)  # Steel
    assert by_id[2].youngs_modulus == pytest.approx(5.0e6)  # Rubber
    assert by_id[1].poisson_ratio == pytest.approx(0.3)
    assert by_id[2].poisson_ratio == pytest.approx(0.49)


def test_plasticity_in_one_material_does_not_touch_another() -> None:
    deck = _TWO_MATERIALS.replace(
        "*Material, name=Rubber\n*Elastic\n5., 0.49",
        "*Material, name=Rubber\n*Elastic\n5., 0.49\n*Plastic\n1., 0.",
    )
    by_id = {m.material_id: m for m in _read(deck).materials}
    assert by_id[1].canonical_model == "linear_elastic"  # Steel, untouched
    assert by_id[2].canonical_model == "elastic_plastic_isotropic"  # Rubber


def test_a_material_card_the_reader_skips_is_recorded_not_ignored() -> None:
    """`unstated_or_unread` and `input_gap` key ownership off `unparsable`.

    With the token absent, a deck that tabulates three yield knots was
    reported as stating no yield stress at all.
    """
    deck = _TWO_MATERIALS.replace(
        "*Material, name=Steel\n*Elastic\n210000., 0.3",
        "*Material, name=Steel\n*Conductivity\n45.\n*Elastic\n210000., 0.3"
        "\n*Plastic, hardening=KINEMATIC\n250., 0.",
    )
    facts = _read(deck)
    assert "unread_card:PLASTIC" in facts.unparsable
    assert "unread_card:CONDUCTIVITY" in facts.unparsable
    by_id = {m.material_id: m for m in facts.materials}
    assert by_id[1].canonical_model is None  # kinematic is not ADR-0070's class


def test_a_typed_elastic_card_is_not_read_positionally() -> None:
    """`type=ENGINEERING CONSTANTS` put a 9000 MPa stiffness in poisson_ratio."""
    deck = _TWO_MATERIALS.replace(
        "*Material, name=Rubber\n*Elastic\n5., 0.49",
        "*Material, name=Rubber\n*Elastic, type=ENGINEERING CONSTANTS\n"
        "150000., 9000., 9000., 0.3, 0.3, 0.4, 5000., 5000.",
    )
    by_id = {m.material_id: m for m in _read(deck).materials}
    assert by_id[2].poisson_ratio is None
    assert by_id[2].youngs_modulus is None
    assert by_id[2].canonical_model is None
    assert "unknown_card_layout:ELASTIC" in _read(deck).unparsable


def test_an_element_code_with_odd_characters_does_not_raise() -> None:
    """A token is validated, and an element code is a raw option value."""
    for bad in ("", '"S4R"', "A/B"):
        facts = _read(_TWO_MATERIALS.replace("type=C3D8I", f"type={bad}"))
        assert any(t.startswith("unknown_element_type") for t in facts.unparsable), bad


def test_a_node_lookahead_that_lands_on_a_keyword_is_not_read_as_coordinates() -> None:
    deck = _TWO_MATERIALS.replace(
        "*Node\n      1,   0.,   0.,   0.", "*Node, input=nodes.inp\n*Nset, nset=All"
    )
    facts = _read(deck)
    assert facts.dimension is None  # not 2, which the *Nset line would have given
    assert "include" in facts.unparsable  # `input=` hides content, like *Include


# --- 2D decks as the Abaqus pipeline writes them (plan 2, Task 4) -----------

_FLAT_2D = """*HEADING
toy
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 0.0, 1.0
4, 1.0, 1.0
*ELEMENT, TYPE=CAX4R, ELSET=E
1, 1, 2, 4, 3
*NSET, NSET=ALLN
1, 2, 3, 4
*MATERIAL, NAME=M
*DENSITY
7e-09
*ELASTIC
200000.0, 0.3
*PLASTIC
250.0, 0.0
1250.0, 10.0
*SECTION CONTROLS, NAME=HG_ENHANCED, HOURGLASS=ENHANCED
*SOLID SECTION, ELSET=E, MATERIAL=M, CONTROLS=HG_ENHANCED
*INITIAL CONDITIONS, TYPE=VELOCITY
ALLN, 2, -30000.0
*INITIAL CONDITIONS, TYPE=HARDENING
1, 0.15
*STEP, NAME=S, NLGEOM=YES
*DYNAMIC, EXPLICIT
, 0.001
*END STEP
"""


def test_flat_axisymmetric_deck():
    f = read_abaqus_input_facts(_FLAT_2D, source_units="t-mm-s")
    assert f.dimension == 2 and f.plane_strain is False
    assert f.time_integration == "explicit" and f.end_time == pytest.approx(0.001)
    (part,) = f.parts
    assert (part.discretisation, part.under_integrated) == ("solid", True)
    (m,) = f.materials
    assert m.canonical_model == "elastic_plastic_isotropic"
    assert m.density == pytest.approx(7000.0)  # t/mm^3 -> kg/m^3
    assert m.yield_table == ((0.0, 10.0), pytest.approx((250e6, 1250e6)))
    assert f.initial_velocity == ((frozenset({1, 2, 3, 4}), 2, pytest.approx(-30.0)),)
    assert f.initial_hardening == ((1, pytest.approx(0.15)),)
    assert not {t for t in f.unparsable if t.startswith("unread_card")}


def test_single_row_plastic_is_flat_and_cpe_is_plane_strain():
    deck = _FLAT_2D.replace("CAX4R", "CPE4R").replace("1250.0, 10.0\n", "")
    f = read_abaqus_input_facts(deck, source_units="t-mm-s")
    assert f.plane_strain is True
    assert f.materials[0].yield_table == ((0.0, 1.0), pytest.approx((250e6, 250e6)))
    assert f.parts[0].under_integrated is None  # only CAX4R is established


def test_a_deck_stating_no_initial_conditions_states_none() -> None:
    """`()` is the input's positive claim of none; `None` is "not established"."""
    deck = _FLAT_2D.split("*INITIAL CONDITIONS")[0] + "*STEP, NAME=S\n*END STEP\n"
    f = read_abaqus_input_facts(deck, source_units="t-mm-s")
    assert (f.initial_velocity, f.initial_hardening) == ((), ())


def test_a_generated_node_set_leaves_the_initial_velocity_unestablished() -> None:
    """Review Focus 3: an unresolved target is not an empty target."""
    deck = _FLAT_2D.replace(
        "*NSET, NSET=ALLN\n1, 2, 3, 4", "*NSET, NSET=ALLN, GENERATE\n1, 4, 1"
    )
    f = read_abaqus_input_facts(deck, source_units="t-mm-s")
    assert f.initial_velocity is None
    assert {
        "unread_card:NSET_GENERATE",
        "unresolved_initial_condition_target",
    } <= f.unparsable


def test_hardening_on_an_element_set_is_refused_by_name() -> None:
    deck = _FLAT_2D.replace("TYPE=HARDENING\n1, 0.15", "TYPE=HARDENING\nE, 0.15")
    f = read_abaqus_input_facts(deck, source_units="t-mm-s")
    assert f.initial_hardening is None
    assert "unread_card:HARDENING_ELSET" in f.unparsable


def test_a_rate_or_temperature_column_is_not_a_strain_only_table() -> None:
    """A third column makes the yield stress depend on more than PEEQ."""
    deck = _FLAT_2D.replace(
        "250.0, 0.0\n1250.0, 10.0", "250.0, 0.0, 20.0\n1250.0, 10.0, 20.0"
    )
    (m,) = read_abaqus_input_facts(deck, source_units="t-mm-s").materials
    assert m.yield_table is None and m.canonical_model is None


def test_two_steps_establish_no_single_end_time() -> None:
    deck = _FLAT_2D + "*STEP, NAME=T\n*DYNAMIC, EXPLICIT\n, 0.002\n*END STEP\n"
    assert read_abaqus_input_facts(deck, source_units="t-mm-s").end_time is None


def test_a_flat_deck_with_the_section_after_the_material_has_one_part() -> None:
    f = read_abaqus_input_facts(_FLAT_2D, source_units="t-mm-s")
    assert [(p.part_id, p.material_id) for p in f.parts] == [(1, 1)]
    assert "unresolved_section_material" not in f.unparsable


def test_a_rate_dependent_suboption_withdraws_the_isotropic_class() -> None:
    deck = _FLAT_2D.replace(
        "1250.0, 10.0\n", "1250.0, 10.0\n*RATE DEPENDENT\n40.0, 5.0\n"
    )
    (m,) = read_abaqus_input_facts(deck, source_units="t-mm-s").materials
    assert m.canonical_model is None
