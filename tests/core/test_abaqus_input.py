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
    assert material.canonical_model is None


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
