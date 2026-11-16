"""A rigid body whose reference node carries a point mass, deck to canonical case.

The fixture ``fixtures/drop_mass_export.{inp,npz}`` is a real run: one pilot
case of the shipped example ``abaqus_drop_mass`` (a CAX4R cylinder upset
between a fixed analytical rigid base and a dropped analytical rigid plate
whose reference node carries a 4 kg ``MASS`` element at 5 m/s) solved by
Abaqus/Explicit 2025 and exported by ``odb_export.py``.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from structbench.core.io.abaqus import abaqus_export_to_case  # noqa: E402
from structbench.core.io.abaqus_run import (  # noqa: E402
    abaqus_point_masses,
    read_abaqus_input_facts,
)
from structbench.datagen import definition, sampling  # noqa: E402
from structbench.datagen.abaqus import deck  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLE = resources.files("structbench.datagen") / "examples" / "abaqus_drop_mass"
DECK = (FIXTURES / "drop_mass_export.inp").read_text(encoding="utf-8")
PLATE = 46  # the plate's reference node, the label after the cylinder's 45 nodes


def _case():
    return abaqus_export_to_case(
        FIXTURES / "drop_mass_export.npz",
        DECK,
        source_units="t-mm-s",
        dimension=2,
        case_id="drop",
    )


# --- the deck helpers ----------------------------------------------------------


def test_the_point_mass_is_an_element_and_its_mass_card():
    assert deck.point_mass("DROP_MASS", 33, 46, 0.004) == (
        "*ELEMENT, TYPE=MASS, ELSET=DROP_MASS\n33, 46\n*MASS, ELSET=DROP_MASS\n0.004\n"
    )


def test_a_single_surface_pair_names_one_surface():
    assert deck.self_contact_pair("WALL", "ROUGH") == (
        "*CONTACT PAIR, INTERACTION=ROUGH\nWALL\n"
    )
    assert deck.self_contact_pair("WALL", "ROUGH", mechanical_constraint="PENALTY") == (
        "*CONTACT PAIR, INTERACTION=ROUGH, MECHANICAL CONSTRAINT=PENALTY\nWALL\n"
    )


def test_friction_is_a_card_under_the_interaction_and_absent_without_it():
    assert deck.surface_interaction("SMOOTH") == "*SURFACE INTERACTION, NAME=SMOOTH\n"
    assert deck.surface_interaction("ROUGH", friction=0.1) == (
        "*SURFACE INTERACTION, NAME=ROUGH\n*FRICTION\n0.1\n"
    )


def test_the_drop_mass_example_writes_the_fixtures_deck():
    with resources.as_file(EXAMPLE) as folder:
        d, problem = definition.load_definition(folder), definition.load_problem(folder)
    point = sampling.sample_split(d.variables, d.regions, d.split("pilot"))[1]
    params = {**d.fixed, **point.params, d.levels.refine_key: d.levels.production}
    text = problem.input_deck(params, None)
    assert (
        hashlib.sha256(text.encode()).digest() == hashlib.sha256(DECK.encode()).digest()
    )
    # The mesh is not made of rectangles, and a finer level contains it.
    coarse, fine = problem.mesh(params), problem.mesh({**params, "refine": 2})
    assert np.ptp(coarse.coords[coarse.connectivity - 1][..., 0], axis=1).min() > 0
    wall = coarse.coords[[coarse.node(coarse.nx, j) - 1 for j in range(coarse.ny + 1)]]
    assert np.ptp(wall[:, 0]) > 0.01  # the wall is not straight
    every_other = np.array(
        [
            fine.node(2 * i, 2 * j) - 1
            for j in range(coarse.ny + 1)
            for i in range(coarse.nx + 1)
        ]
    )
    np.testing.assert_allclose(fine.coords[every_other], coarse.coords, atol=1e-12)


# --- the reader ----------------------------------------------------------------


def test_the_reader_does_not_take_the_point_mass_for_the_parts_kind():
    facts = read_abaqus_input_facts(DECK, source_units="t-mm-s")
    # The mass element stands after the continuum block in this deck: it used
    # to decide the part's kind and to hide the formulation.
    assert [(p.discretisation, p.under_integrated) for p in facts.parts] == [
        ("solid", True)
    ]
    assert facts.plane_strain is False
    assert not any("MASS" in token for token in facts.unparsable)
    # Order does not matter: the same deck with the mass element first.
    start, stop = DECK.index("*ELEMENT, TYPE=MASS"), DECK.index("*NSET")
    block = DECK[start:stop]
    moved = DECK.replace(block, "").replace(
        "*ELEMENT, TYPE=CAX4R", block + "*ELEMENT, TYPE=CAX4R"
    )
    assert read_abaqus_input_facts(moved, source_units="t-mm-s") == facts


def test_the_mass_is_read_from_the_input_in_kilograms_by_node_label():
    assert abaqus_point_masses(DECK, source_units="t-mm-s") == {PLATE: 4.0}
    assert abaqus_point_masses("*NODE\n1, 0.0, 0.0\n", source_units="t-mm-s") == {}
    two = (
        "*ELEMENT, TYPE=MASS, ELSET=A\n7, 3\n8, 4\n*MASS, ELSET=A\n0.5\n"
        "*ELEMENT, TYPE=MASS, ELSET=B\n9, 3\n*MASS, ELSET=B\n0.25\n"
    )
    assert abaqus_point_masses(two, source_units="kg-m-s") == {3: 0.75, 4: 0.5}


@pytest.mark.parametrize(
    "text",
    [
        "*ELEMENT, TYPE=MASS, ELSET=A\n7, 3\n",  # no *MASS card
        "*ELEMENT, TYPE=MASS, ELSET=A\n7, 3\n*MASS, ELSET=A, ALPHA=0.1\n0.5\n",
        "*ELEMENT, TYPE=MASS, ELSET=A\n7, 3\n*MASS, ELSET=A\n0.5, 0.5, 0.5\n",
        "*ELEMENT, TYPE=MASS\n7, 3\n*MASS, ELSET=A\n0.5\n",  # no element set
    ],
)
def test_a_mass_that_cannot_be_read_is_refused_not_guessed(text):
    with pytest.raises(ValueError):
        abaqus_point_masses(text, source_units="t-mm-s")


# --- the adapter ---------------------------------------------------------------


def test_the_real_export_converts_with_the_point_element_left_out():
    manifest_types = {
        key.split("/")[3]
        for key in np.load(FIXTURES / "drop_mass_export.npz").files
        if key.startswith("mesh/PART-1-1/elements/")
    }
    assert manifest_types == {"CAX4R", "MASS"}  # what used to be refused
    case = _case()
    assert list(case.elements) == ["solid"]
    assert case.elements["solid"].connectivity.shape == (32, 4)
    assert sorted(case.response.element["solid"]) == [
        "effective_plastic_strain",
        "stress",
    ]
    # No duplicate end frame in this model: 181 frames for 180 intervals.
    assert len(case.response.time) == 181
    # The plate's reference node was in the field output: a node no element uses.
    assert case.nodes.node_id.tolist() == list(range(1, 47))
    assert PLATE - 1 not in np.unique(case.elements["solid"].connectivity)
    # The base's reference node wrote only its reaction: a global.
    assert "reaction_force_2_reference_node" in case.response.globals_
    assert json.loads(json.dumps(sorted(case.response.node))) == [
        "acceleration",
        "displacement",
        "velocity",
    ]


def test_the_mass_read_from_the_case_gives_the_force_and_the_energy():
    case = _case()
    masses = abaqus_point_masses(
        case.metadata.source_deck, source_units=case.metadata.source_units
    )
    ((node, mass),) = masses.items()
    at = int(np.flatnonzero(case.nodes.node_id == node)[0])
    velocity = case.response.node["velocity"][:, at, 1].astype(np.float64)
    acceleration = case.response.node["acceleration"][:, at, 1].astype(np.float64)
    # The mass is that of the whole revolution, as the energies are.
    kinetic = case.response.globals_["kinetic_energy"].astype(np.float64)
    assert velocity[0] == pytest.approx(-5.0)
    assert 0.5 * mass * velocity[0] ** 2 == pytest.approx(kinetic[0], rel=1e-6)
    # Mass times the plate's deceleration is the force the base reacts.
    reaction = case.response.globals_["reaction_force_2_reference_node"]
    peak_inertial = np.abs(mass * acceleration).max()
    assert peak_inertial == pytest.approx(np.abs(reaction).max(), rel=5e-3)


def test_two_continuum_types_in_one_instance_are_still_refused(tmp_path):
    arrays = dict(np.load(FIXTURES / "drop_mass_export.npz"))
    arrays["mesh/PART-1-1/elements/CAX3/labels"] = np.array([99])
    arrays["mesh/PART-1-1/elements/CAX3/connectivity"] = np.array([[1, 2, 3]])
    path = tmp_path / "two.npz"
    np.savez(path, **arrays)
    with pytest.raises(NotImplementedError, match="several element types"):
        abaqus_export_to_case(
            path, DECK, source_units="t-mm-s", dimension=2, case_id="two"
        )
