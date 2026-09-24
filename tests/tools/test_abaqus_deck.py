"""Tests for the shared Abaqus keyword writers (ADR-0069). Text only."""

import abaqus_paths  # noqa: F401
import deck
import numpy as np


def test_structured_mesh_numbering_and_orientation():
    mesh = deck.structured_quad_mesh(2, 1, 0.0, 2.0, 0.0, 1.0)
    assert mesh.node_labels.tolist() == [1, 2, 3, 4, 5, 6]
    assert mesh.coords.tolist() == [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
    assert mesh.connectivity.tolist() == [[1, 2, 5, 4], [2, 3, 6, 5]]  # CCW
    assert mesh.node(2, 1) == 6 and mesh.element(1, 0) == 2
    assert mesh.centroids().tolist() == [[0.5, 0.5], [1.5, 0.5]]


def test_numbers_round_trip_and_sets_wrap_at_sixteen():
    assert deck.num(2.5e-5) == "2.5e-05" and deck.num(3000) == "3000.0"
    text = deck.node_set("ALL", range(1, 18))
    lines = text.splitlines()
    assert lines[0] == "*NSET, NSET=ALL"
    assert lines[1].count(",") == 15 and lines[2] == "17"


def test_small_deck_snapshot():
    mesh = deck.structured_quad_mesh(1, 1, 0.0, 1.0, 0.0, 2.0)
    text = (
        deck.heading("toy")
        + deck.node_block(mesh.node_labels, mesh.coords)
        + deck.element_block("CAX4R", "E", mesh.element_labels, mesh.connectivity)
        + deck.material(
            "M",
            density=7.0e-9,
            youngs=200000.0,
            poisson=0.3,
            plastic=[(250.0, 0.0), (1250.0, 5.0)],
        )
        + deck.solid_section("E", "M")
        + deck.initial_velocity("N", 2, -3.0e4)
        + deck.initial_hardening([1], np.array([0.15]))
        + deck.explicit_step(
            "S",
            1.0e-3,
            deck.standard_output(
                1.0e-5,
                node_set="N",
                node_vars=("U",),
                element_set="E",
                element_vars=("S", "PEEQ"),
                history_node_set="RP",
                history_node_vars=("RF2",),
            ),
        )
    )
    assert text == EXPECTED


EXPECTED = """\
*HEADING
toy
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 0.0, 2.0
4, 1.0, 2.0
*ELEMENT, TYPE=CAX4R, ELSET=E
1, 1, 2, 4, 3
*MATERIAL, NAME=M
*DENSITY
7e-09
*ELASTIC
200000.0, 0.3
*PLASTIC
250.0, 0.0
1250.0, 5.0
*SECTION CONTROLS, NAME=HG_ENHANCED, HOURGLASS=ENHANCED
*SOLID SECTION, ELSET=E, MATERIAL=M, CONTROLS=HG_ENHANCED
*INITIAL CONDITIONS, TYPE=VELOCITY
N, 2, -30000.0
*INITIAL CONDITIONS, TYPE=HARDENING
1, 0.15
*STEP, NAME=S, NLGEOM=YES
*DYNAMIC, EXPLICIT
, 0.001
*OUTPUT, FIELD, TIME INTERVAL=1e-05, TIME MARKS=YES
*NODE OUTPUT, NSET=N
U
*ELEMENT OUTPUT, ELSET=E
S, PEEQ
*OUTPUT, HISTORY, TIME INTERVAL=1e-05
*ENERGY OUTPUT
ALLAE, ALLCD, ALLFD, ALLIE, ALLKE, ALLPD, ALLSE, ALLVD, ALLWK, ETOTAL
*NODE OUTPUT, NSET=RP
RF2
*END STEP
"""


def test_contact_and_rigid_surface_writers():
    assert deck.element_surface("F", [("ROW", "S1"), ("COL", "S2")]) == (
        "*SURFACE, TYPE=ELEMENT, NAME=F\nROW, S1\nCOL, S2\n"
    )
    assert deck.analytical_surface_2d("W", [(-0.5, 0.0), (10.0, 0.0)]) == (
        "*SURFACE, TYPE=SEGMENTS, NAME=W\nSTART, -0.5, 0.0\nLINE, 10.0, 0.0\n"
    )
    assert deck.rigid_body(7, "W") == "*RIGID BODY, ANALYTICAL SURFACE=W, REF NODE=7\n"
    assert deck.surface_interaction("SMOOTH") == "*SURFACE INTERACTION, NAME=SMOOTH\n"
    assert deck.contact_pair("F", "W", "SMOOTH") == (
        "*CONTACT PAIR, INTERACTION=SMOOTH\nF, W\n"
    )
    assert deck.boundary("AXIS", 1) == "*BOUNDARY\nAXIS, 1, 1\n"
    assert deck.encastre("RP") == "*BOUNDARY\nRP, ENCASTRE\n"
