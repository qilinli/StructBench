"""The example problem: an axisymmetric rod striking a frictionless rigid wall.

The four hooks the pipeline calls are ``input_deck``, ``mesh``, ``qoi`` and
(optionally) ``feasible``. Everything the deck contains is written by
``structbench.datagen.abaqus.deck``; nothing is edited as text. Geometry: r
along 1, z along 2; the impact face is z = 0 and the rod moves in -z.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from structbench.core import Case
from structbench.datagen.abaqus import deck

ACROSS_RADIUS = 12  # elements across the radius at level 1
WALL_REACH = 10.0  # the wall spans r in [-0.1 R, WALL_REACH R]
HARDENING_TO = 10.0  # the yield table's last plastic strain


def mesh(params: dict[str, Any]) -> deck.QuadMesh:
    """Square elements of edge R / (12 * refine); level k nests level 1 exactly."""
    length, radius = float(params["L"]), float(params["D"]) / 2.0
    refine = int(params.get("refine", 1))
    rows = refine * round(ACROSS_RADIUS * length / radius)
    return deck.structured_quad_mesh(
        ACROSS_RADIUS * refine, rows, 0.0, radius, 0.0, length
    )


def feasible(params: dict[str, Any]) -> bool:
    """No limit in the example; a dataset states its own here (see the design)."""
    return True


def input_deck(params: dict[str, Any], variant: str | None) -> str:
    radius = float(params["D"]) / 2.0
    grid = mesh(params)
    across, n_axial = grid.nx, grid.ny
    rp = int(grid.node_labels[-1]) + 1
    sigma0, slope = float(params["sigma0"]), float(params["H"])
    table = [(sigma0, 0.0), (sigma0 + slope * HARDENING_TO, HARDENING_TO)]
    interval = float(params["frame_interval"])

    text = deck.heading("Abaqus conformance: rod on a rigid wall")
    text += deck.node_block(grid.node_labels, grid.coords)
    text += deck.node_block([rp], [[0.0, -radius]])
    text += deck.element_block("CAX4R", "ROD", grid.element_labels, grid.connectivity)
    text += deck.node_set("ROD_NODES", grid.node_labels)
    text += deck.node_set("AXIS", [grid.node(0, j) for j in range(n_axial + 1)])
    text += deck.node_set("RP", [rp])
    text += deck.element_set("IMPACT_ROW", [grid.element(i, 0) for i in range(across)])
    text += deck.element_set(
        "OUTER_COL", [grid.element(across - 1, j) for j in range(n_axial)]
    )
    faces = [("IMPACT_ROW", "S1"), ("OUTER_COL", "S2")]
    text += deck.element_surface("ROD_SURF", faces)
    text += deck.analytical_surface_2d(
        "WALL", [(-0.1 * radius, 0.0), (WALL_REACH * radius, 0.0)]
    )
    text += deck.rigid_body(rp, "WALL")
    text += deck.material(
        "ROD_MATERIAL",
        density=float(params["rho"]),
        youngs=float(params["E"]),
        poisson=float(params["nu"]),
        plastic=table,
    )
    text += deck.solid_section("ROD", "ROD_MATERIAL")
    text += deck.surface_interaction("SMOOTH")
    text += deck.boundary("AXIS", 1)
    text += deck.encastre("RP")
    text += deck.initial_velocity("ROD_NODES", 2, -float(params["v0"]))
    body = deck.contact_pair("ROD_SURF", "WALL", "SMOOTH")
    body += deck.standard_output(
        interval,
        node_set="ROD_NODES",
        node_vars=("U", "V", "A"),
        element_set="ROD",
        element_vars=("S", "PEEQ"),
        history_node_set="RP",
        history_node_vars=("RF2",),
    )
    period = float(f"{interval * int(params['n_intervals']):.12g}")
    scale = params.get("dt_scale")
    return text + deck.explicit_step(
        "IMPACT", period, body, scale_factor=None if scale is None else float(scale)
    )


def qoi(case: Case) -> dict[str, float]:
    """Final length and impact-face radius at the last frame, in the case's units."""
    assert case.response is not None
    x0 = case.nodes.coords
    used = np.unique(case.elements["solid"].connectivity)
    x = x0[used] + case.response.node["displacement"][-1][used]
    face = np.isclose(x0[used, 1], x0[used, 1].min())
    return {
        "final_length": float(np.ptp(x[:, 1])),
        "face_radius": float(x[face, 0].max()),
    }
