"""The example problem: a short cylinder upset by a dropped rigid plate.

An axisymmetric cylinder (r along 1, z along 2) stands on a fixed rigid base
and is struck from above by a rigid plate. The plate is an analytical rigid
surface, which has no inertia of its own: its reference node carries the drop
mass as a point mass and the initial velocity. The cylinder's wall carries a
small barrel-shaped imperfection, so its quads are not rectangles; the levels
still nest, because a finer mesh places its nodes by the same function. The
wall is also in contact with itself (single-surface contact), with friction.
Everything the deck contains is written by ``structbench.datagen.abaqus.deck``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from structbench.core import Case
from structbench.datagen.abaqus import deck

ACROSS_RADIUS = 4  # elements across the radius at level 1
PLATE_REACH = 4.0  # both rigid surfaces span r in [-0.1 R, PLATE_REACH R]
HARDENING_TO = 10.0  # the yield table's last plastic strain


def mesh(params: dict[str, Any]) -> deck.QuadMesh:
    """A structured grid whose radius swells by ``imperfection`` at mid-height.

    Node ``(r, z)`` of the straight cylinder moves to ``r (1 + a sin(pi z /
    H))``: a function of position alone, so level k nests level 1 by index.
    """
    radius, height = float(params["D"]) / 2.0, float(params["H"])
    refine = int(params.get("refine", 1))
    rows = refine * round(ACROSS_RADIUS * height / radius)
    grid = deck.structured_quad_mesh(
        ACROSS_RADIUS * refine, rows, 0.0, radius, 0.0, height
    )
    r, z = grid.coords[:, 0], grid.coords[:, 1]
    swell = 1.0 + float(params["imperfection"]) * np.sin(np.pi * z / height)
    return replace(grid, coords=np.column_stack([r * swell, z]))


def feasible(params: dict[str, Any]) -> bool:
    return True


def input_deck(params: dict[str, Any], variant: str | None) -> str:
    radius, height = float(params["D"]) / 2.0, float(params["H"])
    grid = mesh(params)
    across, n_axial = grid.nx, grid.ny
    plate = int(grid.node_labels[-1]) + 1  # the plate's reference node
    base = plate + 1  # the base's reference node
    sigma0, slope = float(params["sigma0"]), float(params["hardening"])
    table = [(sigma0, 0.0), (sigma0 + slope * HARDENING_TO, HARDENING_TO)]
    interval = float(params["frame_interval"])
    reach = (-0.1 * radius, PLATE_REACH * radius)

    text = deck.heading("Abaqus example: a cylinder upset by a dropped plate")
    text += deck.node_block(grid.node_labels, grid.coords)
    text += deck.node_block([plate, base], [[0.0, height], [0.0, 0.0]])
    text += deck.element_block(
        "CAX4R", "CYLINDER", grid.element_labels, grid.connectivity
    )
    text += deck.point_mass(
        "DROP_MASS", int(grid.element_labels[-1]) + 1, plate, float(params["mass"])
    )
    text += deck.node_set("CYLINDER_NODES", grid.node_labels)
    text += deck.node_set("FIELD_NODES", [*grid.node_labels, plate])
    text += deck.node_set("AXIS", [grid.node(0, j) for j in range(n_axial + 1)])
    text += deck.node_set("PLATE_RP", [plate])
    text += deck.node_set("BASE_RP", [base])
    text += deck.element_set(
        "TOP_ROW", [grid.element(i, n_axial - 1) for i in range(across)]
    )
    text += deck.element_set("BOTTOM_ROW", [grid.element(i, 0) for i in range(across)])
    text += deck.element_set(
        "OUTER_COL", [grid.element(across - 1, j) for j in range(n_axial)]
    )
    text += deck.element_surface("TOP_SURF", [("TOP_ROW", "S3")])
    text += deck.element_surface("BOTTOM_SURF", [("BOTTOM_ROW", "S1")])
    text += deck.element_surface("WALL_SURF", [("OUTER_COL", "S2")])
    # Each analytical surface keeps the cylinder on its left as it is drawn.
    text += deck.analytical_surface_2d(
        "PLATE", [(reach[1], height), (reach[0], height)]
    )
    text += deck.rigid_body(plate, "PLATE")
    text += deck.analytical_surface_2d("BASE", [(reach[0], 0.0), (reach[1], 0.0)])
    text += deck.rigid_body(base, "BASE")
    text += deck.material(
        "CYLINDER_MATERIAL",
        density=float(params["rho"]),
        youngs=float(params["E"]),
        poisson=float(params["nu"]),
        plastic=table,
    )
    text += deck.solid_section("CYLINDER", "CYLINDER_MATERIAL")
    text += deck.surface_interaction("SMOOTH")
    text += deck.surface_interaction("ROUGH", friction=float(params["friction"]))
    text += deck.boundary("AXIS", 1)
    text += deck.encastre("BASE_RP")
    text += deck.boundary("PLATE_RP", 1)
    text += deck.boundary("PLATE_RP", 6)
    text += deck.initial_velocity("PLATE_RP", 2, -float(params["v0"]))
    body = deck.contact_pair("TOP_SURF", "PLATE", "SMOOTH")
    body += deck.contact_pair("BOTTOM_SURF", "BASE", "SMOOTH")
    body += deck.self_contact_pair("WALL_SURF", "ROUGH")
    body += deck.standard_output(
        interval,
        node_set="FIELD_NODES",
        node_vars=("U", "V", "A"),
        element_set="CYLINDER",
        element_vars=("S", "PEEQ"),
        history_node_set="BASE_RP",
        history_node_vars=("RF2",),
    )
    period = float(f"{interval * int(params['n_intervals']):.12g}")
    scale = params.get("dt_scale")
    return text + deck.explicit_step(
        "DROP", period, body, scale_factor=None if scale is None else float(scale)
    )


def qoi(case: Case) -> dict[str, float]:
    """The cylinder's height and largest radius at the last frame."""
    assert case.response is not None
    used = np.unique(case.elements["solid"].connectivity)
    x = case.nodes.coords[used] + case.response.node["displacement"][-1][used]
    return {
        "final_height": float(np.ptp(x[:, 1])),
        "final_radius": float(x[:, 0].max()),
    }
