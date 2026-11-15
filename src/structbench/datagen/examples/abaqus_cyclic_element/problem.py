"""The example problem: a block of elements under one cycle of axial strain.

An axisymmetric block (r along 1, z along 2) of a material with combined
hardening: its base is held in z, its axis in r, and its top face is driven
up, down and back by a prescribed displacement, so the stress state is
uniaxial and uniform. Each case may start from a non-zero equivalent plastic
strain and backstress. Everything the deck contains is written by
``structbench.datagen.abaqus.deck``; nothing is edited as text.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from structbench.core import Case
from structbench.datagen.abaqus import deck


def mesh(params: dict[str, Any]) -> deck.QuadMesh:
    """``refine`` x ``refine`` square elements; level 1 is one element."""
    edge, refine = float(params["edge"]), int(params.get("refine", 1))
    return deck.structured_quad_mesh(refine, refine, 0.0, edge, 0.0, edge)


def feasible(params: dict[str, Any]) -> bool:
    """The initial state must lie inside the initial yield surface.

    The block starts stress-free, so the yield function at frame 0 is the
    von Mises measure of the backstress alone: ``|alpha0|`` for the axial
    backstress this example writes.
    """
    sigma_y = float(params["sigma0"]) + float(params["Q_inf"]) * (
        1.0 - np.exp(-float(params["b"]) * float(params["peeq0"]))
    )
    return bool(abs(float(params["alpha0"])) < sigma_y)


def input_deck(params: dict[str, Any], variant: str | None) -> str:
    edge = float(params["edge"])
    grid = mesh(params)
    n = grid.nx
    interval = float(params["frame_interval"])
    period = float(f"{interval * int(params['n_intervals']):.12g}")
    n_elements = len(grid.element_labels)
    # An axial backstress, as the solver itself builds one under axial stress.
    alpha = np.tile([0.0, float(params["alpha0"]), 0.0, 0.0], (n_elements, 1))

    text = deck.heading("Abaqus example: one cycle of axial strain")
    text += deck.node_block(grid.node_labels, grid.coords)
    text += deck.element_block("CAX4R", "BLOCK", grid.element_labels, grid.connectivity)
    text += deck.node_set("BLOCK_NODES", grid.node_labels)
    text += deck.node_set("AXIS", [grid.node(0, j) for j in range(n + 1)])
    text += deck.node_set("BASE", [grid.node(i, 0) for i in range(n + 1)])
    text += deck.node_set("TOP", [grid.node(i, n) for i in range(n + 1)])
    text += deck.material_combined_hardening(
        "BLOCK_MATERIAL",
        density=float(params["rho"]),
        youngs=float(params["E"]),
        poisson=float(params["nu"]),
        yield_stress=float(params["sigma0"]),
        kinematic_modulus=float(params["C"]),
        kinematic_rate=float(params["gamma"]),
        isotropic_saturation=float(params["Q_inf"]),
        isotropic_rate=float(params["b"]),
    )
    text += deck.solid_section("BLOCK", "BLOCK_MATERIAL")
    text += deck.boundary("AXIS", 1)
    text += deck.boundary("BASE", 2)
    text += deck.amplitude(
        "CYCLE",
        [(0.0, 0.0), (0.25 * period, 1.0), (0.75 * period, -1.0), (period, 0.0)],
    )
    text += deck.initial_hardening(
        grid.element_labels, np.full(n_elements, float(params["peeq0"])), alpha
    )
    body = deck.prescribed_displacement(
        "TOP", 2, float(params["strain_amplitude"]) * edge, amplitude="CYCLE"
    )
    body += deck.standard_output(
        interval,
        node_set="BLOCK_NODES",
        node_vars=("U", "V", "A"),
        element_set="BLOCK",
        element_vars=("S", "PEEQ", "ALPHA"),
    )
    scale = params.get("dt_scale")
    return text + deck.explicit_step(
        "CYCLE", period, body, scale_factor=None if scale is None else float(scale)
    )


def qoi(case: Case) -> dict[str, float]:
    """The mean equivalent plastic strain and the block's radius at the last
    frame, in the case's units. The top face is back at its height by then, so
    what the cycle leaves in the kinematics is the radius."""
    assert case.response is not None
    peeq = case.response.element["solid"]["effective_plastic_strain"]
    radius = case.nodes.coords[:, 0] + case.response.node["displacement"][-1][:, 0]
    return {
        "final_plastic_strain": float(peeq[-1].mean()),
        "final_radius": float(radius.max()),
    }
