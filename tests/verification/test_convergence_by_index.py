"""Field errors between nested meshes whose quads are not rectangles.

A structured mesh that nests by index (each coarse element the union of its
``r x r`` children on the grid) is restricted by index, whatever the quads'
shape; rectangles keep the restriction by place, bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from structbench.core import (  # noqa: E402
    Case,
    ElementBlock,
    Metadata,
    Nodes,
    Response,
)
from structbench.datagen.abaqus import deck  # noqa: E402
from structbench.verification import convergence as cv  # noqa: E402


def grid(nx, ny):
    g = deck.structured_quad_mesh(nx, ny, 0.0, 2.0, 0.0, 3.0)
    return g.coords.copy(), g.connectivity - 1


def warp(xy):
    """A smooth map of the plane: every quad sheared, the nesting kept."""
    return np.column_stack(
        [xy[:, 0] * (1 + 0.05 * np.sin(xy[:, 1])), xy[:, 1] + 0.03 * xy[:, 0] ** 2]
    )


def scrambled(conn, rng):
    """The same elements in another order, each from another starting corner."""
    order = rng.permutation(len(conn))
    return np.array([np.roll(conn[e], rng.integers(4)) for e in order]), order


def case(xy, conn, node, element):
    frames = next(iter(node.values())).shape[0]
    return Case(
        metadata=Metadata(case_id="x", dimension=2),
        nodes=Nodes(coords=xy, node_id=np.arange(1, len(xy) + 1)),
        elements={
            "solid": ElementBlock(
                connectivity=conn,
                element_id=np.arange(1, len(conn) + 1),
                part_id=np.ones(len(conn), np.int64),
            )
        },
        materials=[],
        response=Response(
            time=np.linspace(0.0, 1.0, frames), node=node, element={"solid": element}
        ),
    )


@pytest.mark.parametrize("ratio", [(2, 2), (3, 3), (2, 4)])
def test_parents_by_index_are_the_parents_by_place(ratio):
    rng = np.random.default_rng(1)
    cxy, cconn = grid(3, 4)
    fxy, fconn = grid(3 * ratio[0], 4 * ratio[1])
    by_place = cv.Restriction._parents_by_place(cxy, cconn, fxy, fconn)
    c2, c_order = scrambled(cconn, rng)
    f2, f_order = scrambled(fconn, rng)
    index = cv.match_nodes(warp(cxy), warp(fxy))
    restrict = cv.Restriction(warp(cxy), c2, warp(fxy), f2, fine_of_coarse=index)
    # Restrict each fine element's reference parent: every coarse element must
    # read its own number back.
    ids = by_place[f_order][None, :, None].astype(float)
    np.testing.assert_allclose(restrict.apply(ids)[0, :, 0], c_order)


def test_the_restriction_by_index_is_weighted_as_the_one_by_place():
    cxy, cconn = grid(2, 2)
    fxy, fconn = grid(4, 4)
    wc, wf = warp(cxy), warp(fxy)
    index = cv.match_nodes(wc, wf)
    values = np.random.default_rng(2).normal(size=(3, len(fconn), 2))
    for axisymmetric in (True, False):
        restrict = cv.Restriction(
            wc, cconn, wf, fconn, axisymmetric=axisymmetric, fine_of_coarse=index
        )
        weight = cv.element_weights(wf, fconn, axisymmetric=axisymmetric)
        parent = cv.Restriction._parents_by_place(cxy, cconn, fxy, fconn)
        expected = np.stack(
            [
                (values[:, parent == e] * weight[parent == e, None]).sum(axis=1)
                / weight[parent == e].sum()
                for e in range(len(cconn))
            ],
            axis=1,
        )
        np.testing.assert_allclose(restrict.apply(values), expected, rtol=1e-12)


def test_rectangles_keep_the_restriction_by_place_bit_for_bit():
    cxy, cconn = grid(3, 4)
    fxy, fconn = grid(6, 8)
    values = np.random.default_rng(3).normal(size=(2, len(fconn), 4))
    index = cv.match_nodes(cxy, fxy)
    plain = cv.Restriction(cxy, cconn, fxy, fconn).apply(values)
    given = cv.Restriction(cxy, cconn, fxy, fconn, fine_of_coarse=index).apply(values)
    assert plain.tobytes() == given.tobytes()
    # Without the node map a mesh that is not rectangles is refused, as before.
    with pytest.raises(cv.NestingError, match="axis-aligned"):
        cv.Restriction(warp(cxy), cconn, warp(fxy), fconn)


def test_what_does_not_nest_by_index_is_refused_by_name():
    cxy, cconn = grid(3, 4)
    fxy, fconn = grid(6, 8)
    wc, wf = warp(cxy), warp(fxy)
    index = cv.match_nodes(wc, wf)
    wrong = index.copy()
    wrong[0] = index[1]
    with pytest.raises(cv.NestingError, match="do not nest by index"):
        cv.Restriction(wc, cconn, wf, fconn, fine_of_coarse=wrong)
    # A mesh with a hole is not one structured grid.
    holed = np.delete(fconn, 9, axis=0)
    with pytest.raises(cv.NestingError, match="not a structured quad grid"):
        cv.Restriction(wc, cconn, wf, holed, fine_of_coarse=index)
    # Two grids that only touch are not one either.
    with pytest.raises(cv.NestingError, match="not a structured quad grid"):
        cv._grid_indices(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), 8)


def _pair(element_xy=warp):
    cxy, cconn = grid(2, 2)
    fxy, fconn = grid(4, 4)
    wc, wf = element_xy(cxy), element_xy(fxy)
    rng = np.random.default_rng(4)
    fine_node = rng.normal(size=(3, len(wf), 2)).astype(np.float32)
    index = cv.match_nodes(wc, wf)
    fine_element = np.ones((3, len(fconn)), np.float32)
    coarse = case(
        wc,
        cconn,
        {"displacement": fine_node[:, index] * 1.1},
        {"effective_plastic_strain": np.full((3, len(cconn)), 1.2, np.float32)},
    )
    fine = case(
        wf,
        fconn,
        {"displacement": fine_node},
        {"effective_plastic_strain": fine_element},
    )
    return coarse, fine


def test_field_errors_cover_node_and_element_fields_on_sheared_quads():
    errors = cv.field_errors(*_pair())
    assert errors["displacement"] == pytest.approx(0.1, rel=1e-5)
    assert errors["effective_plastic_strain"] == pytest.approx(0.2, rel=1e-5)


def test_node_fields_survive_element_fields_that_cannot_be_restricted():
    coarse, fine = _pair()
    # A fine mesh with a hole still contains every coarse node here, but its
    # element fields have no grid to be restricted on.
    solid = fine.elements["solid"]
    keep = np.arange(len(solid.connectivity)) != 5
    fine.elements["solid"] = ElementBlock(
        solid.connectivity[keep], solid.element_id[keep], solid.part_id[keep]
    )
    fine.response.element["solid"] = {
        k: v[:, keep] for k, v in fine.response.element["solid"].items()
    }
    with pytest.raises(cv.NestingError):
        cv.field_errors(coarse, fine)
    notes: list[str] = []
    errors = cv.field_errors(coarse, fine, notes=notes)
    assert list(errors) == ["displacement"]
    assert errors["displacement"] == pytest.approx(0.1, rel=1e-5)
    assert len(notes) == 1 and notes[0].startswith("element fields not compared")


def test_a_reference_node_standing_on_a_mesh_node_is_not_taken_for_it():
    """Found on a real run: a rigid plate's reference node at the place of the
    mesh's top axis node, stored in the fine case after the mesh nodes."""
    coarse, fine = _pair()
    at = int(np.argmax(fine.nodes.coords[:, 1] - fine.nodes.coords[:, 0]))
    extra = fine.nodes.coords[[at]]
    crowded = case(
        np.vstack([extra, fine.nodes.coords]),  # the stranger first: it wins a tie
        fine.elements["solid"].connectivity + 1,
        {
            "displacement": np.concatenate(
                [
                    np.full((3, 1, 2), 99.0, np.float32),
                    fine.response.node["displacement"],
                ],
                axis=1,
            )
        },
        fine.response.element["solid"],
    )
    errors = cv.field_errors(coarse, crowded)
    assert errors["displacement"] == pytest.approx(0.1, rel=1e-5)
    assert errors["effective_plastic_strain"] == pytest.approx(0.2, rel=1e-5)
