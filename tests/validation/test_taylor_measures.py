"""The Taylor measures: outline of a canonical case, L_f, R_f, W_f, the settled band."""

import numpy as np
import pytest
from conftest import rod_case, two_body_case

from structbench.validation.measures import taylor


def test_outline_is_one_closed_loop_from_the_impact_face():
    rz = taylor.outline(rod_case())
    assert rz.ndim == 2 and rz.shape[1] == 2
    assert np.allclose(rz[0], rz[-1])  # closed
    assert rz[:, 1].min() == pytest.approx(0.0)  # z from the impact face
    assert not (rz == 9.0).all(axis=1).any()  # the unused node is not on it


def test_measures_of_the_known_deformation():
    # z' = 0.8 z and r' = r (1 + 0.5 (1 - z/L)) give
    # L_f = 4, R_f = 1.5 and W_f(f) = 1 + 0.5 (1 - f)
    rz = taylor.outline(rod_case())
    assert taylor.final_length(rz) == pytest.approx(4.0)
    assert taylor.largest_radius(rz) == pytest.approx(1.5)
    got = taylor.lateral_radii(rz, 4.0, taylor.FRACTIONS)
    assert got == pytest.approx([1 + 0.5 * (1 - f) for f in taylor.FRACTIONS])


def test_a_vertex_exactly_on_a_fraction_height_counts_once():
    # nz = 10 puts deformed vertices at z' = 0.4 k; f = 0.5 -> h = 2.0 is a vertex row
    rz = taylor.outline(rod_case(nz=10))
    assert taylor.lateral_radii(rz, 4.0, (0.5,)) == pytest.approx([1.25])


def test_a_height_with_no_crossing_is_nan():
    rz = taylor.outline(rod_case())
    assert np.isnan(taylor.lateral_radii(rz, 4.0, (1.5,))[0])


def test_outcome_reports_the_settled_band_of_the_length():
    still = taylor.taylor_outcome(rod_case(wobble=0.0))
    assert still.length == pytest.approx(4.0) and still.length_band == 0.0
    assert still.largest_radius == pytest.approx(1.5)
    moving = taylor.taylor_outcome(rod_case(wobble=0.1, frames=9))
    # half the peak-to-peak of the length over the last quarter of the frames
    assert moving.length_band == pytest.approx(0.1)
    expected = [1 + 0.5 * (1 - f) for f in taylor.FRACTIONS]
    assert list(moving.lateral_radii) == pytest.approx(expected, rel=0.05)


def test_two_bodies_are_refused():
    with pytest.raises(taylor.OutlineError, match="one closed loop"):
        taylor.outline(two_body_case())


def test_a_three_dimensional_case_is_refused():
    import dataclasses

    from structbench.core import Nodes

    case = rod_case()
    coords = np.column_stack([case.nodes.coords, np.zeros(len(case.nodes.coords))])
    u = np.concatenate(
        [case.response.node["displacement"], np.zeros((5, len(coords), 1), np.float32)],
        axis=2,
    )
    nodes = Nodes(coords=coords, node_id=case.nodes.node_id)
    response = dataclasses.replace(case.response, node={"displacement": u})
    solid = dataclasses.replace(case, nodes=nodes, response=response)
    with pytest.raises(taylor.OutlineError, match="axisymmetric"):
        taylor.outline(solid)
