"""Convergence engine: Richardson and its statuses, nesting, restriction, errors."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("scipy")

from structbench.core import (  # noqa: E402
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
)
from structbench.verification import convergence as cv  # noqa: E402


def grid(nr, nz, radius=1.0, length=2.0):
    rs, zs = np.linspace(0.0, radius, nr + 1), np.linspace(0.0, length, nz + 1)
    gr, gz = np.meshgrid(rs, zs)
    coords = np.column_stack([gr.ravel(), gz.ravel()])
    i, j = (a.ravel() for a in np.meshgrid(np.arange(nr), np.arange(nz)))
    row = nr + 1
    conn = np.column_stack(
        [i + j * row, i + 1 + j * row, i + 1 + (j + 1) * row, i + (j + 1) * row]
    )
    return coords, conn


def case(xy, conn, node=None, element=None, frames=3):
    n, e = len(xy), len(conn)
    node = node or {"displacement": np.zeros((frames, n, 2), np.float32)}
    element = element or {"stress": np.zeros((frames, e, 6), np.float32)}
    return Case(
        metadata=Metadata(case_id="c", dimension=2, source_units="t-mm-s"),
        nodes=Nodes(coords=xy, node_id=np.arange(1, n + 1)),
        elements={
            "solid": ElementBlock(
                connectivity=conn,
                element_id=np.arange(1, e + 1),
                part_id=np.ones(e, np.int64),
            )
        },
        materials=[Material(material_id=1, source_model="toy", source_params={})],
        response=Response(
            time=np.linspace(0.0, 1.0, frames),
            node=node,
            element={"solid": element},
            globals_={},
        ),
    )


def fields(xy, conn, frames=3):
    t = np.arange(1, frames + 1)[:, None, None]
    u = t * np.stack([xy[:, 0] * 0.1, -xy[:, 1] * 0.2], axis=-1)[None]
    c = xy[conn].mean(axis=1)
    s = (
        t
        * np.stack(
            [
                c[:, 0],
                c[:, 1],
                c[:, 0] + c[:, 1],
                0 * c[:, 0],
                2 * c[:, 1],
                3 * c[:, 0],
            ],
            -1,
        )[None]
    )
    node = {"displacement": u, "velocity": 2 * u}
    element = {"stress": s, "effective_plastic_strain": t[..., 0] * (1 + c[:, 1])[None]}
    return node, element


# --- Richardson ---------------------------------------------------------------


def test_second_order_data_give_order_two_and_the_exact_limit():
    # f = 1 + 0.04 (h/h0)^2 on h, h/2, h/4
    x = cv.richardson({"1": 1.04, "2": 1.01, "4": 1.0025}, ratio=2.0)
    assert x.status == "monotone" and x.order == pytest.approx(2.0)
    assert x.extrapolated == pytest.approx(1.0)
    assert x.error_vs_extrapolated == pytest.approx({"1": 0.04, "2": 0.01, "4": 0.0025})
    assert x.gci_fine == pytest.approx(1.25 * 0.0075 / 1.0025 / 3.0)
    assert x.coarsest_vs_finest == pytest.approx(0.0375 / 1.0025)


@pytest.mark.parametrize(
    "values, status",
    [
        ({"1": 1.04, "2": 0.99, "4": 1.0}, "oscillatory"),
        ({"1": 1.00, "2": 1.01, "4": 1.05}, "diverging"),
        ({"1": 1.02, "2": 1.0, "4": 1.0}, "flat"),
    ],
)
def test_data_without_an_order_say_why(values, status):
    x = cv.richardson(values, ratio=2.0)
    assert x.status == status and x.order is None and x.extrapolated is None
    assert x.error_vs_extrapolated is None and x.coarsest_vs_finest > 0


def test_richardson_needs_exactly_three_levels():
    with pytest.raises(ValueError, match="three"):
        cv.richardson({"1": 1.0, "2": 1.0}, ratio=2.0)


# --- nesting and restriction --------------------------------------------------


@pytest.mark.parametrize("k", [2, 4])
def test_every_coarse_node_is_found_in_a_nested_mesh(k):
    coarse, _ = grid(3, 5, 1.0, 2.1)
    fine, _ = grid(3 * k, 5 * k, 1.0, 2.1)
    idx = cv.match_nodes(coarse, fine)
    np.testing.assert_allclose(fine[idx], coarse, atol=1e-12)


def test_a_mesh_that_does_not_nest_is_refused():
    coarse, _ = grid(3, 5, 1.0, 2.1)
    fine, _ = grid(6, 11, 1.0, 2.1)
    with pytest.raises(cv.NestingError, match="nest"):
        cv.match_nodes(coarse, fine)


def test_axisymmetric_weights_are_the_first_moment_and_planar_the_area():
    xy, conn = grid(2, 1, 1.0, 1.0)  # r in [0, .5] and [.5, 1], z in [0, 1]
    assert cv.element_weights(xy, conn, axisymmetric=True) == pytest.approx(
        [0.125, 0.375]
    )
    assert cv.element_weights(xy, conn, axisymmetric=False) == pytest.approx([0.5, 0.5])


def test_restriction_is_the_weighted_average_of_the_children():
    # a field equal to r, averaged with weight r: exactly 2/3 (b^3 - a^3)/(b^2 - a^2)
    cxy, cconn = grid(2, 3, 1.0, 1.5)
    fxy, fconn = grid(8, 12, 1.0, 1.5)

    def r_mean(xy, conn):
        r = xy[conn][..., 0]
        a, b = r.min(axis=1), r.max(axis=1)
        return 2.0 / 3.0 * (b**3 - a**3) / (b**2 - a**2)

    values = np.stack([r_mean(fxy, fconn)] * 2)
    out = cv.Restriction(cxy, cconn, fxy, fconn, axisymmetric=True).apply(values)
    np.testing.assert_allclose(out, np.stack([r_mean(cxy, cconn)] * 2), rtol=1e-12)
    planar = cv.Restriction(cxy, cconn, fxy, fconn, axisymmetric=False).apply(values)
    assert planar.shape == (2, len(cconn))
    assert not np.allclose(planar, out)  # the two weightings differ off the axis


def test_restriction_keeps_trailing_components_and_refuses_a_stranger():
    cxy, cconn = grid(2, 2, 1.0, 1.0)
    fxy, fconn = grid(4, 4, 1.0, 1.0)
    values = np.ones((3, len(fconn), 6)) * np.arange(6)
    out = cv.Restriction(cxy, cconn, fxy, fconn).apply(values)
    assert out.shape == (3, len(cconn), 6)
    np.testing.assert_allclose(out, np.ones((3, len(cconn), 6)) * np.arange(6))
    oxy, oconn = grid(4, 4, 1.2, 1.0)  # wider: children outside every coarse element
    with pytest.raises(cv.NestingError):
        cv.Restriction(cxy, cconn, oxy, oconn)


# --- field errors on cases ----------------------------------------------------


def _pair(scale_disp=1.0, drop=None):
    cxy, cconn = grid(2, 4)
    fxy, fconn = grid(4, 8)
    fn, fe = fields(fxy, fconn)
    idx = cv.match_nodes(cxy, fxy)
    restrict = cv.Restriction(cxy, cconn, fxy, fconn)
    cn = {
        k: v[:, idx] * (scale_disp if k == "displacement" else 1.0)
        for k, v in fn.items()
    }
    ce = {k: restrict.apply(v) for k, v in fe.items()}
    if drop:
        fn = {k: v for k, v in fn.items() if k != drop}
    return case(cxy, cconn, cn, ce), case(fxy, fconn, fn, fe)


def test_identical_solutions_have_zero_error_in_every_stored_field():
    errors = cv.field_errors(*_pair())
    assert set(errors) == {
        "displacement",
        "velocity",
        "stress",
        "effective_plastic_strain",
    }
    assert all(e == pytest.approx(0.0, abs=1e-6) for e in errors.values())


def test_a_ten_percent_displacement_error_reads_as_ten_percent():
    errors = cv.field_errors(*_pair(scale_disp=1.1))
    assert errors["displacement"] == pytest.approx(0.1, rel=1e-5)
    assert errors["stress"] == pytest.approx(0.0, abs=1e-6)


def test_a_field_missing_on_one_side_is_left_out_not_raised():
    errors = cv.field_errors(*_pair(drop="velocity"))
    assert "velocity" not in errors and "displacement" in errors


def test_different_time_grids_are_refused():
    coarse, fine = _pair()
    response = dataclasses.replace(fine.response, time=fine.response.time * 2)
    fine = dataclasses.replace(fine, response=response)
    with pytest.raises(ValueError, match="time"):
        cv.field_errors(coarse, fine)
