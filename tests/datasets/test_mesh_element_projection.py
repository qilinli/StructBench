"""Mesh cases whose state lives on the elements (2026-11-06).

A registered aux name that is not a ``response.node`` key is computed from
the case's element block and projected onto the nodes by an area-weighted
mean of the adjacent elements; a case without node types has one type.
"""

import numpy as np
import pytest

from structbench.core import (
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
    write_case,
)
from structbench.datasets import aux_channel_labels, available_aux_fields
from structbench.datasets.canonical import load_case_trajectory, von_mises_from_voigt

# Two quads sharing the edge (1, 4): the left one 1 x 1, the right one 2 x 1,
# so the shared nodes see an area-weighted mean with weights 1 and 2.
COORDS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [1.0, 1.0], [3.0, 1.0]]
)
CELLS = np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.int64)


def _quad_case(tmp_path, *, node_type=None, T=3):
    n_nodes, n_cells = len(COORDS), len(CELLS)
    disp = np.zeros((T, n_nodes, 2), dtype=np.float32)
    disp[:, :, 1] = -np.arange(T, dtype=np.float32)[:, None] * 1e-3  # -1 mm/frame
    stress = np.zeros((T, n_cells, 6), dtype=np.float32)
    stress[:, 0, :4] = [100e6, 200e6, 300e6, 10e6]  # left element, Pa
    stress[:, 1, :4] = [400e6, 500e6, 600e6, 40e6]  # right element
    stress[:, :, 4:] = 7e6  # out-of-plane shears the 2D selection must drop
    peeq = np.zeros((T, n_cells), dtype=np.float32)
    peeq[:, 0], peeq[:, 1] = 0.1, 0.4
    case = Case(
        metadata=Metadata(case_id="Q-test", dimension=2, source_units="t-mm-s"),
        nodes=Nodes(
            coords=COORDS * 1e-3,  # SI metres
            node_id=np.arange(1, n_nodes + 1, dtype=np.int64),
            node_type=None if node_type is None else np.asarray(node_type, np.int64),
        ),
        elements={
            "solid": ElementBlock(
                connectivity=CELLS,
                element_id=np.arange(1, n_cells + 1, dtype=np.int64),
                part_id=np.ones(n_cells, dtype=np.int64),
            )
        },
        materials=[Material(1, "elastic_plastic_isotropic", {}, None)],
        response=Response(
            time=np.arange(T, dtype=np.float64) * 1e-6,
            node={"displacement": disp},
            element={"solid": {"stress": stress, "effective_plastic_strain": peeq}},
        ),
    )
    path = tmp_path / "quad.h5"
    write_case(case, path)
    return path


def test_stress_axisymmetric_is_registered_with_four_channels():
    assert "stress_axisymmetric" in available_aux_fields()
    assert aux_channel_labels("stress_axisymmetric") == ("s_rr", "s_zz", "s_tt", "s_rz")


def test_element_fields_project_to_nodes_area_weighted(tmp_path):
    traj = load_case_trajectory(
        _quad_case(tmp_path),
        aux_field=("stress_axisymmetric", "effective_plastic_strain"),
    )
    assert traj.aux.shape == (3, 6, 5)
    # Nodes of the left element only (0, 3): its values, in MPa.
    np.testing.assert_allclose(traj.aux[0, 0], [100, 200, 300, 10, 0.1], rtol=1e-6)
    # Nodes of the right element only (2, 5).
    np.testing.assert_allclose(traj.aux[0, 5], [400, 500, 600, 40, 0.4], rtol=1e-6)
    # Shared nodes (1, 4): weights 1 (left) and 2 (right).
    expected = (
        np.array([100, 200, 300, 10, 0.1]) + 2 * np.array([400, 500, 600, 40, 0.4])
    ) / 3
    np.testing.assert_allclose(traj.aux[0, 1], expected, rtol=1e-6)
    np.testing.assert_allclose(traj.aux[2, 4], expected, rtol=1e-6)
    # The projection is per frame and the mesh fields are kept.
    np.testing.assert_array_equal(traj.cells, CELLS)
    np.testing.assert_allclose(traj.positions[2, :, 1], COORDS[:, 1] - 2.0, atol=1e-5)


def test_projection_preserves_admissibility_under_linear_hardening(tmp_path):
    # Both element states satisfy vm(sigma) <= sigma_0 + H * peeq; so must
    # every node's weighted mean (convexity of vm, linearity of sigma_y).
    traj = load_case_trajectory(
        _quad_case(tmp_path),
        aux_field=("stress_axisymmetric", "effective_plastic_strain"),
    )
    s4, peeq = traj.aux[0, :, :4], traj.aux[0, :, 4]
    voigt = np.concatenate([s4, np.zeros((6, 2))], axis=-1)
    sigma_0, hardening = 300.0, 1000.0  # MPa; chosen so both elements are admissible
    elem_vm = von_mises_from_voigt(
        np.array([[100, 200, 300, 10, 0, 0], [400, 500, 600, 40, 0, 0]], float)
    )
    assert np.all(elem_vm <= sigma_0 + hardening * np.array([0.1, 0.4]))
    assert np.all(von_mises_from_voigt(voigt) <= sigma_0 + hardening * peeq + 1e-9)


def test_missing_node_type_means_one_type(tmp_path):
    traj = load_case_trajectory(
        _quad_case(tmp_path), aux_field="effective_plastic_strain"
    )
    np.testing.assert_array_equal(traj.particle_type, np.zeros(6, dtype=np.int64))
    typed = load_case_trajectory(
        _quad_case(tmp_path, node_type=[0, 0, 1, 0, 0, 1]),
        aux_field="effective_plastic_strain",
    )
    np.testing.assert_array_equal(typed.particle_type, [0, 0, 1, 0, 0, 1])


def test_node_field_still_wins_over_the_element_route(tmp_path):
    # A response.node key keeps the DeformingPlate reading (first component,
    # stress-scaled) even when the same name is registered for SPH.
    path = _quad_case(tmp_path)
    from structbench.core import read_case

    case = read_case(path)
    assert case.response is not None
    case.response.node["effective_plastic_strain"] = np.full((3, 6, 1), 2e6, np.float32)
    write_case(case, path)
    traj = load_case_trajectory(path, aux_field="effective_plastic_strain")
    np.testing.assert_allclose(traj.aux[..., 0], 2.0)


def test_unknown_name_names_both_routes_and_missing_input_is_explained(tmp_path):
    path = _quad_case(tmp_path)
    with pytest.raises(ValueError, match="response.node") as excinfo:
        load_case_trajectory(path, aux_field="volumetric_strain")
    assert "element block 'solid'" in str(excinfo.value)
    with pytest.raises(ValueError, match="needs element field 'density'"):
        load_case_trajectory(path, aux_field="density")
