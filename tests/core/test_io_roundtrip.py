"""Round-trip tests for the canonical HDF5 case I/O."""

from __future__ import annotations

import numpy as np
import pytest

from structbench.core import (
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Provenance,
    Response,
    SchemaError,
    read_case,
    write_case,
)


def _shell_case() -> Case:
    """A small 2D case: one shell quad over four nodes, with response."""
    rng = np.random.default_rng(0)
    return Case(
        metadata=Metadata(
            case_id="synthetic-shell-001",
            dimension=2,
            provenance=Provenance("LS-DYNA", "R13", "2026-05-22"),
            source_deck="*KEYWORD\n*TITLE\nsynthetic\n*END\n",
        ),
        nodes=Nodes(
            coords=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
            node_id=np.array([10, 11, 12, 13], dtype=np.int64),
        ),
        elements={
            "shell": ElementBlock(
                connectivity=np.array([[0, 1, 2, 3]], dtype=np.int64),
                element_id=np.array([100], dtype=np.int64),
                part_id=np.array([1], dtype=np.int64),
            )
        },
        materials=[
            Material(
                material_id=1,
                source_model="MAT_ELASTIC",
                source_params={"ro": 7850.0, "e": 2.1e11, "pr": 0.3},
                canonical_model="linear_elastic",
            )
        ],
        response=Response(
            time=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            node={"displacement": rng.random((3, 4, 2), dtype=np.float32)},
            globals_={"kinetic_energy": rng.random(3, dtype=np.float32)},
        ),
    )


def _sph_taylor_like_case() -> Case:
    """A Taylor-like SPH case: degenerate single-node connectivity, EOS-bearing
    material. Mirrors the structure of the Taylor.k reference deck at tiny size.
    """
    n = 5
    return Case(
        metadata=Metadata(case_id="taylor-mini-001", dimension=2),
        nodes=Nodes(
            coords=np.column_stack(
                [np.full(n, 0.25), np.linspace(-9.75, -7.75, n)]
            ).astype(np.float64),
            node_id=np.arange(1, n + 1, dtype=np.int64),
        ),
        elements={
            "sph": ElementBlock(
                connectivity=np.arange(n, dtype=np.int64).reshape(n, 1),
                element_id=np.arange(1, n + 1, dtype=np.int64),
                part_id=np.ones(n, dtype=np.int64),
            )
        },
        materials=[
            Material(
                material_id=2,
                source_model="MAT_ELASTIC_PLASTIC_HYDRO",
                source_params={
                    "ro": 0.0089,
                    "g": 37590.0,
                    "eps": [0.0, 0.5, 1.0],
                    "es": [199.3, 251.1, 250.9],
                    "eos": {"model": "GRUNEISEN", "c": 3940.0, "gamao": 2.02},
                },
                canonical_model="elastic_plastic_hydro",
            )
        ],
        response=Response(
            time=np.array([0.0, 0.1], dtype=np.float64),
            node={"displacement": np.zeros((2, n, 2), dtype=np.float32)},
        ),
    )


@pytest.mark.parametrize(
    "build", [_shell_case, _sph_taylor_like_case], ids=["shell", "sph"]
)
def test_roundtrip(build, tmp_path):
    case = build()
    path = tmp_path / "case.h5"
    write_case(case, path)
    back = read_case(path)

    assert back.metadata == case.metadata
    np.testing.assert_array_equal(back.nodes.coords, case.nodes.coords)
    np.testing.assert_array_equal(back.nodes.node_id, case.nodes.node_id)
    assert back.nodes.coords.dtype == np.float64
    assert back.nodes.node_id.dtype == np.int64

    assert set(back.elements) == set(case.elements)
    for etype, block in case.elements.items():
        np.testing.assert_array_equal(
            back.elements[etype].connectivity, block.connectivity
        )

    assert back.materials == case.materials

    assert back.response is not None and case.response is not None
    np.testing.assert_array_equal(back.response.time, case.response.time)
    for name, arr in case.response.node.items():
        np.testing.assert_array_equal(back.response.node[name], arr)
        assert back.response.node[name].dtype == np.float32
    for name, arr in case.response.globals_.items():
        np.testing.assert_array_equal(back.response.globals_[name], arr)


def test_stub_case_without_response_roundtrips(tmp_path):
    case = _shell_case()
    case.response = None
    path = tmp_path / "stub.h5"
    write_case(case, path)
    back = read_case(path)
    assert back.response is None


def test_validate_rejects_dimension_mismatch(tmp_path):
    case = _shell_case()
    case.metadata.dimension = 3  # coords are still (n, 2)
    with pytest.raises(SchemaError, match="nodes.coords"):
        write_case(case, tmp_path / "bad.h5")


def test_validate_rejects_simulated_case_without_displacement(tmp_path):
    case = _shell_case()
    assert case.response is not None
    case.response.node = {}  # remove required displacement
    with pytest.raises(SchemaError, match="displacement"):
        write_case(case, tmp_path / "bad.h5")


def test_validate_rejects_response_element_type_absent_from_elements(tmp_path):
    case = _shell_case()
    assert case.response is not None
    # response references an element type the case has no block for
    case.response.element = {"solid": {"stress": np.zeros((3, 1, 6), np.float32)}}
    with pytest.raises(SchemaError, match="solid"):
        write_case(case, tmp_path / "bad.h5")


def test_validate_rejects_zero_width_connectivity(tmp_path):
    case = _shell_case()
    case.elements["shell"].connectivity = np.zeros((1, 0), dtype=np.int64)
    with pytest.raises(SchemaError, match="node per element|connectivity"):
        write_case(case, tmp_path / "bad.h5")


def test_validate_rejects_zero_frame_response(tmp_path):
    case = _shell_case()
    case.response = Response(
        time=np.zeros((0,), dtype=np.float64),
        node={"displacement": np.zeros((0, 4, 2), dtype=np.float32)},
        element={},
        globals_={},
    )
    with pytest.raises(SchemaError, match="frame"):
        write_case(case, tmp_path / "bad.h5")


def test_validate_accepts_optional_node_fields():
    from structbench.core import validate

    case = _shell_case()  # existing 2D 4-node helper
    case.nodes.node_type = np.array([0, 0, 3, 3], dtype=np.int64)
    case.nodes.reference_coords = case.nodes.coords.copy()
    validate(case)  # must not raise


def test_validate_rejects_bad_node_type_shape():
    from structbench.core import validate
    from structbench.core.exceptions import SchemaError

    case = _shell_case()
    case.nodes.node_type = np.array([0, 0, 3], dtype=np.int64)  # 3 != 4 nodes
    with pytest.raises(SchemaError, match="node_type"):
        validate(case)


def test_validate_rejects_bad_reference_coords_shape():
    from structbench.core import validate
    from structbench.core.exceptions import SchemaError

    case = _shell_case()
    case.nodes.reference_coords = np.zeros((4, 3), dtype=np.float64)  # dim=2, not 3
    with pytest.raises(SchemaError, match="reference_coords"):
        validate(case)


def test_validate_accepts_per_node_scalar_field():
    from structbench.core import validate

    case = _shell_case()  # 2D, 4 nodes, 3 frames
    case.response.node["von_mises_stress"] = np.zeros((3, 4, 1), dtype=np.float32)
    validate(case)  # (T, N, 1) must be allowed now


def test_validate_still_requires_displacement_dim_wide():
    from structbench.core import validate
    from structbench.core.exceptions import SchemaError

    case = _shell_case()
    # not dim=2
    case.response.node["displacement"] = np.zeros((3, 4, 1), dtype=np.float32)
    with pytest.raises(SchemaError, match="displacement"):
        validate(case)


def test_validate_rejects_per_node_field_wrong_node_count():
    from structbench.core import validate
    from structbench.core.exceptions import SchemaError

    case = _shell_case()
    # 5 != 4
    case.response.node["von_mises_stress"] = np.zeros((3, 5, 1), dtype=np.float32)
    with pytest.raises(SchemaError, match="von_mises_stress"):
        validate(case)


def test_roundtrip_per_node_fields(tmp_path):
    from structbench.core import read_case, write_case

    case = _shell_case()
    case.nodes.node_type = np.array([0, 1, 3, 3], dtype=np.int64)
    case.nodes.reference_coords = case.nodes.coords.copy()
    case.response.node["von_mises_stress"] = np.arange(12, dtype=np.float32).reshape(
        3, 4, 1
    )
    path = tmp_path / "c.h5"
    write_case(case, path)
    back = read_case(path)
    np.testing.assert_array_equal(back.nodes.node_type, case.nodes.node_type)
    assert back.nodes.node_type.dtype == np.int64
    np.testing.assert_array_equal(
        back.nodes.reference_coords, case.nodes.reference_coords
    )
    np.testing.assert_array_equal(
        back.response.node["von_mises_stress"], case.response.node["von_mises_stress"]
    )
    assert back.metadata.schema_version == "0.2.0"


def test_roundtrip_without_optional_node_fields_is_backward_compatible(tmp_path):
    from structbench.core import read_case, write_case

    case = _shell_case()  # no node_type / reference_coords set
    path = tmp_path / "c.h5"
    write_case(case, path)
    back = read_case(path)
    assert back.nodes.node_type is None
    assert back.nodes.reference_coords is None
