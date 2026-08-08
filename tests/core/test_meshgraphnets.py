from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from structbench.core import validate
from structbench.core.io.meshgraphnets import build_deforming_plate_case, parse_meta

_DEFORMING_PLATE_META = {
    "simulator": "comsol",
    "dt": 0,
    "trajectory_length": 400,
    "field_names": ["cells", "node_type", "mesh_pos", "world_pos", "stress"],
    "features": {
        "cells": {"type": "static", "dtype": "int32", "shape": [1, -1, 4]},
        "node_type": {"type": "static", "dtype": "int32", "shape": [1, -1, 1]},
        "mesh_pos": {"type": "static", "dtype": "float32", "shape": [1, -1, 3]},
        "world_pos": {"type": "dynamic", "dtype": "float32", "shape": [400, -1, 3]},
        "stress": {"type": "dynamic", "dtype": "float32", "shape": [400, -1, 1]},
    },
}


def test_parse_meta_reads_all_fields():
    specs = parse_meta(_DEFORMING_PLATE_META)
    assert set(specs) == set(_DEFORMING_PLATE_META["field_names"])
    assert specs["cells"].ftype == "static" and specs["cells"].shape == (1, -1, 4)
    assert specs["world_pos"].ftype == "dynamic"
    assert specs["world_pos"].dtype == "float32"


def test_parse_meta_rejects_unknown_type():
    bad = {
        "field_names": ["x"],
        "trajectory_length": 1,
        "features": {"x": {"type": "bogus", "dtype": "float32", "shape": [1]}},
    }
    with pytest.raises(ValueError, match="bogus"):
        parse_meta(bad)


def _synthetic_traj(n_nodes=5, n_cells=2, T=4):
    rng = np.random.default_rng(0)
    world0 = rng.random((n_nodes, 3)).astype(np.float32)
    world = np.stack([world0 + i * 0.1 for i in range(T)]).astype(np.float32)  # (T,N,3)
    return {
        "cells": rng.integers(0, n_nodes, (n_cells, 4)).astype(np.int32),
        "node_type": np.array([0, 0, 1, 3, 0][:n_nodes], dtype=np.int32),
        "mesh_pos": world0.copy(),
        "world_pos": world,
        "stress": rng.random((T, n_nodes, 1)).astype(np.float32),
    }


def test_build_case_maps_fields_and_validates():
    a = _synthetic_traj()
    case = build_deforming_plate_case(a, source_units="kg-m-s", case_id="dp-000")
    validate(case)  # must not raise
    assert case.metadata.dimension == 3
    assert case.metadata.units_convention == "SI"
    # coords == world_pos[0]; displacement is delta-from-initial
    np.testing.assert_allclose(case.nodes.coords, a["world_pos"][0], rtol=1e-6)
    np.testing.assert_allclose(
        case.response.node["displacement"][0], np.zeros((5, 3)), atol=1e-6
    )
    np.testing.assert_allclose(
        case.response.node["displacement"][2],
        a["world_pos"][2] - a["world_pos"][0],
        rtol=1e-5,
    )
    # per-node static fields
    np.testing.assert_array_equal(case.nodes.node_type, a["node_type"].astype(np.int64))
    np.testing.assert_allclose(case.nodes.reference_coords, a["mesh_pos"], rtol=1e-6)
    # mesh edges live as tetra connectivity
    np.testing.assert_array_equal(
        case.elements["tetra"].connectivity, a["cells"].astype(np.int64)
    )
    # per-node von Mises stress
    assert case.response.node["von_mises_stress"].shape == (4, 5, 1)
    # exactly one material with non-empty source_model
    assert len(case.materials) == 1 and case.materials[0].source_model


def test_build_case_identity_units_are_si():
    a = _synthetic_traj()
    case = build_deforming_plate_case(a, source_units="kg-m-s", case_id="dp-000")
    # kg-m-s is SI identity: coords equal raw world_pos[0]
    np.testing.assert_allclose(case.nodes.coords, a["world_pos"][0], rtol=1e-6)


def test_build_case_applies_non_identity_units():
    # g-mm-ms is a non-identity source convention: unit_factors("g-mm-ms")
    # gives length=1e-3, stress=1e6. kg-m-s (the other tests) is SI identity
    # and would silently pass even if the factor/key wiring were broken.
    a = _synthetic_traj()
    case = build_deforming_plate_case(a, source_units="g-mm-ms", case_id="dp-u")
    np.testing.assert_allclose(case.nodes.coords, a["world_pos"][0] * 1e-3, rtol=1e-5)
    np.testing.assert_allclose(
        case.nodes.reference_coords, a["mesh_pos"] * 1e-3, rtol=1e-5
    )
    np.testing.assert_allclose(
        case.response.node["displacement"][2],
        (a["world_pos"][2] - a["world_pos"][0]) * 1e-3,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        case.response.node["von_mises_stress"], a["stress"] * 1e6, rtol=1e-4
    )


def test_import_structbench_does_not_import_tensorflow():
    # Run in a fresh subprocess: this test file's own top-level import of
    # structbench.core.io.meshgraphnets (see above) already populates
    # sys.modules by the time any test runs, so importlib.import_module()
    # in-process would just return the cached module without re-executing
    # its body. A subprocess has no such cache, so it genuinely re-runs the
    # module's top-level code and can actually fail if a top-level
    # `import tensorflow` were ever (re-)introduced.
    code = (
        "import sys, structbench.core.io.meshgraphnets\n"
        "assert 'tensorflow' not in sys.modules, "
        "sorted(m for m in sys.modules if 'tensor' in m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    not os.environ.get("STRUCTBENCH_DEFORMING_PLATE_DIR"),
    reason="requires STRUCTBENCH_DEFORMING_PLATE_DIR (meta.json + valid.tfrecord)",
)
def test_read_first_trajectory_builds_valid_case():
    pytest.importorskip("tensorflow")
    from structbench.core.io.meshgraphnets import (
        build_deforming_plate_case,
        read_deforming_plate,
    )

    d = os.environ["STRUCTBENCH_DEFORMING_PLATE_DIR"]
    a = next(read_deforming_plate(d, "valid"))
    assert a["world_pos"].ndim == 3 and a["world_pos"].shape[2] == 3
    case = build_deforming_plate_case(a, source_units="kg-m-s", case_id="dp-valid-000")
    validate(case)
