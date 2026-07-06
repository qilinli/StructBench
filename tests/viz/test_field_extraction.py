"""Viz field-extraction tests that do not need matplotlib.

Kept out of test_fringe.py (which has a module-level matplotlib importorskip)
so these run everywhere: FIELDS registry completeness and load_case_field
extraction touch no plotting code.
"""

from __future__ import annotations

import numpy as np

from structbench.core import (
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
    write_case,
)


def _sph_case_with_stress_and_strain(tmp_path):
    """A 2-frame, 1-particle SPH case carrying known stress and strain."""
    coords = np.array([[0.0, 0.0]])
    disp = np.zeros((2, 1, 2), dtype=np.float32)
    stress = np.zeros((2, 1, 6), dtype=np.float32)
    stress[..., 0] = 100e6  # sigma_xx = 100 MPa in Pa
    strain = np.zeros((2, 1, 6), dtype=np.float32)
    strain[..., :] = np.array([0.02, -0.01, 0.0, 0.02, 0.0, 0.0], np.float32)
    case = Case(
        metadata=Metadata(case_id="F-x", dimension=2, source_units="g-mm-ms"),
        nodes=Nodes(coords=coords, node_id=np.array([1], dtype=np.int64)),
        elements={
            "sph": ElementBlock(
                connectivity=np.array([[0]], dtype=np.int64),
                element_id=np.array([1], dtype=np.int64),
                part_id=np.array([1], dtype=np.int64),
            )
        },
        materials=[Material(1, "MAT_ELASTIC", {"data": [[1]]}, None)],
        response=Response(
            time=np.array([0.0, 1e-6]),
            node={"displacement": disp},
            element={"sph": {"stress": stress, "strain": strain}},
        ),
    )
    path = tmp_path / "field_case.h5"
    write_case(case, path)
    return path


def test_every_benchmark_aux_field_is_renderable():
    """Every registered benchmark's aux field must be in the viz registry."""
    from structbench.benchmarks import available_benchmarks, get_benchmark
    from structbench.viz.fringe import FIELDS

    for name in available_benchmarks():
        spec = get_benchmark(name)
        assert spec.aux_field in FIELDS, f"{name}: {spec.aux_field!r} not renderable"


def test_load_case_field_extracts_axial_stress(tmp_path):
    from structbench.viz.fringe import load_case_field

    cf = load_case_field(_sph_case_with_stress_and_strain(tmp_path), "axial_stress")
    np.testing.assert_allclose(cf.values, 100.0, rtol=1e-5)  # Pa -> MPa


def test_load_case_field_extracts_max_principal_strain(tmp_path):
    from structbench.viz.fringe import load_case_field

    cf = load_case_field(
        _sph_case_with_stress_and_strain(tmp_path), "max_principal_strain"
    )
    np.testing.assert_allclose(cf.values, 0.0230278, rtol=1e-5)
