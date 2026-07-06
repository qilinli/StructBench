"""Tests for the LS-DYNA -> canonical-case adapter (ADR-0016).

These tests are data-free: the deck parser runs on synthetic ``*``-card text,
and the d3plot extractors run on synthetic ``arrays`` dicts that mimic the
shape of ``lasso.dyna.D3plot.arrays``. No d3plot binary or large file is
required, per PRINCIPLES.md.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from structbench.core import read_case, write_case
from structbench.core.io.lsdyna import (
    build_case,
    extract_geometry,
    extract_response,
    parse_deck_materials,
    unit_factors,
)

_GMMMS = unit_factors("g-mm-ms")


def _synthetic_arrays() -> dict:
    """A 2-frame case: 3 SPH particles (part 1) + 1 viz shell (part 2).

    Mimics the shape of ``lasso.dyna.D3plot.arrays`` for a 2D model (z == 0).
    Like lasso, ``node_displacement`` holds the *deformed position* per state
    (frame 0 == initial coords), not displacement-from-initial.
    """
    coords = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [2, 0, 0],
            [3, 0, 0],
            [3, 1, 0],
            [2, 1, 0],
        ],
        dtype=float,
    )
    vel = np.zeros((2, 7, 3))
    vel[..., 0] = -100.0  # mm/ms == -100 m/s in SI
    # frame 0 = initial position; frame 1 shifted +0.5 mm in x.
    position = np.stack([coords, coords + [0.5, 0.0, 0.0]])
    return {
        "node_coordinates": coords,
        "node_ids": np.array([1, 2, 3, 4, 5, 6, 7]),
        "sph_node_indexes": np.array([0, 1, 2]),
        "sph_node_material_index": np.array([0, 0, 0]),
        "part_titles_ids": np.array([1, 2]),
        "element_shell_node_indexes": np.array([[3, 4, 5, 6]]),
        "element_shell_ids": np.array([100]),
        "element_shell_part_indexes": np.array([1]),
        "element_solid_node_indexes": np.zeros((0, 8), dtype=int),
        "timesteps": np.array([0.0, 0.002]),
        "node_displacement": position,
        "node_velocity": vel,
        "sph_stress": np.ones((2, 3, 6)),
        "sph_pressure": np.full((2, 3), 2.0),
        "sph_deletion": np.zeros((2, 3), dtype=bool),
        "global_kinetic_energy": np.array([5.0, 6.0]),
    }


def _row(*vals: object) -> str:
    """Format values as an LS-DYNA fixed-width (10-col) data line."""
    return "".join(f"{v:>10}" for v in vals)


def _taylor_like_deck() -> str:
    """A minimal deck with a part linking one MAT to an EOS and hourglass."""
    return "\n".join(
        [
            "*KEYWORD",
            "*PART",
            "$#  title",
            "Copper bar",
            "$#  pid     secid       mid     eosid      hgid",
            _row(1, 1, 2, 2, 1),
            "*SECTION_SPH",
            "$#  secid      cslh",
            _row(1, 1.2),
            "*MAT_ELASTIC_PLASTIC_HYDRO",
            "$#  mid        ro         g      sigy",
            _row(2, 0.0089, 37590.0, 0.0),
            "$#  eps1      eps2      eps3",
            _row(0.0, 0.5, 1.0),
            "$#  es1       es2       es3",
            _row(199.3, 251.1, 250.9),
            "*EOS_GRUNEISEN",
            "$#  eosid         c        s1",
            _row(2, 3940.0, 1.489),
            "*HOURGLASS",
            "$#  hgid       ihq        qm",
            _row(1, 0, 0.1),
            "*END",
        ]
    )


def test_unit_factors_g_mm_ms_matches_hand_derivation():
    f = unit_factors("g-mm-ms")
    assert math.isclose(f["length"], 1e-3)
    assert math.isclose(f["velocity"], 1.0)  # mm/ms == m/s
    assert math.isclose(f["acceleration"], 1e3)
    assert math.isclose(f["density"], 1e6)  # g/mm^3 -> kg/m^3
    assert math.isclose(f["stress"], 1e6)  # MPa -> Pa
    assert math.isclose(f["pressure"], 1e6)
    assert math.isclose(f["energy"], 1e-3)  # g*mm^2/ms^2 == mJ -> J
    assert math.isclose(f["time"], 1e-3)
    assert math.isclose(f["strain"], 1.0)  # dimensionless


def test_unit_factors_si_is_identity():
    f = unit_factors("kg-m-s")
    for q in (
        "length",
        "velocity",
        "acceleration",
        "density",
        "stress",
        "energy",
        "time",
        "strain",
        "mass",
        "force",
    ):
        assert math.isclose(f[q], 1.0), q


def test_unit_factors_rejects_unknown_token():
    with pytest.raises(ValueError):
        unit_factors("g-furlong-ms")


def test_parse_deck_materials_extracts_one_material():
    mats = parse_deck_materials(_taylor_like_deck())
    assert len(mats) == 1
    m = mats[0]
    assert m.material_id == 2
    assert m.source_model == "MAT_ELASTIC_PLASTIC_HYDRO"
    assert m.canonical_model == "elastic_plastic_hydro"


def test_parse_deck_materials_captures_raw_fields_verbatim():
    m = parse_deck_materials(_taylor_like_deck())[0]
    data = m.source_params["data"]
    assert data[0][0] == 2.0  # mid
    assert data[0][1] == pytest.approx(0.0089)  # ro
    assert data[0][2] == pytest.approx(37590.0)  # g
    assert data[1] == pytest.approx([0.0, 0.5, 1.0])  # eps row


def test_parse_deck_materials_links_eos_and_hourglass_via_part():
    m = parse_deck_materials(_taylor_like_deck())[0]
    assert m.source_params["eos"]["source_model"] == "EOS_GRUNEISEN"
    assert m.source_params["eos"]["data"][0][1] == pytest.approx(3940.0)  # c
    assert m.source_params["hourglass"]["source_model"] == "HOURGLASS"
    assert m.source_params["hourglass"]["data"][0][2] == pytest.approx(0.1)  # qm


def test_parse_deck_materials_blank_part_field_does_not_shift_linkage():
    """A blank interior *PART field must not slide later fields left.

    Here ``eosid`` is blank (LS-DYNA default 0 -> no EOS) and ``hgid`` is 2.
    If blank columns are dropped, ``hgid`` slides into the ``eosid`` slot and
    the wrong EOS is linked while the hourglass is lost (ADR-0016 provenance).
    """
    deck = "\n".join(
        [
            "*KEYWORD",
            "*PART",
            "$#  title",
            "Blank-eos part",
            "$#  pid     secid       mid     eosid      hgid",
            _row(1, 1, 2, "", 2),  # eosid blank, hgid 2
            "*MAT_ELASTIC",
            _row(2, 0.0089, 37590.0),
            "*EOS_GRUNEISEN",
            _row(2, 3940.0, 1.489),  # eosid 2 -- must NOT be linked
            "*HOURGLASS",
            _row(2, 0, 0.1),  # hgid 2 -- must be linked
            "*END",
        ]
    )
    m = parse_deck_materials(deck)[0]
    assert "eos" not in m.source_params  # blank eosid -> no EOS
    assert m.source_params["hourglass"]["source_model"] == "HOURGLASS"


def test_parse_deck_materials_reads_every_part_under_one_card():
    """A single *PART card may define several parts; each must be linked."""
    deck = "\n".join(
        [
            "*KEYWORD",
            "*PART",
            "part A",
            _row(1, 1, 2, 2, 0),  # mid 2 -> eos 2
            "part B",
            _row(3, 1, 4, 4, 0),  # mid 4 -> eos 4
            "*MAT_ELASTIC",
            _row(2, 0.0089, 100.0),
            "*MAT_ELASTIC",
            _row(4, 0.0089, 200.0),
            "*EOS_GRUNEISEN",
            _row(2, 3940.0, 1.489),
            "*EOS_GRUNEISEN",
            _row(4, 5000.0, 1.6),
            "*END",
        ]
    )
    mats = {m.material_id: m for m in parse_deck_materials(deck)}
    assert mats[2].source_params["eos"]["data"][0][1] == pytest.approx(3940.0)
    assert mats[4].source_params["eos"]["data"][0][1] == pytest.approx(5000.0)


def test_numeric_fields_fills_blank_interior_columns_with_zero():
    """Blank interior fixed-width columns parse as 0.0; trailing blanks trim."""
    from structbench.core.io.lsdyna import _numeric_fields

    assert _numeric_fields(_row(1, 1, 2, "", 2)) == [1.0, 1.0, 2.0, 0.0, 2.0]
    assert _numeric_fields(_row(1, 2, "", "")) == [1.0, 2.0]  # trailing trimmed
    assert _numeric_fields(_row("copper", 1)) is None  # title line still detected


def test_parse_deck_materials_handles_no_material():
    assert parse_deck_materials("*KEYWORD\n*NODE\n*END\n") == []


def test_extract_geometry_nodes_reduced_to_dimension_and_si():
    nodes, _ = extract_geometry(_synthetic_arrays(), 2, _GMMMS)
    assert nodes.coords.shape == (7, 2)
    assert nodes.coords.dtype == np.float64
    np.testing.assert_allclose(nodes.coords[1], [1e-3, 0.0])  # 1 mm -> 1e-3 m
    np.testing.assert_array_equal(nodes.node_id, [1, 2, 3, 4, 5, 6, 7])


def test_extract_geometry_sph_and_shell_blocks_with_part_ids():
    _, elements = extract_geometry(_synthetic_arrays(), 2, _GMMMS)
    assert set(elements) == {"sph", "shell"}  # empty solid block skipped
    sph = elements["sph"]
    np.testing.assert_array_equal(sph.connectivity, [[0], [1], [2]])
    np.testing.assert_array_equal(sph.element_id, [1, 2, 3])  # particle node ids
    np.testing.assert_array_equal(sph.part_id, [1, 1, 1])  # part_titles_ids[0]
    shell = elements["shell"]
    np.testing.assert_array_equal(shell.connectivity, [[3, 4, 5, 6]])
    np.testing.assert_array_equal(shell.part_id, [2])  # part_titles_ids[1]


def test_extract_geometry_rejects_3d_model_labelled_2d():
    n = 50
    coords = np.zeros((n, 3))
    coords[:, 0] = np.arange(n)
    coords[:, 2] = np.linspace(0.0, 5.0, n)  # z varies across all nodes -> 3D
    arrays = {"node_coordinates": coords, "node_ids": np.arange(1, n + 1)}
    with pytest.raises(ValueError, match="3D|plane"):
        extract_geometry(arrays, 2, _GMMMS)


def test_extract_geometry_allows_sparse_offplane_viz_nodes():
    n = 500
    coords = np.zeros((n, 3))
    coords[:, 0] = np.arange(n)
    coords[:4, 2] = [1.785, -1.785, 1.785, -1.785]  # 4/500 off-plane viz nodes
    arrays = {"node_coordinates": coords, "node_ids": np.arange(1, n + 1)}
    nodes, _ = extract_geometry(arrays, 2, _GMMMS)  # must not raise
    assert nodes.coords.shape == (n, 2)


def test_extract_geometry_constant_offset_plane_is_allowed():
    arrays = _synthetic_arrays()
    arrays["node_coordinates"] = arrays["node_coordinates"].copy()
    arrays["node_coordinates"][:, 2] = 5.0  # planar, just offset in z
    nodes, _ = extract_geometry(arrays, 2, _GMMMS)
    assert nodes.coords.shape == (7, 2)


def test_extract_response_time_and_node_fields_si():
    resp = extract_response(_synthetic_arrays(), 2, _GMMMS)
    np.testing.assert_allclose(resp.time, [0.0, 2e-6])  # 0.002 ms -> s
    assert resp.node["displacement"].shape == (2, 7, 2)
    assert resp.node["displacement"].dtype == np.float32
    np.testing.assert_allclose(resp.node["velocity"][..., 0], -100.0)  # mm/ms->m/s


def test_extract_response_displacement_is_delta_from_initial():
    resp = extract_response(_synthetic_arrays(), 2, _GMMMS)
    disp = resp.node["displacement"]
    np.testing.assert_allclose(disp[0], 0.0, atol=1e-9)  # frame 0: no displacement
    np.testing.assert_allclose(disp[1, :, 0], 0.5e-3)  # +0.5 mm -> 5e-4 m
    np.testing.assert_allclose(disp[1, :, 1], 0.0, atol=1e-9)


def test_extract_response_sph_fields_keep_voigt_and_convert_units():
    resp = extract_response(_synthetic_arrays(), 2, _GMMMS)
    sph = resp.element["sph"]
    assert sph["stress"].shape == (2, 3, 6)  # full 6-component Voigt kept
    np.testing.assert_allclose(sph["stress"], 1e6)  # MPa -> Pa
    np.testing.assert_allclose(sph["pressure"], 2e6)
    assert sph["deletion"].dtype == np.float32  # bool -> float32


def test_extract_response_global_scalar_energy_converted():
    resp = extract_response(_synthetic_arrays(), 2, _GMMMS)
    np.testing.assert_allclose(resp.globals_["kinetic_energy"], [5e-3, 6e-3])


def test_build_case_assembles_valid_case():
    case = build_case(
        _synthetic_arrays(),
        _taylor_like_deck(),
        source_units="g-mm-ms",
        dimension=2,
        case_id="taylor-test",
    )
    assert case.metadata.case_id == "taylor-test"
    assert case.metadata.dimension == 2
    assert case.metadata.units_convention == "SI"
    assert case.metadata.source_units == "g-mm-ms"
    assert case.metadata.source_deck is not None
    assert "*MAT_ELASTIC_PLASTIC_HYDRO" in case.metadata.source_deck
    assert case.metadata.provenance is not None
    assert case.metadata.provenance.solver_name == "LS-DYNA"
    assert set(case.elements) == {"sph", "shell"}
    assert len(case.materials) == 1
    assert case.response is not None


def test_build_case_round_trips_through_hdf5(tmp_path):
    case = build_case(
        _synthetic_arrays(),
        _taylor_like_deck(),
        source_units="g-mm-ms",
        dimension=2,
        case_id="taylor-test",
    )
    path = tmp_path / "case.h5"
    write_case(case, path)  # validates on write
    back = read_case(path)
    np.testing.assert_allclose(back.response.node["velocity"][..., 0], -100.0)
    np.testing.assert_allclose(back.response.element["sph"]["stress"], 1e6)
    assert back.materials[0].canonical_model == "elastic_plastic_hydro"


def test_build_case_rejects_unknown_units():
    with pytest.raises(ValueError):
        build_case(
            _synthetic_arrays(),
            _taylor_like_deck(),
            source_units="g-furlong-ms",
            dimension=2,
            case_id="x",
        )
