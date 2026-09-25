"""Tests for the Abaqus adapter: abaqus-npz/1 -> canonical case (plan 2, Task 5).

The npz is synthetic, written in exactly the layout `odb_export.py` writes
(STANDARD_INPUT_BLOCK.md, "The abaqus-npz/1 intermediate"): one CAX4R element
on four field nodes, a rigid-body reference node (label 5) with no field
output, and three frames of which the last is the solver's duplicate
end-of-step frame.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from structbench.core.io.abaqus import (
    ABAQUS_NPZ_FORMAT,
    abaqus_export_to_case,
    abaqus_ledger,
)
from structbench.core.validation import validate

_ADAPTER_DECK = """*HEADING
toy
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 0.0, 1.0
4, 1.0, 1.0
*ELEMENT, TYPE=CAX4R, ELSET=E
1, 1, 2, 4, 3
*NSET, NSET=ALLN
1, 2, 3, 4
*MATERIAL, NAME=M
*DENSITY
7e-09
*ELASTIC
200000.0, 0.3
*PLASTIC
250.0, 0.0
1250.0, 10.0
*SECTION CONTROLS, NAME=HG_ENHANCED, HOURGLASS=ENHANCED
*SOLID SECTION, ELSET=E, MATERIAL=M, CONTROLS=HG_ENHANCED
*INITIAL CONDITIONS, TYPE=VELOCITY
ALLN, 2, -30000.0
*STEP, NAME=S, NLGEOM=YES
*DYNAMIC, EXPLICIT
, 0.001
*END STEP
"""

_INST = "PART-1-1"
_TERMS = ("ALLAE", "ALLCD", "ALLFD", "ALLIE", "ALLKE", "ALLPD", "ALLSE", "ALLVD")
_TERMS += ("ALLWK", "ETOTAL")


def _arrays() -> dict[str, np.ndarray]:
    """The export as `odb_export.py` lays it out, in the deck's t-mm-s units."""
    t = np.array([0.0, 0.0005, 0.0005])  # the last frame is the end duplicate
    n, e = 4, 1
    rng = np.random.default_rng(0)

    def frames(shape):
        data = rng.normal(size=(3, *shape)).astype(np.float32)
        data[2] = data[1]
        return data

    manifest = {
        "format": ABAQUS_NPZ_FORMAT,
        "odb_sha256": "0" * 64,
        "abaqus_release": "Abaqus/Explicit 2025",
        "precision": "DOUBLE_PRECISION",
        "materials": ["M"],
        "sections": ["Section-E"],
        "fields": {},
        "skipped": [],
    }
    out = {
        "manifest": np.array(json.dumps(manifest)),
        f"mesh/{_INST}/node_labels": np.array([1, 2, 3, 4, 5]),
        f"mesh/{_INST}/node_coords": np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, -1, 0]], dtype=float
        ),
        f"mesh/{_INST}/elements/CAX4R/labels": np.array([1]),
        f"mesh/{_INST}/elements/CAX4R/connectivity": np.array([[1, 2, 4, 3]]),
        "mesh/WALL/node_labels": np.zeros(0, dtype=np.int64),
        "mesh/WALL/node_coords": np.zeros(0),
        "step/S/frame_times": t,
    }
    for name in ("U", "V", "A"):
        out[f"field/S/{name}/{_INST}/data"] = frames((n, 2))
        out[f"field/S/{name}/{_INST}/node_labels"] = np.array([1, 2, 3, 4])
    for name, width in (("S", 4), ("PEEQ", 1)):
        out[f"field/S/{name}/{_INST}/data"] = frames((e, width))
        out[f"field/S/{name}/{_INST}/element_labels"] = np.array([1])
        out[f"field/S/{name}/{_INST}/integration_points"] = np.array([1])
    for k, term in enumerate(_TERMS):
        values = np.array([0.0, 1.0 + k, 1.0 + k])
        out[f"history/S/Assembly Assembly-1/{term}"] = np.stack([t, values], -1)
    out[f"history/S/Node {_INST}.5/RF2"] = np.stack([t, [0.0, 7.0, 7.0]], -1)
    return out


def _case(tmp_path, arrays=None):
    path = tmp_path / "toy.npz"
    np.savez(path, **(arrays if arrays is not None else _arrays()))
    return abaqus_export_to_case(
        path, _ADAPTER_DECK, source_units="t-mm-s", dimension=2, case_id="toy"
    )


def test_units_and_shapes(tmp_path):
    a = _arrays()
    case = _case(tmp_path, a)
    r = case.response
    np.testing.assert_allclose(
        case.nodes.coords, a[f"mesh/{_INST}/node_coords"][:4, :2] * 1e-3
    )
    u = a[f"field/S/U/{_INST}/data"]
    np.testing.assert_allclose(r.node["displacement"], u[:2] * 1e-3, rtol=1e-6)
    v = a[f"field/S/V/{_INST}/data"]
    np.testing.assert_allclose(r.node["velocity"], v[:2] * 1e-3, rtol=1e-6)
    s = a[f"field/S/S/{_INST}/data"][:2]
    stress = r.element["solid"]["stress"]
    assert stress.shape == (2, 1, 6)
    np.testing.assert_allclose(stress[..., :4], s * 1e6, rtol=1e-6)
    assert not stress[..., 4:].any()  # yz = zx = 0 in a 2D state
    peeq = r.element["solid"]["effective_plastic_strain"]
    np.testing.assert_allclose(peeq, a[f"field/S/PEEQ/{_INST}/data"][:2, :, 0])
    np.testing.assert_allclose(r.globals_["kinetic_energy"], [0.0, 5.0 * 1e-3])  # mJ
    assert r.globals_["kinetic_energy"].dtype == np.float32
    assert set(r.globals_) >= {
        "hourglass_energy",
        "plastic_dissipation",
        "external_work",
    }
    np.testing.assert_allclose(case.elements["solid"].connectivity, [[0, 1, 3, 2]])


def test_the_duplicate_end_frame_is_dropped(tmp_path):
    case = _case(tmp_path)
    np.testing.assert_allclose(case.response.time, [0.0, 0.0005])
    assert case.response.node["displacement"].shape[0] == 2


@pytest.mark.parametrize(
    "key", [f"field/S/S/{_INST}/data", "history/S/Assembly Assembly-1/ALLKE"]
)
def test_a_duplicate_that_differs_is_refused(tmp_path, key):
    a = _arrays()
    a[key] = a[key].copy()
    a[key][-1, ..., -1] += 1.0
    with pytest.raises(ValueError, match="duplicate"):
        _case(tmp_path, a)


def test_the_reference_node_is_a_global_reaction(tmp_path):
    case = _case(tmp_path)
    assert case.nodes.node_id.tolist() == [1, 2, 3, 4]
    # One reference node: a name that does not move with the mesh numbering,
    # so one declared field list fits every case of a sweep.
    np.testing.assert_allclose(
        case.response.globals_["reaction_force_2_reference_node"], [0, 7]
    )


def test_a_history_clock_off_the_frame_clock_is_refused(tmp_path):
    a = _arrays()
    key = "history/S/Assembly Assembly-1/ALLIE"
    a[key] = a[key].copy()
    a[key][1, 0] += 1e-9
    a[key][2, 0] += 1e-9
    with pytest.raises(ValueError, match="clock"):
        _case(tmp_path, a)


def test_an_unknown_format_is_refused(tmp_path):
    a = _arrays()
    a["manifest"] = np.array(json.dumps({"format": "abaqus-npz/0"}))
    with pytest.raises(ValueError, match="abaqus-npz"):
        _case(tmp_path, a)


def test_metadata_and_materials(tmp_path):
    case = _case(tmp_path)
    p = case.metadata.provenance
    assert (p.solver_name, p.solver_version) == ("Abaqus", "2025")
    assert case.metadata.source_deck == _ADAPTER_DECK
    assert case.metadata.source_units == "t-mm-s"
    validate(case)
    (m,) = case.materials
    assert m.canonical_model == "elastic_plastic_isotropic"
    assert m.source_model == "ABAQUS"
    assert m.source_params["density"] == pytest.approx(7000.0)
    assert m.source_params["youngs_modulus"] == pytest.approx(2e11)
    assert m.source_params["yield_table"] == [[0.0, 10.0], [250e6, 1250e6]]
    assert case.elements["solid"].part_id.tolist() == [1]


def test_two_integration_points_are_not_guessed(tmp_path):
    a = _arrays()
    a[f"field/S/S/{_INST}/integration_points"] = np.array([2])
    with pytest.raises(NotImplementedError):
        _case(tmp_path, a)


def test_the_end_frame_acceleration_is_the_grid_frames(tmp_path):
    """A real end-of-step frame writes A afresh: it differed in every run of a
    2026-09-24 sweep, by up to 64 % of its peak. Frame N is kept: it is on the
    output grid and made the way every other frame is."""
    a = _arrays()
    key = f"field/S/A/{_INST}/data"
    a[key] = a[key].copy()
    a[key][-1] *= 1.04
    case = _case(tmp_path, a)
    np.testing.assert_allclose(
        case.response.node["acceleration"][-1], a[key][1] * 1e-3, rtol=1e-6
    )


def test_an_end_frame_equal_to_float32_storage_is_a_duplicate(tmp_path):
    """A few percent of real end frames re-store a value or two a few ulp
    apart (at most 1.8e-7 of the field's peak); that is still the same state."""
    a = _arrays()
    key = f"field/S/S/{_INST}/data"
    a[key] = a[key].copy()
    a[key][-1, 0, 0] = np.nextafter(a[key][-2, 0, 0], np.float32(np.inf))
    case = _case(tmp_path, a)
    np.testing.assert_array_equal(
        case.response.element["solid"]["stress"][-1, 0, 0], a[key][1, 0, 0] * 1e6
    )


# --- the energy ledger (plan 2, Task 6) -------------------------------------


def _with(a, **terms):
    """`a` with Assembly history terms set to (0, v, v) in mJ."""
    t = a["step/S/frame_times"]
    for term, v in terms.items():
        a[f"history/S/Assembly Assembly-1/{term}"] = np.stack([t, [0.0, v, v]], -1)
    return a


def _ledger(tmp_path, a):
    path = tmp_path / "toy.npz"
    np.savez(path, **a)
    return abaqus_ledger(path, source_units="t-mm-s")


def test_the_ledger_maps_the_established_identity(tmp_path):
    """ETOTAL = ALLKE + ALLIE + ALLVD + ALLFD + ALLCD - ALLWK - ALLPW closed to
    6.6e-8 of the initial kinetic energy on a diagnostic run requesting every
    energy variable (STANDARD_INPUT_BLOCK.md, "Energy identity")."""
    a = _with(_arrays(), ALLCD=0.0, ALLPW=3.0)
    ledger = _ledger(tmp_path, a)
    assert set(ledger.terms) == {
        "kinetic",
        "internal",
        "damping",
        "contact",
        "zero_energy_mode",
        "external_work",
    }
    assert dict(ledger.identity) == {
        "kinetic": 1,
        "internal": 1,
        "damping": 1,
        "contact": 1,
    }
    assert ledger.time == pytest.approx((0.0, 0.0005))  # duplicate dropped
    fd = 1.0 + _TERMS.index("ALLFD")
    assert ledger.terms["contact"] == pytest.approx((0.0, (fd - 3.0) * 1e-3))
    ke, total, work = (1.0 + _TERMS.index(n) for n in ("ALLKE", "ETOTAL", "ALLWK"))
    assert ledger.terms["kinetic"] == pytest.approx((0.0, ke * 1e-3))
    assert ledger.solver_total == pytest.approx((0.0, (total + work) * 1e-3))


def test_without_the_penalty_work_there_is_no_contact_term(tmp_path):
    """The 2026-09-24 production decks did not request ALLPW: the ledger then
    carries no contact term, so the balance rows read source_missing."""
    ledger = _ledger(tmp_path, _with(_arrays(), ALLCD=0.0))
    assert "contact" not in ledger.terms
    assert "contact" not in ledger.identity


@pytest.mark.parametrize("term", ["ALLCD", "ALLCW", "ALLMW"])
def test_a_term_the_identity_was_not_closed_with_is_not_guessed(tmp_path, term):
    """Only zero-valued ALLCD, ALLCW, ALLMW were in the closing run."""
    assert _ledger(tmp_path, _with(_arrays(), ALLPW=3.0, **{term: 2.0})) is None


def test_an_export_without_etotal_has_no_ledger(tmp_path):
    a = _arrays()
    del a["history/S/Assembly Assembly-1/ETOTAL"]
    assert _ledger(tmp_path, a) is None


def test_several_reference_nodes_keep_their_labels(tmp_path):
    a = _arrays()
    a[f"mesh/{_INST}/node_labels"] = np.array([1, 2, 3, 4, 5, 6])
    a[f"mesh/{_INST}/node_coords"] = np.vstack(
        [a[f"mesh/{_INST}/node_coords"], [0, -2, 0]]
    )
    t = a["step/S/frame_times"]
    a[f"history/S/Node {_INST}.6/RF2"] = np.stack([t, [0.0, 3.0, 3.0]], -1)
    names = set(_case(tmp_path, a).response.globals_)
    assert {"reaction_force_2_node_5", "reaction_force_2_node_6"} <= names
