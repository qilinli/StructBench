"""Combined hardening from the deck to the canonical file and the yield property.

The fixture ``fixtures/cyclic_element_export.{inp,npz}`` is a real run: one
pilot case of the shipped example ``abaqus_cyclic_element`` (one CAX4R element,
one cycle of axial strain, initial plastic strain 0.01 and axial backstress
40 MPa) solved by Abaqus/Explicit 2025 and exported by ``odb_export.py``.
"""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from structbench.benchmarks import (  # noqa: E402
    COMBINED_HARDENING_STATE_CHANNELS,
    CombinedHardeningLaw,
    ConstitutiveLaw,
)
from structbench.cli.evaluation.rollout import _state_constraint_properties
from structbench.cli.training.common import _case_law_rows
from structbench.cli.training.transolver_fm import _law_tuple
from structbench.core.io.abaqus import abaqus_export_to_case  # noqa: E402
from structbench.core.io.abaqus_run import read_abaqus_input_facts  # noqa: E402
from structbench.datagen import definition, sampling  # noqa: E402
from structbench.datagen.abaqus import deck  # noqa: E402
from structbench.datasets.canonical import (  # noqa: E402
    _AUX_FIELDS,
    aux_channel_labels,
    available_aux_fields,
)
from structbench.eval import (  # noqa: E402
    combined_yield_stress,
    yield_violation_rate_about_backstress,
)

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLE = resources.files("structbench.datagen") / "examples" / "abaqus_cyclic_element"
#: The example's law, in MPa (its dataset.toml).
LAW = CombinedHardeningLaw(210000.0, 0.3, 300.0, 20000.0, 100.0, 50.0, 10.0)
LABELS = (
    *("s_rr", "s_zz", "s_tt", "s_rz"),
    "effective_plastic_strain",
    *("a_rr", "a_zz", "a_tt", "a_rz"),
)


def _fixture_case():
    return abaqus_export_to_case(
        FIXTURES / "cyclic_element_export.npz",
        (FIXTURES / "cyclic_element_export.inp").read_text(encoding="utf-8"),
        source_units="t-mm-s",
        dimension=2,
        case_id="cyclic",
    )


def _fixture_state() -> np.ndarray:
    """The stored element state of the real run in the working frame (MPa)."""
    solid = _fixture_case().response.element["solid"]
    extract = {name: _AUX_FIELDS[name].extractor for name in LABELS_BY_FIELD}
    blocks = [extract[name](solid, 1e-6) for name in LABELS_BY_FIELD]
    return np.concatenate([b if b.ndim == 3 else b[..., None] for b in blocks], -1)


LABELS_BY_FIELD = COMBINED_HARDENING_STATE_CHANNELS


# --- item 1: the deck ---------------------------------------------------------


def test_the_combined_hardening_material_card():
    text = deck.material_combined_hardening(
        "M",
        density=7.85e-9,
        youngs=210000.0,
        poisson=0.3,
        yield_stress=300.0,
        kinematic_modulus=20000.0,
        kinematic_rate=100.0,
        isotropic_saturation=50.0,
        isotropic_rate=10.0,
    )
    assert text == (
        "*MATERIAL, NAME=M\n*DENSITY\n7.85e-09\n*ELASTIC\n210000.0, 0.3\n"
        "*PLASTIC, HARDENING=COMBINED, DATATYPE=PARAMETERS, NUMBER BACKSTRESSES=1\n"
        "300.0, 20000.0, 100.0\n"
        "*CYCLIC HARDENING, PARAMETERS\n"
        "300.0, 50.0, 10.0\n"
    )


def test_initial_hardening_takes_a_backstress_beside_the_plastic_strain():
    alpha = np.array([[0.0, 40.0, 0.0, 0.0], [1.5, -2.0, 0.5, 3.0]])
    text = deck.initial_hardening([1, 2], np.array([0.01, 0.0]), alpha)
    assert text == (
        "*INITIAL CONDITIONS, TYPE=HARDENING\n"
        "1, 0.01, 0.0, 40.0, 0.0, 0.0\n"
        "2, 0.0, 1.5, -2.0, 0.5, 3.0\n"
    )
    # Without one the card is what it always was.
    assert deck.initial_hardening([1], np.array([0.15])) == (
        "*INITIAL CONDITIONS, TYPE=HARDENING\n1, 0.15\n"
    )
    with pytest.raises(ValueError, match="4 or 6"):
        deck.initial_hardening([1], np.array([0.0]), np.zeros((1, 3)))


def test_the_amplitude_and_the_prescribed_displacement():
    points = [(0.0, 0.0), (0.5, 1.0), (1.5, -1.0), (2.0, 0.0), (3.0, 0.5)]
    assert deck.amplitude("CYCLE", points) == (
        "*AMPLITUDE, NAME=CYCLE\n0.0, 0.0, 0.5, 1.0, 1.5, -1.0, 2.0, 0.0\n3.0, 0.5\n"
    )
    assert deck.prescribed_displacement("TOP", 2, 0.02, amplitude="CYCLE") == (
        "*BOUNDARY, TYPE=DISPLACEMENT, AMPLITUDE=CYCLE\nTOP, 2, 2, 0.02\n"
    )


def test_the_cyclic_example_writes_the_fixtures_deck():
    """The shipped example, at its second pilot point, is the deck the real
    run was solved from: the fixture is the example's own output."""
    with resources.as_file(EXAMPLE) as folder:
        d, problem = definition.load_definition(folder), definition.load_problem(folder)
    assert d.name == "abaqus_cyclic_element"
    point = sampling.sample_split(d.variables, d.regions, d.split("pilot"))[1]
    params = {**d.fixed, **point.params, d.levels.refine_key: d.levels.production}
    text = problem.input_deck(params, None)
    assert text == problem.input_deck(params, None)  # byte-stable
    stored = (FIXTURES / "cyclic_element_export.inp").read_text(encoding="utf-8")
    assert (
        hashlib.sha256(text.encode()).digest()
        == hashlib.sha256(stored.encode()).digest()
    )
    assert problem.feasible(params)
    assert not problem.feasible({**params, "alpha0": 400.0})  # outside the surface


# --- item 2: the export, the canonical file, the channel ---------------------------


def test_the_real_export_stores_the_backstress_as_the_solver_wrote_it():
    case = _fixture_case()
    solid = case.response.element["solid"]
    assert sorted(solid) == ["backstress", "effective_plastic_strain", "stress"]
    back = solid["backstress"]
    assert back.shape == (101, 1, 6) and back.dtype == np.float32
    # Frame 0 is the deck's initial state: 40 MPa axial, in Pa, component 22.
    np.testing.assert_allclose(back[0, 0], [0.0, 40.0e6, 0.0, 0.0, 0.0, 0.0])
    assert solid["effective_plastic_strain"][0, 0] == pytest.approx(0.01)
    # The solver's backstress is not deviatoric: its trace is far from zero.
    assert np.abs(back[..., :3].sum(-1)).max() > 100.0e6
    assert not np.any(back[..., 4:])


def test_the_backstress_channel_is_registered_beside_the_stress():
    assert "backstress_axisymmetric" in available_aux_fields()
    assert aux_channel_labels("backstress_axisymmetric") == (
        *("a_rr", "a_zz", "a_tt", "a_rz"),
    )
    assert COMBINED_HARDENING_STATE_CHANNELS == (
        "stress_axisymmetric",
        "effective_plastic_strain",
        "backstress_axisymmetric",
    )
    state = _fixture_state()
    assert state.shape == (101, 1, 9)
    assert state[0, 0, 5:].tolist() == [0.0, 40.0, 0.0, 0.0]  # MPa


def test_the_deck_reader_says_what_it_did_not_read():
    deck_text = (FIXTURES / "cyclic_element_export.inp").read_text(encoding="utf-8")
    facts = read_abaqus_input_facts(deck_text, source_units="t-mm-s")
    # The plastic strain is read; the backstress beside it is named as unread,
    # and so are the two cards of a law the reader has no class for.
    assert facts.initial_hardening == ((1, 0.01),)
    for token in (
        "unread_card:HARDENING_BACKSTRESS",
        "unread_card:PLASTIC",
        "unread_card:CYCLIC_HARDENING",
    ):
        assert token in facts.unparsable, token
    assert facts.materials[0].canonical_model is None


# --- item 3: the law -------------------------------------------------------------


def test_the_combined_hardening_law_validates_its_parameters():
    assert LAW.shear_modulus == pytest.approx(210000.0 / 2.6)
    assert LAW.bulk_modulus == pytest.approx(175000.0)
    for bad in (
        {"youngs_modulus": 0.0},
        {"poisson_ratio": 0.5},
        {"yield_stress": 0.0},
        {"kinematic_modulus": -1.0},
        {"kinematic_rate": -1.0},
        {"isotropic_rate": -1.0},
        {"isotropic_saturation": -300.0},
    ):
        fields = {
            **{f: getattr(LAW, f) for f in LAW.__dataclass_fields__},
            **bad,
        }
        with pytest.raises(ValueError):
            CombinedHardeningLaw(**fields)
    # Cyclic softening is a negative saturation above minus the yield stress.
    assert CombinedHardeningLaw(2e5, 0.3, 300.0, 0.0, 0.0, -100.0, 5.0)


def test_the_real_run_follows_the_law_as_described():
    state = _fixture_state().astype(np.float64)
    stress, peeq, back = state[..., :4], state[..., 4], state[..., 5:]
    sy = combined_yield_stress(
        peeq, LAW.yield_stress, LAW.isotropic_saturation, LAW.isotropic_rate
    )
    d = stress - back
    vm = np.sqrt(
        0.5
        * (
            (d[..., 0] - d[..., 1]) ** 2
            + (d[..., 1] - d[..., 2]) ** 2
            + (d[..., 2] - d[..., 0]) ** 2
        )
        + 3.0 * d[..., 3] ** 2
    )
    yielding = np.diff(peeq[:, 0], prepend=peeq[0, 0]) > 0
    assert yielding.sum() > 50
    # On the surface while yielding, inside it otherwise, to float32 storage.
    np.testing.assert_allclose(vm[yielding, 0] / sy[yielding, 0], 1.0, atol=1e-5)
    assert (vm / sy).max() < 1.0 + 1e-5
    # Read as an isotropic law the same state is far outside the surface.
    d0 = stress
    vm0 = np.sqrt(
        0.5
        * (
            (d0[..., 0] - d0[..., 1]) ** 2
            + (d0[..., 1] - d0[..., 2]) ** 2
            + (d0[..., 2] - d0[..., 0]) ** 2
        )
        + 3.0 * d0[..., 3] ** 2
    )
    assert (vm0 / sy).max() > 1.3


def test_the_isotropic_decoders_refuse_a_combined_law():
    with pytest.raises(ValueError, match="CombinedHardeningLaw"):
        _law_tuple(LAW)
    iso = ConstitutiveLaw(2e5, 0.3, ((0.0, 1.0), (250.0, 350.0)))
    assert _law_tuple(iso) == (2e5, 0.3, (0.0, 1.0), (250.0, 350.0))

    class _Spec:
        constitutive_law = staticmethod(lambda case_id: LAW)

    class _Trajectory:
        case_id = "x"
        state = np.zeros((2, 3, 9))

    assert _case_law_rows(_Spec(), _Trajectory(), "cpu") is None


# --- item 4: the property ----------------------------------------------------------


def test_the_yield_property_is_measured_about_the_backstress():
    state = _fixture_state()
    tolerances = {"yield": 1e-4, "peeq": 1e-6}
    props = _state_constraint_properties(state, LABELS, LAW, tolerances, 1, None)
    # The reference data honours both constraints at tolerances taken from its
    # own measured excess (ADR-0075 clause 6).
    assert props == {"prop_yield_violation_rate": 0.0, "prop_peeq_decrease_rate": 0.0}
    # A stress pushed outside the surface is counted, frame by frame.
    broken = state.copy()
    broken[50:, :, 1] += 600.0
    rate = _state_constraint_properties(broken, LABELS, LAW, tolerances, 1, None)[
        "prop_yield_violation_rate"
    ]
    assert rate is not None and rate > 0.2
    # Moving the stress and the backstress together changes nothing.
    shifted = state.copy()
    shifted[..., [1, 6]] += 250.0
    assert (
        _state_constraint_properties(shifted, LABELS, LAW, tolerances, 1, None)[
            "prop_yield_violation_rate"
        ]
        == 0.0
    )


def test_the_yield_property_is_undefined_without_the_backstress():
    state = _fixture_state()[..., :5]
    props = _state_constraint_properties(
        state, LABELS[:5], LAW, {"yield": 1e-4, "peeq": 1e-6}, 1, None
    )
    assert props["prop_yield_violation_rate"] is None
    assert props["prop_peeq_decrease_rate"] == 0.0


def test_the_kernel_is_blind_to_the_backstress_trace_and_undefined_on_nan():
    rng = np.random.default_rng(0)
    stress = rng.normal(0, 200, (5, 4, 4))
    back = rng.normal(0, 80, (5, 4, 4))
    peeq = np.abs(rng.normal(0, 0.02, (5, 4)))
    args = (peeq, 300.0, 50.0, 10.0, 0.0)
    rate = yield_violation_rate_about_backstress(stress, back, *args)
    hydrostatic = back.copy()
    hydrostatic[..., :3] += 123.0
    assert yield_violation_rate_about_backstress(stress, hydrostatic, *args) == rate
    assert 0.0 < rate < 1.0
    back[0, 0, 0] = np.nan
    assert np.isnan(yield_violation_rate_about_backstress(stress, back, *args))
    assert combined_yield_stress(0.0, 300.0, 50.0, 10.0) == 300.0
    assert combined_yield_stress(1e9, 300.0, 50.0, 10.0) == pytest.approx(350.0)
