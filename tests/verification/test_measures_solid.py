"""Constitutive and new rows on a meshed ``solid`` block (plan 2, Task 7).

A synthetic 2D case in the shape the Abaqus adapter writes: one ``solid``
block of two elements, one material of class ``elastic_plastic_isotropic``
whose hardening table comes from the input -- a sweep that varies the
hardening declares none (Decision 4).
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from structbench.core import (
    AbsenceReason,
    Case,
    ElementBlock,
    EnergyLedger,
    InputFacts,
    MaterialInput,
    Metadata,
    Nodes,
    PartTraits,
    Response,
    RunEvidence,
)
from structbench.verification.measures import measure_case
from structbench.verification.results import Location

_T = 11  # frames
_TABLE = ((0.0, 1.0), (100.0e6, 200.0e6))  # 100 MPa + 100 MPa per unit strain
_PEEQ0 = (0.1, 0.2)
_V0 = -30.0


def _peeq() -> np.ndarray:
    """Monotone plastic strain from the initial hardening, (T, 2)."""
    return (np.array(_PEEQ0) + np.linspace(0.0, 0.5, _T)[:, None]).astype(np.float32)


def _on_surface(peeq: np.ndarray) -> np.ndarray:
    """Uniaxial stress exactly on the flat-ended table, (T, 2, 6)."""
    stress = np.zeros((*peeq.shape, 6), dtype=np.float32)
    stress[..., 0] = np.interp(peeq, *_TABLE)
    return stress


def _case(**edits) -> Case:
    peeq = edits.pop("peeq", _peeq())
    stress = edits.pop("stress", _on_surface(peeq))
    velocity = np.zeros((_T, 6, 2), dtype=np.float32)
    velocity[:, :, 1] = _V0
    velocity = edits.pop("velocity", velocity)
    globals_ = edits.pop(
        "globals_", {"plastic_dissipation": np.linspace(0, 10, _T).astype(np.float32)}
    )
    assert not edits
    return Case(
        metadata=Metadata(case_id="s", dimension=2, source_units="t-mm-s"),
        nodes=Nodes(
            coords=np.array(
                [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float64
            ),
            node_id=np.arange(1, 7, dtype=np.int64),
        ),
        elements={
            "solid": ElementBlock(
                connectivity=np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.int64),
                element_id=np.array([1, 2], dtype=np.int64),
                part_id=np.array([1, 1], dtype=np.int64),
            )
        },
        materials=[],
        response=Response(
            time=np.linspace(0.0, 1.0e-5, _T),
            node={
                "displacement": np.zeros((_T, 6, 2), np.float32),
                "velocity": velocity,
            },
            element={"solid": {"stress": stress, "effective_plastic_strain": peeq}},
            globals_=globals_,
        ),
    )


def _facts(**overrides) -> InputFacts:
    base = InputFacts(
        parts=(PartTraits(1, 1, "solid", True),),
        materials=(
            MaterialInput(
                1, "elastic_plastic_isotropic", 7000.0, None, 2.0e11, 0.3, _TABLE
            ),
        ),
        time_integration="explicit",
        dimension=2,
        plane_strain=False,
        end_time=1.0e-5,
        other_termination_criteria=frozenset(),
        mass_scaling_enabled=False,
        erosion_enabled=False,
        contact_defined=True,
        prescribed_motion_defined=False,
        damping_defined=False,
        rigid_planes=(),
        particle_pairwise_conservative=None,
        smoothing_length_scale_bounds=None,
        energy_terms_computed=None,
        databases_requested=None,
        unparsable=frozenset(),
        solver="abaqus",
        initial_velocity=((frozenset(range(1, 7)), 2, _V0),),
        initial_hardening=((1, _PEEQ0[0]), (2, _PEEQ0[1])),
    )
    return dataclasses.replace(base, **overrides)


def _row(name, case=None, facts=None, run=None):
    result = measure_case(
        case if case is not None else _case(),
        facts if facts is not None else _facts(),
        None,
        case_id="s",
        run=run,
    )
    return next(m for m in result.measurements if m.quantity == name)


# --- constitutive rows on a solid block -------------------------------------


def test_yield_ratio_uses_the_input_table_held_flat_past_its_end() -> None:
    peeq = _peeq()
    peeq[-1, 1] = 1.5  # beyond the last knot at 1.0
    stress = _on_surface(peeq)
    stress[-1, 1, 0] = 210.0e6  # 1.05 of the flat 200 MPa; 0.84 if extrapolated
    row = _row("yield_ratio_max", _case(peeq=peeq, stress=stress))
    assert row.value == pytest.approx(1.05, rel=1e-6)
    assert Location.INPUT in row.locations


def test_a_planted_decrease_and_a_negative_state_are_measured() -> None:
    peeq = _peeq()
    peeq[5, 0] = peeq[4, 0] - 0.01
    assert _row("state_variable_decrease_max", _case(peeq=peeq)).value == pytest.approx(
        0.01, rel=1e-5
    )
    peeq = _peeq()
    peeq[0, 0] = -0.001
    assert _row("state_variable_min", _case(peeq=peeq)).value == pytest.approx(-0.001)


# --- initial_state_matches_input ------------------------------------------------


def test_the_initial_state_matches_the_input() -> None:
    assert _row("initial_state_matches_input").value == pytest.approx(0.0, abs=1e-7)


def test_a_one_percent_initial_velocity_error_measures_one_percent() -> None:
    case = _case()
    case.response.node["velocity"][0, 3, 1] = _V0 * 1.01
    assert _row("initial_state_matches_input", case).value == pytest.approx(
        0.01, rel=1e-4
    )


def test_a_one_percent_initial_hardening_error_measures_one_percent() -> None:
    case = _case()
    case.response.element["solid"]["effective_plastic_strain"][0, 1] *= 1.01
    assert _row("initial_state_matches_input", case).value == pytest.approx(
        0.01, rel=1e-4
    )


def test_no_initial_conditions_is_not_applicable() -> None:
    facts = _facts(initial_velocity=(), initial_hardening=())
    assert _row("initial_state_matches_input", facts=facts).not_applicable


def test_a_reader_that_does_not_parse_initial_conditions_is_the_platforms_gap() -> None:
    facts = _facts(solver="lsdyna", initial_velocity=None, initial_hardening=None)
    row = _row("initial_state_matches_input", facts=facts)
    assert row.absence.reason is AbsenceReason.UNSUPPORTED


def test_an_unresolved_initial_condition_is_unparsable() -> None:
    """Review Focus 3: a target the reader refused is not an empty target."""
    facts = _facts(
        initial_velocity=None,
        unparsable=frozenset({"unresolved_initial_condition_target"}),
    )
    row = _row("initial_state_matches_input", facts=facts)
    assert row.absence.reason is AbsenceReason.UNPARSABLE


# --- plastic_dissipation_late_growth --------------------------------------------


def test_late_plastic_growth_is_the_last_tenth_over_the_final_value() -> None:
    # P = 0, 1, ..., 10: frame floor(0.9 * 11) = 9 holds 9, so (10 - 9) / 10
    assert _row("plastic_dissipation_late_growth").value == pytest.approx(0.1)


def test_no_plastic_dissipation_is_not_applicable() -> None:
    case = _case(globals_={"plastic_dissipation": np.zeros(_T, np.float32)})
    assert _row("plastic_dissipation_late_growth", case).not_applicable


def test_without_the_global_the_source_is_missing() -> None:
    row = _row("plastic_dissipation_late_growth", _case(globals_={}))
    assert row.absence.reason is AbsenceReason.SOURCE_MISSING


# --- sampling_clock_consistent --------------------------------------------------


def _run(times) -> RunEvidence:
    n = len(times)
    ramp = tuple(float(i) for i in range(n))
    ledger = EnergyLedger(
        time=tuple(float(t) for t in times),
        terms={"kinetic": ramp, "internal": ramp, "external_work": (0.0,) * n},
        identity={"kinetic": 1, "internal": 1},
    )
    return RunEvidence(ledger=ledger)


def test_a_ledger_on_the_field_clock_misses_no_frame() -> None:
    run = _run(np.linspace(0.0, 1.0e-5, _T))
    assert _row("sampling_clock_consistent", run=run).value == 0.0


def test_a_frame_without_a_ledger_sample_is_counted() -> None:
    run = _run(np.delete(np.linspace(0.0, 1.0e-5, _T), 4))
    assert _row("sampling_clock_consistent", run=run).value == 1.0


# --- particles unchanged ---------------------------------------------------------

_SPEC = importlib.util.spec_from_file_location(
    "sph_measure_fixture", Path(__file__).resolve().parent / "test_measures.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_SPH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SPH)


@pytest.mark.parametrize(
    "name",
    [
        "yield_ratio_max",
        "yield_saturation_min",
        "state_variable_decrease_max",
        "state_variable_min",
    ],
)
def test_a_particle_case_measures_as_before(name: str) -> None:
    got = _SPH._get(_SPH._run(_SPH._case()), name)
    assert got.value == pytest.approx(_SPH._HEALTHY[name])


def test_a_terminal_artifact_frame_is_not_a_clock_mismatch() -> None:
    """LS-DYNA's state at the termination time is off the output grid; it is
    reported by terminal_artifact_frames, not judged again here."""
    case = _case()
    r = case.response
    r.time = np.append(r.time, r.time[-1] + 0.1 * (r.time[1] - r.time[0]))
    r.node = {k: np.concatenate([v, v[-1:]]) for k, v in r.node.items()}
    r.element = {
        b: {k: np.concatenate([v, v[-1:]]) for k, v in f.items()}
        for b, f in r.element.items()
    }
    r.globals_ = {k: np.append(v, v[-1]) for k, v in r.globals_.items()}
    run = _run(np.linspace(0.0, 1.0e-5, _T))
    assert _row("sampling_clock_consistent", case, run=run).value == 0.0
