"""Tests for the run-evidence measures and the energy indicator (ADR-0066 cl. 7).

Hand-sized ledgers whose indicator values can be checked on paper.
"""

from __future__ import annotations

import numpy as np
import pytest

from structbench.core import (
    AbsenceReason,
    EnergyLedger,
    EvidenceItem,
    InputFacts,
    MaterialInput,
    PartTraits,
    RigidPlane,
    RunEvidence,
    SolverIdentity,
    TerminationRecord,
)
from structbench.verification import Verdict
from structbench.verification.criteria import judge
from structbench.verification.kernels import energy_residual
from structbench.verification.measures import measure_case
from structbench.verification.results import (
    CaseMeasurements,
    DatasetMeasurements,
    Measurement,
)

E = EvidenceItem


def _facts(**overrides: object) -> InputFacts:
    base: dict[str, object] = {
        "parts": (PartTraits(1, 2, "particle", None),),
        "materials": (
            MaterialInput(2, "elastic_plastic_hydro", 8900.0, 4e10, None, None, None),
        ),
        "time_integration": "explicit",
        "dimension": 2,
        "plane_strain": True,
        "end_time": 4.0e-6,
        "other_termination_criteria": frozenset(),
        "mass_scaling_enabled": False,
        "erosion_enabled": False,
        "contact_defined": False,
        "prescribed_motion_defined": False,
        "rigid_planes": (RigidPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        "particle_pairwise_conservative": None,
        "smoothing_length_scale_bounds": None,
        "unparsable": frozenset(),
    }
    return InputFacts(**{**base, **overrides})  # type: ignore[arg-type]


def _ledger(**terms: tuple[float, ...]) -> EnergyLedger:
    """An impact: 2 J of kinetic energy at the start, none from outside."""
    series = {
        "kinetic": (2.0, 1.5, 0.1),
        "internal": (0.0, 0.52, 2.0),
        "rigid_surface": (0.0, 0.01, 0.04),
        "external_work": (0.0, 0.0, 0.0),
        **terms,
    }
    identity = {k: 1 for k in series if k != "external_work"}
    return EnergyLedger((0.0, 2.0e-6, 4.0e-6), series, identity)


def _run(**overrides: object) -> RunEvidence:
    base: dict[str, object] = {
        "identity": SolverIdentity(
            "ls-dyna", "mpp.123456", "123999", "double", "mpp:4"
        ),
        "termination": (TerminationRecord("normal", 4.002e-6, 52, "end_time"),),
        "n_errors": 0,
        "n_warnings": 3,
        "timestep": ((0.0, 2.0e-6, 4.0e-6), (8.0e-8, 6.0e-8, 7.6e-8)),
        "ledger": _ledger(),
    }
    return RunEvidence(**{**base, **overrides})  # type: ignore[arg-type]


def _measure(run: RunEvidence | None, facts: InputFacts | None) -> CaseMeasurements:
    return measure_case(None, facts, None, case_id="r", run=run)  # no case at all


def _get(result: CaseMeasurements, name: str) -> Measurement:
    return next(m for m in result.measurements if m.quantity == name)


# --- the indicator ------------------------------------------------------------


def test_an_impact_reads_zero_at_the_start_not_one() -> None:
    total = np.array([2.0, 2.03, 2.14])
    r = energy_residual(total, np.zeros(3), np.array([2.0, 1.5, 0.1]))
    assert r[0] == 0.0
    assert r[1] == pytest.approx(0.03 / 2.0)  # the initial total is the largest scale
    # at the end the non-kinetic bucket, 2.04, exceeds the initial total
    assert r[2] == pytest.approx(0.14 / 2.04)


def test_a_driven_system_from_rest_is_scaled_by_the_work_put_in() -> None:
    work = np.array([0.0, 1.0, 4.0])
    total = np.array([0.0, 0.98, 3.8])  # 0.02 then 0.2 J unaccounted for
    r = energy_residual(total, work, np.array([0.0, 0.3, 0.5]))
    assert r == pytest.approx([0.0, -0.02, -0.05])


def test_nothing_happening_is_a_zero_residual_not_a_division_by_zero() -> None:
    assert energy_residual(np.zeros(3), np.zeros(3), np.zeros(3)).tolist() == [0.0] * 3


# --- the measures -------------------------------------------------------------


def test_a_run_with_no_case_is_measured_on_its_record() -> None:
    result = _measure(_run(), _facts())
    expected = {
        "terminated_normally": 1.0,
        "solver_error_count": 0.0,
        "solver_warning_count": 3.0,
        "solver_identity_complete": 0.0,
        "timestep_min_ratio": 0.75,
        "energy_gain_max": 0.14 / 2.04,
        "energy_loss_max": 0.0,
        "energy_residual_final": 0.14 / 2.04,
        "total_energy_change_final": 0.07,
    }
    for name, number in expected.items():
        assert _get(result, name).value == pytest.approx(number), name
    assert _get(result, "terminated_normally").detail == {
        "segments": 1,
        "steps": 52,
        "ended_by": "end_time",
    }
    assert _get(result, "timestep_min_ratio").detail["sample"] == 1
    assert _get(result, "energy_gain_max").detail["time"] == pytest.approx(4.0e-6)
    assert _get(result, "nonfinite_count").absence is not None  # needs the case


def test_energy_loss_is_reported_as_a_positive_magnitude() -> None:
    leaking = _ledger(internal=(0.0, 0.40, 1.80))  # totals 2.0, 1.91, 1.94
    result = _measure(_run(ledger=leaking), _facts())
    loss = _get(result, "energy_loss_max")
    assert loss.value == pytest.approx(0.09 / 2.0)
    assert loss.detail["sample"] == 1
    assert _get(result, "energy_gain_max").value == 0.0
    assert _get(result, "energy_residual_final").value == pytest.approx(-0.06 / 2.0)


def test_a_term_the_traits_require_but_the_ledger_lacks_blocks_every_energy_row() -> (
    None
):
    contact = _facts(contact_defined=True)  # the ledger has no contact term
    result = _measure(_run(), contact)
    for name in (
        "energy_gain_max",
        "energy_residual_final",
        "total_energy_change_final",
    ):
        absence = _get(result, name).absence
        assert absence is not None, name
        assert (absence.reason, absence.missing) == (
            AbsenceReason.SOURCE_MISSING,
            {E.E5},
        )
    assert _get(result, "terminated_normally").value == 1.0  # other rows stand


def test_unknown_integration_of_a_meshed_part_requires_the_zero_energy_term() -> None:
    meshed = _facts(parts=(PartTraits(1, 2, "solid", None),))
    assert _get(_measure(_run(), meshed), "energy_gain_max").absence is not None
    full = _facts(parts=(PartTraits(1, 2, "solid", False),))
    assert _get(_measure(_run(), full), "energy_gain_max").value is not None


def test_without_the_input_the_required_terms_are_unknown() -> None:
    absence = _get(_measure(_run(), None), "energy_gain_max").absence
    assert absence is not None and absence.missing == {E.E1}
    assert _get(_measure(_run(), None), "solver_error_count").value == 0.0


def test_no_energy_at_the_start_means_no_start_to_end_ratio() -> None:
    from_rest = EnergyLedger(
        (0.0, 1.0),
        {
            "kinetic": (0.0, 0.5),
            "internal": (0.0, 0.5),
            "rigid_surface": (0.0, 0.0),
            "external_work": (0.0, 1.0),
        },
        {"kinetic": 1, "internal": 1, "rigid_surface": 1},
    )
    result = _measure(_run(ledger=from_rest), _facts())
    assert _get(result, "total_energy_change_final").not_applicable
    assert _get(result, "energy_residual_final").value == 0.0


def test_each_row_names_the_evidence_item_the_run_left_out() -> None:
    bare = _run(identity=None, termination=None, timestep=None, ledger=None)
    result = _measure(bare, _facts())
    for name, item in {
        "solver_identity_complete": E.E2,
        "terminated_normally": E.E3,
        "timestep_min_ratio": E.E4,
        "energy_gain_max": E.E5,
    }.items():
        absence = _get(result, name).absence
        assert absence is not None and absence.missing == {item}, name
        assert absence.reason is AbsenceReason.SOURCE_MISSING


def test_a_record_the_reader_refused_reads_unparsable_not_missing() -> None:
    refused = _run(ledger=None, unparsable=frozenset({"unknown_ledger_label"}))
    absence = _get(_measure(refused, _facts()), "energy_gain_max").absence
    assert absence is not None
    assert (absence.reason, absence.missing) == (AbsenceReason.UNPARSABLE, {E.E5})


def test_an_implicit_run_has_no_explicit_time_step_row() -> None:
    assert _get(
        _measure(_run(), _facts(time_integration="implicit")), "timestep_min_ratio"
    ).not_applicable


def test_an_incomplete_identity_is_counted() -> None:
    thin = _run(identity=SolverIdentity("ls-dyna", "mpp.123456"))
    assert _get(_measure(thin, _facts()), "solver_identity_complete").value == 3.0


# --- judged -------------------------------------------------------------------


def test_the_energy_rows_of_an_unscoped_particle_run_get_no_verdict() -> None:
    case = _measure(_run(n_errors=2), _facts())
    (report,) = judge(DatasetMeasurements(None, None, "0", {}, (case,))).cases
    by_name = {r.quantity: r for r in report.results}
    gain = by_name["energy_gain_max"]
    assert (gain.verdict, gain.reason) == (
        Verdict.NOT_ASSESSABLE,
        "no_ratified_criterion",
    )
    assert len(gain.out_of_scope_levels) == 2  # the levels that exist, for a person
    assert by_name["terminated_normally"].verdict is Verdict.PASS
    assert by_name["solver_identity_complete"].verdict is Verdict.PASS
    assert by_name["solver_error_count"].verdict is Verdict.FAIL
    assert by_name["solver_warning_count"].verdict is Verdict.NOT_ASSESSABLE


def test_a_meshed_explicit_run_is_judged_against_the_sourced_level() -> None:
    meshed = _facts(parts=(PartTraits(1, 2, "solid", False),))
    case = _measure(_run(), meshed)
    (report,) = judge(DatasetMeasurements(None, None, "0", {}, (case,))).cases
    gain = next(r for r in report.results if r.quantity == "energy_gain_max")
    assert (gain.verdict, gain.reason) == (Verdict.REVIEW, "exceeds_reference_level")
    loss = next(r for r in report.results if r.quantity == "energy_loss_max")
    assert loss.verdict is Verdict.PASS
