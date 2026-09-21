"""Tests for the run-record reader and its records (ADR-0066 clauses 2 and 3).

Synthetic text with invented numbers on the layout of an LS-DYNA R12 MPP
double-precision run (``mpp`` build, 2020): a global-statistics file of
labelled ``name....  value`` blocks and a message file. Nothing here is
copied from a real run.
"""

from __future__ import annotations

import pytest

from structbench.core import (
    EnergyLedger,
    RunEvidence,
    SolverIdentity,
    TerminationRecord,
    read_run_evidence,
)


def _stat(label: str, value: float | int, tail: str = "") -> str:
    number = f"{value:>13d}" if isinstance(value, int) else f"{value:>13.5E}"
    return f" {label:.<31}{number}{tail}"


def _block(
    cycle: int, time: float, kinetic: float, internal: float, wall: float
) -> str:
    total = kinetic + internal + wall
    return "\n".join(
        [
            "",
            f" dt of cycle{cycle:>8} is controlled by sph             1000",
            "",
            _stat("time", time),
            _stat("time step", 8.0e-5 if cycle < 50 else 7.6e-5),
            _stat("kinetic energy", kinetic),
            _stat("internal energy", internal),
            _stat("stonewall energy", wall, " wall#    1"),
            _stat("spring and damper energy", 8.0e-20),
            _stat("system damping energy", 0.0),
            _stat("sliding interface energy", 0.0),
            _stat("external work", 0.0),
            _stat("eroded kinetic energy", 0.0),
            _stat("eroded internal energy", 0.0),
            _stat("eroded hourglass energy", 0.0),
            _stat("total energy", total),
            _stat("total energy / initial energy", total / 2000.0),
            _stat("energy ratio w/o eroded energy.", total / 2000.0),
            _stat("global x velocity", -50.0),
            _stat("global y velocity", 0.0),
            _stat("global z velocity", 0.0),
            _stat("time per zone cycle.(nanosec)", 0),
        ]
    )


_STAMP = "                         ls-dyna mpp.123456 d           date 01/01/2020"
_HEAD = f" an invented impact\n{_STAMP}\n"
_STATS = _HEAD + "".join(
    [
        _block(1, 0.0, 2000.0, 0.0, 0.0),
        _block(26, 0.002, 1500.0, 520.0, 10.0),
        _block(52, 0.004, 100.0, 2000.0, 40.0),
    ]
)

_MESSAGES = "\n".join(
    [
        "     |  Licensed to: An Invented Customer                |",
        "     |  Precision  : Double precision (I8R8)           |",
        "     |  SVN Version: 123999                            |",
        " ************************************************************",
        " * errors encountered in either the documentation or results  *",
        " ************************************************************",
        " Input file: Q:\\secret\\folder\\model.k",
        " hostname build-box-17",
        " MPP execution with       4 procs",
        _STAMP,
        "      52 t 4.0000E-03 dt 7.60E-05 write d3plot file     01/01/20 10:00:00",
        " ",
        " *** termination time reached ***",
        " N o r m a l    t e r m i n a t i o n                   01/01/20 10:00:01",
        " Problem time       =    4.0020E-03",
        " Problem cycle      =        52",
    ]
)


def _read(messages: str | None = _MESSAGES, stats: str | None = _STATS) -> RunEvidence:
    return read_run_evidence(
        messages_text=messages, global_statistics_text=stats, source_units="g-mm-ms"
    )


def test_the_ledger_is_read_in_si_with_its_identity() -> None:
    ledger = _read().ledger
    assert ledger is not None
    assert ledger.time == pytest.approx((0.0, 2.0e-6, 4.0e-6))  # ms -> s
    assert ledger.terms["kinetic"] == pytest.approx((2.0, 1.5, 0.1))  # g mm2/ms2 -> J
    assert ledger.terms["rigid_surface"] == pytest.approx((0.0, 0.01, 0.04))
    # eroded terms are contained in others, external work is the input side
    assert set(ledger.identity) == {
        "kinetic",
        "internal",
        "rigid_surface",
        "damping",
        "contact",
    }
    assert (
        "discrete_element" in ledger.terms and "discrete_element" not in ledger.identity
    )
    assert ledger.solver_total == pytest.approx((2.0, 2.03, 2.14))
    assert ledger.origin == "solver"


def test_the_time_step_history_comes_with_the_ledger() -> None:
    steps = _read().timestep
    assert steps is not None
    assert steps[1] == pytest.approx((8.0e-8, 8.0e-8, 7.6e-8))


def test_two_rigid_surfaces_add_up() -> None:
    doubled = _STATS.replace(
        _stat("stonewall energy", 40.0, " wall#    1"),
        _stat("stonewall energy", 25.0, " wall#    1")
        + "\n"
        + _stat("stonewall energy", 15.0, " wall#    2"),
    )
    ledger = _read(stats=doubled).ledger
    assert ledger is not None
    assert ledger.terms["rigid_surface"][-1] == pytest.approx(0.04)


def test_an_unknown_label_yields_no_ledger_rather_than_a_partial_one() -> None:
    odd = _STATS.replace("system damping energy.", "drilling energy.......")
    evidence = _read(stats=odd)
    assert evidence.ledger is None and evidence.timestep is None
    assert "unknown_ledger_label" in evidence.unparsable


def test_a_total_the_identity_does_not_reproduce_is_refused() -> None:
    # a term the reader does not know to be part of the total carries energy
    skewed = _STATS.replace(
        _stat("total energy", 2140.0), _stat("total energy", 2400.0)
    )
    evidence = _read(stats=skewed)
    assert evidence.ledger is None
    assert "ledger_identity_unverified" in evidence.unparsable
    assert evidence.timestep is not None  # the step history does not depend on it


def test_blocks_that_disagree_on_their_terms_are_refused() -> None:
    lines = _STATS.splitlines()
    last = max(i for i, ln in enumerate(lines) if "sliding interface energy" in ln)
    ragged = "\n".join(lines[:last] + lines[last + 1 :])
    assert "ragged_ledger" in _read(stats=ragged).unparsable


def test_identity_and_termination_are_read_from_whitelisted_lines() -> None:
    evidence = _read()
    assert evidence.identity == SolverIdentity(
        "ls-dyna", "mpp.123456", "123999", "double", "mpp:4"
    )
    assert evidence.termination == (
        TerminationRecord("normal", pytest.approx(4.002e-6), 52, "end_time"),
    )
    assert (evidence.n_errors, evidence.n_warnings) == (
        0,
        0,
    )  # the disclaimer is not one
    assert evidence.unparsable == frozenset()


def test_diagnostics_are_counted_fail_closed() -> None:
    noisy = _MESSAGES.replace(
        " MPP execution",
        " *** Warning 99999 (XYZ+999)\n     something was odd\n"
        " *** Error 88888 (XYZ+888)\n WARNING: a second one\n MPP execution",
    )
    evidence = _read(messages=noisy)
    assert (evidence.n_errors, evidence.n_warnings) == (1, 2)


def test_a_record_that_just_stops_did_not_terminate_normally() -> None:
    cut = _MESSAGES.split(" *** termination time reached")[0]
    (record,) = _read(messages=cut).termination or ()
    assert (record.status, record.criterion, record.n_steps) == ("none", None, None)
    failed = _MESSAGES.replace("N o r m a l ", "E r r o r   ").replace(
        " *** termination time reached ***\n", ""
    )
    (record,) = _read(messages=failed).termination or ()
    assert record.status == "error"


def test_a_normal_end_by_an_unrecognised_criterion_is_flagged() -> None:
    other = _MESSAGES.replace("termination time reached", "termination cycle reached")
    evidence = _read(messages=other)
    assert evidence.termination is not None
    assert evidence.termination[0].criterion is None
    assert "termination_criterion" in evidence.unparsable


def test_nothing_private_in_the_message_file_reaches_the_record() -> None:
    text = repr(_read())
    for leak in ("Invented Customer", "secret", "build-box", "Q:\\", "01/01/20"):
        assert leak not in text


def test_each_source_is_optional() -> None:
    only_stats = _read(messages=None)
    assert only_stats.identity is None and only_stats.termination is None
    assert only_stats.n_errors is None and only_stats.ledger is not None
    only_messages = _read(stats=None)
    assert only_messages.ledger is None and only_messages.identity is not None
    with pytest.raises(ValueError, match="source_units"):
        read_run_evidence(
            messages_text=None, global_statistics_text=None, source_units="SI"
        )


def test_records_refuse_free_text_and_ragged_series() -> None:
    with pytest.raises(ValueError, match="version"):
        SolverIdentity("ls-dyna", version="C:\\path with spaces")
    with pytest.raises(ValueError, match="name"):
        SolverIdentity("LS DYNA")
    with pytest.raises(ValueError, match="criterion"):
        TerminationRecord("normal", criterion="time reached!")
    with pytest.raises(ValueError, match="unknown ledger terms"):
        EnergyLedger((0.0,), {"kinetic": (1.0,), "magic": (0.0,)}, {"kinetic": 1})
    with pytest.raises(ValueError, match="one value per sample"):
        EnergyLedger((0.0, 1.0), {"kinetic": (1.0,)}, {"kinetic": 1})
    with pytest.raises(ValueError, match="identity"):
        EnergyLedger(
            (0.0,), {"kinetic": (1.0,), "external_work": (0.0,)}, {"external_work": 1}
        )
    with pytest.raises(ValueError, match="kinetic"):
        EnergyLedger((0.0,), {"internal": (1.0,)}, {"internal": 1})
    with pytest.raises(ValueError, match="tokens"):
        RunEvidence(unparsable=frozenset({"Not A Token"}))
