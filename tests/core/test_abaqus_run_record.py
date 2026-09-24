"""Tests for the Abaqus run-record reader (ADR-0068, ADR-0066 clauses 2-3).

Synthetic text on the layout of an Abaqus/Standard 2025 job. Nothing here is
copied from a real run: the layout was observed once and the numbers are
invented.

Abaqus spreads the run record over three files, and which ones exist is
itself evidence. A job that completes writes `.sta`, `.msg` and `.dat`; a job
the pre-processor rejects writes only `.dat` -- no `.sta`, no `.msg` -- so the
reader takes three optional texts and must not require the two that a failed
job never produces.
"""

from __future__ import annotations

import pytest

from structbench.core.io.abaqus_run import read_abaqus_run_evidence

_STA = "\n".join(
    [
        " Abaqus/Standard 2025                  DATE 01-Jan-2026 TIME 09:00:00",
        " SUMMARY OF JOB INFORMATION:",
        " STEP  INC ATT SEVERE EQUIL TOTAL  TOTAL      STEP       INC OF",
        "               DISCON ITERS ITERS  TIME/    TIME/LPF    TIME/LPF",
        "               ITERS               FREQ",
        "   1     1   1     0     1     1  1.00       1.00       1.000     ",
        "   1     2   1     0     2     3  2.00       2.00       1.000     ",
        "                          ",
        " THE ANALYSIS HAS COMPLETED SUCCESSFULLY",
    ]
)

_MSG = "\n".join(
    [
        "1",
        "",
        "   Abaqus 2025                    Date 01-Jan-2026   Time 09:00:00",
        "   For use by An Invented Customer under license from Dassault Systemes",
        "",
        " STEP    1     INCREMENT     1     STEP TIME    0.00    ",
        "",
        "     ANALYSIS SUMMARY:",
        "     TOTAL OF          2  INCREMENTS",
        "                       0  CUTBACKS IN AUTOMATIC INCREMENTATION",
        "",
        "          THE ANALYSIS HAS BEEN COMPLETED",
    ]
)

_DAT_OK = "\n".join(
    [
        "     Abaqus 2025                                  Date 01-Jan-2026",
        "",
        "     JOB TIME SUMMARY",
        "       USER TIME (SEC)      =     0.20    ",
        "       TOTAL CPU TIME (SEC) =     0.20    ",
    ]
)

_DAT_FAIL = "\n".join(
    [
        "     Abaqus 2025                                  Date 01-Jan-2026",
        "",
        ' ***ERROR: Unknown keyword "elastik". It may be misspelled or obsolete,',
        " ***NOTE: DUE TO AN INPUT ERROR THE ANALYSIS PRE-PROCESSOR HAS BEEN UNABLE TO",
        " ***WARNING: something was odd about a surface definition",
        " ***ERROR: 640 elements are missing elastic property reference. The elements",
        "          are listed below. This message continues onto a third marker line.",
        # INVENTED: no observed Abaqus file repeats the marker on a continuation
        # line (fail_test/Bad.dat indents them unmarked). Kept only to prove the
        # stated count is preferred over marker counting.
        " ***ERROR: an invented third marker, to make the two counts differ",
        "          THE PROGRAM HAS DISCOVERED     2 FATAL ERRORS",
        "",
        "     JOB TIME SUMMARY",
    ]
)


def _read(sta=None, msg=None, dat=None):
    return read_abaqus_run_evidence(
        status_text=sta,
        messages_text=msg,
        printed_text=dat,
        source_units="t-mm-s",  # tonne-mm-s gives N and MPa
    )


def test_a_completed_job_is_read_from_its_three_files() -> None:
    evidence = _read(_STA, _MSG, _DAT_OK)
    assert evidence.identity is not None
    assert evidence.identity.name == "abaqus"
    assert evidence.identity.version == "2025"
    assert evidence.termination is not None
    (record,) = evidence.termination
    assert record.status == "normal"
    assert record.n_steps == 2  # two increments in the .sta table
    assert evidence.unparsable == frozenset()


def test_a_rejected_job_leaves_only_the_printed_output() -> None:
    """No `.sta` and no `.msg` at all -- the reader must not require them."""
    evidence = _read(None, None, _DAT_FAIL)
    assert evidence.termination is not None
    (record,) = evidence.termination
    assert record.status == "error"
    assert evidence.n_errors == 2  # the count the solver itself printed
    assert evidence.identity is not None and evidence.identity.version == "2025"


def test_the_printed_error_count_is_preferred_over_counting_lines() -> None:
    """Three `***ERROR:` markers, and the solver itself says two.

    Its own count is authoritative: one diagnostic wraps over several lines,
    each carrying the marker, so counting markers over-reports. Warnings have
    no printed count, so there the markers are all there is.
    """
    evidence = _read(None, None, _DAT_FAIL)
    assert evidence.n_errors == 2  # not 3
    assert evidence.n_warnings == 1


def test_nothing_of_the_licence_line_reaches_the_record() -> None:
    evidence = _read(_STA, _MSG, _DAT_OK)
    text = repr(evidence)
    for leak in ("Invented", "Dassault", "license", "licence"):
        assert leak not in text


def test_a_job_with_no_record_at_all_establishes_nothing() -> None:
    evidence = _read(None, None, None)
    assert evidence.identity is None
    assert evidence.termination is None
    assert evidence.n_errors is None and evidence.n_warnings is None


def test_an_unrecognised_termination_is_refused_rather_than_guessed() -> None:
    odd = _STA.replace(
        "THE ANALYSIS HAS COMPLETED SUCCESSFULLY", "THE ANALYSIS DID A THING"
    )
    evidence = _read(odd, None, _DAT_OK)
    assert evidence.termination is not None
    (record,) = evidence.termination
    assert record.status == "none"
    assert "termination_wording" in evidence.unparsable


def test_a_clean_run_reports_zero_rather_than_an_absence() -> None:
    """Abaqus prints no count line on a clean run, but the markers are countable.

    Zero markers IS zero diagnostics, which is a reading rather than a guess.
    Reporting an absence here would hand `solver_error_count` a
    contributor-owned reason for a run that was simply clean.
    """
    evidence = _read(_STA, _MSG, _DAT_OK)
    assert evidence.n_errors == 0
    assert evidence.n_warnings == 0


# --- the authoritative counts live in the .msg, not the .dat ------------------

_MSG_WITH_COUNTS = "\n".join(
    [
        "   Abaqus 2025                    Date 01-Jan-2026   Time 09:00:00",
        "     ANALYSIS SUMMARY:",
        "     TOTAL OF          2  INCREMENTS",
        "                       4  WARNING MESSAGES DURING USER INPUT PROCESSING",
        "                       8  WARNING MESSAGES DURING ANALYSIS",
        "                       3  ERROR MESSAGES",
        "          THE ANALYSIS HAS BEEN COMPLETED",
    ]
)


def test_the_message_files_stated_counts_are_authoritative() -> None:
    """The `.dat` carries no diagnostics summary at all; the `.msg` does.

    Reading only the `.dat` let a run with analysis errors report zero and
    PASS the ratified must-be-zero requirement -- an invented positive claim
    that exonerates an untrustworthy run.
    """
    evidence = _read(_STA, _MSG_WITH_COUNTS, _DAT_OK)
    assert evidence.n_errors == 3
    assert evidence.n_warnings == 12  # 4 during input processing + 8 during analysis


def test_a_rejected_job_still_reads_its_count_from_the_printed_output() -> None:
    """It has no `.msg` at all, so the `.dat` path must remain."""
    evidence = _read(None, None, _DAT_FAIL)
    assert evidence.n_errors == 2


def test_a_clean_message_file_reports_zero_from_its_own_summary() -> None:
    evidence = _read(_STA, _MSG, _DAT_OK)
    assert evidence.n_errors == 0 and evidence.n_warnings == 0


# --- Abaqus/Explicit (format of the 2026-09-24 conformance run; values invented)

_EXPLICIT_HEAD = (
    "Abaqus/Explicit 2025                             DATE 01-Jan-2026  TIME 00:00:00\n"
    " NUMERICAL PRECISION USED FOR THIS Abaqus/Explicit ANALYSIS\n"
    "Double precision package and explicit executables will be used in this analysis.\n"
    "***WARNING: There are 1 warning messages in the data (.dat) file.  Please\n"
    "            check the data file for possible errors in the input file.\n"
    "***WARNING: Each of the nodes listed below participates in a boundary condition\n"
    "  STEP  TOTAL      STEP      CPU       STABLE       CRITICAL    KINETIC    TOTAL\n"
    "INCREMENT     TIME      TIME      TIME     INCREMENT     ELEMENT     ENERGY     ENERGY\n"  # noqa: E501 - the real fixed-width format
    "        0  0.000E+00 0.000E+00  00:00:00 5.00000E-08          12  1.000E+04  1.000E+04\n"  # noqa: E501 - the real fixed-width format
    "       20  1.000E-06 1.000E-06  00:00:00 4.00000E-08          12  9.500E+03  9.990E+03\n"  # noqa: E501 - the real fixed-width format
)
_EXPLICIT_DONE = _EXPLICIT_HEAD + (
    "       40  2.000E-06 2.000E-06  00:00:01 3.00000E-08           7  9.000E+03  9.990E+03\n"  # noqa: E501 - the real fixed-width format
    "\n  THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n"
)
_EXPLICIT_ABORTED = _EXPLICIT_HEAD + (
    "       30  1.500E-06 1.500E-06  00:00:01 1.00000E-14           7  9.100E+03  9.990E+03\n"  # noqa: E501 - the real fixed-width format
    "***ERROR: Excessive distortion of element number 7\n"
    "\n  THE ANALYSIS HAS NOT BEEN COMPLETED\n"
)
_EXPLICIT_MSG = "\n STEP 1  ORIGIN 0.0000\n"
_EXPLICIT_DAT = (
    "   Abaqus 2025\n ***WARNING: THE PARAMETER HOURGLASS ON THE *SECTION CONTROLS\n"
)


def test_explicit_completed_run_reads_series_precision_and_counts() -> None:
    ev = _read(_EXPLICIT_DONE, _EXPLICIT_MSG, _EXPLICIT_DAT)
    assert ev.termination is not None and ev.identity is not None
    (record,) = ev.termination
    assert record.status == "normal"
    assert record.final_time == pytest.approx(2.0e-6)
    assert record.n_steps == 3
    assert ev.identity.precision == "double"
    assert ev.timestep is not None
    times, steps = ev.timestep
    assert times == pytest.approx((0.0, 1.0e-6, 2.0e-6))
    assert steps == pytest.approx((5.0e-8, 4.0e-8, 3.0e-8))
    # the .dat hourglass warning plus the .sta boundary/contact warning; the
    # .sta pointer to the .dat's own warning is not a second warning
    assert (ev.n_errors, ev.n_warnings) == (0, 2)


def test_explicit_analysis_failure_is_an_error_with_its_markers_counted() -> None:
    ev = _read(_EXPLICIT_ABORTED, _EXPLICIT_MSG, _EXPLICIT_DAT)
    assert ev.termination is not None
    (record,) = ev.termination
    assert record.status == "error"
    assert record.criterion == "analysis_not_completed"
    assert record.final_time == pytest.approx(1.5e-6)
    assert ev.n_errors == 1
    assert "termination_wording" not in ev.unparsable
