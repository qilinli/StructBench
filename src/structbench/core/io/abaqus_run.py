"""Fail-closed readers for an Abaqus run's text record (ADR-0068).

Abaqus spreads what LS-DYNA puts in two files across three, and **which files
exist is itself evidence**:

- ``.sta`` — the increment table and the termination banner. A job the
  pre-processor rejects never writes one.
- ``.msg`` — the per-increment record and the analysis summary. Likewise
  absent from a rejected job. Its header carries a licence line naming the
  seat; nothing of it is kept.
- ``.dat`` — the printed input echo, the diagnostics, and on failure the
  solver's own fatal-error count. Written whatever happens.

So the reader takes three optional texts. It reads only whitelisted patterns
and returns numbers, enum values and version tokens — never text from the
files (ADR-0066 clause 3).

Everything here was derived from observed output, not from the Keywords
Reference. Where a pattern is not recognised the reader records a token in
``unparsable`` and leaves the field ``None``; it never guesses, because a
guess about a run is what ADR-0068 exists to prevent.
"""

from __future__ import annotations

import re

from ..evidence import RunEvidence, SolverIdentity, TerminationRecord
from .lsdyna import unit_factors

__all__ = ["read_abaqus_run_evidence"]

#: ``Abaqus/Standard 2025 ...`` in the status file, ``Abaqus 2025 ...`` in the
#: message and printed files. The release year is the version token.
_VERSION = re.compile(r"^\s*Abaqus(?:/\w+)?\s+(\d{4})\b", re.MULTILINE)
#: The banner a completed job writes to its status file.
_COMPLETED = re.compile(r"THE ANALYSIS HAS COMPLETED SUCCESSFULLY")
#: What the pre-processor writes when it refuses the input.
_FATAL = re.compile(r"THE PROGRAM HAS DISCOVERED\s+(\d+)\s+FATAL ERRORS")
#: A status-file increment row: step, increment, then further integer columns.
_INCREMENT = re.compile(r"^\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s", re.MULTILINE)
#: Diagnostic markers. A single diagnostic wraps over several lines, each
#: carrying the marker, so these OVER-count; the solver's own printed count
#: wins wherever it exists.
_ERROR_MARK = re.compile(r"^\s*\*{3}ERROR\b", re.MULTILINE)
_WARNING_MARK = re.compile(r"^\s*\*{3}WARNING\b", re.MULTILINE)


def _version(*texts: str | None) -> str | None:
    for text in texts:
        if text:
            found = _VERSION.search(text)
            if found:
                return found.group(1)
    return None


def _termination(
    status_text: str | None, printed_text: str | None, tokens: set[str]
) -> TerminationRecord | None:
    """How the job ended, from whichever files it left behind."""
    if status_text is None and printed_text is None:
        return None
    n_steps = None
    if status_text:
        rows = _INCREMENT.findall(status_text)
        n_steps = len(rows) or None
    if status_text and _COMPLETED.search(status_text):
        return TerminationRecord("normal", None, n_steps, "analysis_completed")
    if printed_text and _FATAL.search(printed_text):
        return TerminationRecord("error", None, n_steps, "fatal_error")
    # A record exists and says neither. Refusing to classify is the point:
    # an unrecognised banner must not read as a clean run.
    tokens.add("termination_wording")
    return TerminationRecord("none", None, n_steps, None)


def _diagnostics(printed_text: str | None) -> tuple[int | None, int | None]:
    """``(errors, warnings)`` from the printed output, or ``(None, None)``.

    The solver prints a fatal-error count only when it has some, so a clean
    run is read from the markers: zero markers is zero diagnostics, which is
    a reading and not an assumption. Where the solver states a count it wins,
    because a wrapped diagnostic repeats its marker on every line.
    """
    if printed_text is None:
        return None, None
    stated = _FATAL.search(printed_text)
    errors = int(stated.group(1)) if stated else len(_ERROR_MARK.findall(printed_text))
    return errors, len(_WARNING_MARK.findall(printed_text))


def read_abaqus_run_evidence(
    *,
    status_text: str | None = None,
    messages_text: str | None = None,
    printed_text: str | None = None,
    source_units: str,
) -> RunEvidence:
    """Read what an Abaqus run's text record establishes (E2–E5).

    Parameters
    ----------
    status_text : str or None
        The ``.sta`` file. Absent from a job the pre-processor rejected.
    messages_text : str or None
        The ``.msg`` file. Likewise absent from a rejected job.
    printed_text : str or None
        The ``.dat`` file, written whatever happens.
    source_units : str
        The input's ``"<mass>-<length>-<time>"`` convention; validated here
        so a record is never built against an unknown unit system, even
        though no field of this record is yet dimensional.

    Returns
    -------
    RunEvidence
        Never raises on content it cannot resolve: it records a token in
        ``unparsable`` and leaves the affected field ``None``. No energy
        ledger is read — an Abaqus job writes none unless asked, and no
        observed run has asked (ADR-0068 clause 8).

    Raises
    ------
    ValueError
        If ``source_units`` is not a known unit system.
    """
    unit_factors(source_units)  # validate; no field of this record is scaled yet
    tokens: set[str] = set()

    version = _version(status_text, messages_text, printed_text)
    identity = None
    if version is not None:
        # Precision and the parallel layout are not printed in any observed
        # file, so they stay unset rather than assumed.
        identity = SolverIdentity("abaqus", version)

    termination = _termination(status_text, printed_text, tokens)
    n_errors, n_warnings = _diagnostics(printed_text)

    return RunEvidence(
        identity,
        (termination,) if termination is not None else None,
        n_errors,
        n_warnings,
        None,  # E4: the increment table is not yet read into a time-step series
        None,  # E5: no energy ledger is written unless the job asks for one
        frozenset(tokens),
    )
