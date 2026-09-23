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

from ..evidence import (
    InputFacts,
    MaterialInput,
    PartTraits,
    RunEvidence,
    SolverIdentity,
    TerminationRecord,
)
from .lsdyna import unit_factors

__all__ = ["mint_ids", "read_abaqus_input_facts", "read_abaqus_run_evidence"]

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


# --- the solver input -------------------------------------------------------

#: Element-code prefixes this reader will classify. Abaqus has hundreds of
#: element types and their naming lives in the Keywords Reference, which has
#: not been sourced (ADR-0068 clause 8). Only codes a real deck was observed
#: to use are mapped; anything else is refused by name rather than guessed
#: from its letters.
_ELEMENT_KINDS: dict[str, str] = {"C3D": "solid"}
#: A keyword introducing inelasticity withdraws a ``linear_elastic`` claim.
_INELASTIC = ("PLASTIC", "CREEP", "DAMAGE", "CONCRETE", "HYPERELASTIC", "VISCOELASTIC")
#: ``*Boundary, type=`` values that prescribe motion rather than fix a degree
#: of freedom. A plain ``*Boundary`` is a fixity.
_MOTION_TYPES = ("VELOCITY", "ACCELERATION", "DISPLACEMENT")


def mint_ids(names: list[str]) -> dict[str, int]:
    """Deterministic integer ids for named entities, by first appearance.

    Abaqus names its parts and materials; ``PartTraits.part_id`` and
    ``MaterialInput.material_id`` are integers. Nothing in the solver's own
    output supplies one -- unlike an LS-DYNA ``*PART``, whose explicit id the
    d3plot repeats -- so the integers are minted, and this function is the
    single authority for that minting (ADR-0068).

    The response side must not mint independently. The conversion glue maps
    stored entities onto these ids **by name** using this function, and the
    ``.odb`` extractor emits names rather than ids. Two processes agreeing on
    an ordering by luck is exactly how ``elements_without_input_part`` would
    come to blame a dataset for a bookkeeping mismatch.
    """
    minted: dict[str, int] = {}
    for name in names:
        if name not in minted:
            minted[name] = len(minted) + 1
    return minted


def _keyword(line: str) -> str:
    """``*Solid Section, elset=..`` -> ``SOLID SECTION``."""
    return line[1:].split(",")[0].strip().upper()


def _options(line: str) -> dict[str, str]:
    """``, name=Steel, type=C3D8I`` -> ``{"NAME": "Steel", "TYPE": "C3D8I"}``."""
    out = {}
    for part in line.split(",")[1:]:
        if "=" in part:
            key, _, value = part.partition("=")
            out[key.strip().upper()] = value.strip()
    return out


def read_abaqus_input_facts(deck_text: str, *, source_units: str) -> InputFacts:
    """Read what an Abaqus input establishes about a run (evidence item E1).

    Implements only what a real deck was observed to contain and fails closed
    on the rest: an unrecognised element code is recorded by name and its part
    reads ``"unknown"``, and an ``*Include`` marks the deck as hiding content
    so nothing is asserted absent.

    Parameters
    ----------
    deck_text : str
        The complete ``.inp``, e.g. ``Metadata.source_deck``.
    source_units : str
        The deck's ``"<mass>-<length>-<time>"`` convention. An Abaqus input
        carries no units, so it is supplied per dataset; ``t-mm-s`` is the
        consistent set giving N and MPa.

    Returns
    -------
    InputFacts
        Never raises on content it cannot resolve.

    Raises
    ------
    ValueError
        If ``source_units`` is not a known unit system.
    """
    f = unit_factors(source_units)
    tokens: set[str] = set()
    lines = [ln.rstrip() for ln in deck_text.splitlines() if not ln.startswith("**")]
    keywords = {_keyword(ln) for ln in lines if ln.startswith("*")}
    if "INCLUDE" in keywords:
        tokens.add("include")
    hidden = "include" in tokens

    def requested(found: bool) -> bool | None:
        return True if found else (None if hidden else False)

    part_names: list[str] = []
    material_names: list[str] = []
    element_kind: str | None = None
    section_material: str | None = None
    dimension: int | None = None
    time_integration: str | None = None
    youngs: float | None = None
    poisson: float | None = None
    inelastic = False
    motion = False

    for index, line in enumerate(lines):
        if not line.startswith("*"):
            continue
        word, options = _keyword(line), _options(line)
        row = lines[index + 1] if index + 1 < len(lines) else ""
        if word == "PART" and "NAME" in options:
            part_names.append(options["NAME"])
        elif word == "MATERIAL" and "NAME" in options:
            material_names.append(options["NAME"])
        elif word == "ELEMENT" and "TYPE" in options:
            code = options["TYPE"].upper()
            kind = next(
                (v for k, v in _ELEMENT_KINDS.items() if code.startswith(k)), None
            )
            if kind is None:
                tokens.add(f"unknown_element_type:{code}")
            element_kind = kind or "unknown"
        elif word == "SOLID SECTION" and "MATERIAL" in options:
            section_material = options["MATERIAL"]
        elif word == "NODE":
            columns = [c for c in row.split(",") if c.strip()]
            if len(columns) >= 3:
                dimension = len(columns) - 1  # the first column is the label
        elif word == "STATIC":
            time_integration = "implicit"
        elif word == "DYNAMIC":
            explicit = "EXPLICIT" in line.upper()
            time_integration = "explicit" if explicit else "implicit"
        elif word == "ELASTIC":
            values = [c.strip() for c in row.split(",") if c.strip()]
            if len(values) >= 2:
                try:
                    youngs = float(values[0]) * f["stress"]
                    poisson = float(values[1])
                except ValueError:
                    tokens.add("unknown_card_layout:ELASTIC")
        elif word == "BOUNDARY":
            motion = motion or options.get("TYPE", "").upper() in _MOTION_TYPES
        if any(bad in word for bad in _INELASTIC):
            inelastic = True

    materials_by_name = mint_ids(material_names)
    parts_by_name = mint_ids(part_names)
    canonical = "linear_elastic" if youngs is not None and not inelastic else None
    materials = tuple(
        MaterialInput(mid, canonical, None, None, youngs, poisson, None)
        for _, mid in sorted(materials_by_name.items(), key=lambda kv: kv[1])
    )
    section_id = materials_by_name.get(section_material or "", 0)
    parts = tuple(
        PartTraits(pid, section_id, element_kind or "unknown", None)  # type: ignore[arg-type]
        for _, pid in sorted(parts_by_name.items(), key=lambda kv: kv[1])
    )

    return InputFacts(
        parts=parts,
        materials=materials,
        time_integration=time_integration,  # type: ignore[arg-type]
        dimension=dimension,  # type: ignore[arg-type]
        plane_strain=None,
        # A `*Static` step's time period is a dimensionless load parameter,
        # not a duration; converting it to seconds would be a category error.
        end_time=None,
        other_termination_criteria=frozenset(),
        mass_scaling_enabled=requested("MASS SCALING" in keywords),
        erosion_enabled=None,  # element deletion is a `*Section Controls` option
        contact_defined=requested(
            any(k.startswith("CONTACT") or k == "SURFACE INTERACTION" for k in keywords)
        ),
        prescribed_motion_defined=requested(motion),
        damping_defined=requested("DAMPING" in keywords),
        rigid_planes=(),
        particle_pairwise_conservative=None,
        smoothing_length_scale_bounds=None,
        # ADR-0068 clause 8: the Abaqus output-request vocabulary is deferred
        # until the claim dossier exists, so neither of these is established.
        energy_terms_computed=None,
        databases_requested=None,
        unparsable=frozenset(tokens),
        solver="abaqus",
    )
