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
#: Diagnostic markers in the printed output. The solver's own stated counts are
#: preferred wherever it gives them, because a diagnostic can span lines and
#: because a marker count cannot separate the severities the solver does.
_ERROR_MARK = re.compile(r"^\s*\*{3}ERROR\b", re.MULTILINE)
_WARNING_MARK = re.compile(r"^\s*\*{3}WARNING\b", re.MULTILINE)
#: The message file's ANALYSIS SUMMARY states the run's own totals, and it is
#: the authoritative source: the printed output carries no such summary at all
#: (``grep MESSAGES Cantilever.dat`` finds nothing), so reading only the
#: printed file let a run with analysis errors report zero and PASS a ratified
#: must-be-zero requirement.
_MSG_ERRORS = re.compile(r"^\s*(\d+)\s+ERROR MESSAGES\b", re.MULTILINE)
_MSG_WARNINGS = re.compile(r"^\s*(\d+)\s+WARNING MESSAGES\b", re.MULTILINE)


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


def _diagnostics(
    messages_text: str | None, printed_text: str | None
) -> tuple[int | None, int | None]:
    """``(errors, warnings)`` from the run's own record, or ``(None, None)``.

    Preference order, and the reason for each step:

    1. The message file's ANALYSIS SUMMARY, which states the run's own totals.
       It separates warnings raised during input processing from those raised
       during the analysis, so both are summed.
    2. Otherwise the printed output's fatal-error count, which a rejected job
       states and which has no message file to be read from at all.
    3. Otherwise the printed markers. Zero markers is zero diagnostics -- a
       reading, not an assumption -- which keeps a clean run off a
       contributor-owned absence.
    """
    if messages_text is None and printed_text is None:
        return None, None
    errors: int | None = None
    warnings: int | None = None
    if messages_text is not None:
        stated_errors = _MSG_ERRORS.findall(messages_text)
        stated_warnings = _MSG_WARNINGS.findall(messages_text)
        if stated_errors:
            errors = sum(int(n) for n in stated_errors)
        if stated_warnings:
            warnings = sum(int(n) for n in stated_warnings)
    if printed_text is not None:
        if errors is None:
            fatal = _FATAL.search(printed_text)
            errors = (
                int(fatal.group(1)) if fatal else len(_ERROR_MARK.findall(printed_text))
            )
        if warnings is None:
            warnings = len(_WARNING_MARK.findall(printed_text))
    return errors, warnings


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
    n_errors, n_warnings = _diagnostics(messages_text, printed_text)

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
#: Keywords that close a material definition, so a card after one of these is
#: no longer attributed to the material above it.
_ENDS_MATERIAL = frozenset(
    {
        "MATERIAL",
        "PART",
        "END PART",
        "ASSEMBLY",
        "END ASSEMBLY",
        "INSTANCE",
        "END INSTANCE",
        "STEP",
        "END STEP",
        "NODE",
        "ELEMENT",
        "NSET",
        "ELSET",
        "SURFACE",
        "SOLID SECTION",
        "SHELL SECTION",
        "HEADING",
        "INCLUDE",
        "RESTART",
        "PREPRINT",
        "OUTPUT",
        "BOUNDARY",
        "DSLOAD",
        "CLOAD",
        "STATIC",
        "DYNAMIC",
        "AMPLITUDE",
        "SYSTEM",
    }
)
#: Keywords that close a part definition.
_ENDS_PART = frozenset({"END PART", "ASSEMBLY", "MATERIAL", "STEP"})
#: Cards whose payload is a data block, so an ``input=`` option on one of them
#: moves that payload into another file and hides it from this reader.
_DATA_BEARING = frozenset({"NODE", "ELEMENT", "NSET", "ELSET", "SURFACE"})


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


#: A token must match ``[A-Za-z0-9_.+-]+``; an element code is a raw option
#: value, so ``type=`` empty or ``type="S4R"`` would otherwise raise out of a
#: reader documented never to raise -- and the CLI turns that into every row
#: reading ``source_unreadable``.
_TOKENLIKE = re.compile(r"^[A-Za-z0-9_.+-]+$")


def _element_token(code: str) -> str:
    """``unknown_element_type:<code>``, or the bare token if the code is odd."""
    return (
        f"unknown_element_type:{code}"
        if _TOKENLIKE.fullmatch(code)
        else "unknown_element_type"
    )


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

    # Per-block, never deck-wide. A single set of locals accumulated over the
    # whole file and replicated into every record reported one material's
    # stiffness under another's name, with `unparsable` empty -- a fabricated
    # positive claim, which is the failure mode this reader exists to avoid.
    part_names: list[str] = []
    material_names: list[str] = []
    per_material: dict[str, dict[str, object]] = {}
    per_part: dict[str, dict[str, object]] = {}
    current_material: str | None = None
    current_part: str | None = None
    dimension: int | None = None
    time_integration: str | None = None
    motion = False

    for index, line in enumerate(lines):
        if not line.startswith("*"):
            continue
        word, options = _keyword(line), _options(line)
        nxt = lines[index + 1] if index + 1 < len(lines) else ""
        # A data line is what the lookahead may read. Landing on a keyword and
        # parsing it as data is how `*Node, input=…` yielded a dimension.
        row = "" if nxt.startswith("*") else nxt
        # `input=` on a data-bearing card hides content exactly as `*Include`
        # does, so nothing may be asserted absent afterwards.
        if "INPUT" in options and word in _DATA_BEARING:
            tokens.add("include")

        if word in _ENDS_MATERIAL:
            current_material = None
        if word in _ENDS_PART:
            current_part = None

        if word == "PART" and "NAME" in options:
            current_part = options["NAME"]
            part_names.append(current_part)
            per_part.setdefault(current_part, {})
        elif word == "MATERIAL" and "NAME" in options:
            current_material = options["NAME"]
            material_names.append(current_material)
            per_material.setdefault(current_material, {})
        elif word == "ELEMENT" and "TYPE" in options:
            code = options["TYPE"].upper()
            kind = next(
                (v for k, v in _ELEMENT_KINDS.items() if code.startswith(k)), None
            )
            if kind is None:
                tokens.add(_element_token(code))
            if current_part is not None:
                per_part.setdefault(current_part, {})["kind"] = kind or "unknown"
        elif word == "SOLID SECTION" and "MATERIAL" in options:
            if current_part is not None:
                per_part.setdefault(current_part, {})["material"] = options["MATERIAL"]
        elif word == "NODE":
            columns = [c for c in row.split(",") if c.strip()]
            if len(columns) >= 3:
                dimension = len(columns) - 1  # the first column is the label
        elif word == "STATIC":
            time_integration = "implicit"
        elif word == "DYNAMIC":
            time_integration = "explicit" if "EXPLICIT" in line.upper() else "implicit"
        elif word == "ELASTIC":
            kind = options.get("TYPE", "ISOTROPIC").upper()
            values = [c.strip() for c in row.split(",") if c.strip()]
            if kind != "ISOTROPIC":
                # A typed card's columns are not (E, nu): reading them
                # positionally stored a stiffness in `poisson_ratio`.
                tokens.add("unknown_card_layout:ELASTIC")
            elif len(values) >= 2 and current_material is not None:
                try:
                    block = per_material.setdefault(current_material, {})
                    block["youngs"] = float(values[0]) * f["stress"]
                    block["poisson"] = float(values[1])
                except ValueError:
                    tokens.add("unknown_card_layout:ELASTIC")
        elif word == "BOUNDARY":
            motion = motion or options.get("TYPE", "").upper() in _MOTION_TYPES
        elif current_material is not None:
            # A material card this reader does not implement. Recording it is
            # what lets `unstated_or_unread` and `input_gap` report a typed
            # absence instead of asserting the input states nothing.
            tokens.add(f"unread_card:{word.replace(' ', '_')}")
            if any(bad in word for bad in _INELASTIC):
                per_material.setdefault(current_material, {})["inelastic"] = True

    hidden = "include" in tokens
    materials_by_name = mint_ids(material_names)
    parts_by_name = mint_ids(part_names)
    materials = tuple(
        MaterialInput(
            mid,
            "linear_elastic"
            if per_material.get(name, {}).get("youngs") is not None
            and not per_material.get(name, {}).get("inelastic")
            else None,
            None,
            None,
            per_material.get(name, {}).get("youngs"),  # type: ignore[arg-type]
            per_material.get(name, {}).get("poisson"),  # type: ignore[arg-type]
            None,
        )
        for name, mid in sorted(materials_by_name.items(), key=lambda kv: kv[1])
    )
    built: list[PartTraits] = []
    for name, pid in sorted(parts_by_name.items(), key=lambda kv: kv[1]):
        block = per_part.get(name, {})
        material_name = block.get("material")
        material_id = materials_by_name.get(str(material_name), 0)
        if material_id == 0:
            # No material owns this part's section. Silently minting 0 -- an id
            # no material has -- would drop the part from every class-masked
            # check without saying so.
            tokens.add("unresolved_section_material")
        built.append(
            PartTraits(pid, material_id, block.get("kind", "unknown"), None)  # type: ignore[arg-type]
        )
    parts = tuple(built)

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
