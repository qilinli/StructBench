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
#: The banner an Explicit job writes when its analysis phase stops early
#: (2026-09-24 conformance run: excessive distortion).
_NOT_COMPLETED = re.compile(r"THE ANALYSIS HAS NOT BEEN COMPLETED")
#: An Explicit increment row: increment, total time, step time, CPU hh:mm:ss,
#: stable increment, critical element, kinetic energy, total energy.
_EXPLICIT_ROW = re.compile(
    r"^\s+(\d+)\s+(\S+)\s+(\S+)\s+\d+:\d\d:\d\d\s+(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s*$",
    re.MULTILINE,
)
#: The precision line an Explicit status file prints.
_PRECISION = re.compile(r"(Double|Single) precision package and explicit", re.I)
#: The status file's pointer to the printed file's own warnings: not a new one.
_DAT_POINTER = re.compile(
    r"^\s*\*{3}WARNING: There (?:is|are) \d+ warning messages? in the data",
    re.MULTILINE,
)
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
    status_text: str | None,
    printed_text: str | None,
    tokens: set[str],
    time_factor: float = 1.0,
) -> TerminationRecord | None:
    """How the job ended, from whichever files it left behind."""
    if status_text is None and printed_text is None:
        return None
    n_steps = None
    final_time = None
    if status_text:
        explicit = _EXPLICIT_ROW.findall(status_text)
        if explicit:
            n_steps = len(explicit)
            final_time = float(explicit[-1][2]) * time_factor
        else:
            n_steps = len(_INCREMENT.findall(status_text)) or None
    if status_text and _COMPLETED.search(status_text):
        return TerminationRecord("normal", final_time, n_steps, "analysis_completed")
    if status_text and _NOT_COMPLETED.search(status_text):
        return TerminationRecord("error", final_time, n_steps, "analysis_not_completed")
    if printed_text and _FATAL.search(printed_text):
        return TerminationRecord("error", final_time, n_steps, "fatal_error")
    # A record exists and says neither. Refusing to classify is the point:
    # an unrecognised banner must not read as a clean run.
    tokens.add("termination_wording")
    return TerminationRecord("none", None, n_steps, None)


def _diagnostics(
    messages_text: str | None,
    printed_text: str | None,
    status_text: str | None = None,
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
    # Explicit writes its analysis-phase diagnostics to the status file, and its
    # message file states no totals (2026-09-24 conformance run).
    sta = status_text or ""
    sta_errors = len(_ERROR_MARK.findall(sta))
    sta_warnings = len(_WARNING_MARK.findall(sta)) - len(_DAT_POINTER.findall(sta))
    if printed_text is not None:
        if errors is None:
            fatal = _FATAL.search(printed_text)
            errors = (
                int(fatal.group(1))
                if fatal
                else len(_ERROR_MARK.findall(printed_text)) + sta_errors
            )
        if warnings is None:
            warnings = len(_WARNING_MARK.findall(printed_text)) + sta_warnings
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
    time_factor = unit_factors(source_units)["time"]
    tokens: set[str] = set()

    version = _version(status_text, messages_text, printed_text)
    identity = None
    if version is not None:
        # An Explicit status file prints its precision; a Standard job printed
        # none, so there it stays unset. No file prints the parallel layout.
        found = _PRECISION.search(status_text or "")
        precision = found.group(1).lower() if found else None
        identity = SolverIdentity("abaqus", version, precision=precision)  # type: ignore[arg-type]

    termination = _termination(status_text, printed_text, tokens, time_factor)
    n_errors, n_warnings = _diagnostics(messages_text, printed_text, status_text)
    rows = _EXPLICIT_ROW.findall(status_text or "")
    timestep = None
    if rows:  # E4: the Explicit increment table is a stable-increment series
        timestep = (
            tuple(float(r[2]) * time_factor for r in rows),
            tuple(float(r[3]) * time_factor for r in rows),
        )

    return RunEvidence(
        identity,
        (termination,) if termination is not None else None,
        n_errors,
        n_warnings,
        timestep,  # E4: Explicit only; a Standard table is read as a count
        None,  # E5: no energy ledger is written unless the job asks for one
        frozenset(tokens),
    )


# --- the solver input -------------------------------------------------------

#: Element-code prefixes this reader will classify. Abaqus has hundreds of
#: element types and their naming lives in the Keywords Reference, which has
#: not been sourced (ADR-0068 clause 8). Only codes a real deck was observed
#: to use are mapped; anything else is refused by name rather than guessed
#: from its letters.
_ELEMENT_KINDS: dict[str, str] = {"C3D": "solid", "CAX": "solid", "CPE": "solid"}
#: Codes whose reduced integration is established: one integration point per
#: element, non-zero ALLAE (2026-09-24 conformance run). Any other code reads
#: ``under_integrated = None`` rather than a guess from its trailing letter.
_UNDER_INTEGRATED = frozenset({"CAX4R"})
#: The part a deck without ``*Part`` blocks is written as. Abaqus names the
#: ``.odb`` part of a flat input ``PART-1``, so the stored response carries
#: the same name and ``mint_ids`` agrees on both sides.
_IMPLICIT_PART = "PART-1"
#: A keyword introducing inelasticity withdraws a ``linear_elastic`` claim, and
#: an ``elastic_plastic_isotropic`` one: the yield stress is then no longer a
#: function of plastic strain alone.
_INELASTIC = (
    "PLASTIC",
    "CREEP",
    "DAMAGE",
    "CONCRETE",
    "HYPERELASTIC",
    "VISCOELASTIC",
    "RATE DEPENDENT",
)
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
        # Model-level cards a flat deck writes straight after a material.
        "SECTION CONTROLS",
        "SURFACE INTERACTION",
        "RIGID BODY",
        "INITIAL CONDITIONS",
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


def _flags(line: str) -> set[str]:
    """``*Nset, nset=A, generate`` -> ``{"GENERATE"}``: the options with no value."""
    return {
        p.strip().upper() for p in line.split(",")[1:] if "=" not in p and p.strip()
    }


def _data_rows(lines: list[str], index: int) -> list[list[str]]:
    """The data lines after ``lines[index]`` up to the next keyword, as columns."""
    rows = []
    for line in lines[index + 1 :]:
        if line.startswith("*"):
            break
        if line.strip():
            rows.append([c.strip() for c in line.split(",")])
    return rows


def _node_labels(
    columns: list[str], node_sets: dict[str, frozenset[int] | None]
) -> frozenset[int] | None:
    """Node labels and set names -> labels; ``None`` if any entry is unresolved."""
    labels: set[int] = set()
    for entry in columns:
        if not entry:
            continue
        try:
            labels.add(int(entry))
        except ValueError:
            named = node_sets.get(entry.upper())
            if named is None:
                return None
            labels |= named
    return frozenset(labels)


def _plastic_table(
    rows: list[list[str]], stress: float
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """``*Plastic`` rows ``(stress, plastic strain)`` -> ``(strains, stresses)``.

    ``None`` for any other layout: a third column is a rate or a temperature,
    so the stress would no longer be a function of plastic strain alone.
    """
    knots = []
    for columns in rows:
        values = [c for c in columns if c]
        if len(values) != 2:
            return None
        try:
            knots.append((float(values[1]), float(values[0]) * stress))
        except ValueError:
            return None
    if not knots:
        return None
    if len(knots) == 1:
        # ADR-0070: the table is held at its last value, so one row is flat.
        return (0.0, 1.0), (knots[0][1], knots[0][1])
    strains, stresses = zip(*knots, strict=True)
    return tuple(strains), tuple(stresses)


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
    has_parts = "PART" in keywords
    element_codes: set[str] = set()
    part_codes: dict[str, set[str]] = {}
    # Upper-cased name -> labels; `None` marks a set this reader could not
    # resolve, so a card targeting it is refused rather than read as empty.
    node_sets: dict[str, frozenset[int] | None] = {}
    velocity: list[tuple[frozenset[int], int, float]] = []
    hardening: list[tuple[int, float]] = []
    velocity_read = hardening_read = True
    n_steps = 0
    periods: list[float] = []

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
        if (
            not has_parts
            and current_part is None
            and word in ("ELEMENT", "SOLID SECTION")
        ):
            current_part = _IMPLICIT_PART
            if current_part not in per_part:
                part_names.append(current_part)
                per_part[current_part] = {}

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
            element_codes.add(code)
            if current_part is not None:
                block = per_part.setdefault(current_part, {})
                block["kind"] = kind or "unknown"
                part_codes.setdefault(current_part, set()).add(code)
        elif word == "SOLID SECTION" and "MATERIAL" in options:
            if current_part is not None:
                per_part.setdefault(current_part, {})["material"] = options["MATERIAL"]
        elif word == "NODE":
            columns = [c for c in row.split(",") if c.strip()]
            if len(columns) >= 3:
                dimension = len(columns) - 1  # the first column is the label
        elif word == "NSET" and "NSET" in options:
            name = options["NSET"].upper()
            labels: frozenset[int] | None = None
            if "GENERATE" in _flags(line):
                tokens.add("unread_card:NSET_GENERATE")
            elif "ELSET" in options:
                tokens.add("unread_card:NSET_ELSET")
            else:
                columns = [c for r in _data_rows(lines, index) for c in r]
                labels = _node_labels(columns, node_sets)
            # A repeated name adds to the set; an unresolved part poisons it.
            if name in node_sets:
                earlier = node_sets[name]
                labels = None if earlier is None or labels is None else earlier | labels
            node_sets[name] = labels
        elif word == "INITIAL CONDITIONS":
            kind = options.get("TYPE", "").upper()
            rows = _data_rows(lines, index)
            if set(options) - {"TYPE"} or _flags(line):
                # REBAR, SECTION POINTS, USER...: the columns mean something else.
                tokens.add("unknown_card_layout:INITIAL_CONDITIONS")
                velocity_read = velocity_read and kind != "VELOCITY"
                hardening_read = hardening_read and kind != "HARDENING"
            elif kind == "VELOCITY":
                for columns in rows:
                    target = _node_labels(columns[:1], node_sets)
                    try:
                        dof, value = int(columns[1]), float(columns[2]) * f["velocity"]
                    except (IndexError, ValueError):
                        tokens.add("unknown_card_layout:INITIAL_CONDITIONS")
                        velocity_read = False
                        continue
                    if not target:  # unresolved, or an empty label
                        tokens.add("unresolved_initial_condition_target")
                        velocity_read = False
                        continue
                    velocity.append((target, dof, value))
            elif kind == "HARDENING":
                for columns in rows:
                    try:
                        element = int(columns[0])
                    except ValueError:
                        tokens.add("unread_card:HARDENING_ELSET")
                        hardening_read = False
                        continue
                    try:
                        hardening.append((element, float(columns[1])))
                    except (IndexError, ValueError):
                        tokens.add("unknown_card_layout:INITIAL_CONDITIONS")
                        hardening_read = False
            else:
                label = f"INITIAL_CONDITIONS_{kind.replace(' ', '_')}"
                tokens.add(
                    f"unread_card:{label}"
                    if _TOKENLIKE.fullmatch(label)
                    else "unread_card:INITIAL_CONDITIONS"
                )
        elif word == "STEP":
            n_steps += 1
        elif word == "STATIC":
            time_integration = "implicit"
        elif word == "DYNAMIC":
            time_integration = "explicit" if "EXPLICIT" in line.upper() else "implicit"
            if time_integration == "explicit":
                # `, <time period>`: the first column is a fixed increment, unused.
                columns = row.split(",")
                try:
                    periods.append(float(columns[1]) * f["time"])
                except (IndexError, ValueError):
                    tokens.add("unknown_card_layout:DYNAMIC")
        elif word == "DENSITY" and current_material is not None:
            rows = _data_rows(lines, index)
            if len(rows) == 1 and len([c for c in rows[0] if c]) == 1:
                try:
                    density = float(rows[0][0]) * f["density"]
                    per_material.setdefault(current_material, {})["density"] = density
                except ValueError:
                    tokens.add("unknown_card_layout:DENSITY")
            else:  # a temperature column or table: no single reference density
                tokens.add("unknown_card_layout:DENSITY")
        elif (
            word == "PLASTIC"
            and current_material is not None
            and options.get("HARDENING", "ISOTROPIC").upper() == "ISOTROPIC"
            and not set(options) - {"HARDENING"}
            and not _flags(line)
        ):
            block = per_material.setdefault(current_material, {})
            table = _plastic_table(_data_rows(lines, index), f["stress"])
            if table is None:
                tokens.add("unknown_card_layout:PLASTIC")
                block["inelastic"] = True
            else:
                block["yield_table"] = table
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

    def canonical(block: dict[str, object]) -> str | None:
        if block.get("youngs") is None or block.get("inelastic"):
            return None
        # ADR-0070: an isotropic `*Plastic` table and nothing else inelastic.
        return (
            "elastic_plastic_isotropic" if "yield_table" in block else "linear_elastic"
        )

    materials = tuple(
        MaterialInput(
            mid,
            canonical(per_material.get(name, {})),
            per_material.get(name, {}).get("density"),  # type: ignore[arg-type]
            None,
            per_material.get(name, {}).get("youngs"),  # type: ignore[arg-type]
            per_material.get(name, {}).get("poisson"),  # type: ignore[arg-type]
            per_material.get(name, {}).get("yield_table"),  # type: ignore[arg-type]
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
        codes = part_codes.get(name, set())
        under = True if codes and codes <= _UNDER_INTEGRATED else None
        built.append(
            PartTraits(pid, material_id, block.get("kind", "unknown"), under)  # type: ignore[arg-type]
        )
    parts = tuple(built)
    plane_strain = None
    if element_codes and all(c.startswith("CPE") for c in element_codes):
        plane_strain = True
    elif element_codes and all(c.startswith("CAX") for c in element_codes):
        plane_strain = False

    return InputFacts(
        parts=parts,
        materials=materials,
        time_integration=time_integration,  # type: ignore[arg-type]
        dimension=dimension,  # type: ignore[arg-type]
        plane_strain=plane_strain,
        # Only an explicit step's period is a duration: a `*Static` step's is a
        # dimensionless load parameter. Several steps have no single period.
        end_time=periods[0] if n_steps == 1 and len(periods) == 1 else None,
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
        initial_velocity=None if hidden or not velocity_read else tuple(velocity),
        initial_hardening=None if hidden or not hardening_read else tuple(hardening),
    )
