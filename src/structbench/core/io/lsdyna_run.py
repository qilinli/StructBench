"""Fail-closed readers of an LS-DYNA run: its input and its run record (ADR-0066).

``read_input_facts`` reads the keyword input into :class:`InputFacts`;
``read_run_evidence`` reads the message and global-statistics text into
:class:`RunEvidence`.

The canonical adapter's ``_card_blocks`` drops blank and non-numeric rows, so
positional extraction from it can return a plausible but wrong table. This
reader keeps blank data lines and fixed field positions, and *refuses* what it
cannot resolve: an ``*INCLUDE``, a ``&parameter`` reference, a free-format row
or an unknown card layout becomes a token in ``InputFacts.unparsable`` and the
affected field stays ``None``.

Absence is read two ways. A feature that has to be *requested* — mass scaling,
erosion, contact, a prescribed motion, an implicit solve — is off when no card
requests it and nothing could be hiding one. A *setting with a default* — the
particle-control flags — stays ``None`` when its card is absent: a solver
default is never assumed (ADR-0066 clause 3).

All LS-DYNA vocabulary of the verification subsystem lives in this file
(ADR-0004).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

from ..evidence import (
    EnergyLedger,
    InputFacts,
    MaterialInput,
    PartTraits,
    RigidPlane,
    RunEvidence,
    SolverIdentity,
    TerminationRecord,
)
from .lsdyna import canonical_model_for, unit_factors

__all__ = ["read_input_facts", "read_run_evidence"]

_WIDTH = 10
_FIELDS = 8
#: Tokens meaning "a card may exist that this reader did not see".
_HIDING = frozenset({"include", "parameter"})
_SECTION_KIND = {
    "SECTION_SPH": "particle",
    "SECTION_SOLID": "solid",
    "SECTION_SHELL": "shell",
    "SECTION_BEAM": "beam",
}


@dataclass
class _Card:
    keyword: str
    lines: list[str] = field(default_factory=list)


def _cards(text: str, tokens: set[str]) -> list[_Card]:
    """Split the input into cards, keeping blank data lines."""
    cards: list[_Card] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("$"):
            continue
        if line.startswith("*"):
            keyword = line[1:].strip().upper()
            if " " in keyword or keyword.endswith(("+", "-")):
                tokens.add("long_format")
                keyword = keyword.split()[0].rstrip("+-")
            if keyword == "END":
                break
            cards.append(_Card(keyword))
        elif cards:
            cards[-1].lines.append(line)
    return cards


def _numbers(line: str, keyword: str, tokens: set[str]) -> list[float | None] | None:
    """Eight fixed-width fields as numbers; ``None`` for a blank field.

    Returns ``None`` — and records why — when the row is not plain fixed-width
    numeric data.
    """
    if "," in line:
        tokens.add("free_format")
        return None
    if "&" in line:
        tokens.add("parameter")
        return None
    out: list[float | None] = []
    for i in range(_FIELDS):
        text = line[i * _WIDTH : (i + 1) * _WIDTH].strip()
        if not text:
            out.append(None)
            continue
        try:
            out.append(float(text))
        except ValueError:
            tokens.add(f"unreadable_number:{keyword}")
            return None
    return out


def _find(cards: list[_Card], prefix: str) -> list[_Card]:
    return [c for c in cards if c.keyword == prefix or c.keyword.startswith(prefix)]


def _first_row(
    cards: list[_Card], keyword: str, tokens: set[str]
) -> list[float | None] | None:
    for card in cards:
        if card.keyword == keyword and card.lines:
            return _numbers(card.lines[0], keyword, tokens)
    return None


def _body(card: _Card, *, titled: bool) -> list[str]:
    """Data lines, without the title line a ``_TITLE`` / ``_ID`` option adds."""
    return card.lines[1:] if titled else card.lines


def _yield_table(
    rows: list[list[float | None] | None], stress_factor: float
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Sixteen (strain, stress) knots from four rows; trailing empty pairs dropped."""
    if len(rows) != 4 or any(r is None for r in rows):
        return None
    flat = [[0.0 if v is None else v for v in r] for r in rows if r is not None]
    strain, stress = flat[0] + flat[1], flat[2] + flat[3]
    pairs = list(zip(strain, stress, strict=True))
    while len(pairs) > 1 and pairs[-1] == (0.0, 0.0):
        pairs.pop()
    if len(pairs) < 2 or not any(s for _, s in pairs):
        return None  # no table: the scalar yield-stress branch is in use
    if any(b[0] <= a[0] for a, b in zip(pairs, pairs[1:], strict=False)):
        return None  # strain knots must increase; stress knots are kept verbatim
    return (
        tuple(e for e, _ in pairs),
        tuple(s * stress_factor for _, s in pairs),
    )


def _material(
    card: _Card, f: dict[str, float], tokens: set[str]
) -> MaterialInput | None:
    titled = card.keyword.endswith("_TITLE")
    base = card.keyword.removesuffix("_TITLE")
    spall = base.endswith("_SPALL")
    model = base.removesuffix("_SPALL")
    lines = _body(card, titled=titled)
    head = _numbers(lines[0], base, tokens) if lines else None
    if head is None or head[0] is None:
        return None
    mid = int(head[0])
    canonical = canonical_model_for(model)

    def scaled(value: float | None, key: str) -> float | None:
        return None if value is None else value * f[key]

    if model == "MAT_ELASTIC_PLASTIC_HYDRO":
        first = 2 if spall else 1  # the spall option inserts one card
        rows = [_numbers(line, base, tokens) for line in lines[first : first + 4]]
        return MaterialInput(
            mid,
            canonical,
            scaled(head[1], "density"),
            scaled(head[2], "stress"),
            None,
            None,
            _yield_table(rows, f["stress"]),
        )
    if model in ("MAT_ELASTIC", "MAT_RIGID"):
        return MaterialInput(
            mid,
            canonical,
            scaled(head[1], "density"),
            None,
            scaled(head[2], "stress"),
            head[3],
            None,
        )
    if model == "MAT_NULL":
        return MaterialInput(
            mid, canonical, scaled(head[1], "density"), None, None, None, None
        )
    if model == "MAT_CONCRETE_DAMAGE_REL3":
        # mid, ro, pr. Its yield surface is pressure-dependent and generated
        # from the unconfined compressive strength, not tabulated, so no
        # yield table is claimed for it.
        return MaterialInput(
            mid, canonical, scaled(head[1], "density"), None, None, head[2], None
        )
    if model == "MAT_PLASTIC_KINEMATIC":
        # mid, ro, e, pr, sigy, etan, beta. The yield law is bilinear, stated
        # by sigy and etan rather than by knots, so it is not a yield table:
        # `yield_table` holds knots a deck states verbatim, and two would have
        # to be invented here.
        return MaterialInput(
            mid,
            canonical,
            scaled(head[1], "density"),
            None,
            scaled(head[2], "stress"),
            head[3],
            None,
        )
    tokens.add(f"unknown_card_layout:{model}")
    return MaterialInput(mid, None, None, None, None, None, None)


def _rigid_planes(
    cards: list[_Card], length: float, tokens: set[str]
) -> tuple[RigidPlane, ...]:
    planes: list[RigidPlane] = []
    for card in _find(cards, "RIGIDWALL"):
        if card.keyword not in ("RIGIDWALL_PLANAR", "RIGIDWALL_PLANAR_ID"):
            tokens.add(f"unknown_card_layout:{card.keyword}")
            continue
        lines = _body(card, titled=card.keyword.endswith("_ID"))
        row = _numbers(lines[1], card.keyword, tokens) if len(lines) > 1 else None
        if row is None or any(v is None for v in row[:6]):
            tokens.add(f"unknown_card_layout:{card.keyword}")
            continue
        tail = [v * length for v in row[:3] if v is not None]
        head = [v * length for v in row[3:6] if v is not None]
        span = [h - t for h, t in zip(head, tail, strict=True)]
        norm = math.hypot(*span)
        if norm == 0.0:
            tokens.add(f"unknown_card_layout:{card.keyword}")
            continue
        # The wall normal runs from its tail to its head. That the head side is
        # the admissible one is confirmed on real data, where the body must sit
        # at positive distance (stage-1 hand-run acceptance).
        planes.append(
            RigidPlane(
                (tail[0], tail[1], tail[2]),
                (span[0] / norm, span[1] / norm, span[2] / norm),
            )
        )
    return tuple(planes)


def _parts(cards: list[_Card], tokens: set[str]) -> tuple[PartTraits, ...]:
    kinds: dict[int, str] = {}
    for card in _find(cards, "SECTION_"):
        base = card.keyword.removesuffix("_TITLE")
        lines = _body(card, titled=card.keyword.endswith("_TITLE"))
        row = _numbers(lines[0], base, tokens) if lines else None
        if row is not None and row[0] is not None:
            kinds[int(row[0])] = _SECTION_KIND.get(base, "unknown")
    parts: list[PartTraits] = []
    for card in cards:
        if card.keyword != "PART":
            if card.keyword.startswith("PART_"):
                tokens.add(f"unknown_card_layout:{card.keyword}")
            continue
        for data in card.lines[1::2]:  # (title, data) pairs
            row = _numbers(data, "PART", tokens)
            if row is None or row[0] is None or row[2] is None:
                continue
            section = None if row[1] is None else int(row[1])
            kind = kinds.get(section, "unknown") if section is not None else "unknown"
            # Whether a meshed element formulation is under-integrated needs a
            # table this reader does not have yet; it arrives, from the manual,
            # with the first meshed dataset.
            parts.append(PartTraits(int(row[0]), int(row[2]), kind, None))  # type: ignore[arg-type]
    return tuple(parts)


def _smoothing_bounds(
    cards: list[_Card], tokens: set[str]
) -> tuple[float, float] | None:
    bounds: list[tuple[float, float]] = []
    for card in _find(cards, "SECTION_SPH"):
        lines = _body(card, titled=card.keyword.endswith("_TITLE"))
        row = _numbers(lines[0], "SECTION_SPH", tokens) if lines else None
        if row is None or row[2] is None or row[3] is None:
            return None
        bounds.append((row[2], row[3]))
    if not bounds:
        return None
    return min(b[0] for b in bounds), max(b[1] for b in bounds)


#: ``*CONTROL_ENERGY`` field order -> the ledger term each one governs. A
#: field set to 2 computes the term; 1 leaves it out of the balance entirely,
#: so it is absent from the solver's own total and the ledger cannot be told
#: apart from one whose term is genuinely zero.
_ENERGY_FIELDS = ("zero_energy_mode", "rigid_surface", "contact", "damping")
#: ``*DATABASE_`` cards whose first field is not an output interval: they
#: configure what output contains, and request none of it.
_DATABASE_SETTINGS = ("DATABASE_EXTENT", "DATABASE_FORMAT")


def _energy_terms(cards: list[_Card], tokens: set[str]) -> frozenset[str] | None:
    """Ledger terms ``*CONTROL_ENERGY`` switches on, or ``None`` if unstated."""
    row = _first_row(cards, "CONTROL_ENERGY", tokens)
    if row is None:
        return None  # no card, or a card that could not be read
    return frozenset(
        term
        for term, field_value in zip(_ENERGY_FIELDS, row, strict=False)
        if field_value == 2.0
    )


def _databases(cards: list[_Card], tokens: set[str]) -> frozenset[str]:
    """``*DATABASE_`` keywords requested with a non-zero output interval."""
    requested = set()
    for card in _find(cards, "DATABASE_"):
        if card.keyword.startswith(_DATABASE_SETTINGS) or not card.lines:
            continue
        row = _numbers(card.lines[0], card.keyword, tokens)
        if row is not None and row[0]:
            requested.add(card.keyword)
    return frozenset(requested)


def read_input_facts(deck_text: str, *, source_units: str) -> InputFacts:
    """Read what a keyword input establishes about a run (evidence item E1).

    Parameters
    ----------
    deck_text : str
        The complete solver input, e.g. ``Metadata.source_deck``.
    source_units : str
        The input's ``"<mass>-<length>-<time>"`` unit system, e.g.
        ``"g-mm-ms"``; every returned value is SI.

    Returns
    -------
    InputFacts
        Never raises on content it cannot resolve: it records a token in
        ``unparsable`` and leaves the affected field ``None``.

    Raises
    ------
    ValueError
        If ``source_units`` is not a known unit system.
    """
    f = unit_factors(source_units)
    tokens: set[str] = set()
    cards = _cards(deck_text, tokens)
    keywords = {c.keyword for c in cards}
    if any(k.startswith("INCLUDE") for k in keywords):
        tokens.add("include")
    if any(k.startswith("PARAMETER") for k in keywords):
        tokens.add("parameter")

    termination = _first_row(cards, "CONTROL_TERMINATION", tokens)
    timestep = _first_row(cards, "CONTROL_TIMESTEP", tokens)
    particle = _first_row(cards, "CONTROL_SPH", tokens)
    implicit = _first_row(cards, "CONTROL_IMPLICIT_GENERAL", tokens)
    materials = tuple(
        m for c in _find(cards, "MAT_") if (m := _material(c, f, tokens)) is not None
    )
    parts = _parts(cards, tokens)
    planes = _rigid_planes(cards, f["length"], tokens)
    bounds = _smoothing_bounds(cards, tokens)
    energy_terms = _energy_terms(cards, tokens)
    databases = _databases(cards, tokens)

    hidden = bool(_HIDING & tokens)

    def requested(found: bool) -> bool | None:
        return True if found else (None if hidden else False)

    other: set[str] = set()
    end_time = None
    if termination is not None:
        end_time = None if termination[0] is None else termination[0] * f["time"]
        if termination[1]:
            other.add("step_limit")
        if termination[2]:
            other.add("min_timestep")
        if termination[3]:
            other.add("energy_change")
        if termination[4] and termination[4] < 1.0e8:
            other.add("mass_change")

    if timestep is not None:
        mass_scaling: bool | None = bool(timestep[4])
    elif "CONTROL_TIMESTEP" in keywords:
        mass_scaling = None  # the card is there but could not be read
    else:
        mass_scaling = requested(False)

    if implicit is not None:
        time_integration = {0.0: "explicit", 1.0: "implicit"}.get(implicit[0] or 0.0)
    elif "CONTROL_IMPLICIT_GENERAL" in keywords or hidden:
        time_integration = None
    else:
        time_integration = "explicit"

    dimension: int | None = None
    plane_strain: bool | None = None
    if particle is not None and particle[3] is not None:
        idim = int(particle[3])
        if idim == 3:
            dimension = 3
        elif idim in (2, -2):
            dimension, plane_strain = 2, idim == 2

    return InputFacts(
        parts=parts,
        materials=materials,
        time_integration=time_integration,  # type: ignore[arg-type]
        dimension=dimension,  # type: ignore[arg-type]
        plane_strain=plane_strain,
        end_time=end_time,
        other_termination_criteria=frozenset(other),
        mass_scaling_enabled=mass_scaling,
        erosion_enabled=requested(
            any(k.startswith("MAT_ADD_EROSION") for k in keywords)
        ),
        contact_defined=requested(any(k.startswith("CONTACT_") for k in keywords)),
        prescribed_motion_defined=requested(
            any(k.startswith("BOUNDARY_PRESCRIBED_MOTION") for k in keywords)
        ),
        damping_defined=requested(any(k.startswith("DAMPING_") for k in keywords)),
        rigid_planes=planes,
        # No verified source yet classifies the solver's particle formulations
        # as pairwise conservative or not; unknown until one is recorded.
        particle_pairwise_conservative=None,
        smoothing_length_scale_bounds=bounds,
        energy_terms_computed=None if hidden else energy_terms,
        databases_requested=None if hidden else databases,
        unparsable=frozenset(tokens),
        solver="lsdyna",
    )


# --- the run record (E2-E5) ---------------------------------------------------

#: Global-statistics labels that are ledger terms, in the solver-neutral
#: vocabulary of ``core.evidence.LEDGER_TERMS``.
_LEDGER_LABELS = {
    "kinetic energy": "kinetic",
    "internal energy": "internal",
    "hourglass energy": "zero_energy_mode",
    "stonewall energy": "rigid_surface",
    "spring and damper energy": "discrete_element",
    "system damping energy": "damping",
    "sliding interface energy": "contact",
    "external work": "external_work",
    "eroded kinetic energy": "eroded_kinetic",
    "eroded internal energy": "eroded_internal",
    "eroded hourglass energy": "eroded_zero_energy_mode",
}
#: Labels that are read past: ratios and velocities derived from the above.
_DERIVED_LABELS = frozenset(
    {
        "total energy / initial energy",
        "energy ratio w/o eroded energy",
        "global x velocity",
        "global y velocity",
        "global z velocity",
        "time per zone cycle.(nanosec)",
    }
)
#: The terms the solver's total is the sum of. Eroded terms are contained in
#: the kinetic and internal terms; whether the discrete-element term is
#: contained in the internal term is not established, so it is left out and
#: the identity is checked against the solver's own total before it is used.
_TOTAL_IS_SUM_OF = (
    "kinetic",
    "internal",
    "zero_energy_mode",
    "rigid_surface",
    "damping",
    "contact",
)
#: Six printed digits per term, six terms.
_IDENTITY_RTOL = 1.0e-4

_STAT_LINE = re.compile(r"^ (.{31})\s+(-?\d+\.\d+E[+-]\d+|\d+)(?:\s+wall#\s*\d+)?\s*$")
_BANNER_DISCLAIMER = re.compile(r"errors encountered in either the documentation")
#: A diagnostic names its severity at the head of its first line, after any
#: banner asterisks; the lines below it are the diagnostic's own explanation,
#: which may name an "error" it is *reporting* rather than one the run hit.
_ERROR_LINE = re.compile(r"^[\s*]*errors?\b", re.IGNORECASE)
_WARNING_LINE = re.compile(r"^[\s*]*warnings?\b", re.IGNORECASE)
#: R12.0 prints an SVN number beside a ``Revision:`` line; later builds print
#: a ``Revision:`` describe string and no SVN line. Where both are printed the
#: SVN number stays the revision of record.
_SVN_REVISION = re.compile(r"SVN Version:\s*(\d+)")
_BANNER_REVISION = re.compile(r"Revision:\s*(\S+)")
_VERSION_TOKEN = r"[A-Za-z0-9_.+-]{1,32}"


def _ledger(
    text: str, energy: float, time: float, tokens: set[str]
) -> tuple[EnergyLedger | None, tuple[tuple[float, ...], tuple[float, ...]] | None]:
    """The energy ledger and time-step history of a global-statistics file."""
    blocks: list[dict[str, float]] = []
    for line in text.splitlines():
        found = _STAT_LINE.match(line)
        if found is None:
            continue
        label, number = found.group(1).rstrip(". "), float(found.group(2))
        if label == "time":
            blocks.append({})
        if not blocks or label in _DERIVED_LABELS:
            continue
        if label in _LEDGER_LABELS or label in ("time", "time step", "total energy"):
            key = _LEDGER_LABELS.get(label, label)
            blocks[-1][key] = blocks[-1].get(key, 0.0) + number  # walls add up
        else:
            tokens.add("unknown_ledger_label")
    if not blocks or "unknown_ledger_label" in tokens:
        return None, None
    if any(block.keys() != blocks[0].keys() for block in blocks):
        tokens.add("ragged_ledger")
        return None, None
    if not {"time", "time step", "kinetic", "total energy"} <= blocks[0].keys():
        tokens.add("incomplete_ledger")
        return None, None

    def series(key: str, factor: float) -> tuple[float, ...]:
        return tuple(block[key] * factor for block in blocks)

    times = series("time", time)
    steps = (times, series("time step", time))
    terms = {
        key: series(key, energy) for key in blocks[0] if key in _LEDGER_LABELS.values()
    }
    total = series("total energy", energy)
    identity = {key: 1 for key in _TOTAL_IS_SUM_OF if key in terms}
    scale = max(abs(x) for x in total) or 1.0
    for i, printed in enumerate(total):
        if (
            abs(sum(terms[key][i] for key in identity) - printed)
            > _IDENTITY_RTOL * scale
        ):
            tokens.add("ledger_identity_unverified")
            return None, steps
    return EnergyLedger(times, terms, identity, total), steps


def _messages(
    text: str, time: float, tokens: set[str]
) -> tuple[SolverIdentity | None, TerminationRecord, int, int]:
    """Identity, termination and diagnostic counts from a solver message file.

    Only whitelisted patterns are read; nothing of the text is kept.
    Diagnostics are counted by the lines that *raise* one -- the severity
    word at the head of the line -- so a diagnostic's own explanation does
    not count a second time, and the banner's standing disclaimer not at all.
    """
    identity = None
    stamp = re.search(r"^\s*ls-dyna\s+(\S+)\s+([sd])\s+date\b", text, re.MULTILINE)
    if stamp is not None:
        found = _SVN_REVISION.search(text) or _BANNER_REVISION.search(text)
        procs = re.search(r"MPP execution with\s+(\d+)\s+procs", text)
        version = stamp.group(1)
        revision = found.group(1) if found else None
        if revision is not None and not re.fullmatch(_VERSION_TOKEN, revision):
            tokens.add("solver_revision")  # a guess is worse than no revision
            revision = None
        if not re.fullmatch(_VERSION_TOKEN, version):
            tokens.add("solver_version")
        else:
            identity = SolverIdentity(
                "ls-dyna",
                version,
                revision,
                "double" if stamp.group(2) == "d" else "single",
                f"mpp:{procs.group(1)}" if procs else None,
            )

    status: Literal["normal", "error", "none"] = "none"
    if re.search(r"N o r m a l\s+t e r m i n a t i o n", text):
        status = "normal"
    elif re.search(r"E r r o r\s+t e r m i n a t i o n", text):
        status = "error"
    criterion = None
    if "*** termination time reached ***" in text:
        criterion = "end_time"
    elif status == "normal":
        tokens.add("termination_criterion")
    final = re.search(r"Problem time\s*=\s*(\S+)", text)
    cycles = re.search(r"Problem cycle\s*=\s*(\d+)", text)
    record = TerminationRecord(
        status,
        float(final.group(1)) * time if final else None,
        int(cycles.group(1)) if cycles else None,
        criterion,
    )

    lines = [ln for ln in text.splitlines() if not _BANNER_DISCLAIMER.search(ln)]
    n_errors = sum(bool(_ERROR_LINE.match(ln)) for ln in lines)
    n_warnings = sum(bool(_WARNING_LINE.match(ln)) for ln in lines)
    return identity, record, n_errors, n_warnings


def read_run_evidence(
    *,
    messages_text: str | None,
    global_statistics_text: str | None,
    source_units: str,
) -> RunEvidence:
    """Read what a run's text records establish into ``RunEvidence`` (ADR-0066).

    Takes text, never a directory, and keeps none of it: the record holds
    numbers, enum values and version tokens only, so a licence number, a
    host name or a path in the message file cannot reach a report. Unknown
    ledger labels, a ragged ledger, or a balance identity the solver's own
    total does not confirm yield no ledger and an ``unparsable`` token —
    never a partial one.

    Parameters
    ----------
    messages_text : str or None
        The solver's message file (rank 0 of a parallel run).
    global_statistics_text : str or None
        The ASCII global-statistics file.
    source_units : str
        Unit system of the run, ``"mass-length-time"``.

    Raises
    ------
    ValueError
        If ``source_units`` is not a known unit system.
    """
    f = unit_factors(source_units)
    tokens: set[str] = set()
    ledger, steps = None, None
    if global_statistics_text is not None:
        ledger, steps = _ledger(global_statistics_text, f["energy"], f["time"], tokens)
    if messages_text is None:
        return RunEvidence(timestep=steps, ledger=ledger, unparsable=frozenset(tokens))
    identity, record, n_errors, n_warnings = _messages(messages_text, f["time"], tokens)
    return RunEvidence(
        identity, (record,), n_errors, n_warnings, steps, ledger, frozenset(tokens)
    )
