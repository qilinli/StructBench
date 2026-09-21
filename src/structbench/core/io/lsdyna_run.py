"""Fail-closed reader of an LS-DYNA keyword input into :class:`InputFacts` (ADR-0066).

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
from dataclasses import dataclass, field

from ..evidence import InputFacts, MaterialInput, PartTraits, RigidPlane
from .lsdyna import _CANONICAL_MAT, unit_factors

__all__ = ["read_input_facts"]

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
    canonical = _CANONICAL_MAT.get(model)

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
        rigid_planes=planes,
        # No verified source yet classifies the solver's particle formulations
        # as pairwise conservative or not; unknown until one is recorded.
        particle_pairwise_conservative=None,
        smoothing_length_scale_bounds=bounds,
        unparsable=frozenset(tokens),
    )
