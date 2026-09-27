"""Pairs of (test, canonical case) against a reference set: deviations, never verdicts.

A ``pairs.toml`` names the reference set, labels the variants compared side
by side, and lists one pair per (variant, test): either a canonical case file
or ``status = "aborted"`` with a reason, for a run that did not complete.
"""

from __future__ import annotations

import statistics
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from structbench.core import Case
from structbench.core.io import read_case
from structbench.validation.measures import taylor
from structbench.validation.reference import ReferenceSet, Test

#: Canonical cases are SI; reference sets are in millimetres.
CANONICAL_TO_MM = 1e3
#: The one status a pairs file may state; the others are the comparison's own.
STATED_STATUSES = ("aborted",)
#: The measures the summary is built over, with the ``Result`` field each reads.
MEASURES = (
    ("length", "dev_length"),
    ("largest_radius", "dev_radius"),
    ("lateral_rms", "dev_lateral_rms"),
)


class PairsError(ValueError):
    """A pairs file the comparison cannot use; says which entry."""


@dataclass(frozen=True)
class Pair:
    variant: str
    test: str
    case: Path | None  # resolved, for reading
    case_text: str | None  # as written in the pairs file, for the record
    status: str | None
    reason: str | None


@dataclass(frozen=True)
class Result:
    variant: str
    test: str
    case_id: str | None
    case_path: str | None  # as written in the pairs file
    status: str  # completed | aborted | missing | unmeasurable
    reason: str | None
    measured_length_mm: float
    measured_radius_mm: float
    measured_lateral_mm: tuple[float, ...]
    length_mm: float | None
    length_band_mm: float | None
    largest_radius_mm: float | None
    lateral_radii_mm: tuple[float, ...] | None
    dev_length: float | None
    dev_radius: float | None
    dev_lateral_rms: float | None


Summary = dict[str, dict[str, dict[str, float | int]]]


@dataclass(frozen=True)
class Comparison:
    results: tuple[Result, ...]
    summary: Summary  # variant -> measure -> {min, median, max, n} (n only when 0)


def load_pairs(path: Path) -> tuple[str, dict[str, str], list[Pair]]:
    """``(reference name, variant labels, pairs)`` from a ``pairs.toml``."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise PairsError(f"{path.name}: {exc}") from exc
    name = data.get("reference")
    if not isinstance(name, str):
        raise PairsError(f"{path.name}: reference: a set name is required")
    raw_variants = data.get("variants", {})
    if not isinstance(raw_variants, dict):
        raise PairsError(f"{path.name}: [variants] must be a table of labels")
    variants = {str(k): str(v) for k, v in raw_variants.items()}
    raw_pairs = data.get("pair", [])
    if not isinstance(raw_pairs, list) or not all(
        isinstance(p, dict) for p in raw_pairs
    ):
        raise PairsError(f"{path.name}: pairs are [[pair]] tables")
    pairs: list[Pair] = []
    seen: set[tuple[str, str]] = set()
    for k, raw in enumerate(raw_pairs):
        where = f"{path.name}: pair {k + 1}"
        variant, test = raw.get("variant"), raw.get("test")
        if variant not in variants:
            raise PairsError(f"{where}: variant {variant!r} is not in [variants]")
        if test is None:
            raise PairsError(f"{where}: test is required")
        key = (str(variant), str(test))
        if key in seen:
            raise PairsError(
                f"{where}: variant {variant!r} test {test!r} appears twice"
            )
        seen.add(key)
        if "case" in raw and "status" in raw:
            raise PairsError(f"{where}: give a case file or a status, not both")
        case: Path | None
        case_text: str | None
        status: str | None
        if "case" in raw:
            case_text = str(raw["case"])
            given = Path(case_text)
            case = given if given.is_absolute() else (path.parent / given).resolve()
            status = None
        elif "status" in raw:
            status = str(raw["status"])
            if status not in STATED_STATUSES:
                raise PairsError(
                    f"{where}: status {status!r}; a pairs file may state only "
                    f"{', '.join(STATED_STATUSES)} (with a reason)"
                )
            case = case_text = None
        else:
            raise PairsError(f"{where}: give a case file or a status")
        reason = raw.get("reason")
        pairs.append(
            Pair(*key, case, case_text, status, None if reason is None else str(reason))
        )
    return name, variants, pairs


def _measure_mm(case: Case, fractions: tuple[float, ...]) -> taylor.TaylorOutcome:
    o = taylor.taylor_outcome(case, fractions)
    s = CANONICAL_TO_MM
    return taylor.TaylorOutcome(
        o.length * s,
        o.length_band * s,
        o.largest_radius * s,
        tuple(w * s for w in o.lateral_radii),
    )


def _gap(
    pair: Pair, t: Test, status: str, reason: str | None, case_id: str | None = None
) -> Result:
    return Result(
        pair.variant,
        pair.test,
        case_id,
        pair.case_text,
        status,
        reason,
        t.Lf_mm,
        t.Rf_mm,
        tuple(t.Wf_mm),
        *(None,) * 7,
    )


def compare(
    ref: ReferenceSet,
    pairs: list[Pair],
    variants: dict[str, str],
    *,
    case_loader: Callable[[Path], Case] = read_case,
) -> Comparison:
    """Every pair measured against its test at the set's own heights; gaps named."""
    results: list[Result] = []
    for p in pairs:
        if p.variant not in variants:
            raise PairsError(f"variant {p.variant!r} is not in [variants]")
        try:
            t = ref.test(p.test)
        except KeyError:
            raise PairsError(
                f"test {p.test!r} is not in reference set {ref.name!r}"
            ) from None
        if p.case is None:
            results.append(_gap(p, t, p.status or "aborted", p.reason))
            continue
        if not p.case.is_file():
            results.append(_gap(p, t, "missing", f"no file at {p.case_text}"))
            continue
        try:
            case = case_loader(p.case)
        except Exception as exc:  # h5py, the schema reader: named, not raised
            results.append(_gap(p, t, "unmeasurable", f"{type(exc).__name__}: {exc}"))
            continue
        try:
            o = _measure_mm(case, ref.fractions)
        except (taylor.OutlineError, KeyError, ValueError, IndexError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            results.append(_gap(p, t, "unmeasurable", reason, case.metadata.case_id))
            continue
        dev_w = [(a - b) / b for a, b in zip(o.lateral_radii, t.Wf_mm, strict=True)]
        results.append(
            Result(
                p.variant,
                p.test,
                case.metadata.case_id,
                p.case_text,
                "completed",
                None,
                t.Lf_mm,
                t.Rf_mm,
                tuple(t.Wf_mm),
                o.length,
                o.length_band,
                o.largest_radius,
                o.lateral_radii,
                o.length / t.Lf_mm - 1.0,
                o.largest_radius / t.Rf_mm - 1.0,
                float(np.sqrt(np.mean(np.square(dev_w)))),
            )
        )
    summary: Summary = {}
    for v in variants:
        done = [r for r in results if r.variant == v and r.status == "completed"]
        summary[v] = {}
        for measure, attr in MEASURES:
            values = [float(getattr(r, attr)) for r in done]
            summary[v][measure] = (
                {
                    "min": min(values),
                    "median": statistics.median(values),
                    "max": max(values),
                    "n": len(values),
                }
                if values
                else {"n": 0}
            )
    return Comparison(tuple(results), summary)
