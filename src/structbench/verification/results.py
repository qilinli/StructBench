"""Result types for reference-data verification (ADR-0066).

``measure`` produces threshold-free :class:`Measurement` records; ``judge``
turns them into verdicts. These types sit below ``eval`` and ``benchmarks``
in the dependency graph so both can use them without a cycle.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from ..core import Absence

__all__ = [
    "CaseMeasurements",
    "Category",
    "DatasetMeasurements",
    "Location",
    "Measurement",
    "Status",
    "Verdict",
]

#: ``Measurement.detail`` strings are tokens, never free text (ADR-0066 cl. 3).
_DETAIL_TOKEN = re.compile(r"^[a-z0-9_:.+-]{1,64}$")


class Verdict(StrEnum):
    """The five verdicts, each owned by one stage of ``judge`` (clause 5)."""

    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"  # an indicator above its reference level; a person decides
    NOT_APPLICABLE = "not_applicable"
    NOT_ASSESSABLE = "not_assessable"


class Category(StrEnum):
    """Catalogue categories."""

    NUMERICAL_HEALTH = "numerical_health"
    CONSERVATION = "conservation"
    CONSTITUTIVE = "constitutive"
    UNITS = "units"
    DATA_INTEGRITY = "data_integrity"


class Location(StrEnum):
    """Where a measurement's evidence was read — a locator, not a definition."""

    CASE = "case"
    INPUT = "input"
    RUN = "run"
    DECLARED = "declared"


class Status(StrEnum):
    """Whether a catalogue row has a measure yet."""

    SPECIFIED = "specified"
    IMPLEMENTED = "implemented"


@dataclass(frozen=True)
class Measurement:
    """One quantity measured on one case — a value, an absence, or neither.

    Exactly one of three states holds: a finite ``value``; an ``absence``
    (the quantity exists but could not be measured); or ``not_applicable``
    (the quantity does not exist for this formulation).

    Parameters
    ----------
    quantity : str
        Catalogue name.
    value : float or None
        The measured value in ``unit``.
    unit : str
        SI unit; ``"1"`` for a dimensionless quantity.
    locations : frozenset of Location
        Where the evidence was read.
    n_samples : int or None
        How many samples the value summarises.
    absence : Absence or None
    not_applicable : bool
    detail : mapping
        Small numeric or token-valued context (a frame index, a knot
        distance). String values must be tokens.

    Raises
    ------
    ValueError
        If the three-state invariant is broken, the value is not finite, or
        a ``detail`` string is not a token.
    """

    quantity: str
    value: float | None
    unit: str
    locations: frozenset[Location] = frozenset()
    n_samples: int | None = None
    absence: Absence | None = None
    not_applicable: bool = False
    detail: Mapping[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.not_applicable:
            if self.value is not None or self.absence is not None:
                raise ValueError(
                    f"{self.quantity}: not_applicable carries no value and no absence"
                )
        elif (self.value is None) == (self.absence is None):
            raise ValueError(
                f"{self.quantity}: exactly one of value and absence must be set"
            )
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError(f"{self.quantity}: value must be finite, got {self.value}")
        for key, item in self.detail.items():
            if isinstance(item, str) and not _DETAIL_TOKEN.fullmatch(item):
                raise ValueError(
                    f"{self.quantity}: detail[{key!r}] must be a token, got {item!r}"
                )
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))


@dataclass(frozen=True)
class CaseMeasurements:
    """Every catalogue quantity measured on one case.

    Parameters
    ----------
    case_id : str
    file_sha256 : str or None
        Digest of the measured case file; ``None`` for a run with no case.
    run_traits : frozenset of str
        Run-trait tokens derived from the input; reference levels are scoped
        on them, so they travel with the measurements.
    declared_intent : frozenset of str
        ``{"quasi_static"}`` or empty.
    measurements : tuple of Measurement
        Sorted by quantity name, one per catalogue row.
    """

    case_id: str
    file_sha256: str | None
    run_traits: frozenset[str]
    declared_intent: frozenset[str]
    measurements: tuple[Measurement, ...]

    def __post_init__(self) -> None:
        names = [m.quantity for m in self.measurements]
        if names != sorted(set(names)):
            raise ValueError(
                f"{self.case_id}: measurements must be sorted by quantity and unique"
            )


@dataclass(frozen=True)
class DatasetMeasurements:
    """The committed record: measurements for every case of a dataset.

    Verdicts are generated from this plus the criteria; nothing else is
    needed to judge (ADR-0066 clause 8).

    Parameters
    ----------
    benchmark : str or None
        Registry name, or ``None`` for an unregistered set of runs.
    dataset_revision : str or None
        The dataset tag the cases came from.
    structbench_version : str
    definition_versions : mapping of str to int
        Quantity name -> ``definition_version`` at measurement time.
    cases : tuple of CaseMeasurements
        Sorted by ``case_id``.
    """

    benchmark: str | None
    dataset_revision: str | None
    structbench_version: str
    definition_versions: Mapping[str, int]
    cases: tuple[CaseMeasurements, ...]

    def __post_init__(self) -> None:
        ids = [c.case_id for c in self.cases]
        if ids != sorted(set(ids)):
            raise ValueError("cases must be sorted by case_id and unique")
        object.__setattr__(
            self,
            "definition_versions",
            MappingProxyType(dict(self.definition_versions)),
        )
