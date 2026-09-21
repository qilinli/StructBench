"""Solver-neutral evidence records for reference-data verification (ADR-0066).

A run is judged from *evidence*: facts read from the solver input, facts a
benchmark declares, and (from stage 2) facts read from the solver's run
record. The records here carry that evidence in strict SI with no solver
vocabulary (ADR-0004) and no free text: every string is an enum value or
matches a fixed token pattern, so licence numbers, host names, and local
paths cannot travel through them into a report.

Adapters in ``core/io`` produce these records; ``structbench.verification``
consumes them. Nothing here is persisted inside a case file.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

__all__ = [
    "PLATFORM_REASONS",
    "Absence",
    "AbsenceReason",
    "DeclaredFacts",
    "Discretisation",
    "EvidenceItem",
    "InputFacts",
    "MaterialInput",
    "PartTraits",
    "RigidPlane",
    "UnitsAnchor",
]

#: Construct tokens and quantity locators: lower-case words, optionally
#: followed by ``:``-separated identifiers (a keyword name, a material id).
_TOKEN = re.compile(r"^[a-z][a-z0-9_]*(:[A-Za-z0-9_.+-]+)*$")
_UNIT_SYSTEM = re.compile(r"^[a-z]+-[a-z]+-[a-z]+$")


class EvidenceItem(StrEnum):
    """One item of the run-evidence requirement (ADR-0066 clause 2)."""

    E1 = "E1"  # complete solver input
    E2 = "E2"  # solver identity
    E3 = "E3"  # termination record and diagnostics
    E4 = "E4"  # time-integration record
    E5 = "E5"  # global energy ledger
    E6 = "E6"  # per-part and per-interface energy and mass
    E7 = "E7"  # applied-load resultants and reactions
    E8 = "E8"  # field output at the constitutive points
    E9 = "E9"  # sampling clock of ledger against fields
    E10A = "E10a"  # declared units of mass, length, time
    E10B = "E10b"  # three SI anchors of independent dimension


class AbsenceReason(StrEnum):
    """Why a quantity is ``not_assessable`` — and so who can close the gap."""

    # closed by the contributor
    NOT_REQUESTED = "not_requested"
    NOT_COMPUTED = "not_computed"
    SOURCE_MISSING = "source_missing"
    UNPARSABLE = "unparsable"
    SOURCE_UNREADABLE = "source_unreadable"
    # closed by nobody, for this solver
    NOT_AVAILABLE_FROM_SOLVER = "not_available_from_solver"
    # closed by the platform
    NOT_INGESTED = "not_ingested"
    UNSUPPORTED = "unsupported"
    NO_RATIFIED_CRITERION = "no_ratified_criterion"
    NO_DECLARATION_HOME = "no_declaration_home"
    STALE_DEFINITION = "stale_definition"


#: Reasons the platform must close. Rows carrying one are reported as "not
#: yet checked by this instrument" and never count as a dataset's gap.
PLATFORM_REASONS: frozenset[AbsenceReason] = frozenset(
    {
        AbsenceReason.NOT_INGESTED,
        AbsenceReason.UNSUPPORTED,
        AbsenceReason.NO_RATIFIED_CRITERION,
        AbsenceReason.NO_DECLARATION_HOME,
        AbsenceReason.STALE_DEFINITION,
    }
)

Discretisation = Literal["particle", "solid", "shell", "beam", "unknown"]


@dataclass(frozen=True)
class Absence:
    """A typed reason a measurement could not be made.

    Parameters
    ----------
    reason : AbsenceReason
        Why the evidence is absent.
    missing : frozenset of EvidenceItem
        The evidence items whose absence blocks the quantity; empty when the
        reason is not about evidence (for example ``UNSUPPORTED``).
    """

    reason: AbsenceReason
    missing: frozenset[EvidenceItem] = frozenset()

    def __post_init__(self) -> None:
        if not all(isinstance(item, EvidenceItem) for item in self.missing):
            raise TypeError("Absence.missing must contain EvidenceItem members")


@dataclass(frozen=True)
class MaterialInput:
    """One material as the solver input defines it, in SI.

    Parameters
    ----------
    material_id : int
        The input's material identifier.
    canonical_model : str or None
        Material class in the ADR-0012 vocabulary; ``None`` when unmapped.
    density : float or None
        Reference density [kg/m^3].
    shear_modulus, youngs_modulus : float or None
        Elastic moduli [Pa].
    poisson_ratio : float or None
        Dimensionless.
    yield_table : tuple of two tuples, or None
        ``(plastic strain [-], yield stress [Pa])`` knots, verbatim from the
        input (a non-monotone knot is kept — it is a finding, not an error).
    """

    material_id: int
    canonical_model: str | None
    density: float | None
    shear_modulus: float | None
    youngs_modulus: float | None
    poisson_ratio: float | None
    yield_table: tuple[tuple[float, ...], tuple[float, ...]] | None

    def __post_init__(self) -> None:
        if self.yield_table is not None:
            strain, stress = self.yield_table
            if len(strain) != len(stress) or len(strain) < 2:
                raise ValueError(
                    "yield_table needs two equal-length columns of at least two "
                    f"knots, got {len(strain)} and {len(stress)}"
                )


@dataclass(frozen=True)
class PartTraits:
    """Formulation traits of one part, derived from the solver input.

    ``under_integrated`` is ``None`` when it does not apply (a particle part)
    or cannot be established.
    """

    part_id: int
    material_id: int
    discretisation: Discretisation
    under_integrated: bool | None


@dataclass(frozen=True)
class RigidPlane:
    """An infinite rigid plane.

    Parameters
    ----------
    point : tuple of 3 float
        A point on the plane [m].
    normal : tuple of 3 float
        Unit normal pointing into the admissible half-space.
    """

    point: tuple[float, float, float]
    normal: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not math.isclose(math.hypot(*self.normal), 1.0, abs_tol=1e-9):
            raise ValueError(f"normal must be a unit vector, got {self.normal}")


@dataclass(frozen=True)
class InputFacts:
    """What the solver input establishes about a run (evidence item E1).

    A field is ``None`` when the input does not establish it — an absent
    setting is never replaced by an assumed solver default (ADR-0066
    clause 3). Times are in seconds, lengths in metres.

    Parameters
    ----------
    parts, materials
        Per-part traits and the materials they reference.
    time_integration : {"explicit", "implicit"} or None
    dimension : {2, 3} or None
    plane_strain : bool or None
        ``True`` plane strain, ``False`` axisymmetric; ``None`` in 3D.
    end_time : float or None
        Requested termination time [s].
    other_termination_criteria : frozenset of str
        Active criteria that can end a run before ``end_time``; tokens
        ``"step_limit"``, ``"min_timestep"``, ``"energy_change"``,
        ``"mass_change"``.
    mass_scaling_enabled, erosion_enabled, contact_defined,
    prescribed_motion_defined : bool or None
    rigid_planes : tuple of RigidPlane
    particle_pairwise_conservative : bool or None
        Whether the particle formulation conserves momentum pairwise;
        ``None`` with no particle part, or when unknown.
    smoothing_length_scale_bounds : (float, float) or None
        Minimum and maximum factors on the initial smoothing length.
    unparsable : frozenset of str
        Tokens for constructs the reader refused to resolve, e.g.
        ``"include"``, ``"unknown_card_layout:<KEYWORD>"``.
    """

    parts: tuple[PartTraits, ...]
    materials: tuple[MaterialInput, ...]
    time_integration: Literal["explicit", "implicit"] | None
    dimension: Literal[2, 3] | None
    plane_strain: bool | None
    end_time: float | None
    other_termination_criteria: frozenset[str]
    mass_scaling_enabled: bool | None
    erosion_enabled: bool | None
    contact_defined: bool | None
    prescribed_motion_defined: bool | None
    rigid_planes: tuple[RigidPlane, ...]
    particle_pairwise_conservative: bool | None
    smoothing_length_scale_bounds: tuple[float, float] | None
    unparsable: frozenset[str]

    def __post_init__(self) -> None:
        for token in self.unparsable | self.other_termination_criteria:
            if not _TOKEN.fullmatch(token):
                raise ValueError(
                    f"unparsable / termination entries must be tokens, got {token!r}"
                )


@dataclass(frozen=True)
class UnitsAnchor:
    """One declared SI anchor for the units check (evidence item E10b).

    The anchor states the SI value of a *named input quantity*, so agreement
    is an equality to ``digits`` significant digits rather than a judgement
    about how close an input sits to a handbook value.

    Parameters
    ----------
    kind : {"density", "stress", "velocity", "length"}
    si_value : float
        Positive and finite, in the SI unit of ``kind``.
    digits : int
        Significant digits to which equality is judged (>= 1).
    input_quantity : str
        Locator token, e.g. ``"material:2:density"`` or
        ``"geometry:extent_x"``.
    source_kind : {"handbook", "specimen_measurement", "test_report", "synthetic"}
        ``"synthetic"`` marks a deliberately non-physical value: the anchor
        then verifies arithmetic only.
    """

    kind: Literal["density", "stress", "velocity", "length"]
    si_value: float
    digits: int
    input_quantity: str
    source_kind: Literal["handbook", "specimen_measurement", "test_report", "synthetic"]

    def __post_init__(self) -> None:
        if not (math.isfinite(self.si_value) and self.si_value > 0.0):
            raise ValueError(f"si_value must be positive and finite: {self.si_value}")
        if self.digits < 1:
            raise ValueError(f"digits must be >= 1, got {self.digits}")
        if not _TOKEN.fullmatch(self.input_quantity):
            raise ValueError(
                f"input_quantity must be a locator token, got {self.input_quantity!r}"
            )


@dataclass(frozen=True)
class DeclaredFacts:
    """Facts a benchmark declares because they cannot be derived.

    The caller (the CLI) fills this from the benchmark's spec and card;
    ``structbench.verification`` never imports ``benchmarks``. A declaration
    is a *claim*: where the solver input can confirm it, a verification
    quantity cross-checks it.

    Parameters
    ----------
    unit_system : str or None
        ``"<mass>-<length>-<time>"`` token such as ``"g-mm-ms"`` (E10a).
    anchors : tuple of UnitsAnchor
        E10b; empty means none declared.
    fields : frozenset of str or None
        Namespaced response fields the dataset declares.
    discretisation : str or None
        ``"SPH"``, ``"FEM"`` or ``"coupled"``.
    erosion : bool or None
    yield_table : tuple of two tuples, or None
        ``(plastic strain [-], yield stress [Pa])`` — the operative table
        (ADR-0064's one authoritative hardening curve, converted to Pa).
    material_family : str or None
        Token such as ``"copper_alloys"``; sharpens the plausibility screens.
    quasi_static : bool
        Declared intent; the one run trait that cannot be derived.
    """

    unit_system: str | None
    anchors: tuple[UnitsAnchor, ...] = ()
    fields: frozenset[str] | None = None
    discretisation: str | None = None
    erosion: bool | None = None
    yield_table: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    material_family: str | None = None
    quasi_static: bool = False

    def __post_init__(self) -> None:
        if self.unit_system is not None and not _UNIT_SYSTEM.fullmatch(
            self.unit_system
        ):
            raise ValueError(
                f"unit_system must look like 'g-mm-ms', got {self.unit_system!r}"
            )
        if self.material_family is not None and not _TOKEN.fullmatch(
            self.material_family
        ):
            raise ValueError(
                f"material_family must be a token, got {self.material_family!r}"
            )
