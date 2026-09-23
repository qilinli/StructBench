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
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

__all__ = [
    "LEDGER_TERMS",
    "PLATFORM_REASONS",
    "Absence",
    "AbsenceReason",
    "DeclaredFacts",
    "EnergyLedger",
    "Discretisation",
    "EvidenceItem",
    "InputFacts",
    "MaterialInput",
    "PartTraits",
    "RigidPlane",
    "RunEvidence",
    "SolverIdentity",
    "TerminationRecord",
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
    prescribed_motion_defined, damping_defined : bool or None
    rigid_planes : tuple of RigidPlane
    particle_pairwise_conservative : bool or None
        Whether the particle formulation conserves momentum pairwise;
        ``None`` with no particle part, or when unknown.
    smoothing_length_scale_bounds : (float, float) or None
        Minimum and maximum factors on the initial smoothing length.
    energy_terms_computed : frozenset of str or None
        Ledger terms the input switches on for computation; a term the
        solver would not compute is missing from its own printed total, so
        the ledger cannot be read as complete without this. ``None`` when
        the input does not establish it -- a solver default is never
        assumed for an absent setting.
    databases_requested : frozenset of str or None
        Output databases the input asks for with a non-zero interval, by
        keyword (``"DATABASE_GLSTAT"``). Cards that configure output rather
        than request it are not listed. ``None`` when not established.
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
    damping_defined: bool | None
    rigid_planes: tuple[RigidPlane, ...]
    particle_pairwise_conservative: bool | None
    smoothing_length_scale_bounds: tuple[float, float] | None
    energy_terms_computed: frozenset[str] | None
    databases_requested: frozenset[str] | None
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


#: Version-like identifiers: no spaces, no path separators, bounded length.
_VERSION = re.compile(r"^[A-Za-z0-9_.+-]{1,32}$")

#: The closed vocabulary of energy-ledger terms (E5). ``external_work`` is
#: the input side of the balance; every other term is a store or a sink.
LEDGER_TERMS: frozenset[str] = frozenset(
    {
        "kinetic",
        "internal",
        "external_work",
        "zero_energy_mode",
        "contact",
        "rigid_surface",
        "damping",
        "discrete_element",
        "eroded_kinetic",
        "eroded_internal",
        "eroded_zero_energy_mode",
    }
)


@dataclass(frozen=True)
class SolverIdentity:
    """Which solver produced the run (evidence item E2).

    The precision of each *output stream* is part of E2 but no run has
    supplied it yet, so it has no field (ADR-0066 clause 3).

    Parameters
    ----------
    name : str
        Lower-case token, e.g. ``"ls-dyna"``.
    version, revision : str or None
        Version-like tokens as the solver prints them.
    precision : {"single", "double"} or None
        Floating-point precision of the solver executable.
    parallel_layout : str or None
        Token ``"<kind>:<n>"``, e.g. ``"mpp:8"``.
    """

    name: str
    version: str | None = None
    revision: str | None = None
    precision: Literal["single", "double"] | None = None
    parallel_layout: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", self.name):
            raise ValueError(f"solver name must be a token, got {self.name!r}")
        for label in ("version", "revision"):
            text = getattr(self, label)
            if text is not None and not _VERSION.fullmatch(text):
                raise ValueError(f"{label} must be a version token, got {text!r}")
        layout = self.parallel_layout
        if layout is not None and not re.fullmatch(r"[a-z]+:[0-9]+", layout):
            raise ValueError(f"parallel_layout must be 'kind:n', got {layout!r}")


@dataclass(frozen=True)
class TerminationRecord:
    """How one analysis phase or restart segment ended (evidence item E3).

    Parameters
    ----------
    status : {"normal", "error", "none"}
        ``"none"``: the record ends without a termination statement.
    final_time : float or None
        Solver time at the end of the segment [s].
    n_steps : int or None
        Steps or increments taken.
    criterion : str or None
        What ended it: ``"end_time"`` or a token from
        ``InputFacts.other_termination_criteria``; ``None`` when the record
        does not say.
    """

    status: Literal["normal", "error", "none"]
    final_time: float | None = None
    n_steps: int | None = None
    criterion: str | None = None

    def __post_init__(self) -> None:
        if self.criterion is not None and not _TOKEN.fullmatch(self.criterion):
            raise ValueError(f"criterion must be a token, got {self.criterion!r}")


@dataclass(frozen=True)
class EnergyLedger:
    """The global energy ledger over time, with its balance identity (E5).

    Parameters
    ----------
    time : tuple of float
        Sample times [s].
    terms : mapping of str to tuple of float
        One series [J] per term; names from :data:`LEDGER_TERMS`.
    identity : mapping of str to int
        The declared balance identity: the terms that sum to the total
        energy, each with its sign (+1 or -1). A term present in ``terms``
        but absent here is contained in another term or is external work.
    solver_total : tuple of float or None
        The solver's own total [J], kept to check the identity against.
    origin : {"solver", "derived"}
        Whether the solver kept the ledger or it was derived from outputs.
    """

    time: tuple[float, ...]
    terms: Mapping[str, tuple[float, ...]]
    identity: Mapping[str, int]
    solver_total: tuple[float, ...] | None = None
    origin: Literal["solver", "derived"] = "solver"

    def __post_init__(self) -> None:
        unknown = set(self.terms) - LEDGER_TERMS
        if unknown:
            raise ValueError(f"unknown ledger terms {sorted(unknown)}")
        series = [*self.terms.values()]
        if self.solver_total is not None:
            series.append(self.solver_total)
        if any(len(s) != len(self.time) for s in series):
            raise ValueError("every ledger series must have one value per sample")
        if not all(math.isfinite(x) for s in (self.time, *series) for x in s):
            raise ValueError("ledger values must be finite")
        if not set(self.identity) <= set(self.terms) - {"external_work"}:
            raise ValueError("identity terms must be ledger terms other than the input")
        if "kinetic" not in self.identity or set(self.identity.values()) - {1, -1}:
            raise ValueError("the identity needs kinetic energy and signs of +1 or -1")
        object.__setattr__(self, "terms", MappingProxyType(dict(self.terms)))
        object.__setattr__(self, "identity", MappingProxyType(dict(self.identity)))


@dataclass(frozen=True)
class RunEvidence:
    """What the solver's run record establishes (evidence items E2-E5).

    A field is ``None`` when the run did not supply that item. E6, E7 and E9
    have no field yet: no run has supplied them (ADR-0066 clause 3).

    Parameters
    ----------
    identity : SolverIdentity or None
        E2.
    termination : tuple of TerminationRecord or None
        E3, one record per phase or restart segment.
    n_errors, n_warnings : int or None
        E3, the solver's diagnostics reduced to counts.
    timestep : (tuple of float, tuple of float) or None
        E4 for explicit integration: sample times and time steps [s].
    ledger : EnergyLedger or None
        E5.
    unparsable : frozenset of str
        Tokens for constructs the reader refused to resolve.
    """

    identity: SolverIdentity | None = None
    termination: tuple[TerminationRecord, ...] | None = None
    n_errors: int | None = None
    n_warnings: int | None = None
    timestep: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    ledger: EnergyLedger | None = None
    unparsable: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for token in self.unparsable:
            if not _TOKEN.fullmatch(token):
                raise ValueError(f"unparsable entries must be tokens, got {token!r}")
        if self.timestep is not None and len(self.timestep[0]) != len(self.timestep[1]):
            raise ValueError("timestep needs one step per sample time")
