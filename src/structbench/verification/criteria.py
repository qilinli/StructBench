"""The platform's criteria, and ``judge`` (ADR-0066 clauses 5 and 7).

``judge`` is a pure function of committed measurements and the records in
this file; it never touches data. A criterion is one of three kinds, fixed by
where its bound comes from:

* a **requirement**'s bound is definitional (zero, equality, the input's own
  value) — ``pass`` or ``fail``;
* an **instrument tolerance**'s bound is computed from storage precision or a
  named mechanism, never fitted to a measured value — ``pass`` or ``fail``;
* an **indicator**'s bound is a sourced *reference level*, scoped by run
  traits — ``pass`` or ``review``; the call on a ``review`` is a person's.

Source tokens (``W-…``, ``B-…``, ``M-…``) are claim ids in
``docs/plans/2026-09-21-reference-data-verification-sources.md``. A quantity
with no record here is measured and published with no verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .quantities import get_quantity
from .results import CaseMeasurements, DatasetMeasurements, Measurement, Verdict
from .traits import RunTrait

__all__ = [
    "CRITERIA",
    "CaseReport",
    "CheckResult",
    "Criterion",
    "CriterionKind",
    "DatasetReport",
    "Scope",
    "judge",
]

T = RunTrait


class CriterionKind(StrEnum):
    """Where a criterion's bound comes from."""

    REQUIREMENT = "requirement"
    INSTRUMENT = "instrument"
    INDICATOR = "indicator"


@dataclass(frozen=True)
class Scope:
    """The runs a criterion speaks for: every token must be established."""

    run_traits: frozenset[RunTrait] = frozenset()
    declared_intent: frozenset[str] = frozenset()

    def covers(self, case: CaseMeasurements) -> bool:
        traits = {str(t) for t in self.run_traits}
        return (
            traits <= case.run_traits and self.declared_intent <= case.declared_intent
        )

    def label(self) -> str:
        tokens = sorted(str(t) for t in self.run_traits) + sorted(self.declared_intent)
        return "{" + ", ".join(tokens) + "}" if tokens else "any run"


@dataclass(frozen=True)
class Criterion:
    """One bound on one quantity.

    Parameters
    ----------
    quantity : str
        A catalogue name.
    lo, hi : float or None
        The admissible interval; at least one is set.
    strict : bool
        Whether the bound excludes its end point, as its source states it.
    kind : CriterionKind
    scope : Scope
    source : str
        Dossier claim ids for an indicator; ``""`` otherwise.
    rationale : str
        Why this bound; required and rendered.
    provisional : bool
        The bound awaits a confirmation its rationale names.
    """

    quantity: str
    lo: float | None
    hi: float | None
    strict: bool
    kind: CriterionKind
    scope: Scope
    source: str
    rationale: str
    provisional: bool = False

    def __post_init__(self) -> None:
        get_quantity(self.quantity)  # raises on an unknown name
        if self.lo is None and self.hi is None:
            raise ValueError(f"{self.quantity}: a criterion needs lo or hi")
        if not self.rationale.strip():
            raise ValueError(f"{self.quantity}: a criterion needs a rationale")
        if self.kind is CriterionKind.INDICATOR and not self.source:
            raise ValueError(f"{self.quantity}: a reference level cites its source")

    def admits(self, value: float) -> bool:
        if self.strict:
            above = self.lo is None or value > self.lo
            return above and (self.hi is None or value < self.hi)
        above = self.lo is None or value >= self.lo
        return above and (self.hi is None or value <= self.hi)

    def label(self) -> str:
        """The bound as rendered, e.g. ``"<= 0.01 {explicit, lagrangian_mesh}"``."""
        lt = "<" if self.strict else "<="
        if self.lo is not None and self.hi is not None:
            bound = f"{self.lo:g} {lt} x {lt} {self.hi:g}"
        elif self.hi is not None:
            bound = f"{lt} {self.hi:g}"
        else:
            bound = f"{'>' if self.strict else '>='} {self.lo:g}"
        return f"{bound} {self.scope.label()}"


def _zero(quantity: str, rationale: str) -> Criterion:
    return Criterion(
        quantity, None, 0.0, False, CriterionKind.REQUIREMENT, Scope(), "", rationale
    )


def _tolerance(
    quantity: str, lo: float | None, hi: float | None, rationale: str
) -> Criterion:
    # provisional: each bound below is argued from float32 storage and has not
    # yet been confirmed on a second solver's output
    return Criterion(
        quantity,
        lo,
        hi,
        False,
        CriterionKind.INSTRUMENT,
        Scope(),
        "",
        rationale,
        provisional=True,
    )


def _level(
    quantity: str,
    lo: float | None,
    hi: float | None,
    strict: bool,
    traits: set[RunTrait],
    source: str,
    rationale: str,
    *,
    provisional: bool = False,
) -> Criterion:
    return Criterion(
        quantity,
        lo,
        hi,
        strict,
        CriterionKind.INDICATOR,
        Scope(frozenset(traits)),
        source,
        rationale,
        provisional,
    )


_MESH = {T.EXPLICIT, T.LAGRANGIAN_MESH}
_KNOWN_FIRST = (
    " Fixed from the source alone; one test-bed value (a legacy particle run whose"
    " total energy rises by several percent) was known beforehand."
)
_BLM = (
    "Over-the-run energy residual 'generally on the order of 10^-2', rendered as"
    " the one number the source prints. A stability check, not an accuracy limit;"
    " the source checks every step, so a sampled pass is weaker. Read as snippets"
    " only." + _KNOWN_FIRST
)

_REQUIREMENTS = (
    _zero("nonfinite_count", "A stored response contains no NaN or infinity."),
    _zero("time_axis_monotone", "Stored times strictly increase."),
    _zero(
        "terminal_artifact_frames",
        "Every stored interval is a sampling interval: a frame written off the"
        " interval makes index-based rates wrong (ADR-0028).",
    ),
    _zero(
        "elements_without_input_part",
        "Every stored element belongs to a part the solver input defines.",
    ),
    _zero(
        "fields_match_declaration",
        "The stored fields are exactly the fields the benchmark declares.",
    ),
    _zero(
        "declared_traits_match_input",
        "What the benchmark declares about the run agrees with the solver input.",
    ),
    _zero(
        "particle_deactivated_count",
        "With erosion off, no particle is ever deactivated.",
    ),
    _zero(
        "state_variable_decrease_max",
        "The material class says its state variable never decreases; rounding to"
        " float32 is monotone, so storage adds no tolerance.",
    ),
    Criterion(
        "state_variable_min",
        0.0,
        None,
        False,
        CriterionKind.REQUIREMENT,
        Scope(),
        "",
        "The material class bounds its state variable below by zero.",
    ),
    _zero(
        "yield_table_monotone",
        "A tabulated yield stress does not fall with its state variable; a dip is"
        " an input defect.",
    ),
    Criterion(
        "yield_table_covers_range",
        None,
        1.0,
        False,
        CriterionKind.REQUIREMENT,
        Scope(),
        "",
        "The largest state reached lies within the table, so no yield stress was"
        " extrapolated.",
    ),
    Criterion(
        "yield_table_matches_input",
        None,
        1.0e-9,
        False,
        CriterionKind.REQUIREMENT,
        Scope(),
        "",
        "The declared table equals the input's. Both reach SI through their own"
        " float64 conversion, so equality is taken to the nine significant digits"
        " the report stores.",
    ),
    _zero(
        "units_anchors_consistent",
        "Each anchor states the SI value of a named input quantity to declared"
        " digits, so agreement is an equality.",
    ),
    Criterion(
        "yield_saturation_min",
        0.5,
        None,
        False,
        CriterionKind.REQUIREMENT,
        Scope(),
        "",
        "A scale screen, not a return-mapping tolerance: among points loading"
        " through a stored frame, one must sit near the yield surface. One half is"
        " far below one and above 0.145, the reciprocal of the smallest stress"
        " factor between common consistent unit systems (psi against kPa, 6.9)."
        " Chosen after the test-bed value (about one) was known, not from it.",
        provisional=True,
    ),
)

_TOLERANCES = (
    _tolerance(
        "reached_end_time",
        1.0 - 1.0e-6,
        None,
        "The last stored time reaches the requested end time. float32 time stamps"
        " resolve 1.2e-7 of their value; 1e-6 leaves a decade. No upper bound: an"
        " explicit run may overshoot by one step.",
    ),
    _tolerance(
        "smoothing_length_within_input_bounds",
        None,
        1.0e-5,
        "The scale is a ratio of two float32 values, resolved to about 2.4e-7;"
        " 1e-5 leaves over a decade.",
    ),
    _tolerance(
        "active_mass_drift",
        None,
        1.0e-6,
        "With mass scaling and deletion off, every stored mass is one constant"
        " rounded the same way each frame; 1e-6 is a decade above float32"
        " resolution.",
    ),
    _tolerance(
        "plane_strain_ezz_max",
        None,
        1.0e-6,
        "Plane strain sets the out-of-plane normal strain to zero; 1e-6 is a"
        " decade above float32 resolution of a strain of order one, and three"
        " decades below the smallest hoop strain of an axisymmetric run.",
    ),
    _tolerance(
        "out_of_plane_shear_max",
        None,
        0.0,
        "A two-dimensional formulation carries no out-of-plane shear; the slots"
        " hold exact zeros, so any other value is a slot mix-up.",
    ),
    _tolerance(
        "density_slot_matches_input",
        None,
        1.0e-5,
        "At the first stored state an unloaded part has its input density, to"
        " float32 resolution (1.2e-7); a wrong slot is off by orders of magnitude."
        " A preloaded first state needs its own bound.",
    ),
)

_LEVELS = (
    *(
        _level(
            name, None, 0.01, False, traits, "B-BLM-1, B-BLM-2", _BLM, provisional=True
        )
        for name in ("energy_gain_max", "energy_loss_max")
        for traits in (_MESH, {T.EXPLICIT, T.PARTICLE_CONSERVATIVE})
    ),
    _level(
        "total_energy_change_final",
        -0.10,
        0.10,
        False,
        _MESH | {T.INITIAL_ENERGY_DRIVEN},
        "W-W179-02, W-W179-12",
        "Total energy 'must not vary more than 10 percent from the beginning of"
        " the run to the end'; the denominator is the initial total and the source"
        " has no external-work term. Roadside-crash practice with finite"
        " elements." + _KNOWN_FIRST,
    ),
    _level(
        "zero_energy_mode_final_over_initial_total",
        None,
        0.05,
        True,
        _MESH | {T.INITIAL_ENERGY_DRIVEN},
        "W-W179-03",
        "Final zero-energy-mode energy under five percent of the initial total;"
        " the pages read state no provenance for the figure.",
    ),
    _level(
        "zero_energy_mode_final_over_internal_final",
        None,
        0.10,
        True,
        _MESH,
        "W-W179-04",
        "Final zero-energy-mode energy under ten percent of final internal energy.",
    ),
    _level(
        "zero_energy_mode_top_part_final_over_internal_final",
        None,
        0.10,
        True,
        _MESH,
        "W-W179-05",
        "The one part with the most zero-energy-mode energy, not the worst ratio"
        " over all parts.",
    ),
    _level(
        "zero_energy_mode_peak_over_internal_peak",
        None,
        0.10,
        True,
        _MESH,
        "W-ENCAP-01, W-ENCAP-02",
        "Occupant virtual-testing protocols, 'max. internal energy' read as the"
        " full setup's; use beyond that domain is the platform's extension.",
    ),
    _level(
        "added_mass_fraction",
        None,
        0.05,
        True,
        {T.EXPLICIT, T.MASS_SCALED},
        "W-ENCAP-01, W-ENCAP-02",
        "Maximum added mass under five percent of the model's. The source's 2.5"
        " percent for separately built models is not adopted.",
    ),
    _level(
        "added_mass_top_part_fraction",
        None,
        0.10,
        True,
        {T.EXPLICIT, T.MASS_SCALED},
        "W-W179-07",
        "The one part with the most added mass, at the last sample.",
    ),
    _level(
        "added_mass_moving_fraction",
        None,
        0.05,
        True,
        {T.EXPLICIT, T.MASS_SCALED, T.INITIAL_ENERGY_DRIVEN},
        "W-W179-08",
        "Parts given an initial velocity, at the last sample.",
    ),
    _level(
        "input_density_plausible",
        16.0,
        22590.0,
        False,
        set(),
        "M-D6, M-D5, M-D10",
        "Flexible polymer foam to osmium. The only family-free screen that sees a"
        " mass-unit error.",
    ),
    _level(
        "input_constants_plausible",
        3.0e5,
        1.0e12,
        False,
        set(),
        "M-E4, M-D7",
        "Young's modulus from flexible foam to diamond: sees length or time errors"
        " of several decades, not a factor of a thousand within a family.",
    ),
    _level(
        "response_magnitudes_plausible",
        None,
        3000.0,
        False,
        set(),
        "M-S9",
        "Above 3 km/s is the hypervelocity regime, outside structural impact."
        " Blind to the mass unit.",
    ),
)

CRITERIA: tuple[Criterion, ...] = (*_REQUIREMENTS, *_TOLERANCES, *_LEVELS)


@dataclass(frozen=True)
class CheckResult:
    """One quantity's verdict for one case.

    ``reason`` is ``""`` on a pass, an ``AbsenceReason`` value when not
    assessable, else ``"outside_bound"`` or ``"exceeds_reference_level"``.
    ``out_of_scope_levels`` lists the levels that exist for other scopes when
    none covers this run, so a person can make the call.
    """

    quantity: str
    verdict: Verdict
    value: float | None
    unit: str
    reason: str = ""
    missing: tuple[str, ...] = ()
    criterion: str = ""
    criterion_source: str = ""
    provisional: bool = False
    out_of_scope_levels: tuple[str, ...] = ()


@dataclass(frozen=True)
class CaseReport:
    case_id: str
    results: tuple[CheckResult, ...]


@dataclass(frozen=True)
class DatasetReport:
    measurements: DatasetMeasurements
    cases: tuple[CaseReport, ...]


def _judge_one(
    m: Measurement,
    case: CaseMeasurements,
    criteria: Sequence[Criterion],
    stale: bool,
) -> CheckResult:
    def not_assessable(
        reason: str, missing: tuple[str, ...] = (), levels: tuple[str, ...] = ()
    ) -> CheckResult:
        return CheckResult(
            m.quantity,
            Verdict.NOT_ASSESSABLE,
            m.value,
            m.unit,
            reason,
            missing,
            out_of_scope_levels=levels,
        )

    if m.not_applicable:
        return CheckResult(m.quantity, Verdict.NOT_APPLICABLE, None, m.unit)
    if m.absence is not None:
        missing = tuple(sorted(str(item) for item in m.absence.missing))
        return not_assessable(str(m.absence.reason), missing)
    if stale:
        return not_assessable("stale_definition")
    assert m.value is not None
    mine = [c for c in criteria if c.quantity == m.quantity]
    covering = [c for c in mine if c.scope.covers(case)]
    if len(covering) > 1:
        raise ValueError(f"{m.quantity}: overlapping criteria scopes")
    if not covering:
        return not_assessable(
            "no_ratified_criterion",
            levels=tuple(f"{c.label()} [{c.source}]" for c in mine),
        )
    criterion = covering[0]
    if criterion.admits(m.value):
        verdict, reason = Verdict.PASS, ""
    elif criterion.kind is CriterionKind.INDICATOR:
        verdict, reason = Verdict.REVIEW, "exceeds_reference_level"
    else:
        verdict, reason = Verdict.FAIL, "outside_bound"
    return CheckResult(
        m.quantity,
        verdict,
        m.value,
        m.unit,
        reason,
        criterion=criterion.label(),
        criterion_source=criterion.source,
        provisional=criterion.provisional,
    )


def judge(
    measurements: DatasetMeasurements, criteria: Sequence[Criterion] = CRITERIA
) -> DatasetReport:
    """Verdicts for every measurement, from the measurements and criteria alone.

    A measurement taken under a ``definition_version`` other than the
    catalogue's current one is ``not_assessable / stale_definition`` — it
    asks to be re-measured, it is never an error.
    """
    stale = {
        name
        for name, version in measurements.definition_versions.items()
        if get_quantity(name).definition_version != version
    }
    cases = tuple(
        CaseReport(
            case.case_id,
            tuple(
                _judge_one(m, case, criteria, m.quantity in stale)
                for m in case.measurements
            ),
        )
        for case in measurements.cases
    )
    return DatasetReport(measurements, cases)
