# Design: reference-data verification (`verification/`) and the first slice

**Date**: 2026-09-21
**Status**: Approved with ADR-0066 (maintainer, in-session 2026-09-21)
**Scope**: new module `src/structbench/verification/`; two new `core` files
(`core/evidence.py`, `core/io/lsdyna_run.py`); `cli/datacheck.py`; per-dataset
glue under `data_generation/`; `docs/datachecks/`; `docs/ARCHITECTURE.md`;
README Roadmap. Governing decision: ADR-0066 (Accepted), which is normative
wherever the two overlap.

---

## Problem

ADR-0065 asks the platform to ship reference data "whose uncertainty is
declared" and says none of the four benchmarks would be admitted today. The
repository cannot currently say *why* for any single run: no code measures
termination, time integration, energy balance, particle health, constitutive
consistency, or data integrity; the only such checks are asserts in
exploratory `tools/state_probe/`; provenance says `"unknown"`.

The goal is to encode that judgment once, as deterministic checks, so that
(a) what varies by solver, by formulation, by material class, and by
benchmark each has exactly one home; (b) the same kernels later serve
model-side verification (ADR-0065 follow-up 1); (c) what a run must supply is
stated as a requirement future data can be built to; and (d) the footprint is
the smallest that works.

## Decisions (maintainer, in-session 2026-09-21)

1. **New top-level module, named `verification/`**, layered between
   `datasets/` and the modules that use it.
2. **Criteria are platform-level and independent of our data**: requirement
   and tolerance bounds are definitional or mechanism-derived, reference
   levels are fixed in advance from a verified external source as the source
   states it, and none is adjusted in light of a StructBench measurement.
   Until one exists a row reads `not_assessable / no_ratified_criterion`.
3. **The published record is measurements, not verdicts**:
   `docs/datachecks/<benchmark>.json` plus a generated `.md`, with a
   data-free drift test.
4. **The first stage covers the full set of quantities measurable from a
   case file and its stored input**, not a lean port.
5. **Taylor is the first test bed**; wave-1D and notch-impact follow.
6. **Design from what verification requires, not from what the existing
   archives contain** (logged in `docs/CORRECTIONS.md`, 2026-09-21). The
   legacy sweeps are small, pre-date the repository, and are
   unrepresentative. What a quantity needs becomes a stated requirement;
   the legacy archives are a test bed, and nothing is shaped to make them
   assessable.
7. **The energy criterion is a fixed general indicator, not a single hard
   limit** — a residual in Belytschko's form, with levels fixed in advance
   and related to the problem and the solver, and the final call on an
   exceedance left to a person.
8. **Units are checked, as a category of their own** — they are a large
   source of error, and the one units error this repository has had was
   invisible to every self-consistent check.
9. **The indicator treatment extends beyond energy** to the zero-energy-mode
   ratio, added mass, contact energy, time-step collapse, the quasi-static
   kinetic ratio, warning counts, particle-neighbour statistics, and the
   plausibility screens. Where the source survey found no published limit,
   the quantity stays measured with no level until a source is found.

Taken after stages 1 and 2 had run on the test bed (same day):

10. **A frame written off the sampling interval is a fact, not a defect.**
    A solver that writes its state at the exact termination time did
    nothing wrong, and loaders already drop that frame (ADR-0028).
    `terminal_artifact_frames` is measured and reported with no
    criterion; it was first catalogued as a requirement.
11. **Strength gets its own plausibility row**, `input_strength_plausible`,
    against the sourced family-free range. It is the one stress-unit
    screen that works on an input with no Young's modulus;
    `input_constants_plausible` is Young's modulus only, because a level
    attaches only to the statistic its source states.
12. **For the kinetic-energy closure, shared sample instants found in the
    files stand in for a declared sampling clock (E9).** A stored frame
    and a ledger sample coincide within a hundredth of a stored interval;
    a frame with no partner is left out. The closure ships measured, with
    no tolerance yet. The other three E9 rows are decided after it has
    been seen on the test bed.
13. **No sourced reference level is ratified, so none gives a verdict.**
    The maintainer cannot confirm the levels against their sources, two
    of which were read only as page images or snippets. They are not
    published as this platform's standard: every indicator is reported
    as a measurement, with the published level shown beside it for
    context (`not_assessable / no_ratified_criterion`). Verdicts come
    from definitional requirements and instrument tolerances only. A
    level starts judging when its record in `criteria.py` is marked
    ratified — a one-line change that re-renders every report with no
    data access. The record was published on this footing
    (`docs/datachecks/taylor_impact_2d.{json,md}`).

### Calls made in drafting

*Confirmed by the acceptance of ADR-0066 (2026-09-21), which encodes them —
except the conformance run, which stays the maintainer's to decide and
execute.*

- **Catalogue rows exist for every specified quantity, as data.** The trait
  gate and the evidence gate run for all of them, so a contributor is told
  what is missing even where no measure exists yet; only measures are
  deferred. (The alternative — unbuilt quantities in documentation only —
  hides exactly the gaps the requirement exists to show.)
- **Energy closures are defined against the ledger**, not against globals a
  case file happens to store. They move to the run-evidence stage and read
  `not_assessable` on a run with no ledger.
- **One conformance run** (below): an unpublished re-run of a single test-bed
  case with the standard input block. It needs the licensed solver and is
  therefore the maintainer's to decide and to execute.
- **An exceedance is a fifth verdict, `review`,** rather than a `fail` the
  user overrides; and whoever is accountable for a dataset may record a
  dated *disposition* beside it that never changes the verdict.
- **"Related to the problem and the solver" is read as scoping by run
  traits derived from the input and the ledger** — time integrator, spatial
  conservation class, energy source, mass scaling, friction, dimension —
  not by solver product (ADR-0004) and not by a label a dataset gives
  itself, which would let it choose its own level. Only quasi-static intent
  is declared, because it cannot be derived.
- **Levels are scoped no wider than their source's domain** *(settled —
  maintainer, 2026-09-21: keep it simple)*. The general energy level comes
  from a finite-element text, so it covers a Lagrangian mesh and a
  pairwise-conservative particle method. A run outside every scope has its
  indicator measured and published with the out-of-scope levels as context,
  and no verdict. An earlier draft argued a wider scope from a test-bed
  measurement; that is withdrawn (ADR-0066, rejected alternatives).

## How the design was reached

A first draft came from three independent module designs and three
adversarial reviews (governance, computational mechanics, buildability). The
maintainer reversed its evidence model (decision 6); a first revision still
copied the requirement from one solver's output list, which a second review
on the same three lenses caught. What follows is the result.

## The model

**Two verbs.** `measure`: evidence → threshold-free measurement, or a typed
absence. `judge`: measurements + criteria → verdicts; never touches data.

**Five verdicts, fixed ownership and order.**

| Order | Owner | Verdict |
|---|---|---|
| 1 | formulation traits | `not_applicable` — the quantity does not exist for this formulation |
| 2 | evidence absence | `not_assessable` + reason + the missing evidence items |
| 3 | measurement vs a requirement or instrument tolerance | `pass` / `fail` |
| 3 | measurement vs an indicator's reference level | `pass` / `review` — above the level; a person decides whether it matters for their use |

**Three kinds of criterion, fixed by where the bound comes from.** A
*requirement*'s bound is definitional — zero, exact equality, or the input's
own value. An *instrument tolerance*'s bound is computed from storage
precision or a confirmed mechanism, including where it serves a requirement.
An *indicator*'s bound is a sourced reference level: one fixed definition,
levels scoped by run traits, each attachable only where the source's
statistic, normalisation, and evaluation time match the quantity's
definition. A `review` may carry a dated, attributed *disposition* —
accepted or rejected, with a rationale, naming the definition version and
level it answers — rendered beside the row; it never changes the measurement
or the verdict, and a reader is free to disagree with it.

The test between rows 1 and 2: *could better evidence change the answer?*
Unknown traits give `not_assessable`, never `not_applicable`. Traits go first
because the same input can mislead otherwise: zero-energy-mode work that the
solver was told not to compute is `not_applicable` on a particle part (no
such modes exist) and `not_assessable / not_computed` on an under-integrated
element part, where it is a real gap in the energy balance.

**Reasons say who can close the gap.**

| Reason | Closed by | Meaning |
|---|---|---|
| `not_requested` | contributor | the input shows the output was never asked for |
| `not_computed` | contributor | requested, but the solver's accounting for it is switched off |
| `source_missing` | contributor | the evidence item was not supplied with the run — or E1 is absent, so nothing about requests can be established |
| `unparsable` | contributor | the input uses a construct the reader does not resolve |
| `source_unreadable` | contributor | supplied but corrupt |
| `not_available_from_solver` | nobody, for this solver | the solver cannot report it and it cannot be derived |
| `not_ingested` | platform | the solver wrote it; the adapter drops it |
| `unsupported` | platform | evidence is present in a form, or for a material class, this version does not handle |
| `no_ratified_criterion` | platform | measured; no criterion has been ratified |
| `no_declaration_home` | platform | a declared fact is required but benchmarks have nowhere to record it yet |
| `stale_definition` | platform | measured under an older definition; re-measure |

Platform-side reasons are reported in a separate block, "not yet checked by
this instrument", and never count as a dataset's gap.

**Evidence locations.** Each measurement lists where its evidence was read:
`case` (the canonical file), `input` (the solver input stored in it), `run`
(the run-evidence record), `declared` (benchmark declarations). This is a
locator, not part of a quantity's definition.

## The run-evidence requirement

ADR-0066 clause 2 is the normative table. In summary, a run supplies:
**E1** the complete input as the solver read it, plus the solver's echo of it
· **E2** solver identity, including the precision of each output stream ·
**E3** a termination record per phase and restart segment, plus diagnostics
as counts by class · **E4** the time-integration record (explicit: time step,
controlling part, added mass; implicit: iterations, norms, cut-backs,
unconverged increments) · **E5** an energy ledger with every term separate
and its balance identity declared · **E6** per-part and per-interface energy
and mass over time · **E7** applied-load resultants and reactions over time ·
**E8** fields at the constitutive points with their conventions declared,
including every argument of the material's admissibility functions · **E9**
ledger and reactions sampled on a clock that divides the field interval and
resolves the fastest physics · **E10** declared units plus three SI anchors
of independent dimension.

Points that matter in practice:

- **Stated from the mechanics, provisional until a second solver tests it.**
  Every mechanism that does work on, or removes energy or mass from, the
  discretised system has its own ledger term. The first revision's list
  tracked one solver's global-statistics columns and would have excluded
  solvers that keep no ledger and section-level structural solvers; hence
  the declared balance identity, the derived-ledger route, and the
  structural-element clause of E8.
- **Applicability is per run.** A ledger term is required only where the
  input-derived traits make it applicable. A quantity is assessable if and
  only if every required item is present — there is no partial-ledger
  energy check.
- **Two workarounds become declarations.** What each stored state variable
  *means* (E8: a record per material and variable — canonical name from a
  closed vocabulary, definition variant, unit, admissible range, monotonicity,
  initial value) and *units anchors* (E10: three, because one anchor leaves
  two of the three scale factors unchecked; the units error recorded in
  ADR-0030 was invisible to every self-consistent check). Both are declared
  once per dataset. They have no home yet; where they live is decided by the
  ADR that amends ADR-0027.
- **Per-solver realisation lives with the adapter** under `data_generation/`,
  not in the package. For LS-DYNA it is a standard block every generated
  input carries. Keyword names that appear in the shipped input decks
  (`*DATABASE_GLSTAT`, `*DATABASE_MATSUM`, `*CONTROL_ENERGY`,
  `*CONTROL_TIMESTEP`, `*CONTROL_TERMINATION`) are a starting point only; the
  options and values that satisfy E3–E7 and E9, the balance identity, and
  whether text or binary output results, are **unverified** and are checked
  against the keyword manual before the block is written down.
- **Not an admission rule.** Repository data generation is built to meet the
  requirement; a contribution is measured against it; whether an unmet item
  blocks admission is ADR-0065 follow-up 2's decision.

## What is shared by what

| Category | Shared by all | Per solver (`core/io`) | Per formulation (traits) | Per material class | Per benchmark |
|---|---|---|---|---|---|
| Numerical health | verdict logic, criteria | reading the termination record, diagnostics, time-integration record, identity | explicit vs implicit; zero-energy modes only for under-integrated parts; particle health only for particle parts | — | — |
| Conservation | evaluating a declared balance identity; closures; active mass | mapping the solver's terms onto the identity; explaining each absent term | which terms are required (contact, rigid surfaces, damping, erosion, mass scaling) | equation-of-state closure | — |
| Constitutive | kernels on plain arrays | keyword → class (ADR-0012's `canonical_model`, already stored with each case) | plane strain vs axisymmetric vs 3D; constitutive-point vs reduced data | admissibility functions, meaning and bounds of each state variable, which checks are assessable | the one yield table (`BenchmarkSpec.hardening_curve`) |
| Units and dimensions | anchor comparison, plausibility ranges, dimensionless groups | the input's unit token; any unit declaration the input carries | which constants and magnitudes exist to screen | which constants anchor a class (density, modulus, yield stress) | the declared anchors; "synthetic by design" |
| Data integrity | finite values, time axis, declared-vs-derived cross-checks, input lint, sampling clock | source-units token | alive mask under erosion | table matches input, is monotone, covers the strain range | declared fields, traits, state-variable meanings, units anchors |

The constitutive kernels are the part reusable on model predictions; nothing
else in the table is.

Traits are **per part**, derived from the solver input's part → material map,
never from a case file's element keys. An element block with no part in the
input is reported by an integrity row (`elements_without_input_part`) and
left out of per-part checks — a finding, not a presumption that it is benign.
A dataset with no solver input does not meet E1: its traits are unknown,
trait-gated quantities are `not_assessable / source_missing`, and its card
declarations are rendered as unverified claims.

## Repository layout

```
src/structbench/
  core/evidence.py          Absence, AbsenceReason, EvidenceItem, PartTraits, InputFacts, RunEvidence, DeclaredFacts (SI)
  core/io/lsdyna_run.py     input text → InputFacts; message / global-statistics TEXT → RunEvidence
  verification/
    kernels.py              array-level measures returning small mergeable records
    materials.py            canonical_model → admissibility functions, state-variable meaning and bounds
    quantities.py           catalogue rows: name, category, requires, trait gate, meaning, status
    measure.py              measure_case(case | None, input_facts, run_evidence, declared) -> CaseMeasurements
    criteria.py             Criterion records, the platform standard, judge()
    report.py               the JSON record
    markdown.py             the reader-facing report, generated from the record
  cli/datacheck.py          python -m structbench.cli.datacheck measure | judge
data_generation/lsdyna/     the standard input block (README) + per-dataset evidence glue
tests/{core,verification,cli}/
docs/datachecks/<benchmark>.{json,md}
```

Import edges: `verification → {core, datasets}`; `benchmarks`, `eval`, and
`cli` may import `verification`; `models` and `viz` may not. `datasets`
exports `n_valid_frames`. `verification` does not import `benchmarks`: the
CLI fills `DeclaredFacts` from the spec and card and passes it in.

Sketch of the central types (signatures are pinned in the implementation
plan):

```python
class Verdict(str, Enum): PASS; FAIL; REVIEW; NOT_APPLICABLE; NOT_ASSESSABLE
class Location(str, Enum): CASE; INPUT; RUN; DECLARED
class CriterionKind(str, Enum): REQUIREMENT; INSTRUMENT; INDICATOR
class Status(str, Enum): SPECIFIED; IMPLEMENTED

@dataclass(frozen=True)
class Absence:
    reason: AbsenceReason
    missing: frozenset[EvidenceItem]      # subset of the quantity's requires

@dataclass(frozen=True)
class Quantity:                           # a catalogue row — data, not a code path
    name: str; category: Category; unit: str
    requires: frozenset[EvidenceItem]     # E1..E10
    gate: TraitGate                       # when the quantity exists at all
    meaning: str                          # what a violation means
    status: Status
    definition_version: int | None        # None while SPECIFIED

@dataclass(frozen=True)
class Measurement:                        # threshold-free
    quantity: str; value: float | None; unit: str
    locations: frozenset[Location]; n_samples: int | None
    absence: Absence | None               # invariant: (value is None) == (absence is not None)
    detail: Mapping[str, float | int | str]   # strings restricted to enums / fixed patterns

@dataclass(frozen=True)
class Scope:                              # every token must hold; empty = everywhere
    run_traits: frozenset[RunTrait]       # derived from input and ledger: explicit, implicit,
                                          # lagrangian_mesh, particle_conservative,
                                          # particle_nonconservative, advecting_mesh, mass_scaled,
                                          # frictionless_contact, initial_energy_driven,
                                          # externally_driven, dim2, dim3
    declared_intent: frozenset[Intent]    # only what cannot be derived: quasi_static

@dataclass(frozen=True)
class Criterion:                          # platform standard
    quantity: str; statistic: str         # must equal the catalogue row's definition token
    lo: float | None; hi: float | None; strict: bool   # strict as the source states it ("less than" vs "not more than")
    kind: CriterionKind                   # REQUIREMENT/INSTRUMENT -> pass|fail; INDICATOR -> pass|review
    scope: Scope
    source: str                           # claim id in the source dossier; "" only for definitional bounds
    rationale: str; provisional: bool     # rationale non-blank

@dataclass(frozen=True)
class Disposition:                        # deferred — see "Deliberately not built yet"
    quantity: str; case_id: str | None    # None = the whole dataset
    decision: Literal["accepted", "rejected"]
    answers_definition_version: int; answers_level: str   # rendered stale when either changes
    rationale: str; author: str; date: str

def judge(measurements: DatasetMeasurements,       # carries scope_facts: run traits per part, declared intent
          criteria: Sequence[Criterion]) -> DatasetReport: ...
          # a run-global quantity's scope is the union over parts, and a level applies only if it
          # covers all of them; scopes of one (quantity, statistic) are pairwise disjoint (data-free
          # test), so at most one level applies; none -> no_ratified_criterion

def render(report: DatasetReport, dispositions: Sequence[Disposition] = ()) -> str: ...
```

`DeclaredFacts` carries the unit label, the three anchors, per-material
state-variable meanings, an optional material family, and quasi-static
intent. All of it is passed in by the caller.

Every catalogue row carries a one-line **meaning of a violation**, rendered
in the report. PASS on numerical-health and conservation rows is a necessary
solution-verification indicator; the report says so, and says it is not
evidence of accuracy.

## The catalogue

Status: **S1** implemented in stage 1, **S2** in stage 2, **gate** = row and
gates only (measure built when a run first supplies the evidence). Judged
as: **req** requirement (definitional bound), **tol** instrument tolerance,
**ind** indicator with reference levels (`pass` / `review`), **—** measured
and published with no criterion yet. E10a is the unit label, E10b the
anchors.

| Quantity | Requires | Exists when | Judged as | Status |
|---|---|---|---|---|
| *Numerical health* | | | | |
| `terminated_normally` | E3 | always | req | S2 |
| `reached_end_time` | E1, E8 | always | tol | S1 |
| `solver_error_count` (errors, inversions, non-finite kinematics) | E3 | always | req | S2 |
| `solver_warning_count` (by class) | E3 | always | ind (no level yet) | S2 |
| `solver_identity_complete` | E2 | always | req | S2 |
| `timestep_min_ratio` | E4 | explicit integration | ind (no level yet) | S2 |
| `timestep_vs_stability_estimate` (first step; a departure means the step is governed by something other than size and wave speed) | E1, E4, E8 | explicit integration | — | gate |
| `added_mass_fraction` (whole model, maximum over the run) · `added_mass_top_part_fraction` (the one part with the most added mass, last sample) · `added_mass_moving_fraction` (parts given an initial velocity, last sample) | E4 | explicit, mass scaling enabled | ind | gate |
| `implicit_convergence` | E4 | implicit integration | req | gate |
| `zero_energy_mode_final_over_initial_total` · `zero_energy_mode_final_over_internal_final` · `zero_energy_mode_peak_over_internal_peak` (whole model) | E5 | under-integrated elements | ind | gate |
| `zero_energy_mode_top_part_final_over_internal_final` (the one part with the most zero-energy-mode energy) | E5, E6 | under-integrated elements | ind | gate |
| `particle_deactivated_count` | E8 | particle part, erosion off | req | S1 |
| `particle_neighbors_min` · `particle_neighbors_growth` | E8 | particle part | ind, level scoped by dimension and kernel support (no level yet); a req floor of d + 1 for corrected-kernel and moving-least-squares formulations | S1 |
| `smoothing_length_within_input_bounds` (ingestion mapping) | E1, E8 | particle part | tol | S1 |
| `smoothing_length_at_bound_fraction` (upper and lower separately) | E1, E8 | particle part, variable smoothing length | ind (no level yet) | S1 |
| `rigid_surface_penetration_max` | E1, E4, E8 | rigid surface defined | tol where the surface is a kinematic constraint (bound: normal velocity × solver step); ind, normalised by local spacing, where it is a penalty | S1 |
| `prescribed_motion_realised` | E1, E7, E8 | prescribed motion defined | tol | gate |
| *Conservation* | | | | |
| `energy_gain_max` · `energy_loss_max` (over-the-run extremes of r) | E5 | always | ind | S2 |
| `energy_residual_final` (final value of r) | E5 | always | ind (no level; context) | S2 |
| `total_energy_change_final` = [E_tot(t_end) − E_tot(t0)] / E_tot(t0), signed | E5 | energy present at the first sample | ind | S2 |
| `quasi_static_kinetic_ratio` | E5 | declared quasi-static intent | ind | gate |
| `kinetic_energy_closure` (ledger term against the field sum at shared sample times, found in the files — decision 12; particle parts only so far; normalised by the peak ledger value) | E5, E8, E9 | always | — (a tolerance must name the half-step velocity stagger, not yet confirmed) | S2 |
| `internal_energy_closure` | E5, E8, E9 | the field output carries internal energy per constitutive point | tol | S2 |
| `contact_energy_ratio` · `contact_energy_negative_ratio` (per interface, over peak internal energy) | E5, E6 | contact defined | ind · ind (level scoped to frictionless contact) | gate |
| `external_work_closure` | E5, E7 | loads defined | tol | gate |
| `momentum_impulse_balance` | E7, E8 | always | tol | gate |
| `active_mass_drift` | E8 | mass scaling off and deletion off | tol | S1 |
| `mass_closure` (active + deleted − added = initial) | E4, E6, E8 | mass scaling or deletion enabled | tol | gate |
| `eos_closure` | E1, E8 | equation-of-state material | tol | gate |
| *Constitutive* | | | | |
| `yield_ratio_max` (+ detail) | E1, E8, declared | class with an assessable yield law | — (mechanism unconfirmed) | S1 |
| `yield_saturation_min` (largest yield ratio among samples whose plastic strain is increasing; a coarse scale screen, also a units row) | E1, E8, declared | same; constitutive-point data; at least one point yielding across two consecutive stored intervals | req, coarse scale bound | S1 |
| `state_variable_decrease_max` · `state_variable_min` | E8 | class with a monotone, bounded variable | req | S1 |
| `plane_strain_ezz_max` | E1, E8 | plane-strain trait | tol | S1 |
| `out_of_plane_shear_max` | E8 | two-dimensional | tol | S1 |
| `pressure_trace_residual` | E8 | pressure stored as independent state | — | S1 |
| *Units and dimensions* | | | | |
| `units_anchors_consistent` | E1, E10a, E10b | always | req (equality to the anchor's declared digits) | S1 — `not_assessable / no_declaration_home` until benchmarks can declare; the test-bed hand-run is given Taylor's three anchors |
| `input_density_plausible` (condensed-matter range; the only family-free screen that sees a mass-unit error) | E1, E10a | always | ind | S1 |
| `input_constants_plausible` (Young's modulus; discriminating only with a declared material family) | E1, E10a | a material states a Young's modulus | ind | S1 |
| `input_strength_plausible` (the most extreme tabulated yield stress; gross stress-unit errors only — decision 11) | E1, E10a | a material states a strength | ind | S1 |
| `input_dimensionless_groups_plausible` (yield stress over modulus, elastic constants against one another; needs no unit) | E1 | always | ind | S1 |
| `response_magnitudes_plausible` (velocity, strain, stress; no net mass dimension except stress) | E8, E10a | always | ind | S1 |
| `input_unit_declaration_consistent` | E1, E10a | the input carries its own unit declaration | req | gate |
| `density_slot_matches_input` (ingestion mapping — *not* a units check) | E1, E8 | always | tol | S1 |
| *Data integrity* | | | | |
| `nonfinite_count` · `time_axis_monotone` | E8 | always | req | S1 |
| `terminal_artifact_frames` (frames written off the sampling interval; a fact — decision 10) | E8 | always | — | S1 |
| `elements_without_input_part` | E1, E8 | always | req | S1 |
| `fields_match_declaration` | E8, declared | always | req | S1 |
| `declared_traits_match_input` | E1, declared | always | req | S1 |
| `yield_table_matches_input` · `yield_table_monotone` · `yield_table_covers_range` | E1, E8, declared | tabulated yield law | req | S1 |
| `sampling_clock_consistent` | E5, E8, E9 | always | req | S2 |
| `stored_globals_match_ledger` | E5, E8, E9 | the case's globals come from a stream independent of the ledger | tol | S2 |

Every stage-1 row's requirement lies within {E1, E4 for one bound, E8, E10,
declared}; two rows inside that set stay `gate` because no test-bed run
exercises them (`eos_closure`: no equation-of-state class is supported yet;
`input_unit_declaration_consistent`: the test-bed input carries no unit
declaration). A data-free test asserts that the union of all rows'
requirements is E1–E10, and that no two levels share a quantity, statistic,
and scope.

Plausibility ranges are reference levels like any other: each cites a
materials reference and none is set from StructBench data; none has been
surveyed yet. A dataset whose anchors state `kind of source: synthetic` has
no outside to be checked against — the report renders that beside the
plausibility and anchor rows and says the anchors verify arithmetic only. It
is a declared fact, not a disposition.

## Reference levels

Fixed from the source dossier under ADR clause 7's attachment rule: a
source's number becomes a level only on the statistic, normalisation, and
evaluation time the source states, within the source's own domain. Each was
proposed by one pass and attacked by a second; the record is in the
dossier's section V. None was chosen or adjusted in light of a StructBench
measurement — and because one test-bed value (a legacy particle run whose
total energy rises by several percent) was known before these were fixed,
every energy level's rationale says so.

| Quantity | Bound | Scope (run traits) | Source | Notes |
|---|---|---|---|---|
| `energy_gain_max`, `energy_loss_max` | ≤ 0.01 | {explicit, lagrangian_mesh} · {explicit, particle_conservative} | B-BLM-1, B-BLM-2 | Equal under the reading clause 7 fixes (internal work = every non-kinetic term; energy at the first sample is input). "Generally on the order of 10⁻²" is rendered as the one number the source prints. A stability check, not an accuracy limit; the source checks every step and balances large models on subdomains, so a sampled whole-model `pass` is weaker than the source's check. **Provisional**: the book was read as snippets only. |
| `total_energy_change_final` | abs ≤ 0.10 | {explicit, lagrangian_mesh, initial_energy_driven} | W-W179-02, W-W179-12 | "must not vary more than 10 percent from the beginning of the run to the end of the run"; denominator is the initial total energy; the source has no external-work term. Community practice for roadside crash with finite elements. |
| `zero_energy_mode_final_over_initial_total` | < 0.05 | {explicit, lagrangian_mesh, initial_energy_driven} | W-W179-03 | No provenance is stated for this figure in the pages read. |
| `zero_energy_mode_final_over_internal_final` | < 0.10 | {explicit, lagrangian_mesh} | W-W179-04 | — |
| `zero_energy_mode_top_part_final_over_internal_final` | < 0.10 | {explicit, lagrangian_mesh} | W-W179-05 | One part only — the part with the largest zero-energy-mode energy — not the worst ratio over all parts. |
| `zero_energy_mode_peak_over_internal_peak` | < 0.10 | {explicit, lagrangian_mesh} | W-ENCAP-01, W-ENCAP-02 | Occupant virtual-testing protocols; "max. internal energy" read as the full setup's. Extension beyond that domain is the platform's. |
| `added_mass_fraction` | < 0.05 | {explicit, mass_scaled} | W-ENCAP-01, W-ENCAP-02 | The source says "max."; it also prints 2.5 % for separately built models, not adopted. W-W179-06 and B-RAD-3 corroborate the number without stating the evaluation time or the denominator. |
| `added_mass_top_part_fraction` | < 0.10 | {explicit, mass_scaled} | W-W179-07 | The one part with the most added mass, at the last sample. |
| `added_mass_moving_fraction` | < 0.05 | {explicit, mass_scaled, initial_energy_driven} | W-W179-08 | "Moving" is input-derived: a part with a node given a non-zero initial velocity. |

In every zero-energy-mode ratio, "internal energy" has the sources' meaning:
the work of element internal forces as their solver books it — including
stiffness-damping and artificial-viscosity dissipation, excluding
zero-energy-mode, contact, rigid-surface, and mass-damping terms.

**Context only — cited in rationales, attached to nothing.** The
roadside-safety report's worked-example wording of its per-part hourglass
row (W-W179-11), which differs from its normative table; only the table is
used. The solver
vendor's per-part rule of thumb "< 10 %" of peak internal energy (B-LSD-2):
the numerator's evaluation time is unstated. Contact energy "10 % of peak
internal energy might be considered acceptable" (B-LSD-3): whole model versus
per interface is not stated. "Generally with an error of less than 1 %"
(B-ABQ-1): no normalisation. Kinetic energy "typically 5 % to 10 %" of
internal energy "throughout most of the process" (B-ABQ-3): no determinate
evaluation time. "+1 % or +2 % is acceptable" (B-RAD-2): a different ledger
subset. Hourglass over total energy "on the order of 3 % or 5 %" (B-BLM-3):
an error estimate, not a limit.

**No level — measured and published, `no_ratified_criterion`.**
`energy_residual_final`; both contact-energy ratios; the quasi-static
kinetic ratio; time-step collapse; warning counts; neighbour counts;
smoothing-length saturation; penalty penetration; and every energy indicator
for a run outside the scopes above. No published number was found for any
of them.

## Plausibility ranges

From the dossier's section M (37 claims; 36 confirmed, one source typo).
Values are the sources' own; SI conversion is ours. These are reference
levels for the units category: an excursion is `review`.

**Family-free — needs only the unit label.**

| Constant | Range | Source | What it can see |
|---|---|---|---|
| Density | 16 kg/m³ (flexible polymer foam) to 22 590 kg/m³ (osmium); fully dense solids from about 890 kg/m³ | M-D6, M-D5, M-D10 (secondary; M-D8 gives 21 500 for a platinum alloy from a primary table) | a mass-unit error — the one screen that can, without a declared family |
| Young's modulus | 3 × 10⁵ Pa (flexible foam) to 10¹² Pa (diamond) | M-E4, M-D7 | length or time errors of several decades; not a factor of 1000 within a family |
| Poisson's ratio | −1 ≤ ν ≤ ½ for an isotropic solid — a **requirement**, not an indicator | M-P1, M-P2 | a mis-entered constant |
| Elastic-constant relations | G = E / 2(1 + ν), K = E / 3(1 − 2ν) — a **requirement**, to an instrument tolerance | M-P3 (relations (a) and (b); the source's printed relation (c) carries a typo and is not cited) | constants mixed across unit systems within one material |
| Strength | 10⁴ Pa (foam) to 6.8 × 10⁹ Pa (tungsten carbide, compressive) | M-S1, M-S2, M-S3 | gross errors only |
| Bar wave speed √(E/ρ) | 10² to 10⁴ m/s across engineering solids; absolute bound about 36 100 m/s | M-S6, M-S8 | length and time only — blind to mass |
| Response velocity | above 3 km/s is the hypervelocity regime, outside structural impact | M-S9 | length and time only |

**By declared material family** (density in kg/m³, modulus in GPa, yield or
compressive strength in MPa; Cambridge *Materials Data Book* unless noted):

| Family | Density | Young's modulus | Strength | Claims |
|---|---|---|---|---|
| Carbon and low-alloy steels | 7800–7900 | 200–217 | yield 250–395 (low carbon) | M-D1, M-E1, M-S1, M-C5 |
| Aluminium alloys | 2500–2900 | 68–82 | yield 30–500 | M-D2, M-E1, M-S1 |
| Copper alloys | 8930–8940 | 112–148 | yield 30–500 | M-D2, M-E1, M-S1 |
| Metals and alloys, all | 1740–11 400 | 12.5–220 | yield 8–1245 | M-D2, M-E1, M-S1 |
| Concrete, normal weight | 2200–2600 | 25–38 (data book); 27–44 (EN 1992-1-1 Table 3.1) | compressive 12–90 characteristic; tensile 1.6–5.0 mean | M-D3, M-E2, M-C1, M-S5 |
| Concrete, lightweight aggregate | 800–2000 by class; ≤ 2200 by definition | E_cm × (ρ/2200)² | — | M-C3 |
| Stone · brick | 2500–3000 · 1900–2100 | 6.9–21 · 10–50 | compressive 34–248 · 50–140 | M-D3, M-E2, M-S2 |
| Glasses | 2170–2800 | 61–110 | compressive 264–2129 | M-D3, M-E2, M-S2 |
| Technical ceramics | 2300–15 900 | 140–720 | compressive 524–6833 | M-D3, M-E2, M-S2 |
| Thermoplastics · thermosets | 890–2200 · 1040–1400 | 0.2–5 · 2.07–4.83 | yield 8.3–95 · 27.6–71.7 | M-D5, M-E4, M-S3 |
| Elastomers | 900–1800 | 0.0007–0.04 | 2–51 | M-D5, M-E4, M-S3 |
| Polymer foams | 16–470 | 0.0003–0.48 | 0.01–12 | M-D6, M-E4, M-S3 |
| Fibre composites (CFRP · GFRP) | 1500–1600 · 1750–1970 | 69–150 · 15–28 | 550–1050 · 110–192 | M-D4, M-E3, M-S3 |
| Wood (longitudinal) | 600–800 | 6–20 | 30–70 | M-D4, M-E3, M-S3 |

**Still unsourced.** A numeric range of yield strain (strength over modulus)
by family — only the chart's contour labels, 10⁻⁴ to 10⁻¹, were read
(M-S4); a family-by-family ratio can be derived from the two tables above,
but that is a derivation, not a quotation. A primary table of wave speeds by
family. Rock by type. Metal-foam strength. Until then
`input_dimensionless_groups_plausible` judges the elastic-constant
relations and Poisson bound only, and reports yield strain with no level.

## Physics notes

1. The value 1.001 in ADR-0064 bounds a *model head's* float32 saturation
   error (~2e-7). The excess measured on reference data, 3.5e-4, comes from a
   single case and is far above float32 round-off; `tools/state_probe`
   attributes it to round-off, a reviewer to the solver's discretised plastic
   return across a knot of the hardening table. Neither is confirmed. So
   `yield_ratio_max` ships **measured, with no criterion**; the test bed is
   used to test the hypothesis (does the excess track knot distance and
   impact velocity?), and a tolerance is derived from the mechanism once one
   is confirmed. The measurement reports the plastic strain at the maximum,
   the frame, the distance to the nearest knot, and the violation fraction.
2. The yield check needs a **lower-bound companion**: among samples whose
   plastic strain is increasing, the largest yield ratio must not be far
   below one. Without it, stress stored a thousand times too small passes
   silently — the ADR-0030 bug class. It is a *scale screen*, not a
   return-mapping tolerance: a point that yielded between two stored frames
   need not sit on the surface at either, so a fine bound is not
   definitional. The bound is set far below one and far above the reciprocal
   of the smallest unit-error factor, and its rationale says so. It is valid
   only on constitutive-point data: von Mises of an element average is below
   the average von Mises.
3. **Units cannot be checked from inside the data.** A consistently
   mislabelled unit system survives every identity — energy closure, wave
   speed, stored density against input density — so the ADR-0030 error would
   have passed every self-consistent check. Only things outside the label
   can fail. *Anchors*: three, of independent dimension (density, a
   stress-dimension constant, a length — their exponent matrix over mass,
   length, time is non-singular, so every single-factor mislabel fails at
   least one); each states the SI value of a named input quantity to
   declared digits, which makes agreement an equality, not a judgement about
   handbook accuracy. *Plausibility*: without a declared material family
   only density sees a mass-unit error — a family-free modulus range spans
   several decades, and steel entered a thousand times too soft reads as a
   plausible polymer. *Dimensionless groups* inside the input need no unit
   and catch units mixed within a material definition. The solver's time
   step against a stability estimate is **not** a units check — solver and
   estimate use the same input numbers, and wave speed is blind to the mass
   unit — so it sits in numerical health, where a departure means the step
   is governed by something other than size and wave speed.
4. **Energy is one general indicator, defined on the ledger and nowhere
   else.** With `E_tot` the sum of every term of the declared balance
   identity (each once, with its sign) and `t0` the first ledger sample:
   `R(t) = [E_tot(t) − E_tot(t0)] − [W_ext(t) − W_ext(t0)]` and
   `r(t) = R(t) / max(|E_tot(t0) + W_ext(t) − W_ext(t0)|, E_kin(t), E_tot(t) − E_kin(t))`.
   This is Belytschko's form (dossier B-BLM-1) with its internal work read as
   the work of *all* internal forces and with energy present at `t0`
   counted as input — without that term an initial-velocity impact would
   read about one from the first sample. `r > 0` is energy created. For a
   run with no external work, `r` equals the solver-reported energy ratio
   minus one (dossier L-C12) only while the first argument of the
   denominator is the largest; once the kinetic or the non-kinetic bucket
   exceeds the initial total — the normal end state of an impact that has
   gained energy and come to rest — the denominator is larger and `r` is
   smaller. That is why the roadside-safety level has its own quantity,
   `total_energy_change_final`, which *is* the ratio minus one at the last
   sample. The levels, and the statistic each attaches to, are in
   *Reference levels* above. Closures compare a ledger term with the field
   sum at shared sample times and need both; the kinetic closure's tolerance
   must name the half-step velocity stagger of central-difference
   integration.
5. Termination and integrity checks read the **raw** time axis and all
   stored frames. Only checks that need a uniform interval apply
   `n_valid_frames` (ADR-0028); on the artifact-dropped axis a healthy run
   would fail `reached_end_time`.
6. Plane strain is an **input trait**, tested by zero out-of-plane normal
   strain; zero out-of-plane shear only proves "two-dimensional".
7. Particle-method numerical health — deactivated particles, neighbour
   starvation and clumping, smoothing length against the input's own bounds,
   rigid-surface penetration — is required of any particle run by E8.
8. Input lint is evidence: a non-monotone hardening knot is an input defect
   and appears as a recorded finding.
9. Ledger extremes are *sampled* extremes; E9 exists so that the sampling
   means something, and the report states the sample count.

## Stages

Taylor is the **test bed for the instrument**, chosen because its files are
at hand and a prior probe recorded numbers to reproduce. Its gaps against
E1–E10 are reported, not designed around.

**Stage 0 — the requirement, written down.** E1–E10 land with ADR-0066. The
LS-DYNA realisation (the standard input block) is verified against the
keyword manual and documented under `data_generation/lsdyna/` — written as
`data_generation/lsdyna/STANDARD_INPUT_BLOCK.md`, a draft until the
conformance run has exercised it; its open points are listed there.

**Stage 0b — one conformance run** *(maintainer's decision; needs the
licensed solver)*. A single test-bed case re-run with the standard input
block: not a dataset, unpublished, no benchmark version, kept under
gitignored `runs/`. It proves the realisation yields E1–E10 in a form the
text extractor reads, and it is the first run that supplies E4–E7 in full —
so it, not the legacy archive, triggers those measures. Without it, the
requirement is normative for contributors while never having been produced
or read once.

**Stage 1 — evidence in the case file and its stored input.** Order: result
types and invariants → catalogue rows and gates (all rows) → kernels →
faithful input reader, `InputFacts`, `PartTraits` → measures (including the
units category: anchors against passed-in declarations, and the two
plausibility screens) → criteria with scopes and `judge` → report and CLI. Synthetic fixtures mirror real traps: a case with
an element block that has no input part, and an input whose material card
has a blank second row.
*Hand-run acceptance*: on `T-20-60-100` reproduce the probe's recorded
numbers (maximum yield ratio 1.000353; zero monotonicity violations in
720,000 transitions; 151 valid frames; 4800 particles); take one timing on a
hydrated and a dehydrated file; then run the 33 benchmark cases and the
held-aside convergence case through the per-case quantities.

**Stage 2 — the run-evidence record.** Message-file and global-statistics
parsers (fixtures use invented numbers on the real layout, name the solver
version, fail closed on unknown labels); the energy ledger evaluated through
the declared identity; the S2 rows of the catalogue; glue writing the
run-evidence record; a privacy test over every rendered artefact.
*Hand-run acceptance*: first record which required ledger terms the
inspected run carries — if the applicability rule yields `not_assessable`,
that row is the acceptance fact. The hand-read facts to reproduce on Taylor
`20100/100` (case `T-20-100-100`): normal termination, 3918 steps, time step
7.46e-5 – 7.78e-5 ms, total-to-initial energy ratio peaking at 1.089 and
ending at 1.070, rigid-surface energy 311 against initial kinetic energy
4.45e4 (input units). From the last ledger sample (total 4.76245e4,
kinetic 76.8, no external work) the measures must return
`total_energy_change_final` = 0.0702 and `energy_residual_final` = 0.0657 —
the second is smaller because the non-kinetic bucket, 4.7548e4, exceeds the
initial total and becomes the denominator. `energy_gain_max` is at most
0.089 and is computed from the ledger. This run is a particle formulation
that is not pairwise conservative, so no level covers it: the rows are
published with values and out-of-scope levels, and no verdict.

**Stage 3 — confirm the levels and publish.** *(As carried out: decision
13 — published with no level ratified.)* The maintainer confirms the
reference levels and plausibility ranges tabulated above, spot-checking the
two sources read only as images or snippets; `docs/datachecks/<benchmark>.json` and its
generated `.md` land with the drift test. The energy levels' rationale
records that one test-bed value was known before they were fixed. Rows that
read `review` are published as such; a disposition is the dataset owner's
to add, or not.

## Deliberately not built yet

| Item | Trigger |
|---|---|
| Measures for `gate` rows (added mass, implicit convergence, zero-energy-mode ratio, contact-energy sign, external-work and momentum balances, prescribed-motion realisation, EOS closure) | the first run that supplies the evidence — the conformance run for most |
| A dataset generated under E1–E10 (new, or a regenerated legacy sweep as a new dataset version) | the maintainer's decision |
| Revision of E1–E10 and its term names | the second solver's adapter |
| A home in the benchmark package for the declared facts — unit label and anchors, state-variable meanings, material family, quasi-static intent (the checks themselves are built against passed-in `DeclaredFacts`) | the ADR that amends ADR-0027 (ADR-0065 follow-up 2) |
| Refinement, noise-floor, and cross-source quantities | ADR-0065 follow-up 2 defines those standards; the first dataset that meets them |
| Selective or streamed reader in `core/io`; chunk-merged measures | cases of hundreds of MB |
| Material classes beyond tabulated J2; the material-class enum ADR | the next material |
| Disposition records (typed, dated literals; the ADR-0033 registry pattern is the candidate home) | the first published row that reads `review` |
| Reference levels no source yet covers (energy for non-conservative particle and advecting-mesh formulations, contact energy, the quasi-static ratio, time-step collapse, warning counts, neighbour counts, smoothing-length saturation); numeric yield-strain ranges; wave speeds and rock by type | a verified source, or a dated note recording that none exists |
| Input-derived operative yield table | the first dataset with no `BenchmarkSpec` |
| Carrying the ledger inside the canonical file; ingesting the arrays the adapter discards; populating `Provenance` from E2 | a dated note on ADR-0016; the first dataset generated under E9 |
| Binary solver-output reading | the first run whose ledger exists only in binary form; flag-first — whether it needs an import outside the approved list is unverified |
| Compliance table, card fields, archive artefacts | ADR-0065 follow-up 2 |
| Kernels called from `eval/` | ADR-0065 follow-up 1 |
| Console script; benchmark-free CLI mode; `conftest.py` | the first unregistered run; the first shared fixture need |

## Open items for the maintainer

1. **Decide on the conformance run** (stage 0b); the other drafting calls
   were confirmed with ADR-0066.
2. **Spot-check two sources before the levels are relied on**: the
   roadside-safety report was read as page images, and the energy-balance
   text as snippets (derivation record: dossier section V; material claims:
   section M).
3. **Finalise the LS-DYNA realisation of E1–E10** from the dossier's draft
   table, against the release actually used (R13 and R15 were read; the
   legacy runs used R12). Two points need a run to settle: MPP executables
   write the statistics databases in binary only, with ASCII by post-run
   conversion, yet a legacy MPP folder holds both forms; and per-part added
   mass under MPP is unconfirmed.
4. **Whether to generate a dataset under the requirement**, so that one
   benchmark actually meets the standard. Not part of this work.
5. **Re-run policy**: with no CI, the published record is regenerated by
   hand on a dataset re-release or a `definition_version` bump.

## Verification

- Every stage: `ruff format --check . && ruff check . && mypy src && pytest`.
- Synthetic tests exercise every verdict branch per implemented quantity
  (including `review` for indicators and scope selection in `judge`), both gates for every row, every fail-closed input path, a
  byte-identical re-run, the privacy scan, the union-equals-E1–E10 check, and
  an import-boundary test that resolves relative imports.
- Real data is never a test dependency; environment-gated tests assert the
  acceptance facts above and skip when unset.
- `python tools/gen_benchmark_docs.py --check` stays green — nothing
  generated by the existing pipeline is touched.
