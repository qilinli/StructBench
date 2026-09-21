# Design: reference-data verification (`verification/`) and the first slice

**Date**: 2026-09-21
**Status**: Draft for maintainer review
**Scope**: new module `src/structbench/verification/`; two new `core` files
(`core/evidence.py`, `core/io/lsdyna_run.py`); `cli/datacheck.py`; per-dataset
glue under `data_generation/`; `docs/datachecks/`; `docs/ARCHITECTURE.md`;
README Roadmap. Governing decision: ADR-0066 (Proposed), which is normative
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
2. **Acceptance criteria are platform-level and independent of our data**:
   taken from a verified external source as the source states them, never
   adjusted in light of a StructBench measurement. Until ratified a row reads
   `not_assessable / no_ratified_criterion`.
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

### Calls made in drafting — for the maintainer to confirm

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

## How the design was reached

A first draft came from three independent module designs and three
adversarial reviews (governance, computational mechanics, buildability). The
maintainer reversed its evidence model (decision 6); a first revision still
copied the requirement from one solver's output list, which a second review
on the same three lenses caught. What follows is the result.

## The model

**Two verbs.** `measure`: evidence → threshold-free measurement, or a typed
absence. `judge`: measurements + criteria → verdicts; never touches data.

**Four verdicts, fixed ownership and order.**

| Order | Owner | Verdict |
|---|---|---|
| 1 | formulation traits | `not_applicable` — the quantity does not exist for this formulation |
| 2 | evidence absence | `not_assessable` + reason + the missing evidence items |
| 3 | measurement vs criterion | `pass` / `fail` |

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
    report.py               JSON and markdown
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
class Verdict(str, Enum): PASS; FAIL; NOT_APPLICABLE; NOT_ASSESSABLE
class Location(str, Enum): CASE; INPUT; RUN; DECLARED
class CriterionKind(str, Enum): ACCEPTANCE; INSTRUMENT
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
class Criterion:                          # platform standard
    quantity: str; lo: float | None; hi: float | None
    kind: CriterionKind; rationale: str; provisional: bool   # rationale non-blank

def judge(measurements: DatasetMeasurements,
          criteria: Mapping[str, Criterion]) -> DatasetReport: ...
```

Every catalogue row carries a one-line **meaning of a violation**, rendered
in the report. PASS on numerical-health and conservation rows is a necessary
solution-verification indicator; the report says so, and says it is not
evidence of accuracy.

## The catalogue

Status: **S1** implemented in stage 1, **S2** in stage 2, **gate** = row and
gates only (measure built when a run first supplies the evidence).

| Quantity | Requires | Exists when | Status |
|---|---|---|---|
| *Numerical health* | | | |
| `terminated_normally` | E3 | always | S2 |
| `reached_end_time` | E1, E8 | always | S1 |
| `solver_diagnostics` (counts by class) | E3 | always | S2 (errors, warnings; further classes as first seen) |
| `solver_identity_complete` | E2 | always | S2 |
| `timestep_min_ratio` | E4 | explicit integration | S2 |
| `added_mass_fraction` | E4 | explicit, mass scaling enabled | gate |
| `implicit_convergence` | E4 | implicit integration | gate |
| `zero_energy_mode_ratio` | E5, E6 | per part: under-integrated elements | gate |
| `particle_deactivated_count` | E8 | particle part | S1 |
| `particle_neighbors_min` · `particle_neighbors_growth` | E8 | particle part | S1 |
| `smoothing_length_range` | E1, E8 | particle part | S1 |
| `rigid_surface_penetration_max` | E1, E8 | rigid surface defined | S1 |
| `prescribed_motion_realised` | E1, E7, E8 | prescribed motion defined | gate |
| *Conservation* | | | |
| `energy_gain_max` · `energy_loss_max` · `energy_residual_final` | E5 | always | S2 |
| `quasi_static_kinetic_ratio` | E5 | task declared quasi-static | gate |
| `kinetic_energy_closure` · `internal_energy_closure` | E5, E8, E9 | always | S2 |
| `contact_energy_sign` | E6 | contact defined | gate |
| `external_work_closure` | E5, E7 | loads defined | gate |
| `momentum_impulse_balance` | E7, E8 | always | gate |
| `active_mass_drift` | E8 | always | S1 |
| `eos_closure` | E1, E8 | equation-of-state material | gate |
| *Constitutive* | | | |
| `yield_ratio_max` (+ detail) | E1, E8, declared | class with an assessable yield law | S1 — measured, no criterion |
| `yield_saturation_min` | E1, E8, declared | same; constitutive-point data | S1 |
| `state_variable_decrease_max` · `state_variable_min` | E8 | class with a monotone, bounded variable | S1 |
| `plane_strain_ezz_max` | E1, E8 | plane-strain trait | S1 |
| `out_of_plane_shear_max` | E8 | two-dimensional | S1 |
| `pressure_trace_residual` | E8 | pressure stored as independent state | S1 — measured, no criterion |
| *Data integrity* | | | |
| `nonfinite_count` · `time_axis_monotone` · `terminal_artifact_frames` | E8 | always | S1 |
| `elements_without_input_part` | E1, E8 | always | S1 |
| `fields_match_declaration` | E8, declared | always | S1 |
| `declared_traits_match_input` | E1, declared | always | S1 |
| `density_slot_matches_input` | E1, E8 | always | S1 |
| `yield_table_matches_input` · `yield_table_monotone` · `yield_table_covers_range` | E1, E8, declared | tabulated yield law | S1 |
| `sampling_clock_consistent` | E5, E8, E9 | always | S2 |
| `stored_globals_match_ledger` | E5, E8, E9 | the case's globals come from a stream independent of the ledger | S2 |
| `units_anchors_consistent` | E1, E10 | always | gate (no home for the declaration yet) |

Stage 1 is exactly the rows whose requirement lies within {E1, E8,
declared}. A data-free test asserts that the union of all rows'
requirements is E1–E10.

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
2. The yield check needs a **lower-bound companion**: if any sample has
   yielded, some sample must sit near the surface. Without it, stress stored
   a thousand times too small passes silently — the ADR-0030 bug class. It
   is valid per case, not per sample, and only on constitutive-point data:
   von Mises of an element average is below the average von Mises.
3. Comparing stored density with input density is **not a units check** —
   both pass through the same conversion factor — so it keeps the honest name
   `density_slot_matches_input`, and E10 asks for anchors that can fail.
4. **Energy is defined on the ledger and nowhere else**: signed excursions
   (largest gain, largest loss, final), normalised by the largest of kinetic,
   internal, and external-work magnitudes so that driven and from-rest
   systems are defined, evaluated through the declared balance identity.
   Closures compare a ledger term with the field sum at shared sample times
   and need both; kinetic closure includes rotary and rigid-body terms where
   the formulation has them, and its tolerance must name the half-step
   velocity stagger of central-difference integration.
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
keyword manual and documented under `data_generation/lsdyna/`.

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
faithful input reader, `InputFacts`, `PartTraits` → measures → criteria and
`judge` → report and CLI. Synthetic fixtures mirror real traps: a case with
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
4.45e4 (input units).

**Stage 3 — ratify and publish.** The maintainer verifies external sources
and commits acceptance criteria verbatim; `docs/datachecks/<benchmark>.json`
and its generated `.md` land with the drift test. The energy criterion's
rationale records that one test-bed value was known before ratification.

## Deliberately not built yet

| Item | Trigger |
|---|---|
| Measures for `gate` rows (added mass, implicit convergence, zero-energy-mode ratio, contact-energy sign, external-work and momentum balances, prescribed-motion realisation, EOS closure) | the first run that supplies the evidence — the conformance run for most |
| A dataset generated under E1–E10 (new, or a regenerated legacy sweep as a new dataset version) | the maintainer's decision |
| Revision of E1–E10 and its term names | the second solver's adapter |
| A home for E8's state-variable declarations and E10's anchors; `units_anchors_consistent` | the ADR that amends ADR-0027 (ADR-0065 follow-up 2) |
| Refinement, noise-floor, and cross-source quantities | ADR-0065 follow-up 2 defines those standards; the first dataset that meets them |
| Selective or streamed reader in `core/io`; chunk-merged measures | cases of hundreds of MB |
| Material classes beyond tabulated J2; the material-class enum ADR | the next material |
| Waiver records | the first ratified criterion a shipped benchmark fails |
| Input-derived operative yield table | the first dataset with no `BenchmarkSpec` |
| Carrying the ledger inside the canonical file; ingesting the arrays the adapter discards; populating `Provenance` from E2 | a dated note on ADR-0016; the first dataset generated under E9 |
| Binary solver-output reading | the first run whose ledger exists only in binary form; flag-first — whether it needs an import outside the approved list is unverified |
| Compliance table, card fields, archive artefacts | ADR-0065 follow-up 2 |
| Kernels called from `eval/` | ADR-0065 follow-up 1 |
| Console script; benchmark-free CLI mode; `conftest.py` | the first unregistered run; the first shared fixture need |

## Open items for the maintainer

1. **Confirm the three drafting calls** at the top of this document.
2. **External sources for the acceptance criteria.** Candidates were named
   from memory during review and are *unverified*: a roadside-safety
   solution-verification table and a textbook energy-balance criterion for
   explicit integration. Nothing numeric enters code or an ADR until
   verified, and the criterion is then taken as the source states it.
3. **The LS-DYNA realisation of E1–E10**, against the keyword manual.
4. **Whether to generate a dataset under the requirement**, so that one
   benchmark actually meets the standard. Not part of this work.
5. **Re-run policy**: with no CI, the published record is regenerated by
   hand on a dataset re-release or a `definition_version` bump.

## Verification

- Every stage: `ruff format --check . && ruff check . && mypy src && pytest`.
- Synthetic tests exercise all four verdict branches per implemented
  quantity, both gates for every row, every fail-closed input path, a
  byte-identical re-run, the privacy scan, the union-equals-E1–E10 check, and
  an import-boundary test that resolves relative imports.
- Real data is never a test dependency; environment-gated tests assert the
  acceptance facts above and skip when unset.
- `python tools/gen_benchmark_docs.py --check` stays green — nothing
  generated by the existing pipeline is touched.
