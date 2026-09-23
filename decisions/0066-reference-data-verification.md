# 0066 — Reference-data verification: the `verification/` module

**Status**: Accepted (maintainer, in-session 2026-09-21)
**Type**: Durable
**Date**: 2026-09-21

## Context

ADR-0065 gives StructBench "the data-standard side of the data layer" and
records that none of the four shipped benchmarks would be admitted under the
data standards it names, a gap to be made visible rather than hidden. Today
nothing in the repository can measure whether a reference simulation run is
trustworthy:

- `core/validation.py` checks structure and shapes only — no NaN scan, no
  physics, no non-fatal finding channel.
- The constitutive checks on reference data (plane strain, von Mises
  identity, yield surface, plastic-strain monotonicity) exist only as
  `assert`s in `tools/state_probe/`, which is explicitly exploratory, and
  their outcomes are prose in its README. The model-side admissibility
  diagnostics ADR-0064 names (D2/D3) live in untracked `scratch/`.
- Provenance is `("LS-DYNA", "unknown", <d3plot mtime>)` on every ingested
  case. No reader exists for any solver output other than d3plot.

**The design starts from what verification requires, not from what the
existing archives happen to contain.** The three LS-DYNA archives are small
sweeps produced years before this repository existed. Which solver outputs
they requested, and which files were kept, is an accident of how they were
run; it says nothing about what the platform should ask for or what a run
can supply. So this ADR works from the mechanics outward: each quantity
states the evidence a run must supply for it to be assessable, and the union
of those statements is what future data generation is asked to supply and
what a contribution is measured against. Applied to a legacy archive, the
requirement produces honest `not_assessable` rows that name what is missing,
and nothing else. The legacy archives serve one purpose in the design: a
test bed for the instrument.

That a run has to be measured at all is not hypothetical. On one test-bed
run whose global energy record was read by hand, the solver reports normal
termination while its recorded total energy rises by several percent with no
external work. Whether that is a genuine imbalance or an accounting artefact
is not established — nothing in the repository can say, which is the gap
this ADR closes. The figures are in the design doc's test plan.

A first draft was reviewed in-session (2026-09-21) for repository
governance, computational mechanics, and buildability. The maintainer then
reversed its evidence model (see the first rejected alternative), and the
revision was reviewed again on the same three lenses. External sources for
criteria and for the LS-DYNA realisation were then read and independently
re-checked; the claims, quotes, and locators are in
`docs/plans/2026-09-21-reference-data-verification-sources.md`. This ADR
records the result.

Binding constraints: ADR-0004 (no solver vocabulary outside adapters),
ADR-0010 (a solver abstraction layer was rejected as premature — "needs at
least two concrete solvers to be designed correctly"), ADR-0016 §4/§6 (no
feature engineering at ingestion; glue never manipulates response data),
ADR-0027 (per-benchmark declarations are typed dataclasses; derivable numbers
are computed, not declared), ADR-0031 and ADR-0040 (the archive content list
and the public mirror built from it), ADR-0033 (results never bump a
benchmark version), ADR-0064 (one authoritative hardening table), the
dependency graph in `docs/ARCHITECTURE.md`, and the active corrections of
2026-07-03 (prefer the smallest artifact that encodes the behaviour) and
2026-09-21 (design for the future, not for the legacy runs).

## Decision

1. **A new top-level module `src/structbench/verification/`** holds the
   deterministic checks: array-level kernels, material-class semantics, the
   quantity catalogue, `measure`, the criteria with `judge`, and the report.
   It sits between `datasets/` and the modules that may use it:

   ```
   core ← datasets ← verification ← {eval, benchmarks} ← cli
   ```

   It imports `core` and `datasets` only; `models/` and `viz/` gain no edge
   to it. No code moves; `datasets` gains one export (`n_valid_frames`). The
   name follows ADR-0065 clause 4 ("verification is reported before accuracy
   is compared") and leaves *validation* free for the reality layer; it is
   unrelated to `core/validation.py`, which stays the schema-validity check.
   The kernels are written on plain arrays (e.g. `(von_mises,
   yield_stress)`), so `eval/` can later call the same functions on predicted
   fields with no new edge.

2. **The run-evidence requirement.** Every catalogue quantity declares the
   evidence it needs. The union is what a run has to supply for every
   applicable quantity to be assessable:

   | | A run supplies |
   |---|---|
   | E1 | the complete solver input as the solver read it — every included file, parameter values resolved or recorded, user subroutines and driving scripts by source or content hash — and, where the solver writes one, its echo of the resolved input with defaults applied |
   | E2 | solver identity: name, version, revision, floating-point precision of the solver and of each output stream, parallel layout |
   | E3 | for every analysis phase and restart segment: status, final time, number of steps or increments, and the criterion that ended it; plus the solver's diagnostics reduced to counts by class (errors, warnings, element inversion, non-finite or out-of-range kinematics, entities deleted for numerical reasons, initial contact overclosures) |
   | E4 | the time-integration record — explicit: time-step history, the part controlling it, and non-physical mass added over time, in total and per part; implicit: per increment, iterations, residual and energy norms against their tolerances, cut-backs, and any increment accepted without convergence |
   | E5 | a global energy ledger over time in which every term of the solver's energy balance is separate and the balance identity is declared (which terms sum to the total, with what sign, which are contained in others): at minimum kinetic, internal, and external work, plus each non-physical or dissipative term the formulation has — zero-energy-mode control, contact and penalty work including rigid surfaces, damping, artificial viscosity, work done by mass scaling, energy removed by deletion |
   | E6 | per-part energy and mass, and per-contact-interface energy, over time |
   | E7 | applied-load resultants, including loads the solver computes internally, and boundary and contact reactions, over time |
   | E8 | field output at the points where the constitutive update is performed, or with the reduction declared: displacement, velocity, mass; stress and strain with their measure, frame, and shear convention declared; every argument of the material's admissibility functions as the update used it, declared per material from a closed vocabulary; pressure, density, and specific internal energy where an equation of state makes them independent state; section-point stresses, or section resultants with their conjugate deformations, for structural elements; neighbour count, smoothing length, and activity flag for particle methods; the deletion flag under erosion |
   | E9 | ledger, loads, and reactions sampled at an interval that divides the field-output interval and is no longer than a declared fraction of the shortest physical time scale of interest; both declared |
   | E10 | (a) the input's units of mass, length, and time (and temperature where thermal); (b) three SI anchors of independent dimension — a density; a stress-dimension constant (a modulus or a strength), or a wave velocity for a material that has none; a characteristic length — each naming the input quantity it corresponds to, stating the SI value of that input quantity to a declared number of significant digits, and giving its kind of source (handbook, specimen measurement, test report, synthetic) |

   The items are stated from the mechanics — every mechanism that does work
   on, or removes energy or mass from, the discretised system has its own
   ledger term — not from any solver's output list. With one solver
   implemented, the item list and term names are **provisional**; the second
   solver's adapter is the trigger for a dated note revising them.

   *Applicability.* An E5–E7 term is required of a run only when the run's
   input-derived traits make it applicable (zero-energy-mode work for an
   under-integrated part, contact work for a contact definition, and so on).
   A quantity is assessable if and only if every required item is present;
   a term that is not required is not an absence. Where a solver keeps no
   energy ledger, E5/E6 may be met by one derived from its outputs with the
   method declared; the quantity and its criterion are unchanged, and the
   measurement records the ledger's origin. E9 is judged by its own
   integrity quantity and is not a precondition of the others. The
   declarations inside E8 and E10 are made once per dataset.

   *One definition per quantity.* When required evidence is absent the
   verdict is `not_assessable` with the missing items named; there are no
   degraded-mode substitutes computed from whatever a file happens to hold.
   The same row is therefore the feedback a contributor needs.

   *Units are verified from outside the label.* A consistently mislabelled
   unit system cannot be detected from inside the data — every identity
   survives the relabel, so the error recorded in ADR-0030 would have passed
   every self-consistent check. Units therefore form their own catalogue
   category, judged only against things outside the label. *Anchors*
   (E10b): the named input quantity, converted with the declared units, must
   equal the anchor to the anchor's declared digits, otherwise `fail`;
   anchors verify the label for the anchored constants only. *Plausibility
   screens* on the SI-converted input constants and response magnitudes need
   the unit label (E10a) but no anchors, and an excursion is `review`.
   Without a declared material family only density, judged against the range
   of condensed matter, can catch a mass-unit error; modulus and strength
   ranges discriminate only when the dataset declares a material family, and
   each range cites a materials reference; quantities with no net mass
   dimension (wave speed, velocity, strain) screen length and time only.
   *Dimensionless groups* formed inside the input (yield stress over
   modulus, the elastic constants against one another) need no unit at all
   and catch an input that mixes units within a material definition. Any
   unit declaration the input itself carries is cross-checked. A dataset
   that declares its anchors synthetic has no outside to be checked against:
   its anchors verify arithmetic only, and the report says so. Two
   comparisons stay in the catalogue but are *not* units checks, because
   both sides use the input's own numbers: stored density against the
   input's density (an ingestion-mapping check), and the solver's time step
   against a stability estimate (numerical health). The anchor check is
   built in the first slice as a deliberate exception to clause 5's
   build-on-first-evidence rule, because its evidence is a declaration, not
   solver output: the maintainer supplies the test bed's three anchors to
   the hand-run. Where a benchmark records its declarations is decided by
   the ADR that amends ADR-0027; until then their absence reads
   `not_assessable / no_declaration_home`, a platform-side reason.

   *What this ADR does not decide.* Data generated in this repository is
   built to meet the requirement, and a contributed dataset is measured
   against it. Whether an unmet item blocks admission, and how datasets that
   can never meet it (third-party public data with no solver input) are
   treated, is ADR-0065 follow-up 2's decision. Re-judging needs only the
   committed measurements (clause 8); re-*measuring* needs the evidence, and
   what may ever be published is the whitelisted record of clause 3, never
   raw solver files. Whether and how that record ships with a dataset is
   follow-up 2's decision together with ADR-0031 and ADR-0040; until then
   archive contents are untouched. Dataset-level standards (a record of
   discarded runs; mesh, time-step, and numerical-parameter sensitivity
   studies; a repeat-run noise floor including parallel decomposition;
   cross-solver and constitutive-sensitivity studies; a physical-test
   anchor) remain ADR-0065's. E2 records the parallel layout; it does not
   establish reproducibility across layouts. How each solver meets E1–E10 is
   documented with its adapter under `data_generation/`; for LS-DYNA that
   mapping is checked against the keyword manual before it is written down.

3. **Evidence is read into record types by one concrete extractor — not a
   solver abstraction.** `core/evidence.py` defines small solver-neutral,
   SI-valued records (`Absence`, `PartTraits`, `InputFacts`, `RunEvidence`,
   `DeclaredFacts`), re-exported through `structbench.core`.
   `core/io/lsdyna_run.py` is the single extractor. There is no protocol,
   registry, or plugin seam (ADR-0010 stands), and LS-DYNA vocabulary stays
   inside `core/io/lsdyna*.py`. A record field is implemented when a run
   first supplies the evidence for it, never before. Each measurement lists
   where its evidence was read from — `case` (the canonical file), `input`
   (the solver input stored in it), `run` (the run-evidence record),
   `declared` (benchmark declarations). That list is a locator, not part of
   a quantity's definition: moving evidence between locations never changes
   a `definition_version`.

   Parsers take *text*, never directories. Per-dataset glue under
   `data_generation/` builds explicit paths from case ids, calls the
   extractor, and writes one whitelisted run-evidence record; the library
   and CLI never receive a run directory. Evidence records carry no unparsed
   text: every string field is an enum or matches a fixed pattern (a version
   token, for example), so licence numbers, host names, and local paths
   cannot reach a report. The verbatim input of E1 is outside that guarantee
   and is scanned by the dataset's glue before ingestion. Input facts are
   read by a new faithful, **fail-closed** reader: blank data lines and
   fixed field positions are preserved, and any construct the reader does
   not resolve yields `unparsable` rather than a guess. Where the solver's
   echo is supplied, effective settings are read from it; an absent setting
   is otherwise `not_assessable`, never assumed to be the default. The
   existing `_card_blocks` drops blank and non-numeric rows and stays
   private to material ingestion. Run-evidence quantities can be measured
   with no `Case`, so a run discarded before it became a case can still be
   measured.

4. **Measuring and judging are separate functions.** `measure` turns
   evidence into threshold-free measurements (quantity, value, unit,
   sources, sample count, detail) or a typed absence. `judge` turns
   measurements plus criteria into verdicts and never touches data. Verdicts
   are therefore a pure function of committed measurements and tracked
   criteria, and no command-line flag alters a criterion.

5. **Five verdicts, one owner each, evaluated in a fixed order.**

   1. Formulation traits own `not_applicable` — the quantity does not exist
      for this formulation.
   2. Evidence absence owns `not_assessable`, always with a typed reason and
      the missing evidence items named.
   3. Measurement against a criterion owns the rest: `pass` or `fail` for a
      requirement or an instrument tolerance; `pass` or `review` for an
      indicator (clause 7). `review` means the indicator exceeds its
      reference level; this instrument reports it and does not decide it.

   The test between the first two is whether better evidence could change
   the answer. Unknown traits yield `not_assessable`, never
   `not_applicable`. Reasons say who can close the gap. *The contributor*:
   `not_requested`, `not_computed`, `source_missing`, `unparsable`,
   `source_unreadable`. *Nobody, for this solver*:
   `not_available_from_solver`. *The platform*: `not_ingested`,
   `unsupported`, `no_ratified_criterion`, `no_declaration_home`,
   `stale_definition` — these are
   reported in a separate block ("not yet checked by this instrument") and
   never count as a dataset's gap.

   The catalogue is *specified* from the requirement. Every quantity has a
   catalogue row from the first slice — name, category, required evidence,
   trait gate, meaning of a violation, and status `specified` or
   `implemented` — and rows are data, not code paths. The trait gate and
   the evidence gate need no solver-output parser, so they run for every
   row: a contributor is told which evidence is missing even for a quantity
   whose measure does not exist yet. Only the *measure* — parser, record
   field, kernel — is built when a run first supplies the evidence, so its
   tests rest on real output. An applicable quantity whose evidence is
   present but whose measure is unbuilt is listed as an instrument gap,
   never rendered as a verdict about the dataset. A data-free test checks
   that the union of the rows' requirements is exactly E1–E10.

6. **Traits are per part and input-derived; declarations are claims.** A
   case may carry several materials, so the extractor reads the
   part-to-material map from the solver input. Traits are never inferred
   from a case file's element keys; an element block with no part in the
   input is reported by an integrity row and left out of the per-part
   checks — a finding, not a presumption that the block is benign. A dataset
   with no solver input does not meet E1: its traits are unknown, every
   trait-gated quantity is `not_assessable / source_missing` naming E1, and
   its card declarations are rendered as unverified claims that never issue
   `not_applicable`. Material knowledge has two homes, one of which exists
   already: *solver keyword → material class* is ADR-0012's `canonical_model`
   (today `_CANONICAL_MAT` in `core/io/lsdyna.py`, stored with each case);
   *class → the admissibility functions, the meaning and bounds of each
   state variable, and which checks are assessable* is new, in
   `verification/materials.py`, keyed on `canonical_model`. No second
   keyword table is added; a null class yields `not_assessable /
   unsupported`, and growing the enum stays the follow-on ADR-0012
   anticipates. Once E8's declaration has a home, the declared meaning is
   cross-checked against the class table. `BenchmarkSpec.hardening_curve`
   stays the only operative yield table (ADR-0064); the input's table is
   read only to verify that the two agree. Constitutive bounds are
   two-sided only on constitutive-point data; on reduced data the upper
   bound is reported as one-sided and its lower-bound companion is
   `not_assessable`.

7. **Criteria are the platform's standard, of three kinds fixed by where
   the bound comes from.** A *requirement*'s bound is definitional — zero,
   exact equality, or the input's own value: no non-finite values, normal
   termination, no increment accepted without convergence, state variables
   within their bounds and monotone where the material says so, declared
   facts that agree with the input. An *instrument tolerance*'s bound is
   computed from storage precision or from a named and confirmed algorithmic
   mechanism, including where it serves a requirement (the end time reached,
   closure sums); it is never fitted to values measured on the test bed, and
   a quantity whose mechanism is unconfirmed is measured and published with
   no criterion. Both yield `pass` or `fail`. An *indicator*'s bound is a
   sourced *reference level*, and it yields `pass` or `review`. The design
   doc's catalogue is authoritative on which quantity is judged how.

   *Why indicators.* Some error measures have no definitional bound: how
   much energy imbalance, zero-energy-mode energy, added mass, or contact
   energy is tolerable depends on the problem and the formulation. The
   source survey behind this ADR found the published limits for these to be
   community-practice limits scoped to a domain — a ten-percent start-to-end
   change in total energy in roadside crash practice; an over-the-run
   residual "on the order of 10⁻²" in a standard text, given as a stability
   check; a descriptive "generally less than 1 %" with no normalisation in
   one vendor's guide; one to two percent for energy creation only, on a
   statistic that excludes zero-energy-mode and contact energy, in
   another's; no number at all in a third's — and none is presented as a
   derivation in the passages read. No published limit was found for
   time-step collapse, and none has yet been surveyed for plausibility
   ranges. So an indicator has one fixed definition and reference levels,
   not a universal limit.

   *The energy indicator* is the signed residual of the run's declared
   balance identity (E5). Let `E_tot(t)` be the sum of every term the
   identity declares, each taken once with its declared sign and no
   contained term counted twice, and `t0` the first ledger sample. Then

   ```
   R(t) = [E_tot(t) − E_tot(t0)] − [W_ext(t) − W_ext(t0)]
   r(t) = R(t) / max( |E_tot(t0) + W_ext(t) − W_ext(t0)|, E_kin(t), E_tot(t) − E_kin(t) )
   ```

   This is the form of Belytschko et al. with two things its three symbols
   leave implicit made explicit: its internal work is the work of all
   internal forces, so every non-kinetic ledger term belongs with it; and
   energy present at `t0` — initial velocity, preload, stored chemical
   energy — counts as input. External work includes the work of reactions at
   prescribed-motion boundaries and of body forces. `r > 0` is energy
   created; `r < 0` is energy unaccounted for. Reported: the largest gain
   and the largest loss over the run, each with its sample time, and the
   signed final value; being sampled, the extremes are lower bounds on the
   solver-step extremes.

   *Reference levels.* A level is fixed in advance, cites a verified source,
   and is never adjusted in light of a StructBench measurement. It may be
   attached to a quantity only where the source's statistic, normalisation,
   and evaluation time are identical to the quantity's definition, or equal
   to it under a condition the level's scope records; otherwise the source
   is context in a rationale and yields no level, and where sources use
   different statistics for one phenomenon, each statistic given a level is
   its own catalogue quantity. Levels are scoped by run traits derived from
   the input and the ledger — time integrator, spatial conservation class,
   energy source, mass scaling, friction, dimension — and by declared intent
   only where intent cannot be derived (a quasi-static task); a dataset
   cannot select a level by labelling itself. For a run-global quantity
   every part must fall inside the scope, and the scopes of one quantity's
   levels are pairwise disjoint. A level is scoped no wider than its source's
   own domain: a text on finite-element time integration yields a level for
   a Lagrangian mesh, and for a particle method whose ledger is likewise
   built from the work of its internal forces, and for nothing else. Where
   no level covers a run's scope — a particle method that is not pairwise
   conservative, an advecting mesh — the indicator is still measured and is
   rendered in the dataset's main table with its value and with the levels
   that exist for other scopes, labelled as out of scope, so that a person
   can make the call the platform has no sourced basis to make. A level
   records whether its bound is strict, as its source states it. At or below the level the verdict is
   `pass`; above it, `review`; with no level for the run's scope,
   `not_assessable / no_ratified_criterion`.

   *The call on a `review`* is not this instrument's. Whoever is
   accountable for the dataset's entry in this repository — the maintainer
   for shipped benchmarks; for a contribution, the contributor named in its
   provenance, merged by the maintainer — may record a dated, attributed
   *disposition*: accepted or rejected, with a rationale. It names the
   `definition_version` and the reference level it answers, is rendered
   beside the row, is shown as stale when either changes, and never changes
   the measurement or the verdict; a reader is free to disagree with it. How
   `fail`, `review`, and dispositions count towards a benchmark's
   compliance or admission is ADR-0065 follow-up 2's decision.

   Every criterion is a record with a required, rendered rationale.
   Benchmarks cannot loosen a criterion or a reference level. Measurements
   may exist before a criterion does; a criterion or level fixed after a
   test-bed value was already known says so in its rationale. A
   grandfathered benchmark may visibly fail or read `review`. Gates,
   severities, and scores are not introduced without a further ADR.

8. **Measurements are the committed record; verdicts are generated.** The
   JSON report carries a schema id, the package version, a per-quantity
   `definition_version`, the SHA-256 of each measured case file and of the
   run-evidence record a `run` row was measured from, values rounded to
   fixed significant digits, and the facts levels are scoped on — the run
   traits derived from the input and the ledger, and any declared intent —
   so that judging needs nothing but this file and the criteria; it carries no timestamp, host, or path, and a
   re-run is byte-identical. Working output goes to gitignored `runs/`. The
   published record is `docs/datachecks/<benchmark>.json` with a generated
   `.md`, pinned by a data-free test that the markdown equals
   `render(judge(json, criteria), dispositions)`; dispositions are tracked
   literals, are not part of the JSON, and never enter `judge`. A stale `definition_version` is judged
   `not_assessable / stale_definition` for that row only. The CLI exits `0`
   on completion and `2` on usage or I/O errors; failing verdicts are
   reported in the output, not through the exit code, and an unreadable
   input becomes a recorded row rather than an abort. Changing a criterion
   never bumps a benchmark version (the ADR-0033 rule); removing a case
   because of a verdict is a split change and does (ADR-0019).

9. **Scope.** This ADR discharges none of ADR-0065's four follow-ups. It
   supplies the measurements follow-up 2's compliance table can render, the
   run-evidence requirement its admission rules can cite, and the kernels
   follow-up 1's properties block can call. Benchmark cards, results
   registries, `render.py`, the generated benchmark pages, the case schema,
   and the archive contents are untouched. Nothing in the catalogue, the
   records, or the criteria is shaped to make the legacy archives
   assessable; their published records are the visibility ADR-0065 asks for.

## Alternatives considered

- **Shape the evidence model around what the existing archives kept** — the
  first draft of this ADR did: evidence tiers defined by who holds the
  files, a coarse energy check computed from stored globals wherever no
  ledger was written, a precedence rule for when the two disagreed, a
  declared-traits fallback for data with no solver input, tolerances
  calibrated on the legacy cases, and a two-grid stand-in for a refinement
  study. Rejected (maintainer, 2026-09-21). The archives are small, pre-date
  the repository, and are unrepresentative of what future runs can supply; a
  fallback gives one quantity two definitions; and the right response to
  missing evidence is a stated requirement plus an honest `not_assessable`,
  not a workaround that becomes permanent. A first revision then copied the
  requirement from one solver's output list; the second review caught that,
  and clause 2 is stated from the mechanics instead.
- **No new module — measures in `datasets/`, verdicts and catalogue in
  `benchmarks/`** (the `timeline.py` precedent). Rejected.
  `benchmarks/registry.py` imports `eval`, so `eval` could never import the
  result types; `datasets/` is chartered for data loading; a run with no
  registered benchmark and no case could not be judged; and the catalogue is
  several times `timeline.py`'s size. The smaller footprint buys a forced
  public-API move later.
- **Host the checks in `eval/`.** Rejected: everything in `eval/` scores a
  model's prediction against ground truth, whereas these checks judge the
  ground truth itself, and `benchmarks → eval` is already a flagged edge.
- **Move the tensor kernels into `core/` for a torch-free import path.**
  Rejected for now. It moves public functions for an import-time benefit
  only (torch is a hard dependency, ADR-0018). Revisit if per-call import
  cost becomes a measured problem.
- **A protocol or plugin registry for evidence extractors.** Rejected under
  ADR-0010: a seam designed against one solver over-fits it. The requirement
  itself is stated physically and marked provisional until a second solver
  tests it.
- **Catalogue rows for unbuilt quantities kept in documentation only.**
  Rejected: a contributor whose run omits evidence for an unbuilt quantity
  would see no row at all, so whether a gap is visible would depend on what
  the test bed happened to exercise. Rows are declarative data; only
  measures are deferred.
- **Add the solver's own global energies to E8 so that energy closures are
  assessable without a ledger.** Rejected: it keeps a second, ledger-free
  route to the same quantity. Closures compare a ledger term with the field
  sum and need both.
- **One hard energy limit taken from a single source.** Rejected
  (maintainer, 2026-09-21). The sources read disagree by an order of
  magnitude, measure different statistics (start-to-end change against an
  over-the-run residual), are scoped to their own domains, and rest on
  community practice rather than derivation. A hard pass/fail would claim
  an authority no source has. Hence a general indicator, reference levels
  scoped by run traits, and a human call on `review`.
- **Apply the general explicit-integration energy level to every spatial
  discretisation.** An earlier draft did, arguing that otherwise the test
  bed's energy rise would go unflagged. Withdrawn: that is a scope argued
  from a measurement, which clause 7 forbids, and the source is a
  finite-element text — for a formulation whose ledger is not built from
  the work of its internal forces, the quantity is not the source's
  statistic. The value is still published, with the out-of-scope levels as
  context, so the person relying on the data sees exactly what the platform
  does and does not know.
- **Per-benchmark bounds ratified after measuring the benchmark.** Rejected:
  the criterion becomes a function of the data it judges, and no common
  standard remains. Reference levels differ from this in both respects: they
  are scoped by run traits derived from the input and the ledger, not by
  benchmark, and they
  are fixed from a source before they are applied.
- **A units check built from internal consistency** (stored density against
  input density, wave speed, energy closure). Rejected: all of these survive
  a consistent relabel, so none can fail on the error they are meant to
  catch.
- **Commit verdicts rather than measurements.** Rejected: every criterion
  change would need the data again, and a verdict stale against the code
  would be invisible.
- **Store verdicts in the case file, or as per-case sidecar files.**
  Rejected: the first is a schema change (ADR-0012/0013), the second changes
  ADR-0031's archive list. Any archive artefact belongs to ADR-0065
  follow-up 2.
- **Profiles or criteria in TOML/YAML.** Rejected for ADR-0027's reasons.
- **A hand-committed markdown report** (the `docs/timelines/` precedent).
  Rejected: no drift gate, so a criterion could change while the published
  table stays old.
- **Reuse `_card_blocks` for named input parameters.** Rejected: it silently
  drops blank and non-numeric rows, so positional extraction can return a
  plausible but wrong table.
- **Adopt ADR-0064's 1.001 as the reference-data yield tolerance.**
  Rejected: that number bounds a model head's float32 saturation error
  (~2e-7). The excess measured on reference data (3.5e-4, one case) is three
  orders larger and its cause is unconfirmed, so under clause 7 the quantity
  ships measured, with no criterion.

## Consequences

- **Flag-first items approved with this ADR**: the new module; the two new
  `core` files and their re-exports; the `n_valid_frames` export; a new
  `cli/datacheck.py` (module entry point, no console script); glue scripts
  under `data_generation/<solver>/<dataset>/`; the `docs/datachecks/` tree.
  `docs/ARCHITECTURE.md` (package layout, module responsibilities,
  dependency graph) is updated in the change that lands the module, and the
  README Roadmap gains the corresponding entries.
- **A run-evidence requirement now exists.** Data generation in this
  repository requests it from the solver by construction, and a
  contribution is checked by running the instrument on it: every
  `not_assessable` row names the evidence that is missing and who can supply
  it. This ADR makes unmet items visible; it does not make them blocking.
- **Delivery is staged, with Taylor as the test bed**: the requirement and
  its LS-DYNA realisation first; then quantities whose evidence is in the
  case file and its stored input; then run-evidence quantities; then
  fixing reference levels and publication. Because the legacy runs do not meet the
  requirement, the staging includes — as the maintainer's decision — one
  unpublished *conformance run* made with the standard input block, so that
  the realisation and the measures for E4–E7 rest on real output. The
  implementation plan lives in `docs/plans/`.
- **`tools/state_probe` is the acceptance oracle**, untouched in the first
  slice: the port must reproduce its recorded numbers before anything
  migrates. Retiring its copies is a later change with a regression first.
- **Tests stay synthetic-only.** Real data enters through environment-gated
  tests that skip when unset. Parser fixtures use invented numbers on the
  real file layout and name the solver version they mimic; unknown labels
  fail closed. A new import-boundary test resolves relative imports (the
  existing torch-free test does not).
- **The grandfathered benchmarks will show honest gaps**, which is the
  visibility ADR-0065 asks for: rows that read `not_assessable` with the
  missing evidence named, a dataset with no solver input that is almost
  entirely `not_assessable`, possibly a failed requirement or a `review`. None of this
  withdraws or re-scores anything.
- **Deferred.** The list with triggers lives in the design doc. The durable
  ones: measures for specified quantities (the first run that supplies the
  evidence); revision of E1–E10 (the second solver); a home for E8's
  declaration and E10's anchors (the ADR that amends 0027, ADR-0065
  follow-up 2); the material-class enum (ADR-0012's follow-on); disposition
  records — typed, dated literals; the ADR-0033 registry pattern is the
  candidate home, decided when first needed (the first published row that
  reads `review`); a declared material family and declared quasi-static
  intent, alongside the other declarations; reference levels for scopes no source yet covers; carrying
  the ledger inside the canonical file, and ingesting the arrays the adapter
  currently discards (a dated note on ADR-0016).

## Catalogue note (2026-09-22, agent + maintainer)

Every catalogue row gains one field, `bears_on`: which artefact a violation
condemns, and so who would have to act — `input` (the solver input deck),
`response` (the stored fields a user loads and trains on), `run` (the run's
numerical conduct and the solver's own record), `declared` (what this
repository claims about the runs). It is carried as a side table keyed by
quantity name, like the reader-facing titles, so a row that is not
classified is a `KeyError` at import.

It is a **definition, not a measurement**. It says where a finding of this
kind lands for any dataset, never what was found in one, so it is not in
the JSON record and adds nothing to it: no re-measurement, and every
published report regenerates from the record it already has. It is
distinct from `Location`, which locates the evidence a measurement was
read from; the two differ exactly where it matters — the evidence for
"stored elements that no input part owns" is read from both the case and
the input, while the violation lands in the stored response alone.

The motivating gap: a reader could not tell which of Taylor's two
archive-level failures touched the arrays they would train on. Both read
`fail`, identically, in all 33 cases. They land in different artefacts —
the hardening-table dip is an input defect, the unowned elements are in
the stored response — and the report now says so in the summary and on
each finding, grouped by artefact so a count is never read off the wrong
name.

This introduces no gate, severity or score, and clause 7's reservation
stands: how `fail`, `review` and dispositions count towards compliance or
admission remains ADR-0065 follow-up 2's decision. `bears_on` states where
a finding lands; it does not weigh it, rank it, or aggregate it, and no
verdict, bound, criterion or measurement changed with it — the regenerated
Taylor report carries the counts it carried before (`fail` 66, `pass` 627,
`not_applicable` 495, `not_assessable` 858). Dispositions remain deferred
on their stated trigger (the first published row that reads `review`), and
this note does not build them: the report still records no maintainer
judgement of any finding.

Additive to clause 2's catalogue, and presentation-only in effect. The
report changes it serves — a per-quantity spread replacing the per-case
table, which did not survive contact with a 110- or 1200-case benchmark;
the removal of a section that restated reasons already on their rows;
deviation-from-unity rendering for ratios whose printed digits hide their
content — carry no decision and are recorded in the commit, not here.

## Widening note (2026-09-23, agent + maintainer)

The instrument has met a second benchmark. All 110 cases of the notch-beam
impact sweep were measured against the canonical archive and each run's
message file, judged, and published as
`docs/datachecks/notch_beam_2d_impact.{json,md}`, linked from the benchmark's
landing page. Every case was readable and carries its SHA-256. The verdicts
are uniform across the sweep. *(Counts as first published: 14 pass, 6
measured but not judged, 28 not checked, 14 not applicable, no findings. The
coverage note below moves them.)*

Nothing was added to the catalogue, no criterion changed, and no reference
level was ratified: the maintainer's 2026-09-21 decision stands, and the
indicators that could be measured are reported as measurements. What the
widening required of the repository was one piece of per-dataset glue
(`data_generation/lsdyna/2DNotchBeam/collect_run_evidence.py`), which maps the
impact grid and the two probe folders to run directories and reads `mes0000`
alone. These decks never requested `*DATABASE_GLSTAT`, so there is no energy
ledger and no time-integration record to read; the nine conservation rows and
the two time-step rows report `not_assessable` naming the missing evidence,
and no substitute was built for them (CORRECTIONS 2026-09-21).

**"No findings" is weaker here than the phrase suggests, and the report says
so.** Twenty-six of the sixty-two checks could not be made. Beyond the absent
ledger, the material rows are `not_assessable / unsupported` because the
platform carries no class for `*MAT_CONCRETE_DAMAGE_REL3` (K&C) or
`*MAT_PLASTIC_KINEMATIC`, and the four input-constant rows are
`not_assessable / unparsable` because `read_input_facts` has no card layout
for them. Adding either is the material-class ADR that ADR-0012 anticipates
and `verification/materials.py` requires; it is not discharged here.
*(Corrected 2026-09-23: this note first called it the single largest
recovery available on notch. It is not. Of the 28 rows notch cannot check,
13 are `source_missing` -- the ledger, time-step history and load
resultants the runs never wrote -- and no material class reaches them. A
class plus the two card layouts recovers about six; the layouts landed
2026-09-23 and were worth five.)*

**The widening's real yield was two defects in the message-file reader that
Taylor could not have exposed.** Diagnostics were counted by any line
mentioning an error, so the explanatory prose under `*** Warning 21329` —
"Curve ID 723 has discretization error of" — made every run of the sweep
report a solver error it never had; the count now follows the line that
*raises* a diagnostic, the severity word at its head. And solver identity read
the revision from an `SVN Version:` line alone, while these runs are a later
R12 build printing `Revision: R12.1-190-gadfcdf9018` and no SVN line, so a
revision present in the file was recorded as absent and E2 failed. The SVN
number still wins where both are printed, so Taylor's published record is
byte-identical — the data-free record test is what confirms it — and a
revision that is not a version token is now dropped with a token rather than
guessed. Both of notch's apparent failures were the instrument's, not the
data's. This is the argument for widening: an instrument that has met one
dataset has been calibrated against one dataset's accidents.

A third defect was found while reading the two records side by side, and is
fixed here too. `not_applicable` is a claim *about the input* — it states no
quantity of this kind — and two of the units screens were making that claim
on an input the reader had not got through: with no card layout for K&C
concrete there is no yield table and no modulus to find, and notch's report
said the screens did not apply where the honest answer is that they could not
be made. Both now route through one helper that separates the two, so a
`not_applicable` on an input screen means every card was read. This moves two
rows per notch case out of `not_applicable` and into `not_assessable /
unparsable` (hence the counts above), and changes nothing measured: Taylor's
materials parse, so its re-measured record is content-identical.

Two measured-but-unjudged numbers are left for a reader to weigh, and both
are candidates for a criterion whenever one can be sourced: the fewest
neighbours of any particle runs 3 to 8 across the sweep, and every run raised
exactly one solver warning.

Open after this note, and unchanged by it: the four ADR-0065 follow-ups; the
kinetic-energy closure's tolerance; the three unbuilt E9 rows; ratifying any
reference level; the conformance run with the standard input block. Newly
visible: the material-class ADR above, and whether a third benchmark is worth
the instrument's time — wave-1D was declined 2026-09-21 and its runs kept no
`glstat` either, so it would report a similar shape.

## Coverage note (2026-09-23, agent + maintainer)

Four gaps found by reviewing the instrument against its own solver-side
requirement are closed here. The catalogue goes to 63 rows, 43 implemented.
No reference level is ratified and no indicator gains a verdict; both new
criteria are a definitional requirement and an instrument tolerance argued
from printed precision, which is the class clause 7 already allows.

**`stored_globals_match_ledger` is built** (E5 + E8 + E9, `bears_on`
response). It compares each stored global channel with the solver's own
series at the instants both were sampled, normalised by that channel's peak,
and reports the worst. It is the only row that sees an *ingestion* error —
a channel dropped, misnamed, or left in the solver's units — because it is
the only place the stored arrays meet the solver's independent account of the
same run. Nothing new was needed to build it: Taylor has supplied all three
evidence items since the instrument existed.

On its first run it found one. Taylor's stored `kinetic_energy` and
`internal_energy` reproduce the ledger to about 1e-6, but `total_energy`
disagrees by 0.6 to 1.2 % of the peak, growing through the run. The stored
channel equals the ledger's kinetic plus internal energy to 2e-6, while the
solver's printed total also carries the rigid-wall term — so the two are
different quantities, both written by the same solver, and the canonical
`global/total_energy` is not the run's total energy. The adapter copies
`d3plot`'s `global_total_energy` verbatim, so this is a mismatch between two
of the solver's own outputs, not an arithmetic slip. Whether `d3plot`'s
definition varies with the terms `*CONTROL_ENERGY` switches on is
unestablished and is now an open point of the standard input block. **What to
do about it is not decided here**: renaming or re-deriving the stored channel
touches the schema and every archive, and is the maintainer's call. Until
then the row reads `fail` on all 33 Taylor cases, truthfully.

**`input_requests_required_evidence` is new** (E1 only, `bears_on` input).
`STANDARD_INPUT_BLOCK.md` stated what a run must ask the solver to write, and
nothing read it back: the instrument could say the runs did not supply the
ledger, never that the input never asked for one. Those are different
failures with different owners, and only the second is fixable — before the
run, for free. The row counts the omitted requests, gated on the features the
model actually has, so an SPH run is not asked to compute hourglass energy and
a run without contact is not asked for contact forces. `read_input_facts`
gained `energy_terms_computed` and `databases_requested` for it, and
`damping_defined` so the damping term can be gated too.

It closes a **silent-acceptance path**. `*CONTROL_ENERGY` leaves three
dissipation terms uncomputed by default; an uncomputed term never appears in
`glstat`, and the ledger reader builds its balance identity over the terms
that are present, so the identity reproduces the solver's printed total and
the ledger is accepted as complete. Nothing could tell a term that was zero
from a term that was never computed. Today both benchmarks are SPH and the
hourglass rows are `not_applicable`, so the path is masked; the first meshed
benchmark is where it would have mattered.

Both legacy sweeps fail the row, as expected and as CORRECTIONS 2026-09-21
prescribes — a requirement on future data generation, with the legacy gap
reported honestly. Taylor omits `*DATABASE_RWFORC` though it has a rigid
wall, so its wall forces can never be read. Notch states `*CONTROL_ENERGY`
in full, computing every term, and then requests no `glstat` at all: the
ledger existed inside the solver and was never written.

**`not_applicable` on the two remaining input screens was a false claim.**
`input_strength_plausible` and `input_constants_plausible` said the input
states no strength and no modulus, on an input whose material cards the
reader could not parse. Both now route through `unstated_or_unread`, so
`not_applicable` on an input screen means every card was read.

**Units anchors have a home**: `BenchmarkCard.units_anchors`, carried into
`DeclaredFacts` by `declared_from_spec`. Taylor declares one — copper at
8.9e3 kg/m³ to two significant digits, `material:2:density`, handbook — and
`units_anchors_consistent` passes on all 33 cases, which is the first time
the check designed to catch a wrong unit label has run at all. **One anchor
is not three.** A density pins mass against length; a time-unit error still
passes. A second anchor of independent dimension needs a value the maintainer
stands behind: the deck's shear modulus is a calibration number, not a
handbook constant, so labelling it `handbook` would be false provenance, and
the resolver has no locator for a velocity. Notch declares none, because its
material cards do not parse and an anchor on them would resolve to nothing.

Counts after this note. Taylor: 20 pass, 4 findings, 17 measured but not
judged, 7 not checked, 15 not applicable. Notch: 16 pass, 1 finding, 8
measured but not judged, 23 not checked, 15 not applicable. Taylor's two new
findings are the stored-globals mismatch and its missing `*DATABASE_RWFORC`;
notch's single finding is its three omitted requests. Nothing else moved: no
existing verdict changed, and the hardening-table and unowned-element
findings stand as before.

**The two card layouts landed the same day, without an ADR.** Reading a card
layout says what numbers a card holds; only a material class says what they
*mean*, so `*MAT_CONCRETE_DAMAGE_REL3` and `*MAT_PLASTIC_KINEMATIC` now give
up their density, Poisson ratio and (for the steel) Young's modulus, while
`canonical_model` stays `None` and every constitutive row stays
`not_assessable / unsupported`. Neither yield law is tabulated — K&C's
surface is generated from the unconfined compressive strength, the steel's is
bilinear — so neither gets a `yield_table`, whose contract is knots a deck
states verbatim.

Five rows moved off `unparsable` on notch: the stored density now matches the
input's to 2e-16, and the density anchor its card already stated now resolves
and agrees, so `units_anchors_consistent` **passes**. That is two independent
confirmations that notch's `kg-mm-ms` declaration is right and that ingestion
converted it correctly — on a benchmark where, an hour earlier, nothing about
units could be checked at all. `input_strength_plausible` is correctly
`not_applicable`: every card was read and none states a tabulated strength.

One observation for the material-class ADR, found reading the deck and not
yet caught by any row: the steel card carries `e = 200.0` and `sigy = 337.0`
in a unit system whose stress unit is 1 GPa, a first-yield strain of 1.7.
`input_dimensionless_groups_plausible` is the row built to see exactly this,
and it cannot, because it reads `yield_table` and a bilinear law has none. A
scalar `yield_stress` on `MaterialInput` would close it, and would change
what two units rows measure, so it waits for a decision rather than being
slipped in. The steel parts are protocol-kinematic — driven by ground truth,
excluded from loss and metrics — so nothing a user trains on depends on it.

Open and untouched: the four ADR-0065 follow-ups; the material-class ADR that
ADR-0012 anticipates, worth two more of notch's 23 unchecked rows (the 13
`source_missing` rows are beyond any class, the runs having written no
ledger);
the kinetic-energy closure's tolerance; E6, E7 and the two remaining E9 rows,
which wait on a run that supplies their files; and what to do about
`global/total_energy`.

## Claim audit (2026-09-23, agent)

A deliberate pass over every place the module reports a quantity as absent
or inapplicable, prompted by three such misreports being found by accident
in one week. Twelve sites: eleven state something true, one did not.

`input_dimensionless_groups_plausible` returned `unsupported` when no
material offers both a tabulated yield stress and an elastic modulus. The
instrument can compute that ratio; what is missing is in the input. Worse,
`unsupported` is in `PLATFORM_REASONS`, whose contract is that such rows
"never count as a dataset's gap" — so the row read as an instrument
limitation *and* exempted the data from a gap that was its own. It now
routes through `unstated_or_unread` like the other input screens: an input
whose cards were all read and offers no such ratio is `not_applicable`; one
with an unparsed card keeps the typed absence. On notch the row moves from
"not checked" to "does not apply", and the sweep's counts move to 18 pass,
1 finding, 8 measured but not judged, 19 not checked, 17 not applicable.
Taylor is re-measured and byte-identical: its material states both.

Two sites are defensible and left alone, recorded so the next audit does not
re-derive them. `kinetic_energy_closure` treats a peak kinetic energy of
zero as "nothing ever moved", which is true; a *negative* peak is physically
impossible and would be a defect reported as inapplicable, so a guard there
would be honest if the case ever arises. `total_energy_change_final` does
not apply when the initial total energy is zero, which is right for a ratio
but means the row can never apply to an externally driven run — a coverage
hole, not a false claim.

The rule this audit was testing, from the ADR-0067 build note: a verdict of
`not_applicable`, and the reason on an absence, are both **claims about the
data**, and an instrument whose purpose is to avoid asserting untrue things
about data has to hold its own reporting to that standard. Four instances in
one week says this is a pattern in how the measures were written, not a run
of coincidences: the easy return when a measurement cannot be made is the
one that blames the platform, and it is wrong whenever the input is what is
lacking.

## Amendment (2026-09-24): clause 3's single extractor becomes one per solver

Clause 3 reads "`core/io/lsdyna_run.py` is the single extractor. There is no
protocol, registry, or plugin seam (ADR-0010 stands)." ADR-0068 keeps the
second sentence and amends the first: there is one extractor **per solver**,
selected by a normalised lookup on `Provenance.solver_name`, and the
solver-neutral record types remain the whole interface. No protocol, seam or
class hierarchy is introduced, so ADR-0010 still stands — what changed is only
that its deferral condition ("at least two concrete solvers") was met.

The amendment is forced rather than aspirational. Measured on a real Abaqus
input, the single extractor reported `time_integration='explicit'` for a
`*Static` job and `databases_requested=frozenset()`, failing
`input_requests_required_evidence` against a correct deck. A single extractor
is only safe while there is a single solver, and that stopped being true.
