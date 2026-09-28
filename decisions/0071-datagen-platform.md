# 0071 — The Abaqus data-generation pipeline becomes a StructBench capability

**Status**: Proposed
**Type**: Durable
**Date**: 2026-09-27
**Amends**: ADR-0069 (the four-stage pipeline as shared scripts)

## Context

ADR-0069's pipeline generated, ran, exported, converted, verified and
dry-archived a 500-case Abaqus dataset end to end with no pipeline failure.
The dataset itself took four production versions. The causes were a severity
limit chosen after production, an energy term the deck did not request, a mesh
chosen before its convergence was measured, a contact enforcement that created
energy without a solver warning, a fix checked on one axis and not the other,
and a disk that filled. One of these is now caught by the verification
instrument (ADR-0066 energy-account note); the rest were caught by people.

The maintainer's goal (2026-09-27) is a data-generation capability of the
platform: a user who arrives with Abaqus and a problem finds the whole
pipeline, a template that says what a dataset must supply, and a gate that
refuses a production sweep until the pre-production checks have passed. The
design is `docs/plans/2026-09-27-abaqus-datagen-platform-design.md`.

## Decision

1. **The pipeline moves into the package** as `structbench.datagen`
   (solver-agnostic: sampling, generation, running, following, preflight,
   archive, card, template) and `structbench.datagen.abaqus` (deck writers and
   the exporter), with one console entry point, `structbench-datagen`. The
   exporter stays a standalone Python 3.10 file that imports nothing from the
   package, shipped as package data and handed to `abaqus python`. The
   readers and adapter stay in `core/io`. `ARCHITECTURE.md`'s rule that
   nothing under `data_generation/` is importable is amended: that directory
   keeps only non-importable glue, and the layering gains `datagen` beside
   `eval` and `benchmarks`.
2. **A dataset is a definition with a stated contract**: `dataset.toml` with
   the tables the design lists (today's `sweep.toml` plus `[levels]`,
   `[pilot]`, `[qoi]`, `[limits]`) and `problem.py` (today's `model.py`) with
   `input_deck`, `feasible`, `mesh` and `qoi`. The files are named for what a
   user opens them for: what the dataset is, and how one case is built and
   read.
   `structbench-datagen new` scaffolds it with every field explained;
   `structbench-datagen check` validates it without a solver. Every solver
   keyword goes through the writer library; a dataset never edits deck text.
3. **Preflight is a stage and a gate.** It runs the pilot split through
   conformance, deck regression, feasibility, the mesh levels with QoI
   extrapolation and field metrics, the energy-account requirements at every
   level, a time-and-disk budget, and the verification instrument, and writes
   a report and a stamp of the definition's hashes. `generate` refuses a
   production split without a passing stamp for the current definition; an
   override exists and is recorded in the provenance of every case it
   produces.
4. **The convergence engine is solution verification** and lives in
   `structbench.verification.convergence`: nested-node matching, field
   restriction, Richardson extrapolation and GCI with observed-order
   statuses, pooled relative L2 per field, and the stress breakdown. Datasets
   supply only their QoI function and level convention.
5. **The runner budgets and stops cleanly**: a free-space check before each
   launch, an estimate of the sweep from the pilot measurements, a clean stop
   with its own exit code when the environment fails, and a `follow` stage
   that exports and converts finished cases while a run proceeds.
6. **Three verification changes**, each with an ADR-0066 note:
   `input_requests_required_evidence` extended to Abaqus decks; the hourglass
   rows measured from the stored hourglass global; the sourced 1 % level
   (B-BLM-1, provisional) ratified for `energy_gain_max` and `energy_loss_max`
   as an indicator whose exceedance is `review`, not as a requirement.
7. **A user guide and a public example**: `docs/DATA_GENERATION.md`, the
   conformance document under `docs/datagen/`, and one public example
   definition (the single-rod conformance case) that the scaffold, the docs
   and the env-gated acceptance test share. No private dataset's box, splits
   or constants enter the repository.
8. **Dataset A migrates without a re-run**: its definition adopts the
   contract and its decks must rebuild byte for byte against their recorded
   hashes.

## Alternatives considered

- **Keep the scripts where they are and document them.** Rejected: the
  imports, the missing entry point and the absent template are what make the
  pipeline unusable to anyone but its authors.
- **A warning instead of a gate.** Rejected: three of the four versions were
  run by people who knew the checklist. A gate that re-arms when the
  definition changes is the mechanism the lessons call for; the override
  keeps it from blocking legitimate work, and its record keeps it honest.
- **Convergence code per dataset.** Rejected: nothing in it is
  dataset-specific except the QoI function; a private copy is why the mesh
  was decided after production.
- **Energy gain as a requirement with a float32 tolerance.** Rejected: healthy
  explicit runs gain small amounts legitimately, and a tolerance chosen to
  pass ours would be fitted to the test bed, which ADR-0066 clause 7 forbids.
  The sourced level as an indicator flags the failure mode seen and leaves the
  call to a person.

## Consequences

- StructBench gains a second kind of user: one who produces data with it.
  The dataset contract and the template are public API and change by ADR.
- The pre-production cost of a dataset rises by the preflight's runs (a
  pilot split at three mesh levels); it replaces the cost of re-running
  production, which for dataset A was three times its size.
- Dataset B's solver-side needs (combined hardening, self-contact, a point
  mass, backstress and plastic-strain fields, a new material class) become
  separate ADRs against a stable pipeline rather than changes to it.
- ADR-0069 remains the record of the stage designs; this ADR changes their
  home, their entry point, and adds the gate, the engine and the template.

## Note 2026-09-27 — the preflight examines three resolutions, not one

Clause 3 names the mesh levels. The maintainer's instruction of 2026-09-27
widens the preparation stage to the three resolutions a stored trajectory has;
each is probed on the pilots, measured, and declared in the data card before
production:

- **Space** — the mesh levels of `[levels]`, as clause 3 says: QoI
  extrapolation and field metrics across nested levels, the production level
  chosen against them.
- **Time** — two settings. The solver's increment: the energy account is
  checked at more than one stable-increment scale, because a contact
  enforcement that creates energy at the full increment and not at half of it
  is a resolution defect the pilots can show (dataset A found it after
  production). The output sampling: one pilot is exported at a finer frame
  interval and the stored fields, globals and QoIs are compared at the common
  instants, so the frame interval is a measured choice rather than an inherited
  one.
- **Duration** — the stored horizon against the response's settling: on every
  pilot the preflight measures when the quantities of interest stop changing
  (and, where the problem has one, when contact ends), reports the share of the
  horizon that lies after it, and requires the slowest pilot to settle inside
  the horizon with margin. The card declares the horizon and the scored
  horizon (fixed, or per case from a stored global); a tail kept beyond
  settling is for property tests, not accuracy. Dataset A's horizon was
  inherited from an early probe, and most of its trajectory turned out to be
  elastic ringing after the rod had left the wall.

The mechanism — which `[pilot]` fields name the increment scales, the
frame-interval probe and the settling rule — is part two's to design; the
design document's `preflight` step 4 carries the sketch and its open points.

## Note 2026-09-27 — the `validate` stage is `verify`

ADR-0072 gives validation its meaning (comparison with experiment) and
renames this pipeline's `validate` stage, which runs the ADR-0066
verification instrument, to `verify`; the old name is refused with a
pointer. Clause 1's stage list reads accordingly.

## Note 2026-09-27 — part two (a) built

Clause 4 is delivered: the convergence engine lives in
`structbench.verification.convergence` and `structbench-datagen converge`
pairs a sweep's runs across run roots, extrapolates each quantity of interest
and measures every stored field of each coarser level against the finest,
writing a byte-stable record. With it: the headline metric moved into
`verification.kernels` (`eval.metrics` re-exports it), the dataset-level
measurement helpers moved into `verification.dataset` so `datagen` no longer
imports `cli`, `problem.py` may import its siblings, and `[levels].symmetry`
names the volume weights. The study's stress-breakdown diagnostic (contact
phase, post-release, time shift) is not generic and stays out. Part two (b)
owes the preflight and its stamp, `generate`'s gate, `run` hardening and
`follow`.

## Note 2026-09-28 — part two (b) built

Clause 3's gate is delivered as `structbench-datagen preflight`: the pilot
split's points become a case set — every pilot at every `[levels].pilot`
level but the finest (the `fine_cases` there too), the pilots at production
with the solver increment scaled, one pilot exported at a finer frame
interval, one conformance run with every energy term requested — run in a
sub-sweep of the work root, and ten steps turn the records of `run`,
`export`, `convert`, `verify` and `converge` into verdicts in ADR-0066's
vocabulary: deck regression, feasibility, conformance, the three resolutions
of the 2026-09-27 note (space, increment, frame, duration), the energy
account, the budget and the verification instrument. The stamp
(`preflight-stamp/1`) binds the verdicts to the sha256 of both definition
files; `generate` refuses a split that is not a probe without a passing
stamp for the current definition, and `--no-preflight` records the
omission in provenance (now `abaqus-provenance/3`, with `preflight` and
`probe` keys). The `[pilot]` fields the note left open are settled with
defaults (`increment_key`, `increment_factors`, `frame_key`,
`frame_count_key`, `frame_factor`, `frame_tolerance`, `settling_margin`,
`contact_force_global`) and `[qoi].tolerance` is the one target the space,
increment and settling rules read. `run` stops launching below the
free-space margin with exit 3 and prints the stamp's estimate; `follow`
exports and converts finished cases beside a running sweep; `converge
--anchor` compares groups that have runs from probe splits only. The
temporal measures live in `verification/temporal.py`. The whole chain is
tested end to end against a fake solver script; the first Abaqus run of the
stage on a real dataset is the maintainer's to schedule. Part three owes the
Abaqus `input_requests_required_evidence` and hourglass rows, the
energy-gain indicator's ratification, the card generator and the full guide.

## Note 2026-09-28 — part three (a) built: a gate a real dataset can pass

The first real preflight, on the maintainer's first Abaqus dataset, could not
pass whatever the data's quality. The frame step judged every stored global,
and a rigid-wall reaction force or a strain energy is never resolved at a
practical clock; the space step's `review` on a quantity that barely moves
between levels had no way to be accepted; and the budget estimated, from pilots
at the box's corners, a production sweep that already existed. Plan 3a
answers each with a declaration that travels with its reason. `[pilot].
frame_reported` names the stored fields the frame step reports rather than
judges; an empty table judges every stored field, including stress and
acceleration, which is stricter than plan 2b's built-in acceleration exception,
now removed (the shipped example declares acceleration). `[pilot].
accepted_reviews` records a person's acceptance of a review, keyed
`<step>.<name>`; it rescues a review only, never a fail or a missing
measurement, and the stamp lists accepted, unaccepted and unused acceptances
so its verdict is recomputable from the stamp alone. The budget sizes only the
production cases not yet completed, from their own meshes at the pilots' rates.
`preflight --rejudge` re-evaluates every step on the runs already made, with no
solver, when the case set is the planned one and every deck and every unit
label is reproduced by the current definition; the stamp records the hashes
the runs were generated under beside its own.
