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
