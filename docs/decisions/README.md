# Decisions

This folder holds the project's Architecture Decision Records (ADRs): one file per decision, with its context, the alternatives considered and its consequences. Together they are the project's decision history, and each one is a constraint on everything decided after it.

---

## What earns an ADR

A record constrains future work, so the log grows only when both of these hold:

1. **The decision binds what comes after it.** Scope and identity; a public contract (the case schema, a file format, the dataset definition, a command); a benchmark protocol that fixes the comparability of registered results; what counts as a pass or as trustworthy; what is published and on what terms; a dependency or a layering rule.
2. **The decision needs the maintainer's judgment.** It rests on context the agent cannot see, its consequences reach beyond the repository, or it is expensive or impossible to reverse.

The test is: *would a future session, without this record, make a choice the maintainer would reject?* If not, there is no ADR. Everything else has another home and goes there:

| Instead of an ADR | Home |
|---|---|
| how something is implemented | the code and its docstrings |
| a design, a plan, a build log | `docs/plans/`; commit messages |
| a training recipe, a run, an experiment arm | the run config; the results registry |
| a measured fact about the data | the published record (`docs/datachecks/`, the benchmark card) |
| a small correction to how the agent works | `docs/CORRECTIONS.md` |
| a change to a decision already recorded | a dated note on that record |

Why the bar is where it is: the log was reviewed in full on 2026-09-29 (ADR-0073). It held 71 records and 86,000 words, most of them accepted unread because they were long, and several contradicted each other or the shipped code. Every record added is a constraint future work must honour and a chance for two constraints to clash; fewer, shorter records are easier to keep true *(maintainer, 2026-10-01)*.

## Shape

An ADR is one page: about 800 words before any appended note, and the test suite refuses a new record over 900. Right after the header it opens with the block the maintainer accepts from:

```
**Your call**: the one to three judgments the maintainer makes by accepting
this record, one line each, with what reversing each would cost.
```

Every judgment the maintainer is accountable for is named in that block. A call that appears only in the body is a call the maintainer has not made. The Decision section states the decision in numbered clauses, plainly. Design detail, measurements and rationale longer than a paragraph go to a linked plan under `docs/plans/`.

Before proposing, the drafter searches the log for every record the decision touches and names them in the `**Amends**:` header line. A clash with an accepted record is resolved in the new one, by amending or superseding the clause it clashes with, never left for a reader to find.

---

## Format

Each ADR is one markdown file with the following structure:

```
# NNNN — Title

**Status**: Proposed | Accepted (maintainer, YYYY-MM-DD) | Superseded by NNNN
**Type**: Ephemeral | Durable
**Date**: YYYY-MM-DD
**Amends**: the records this one changes or constrains, or "none"

**Your call**: see Shape above.

## Context
What problem or question prompted this decision.

## Decision
What was decided, in numbered clauses.

## Alternatives considered
What else was on the table, and why not.

## Consequences
What becomes easier, harder, or constrained as a result.
```

### Filenames

`NNNN-kebab-case-title.md`, where NNNN is a zero-padded sequential number (0001, 0002, ...). Numbers are never reused, even when decisions are superseded.

### Status

- **Proposed** — drafted, not yet approved by the maintainer. Work may be built on a branch while a record is Proposed, but no Accepted record may depend on it, and a record whose build is merged is accepted or withdrawn in the same session. Proposed is not a parking state.
- **Accepted** — the maintainer has read the Your-call block and the Decision and approved them; the status line carries the date.
- **Superseded by NNNN** — replaced by a later record. The superseded record is kept for history; the new one references it.

### Type

- **Ephemeral** — the default. Expected to change as the project evolves; updated in place with a dated note, and supersession is not required.
- **Durable** — only for commitments that are expensive or impossible to reverse: the case schema and format compatibility, what is published and under which licence, a benchmark protocol behind registered results, the project's identity and scope, and the harness itself. Revising one requires a new superseding record with explicit reasoning. Small adjustments that do not reverse the decision (timing slips, parking a sub-item) may instead be recorded as a dated amendment note and reflected in the index Status column *(maintainer, 2026-08-06)*; supersession remains required for genuine reversals.

### Notes

A dated note is appended only when it changes the decision (an amendment), records its outcome (a verdict), or points at a later record that changed it. Build logs, measured counts and narratives do not go in notes; they go in commit messages, the plans or the published records. A record carrying more than three notes, or grown past twice the page, is consolidated: a short successor states the current decision and supersedes it.

### Index and moves

The index row's Status column names every later record that amended, narrowed or superseded the row's, so a reader who opens a record from the index knows when it is not the whole story. When a file the log cites moves, the record that cites it gets "(moved YYYY-MM-DD; was `old path`)" at the first mention; records are history and are not rewritten.

## Review

At each release, or every fifteen records, the whole log is triaged: index rows against the files, text that is no longer true, labels, clashes, and calls the maintainer never made. The triage's decisions go in one record with a dated pointer on each record it touches, as ADR-0073 did.

---

## Index

| # | Title | Type | Status |
|---|-------|------|--------|
| 0001 | Adopt harness engineering methodology | Durable | Accepted |
| 0002 | Project name is StructBench | Durable | Accepted |
| 0003 | v0.1 anchor problem is impact on RC beams | Durable | Superseded by 0015 |
| 0004 | Platform is solver-agnostic; LS-DYNA for v0.1 data generation | Durable | Accepted |
| 0005 | ADR format and decision-log structure | Durable | Accepted |
| 0006 | Three-tier authority model for Claude Code | Durable | Accepted (amended by 0023) |
| 0007 | CORRECTIONS.md mechanism for small corrections | Durable | Accepted |
| 0008 | Principle/mechanism separation between HARNESS and CLAUDE | Durable | Accepted |
| 0009 | Session-start reading list | Ephemeral | Accepted |
| 0010 | FEM solver code lives outside the importable package | Durable | Accepted (amended by 0073) |
| 0011 | Case vocabulary for the data record | Durable | Accepted |
| 0012 | Case schema field-level structure | Durable | Accepted (amended by 0073) |
| 0013 | HDF5 persistence layout for the case schema | Durable | Accepted |
| 0014 | StructBench is the substrate layer of a broader research program | Durable | Superseded by 0065 |
| 0015 | v0.1 ships existing LS-DYNA datasets as benchmarks with prior-paper GNN baselines (supersedes 0003) | Durable | Accepted (amended by 0021, 0024) |
| 0016 | LS-DYNA d3plot is the canonical ingestion path; general adapter on lasso-python | Durable | Accepted |
| 0017 | Relationship to NVIDIA PhysicsNeMo: independent substrate, opt-in model-edge interop | Durable | Accepted |
| 0018 | PyTorch + PyG are hard runtime dependencies of the ML layer | Durable | Accepted |
| 0019 | v0.1 Taylor 2D benchmark: autoregressive surrogate task, split, and eval protocol | Durable | Accepted (amended by 0032, 0035, 0055) |
| 0020 | Native radius_graph; no graph-backend binary dependency | Durable | Accepted |
| 0021 | v0.1 narrows to Taylor 2D; portfolio spreads across releases (amends 0015) | Durable | Accepted |
| 0022 | FEM-convention visualization harness (`viz/`, matplotlib as optional extra) | Durable | Accepted |
| 0023 | Git authority: `main` moves on explicit in-session instruction (amends 0006) | Durable | Accepted (amended 2026-08-28: Hugging Face data-release actions are on-instruction) |
| 0024 | v0.2 ships the 1D wave and notch-beam benchmarks; RC beam moves to v0.3 | Ephemeral | Accepted (amended 2026-08-06: notch-bend parked; v0.3 scope superseded by 0041, 2026-08-07; notch-bend excluded and removed by 0056) |
| 0025 | Wave 1D benchmark: task, split, and eval protocol | Durable | Accepted (amended by 0055) |
| 0026 | Notch-beam 2D benchmark pair: two benchmarks, tasks, splits, eval | Durable | Accepted (amended by 0029, 0039, 0055, 0056; amendments finalised by 0073) |
| 0027 | Benchmark cards: typed per-benchmark metadata with generated views | Durable | Accepted (amended by 0032) |
| 0028 | GNS baseline training-recipe rework after the first full run | Ephemeral | Accepted |
| 0029 | Notch-beam aux is max principal strain, not K&C damage (amends 0026) | Durable | Accepted (amended in place 2026-08-06: 0.01 threshold declared, provisional flag resolved) |
| 0030 | Concrete-Beam decks are kg-mm-ms; canonical data patched in place | Durable | Accepted |
| 0031 | Data archive layout: canonical/raw mirrors named by benchmark | Ephemeral | Accepted (amended by 0037; the bend archive removed by 0056) |
| 0032 | Grouped run configuration and benchmark-protocol governance (amends 0019, 0027) | Durable | Accepted (amended by 0035) |
| 0033 | Official baseline results live in per-benchmark results registries | Durable | Accepted (amended by 0037; extended by 0046) |
| 0034 | The reference baseline is CGN (Concrete Graph Network, Li et al. 2023) | Durable | Accepted |
| 0035 | The model input window is the rollout init; no history backfill (amends 0019, 0032) | Durable | Accepted (amended by 0053; confirmed by 0073) |
| 0036 | Per-benchmark landing pages: one generated docs page per benchmark (extends 0027) | Durable | Accepted (extended by 0046) |
| 0037 | Blessed runs archive: `models/` mirror and registry checkpoint pointers (amends 0031, 0033) | Durable | Accepted |
| 0038 | Auxiliary-channel training knobs: target-space transform and tail weight | Ephemeral | Accepted |
| 0039 | Notch-impact scored horizon: 250 µs evaluation window, matched baseline recipe | Durable | Accepted |
| 0040 | Dataset hosting: maintainer's OneDrive stays the master; archives public on Hugging Face (was: shared on request) | Durable | Accepted (amended 2026-08-28: public Hugging Face mirror; 2026-09-23: data tag v0.1.1, and the on-request promise needs ADR-0068 clause 6 for Abaqus) |
| 0041 | v0.3 pivots to a public multi-method benchmark: DeformingPlate with native MGN/Transolver/GeoFLARE (supersedes ADR-0024's v0.3 scope) | Durable | Accepted (amends 0034; corrected in place 2026-08-07 re schema, see 0042; comparison reframed as a use, not the purpose, by 0065) |
| 0042 | Schema 0.2.0 adds per-node fields; nodal-FE ingestion via download-and-convert (deforming_plate) | Durable | Accepted (corrects 0041) |
| 0043 | DeformingPlate benchmark protocol: task, split, eval, and the MGN blessing gate | Durable | Accepted (narrowed by 0046; amended by 0073) |
| 0044 | Transolver provisional adaptation: native Physics-Attention on the DeformingPlate rollout | Durable | Accepted (AR superseded as Transolver's native scheme by 0054; clause 14 settled by 0073) |
| 0045 | GeoFLARE provisional adaptation: native GALE_FA (GeoTransolver + FLARE) on the DeformingPlate rollout | Durable | Accepted |
| 0046 | Provisional results and the method-comparison table (closes ADR-0041 clause 4) | Durable | Accepted (amended by 0073) |
| 0047 | Taylor 2D multi-method extension: native MGN/Transolver/GeoFLARE on the SPH benchmark | Ephemeral | Accepted |
| 0048 | Notch-impact multi-method extension: native MGN/Transolver/GeoFLARE on the notched-beam SPH benchmark | Ephemeral | Accepted |
| 0049 | Taylor native recipe repair: noise rescale, velocity history, MGN stretch gate | Ephemeral | Accepted |
| 0050 | Prediction-scheme axis: unified k-frames-per-call (autoregressive ↔ bundled ↔ one-shot) | Durable | Superseded by 0051 (source corrections 2026-08-15) |
| 0051 | k-frames-per-call implementation (Transolver): resolved decisions, neural-CFL, pushforward | Durable | Accepted (amended 2026-08-15: one-shot impact-velocity conditioning; time-query scheme recorded) |
| 0053 | Decouple model history (`history_frames`) from the `input_frames` seed / scored-span protocol | Durable | Accepted |
| 0054 | Transolver time-conditioning: the native non-autoregressive prediction scheme | Durable | Accepted |
| 0055 | Relative-L2 as the headline metric (amended: headline, RMSE retained secondary) | Durable | Accepted (amended by 0073) |
| 0056 | Descope notch-bend: the notch-beam benchmark narrows to notch-impact (redundant with impact; amends 0024/0026) | Durable | Accepted (amended 2026-09-23: excluded, not parked) |
| 0057 | Transolver++ eidetic-state adaptation (adaptive temperature + train-only Gumbel Rep-Slice) on the Transolver family | Durable | Accepted (0073) |
| 0058 | `huggingface_hub` as an optional `data` extra | Durable | Accepted (2026-10-01) |
| 0059 | Auxiliary state channels: `aux` generalises from `(T, P)` to `(T, P, C)` | Durable | Accepted |
| 0060 | Aux channels as model inputs: the state-feedback surface (Transolver AR) | Durable | Accepted (2026-10-01; narrowed by 0062; line closed by 0064) |
| 0061 | State-feedback stability: input noise and pushforward on the state channel | Ephemeral | Accepted |
| 0062 | Anchored flow map: state-anchored time-conditioned prediction (Transolver) | Ephemeral | Accepted (verdict note 2026-09-12) |
| 0063 | Anchor-interface contraction training: flow-map pushforward chains + kinematic-anchor noise | Ephemeral | Accepted (amended 2026-09-10; verdict note 2026-09-12) |
| 0064 | Constitutively-structured admissible heads: return-map decoder structure (D2/D3 by construction) + consistency-hinge comparator | Ephemeral | Accepted (note by 0073) |
| 0065 | StructBench is a verification-and-validation platform for learned surrogates (supersedes 0014) | Durable | Accepted (VISION.md rewrite pending, maintainer out-of-session; data standard fixed by 0073) |
| 0066 | Reference-data verification: the `verification/` module | Durable | Accepted (reviewed by 0073) |
| 0067 | Material classes for the notch sweep: K&C concrete and bilinear steel | Durable | Accepted (note by 0073) |
| 0068 | Abaqus is the second solver; the deferred abstraction question is answered (amends 0066 clause 3) | Durable | Accepted (2026-10-01; clause 8 lifted for one row by 0073) |
| 0069 | The Abaqus data-generation pipeline (four stages, shared scripts, `abaqus-npz/1`, `datagen` extra) | Durable | Accepted (2026-10-01; amended by 0071) |
| 0070 | Material class `elastic_plastic_isotropic` (Abaqus `*PLASTIC`, isotropic, no EOS) | Durable | Accepted (2026-10-01) |
| 0071 | The Abaqus data-generation pipeline becomes a StructBench capability: `structbench.datagen`, the dataset template and `check`, the preflight gate, the convergence engine in `verification`, runner budget and `follow` (amends ADR-0069) | Durable | Accepted (0073) |
| 0072 | Validation against experiments: `structbench.validation`, reference-experiment sets with provenance, measures shared by experiment and simulation, a record that reports deviations and never judges; the datagen stage `validate` renamed `verify` | Durable | Accepted (0073) |
| 0073 | Decisions from the maintainer's review of the decision log (amends 0010, 0012, 0026, 0035, 0043, 0046, 0055, 0064, 0065, 0066, 0067, 0068, 0071, 0072; accepts 0057, 0071, 0072) | Durable | Accepted |

---

## Adding a new ADR

1. Check the bar above. If the decision does not meet it, record it in its
   home instead.
2. Claim the next available number by checking the highest NNNN in use.
3. Create `NNNN-kebab-case-title.md` using the format above, and write the
   Your-call block first.
4. Search the log for the records the decision touches; name them in the
   `**Amends**` line and resolve any clash in the new record.
5. Draft the rest within the page. Claude Code may draft; the maintainer
   finalises by reading the Your-call block and the Decision.
6. Add a row to the index in this README.
7. Commit with a message like `docs: add ADR-NNNN on <title>`.
