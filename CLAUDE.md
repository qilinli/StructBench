# CLAUDE.md

*The operational manual for Claude Code working on this project. Read at the start of every session.*

---

## Purpose

This file is the entry point for Claude Code sessions on StructBench. Most rules and conventions live in other documents; this file points to them and covers the pieces that don't have a home elsewhere.

HARNESS.md carries the principles (*why* rules exist); this file carries the mechanisms (*what* to do). When editing either, content drifting across that boundary should be routed back to its proper home. For what the project is, see VISION.md.

---

## Project snapshot

*Where the project stands, with pointers to the files that hold the detail. Keep it that way: build notes, measured counts, fix narratives and open-item lists go in the ADR notes, the published records, the README Roadmap and commit messages. Update this section when a release ships or an ADR changes status, not after every build.*

StructBench is an open platform for establishing whether learned surrogate models of structural response can be trusted: benchmark problems with reference data of declared uncertainty, property tests and evaluation protocols, and reference models that show the tested properties are attainable (ADR-0065, Accepted 2026-09-15, superseding ADR-0014's substrate framing). `docs/VISION.md` keeps its earlier accuracy-comparison wording until the maintainer applies the reviewed rewrite out of session. ADR-0065's four follow-ups (properties block in the registries, card fields and compliance table, task redefinition with a restart family, v0.4 roadmap reframe) are not yet drafted; existing benchmarks and registered results stand. Run by Qilin Li (human) and Claude Code (agent) together under the philosophy in HARNESS.md.

**Released: v0.3.0** (2026-08-27; v0.2.0 2026-08-06, v0.1.0 2026-07-09). Four benchmarks (Taylor 2D, wave-1D, notch-impact, and the 3D `DeformingPlate` on public MeshGraphNets data) and four native model families (CGN, MGN, Transolver, GeoFLARE) under one pipeline (`structbench-train`) and one protocol. Relative L2 is the headline metric (ADR-0055), and prediction schemes are an explicit axis. Blessed and provisional baselines are in README.md; per-benchmark leaderboards are on the landing pages in `docs/benchmarks/`. The three LS-DYNA archives are public on Hugging Face (`StructBench/*`, data tag `v0.1.0`, ADR-0040); DeformingPlate is download-and-convert (ADR-0042); blessed checkpoints stay in the gitignored `models/`.

**In progress since v0.3**: three strands of the ADR-0065 identity. Their open items are in the README Roadmap.

- **Verification of reference data** (ADR-0066, Accepted; material classes ADR-0067, Accepted, and ADR-0070, Proposed). `structbench.verification` measures a simulation run against a catalogue of quantities and judges the measurements; `measure` sets no thresholds and `judge` never touches data. CLI: `python -m structbench.cli.datacheck measure|judge`. Published records: `docs/datachecks/`.
- **Data generation** (Abaqus as second solver ADR-0068, pipeline ADR-0069, platform ADR-0071; all Proposed, ADR-0071 built through part three (a)). `structbench-datagen` runs the stages `new check preflight generate run follow export convert verify converge archive`; `generate` refuses a production split without a passing preflight stamp (`--no-preflight` records the omission). A dataset is a definition (`dataset.toml` + `problem.py`) passed by path, never code in the repository. Guide: `docs/DATA_GENERATION.md`; conformance blocks and lessons in `docs/datagen/`.
- **Validation against experiments** (ADR-0072, Proposed; built). `structbench-validate` measures canonical cases and a reference-experiment set (the first is `taylor_copper`) with the same functions and writes the deviations.

Package layering and module responsibilities: `docs/ARCHITECTURE.md`.

---

## Session workflow

### Starting a session

Read these files, in order, before any work begins:

1. `CLAUDE.md` (this file).
2. `docs/VISION.md`.
3. `docs/HARNESS.md`.
4. `docs/PRINCIPLES.md`.
5. `docs/CORRECTIONS.md` — all entries marked `active`.
6. `decisions/README.md` — the ADR index.
7. `docs/WORKFLOW.md` — session venues and multi-machine git workflow; identify your venue before making any change.

Then, conditionally based on the session's task:

- `docs/ARCHITECTURE.md` — if the task touches the package structure, module interfaces, or the case schema.
- Specific ADRs from the index — whichever are relevant to the task.
- The Roadmap section of `README.md` — if the session is about planning or scoping.
- The maintainer's private research documents — kept **outside the repository**, never tracked; consulted only when the task touches scope, benchmark admission, or the roadmap, and only when present (the maintainer points the session at them). *Context-only: they guide the maintainer's research, of which StructBench is one instrument, and do not define StructBench scope — that is `docs/VISION.md` and the ADRs alone (ADR-0065).*

If the session's task is not clear from the opening message, ask the human what the session is for before beginning work.

Target: the full start-of-session reading should take under 10 minutes of agent time.

### During a session

- **Default to asking when ambiguous.** Silent resolution of ambiguity is how invariants erode.
- **Draft ADRs immediately when decisions are made**, not at end of session.
- **Flag scope expansion.** If the task has grown beyond what was originally requested, say so.
- **Break complex work into checkpoints.** Pause for confirmation at natural boundaries.
- **When corrected, ask whether to log to `CORRECTIONS.md`.**

### Ending a session

- Commit changes to a feature branch; `main` moves only on the human's explicit in-session instruction (ADR-0023).
- Unfinished work persists as `WIP:`-prefixed commits on a feature branch, or as dated notes in `scratch/`.
- After session end, a reader of the repo files (without chat history) should be able to reconstruct what was decided and what was done.
- No formal session summary required; commit messages serve as the record.

---

## Authority tiers

Four tiers govern what Claude Code can do. When in doubt, default to the more restrictive tier.

### Unilateral — do without asking

- Writing, refactoring, or deleting code within existing modules, if the public API doesn't change and no dependencies are added.
- Writing or updating tests.
- Running tests, linters, formatters, the CLI.
- Creating or modifying docstrings and code comments.
- Fixing obvious bugs with local fixes.
- Installing already-approved dependencies in a local environment.
- Reading any file in the repo.
- Writing scratch notes in `scratch/` (gitignored).

### Flag-first — propose and wait for confirmation

- Adding, removing, or upgrading any dependency.
- Modifying the public API of any module.
- Modifying the case schema in any way.
- Creating new top-level modules or new files at repo root.
- Drafting or modifying an ADR (the human finalises).
- Architectural changes affecting how modules interact.
- Deletions exceeding ~50 lines of non-trivial code.
- Running anything with real compute cost — propose with estimated cost and runtime.
- Changes to git state affecting history.

### On explicit instruction — execute when the human directs it in-session

*(Added by ADR-0023, amending ADR-0006.)* These never happen as part of unprompted work — `main` moves only by the human's word — but when the human explicitly instructs them in the session ("merge it", "push"), Claude Code executes them directly instead of handing commands back.

- Merging a feature branch into `main`.
- Pushing to the remote.
- Committing directly to `main`.
- Hugging Face data-release actions on the `StructBench/*` dataset repos — create a repo, upload, flip public, create a *data* tag (ADR-0023, amended 2026-08-28). The token stays the human's: they log in, Claude only invokes the CLI.

### Forbidden — refuse even if asked in-session

These require deliberate human action outside a normal coding session.

- Publishing *code* releases — GitHub releases, code version tags, uploading to PyPI or Zenodo. (Dataset releases on Hugging Face are on-instruction, above.)
- Modifying `LICENSE`, `HARNESS.md`, or `VISION.md` during a coding session.
- Rewriting git history on shared branches.
- Accepting or merging third-party pull requests.
- Changing repository settings.
- Handling secrets, credentials, API keys, or SSH keys.
- Asserting facts about external sources without verification.

---

## Corrections handling

Small corrections that don't warrant an ADR are logged in `CORRECTIONS.md`. Format and workflow specified in that file's header. In-session behaviour:

- When the human corrects something that could plausibly recur, ask: *"should I log this to `CORRECTIONS.md`?"*
- On confirmation, add the entry before continuing.
- Active entries are read at session start and inform behaviour throughout the session.

---

## Where other rules live

- **Coding conventions** (Python version, style, testing, documentation, logging, git): `docs/PRINCIPLES.md`.
- **Repository structure and package layout**: `docs/ARCHITECTURE.md`.
- **Case schema**: `docs/ARCHITECTURE.md`.
- **Dependency policy and approved list**: `docs/PRINCIPLES.md`, with individual additions recorded as ADRs.
- **ADR format and process**: `decisions/README.md`.
- **Session venues and multi-machine git workflow**: `docs/WORKFLOW.md`.
- **Planning, work in progress and open items**: the Roadmap section of `README.md`.
- **Data generation (the dataset contract and the stages)**: `docs/DATA_GENERATION.md`.

If a rule seems missing from all of these, flag it rather than guess. It may belong in one of the existing documents, or it may indicate a gap the harness doesn't yet cover.

---

## Common situations

**The human asks me to do something that crosses into flag-first territory.** Propose with reasoning, wait for confirmation.

**The human asks me to do something forbidden.** Refuse, explain why, suggest the correct out-of-session path.

**I want to do something the rules don't cover.** Flag the ambiguity. The resolution is either a quick human answer or a new `CORRECTIONS.md` entry.

**I realise mid-task that the approach is wrong.** Stop, say so, propose an alternative. No silent pivots.

**The rules here conflict with `VISION.md` or `HARNESS.md`.** The philosophical documents take precedence. Flag the conflict so this file can be revised.

**The rules here feel wrong for the current task.** Flag it. Do not override silently. If the rule is genuinely a bad fit, the path is to revise this file.
