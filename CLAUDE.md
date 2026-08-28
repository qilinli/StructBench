# CLAUDE.md

*The operational manual for Claude Code working on this project. Read at the start of every session.*

---

## Purpose

This file is the entry point for Claude Code sessions on StructBench. Most rules and conventions live in other documents; this file points to them and covers the pieces that don't have a home elsewhere.

HARNESS.md carries the principles (*why* rules exist); this file carries the mechanisms (*what* to do). When editing either, content drifting across that boundary should be routed back to its proper home. For what the project is, see VISION.md.

---

## Project snapshot

StructBench is an open platform for data-driven structural engineering — benchmarks, reference models, and eventually deployment tools. Run by Qilin Li (human) and Claude Code (agent) together under the philosophy in HARNESS.md.

Current stage: **v0.3.0 shipped** — tagged 2026-08-27 (v0.2.0 2026-08-06, v0.1.0 2026-07-09); the GitHub release from `scratch/2026-08-27-v0.3.0-release-notes.md` is the human's remaining out-of-session action. The public repo carries four benchmarks — Taylor 2D (ADR-0019), wave-1D (ADR-0025), notch-impact (ADR-0026, 250 µs scored horizon ADR-0039; notch-bend descoped ADR-0056), and `DeformingPlate` (ADR-0041/0042/0043 — the first 3D benchmark, on public MeshGraphNets data) — and four native model families (CGN, MGN, Transolver with an off-by-default Transolver++ variant per ADR-0057 Proposed, GeoFLARE) under one pipeline and one protocol, with cross-method leaderboards on Taylor, notch-impact, and DeformingPlate (wave-1D's registry is CGN-only). Blessed baselines: CGN on Taylor (s1), wave-1D (x1-s1), notch-impact (h250c-s1); MGN on DeformingPlate (noise-fixed, reproduces the published band); Transolver/GeoFLARE ship provisional (ADR-0046), and the DP registry is a family × scheme matrix. Relative L2 is the headline metric (ADR-0055); prediction schemes are an explicit axis (ADR-0051/0053/0054 — time-conditioning is the operators' native scheme). The substrate is built end to end: canonical case schema + HDF5 I/O in `core/`, the general LS-DYNA adapter (ADR-0016) plus the `tfrecord` ingestion adapter (ADR-0042), benchmark cards with generated landing pages (ADR-0027/0036), grouped run configs and per-benchmark results registries (ADR-0032/0033/0046), and the config-driven pipeline (`structbench-train`). The three LS-DYNA canonical archives are public on Hugging Face (`StructBench/{wave-propagation-1d,taylor-impact-2d,notch-beam-2d-impact}`, dataset tag `v0.1.0`, built by `tools/build_hf_bundle.py`; ADR-0040 amended 2026-08-28 — the maintainer's OneDrive stays the master and on-request sharing continues; DeformingPlate is download-and-convert, not rehosted, ADR-0042); blessed-checkpoint archives live in the gitignored `models/` (private, paths recorded in the cards). After v0.3: RC beam stays deferred, a crash benchmark is a v0.4 candidate gated on public data, and the segmented beam stays parked (ADR-0015). See the Roadmap section of README.md for sequencing.

---

## Session workflow

### Starting a session

Read these files, in order, before any work begins:

1. `CLAUDE.md` (this file).
2. `docs/VISION.md`.
3. `RESEARCH-PROGRAM.md` — *context-only; explains the research program StructBench serves but does not define its scope (see ADR-0014). **Local-only and untracked** (private strategy, 2026-07-02): present on the maintainer's machine but absent from clones — skip without error if missing.*
4. `research/FINDINGS.md` — the private findings index (banked research conclusions with confidence tiers; open the individual `research/findings/F-NNN-*.md` only when a line is relevant). Governance in `research/README.md`. **Local-only and untracked** (same as `RESEARCH-PROGRAM.md`) — skip without error if missing.
5. `docs/HARNESS.md`.
6. `docs/PRINCIPLES.md`.
7. `docs/CORRECTIONS.md` — all entries marked `active`.
8. `decisions/README.md` — the ADR index.
9. `docs/WORKFLOW.md` — session venues and multi-machine git workflow; identify your venue before making any change.

Then, conditionally based on the session's task:

- `docs/ARCHITECTURE.md` — if the task touches the package structure, module interfaces, or the case schema.
- Specific ADRs from the index — whichever are relevant to the task.
- The Roadmap section of `README.md` — if the session is about planning or scoping.

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
- **Long-term trajectory**: the Roadmap section of `README.md`.

If a rule seems missing from all of these, flag it rather than guess. It may belong in one of the existing documents, or it may indicate a gap the harness doesn't yet cover.

---

## Common situations

**The human asks me to do something that crosses into flag-first territory.** Propose with reasoning, wait for confirmation.

**The human asks me to do something forbidden.** Refuse, explain why, suggest the correct out-of-session path.

**I want to do something the rules don't cover.** Flag the ambiguity. The resolution is either a quick human answer or a new `CORRECTIONS.md` entry.

**I realise mid-task that the approach is wrong.** Stop, say so, propose an alternative. No silent pivots.

**The rules here conflict with `VISION.md` or `HARNESS.md`.** The philosophical documents take precedence. Flag the conflict so this file can be revised.

**The rules here feel wrong for the current task.** Flag it. Do not override silently. If the rule is genuinely a bad fit, the path is to revise this file.
