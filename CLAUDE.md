# CLAUDE.md

*The operational manual for Claude Code working on this project. Read at the start of every session.*

---

## Purpose

This file is the entry point for Claude Code sessions on StructBench. Most rules and conventions live in other documents; this file points to them and covers the pieces that don't have a home elsewhere.

HARNESS.md carries the principles (*why* rules exist); this file carries the mechanisms (*what* to do). When editing either, content drifting across that boundary should be routed back to its proper home. For what the project is, see VISION.md.

---

## Project snapshot

StructBench is an open platform for data-driven structural engineering — benchmarks, reference models, and eventually deployment tools. Run by Qilin Li (human) and Claude Code (agent) together under the philosophy in HARNESS.md.

Current stage: pre-v0.1. Foundational documents are being written; v0.1 benchmark problem is being scoped around impact/blast on RC beams. No code has been written yet.

---

## Session workflow

### Starting a session

Read these files, in order, before any work begins:

1. `CLAUDE.md` (this file).
2. `VISION.md`.
3. `RESEARCH-PROGRAM.md` — *context-only; explains the research program StructBench serves but does not define its scope (see ADR-0014)*.
4. `HARNESS.md`.
5. `PRINCIPLES.md` (when it exists).
6. `CORRECTIONS.md` — all entries marked `active`.
7. `decisions/README.md` — the ADR index.

Then, conditionally based on the session's task:

- `ARCHITECTURE.md` — if the task touches the package structure, module interfaces, or the case schema.
- Specific ADRs from the index — whichever are relevant to the task.
- `ROADMAP.md` — if the session is about planning or scoping.

If the session's task is not clear from the opening message, ask the human what the session is for before beginning work.

Target: the full start-of-session reading should take under 10 minutes of agent time.

### During a session

- **Default to asking when ambiguous.** Silent resolution of ambiguity is how invariants erode.
- **Draft ADRs immediately when decisions are made**, not at end of session.
- **Flag scope expansion.** If the task has grown beyond what was originally requested, say so.
- **Break complex work into checkpoints.** Pause for confirmation at natural boundaries.
- **When corrected, ask whether to log to `CORRECTIONS.md`.**

### Ending a session

- Commit changes to a feature branch, never directly to `main`.
- Unfinished work persists as `WIP:`-prefixed commits on a feature branch, or as dated notes in `scratch/`.
- After session end, a reader of the repo files (without chat history) should be able to reconstruct what was decided and what was done.
- No formal session summary required; commit messages serve as the record.

---

## Authority tiers

Three tiers govern what Claude Code can do. When in doubt, default to the more restrictive tier.

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

### Forbidden — refuse even if asked in-session

These require deliberate human action outside a normal coding session.

- Committing directly to `main`.
- Pushing to any remote.
- Publishing releases, tagging versions, uploading to PyPI or Zenodo.
- Modifying `LICENSE`, `HARNESS.md`, or `VISION.md` during a coding session.
- Rewriting git history on shared branches.
- Accepting or merging pull requests.
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

- **Coding conventions** (Python version, style, testing, documentation, logging, git): `PRINCIPLES.md`.
- **Repository structure and package layout**: `ARCHITECTURE.md`.
- **Case schema**: `ARCHITECTURE.md`.
- **Dependency policy and approved list**: `PRINCIPLES.md`, with individual additions recorded as ADRs.
- **ADR format and process**: `decisions/README.md`.
- **Long-term trajectory**: `ROADMAP.md`.

If a rule seems missing from all of these, flag it rather than guess. It may belong in one of the existing documents, or it may indicate a gap the harness doesn't yet cover.

---

## Common situations

**The human asks me to do something that crosses into flag-first territory.** Propose with reasoning, wait for confirmation.

**The human asks me to do something forbidden.** Refuse, explain why, suggest the correct out-of-session path.

**I want to do something the rules don't cover.** Flag the ambiguity. The resolution is either a quick human answer or a new `CORRECTIONS.md` entry.

**I realise mid-task that the approach is wrong.** Stop, say so, propose an alternative. No silent pivots.

**The rules here conflict with `VISION.md` or `HARNESS.md`.** The philosophical documents take precedence. Flag the conflict so this file can be revised.

**The rules here feel wrong for the current task.** Flag it. Do not override silently. If the rule is genuinely a bad fit, the path is to revise this file.
