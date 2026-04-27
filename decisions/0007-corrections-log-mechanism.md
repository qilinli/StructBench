# 0007 — CORRECTIONS.md mechanism for small corrections

**Status**: Accepted
**Type**: Durable
**Date**: 2026-04-24

## Context

Real work produces many small corrections — style preferences, recurring minor mistakes, one-line behavioural adjustments — that do not rise to the level of an architectural decision but still need to be remembered across sessions. Without a mechanism, these corrections decay: the human corrects the same mistake repeatedly, or silently fixes the output, and the agent never learns.

HARNESS.md tenet 4 (after its revision on 2026-04-24) establishes that the human must surface corrections rather than silently absorb them, and that the agent must have a place to record them. The specific mechanism is left to the operational manual.

## Decision

Small corrections are recorded in a root-level file: `CORRECTIONS.md`.

- **Format**: append-only log, dated entries, one line per correction, with a status flag.
- **Statuses**: `active` (informs future behaviour), `resolved` (no longer applies, retained for history), `promoted` (moved into `CLAUDE.md` or `PRINCIPLES.md` as a durable rule).
- **Capture workflow**: when the human corrects something that could plausibly recur, Claude Code asks *"should I log this to `CORRECTIONS.md`?"* On confirmation, the entry is added before continuing. The human may also add corrections directly.
- **Read workflow**: at session start, Claude Code reads all `active` entries as part of the start-of-session reading list (see ADR-0009).
- **Distillation**: every few weeks (cadence not fixed), a review pass is done — durable corrections are promoted into `CLAUDE.md` or `PRINCIPLES.md`, resolved ones are marked, one-offs are deleted.

## Alternatives considered

- **Write an ADR per correction**: rejected as too heavy for style-level corrections. ADRs are for decisions with real rationale; corrections are for preferences and habits.
- **Add rules directly to `CLAUDE.md`**: rejected because `CLAUDE.md` would accumulate noise and become hard to read. Distillation into `CLAUDE.md` is the right path, but only after a correction proves durable.
- **No explicit mechanism** (rely on Claude Code's context memory): rejected because AI agents have no persistent memory across sessions. Without a file, corrections decay.

## Consequences

- Corrections persist across sessions via a lightweight mechanism that doesn't fight the decision-log infrastructure.
- A small ongoing discipline is required: Claude Code asks about logging when corrected, the human confirms, and distillation happens periodically.
- The file may accumulate noise if distillation is skipped. This is a known risk; mitigated by keeping the format trivial and the distillation pass short.
- The detection of "this is a correction" is imperfect — Claude Code will miss some and misidentify others. Better-than-nothing is the design target.
