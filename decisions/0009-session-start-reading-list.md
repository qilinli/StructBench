# 0009 — Session-start reading list

**Status**: Accepted
**Type**: Ephemeral
**Date**: 2026-04-24

## Context

Every Claude Code session begins from a clean slate. To function coherently, the agent must load project context at the start of each session. Without a specified reading list, this loading is ad-hoc and inconsistent across sessions.

## Decision

At the start of every session, Claude Code reads the following files, in order:

**Always read**:
1. `CLAUDE.md`
2. `VISION.md`
3. `HARNESS.md`
4. `PRINCIPLES.md` (once it exists)
5. `CORRECTIONS.md` — all entries marked `active`
6. `decisions/README.md` — the ADR index

**Conditionally read** (based on the session's task):
- `ARCHITECTURE.md` — if the task touches package structure, module interfaces, or the asset model schema.
- Specific ADRs from the index — whichever are relevant to the task.
- `ROADMAP.md` — if the session is about planning or scoping.

Target: the full start-of-session reading should take under 10 minutes of agent time.

This ADR is marked **Ephemeral** because the list is expected to change as the project grows. Revisions happen in place in this ADR (with a dated note) rather than requiring a new superseding ADR.

## Alternatives considered

- **`CLAUDE.md` only**: rejected because important rules and principles would not reliably reach the agent; session-start context would be thinner than needed.
- **A longer always-read list** (including `ARCHITECTURE.md`, specific ADRs): rejected because it would push the start-of-session reading cost too high and encourage skimming.
- **Letting Claude Code decide what to read based on the task**: rejected because this defeats the purpose of a harness — consistent behaviour across sessions requires a consistent reading protocol.

## Consequences

- Consistent context loading at session start, regardless of who or what starts the session.
- Reading cost is real — every session spends several minutes orienting. This is the price of consistency and is expected to be worthwhile.
- As more documents accumulate, some "always read" entries may need to be demoted to "conditionally read" to keep the total cost bounded.
- `HARNESS.md` being in the always-read list is a deliberate choice; the argument for removing it (most sessions don't test the philosophy) was considered and rejected because the cost of rare philosophy drift is high.

## Revision log

*(Dated notes go here as the list is revised.)*
