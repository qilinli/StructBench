# 0005 — ADR format and decision-log structure

**Status**: Accepted
**Type**: Durable
**Date**: 2026-04-24

## Context

HARNESS.md tenet 2 requires that every significant decision have a designated home with recorded rationale. This needs a concrete implementation: what format decisions take, where they live, and how they are indexed.

## Decision

Decisions are recorded as Architecture Decision Records (ADRs), one file per decision, in the `decisions/` folder at repo root.

Format (five sections):

```
# NNNN — Title

**Status**: Accepted | Proposed | Superseded by NNNN
**Type**: Durable | Ephemeral
**Date**: YYYY-MM-DD

## Context
## Decision
## Alternatives considered
## Consequences
```

Filenames: `NNNN-kebab-case-title.md` with zero-padded sequential numbers. Numbers are never reused, even when decisions are superseded.

The `decisions/README.md` file contains the format spec and an index table of active ADRs; it is updated whenever a new ADR is added.

## Alternatives considered

- **Monolithic `DECISIONS.md` file**: rejected. Scales poorly past ~20 entries, invites merge conflicts when two sessions add decisions concurrently, and hurts grep-ability.
- **Looser format** (dated bullet list, minimal structure): rejected because rationale would not be consistently captured, violating HARNESS tenet 2.
- **More elaborate format** (adding implementation notes, validation criteria, stakeholders): rejected because added friction discourages writing ADRs, which defeats HARNESS tenet 1.

## Consequences

- Each decision has a stable, citable address (`decisions/NNNN-slug.md`).
- Supersession is itself a decision event, producing a new ADR that references the old one — preserving the historical record.
- `decisions/README.md` must be maintained as the single index; a missing row there can cause an ADR to be effectively invisible.
- Claude Code may draft ADRs during sessions; the human finalises them before they are marked `Accepted`.
