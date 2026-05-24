# Decisions

This folder holds the project's Architecture Decision Records (ADRs). Each ADR captures one decision, its context, the alternatives considered, and its consequences. Together, they form the project's decision history.

---

## Format

Each ADR is one markdown file with the following structure:

```
# NNNN — Title

**Status**: Accepted | Proposed | Superseded by NNNN
**Type**: Durable | Ephemeral
**Date**: YYYY-MM-DD

## Context
What problem or question prompted this decision.

## Decision
What was decided.

## Alternatives considered
What else was on the table, and why not.

## Consequences
What becomes easier, harder, or constrained as a result.
```

### Filenames

`NNNN-kebab-case-title.md`, where NNNN is a zero-padded sequential number (0001, 0002, ...). Numbers are never reused, even when decisions are superseded.

### Status

- **Proposed** — drafted, not yet approved by the human.
- **Accepted** — current, active decision.
- **Superseded by NNNN** — replaced by a later ADR. The superseded ADR is kept for history; the new one references it.

### Type

- **Durable** — effectively permanent. Revising requires a new superseding ADR with explicit reasoning.
- **Ephemeral** — expected to change as the project evolves. Can be updated in place with a dated note appended to the ADR; supersession is not required.

---

## Index

| # | Title | Type | Status |
|---|-------|------|--------|
| 0001 | Adopt harness engineering methodology | Durable | Accepted |
| 0002 | Project name is StructBench | Durable | Accepted |
| 0003 | v0.1 anchor problem is impact on RC beams | Durable | Accepted |
| 0004 | Platform is solver-agnostic; LS-DYNA for v0.1 data generation | Durable | Accepted |
| 0005 | ADR format and decision-log structure | Durable | Accepted |
| 0006 | Three-tier authority model for Claude Code | Durable | Accepted |
| 0007 | CORRECTIONS.md mechanism for small corrections | Durable | Accepted |
| 0008 | Principle/mechanism separation between HARNESS and CLAUDE | Durable | Accepted |
| 0009 | Session-start reading list | Ephemeral | Accepted |
| 0010 | FEM solver code lives outside the importable package | Durable | Accepted |
| 0011 | Case vocabulary for the data record | Durable | Accepted |
| 0012 | Case schema field-level structure | Durable | Accepted |
| 0013 | HDF5 persistence layout for the case schema | Durable | Accepted |
| 0014 | StructBench is the substrate layer of a broader research program | Durable | Accepted |

---

## Adding a new ADR

1. Claim the next available number by checking the highest NNNN in use.
2. Create `NNNN-kebab-case-title.md` using the format above.
3. Draft the ADR. Claude Code may draft; the human finalises.
4. Add a row to the index in this README.
5. Commit with a message like `docs: add ADR-NNNN on <title>`.
