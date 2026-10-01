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

## Amendment (2026-10-01): the folder moves to `docs/decisions/`

The decision log moves from `decisions/` at the repo root to
`docs/decisions/` (maintainer, 2026-10-01). ADRs are documents, `docs/` is
where the project's other documents live, and the root is kept for the
package, its configuration and the tooling. Nothing else in this ADR
changes: the format, the numbering, the index and the finalisation rule
stand.

Consequences: the citable address is now `docs/decisions/NNNN-slug.md`;
citations by number (`ADR-NNNN`) are unaffected; links to the old path
from outside this repository are not redirected and are their authors' to
update; the index is `docs/decisions/README.md`. ADR bodies that spell the
old path (0009, 0015, 0029, 0056, 0062, 0065) are left as written -- they
are records, and the files they name sit beside them.

*(Drafted by the agent on the maintainer's instruction; the maintainer
finalises.)*

## Amendment (2026-10-01): the bar for a record is raised

After the review of the whole log (ADR-0073: 71 records, 86,000 words, most
accepted unread, several in conflict), the maintainer raised the bar
(2026-10-01). A record is written only for a decision that binds later work
*and* needs the maintainer's judgment; anything else goes to the home
`docs/decisions/README.md` names. A record is one page (the suite refuses a
new one over 900 words), opens with a `**Your call**` block naming the
judgments acceptance makes, and names in `**Amends**` the records it touches,
resolving clashes rather than leaving them. Ephemeral is the default type;
Durable is reserved for what is expensive or impossible to reverse. Notes are
for amendments, verdicts and pointers only; a record with more than three is
consolidated. Proposed is not a parking state. The log is triaged at each
release or every fifteen records. The format's five sections, the numbering
and the index stand; `docs/decisions/README.md` carries the rules in full.

*(Drafted by the agent on the maintainer's instruction; the maintainer
finalises.)*
