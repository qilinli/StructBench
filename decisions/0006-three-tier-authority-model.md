# 0006 — Three-tier authority model for Claude Code

**Status**: Accepted
**Type**: Durable
**Date**: 2026-04-24

## Context

HARNESS.md tenet 4 asserts that the human and the agent have distinct accountabilities, but delegates the specific boundary to the operational manual. A workable boundary needs enough resolution to handle the common cases without being so granular that every action requires lookup.

## Decision

Claude Code operates under a three-tier authority model, specified in detail in `CLAUDE.md`:

- **Unilateral** — actions Claude Code takes without asking (local, reversible, no new commitments).
- **Flag-first** — actions Claude Code proposes and waits for human confirmation (reversible but with downstream consequences).
- **Forbidden** — actions Claude Code refuses even if asked in-session (high-consequence, cross-trust-boundary, or irreversible).

Forbidden items cannot be unlocked by in-session instruction from the human. Unlocking requires a deliberate out-of-session revision to `CLAUDE.md` or `HARNESS.md`. This is called "genuine refuse" and is intentional — a harness that yields to any in-session override is not really a harness.

## Alternatives considered

- **Two tiers (can / cannot)**: rejected because it collapses reversible-with-consequences actions into either excessive autonomy or excessive friction. The flag-first tier absorbs the important middle case.
- **Fully unilateral**: rejected because it invites silent decision accumulation, which HARNESS tenet 4 identifies as a failure mode.
- **Fully flag-first**: rejected because it produces unusable friction on routine work.
- **Soft refuse** (refuse by default but override on explicit ask): rejected because it collapses into de facto flag-first under pressure. The strength of the forbidden tier is that it cannot be negotiated mid-session.

## Consequences

- Predictable agent behaviour: the human can anticipate whether any given request will proceed, pause, or be refused.
- Some in-session friction on forbidden items — the harness has to be revised out-of-session to unlock them. This friction is the point.
- The specific lists in each tier are expected to evolve. Revisions are recorded by updating `CLAUDE.md` and noting in an ADR if the change is substantive.
- If Claude Code finds itself wanting to refuse something not on the forbidden list but feeling it should be, this is flagged as a potential tier-list update, not resolved silently.
