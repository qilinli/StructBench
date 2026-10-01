# 0008 — Principle/mechanism separation between HARNESS and CLAUDE

**Status**: Accepted
**Type**: Durable
**Date**: 2026-04-24

## Context

`HARNESS.md` and `CLAUDE.md` have different purposes — HARNESS explains the philosophy (*why* rules exist); CLAUDE specifies the mechanisms (*what* to do in a session). Content can drift across this boundary: philosophical justifications creeping into CLAUDE, or operational details migrating into HARNESS. Either drift erodes the separation and makes both documents harder to read.

This drift was already observed during initial drafting: an early CLAUDE.md draft contained justificatory prose for the "forbidden" tier, and the corrections handling section had slight philosophical framing. These were caught and trimmed during review, but the pattern is expected to recur.

## Decision

The separation is explicit and maintained:

- **`HARNESS.md`** carries principles (*why*) — the philosophical basis, tenets, and their rationale.
- **`CLAUDE.md`** carries mechanisms (*what*) — the operational rules, workflows, and specific lists.

A reminder of this separation is placed at the top of `CLAUDE.md` (in its Purpose section), because CLAUDE.md is re-read at every session start, making it the best place for the principle to be reinforced when editing is most likely.

When either document is being edited, content drifting across the boundary is routed back to its proper home rather than inlined. If content legitimately needs to exist in both (rare), a cross-reference is used instead of duplication.

## Alternatives considered

- **Merge HARNESS and CLAUDE into one document**: rejected. They have different read frequencies (CLAUDE every session, HARNESS rarely), different audiences (agent vs. human), and different stability (mechanisms evolve, philosophy is stable). Merging would dilute both.
- **No explicit boundary** (trust the drafters to maintain it): rejected because drift was already observed during initial drafting. The boundary needs a written reminder to survive.
- **Place the reminder in HARNESS instead**: rejected because HARNESS is read rarely; CLAUDE's Purpose section is the effective place to reinforce the separation.

## Consequences

- Both documents remain tight in their respective voices — HARNESS stays philosophical, CLAUDE stays operational.
- Editors (human or agent) have a written cue to check which document a piece of content belongs in.
- The boundary is enforced by convention, not by tooling. If a violation slips through, it is corrected during review, not automatically caught.
- When in genuine doubt about where content belongs, the resolution is to flag the ambiguity and decide deliberately rather than guess.
