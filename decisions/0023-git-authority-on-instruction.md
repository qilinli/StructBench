# 0023 — Git authority: `main` moves on explicit in-session instruction (amends 0006)

**Status**: Accepted
**Type**: Durable
**Date**: 2026-07-03

## Context

ADR-0006's three-tier authority model placed committing to `main`, pushing to
the remote, and merging in the forbidden tier — "deliberate human action
outside a normal coding session." The first full cycle under that rule (the
viz harness, 2026-07-03) showed what it costs in practice: after the human
had already reviewed and approved the work in-session, they still had to
manually retype the merge and push commands that Claude handed back. The
retyping added no safety — the decision had already been made and voiced —
only friction and a failure mode of its own (wrong working directory,
missing PATH setup).

## Decision

A fourth authority tier is added between flag-first and forbidden: **on
explicit instruction**. Merging a feature branch into `main`, pushing to the
remote, and committing directly to `main` sit there. Claude Code executes
them directly when — and only when — the human explicitly instructs them in
the session ("merge it", "push"). They never happen as part of unprompted
work: the invariant that **`main` moves only by the human's word** is kept;
what changes is who types the commands after the word is given.

Release actions (tags, PyPI/Zenodo), history rewrites on shared branches,
third-party PR acceptance, and repository settings remain forbidden.

## Alternatives considered

- **Status quo** (human retypes the commands). Rejected: pure transcription
  friction; the authorization already happened in-session.
- **Full git autonomy** (Claude merges/pushes as part of normal work).
  Rejected: `main` and the public GitHub remote are the project's public
  face; moving them should always trace to an explicit human utterance.
- **Feature-branch pushes only.** Rejected: doesn't address the actual
  friction point, which is landing approved work on `main`.

## Consequences

- Approved work lands in one conversational turn; no command handback.
- The audit trail of "who decided" moves from the human's shell history to
  the session instruction plus the commit/merge record (Claude's commits
  carry a co-author trailer).
- The forbidden tier shrinks but keeps the release boundary: nothing
  version-tagged or published ever happens without deliberate human action
  outside a session.
- CLAUDE.md's session-ending rule is softened accordingly: work still lands
  on feature branches by default; `main` moves only on instruction.
