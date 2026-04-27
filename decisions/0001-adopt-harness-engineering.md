# 0001 — Adopt harness engineering methodology

**Status**: Accepted
**Type**: Durable
**Date**: 2026-04-24

## Context

This project is run by a human (Qilin Li) and an AI agent (Claude Code) together, over a multi-year timeline. AI agents lose session context by default: each session starts from a clean slate, and any knowledge not written to files is effectively lost. Without a deliberate operating structure, this produces decision drift, invariant erosion, and gradually-inconsistent behaviour across sessions.

Harness engineering is an emerging practice — evolved from prompt engineering and context engineering — focused on constructing the project environment itself so that human–agent collaboration remains coherent over long time horizons. The core idea is that state lives in files rather than in memory or conversation.

## Decision

The project adopts harness engineering as its operating methodology. The philosophy is documented in `HARNESS.md`. Operational rules derived from the philosophy live in `CLAUDE.md`, with supporting documents (`VISION.md`, `PRINCIPLES.md`, `ARCHITECTURE.md`, `ROADMAP.md`) carrying specific classes of content. Decisions are recorded as ADRs in `decisions/`.

## Alternatives considered

- **Ad-hoc workflow**: rely on the human's memory and moment-to-moment instruction. Rejected because it guarantees drift over a multi-year project.
- **Heavier formal process** (e.g., enterprise SDLC, formal specification): rejected as disproportionate for a small research project and likely to slow work without proportional benefit.

## Consequences

- Initial investment in foundational documents (roughly one week of writing before coding begins).
- Documents must be maintained; the harness itself is subject to revision (see ADR-0009 and later revisions).
- The methodology is intended to be reusable across the author's future projects.
- Project state becomes auditable from files alone; no reliance on external memory.
