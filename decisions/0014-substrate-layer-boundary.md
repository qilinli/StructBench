# 0014 — StructBench is the substrate layer of a broader research program

**Status**: Accepted
**Type**: Durable
**Date**: 2026-05-23

## Context

StructBench exists within a broader research program — the 3–5 year research
direction in foundation models and agentic systems for intelligent
infrastructure monitoring, documented in `RESEARCH-PROGRAM.md`. That program
has three conceptual layers:

- **Substrate** — reproducible data generation, solver-agnostic canonical
  format, standard benchmark problems, evaluation protocols. The measurement
  infrastructure that makes the rest possible.
- **Brain** — foundation models that learn general representations of
  structural behaviour and transfer across structures. The scientific core.
- **Body** — agentic systems that orchestrate sensing, reasoning, and
  intervention on real assets. The deployed impact (DECRA project).

Without an explicit layer separation, two failure modes are likely. (1) The
platform absorbs the brain and body layers into its own scope, becoming a
do-everything project that can be neither maintained nor scoped. (2) The
platform drifts away from the program's direction, optimising for narrow
surrogate-model benchmarks that do not feed the eventual foundation-model
and SHM agenda. This ADR records the layer separation and the test by which
scope decisions are gated.

## Decision

1. **StructBench occupies the substrate layer only.** Its scope is
   reproducible data, the canonical solver-agnostic format, benchmark
   problems, reference models as baselines/calibration artefacts, evaluation
   protocols, and (later) deployment tools that any group can reuse.

2. **`RESEARCH-PROGRAM.md` is adopted as the program-level north star** that
   sits above StructBench. It informs direction and prioritisation; it **does
   not define StructBench's scope**. StructBench scope is defined solely by
   `VISION.md` and the ADRs in this folder.

3. **Substrate-layer litmus test.** Each proposal for new work in
   StructBench is gated by:

   > **What is the proposal's primary output?**
   >
   > - **Reusable benchmark infrastructure** — datasets and data-generation
   >   code, the canonical format, benchmark problems, evaluation
   >   protocols, general tooling, **and reference baselines / reference
   >   implementations** (calibrated starting points released alongside the
   >   benchmarks). → **In StructBench.**
   > - **A scientific contribution** — a novel method, a model whose
   >   existence is the claim of a paper (e.g. a structural foundation
   >   model, a new architecture), a system deployed on a specific real
   >   asset. → **Outside StructBench**, in a separate repository or paper.
   >   StructBench may *evaluate* such artefacts and report their scores
   >   on its leaderboard; it does not host them.
   >
   > The distinguishing question for any trained model is **role**, not
   > quality or training cost: a best-effort GNN released so users have a
   > known-good evaluation pipeline and a number to beat is a *baseline*
   > (substrate); a foundation model whose existence is the paper's
   > contribution is a *research artefact* (brain layer).

## Alternatives considered

- **Fold the program-level vision into `VISION.md`.** Rejected. Merging
  program and platform into one document erases the level separation the
  litmus test depends on; future sessions would read the agentic and
  foundation-model agenda as StructBench's to-do list, which is the failure
  mode this ADR exists to prevent.

- **Keep `RESEARCH-PROGRAM.md` as a private note, outside session-start
  context.** Rejected. The program would still influence prioritisation
  (because the human holds it in their head), but the agent would lack
  access to it — producing recommendations misaligned with direction.

- **Set the substrate boundary as a slogan, without a litmus test.**
  Rejected. "Stay at the substrate layer" is not falsifiable on its own;
  without an operational test, individual decisions erode the boundary over
  months while the slogan survives intact with its meaning quietly changed.

## Consequences

- `RESEARCH-PROGRAM.md` is added to the session-start reading list as
  **context-only**; this is implemented as an in-place revision of ADR-0009.

- Future ADRs proposing new scope reference this ADR's litmus test in their
  *Context* section, naming explicitly which side of the test the proposal
  lands on.

- Work the litmus test rules *out of* StructBench (e.g., training a
  foundation model **as a paper's contribution**, deploying a monitoring
  agent on a real asset) is not blocked — it lives in a separate repository
  and may consume StructBench artefacts including the released baselines.
  The boundary is structural, not prohibitive.

- The platform/program separation is asymmetric: pivots in
  `RESEARCH-PROGRAM.md` can prompt scope reviews here, but in-session edits
  to StructBench scope must not propagate back into `RESEARCH-PROGRAM.md`
  without an explicit out-of-session decision.
