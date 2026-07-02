# 0021 — v0.1 narrows to Taylor 2D; the portfolio spreads across releases

**Status**: Accepted
**Type**: Durable
**Date**: 2026-07-02

## Context

ADR-0015 committed v0.1 to shipping a portfolio of three existing LS-DYNA
datasets (Taylor 2D, RC beam, segmented beam) as benchmarks with prior-paper
GNN baselines. Since then the state has moved asymmetrically: the Taylor 2D
benchmark is code-complete end to end (ADR-0019 protocol implemented, 65
tests green, DUG training recipe staged), and the general adapter (ADR-0016)
has been proven on a second dataset family (a concrete-beam SPH case ingested
unchanged). But the RC beam benchmark still lacks a designated dataset and
prior-model pairing, and the segmented-beam dataset has not been identified
in the archive at all. Both depend on maintainer knowledge and, potentially,
collaborator input (ADR-0015's own data-generation constraint).

Holding the release to the full portfolio gates a working, provable substrate
on the slowest-moving, least-defined items. The maintainer decided on
2026-07-02 (in session, while settling the first ROADMAP.md): a running
pipeline working for Taylor impact is sufficient to call v0.1; no paper is
planned for v0.1 given the limited data.

Under ADR-0014's litmus test everything here remains substrate; this ADR only
resequences releases.

## Decision

1. **v0.1 is the Taylor 2D substrate proof**: the canonical case format and
   HDF5 I/O, the general LS-DYNA adapter, the Taylor 2D benchmark
   (ADR-0019), and its single-scale GNS baseline **trained, with recorded
   metrics** — all exercised through the config-driven pipeline
   (`structbench-train`), released as a **public GitHub repository**.

2. **No paper attaches to v0.1.**

3. **The portfolio commitment stands but spreads across releases**: the RC
   beam and segmented-beam benchmarks (with their prior-paper baselines)
   move to v0.2+, each entering when its dataset and prior model are
   designated. ADR-0015's reasoning — the portfolio pressure-tests the
   schema and adapter — is unchanged; only the version boundary moves.

4. **The MS-GNS second Taylor baseline is not a v0.1 requirement** (it
   remains a Proposed enhancement on its own spec).

5. **Dataset hosting for the release is decided separately.** The raw and
   canonical data currently live in the maintainer's OneDrive; how released
   benchmark data is hosted (Zenodo, HF datasets, institutional, …) is an
   open ROADMAP question, not settled here.

This ADR **amends ADR-0015's decision point 1** (what v0.1 ships). It does
not supersede ADR-0015 wholesale: the portfolio, the baseline framing, and
the data-generation deferral all remain in force.

## Alternatives considered

- **Keep the three-benchmark v0.1.** Rejected: gates the release on an
  unidentified dataset and an undesignated model pairing, with no date
  either can be promised for; the substrate proof is complete with one
  benchmark end to end.

- **Call v0.1 without a trained baseline** (code-only). Rejected: ADR-0015's
  own argument — without at least one real benchmark-plus-baseline through
  the full pipeline, the substrate has no end-to-end validation.

- **Re-litigate the portfolio itself.** Not considered here; the portfolio
  remains the plan (ADR-0015), only its release sequencing changes.

## Consequences

- v0.1 becomes concretely reachable: the remaining work is the
  radius_graph batch-partition fix, the DUG training run, recording the
  baseline metrics, and the (human-only) GitHub publication.

- ADR-0015 stays **Accepted** in the index with a cross-reference to this
  ADR; its portfolio content continues to govern v0.2+ scope.

- ROADMAP.md's v0.1 definition of done is rewritten against this ADR; RC
  beam and segmented beam move to the near horizon.

- The v0.1 GitHub release is an out-of-session human action (CLAUDE.md
  forbidden tier: publishing, pushing to remotes).
