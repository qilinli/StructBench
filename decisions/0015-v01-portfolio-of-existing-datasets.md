# 0015 — v0.1 ships existing LS-DYNA datasets as benchmarks with prior-paper GNN baselines (supersedes 0003)

**Status**: Accepted
**Type**: Durable
**Date**: 2026-05-24

## Context

ADR-0003 set the v0.1 anchor problem as drop-weight impact on RC beams
(`RCBeam-DropImpact-v1`) on 2026-04-24, before ADR-0014 established the
substrate-vs-brain-vs-body layer separation. ADR-0003 assumed v0.1 needed
one well-defined anchor problem because "data generation and schema design
cascade from it." Two facts have since shifted the calculation.

**Existing assets are richer than ADR-0003 assumed.** The user has multiple
existing LS-DYNA datasets from prior published work — Taylor 2D, RC beam,
segmented beam — and prior-paper GNNs trained against each. Under ADR-0014's
substrate-layer litmus test, prior-paper GNNs released as calibrated
reference implementations (so users have a known-good evaluation pipeline
and a number to beat) are *baselines* (substrate), not research artefacts.
They belong in StructBench. Shipping nothing of them in v0.1 would discard
ready-to-use assets that exactly match the substrate-layer role.

**Data-generation autonomy is constrained.** LS-DYNA case production for
this project goes through civil-engineering collaborators using the GUI.
Scripted parametric expansion of any single dataset is not currently
feasible without either deck-templating with collaborator buy-in for batch
runs, or adopting a Python-native solver (each a substantial separate
decision). That makes "pick one anchor and grow it" the wrong shape for
v0.1 — the natural shape is "ship what exists."

Under those two facts, ADR-0003's framing — one anchor, deeply grown — is
no longer the right shape. Substrate validation needs the format and
adapter exercised across diverse cases, which a portfolio provides more
directly than a single deep anchor.

## Decision

1. **v0.1 ships a portfolio of existing LS-DYNA datasets** as benchmarks.
   The initial set comprises Taylor 2D, RC beam, and segmented beam —
   exact benchmark module names settle when the modules are created.
   Datasets ship at their existing sizes; no expansion is in scope for v0.1.

2. **Prior-paper GNNs trained on those datasets are released as baselines**
   under `src/structbench/models/`, paired with each benchmark. Each
   baseline ships with its training code, a checkpoint, and a reference to
   the original paper. Per ADR-0014, these are baselines (reference
   implementations released so users have a known-good evaluation pipeline
   and a number to beat) — not the prior papers' scientific contribution.

3. **Data-generation expansion is deferred.** Growing any dataset requires
   either (a) deck-templating with collaborator buy-in for scripted batch
   runs, or (b) adopting a Python-native solver. Both are out of scope for
   v0.1 and will be addressed in their own future ADRs as the need arises.

4. **ADR-0003 is superseded.** Its content remains in the repository for
   history; its status flips to *Superseded by 0015* on acceptance of
   this ADR.

## Alternatives considered

- **Single Taylor anchor with parametric expansion.** Drafted as the
  original ADR-0015 content; rejected before commit on this revision.
  Discards the RC-beam and segmented-beam assets and their paired GNNs;
  assumes scripted expansion is feasible, which under the collaborator
  constraint it currently is not.

- **Keep ADR-0003 — single RC-beam anchor.** Rejected for the symmetric
  reason: discards Taylor and segmented-beam assets, and confines
  schema validation to RC-specific structure (rebar embedding, concrete
  damage, multi-material contact) on the first real case rather than
  letting simpler cases stress-test the schema first.

- **Defer the v0.1 benchmark decision; ship "just the substrate".**
  Rejected. Without at least one real benchmark plus baseline going
  through the full pipeline, the substrate has no end-to-end validation
  that the canonical format works in practice. v0.1 needs the proof.

- **Wait for data-generation autonomy (deck-templating or solver-switch)
  before shipping any benchmark.** Rejected. That re-imports the brain
  /body-layer worry the substrate is supposed to be independent of. The
  substrate's first version should ship on existing assets; the
  data-generation question can move on its own track.

## Consequences

- v0.1 ships three benchmark modules under
  `src/structbench/benchmarks/` (exact names settled when the modules
  are created) and three baseline models under
  `src/structbench/models/` paired with each benchmark.

- The general LS-DYNA adapter (ADR-0016) must handle all three datasets'
  d3plot output uniformly. This is intentional pressure on the schema —
  SPH-only (Taylor), solid + beam + discrete with rebar coupling (RC
  beam), and segmented-beam structure will each exercise different schema
  surface. Some schema gaps will surface; each is handled in its own
  amendment ADR as it arises.

- Dataset sizes are fixed by what the collaborators have already
  produced. The "is each dataset big enough to be a useful benchmark"
  question is real but is downstream of acceptance here. If sizes prove
  inadequate, the path is to revisit the data-generation question (point
  3 of the Decision), not to re-litigate this ADR.

- The prior papers' scientific contribution remains where it was
  published; only the trained-model artefacts and training code graduate
  here as baselines. The release framing matters: these are
  known-good starting points, not the original papers' claims restated.

- ADR-0003's "community positioning is impact/blast engineering" framing
  is dropped from v0.1. The v0.1 paper, if any, positions StructBench as
  substrate (benchmarks + adapter + format + baselines), not as a single-
  problem community benchmark.

- Downstream documentation edits will be required on acceptance: ADR-0003
  status flip to *Superseded by 0015*, `CLAUDE.md` project snapshot
  (currently "impact/blast on RC beams") rewritten to reflect the
  portfolio framing, and the ADR index in `decisions/README.md` updated
  to reflect the new title. These follow on acceptance, not in the same
  change as this ADR.
