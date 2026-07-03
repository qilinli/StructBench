# 0022 — v0.2 ships the 1D wave and notch-beam benchmarks; RC beam moves to v0.3

**Status**: Proposed
**Type**: Durable
**Date**: 2026-07-03

## Context

v0.1 is the Taylor-only substrate proof (ADR-0021), with release imminent —
the remaining items (DUG training run, publication) are human actions. The
question is what the platform ships next.

Candidates weighed in session on 2026-07-03: 1D wave propagation, the 2D
notched beam (two loading families), the 3D RC beam under drop-weight impact
with erosion, and 3D slab fragmentation under blast. The segmented beam and
other more complex archive sets were deliberately parked as harder than a
second release should carry.

Three facts decided the shape:

1. **Erosion is an open problem twice over.** The 3D RC beam dataset uses
   element erosion, which is numerically delicate in FEM and — more
   importantly here — unsolved for autoregressive surrogates: deleted
   elements mean particles vanishing mid-rollout, which the transition task
   (ADR-0019) has no vocabulary for. Bundling that research problem into the
   next release would gate a data milestone on a modelling breakthrough.

2. **Two candidate datasets are release-ready on disk.** The
   `1DWavePropagation` set (16 runs) and the `2DNotchBeam` set (216 runs per
   its `simulation_specification.xlsx`, plus 5 purpose-built generalisation
   cases) exist complete with d3plot output. Both are SPH; neither uses
   erosion (verified in the decks: `*MAT_ELASTIC` for the bar;
   `*MAT_CONCRETE_DAMAGE_REL3` + `*MAT_PLASTIC_KINEMATIC`, no
   `*MAT_ADD_EROSION`, for the beam). The general adapter (ADR-0016) already
   ingested a notch-beam case unchanged on 2026-07-02.

3. **No model port is needed.** The maintainer decided the v0.1 single-scale
   GNS is the baseline for all new v0.2 benchmarks — retrained per benchmark
   through the existing config-driven pipeline, not a port of a prior-paper
   model.

## Decision

1. **v0.2 ships three benchmarks**: `wave_propagation_1d` (defined in
   ADR-0023), and the pair `notch_beam_2d_bend` / `notch_beam_2d_impact`
   (defined in ADR-0024).

2. **The baseline for each is the v0.1 single-scale GNS retrained per
   benchmark** via `structbench-train`. No new model architecture enters in
   v0.2.

3. **The platform reads as a difficulty ladder**, and v0.2 grows it in both
   directions from Taylor: 1D elastic wave (entry tier; doubles as the docs
   tutorial and fast-CI dataset) → Taylor 2D (metal plasticity, v0.1) →
   notch-beam pair (concrete damage and fracture, no erosion) → RC beam 3D
   with erosion (v0.3) → slab fragmentation under blast (later).

4. **The RC beam benchmark moves to v0.3**, where its erosion problem is the
   headline rather than a stowaway. Slab blast stays on the later horizon;
   the segmented beam stays parked.

5. This ADR **amends ADR-0015's release sequencing** (as ADR-0021 did). The
   portfolio commitment and baseline framing of ADR-0015 are unchanged.

## Alternatives considered

- **3D RC beam next** — the slot ADR-0015 named. Rejected for v0.2 on the
  erosion argument above; deferred, not dropped.

- **Slab blast fragmentation next.** Showcase appeal, but its dataset is not
  designated in the archive, fragmentation is harder to standardise than
  erosion (the ADR-0003 deferral reasons stand), and it would skip two rungs
  of the ladder.

- **1D wave alone as v0.2.** Too thin to headline a release; 16 elastic runs
  are an entry tier, not a milestone.

- **Notch beam only, with 1D wave shipped as a tutorial example rather than
  a benchmark.** Considered seriously — with 16 linear-elastic cases, GNS
  will likely near-solve the wave benchmark, so it differentiates methods
  poorly. Rejected in favour of a formal entry tier: a uniform
  benchmark-card/protocol surface across all datasets is worth more than
  gatekeeping the word "benchmark", and the card states the tier honestly.

## Consequences

- Three new benchmark modules under `src/structbench/benchmarks/`, three new
  GNS checkpoints, and ingestion of ~237 runs (16 + 216 + 5) from the
  OneDrive archive — batched, specific paths only (per the standing
  CORRECTIONS.md entry on OneDrive hydration).

- The config pipeline needs a benchmark-selection mechanism:
  `cli/train.py` currently hard-imports the Taylor module, which does not
  scale to four benchmarks.

- The dataset-hosting question (ADR-0021 §5, open in ROADMAP.md) becomes
  more pressing at roughly seven times the v0.1 case count; it must be
  settled before the v0.2 release.

- The part-id→embedding-index remap concern (ROADMAP training-robustness
  item) is checked against the notch-beam decks at ingestion; the spec
  sheet's ID tables look small and contiguous, so it is not expected to
  block.

- ROADMAP.md's near horizon is rewritten against this ADR on acceptance
  (RC beam → v0.3; wave + notch-beam pair become the v0.2 definition of
  done).
