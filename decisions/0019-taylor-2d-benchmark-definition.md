# 0019 — v0.1 Taylor 2D benchmark: autoregressive surrogate task, split, and eval protocol

**Status**: Proposed
**Type**: Durable
**Date**: 2026-06-29

## Context

ADR-0003 made impact on a structure the v0.1 anchor problem; ADR-0015 reframed
v0.1 as a portfolio of existing LS-DYNA datasets shipped as benchmarks with
prior-paper GNN baselines. The 2D Copper Bar Taylor Impact dataset is now in
canonical HDF5 (34 cases, ADR-0016 adapter). What is still undefined is the
**benchmark** built on that data: the learning task, the evaluation protocol,
and the train/validation/test split. ARCHITECTURE.md states a benchmark owns
exactly these. This ADR fixes them for the Taylor 2D benchmark.

The reference baseline is a Graph Network Simulator (GNS) ported from the
user's existing `sgnn` code — a faithful encode-process-decode learned
simulator (Sanchez-Gonzalez et al. 2020) with a per-particle auxiliary head and
a wall-distance input feature. Its architecture is an implementation detail
recorded in ARCHITECTURE.md and the design spec, not here; this ADR records the
benchmark *task and protocol*, which any model is evaluated against.

## Decision

1. **Task — autoregressive next-step surrogate (learned simulator).** Given a
   short history of per-particle positions, the model predicts the next-step
   per-particle acceleration; an Euler step integrates to the next position,
   and the model is rolled out autoregressively to reconstruct the full
   trajectory. This is the schema's `transition` unit (ADR-0011) used as the
   training example.

2. **Auxiliary field — von Mises stress.** Alongside position, the benchmark
   predicts a per-particle scalar: **von Mises stress**, computed from the
   canonical 6-component `response/element/sph/stress`. (The prior `sgnn` model
   predicted this same quantity but mislabelled it "strain"; the canonical
   benchmark names it correctly. The real `effective_plastic_strain` is also
   in the data and may be added as a second auxiliary target later.)

3. **Particles.** The model operates on the SPH particles only; the 4 viz-shell
   nodes (the null shell included at ingestion, ADR-0016) are excluded from the
   model graph.

4. **Fixed split.** Velocities are {100,110,…,200} m/s; geometries are
   {60,80,100}. The split is documented and immutable for the benchmark:

   | split | cases | n |
   |---|---|---|
   | train | vel {100,110,120,140,160,180,190} × all geom | 21 |
   | val | vel {150} × all geom | 3 |
   | test — interpolation (headline) | vel {130,170} × all geom | 6 |
   | test — extrapolation | vel {200} × all geom | 3 |
   | held aside | `T-20-80-Convergence` | 1 |

   Interpolation is the headline test (held-out interior velocities, neighbours
   present in train); extrapolation (the top velocity, beyond the training
   range) is reported separately as a small, harder probe. The `Convergence`
   run is excluded from train/val/test and reserved for an optional
   mesh-resolution-invariance check.

5. **Evaluation protocol.** A model is seeded with the first `L` ground-truth
   frames, then rolled out autoregressively to the end of the trajectory.
   Metrics, reported per split:
   - one-step and full-rollout **position RMSE**;
   - **von Mises RMSE** over the rollout;
   - **quantities of interest**: final bar length and mushroom-width error,
     for physical interpretability.

## Alternatives considered

- **Parametric field / QoI-only tasks** (predict a field at a queried time, or
  predict scalar outcomes, both without autoregression). Rejected during
  design: weaker surrogates that exercise little of the per-frame response and
  do not match the schema's `transition` design or the prior practice.

- **Random or mirrored split.** A seeded random split tests only interpolation
  and is undocumented as a benchmark; mirroring the prior `sgnn` split optimises
  for reproducing old numbers rather than defining a clean, generalisation-aware
  benchmark. Rejected in favour of the fixed interpolation-plus-small-
  extrapolation split above.

- **Holding out an entire geometry** (e.g. all of geom 80) as the test.
  Considered, then narrowed: the maintainer chose interpolation as the main
  test with only a small extrapolation set, so all geometries appear in train
  and generalisation is probed in velocity, not geometry.

## Consequences

- The split lives in `benchmarks/taylor_impact_2d/` as case-id lists and is
  treated as immutable; changing it is a new benchmark version with its own
  ADR, so leaderboard numbers stay comparable.

- Reporting interpolation and extrapolation separately makes the benchmark
  honest about where a surrogate generalises and where it does not.

- von Mises as the auxiliary target couples the benchmark to a derived quantity;
  the derivation (Voigt → von Mises) lives in `datasets/` and is tested.

- The protocol generalises to the other v0.1 datasets (RC beam, segmented beam)
  with their own splits and QoIs; each gets its own benchmark definition, not a
  change to this one.
