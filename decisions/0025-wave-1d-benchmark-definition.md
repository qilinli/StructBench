# 0025 — Wave 1D benchmark: task, split, and eval protocol

**Status**: Accepted
**Type**: Durable
**Date**: 2026-07-03

## Context

ADR-0024 puts `wave_propagation_1d` in v0.2 as the platform's entry tier.
The dataset is `../data/Concrete-Beam/1DWavePropagation`: 16 LS-DYNA SPH
runs of an elastic bar (`*MAT_ELASTIC`) excited by an initial velocity. Per
the collaborators' `simulation_specification.xlsx`, the sweep is bar length
{200, 300, 400, 500} mm × initial velocity {1, 2, 4, 8} mm/ms, simulated
for 30 ms with 0.1 ms output interval (300 frames per case; frame counts
and the deck unit system are verified at ingestion, per the Taylor
precedent).

This ADR fixes the benchmark built on that data: the learning task, the
immutable split, and the evaluation protocol. The benchmark's stated role
is the entry tier — onboarding, the docs tutorial, and a fast CI-scale
dataset. With 16 linear-elastic cases it is not expected to differentiate
strong methods, and its card says so.

## Decision

1. **Task — the ADR-0019 autoregressive transition task, unchanged.** A
   history of per-particle states in, next-step prediction out, rolled out
   autoregressively at evaluation.

2. **Auxiliary field — axial stress**, extracted from the canonical
   6-component `response/element/sph/stress`. In an elastic wave problem
   the particles barely move; the travelling stress wave is the physics, so
   the auxiliary field is the headline target and position RMSE is the
   sanity metric — the reverse of Taylor's emphasis.

3. **Fixed split — interpolation only, no extrapolation** (maintainer
   decision, 2026-07-03: with a 16-case set, do not spend cases on an
   extrapolation probe). The split is a symmetric interior holdout on the
   length × velocity grid; every length and every velocity appears in
   train:

   | split | cases | n |
   |---|---|---|
   | train | all of L200 and L500; (300,1), (300,8), (400,1), (400,8) | 12 |
   | val | (300,2), (400,4) | 2 |
   | test — interpolation | (300,4), (400,2) | 2 |

   The split is immutable once committed; changing it is a new benchmark
   version with its own ADR (ADR-0019 precedent).

4. **Evaluation protocol.** A model is seeded with the first `L`
   ground-truth frames, then rolled out to the end of the trajectory.
   Metrics, per split:
   - one-step and full-rollout **axial-stress RMSE** (headline);
   - one-step and full-rollout **position RMSE** (sanity);
   - **quantities of interest**: wave-front arrival-time error at fixed
     gauge stations along the bar, and peak-stress error.

## Alternatives considered

- **An extrapolation split** (e.g. holding out the (500, 8) corner).
  Rejected by the maintainer: the set is too small to split three ways, and
  the entry tier does not need a generalisation probe — the notch-beam
  benchmarks carry that duty in v0.2 (ADR-0026).

- **Position as the headline target.** Rejected: displacements are near
  zero in an elastic wave; a model could score well while learning nothing.
  The stress field is the signal.

- **Shipping it as a tutorial example, not a benchmark.** Rejected in
  ADR-0024; the honesty about its role lives in its benchmark card.

## Consequences

- The split lives in `benchmarks/wave_propagation_1d/` as case-id lists,
  immutable, with the module following the Taylor layout (`benchmark.py`
  constants + `card.py` per ADR-0027).

- The tutorial and CI use this dataset: small (≈0.5–1.3k particles per
  case, 16 cases), fast to train, end-to-end through the same pipeline as
  every other benchmark.

- The axial-stress extraction (Voigt component selection) lives in
  `datasets/` and is tested, mirroring the von Mises derivation for Taylor
  (ADR-0019).
