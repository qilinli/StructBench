# 0039 — Notch-impact scored horizon: 250 µs evaluation window, matched baseline recipe

**Status**: Accepted
**Type**: Durable
**Date**: 2026-07-20

## Context

The notch-impact dataset stores 502 frames at 1 µs output (0.5 ms per event;
card corrected 2026-07-20, this branch). Measured against that grid, the
physics is front-loaded:

- Maximum plastic strain saturates by ~frame 10–20 — the fracture outcome is
  decided within the first ~2–4% of the trajectory.
- Global internal energy reaches 90% of its final value by frame 22–38 and
  **99% by frame 77–213** (width-dependent; 640 mm settles slowest).
- Everything after is ballistic separation of the broken pieces plus elastic
  ringing: no further fracture physics.

The full-horizon rollout metric is dominated by that tail. From the
2026-07-17 bless fleet's saved per-frame RMSE (seed s1, 18 test rollouts):
2.4% of the final position error has accrued by frame 51, 8.5% by frame 101,
and **half accrues after frame 301** — the headline number mostly measures
rigid-body drift of already-broken pieces, not the fracture prediction the
QoIs (`cracked_fraction`, `midspan_deflection_peak`) exist to test. Worse,
`midspan_deflection_peak` ("peak over the trajectory") is silently
horizon-dependent for fractured beams, whose mid-span gauge keeps falling —
its current value is pinned by wherever the stored trajectory happens to end.

A survey of field practice (2026-07-20, verified against sources) found no
benchmark that scores ~500-step rollouts at native solver-output cadence:
MeshGraphNets steps 16–100 solver substeps per learned step and headlines
50-step rollout RMSE; LagrangeBench coarse-grains 100× by design and scores
5- and 20-step windows; NVIDIA PhysicsNeMo's transient examples truncate both
training and evaluation (vortex shedding 300 of 600 frames, deforming plate
200 of 400) and its crash surrogates use 14–51 frames per event.

## Decision

### 1. The dataset is unchanged

All 502 frames at 1 µs ship. The fine cadence is the dataset's
differentiating asset (no surveyed public dataset resolves impact fracture at
this resolution), and the tail feeds the diagnostic below.

### 2. Scored horizon: 250 µs

Evaluation rollouts are seeded per ADR-0035 and **scored on frames
[`input_frames`, 250]**. Rationale for 250: it covers internal-energy
settling (99% by frame 213 in the slowest-settling width) with margin, so every scored
frame contains physics; beyond it the trajectory is drift. All reported
rollout metrics are computed over this window, and both QoIs are pinned to
it: `midspan_deflection_peak` is the peak within frames [0, 250];
`cracked_fraction` is evaluated at frame 250 (insensitive in practice — the
crack field saturates by ~frame 40).

### 3. Full-horizon curve stays as a diagnostic

The per-frame error curve to frame 502 is reported alongside, as a
non-leaderboard long-horizon-stability diagnostic.

### 4. Baseline recipe matches the protocol

The reference CGN baseline trains on windows whose target frame is ≤ 250.
This roughly doubles the transient's share of the window pool (~8% → ~16%),
aligns train and eval distributions, and follows the PhysicsNeMo truncation
precedent. Normalization statistics are recomputed over the truncated window
pool. This is **recipe, not protocol**: submitters may train on the tail.

### 5. Scope

This ADR covers `notch_beam_2d_impact` only. The bend benchmark has a
different timescale (500 ms quasi-static, ADR-0026) and needs its own
timeline analysis before any horizon decision.

## Alternatives considered

- **Temporal stride 4–5× (dt 4–5 µs, ~100-step rollouts; the maintainer's
  prior CGN paper's regime).** Rejected for the headline protocol: at stride
  ≥ 4 the 6-frame input window (ADR-0035) spans ≥ 24 µs — past
  plastic-strain saturation — moving fracture *initiation* from the
  predicted region into the observed one. The task becomes propagation of an
  observed fracture, misaligned with the fracture-centric QoIs and with
  predictive deployment (a surrogate seeded with post-impact ground truth
  cannot assess a design from initial conditions). It also degrades per-step
  locality: a stress wave crosses ~1.4 particle spacings per frame at
  stride 1, ~7 at stride 5. A coarse companion track can be added later
  without superseding this ADR.
- **Keep full-horizon scoring (status quo).** Rejected: the headline metric
  is then half-determined by frames after 301 — rigid-body drift of broken
  pieces — and `midspan_deflection_peak` stays implicitly pinned to the
  storage end. No surveyed benchmark scores a comparable tail.
- **Shrink the shipped dataset to 250 frames.** Rejected: the tail is cheap,
  feeds the long-horizon diagnostic, and full-resolution completeness is the
  dataset's differentiating asset.

## Consequences

- Benchmark version change (ADR-0019 precedent): card protocol updated and
  pages regenerated with this ADR; the eval pipeline gains a machine-readable
  scored-horizon field at implementation time.
- The 2026-07-17 bless fleet was trained full-window. Plan: rescore it at the
  250-frame horizon (cheap — per-frame RMSE is saved in every rollout file)
  as the full-window reference, and run a 2-seed truncated-recipe arm; bless
  the v0.2 baseline from whichever recipe wins on the 250-horizon metrics.
- In-training validation may adopt the same truncation (~2× cheaper per val
  point).
- The headline rollout numbers drop relative to earlier logs (e.g. s1
  test-interp ~2.8 mm full-horizon → ~1.1 mm at 250) — a metric-definition
  change, not a model improvement; results registries must not mix the two
  definitions.
