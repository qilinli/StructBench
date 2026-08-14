# 0051 — Prediction-scheme axis: k-frames-per-call implementation (Transolver)

**Status**: Proposed — supersedes the BACKLOG status of ADR-0050 on acceptance
**Type**: Durable
**Date**: 2026-08-14

## Context

ADR-0050 recorded the **k-frames-per-call** prediction-scheme axis
(autoregressive `k=1` ↔ temporal bundling `1<k<T` ↔ one-shot `k=T`) as a
BACKLOG design with a physics grounding, a pipeline-impact analysis, and seven
open design decisions, explicitly *not* authorising implementation. This ADR
settles those seven decisions and records the **implemented** Transolver-family
scheme (`config.frames_per_call = k`, default 1, byte-identical to the pre-0051
recipe). It supersedes ADR-0050's backlog status; ADR-0050 is retained as the
design-rationale and physics record (explicit/implicit analogy, neural CFL,
regime hypothesis), which this ADR does not repeat.

The work was scoped and adversarially verified before coding: a whole-repo
touch-point map and a four-claim skeptic pass (design record in
`scratch/2026-08-14-kframes-design.md`) surfaced the load-bearing subtleties
below before they could ship as silent bugs.

## Decision

Implement `k` as a **single config knob** `TransolverConfig.frames_per_call`
(default `1`), Transolver-only, from which all three schemes fall out. The
seven ADR-0050 decisions resolve as:

1. **Output = per-frame velocities**, integrated. Targets are running-
   integration velocities (`v_0 = next[:,0] − x_last`, `v_j = next[:,j] −
   next[:,j−1]`); eval integrates `x_t + cumsum(velocity)`. Every target stays
   one-step-scaled, so the width-`(dim+1)` target normalizer stays valid
   (applied on the row-folded `(P·k, dim+1)` view). Absolute-displacement
   targets were rejected (grow across the bundle, break normalisation).
2. **k=T sentinel = `frames_per_call = 0`**, resolved at train time from the
   loaded (post-`train_frames`-truncation) trajectories to `k = T_working −
   input_frames`, and the **resolved integer is written back into
   `config.json`**. Evaluation reads the concrete `k` and rebuilds the identical
   decoder head with no benchmark spec and no data-dependent re-derivation. One-
   shot requires a uniform trajectory length (fixed-size head; asserted).
3. **Sampler stride = overlapping (stride 1)** for `k=1` and `1<k<T`; `k=T` is
   one window per trajectory by construction.
4. **Noise/target is a k-regime branch** (below), not a smooth dial.
5. **Remainder = predict-k-and-truncate** (fixed head). The rollout runs
   `k_eff = min(k, n_frames − t)` and truncates the final bundle.
6. **Loss weighting = uniform** (mean over the `(n_free, k)` block); a decaying
   late-frame weighting is a later ablation.
7. **Scope = Transolver only.** The scripted-velocity INPUT feature stays a
   single immediate-next actuator velocity (`node_in` unchanged); k-frame
   kinematic handling is confined to the rollout override and the training seam.
   CGN/MGN/GeoFLARE are untouched (see the neural-CFL audit).

### Neural-CFL audit (derives the Transolver-first scope)

`PhysicsAttentionIrregularMesh` aggregates every node of an example into
`slice_num` tokens, attends across tokens, and de-slices — so a single block's
receptive field is the **whole case domain at every depth, independent of k**.
The neural-CFL condition `receptive field ≥ wave_speed · k · frame_dt` is
therefore cleared at all k for Transolver on both structural benchmarks:

| benchmark | frame_dt | wave speed | k=1 need | k=5 need | Transolver | CGN (10 MP × radius) |
|---|---|---|---|---|---|---|
| Taylor | 2 µs | ~4 mm/µs (Cu) | 8 mm | 40 mm | global ✓ (all k) | 15 mm ✓ k=1, ✗ k=5 |
| notch | 1 µs | ~3–4 mm/µs (concrete) | 3–4 mm | 15–20 mm | global ✓ (all k) | — |

Message-passing backbones would need global mixing or ≈3× more rounds to see
far enough at `k=5`; they are out of first scope **on evidence, not preference**.

### The k-regime noise/target branch

The `velocity_history` flag previously conflated two axes — the velocity-history
INPUT feature and the noise/target SCHEME — which `k>1` forces apart. The
feature is valid at any k; the scheme is chosen by regime:

- **k=1** — the reference recipe, lexically unchanged: single-frame Gaussian
  noise (or the ADR-0049 velocity-history random-walk with the GNS adjusted-next
  target). Same RNG draw order → byte-identical.
- **k=T (one-shot)** — no autoregressive feedback exists, so single-step drift
  noise is meaningless (it would bias one-shot toward over-contraction, the
  ADR-0049 pathology). Clean inputs, clean velocity-history feature, clean
  full-sequence per-frame-velocity L2 (the CarCrashNet regime).
- **1<k<T (bundling)** — robustness lives at the **bundle seam** (no feedback
  *within* a bundle). Following MP-PDE (Brandstetter et al., ICLR 2022): run
  bundle1 forward WITHOUT gradient, decode it to positions, clamp kinematic rows
  back to ground truth (clean actuator input on a moving actuator), seed
  bundle2's input window from that drifted output, and backpropagate **only
  through bundle2** (loss on the clean GT bundle2). During normalizer warmup the
  target normalizer's inverse is untrained, so the step instead trains a plain
  clean bundle1; the pushforward engages once the normalizers are ready.

### k=1 byte-identity invariants (the acceptance bar)

k=1 is byte-identical to pre-0051, contingent on a fixed set of invariants
verified by regression tests: `out_size = k*(dim+1)` collapses to `dim+1`;
the `(P,k,dim+1)` reshape lives simulator-side (network.py untouched) and is a
no-op; the target normalizer stays width `dim+1` fed the 2-D view at k=1; the
sampler keeps the pre-0051 index/target; `_mesh_family_noise`'s k=1 branch is
lexically today's code (same RNG order); the rollout `ndim==2` path is
unchanged; `_advance_pointer`'s new `step=1` default and `WindowDataset`'s
`target_frames=1` default keep every k=1 family (CGN/MGN/GeoFLARE) positional
and unaffected; and a legacy `config.json` lacking the field rebuilds as k=1.

## Alternatives considered

- **A standalone one-shot (`k=T`) model** — rejected: the k-parameterised family
  subsumes it and yields the whole spectrum for a controlled comparison.
- **Storing the raw sentinel `0` in `config.json`** (resolve at point of use) —
  rejected: evaluation's `_model_config_from_record` has no benchmark spec and
  cannot turn `0` into a head shape, and the working length is data-derived (the
  152→151 terminal drop), so re-resolving at eval could disagree with training.
  Storing the resolved integer makes the round-trip deterministic.
- **Feeding k future scripted velocities** (widen `node_in`) — deferred: for
  Taylor the wall is static (zero information); it is a refinement for
  moving-actuator benchmarks.
- **Message-passing bundling (CGN/MGN)** — out of first scope per the neural-CFL
  audit above.
- **Implement 1<k<T with plain input noise instead of the seam pushforward** —
  rejected: bundling's pathology is seam drift, which per-window noise does not
  target; the pushforward is the regime-correct robustness.

## Consequences

- StructBench gains a single-backbone k-sweep across structural impact/fracture
  benchmarks — a cleaner controlled comparison than CarCrashNet (one-shot only)
  or MP-PDE (bundling on fluid PDEs) reports.
- k=1 results (ADR-0044/0047/0048/0049 Transolver runs) remain valid and are the
  `k=1` point of any sweep.
- **Validation plan (signed predictions on record, ADR-0050 physics):** k-sweep
  on Taylor, `k ∈ {1, 5, 0(=T)}`, L=8/M=64, `velocity_history=true`, 2 seeds,
  reduced budget, into `runs/kframes-*`. Predictions: one-step RMSE stays
  ~0.0035 mm across k (const-velocity floor); rollout RMSE **improves** as k
  grows (accumulation drops — the direct SAROS test); `peak_von_mises` /
  `t_peak_von_mises` QoIs may **degrade** at large k (spectral blur, the
  implicit-integrator dissipation analogue — the existing QoI set is the
  discriminator, no new metric); watch `test_extrap` for an OOD signal.
- **Deferred:** (a) notch `k=T` horizon divergence (`scored_frames=250 <
  n_frames=502`): the general rollout loop degrades gracefully — the scored span
  is a genuine single one-shot call, frames beyond it are an AR continuation of
  the fixed head — but a notch-time decision (truncate the eval trajectory vs
  accept the AR tail) remains. (b) `k=T` data sparsity: one window per Taylor
  trajectory ≈ 33 samples, a confound the k=1/k=5 arms do not share; a weak k=T
  result may reflect sample count, not one-shot being wrong. (c) results-registry
  representation of `k` (provisional entries by label, per ADR-0046). (d) feeding
  k scripted velocities for moving actuators.
- The autoregressive task definitions in ADR-0019/0026 are descriptive of the
  *default* scheme, not a constraint on models (the metric is scheme-agnostic);
  those ADRs carry a forward-reference note to this ADR.
