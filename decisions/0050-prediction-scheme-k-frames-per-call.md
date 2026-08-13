# 0050 — Prediction-scheme axis: unified k-frames-per-call (autoregressive ↔ bundled ↔ one-shot)

**Status**: Proposed — **BACKLOG** (recorded for the decision log; not scheduled, and NOT approved for implementation. Warrants further discussion and analysis before it becomes an active ADR.)
**Type**: Durable
**Date**: 2026-08-13

## Context

All StructBench native surrogates predict **autoregressively**: given a
history window they emit the next single frame, integrate one step, and roll
that forward (ADR-0044/0045/0047/0049; the benchmarks are defined as
"autoregressive next-step surrogate" tasks, ADR-0019/0026). A recurring
question — surfaced 2026-08-13 while reviewing the Transolver results — is
whether a **one-shot** scheme (predict the whole trajectory in one forward
pass) is better or worse than autoregressive on our impact/fracture
benchmarks, and whether the two are worth comparing head-to-head on the same
backbone.

The one-shot and autoregressive schemes are the two ends of a single axis —
**k frames predicted per forward call**:

| k | scheme | precedent |
|---|--------|-----------|
| `k = 1` | pure autoregressive | our current native families; GNS/MGN |
| `1 < k < T` | temporal bundling | MP-PDE, Brandstetter et al., ICLR 2022 (arXiv 2202.03376; they used k=5) |
| `k = T` | one-shot full trajectory | CarCrashNet / CrashSolver (arXiv 2605.07098): emits `Û^(1:n)` in one pass |

Prior art (grounded 2026-08-13):
- **CarCrashNet** *uses* one-shot full-sequence output but does **not** itself
  ablate one-shot vs autoregressive — so the head-to-head comparison would be
  a StructBench contribution, not a reproduction.
- **MP-PDE temporal bundling** is exactly the "k frames per call" middle
  ground; its stated benefit is fewer inference calls → fewer distribution-
  shift boundaries → less error accumulation.
- The tradeoff is **regime-dependent** (e.g. "No Free Lunch in Flow
  Surrogates", arXiv 2607.23667): one-shot/direct tends to win
  boundary-/forcing-driven transients (no rollout accumulation), while
  autoregressive wins self-sustained systems needing phase memory. Taylor and
  notch are impact-*driven* transients, so larger k is *hypothesised* to help
  — a testable prediction, and the reason the study is interesting.

This ADR records the design direction and a pipeline-impact analysis so the
idea and its cost are not lost. It does **not** authorise implementation.

## Decision

**Defer. Record, do not build.** When picked up, the intended shape is a
**single model family with a `frames_per_call = k` config knob** (default
`k = 1`, byte-identical to today), from which all three schemes fall out —
rather than a separate one-shot model. Transolver is the natural first
backbone (operator-native; k=T is its literature-home mode). The first study
would be a k-sweep (e.g. `k ∈ {1, 5, T}`) on Taylor + notch comparing rollout
RMSE, QoIs, and inference cost against the blessed CGN baseline and the k=1
native.

The axis is **orthogonal to the benchmark protocol** — which is why it is
recordable cleanly and why k=1 is a strict generalisation:

- `input_frames` (input history / rollout seed, ADR-0035) is unchanged; k is
  output-side. The card `input_frames` check (`config.py`) is untouched.
- Metrics, QoIs, and the scored-span bookkeeping (ADR-0039) are
  scheme-agnostic — they consume `(T, P, …)` arrays regardless of how the
  frames were generated.
- Splits, horizon, and the seed→[input_frames, end] scoring convention do not
  move.

## Pipeline-impact analysis (why this is ADR-0044/0047-scale, not a knob)

| Component | Current | With k>1 | Difficulty |
|-----------|---------|----------|------------|
| Decoder head (`forward_train`/`predict_positions`) | 1 frame `(P, dim+aux)` | k frames `(P, k, dim+aux)` | core change |
| Training sampler (`WindowDataset`) | target = `positions[t]`; index `range(input_frames, T)` | target = `positions[t:t+k]`; index `range(input_frames, T-k+1)`; stride choice | moderate |
| Collates | `next_position (ΣP, dim)` | `(ΣP, k, dim)` + `(ΣP, k)` aux | mechanical |
| **Noise + target coupling** (`_mesh_family_noise`, ADR-0049) | GNS single-step; target adjusted by `noise[:,-1]` | **semantics change with k** (below) | **hardest** |
| Loss | one-step L2, kinematic-masked | k-frame L2; per-frame weighting?; normalise per-*velocity* not per-absolute | moderate |
| Rollout loop (`eval/rollout.py`) | step ×1, kinematic override per frame, slide ×1 | step ×k, override all k kinematic frames, re-seed, remainder | rewrite |
| Scripted-velocity + tripwire (`simulator_base`) | 1 next-step scripted velocity; pointer +1 | k future scripted velocities (known for kinematic bodies); pointer +k | moderate |
| `train_frames` truncation guard | `> input_frames+1` | must leave k target frames (`+k`) | trivial |
| Config schema (3 dataclasses, strict loader) | — | add `frames_per_call`; k=T sentinel (horizon varies: Taylor 145 vs notch 244) | moderate |
| Checkpoint | fixed head | head shape depends on k; record in `config.json` | trivial |
| Memory | 1-frame activations | ~×k decode activations; k=T on notch (8k nodes × 244) is large — MGN OOM risk | watch |

### The load-bearing subtlety: noise injection is k-dependent

Noise injection simulates *rollout drift*; that rationale changes with k:
- **k=1** — keep today's GNS single-step noise.
- **k=T (one-shot)** — no autoregressive feedback exists, so the single-step
  machinery is moot; training collapses to plain full-sequence L2 on clean
  inputs (the CarCrashNet regime).
- **1<k<T (bundling)** — no feedback *within* a bundle, but the next bundle is
  seeded from the previous bundle's drifted output, so robustness moves to the
  **bundle seam** — the MP-PDE "pushforward" trick, not per-frame Gaussian
  noise.

So `frames_per_call` selects a **noise/target strategy** (single-step →
pushforward-at-seams → none), a branch rather than a smooth dial. This is the
part most in need of design before any code.

## Open design decisions (must be resolved before implementation)

1. **Output representation** — per-frame velocities (integrate; keeps every
   target at one-step scale so the target normaliser stays valid) vs absolute
   displacements from `x_last` (growing magnitude breaks normalisation).
   Velocities is the presumed answer; confirm.
2. **k=T sentinel** — horizon varies per benchmark, so config cannot hardcode
   145/244; likely `frames_per_call = 0` → "resolve to the scored horizon at
   runtime from the card".
3. **Sampler stride** — overlapping (stride 1, keeps sample density) vs
   non-overlapping (stride k, matches inference tiling).
4. **Noise/target strategy per k-regime** (above) — the hardest call.
5. **Remainder** when k does not divide `(horizon − input_frames)` — short
   final bundle vs predict-k-and-truncate.
6. **Loss weighting** across the k horizon — uniform vs decaying for harder
   late-in-bundle frames.
7. **Scope** — Transolver first; whether CGN/MGN ever bundle (message-passing
   backbones can, but out of first scope).

## Alternatives considered

- **A standalone one-shot model (k=T only)** — rejected as the build target: a
  k-parameterised family subsumes it (k=T is the top of the range) for the
  same effort and yields the whole spectrum for a controlled comparison.
- **A separate "one-shot" benchmark variant** — rejected: the metric is
  scheme-agnostic, so no new benchmark is needed; only a task-definition note
  ("autoregressive" in ADR-0019/0026 is descriptive of the default scheme, not
  a constraint on models) plus the new model family.
- **Implement now** — rejected by the maintainer (2026-08-13): promising but
  not top priority; the noise-strategy design and the memory ceiling on k=T
  warrant discussion first.

## Consequences

- The design direction and its full pipeline cost are on record; picking it up
  later starts from this analysis rather than a cold re-derivation.
- Nothing in the pipeline changes now; the current autoregressive results
  remain valid and would become the `k=1` point of any future sweep.
- Promotion path: this backlog ADR is superseded by (or amended into) an
  active implementation ADR once the seven decisions are settled and the work
  is scheduled.
- If pursued, the payoff is a single-backbone k-sweep across structural
  impact/fracture benchmarks — a cleaner controlled comparison than CarCrashNet
  (one-shot only) or MP-PDE (bundling on fluid PDEs) report.
