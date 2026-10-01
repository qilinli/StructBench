# 0050 — Prediction-scheme axis: unified k-frames-per-call (autoregressive ↔ bundled ↔ one-shot)

**Status**: Superseded by 0051 (was: Proposed — BACKLOG)

> **Superseded (2026-08-15):** the seven open decisions below are settled and the
> scheme is implemented, validated, and merged for the Transolver family — see
> **ADR-0051** (k-frames-per-call implementation, Accepted). This ADR is retained
> as the design-rationale and physics record (the explicit/implicit analogy,
> neural-CFL condition, and regime hypothesis are not repeated in 0051).
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

## Physics grounding (added 2026-08-14)

*Recorded from the 2026-08-14 in-session discussion of how structural systems
evolve in time; the fuller narrative lives in the maintainer's research notes.
This section sharpens the hypothesis and derives part of the scope.*

**The k-axis is the explicit↔implicit axis, relearned.** Explicit vs implicit
time integration is at bottom a trade between how far information travels per
step and how large a step is allowed: explicit updates are local and
CFL-bound (the scheme's information speed must beat the physical wave speed);
implicit steps couple every DOF through a global solve, buying unconditional
stability at the price of that solve. The neural analogue is exact — local
message passing has explicit-like information speed, global attention is
implicit-like — but a learned stepper is neither integrator: it is an
**amortized flow map**, taking implicit-scale steps (the 2 µs Taylor output
frame spans on the order of 20–50 of LS-DYNA's own CFL-bound substeps) at
explicit-scale per-call cost. Learning decouples what classical schemes lock
together: step span (k) becomes a config knob, stability becomes a
training-time property (the noise/pushforward branch below is numerical
dissipation's training-time analogue), and information speed becomes an
architecture property.

**Neural CFL condition.** Per call, the model's receptive field must cover
the physical domain of influence of the frames it predicts:
`receptive field ≥ wave speed × k × frame dt`. Taylor worked example: copper
bulk sound speed ≈ 4 mm/µs × 2 µs ≈ 8 mm of influence per frame, against the
blessed CGN's 10 message-passing steps × 1.5 mm connectivity radius = 15 mm
receptive field — k=1 clears the condition with ~2× margin. At k=5 the last
bundled frame needs ~40 mm ≈ 27 message-passing rounds: local message passing
cannot see far enough. This *derives* the Transolver-first scope (open
decision 7) rather than merely preferring it, and makes a per-benchmark,
per-backbone neural-CFL audit (wave speed, frame dt, hops × radius) a cheap
prerequisite for any k>1 work.

**Regime grounding of the hypothesis.** Taylor and notch are impact-driven
and strongly dissipative — plasticity and fracture contract the dynamics;
nothing self-sustains and no phase-critical resonance must be tracked. The
dominant pathology is therefore rollout error accumulation, which larger k
removes by construction: "larger k helps here" follows from contractivity,
not only from the No-Free-Lunch pattern cited above. The argument is
regime-specific and *flips* for phase-critical, forcing-driven-throughout
problems (vibration/seismic), where stepping or forcing transduction should
win — so the k-sweep's conclusion is not expected to transfer across regimes,
which is itself the publishable framing.

**Predicted failure mode, with the sentinel already in protocol.** Large k
trades accumulation error for spectral blur — the same trade implicit
integrators make via numerical dissipation. Prediction: k=T improves rollout
position RMSE but degrades `peak_von_mises` / `t_peak_von_mises`, exactly the
QoIs the Taylor protocol already flags as penalising temporally coarse
surrogates. The existing QoI set is the discriminator; no new metric is
needed.

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
7. **Scope** — Transolver first; whether CGN/MGN ever bundle (per the
   neural-CFL analysis above, message-passing backbones would need global
   mixing or ~3× more rounds to see far enough at k=5 — out of first scope).

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
- The physics grounding (2026-08-14) turns the sweep from exploratory into
  hypothesis-driven: signed predictions — per regime (impact-dissipative:
  larger k wins) and per metric (rollout RMSE improves, peak-stress QoIs
  degrade) — are on record before any implementation.

## Amendment (2026-08-15): source corrections — Transolver's structural home mode is time-conditioned querying, not k=T

*Recorded after verifying today's sources against the primary code and paper
(maintainer-approved session, 2026-08-15). This note corrects the record this
ADR carries; the forward-looking counterpart (the fourth scheme, its scope and
its boundary) is an amendment on ADR-0051.*

- **Correction to "k=T is its literature-home mode" (Decision section).**
  Verified against thuml/Transolver's own code
  (`PDE-Solving-StandardBenchmark/exp_plas.py`): its only structural-deformation
  benchmark (Plasticity — a die forming an elastoplastic block) predicts each of
  the 20 output frames **independently** via `model(x, fx, T=t)` — geometry-only
  input, a scalar time token per query, no feedback, no stacked trajectory head.
  That is **time-conditioned querying** (a direct operator `u(t) = F(IC, BC, t)`;
  atlas Axis F4), not k=T one-shot. The stacked-channel k=T pattern is the
  CarCrashNet / GeoTransolver-crash-wrapper form. (The same repo's Navier–Stokes
  benchmark is k=1 autoregressive over a 10-frame history window — within one
  architecture the scheme choice tracks the problem regime, not the method.)
- **Correction to the 2026-08-13 session recollection** that the NVIDIA crash
  study "found time-conditioning slightly more accurate than one-shot". Verified
  against the paper (arXiv:2510.15201, NVIDIA + General Motors, Body-in-White
  frontal crash on LS-DYNA data): its three arms are time-conditional (TC),
  autoregressive with one-step training (AR-OT), and autoregressive with rollout
  training (AR-RT) — there is **no one-shot arm**. Finding (Results Part 3,
  Transolver backbone): AR-OT yields noticeably higher rollout L2 position error;
  **TC achieves accuracy comparable to AR-RT with smaller standard deviation**.
  The paper's stated TC weakness: it "fundamentally ignores causal dependencies
  between states, and thus tends to perform poorly when extrapolating or
  forecasting beyond the training time horizon."
- **Why this strengthens rather than weakens the regime hypothesis:** on an
  impact-dissipative crash problem a history-free scheme matched the best
  *stabilised* stepper while the unstabilised stepper lost — i.e. scheme
  comparisons are stabiliser-confounded (each scheme must carry its own
  numerical-dissipation analogue), which is exactly the k-regime noise/target
  branch this ADR identified as the load-bearing subtlety.
