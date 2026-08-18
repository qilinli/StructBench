# 0057 — Transolver++ eidetic-state adaptation (adaptive temperature + Gumbel Rep-Slice)

**Status**: Proposed — drafted by Claude Code, prototype landed on
`feat/adr-0057-transolver-plus`; the human finalises
**Type**: Durable
**Date**: 2026-08-18

## Context

A zero-GPU diagnostic on the blessed time-conditioned Transolver checkpoints
(`scratch/2026-08-17-slice-degeneration-diagnostic.md`) measured KL(slice-weights
‖ uniform) — the Transolver++ Table-4 quantity — and found the "uniform slice
weight → average pooling" degeneration (arXiv:2502.02414) present and seed-robust
on our impact benchmarks: **4 of 8 Physics-Attention blocks fully collapse to
average pooling** in every seed×split (overall normalized KL/logM 0.13–0.27).
Two facts frame the response:

- The degeneration is real even at our small (5–13k-point) meshes, despite the
  paper motivating large meshes. It is a plausible in-distribution accuracy loss.
- It is **decoupled from the off-grid notch probe** (probe KL ≥ in-dist, both
  seeds), so this is an **in-distribution accuracy lever, not the probe fix**
  (the probe stays the local-branch/geometry story — separate work).

Transolver++ (thuml, MIT) attacks exactly this pathology with two edits inside
Physics-Attention's slice-weight path. The mechanism was verified against the
**released code** (`models/Transolver_plus.py`), which differs from the paper
text: the paper writes the adaptive temperature as `τ = τ₀ + Linear(x)`, but the
code uses a 2-layer MLP with a learned per-head bias and a positivity clamp.

## Decision

Implement both edits as **two independent, off-by-default config knobs** on the
native Transolver family, **byte-identical when off**, as a native clean-room
reimplementation in `models/` (no upstream runtime dependency; ADR-0041/0044).

### The two edits (following the released CODE, not the paper text)
1. **`adaptive_temperature`** — replace the single learned per-head temperature
   scalar with a per-point, per-head temperature:
   `τ = clamp(MLP(x_mid) + bias, min=0.01)`, where `MLP = Linear(dim_head, M) →
   GELU → Linear(M, 1) → GELU` and `bias` is a per-head parameter (init 0.5).
2. **`slice_reparam`** — add Gumbel(0,1) noise to the slice logits before the
   temperature-divide + softmax (differentiable categorical sampling).

### Settled decisions
1. **D1 — two independent knobs.** `adaptive_temperature` and `slice_reparam`
   are separate booleans so each edit is ablatable (which one, if either, helps
   on our small meshes is an open empirical question).
2. **D2 — Gumbel is TRAIN-ONLY** (`self.training`-gated); inference is
   deterministic. This is a **declared deviation** from upstream (which samples
   Gumbel at eval too), required by StructBench's eval-determinism/reproducibility
   contract (metrics, gates, the KL diagnostic). Faithful eval-Gumbel is recorded
   as a future ablation, not adopted.
3. **D3 — follow the released code** (MLP + per-head bias + clamp), not the
   paper's single-`Linear` text.
4. **D4 — in-place, off-by-default.** Both edits live inside
   `PhysicsAttentionIrregularMesh`; exactly one temperature parameter set is
   created per mode (scalar `temperature` when off, `proj_temperature` +
   `temperature_bias` when adaptive) so each mode's `state_dict` stays clean.
   Precedent: `time_conditioned`, `impact_velocity_feature`, `history_frames`.

### Surface
`TransolverConfig` gains two `bool = False` fields, threaded config → simulator →
net → block → attention. All 30 `configs/*/transolver*.toml` carry both keys
(the strict loader requires a complete `[model]`); legacy checkpoints reconstruct
with the defaults (`config.json` bypasses the strict TOML loader via dataclass
defaults). **Off-path proven byte-identical on a blessed checkpoint**: the
slice-weight KL is unchanged (0.540 for `taylor-transolver-tc-s2` val case 0,
matching the pre-change diagnostic exactly).

## Consequences

- **+** Attacks a measured, seed-robust degeneration with a published,
  MIT-licensed method; cheap (~10 lines of mechanism); each edit ablatable;
  composes with time-conditioning, k-frames, and a future local branch.
- **−** The paper motivates large meshes; the payoff on our small impact meshes
  is an empirical question the training arms must answer. The train-only-Gumbel
  choice is a declared deviation from upstream.
- Does **not** address the off-grid probe (decoupled; separate local-branch work).

### Validation status
Framed as an in-distribution accuracy lever. Training arms (Taylor + notch:
both-on / adaptive-only / reparam-only, ≥2 seeds each, vs the blessed
Transolver) are a **separate flag-first compute proposal**; results land
provisional per ADR-0046 until blessed.

## Alternatives considered
- **Faithful eval-Gumbel** — rejected for now (breaks eval determinism); kept as
  an ablation.
- **Single combined `transolver_plus` flag** — rejected (cannot ablate the two
  edits independently).
- **Our own anti-collapse treatments** (block-agnostic slice-diversity /
  orthogonality regularizer; physics-informed temperature from a contact /
  strain-rate proxy) — deferred; this ADR establishes the published reference
  baseline first, against which a novel treatment is measured.
