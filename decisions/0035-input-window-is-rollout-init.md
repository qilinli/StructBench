# 0035 — The model input window is the rollout init (no history backfill)

**Status**: Accepted
**Type**: Durable
**Date**: 2026-07-07

## Context

ADR-0032 §4 made the benchmark protocol's `init_frames` (ground-truth frames
observed at rollout start) a separate quantity from the model's input window
(history length), and bridged the gap at rollout with a **constant-velocity
backfill**: when the window was longer than the observed prefix, the missing
history was fabricated by extrapolating the first observed velocity backward.
The Taylor protocol pinned `init_frames = 3` against a model `window = 11`.

Three problems surfaced (2026-07-07 review):

1. **The backfill assumes rigid pre-init motion, which is not general.** It is
   exact only while the observed prefix is constant-velocity. That holds for
   Taylor (free flight until first wall contact ~frame 7; 0.0% of KE dissipated
   within a 3-frame prefix in all 33 cases). It is **false for the wave
   benchmark**: the committed timeline (`docs/timelines/wave_propagation_1d.md`,
   2026-07-06) shows 1.3–3.7% of initial KE already dissipated within a 3-frame
   prefix, so the 8 backfilled frames are a rigid past the wave never had.
2. **The backfilled window is out-of-distribution.** Training only ever sees
   windows of `window` consecutive *real* frames (`WindowDataset`), so wherever
   the prefix is non-rigid the warm-start input the model is asked to predict
   from at rollout start is unlike anything in training — a train/rollout seam
   mismatch on top of the fabricated dynamics.
3. **The two reported metrics covered different spans.** The teacher-forced
   one-step diagnostics scored `[window, end]` while the rollout scored
   `[init_frames, end]`, so within one evaluation the one-step and rollout
   position RMSE were not over the same frames.

The GNS reference (Sanchez-Gonzalez et al. 2023, App. C) uses `C = 5` input
velocities and, because velocity is a finite difference, needs the most recent
`C + 1 = 6` positions. A model that needs 6 positions of history should be
handed 6 observed frames — not 3 plus a fabricated backfill.

## Decision

### 1. Input window == rollout init; no backfill

The number of input frames a model consumes **is** the number of ground-truth
frames it observes to seed a rollout. There is one quantity, `input_frames`:
the observed prefix *is* the input window. A rollout observes exactly
`input_frames` real frames and predicts `[input_frames, end]`, scored there.
The constant-velocity backfill of ADR-0032 §4 and the window/init decoupling
are removed.

### 2. Benchmark-owned; the model must match

`input_frames` is benchmark protocol, pinned on the card. A run whose
`[model].input_frames` differs from its benchmark card's is **rejected at
config load** (and re-checked in `train()`). This keeps ADR-0032 §4's
anti-tuning property — the observation budget is fixed per benchmark, identical
across all models — while removing the separate `init_frames` config concept.
The `[protocol]` research override of ADR-0032 §4 is removed.

### 3. Naming and records

The model config field `window` and the card field `init_frames` are both
renamed to `input_frames` (positions; the network consumes `input_frames − 1`
velocities = `C`). Run records store `model.input_frames` and
`protocol.input_frames`; `read_run_record` renames the legacy `window` /
`init_frames` keys of pre-0035 records so the fleet stays evaluable — a legacy
run seeded its rollout with its window, which is exactly the new rule.

### 4. `input_frames = 6` (C = 5), per-benchmark rationale

- **Taylor**: 6 observes 0.0% of impact KE (first contact ~frame 7).
- **Wave**: 6 observes 14.8% of initial KE worst-case but stays before the
  wave's first-gauge arrival (~frame 7), so the `arrival_time` QoI is predicted,
  not observed. 6 is the ceiling for this benchmark's gauge geometry.
- **Notch (bend/impact)**: `input_frames = 6` provisionally; their mandatory
  timeline analyses (ADR-0032 §5) still gate the first trained baseline.

### 5. One-step diagnostics share the rollout span

The teacher-forced one-step metrics now seed at `input_frames`, so one-step and
full-rollout metrics score the same `[input_frames, end]` span.

## Alternatives considered

- **Keep the ADR-0032 backfill**: rejected — it fabricates non-rigid history
  (demonstrably wrong for the wave) and evaluates the model on an input
  distribution it never trained on.
- **Model-owned `input_frames` (init follows window, no card constraint)**:
  rejected — it lets a model choose its own observation budget, reopening the
  protocol-tuning hole ADR-0032 §4 closed. The card owns the number; the model
  must match.
- **Store `C` (velocities = 5) rather than frames**: rejected — the unified
  quantity is also the rollout seed *count*, which reads naturally in frames;
  `C = input_frames − 1` is documented on the field.

## Consequences

- Absolute metrics shift again versus the init = 3 protocol (init 3→6, backfill
  removed). The ADR-0032 relative fleet conclusions (capacity dominant, w_aux
  load-bearing, noise 0.02) still stand.
- Protocol-sensitivity studies via a `[protocol]` override are gone. To study a
  different observation budget, train at a different `input_frames` — the
  checkpoint architecture is tied to it, so it cannot be re-evaluated at another
  budget without retraining.
- The wave hands the model 14.8% of KE at 6 frames (vs 3.7% at 3), accepted
  deliberately as the ceiling before first-gauge arrival.
- The rename touches the configs, cards, `config.py`, `rollout.py`, `train.py`,
  `render`/`timeline`/`metrics`, `WindowDataset`, `viz`, and the test suite.
  Pre-0035 run records stay evaluable via `read_run_record` key normalization.
- Amends ADR-0019 (Taylor eval protocol) and ADR-0032 §4/§6/§7 (protocol
  governance): the model-independent-init mechanism and the constant-velocity
  warm-start are amended by this ADR.
