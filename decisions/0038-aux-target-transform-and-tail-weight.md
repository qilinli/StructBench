# 0038 — Auxiliary-channel training knobs: target-space transform and tail weight

**Status**: Draft (agent-proposed; human finalises)
**Type**: Durable
**Date**: 2026-07-15

## Context

The notch-impact capacity fleet (runs/fleet-2026-07-13-cap) exposed a
systematic failure in the auxiliary (max principal strain) channel: every
mp10 arm over-predicts `cracked_fraction` on 12/12 test cases (mean bias
+0.26), and the fringe visualisations show spurious periodic crack
striping. A measured diagnosis (2026-07-15) traced part of the failure to
the loss design rather than capacity:

1. **Loss starvation.** At convergence the training loss splits ~96/4
   between the position (noise-floor-limited denoising) and aux terms, so
   late-training gradients barely serve the strain channel.
2. **MSE geometry on a heavy-tailed target.** The strain field's global
   z-scoring puts the 1% crack threshold at −0.21 σ — *below the dataset
   mean* (0.024 = 2.4× the threshold) — while deep cracks sit at +4…+14 σ.
   Under uncertainty an MSE regressor reverts to the conditional mean,
   which itself counts as "cracked": over-prediction is mechanical.
   Squared error simultaneously spends its budget on the +4 σ tail rather
   than the threshold band the QoI reads.

Rescaling `w_aux` cannot fix the bias (a convex loss's argmin is
invariant to scaling); the target's *geometry* has to change, or the
threshold band has to be re-weighted.

## Decision

Two config-keyed, recipe-recorded training knobs, both defaulting to the
reference behaviour (all existing recipes and checkpoints unaffected):

1. **`[model] aux_transform` / `aux_transform_scale`** — an optional
   target-space transform for the auxiliary channel. `"none"` (default)
   is the reference; `"asinh"` trains against
   `asinh(aux / aux_transform_scale)`: linear below the scale knee,
   logarithmic above it. Normalization stats, the training target, and
   the decoder all live in transformed space; `predict_positions` applies
   the exact inverse after de-normalizing, so rollout, metrics, and QoIs
   keep operating on raw physical units — **the evaluation contract is
   untouched**. The transform is parameter-free (checkpoint state-dicts
   are unaffected), is recorded in `config.json`, and keys its own
   normalization-stats cache entry (`"none"` keeps the legacy cache keys).
   For the notch crack field the natural knee is the QoI threshold
   (0.01 strain).
2. **`[train] aux_tail_weight`** — an optional per-particle weight on the
   aux MSE, `1 + aux_tail_weight · relu(z_target)`, upweighting
   above-mean (tail) targets. `0.0` (default) is the plain reference MSE.
   Weights follow the *target*, never the prediction, so the weighting
   cannot self-reinforce.

Per ADR-0032's exact-keys rule, all grouped configs now carry the three
keys explicitly at their defaults; `ablate_notch_impact.slurm` gains
`AUXT`/`AUXS`/`TAILW` overrides.

## Consequences

- The strain-channel ablation round (transform, tail weight, their
  combination, and a `w_aux` falsification arm) becomes a config-only
  fleet against the same trainer.
- A blessed baseline trained with a non-default transform records it in
  the committed recipe verbatim (ADR-0037 config-rewrite step covers it).
- The deeper fixes this ADR does *not* attempt — an irreversible fed-back
  damage state, strain-increment integration — remain future work
  (candidate v0.3), pending the ablation evidence.

## Alternatives considered

- **Raising `w_aux` alone.** Rejected as the primary lever: it cannot move
  the loss minimizer, only reallocate shared-trunk capacity; kept as a
  cheap falsification arm.
- **rank-Gauss / quantile transforms.** Stronger symmetrizers, but they
  need a stored quantile map and distort out-of-distribution magnitudes
  exactly where the probe geometry already misbehaves; asinh is
  parameter-free and exactly invertible.
- **A cracked/not classification head.** Contract-safe as a training-only
  signal and possibly complementary; deferred until the cheaper knobs are
  measured.
