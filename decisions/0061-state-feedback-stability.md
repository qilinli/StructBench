# 0061 — State-feedback stability: input noise and pushforward on the state channel

**Status**: Proposed
**Type**: Durable
**Date**: 2026-09-04

## Context

The state-helpfulness pair (`scratch/2026-09-04-state-pair-results.md`)
measured, via the ADR-0060 accumulation-isolation modes, that state input
makes the AR Transolver ~2.5–3× better on the state fields at one step AND
at full rollout horizon with clean (oracle) state — but self-fed feedback is
*worse than no state input at all* (pooled aux RMSE 35–44 vs the oracle
ceiling 9–12 and the output-only arm's 28–30). Pre-registered branch 2:
the information is real; feedback needs stability work.

Two probes then characterised the failure before choosing mechanisms
(`scratch/2026-09-04-feedback-stability-probes.md`):

- The drift is **variance-type and in-manifold**: peeq error grows
  monotonically (0.06 → 0.56) with near-zero bias, while the fed state's
  yield-violation rate stays flat (~9–12%) — no unphysicality spiral. The
  state-input advantage is visible early (frame 20) before accumulation
  crosses over.
- **Projection-in-the-loop closes only ~5–10% of the gap** (measured on the
  trained checkpoints, no retraining): safe (the monotone-peeq clamp does
  not ratchet), free, insufficient. The gap must close through *trained*
  contraction of perturbed states.

The two mechanisms that target exactly this failure class are training-time
noise on the state input (the GNS lineage's answer for positions, load-
bearing throughout this codebase) and the pushforward trick (train on the
model's own detached one-step state error — the modern PDE-rollout answer,
with an adaptable seam already in-repo from ADR-0051).

## Decision

**Add two off-by-default, independently-ablatable training knobs to the
`aux_input` path of the AR Transolver; select between them by fleet
(pre-registered separately), with the oracle-vs-self gap as the target.**

### Knob 1 — `TransolverConfig.aux_input_noise_std: float | tuple = 0.0`

Gaussian noise on the teacher-forced `input_aux` during training,
**scaled per channel by the batch's per-channel standard deviation** of the
state input: `input_aux += knob * std_batch_per_channel * N(0, 1)`.

- Relative (batch-std) scaling rather than absolute units: the six state
  channels span MPa, dimensionless, J, and kg/m³ — one absolute scalar is
  meaningless across them, while one relative scalar is sweepable; a
  per-channel tuple refines it (ADR-0059 scalar-or-per-channel convention,
  `expand_aux_knob`).
- Injected in the training loop next to the position-noise call (the
  house pattern), on the raw `input_aux` before `forward_train`. The target
  stays the clean GT state at `t` — the model learns to contract perturbed
  states toward the truth. Position noise is untouched and orthogonal.
- `0.0` (default) is byte-identical. Requires `aux_input = true`; rejected
  otherwise at config load.

### Knob 2 — `TransolverConfig.aux_input_pushforward: bool = False`

Two-step training chains on the **state channel only** (positions stay
teacher-forced — position exposure bias is already handled by the recipe's
position noise, and entangling the two would blur attribution):

1. Step A: standard teacher-forced step — window ending at `t−1`, state
   input `aux[t−1]`, predicting the state at `t`.
2. Step B: window slid one frame (GT positions), state input = **step A's
   predicted state, detached** (the pushforward trick — no gradient through
   step A), predicting `t+1`.
3. Loss: the mean of both steps' standard losses — step A keeps the
   teacher-forced signal, step B trains contraction of the model's own
   one-step state error distribution.

- Sampling reuses `WindowDataset(target_frames=2)` (the ADR-0051 seam's
  convention); ~2× forward cost per optimizer step.
- Requires `aux_input = true` and `frames_per_call = 1`; rejected otherwise.
  Composable with knob 1 in principle (both off, either, or both — the
  fleet tests them separately first).
- `False` (default) is byte-identical.

### Explicitly not in this ADR

- **In-loop projection productization**: measured at ~5–10% of the gap; the
  probe harness (`scratch/feedback_projection_probe.py`) remains available
  for eval-time use, but no `aux_feedback="self-projected"` mode ships until
  a trained mechanism makes the residual gap worth polishing.
- **Structured heads**: admissibility is not the failure driver (probe 0);
  they remain a C2 device, not a stability device.
- **Scheduled sampling / full BPTT / damped feedback / k-bundle or flow-map
  feedback-frequency changes**: set aside in the candidate discussion
  (2026-09-04) — dominated, scope-heavy, or previously shelved.

## Alternatives considered

- **Absolute-units noise scale** (like position `noise_std`): rejected —
  not meaningful across mixed-unit channels; the batch-std-relative form
  gives one sweepable scalar.
- **Noise in normalized feature space**: rejected — injection after the
  online normalizer would entangle the knob with normalizer warmup state;
  raw-space injection keeps the house pattern (noise before features).
- **Loss on step B only** (pure pushforward): rejected for the first fleet
  — halving the teacher-forced signal is a second simultaneous change;
  both-steps loss changes exactly one thing (the added contraction term).
- **Jump straight to the better-guessed mechanism**: rejected — the pair of
  knobs is cheap, and the fleet decides on measurement (house rule).

## Consequences

- The stability question becomes a config-level fleet on the existing
  `statefb` recipe; the pre-registered target is closing the oracle-vs-self
  gap (9–12 vs 35–44 pooled aux RMSE), with oracle mode the measured
  ceiling and one-step/oracle metrics guarding against mechanisms that
  destroy the information effect while stabilising.
- **Surface changed** (on acceptance): `config.py` (two fields +
  validation), `cli/train.py` (noise injection + the two-step chain in the
  transolver AR loop; `WindowDataset(target_frames=2)` wiring when
  pushforward is on), tests (byte-identity when off; noise determinism
  under seed; chain shapes). **Not touched**: the ADR-0060 eval modes and
  metrics keys, simulators' predict paths, other families, TC paths,
  benchmarks/registries.

## Relationship to other ADRs

- **Builds on ADR-0060** (the aux-input surface and its eval modes are the
  measurement instrument) and **ADR-0059** (per-channel knob conventions).
- **Adapts the ADR-0051 seam** (two-target sampling / detached second step)
  to the state channel.
- **ADR-0049 precedent** for noise-scale decisiveness: treat the noise
  scale as a sensitive knob; the fleet sweeps two scales rather than
  trusting one.
