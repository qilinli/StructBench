# 0060 — Aux channels as model inputs: the state-feedback surface (Transolver AR)

**Status**: Proposed
**Type**: Durable
**Date**: 2026-09-03

## Context

ADR-0059 made the auxiliary target a channel block `(T, P, C)` — state as
*output*. The Stage-3 TC fleet then measured what output-only state buys
(`scratch/2026-09-03-stage3-tc-fleet-results.md`): admissibility becomes
checkable (D2-own) and freely enforceable (projection), and the TC scheme's
snapshot pathology is quantified (peeq decreases on 27–37% of consecutive
frames) — but composed stress is ~75% less accurate than regressed vm and
biased 13% low, and output-only state cannot *help* prediction: the model
never consumes it.

The maintainer's core question — **is the state helpful at all?** — is a
question about state as *input*. The Stage-1/1b probes answered it at one
teacher-forced step (the deviator increment is unpredictable without state
input; the kinematic arm sits at the predict-zero floor), but the pipeline
has no input-side surface: mesh-family node features are kinematic only, and
the rollout feeds back positions alone.

The known hazard (maintainer directive, 2026-09-03): under autoregression,
state feedback couples the experiment to rollout error accumulation, which
can destroy everything downstream and masquerade as "state doesn't help"
(the G3 precedent: a one-step improvement that tripled rollout
accumulation). The evaluation protocol must therefore **isolate the
information effect from the accumulation effect by construction.**

## Decision

**Give the autoregressive Transolver an off-by-default state input:
`TransolverConfig.aux_input`, consuming the run's aux channel block at the
last input frame, with three evaluation modes that decouple information
value from feedback accumulation.**

### Training

- `aux_input: bool = False` (off = byte-identical). When on, the node
  features gain the `C` aux channels (the ADR-0059-resolved selection) at
  the **last input frame**, normalized by the existing online node
  normalizer (whose width grows by `C`). Teacher-forced: the sample supplies
  the ground-truth aux at that frame (`WindowDataset` gains an `input_aux`
  key, additive to the sample contract like `target_frame`).
- **No noise is injected on the state inputs** in this ADR — deliberate:
  first *measure* the accumulation cost cleanly (mode iii vs ii below), and
  only then decide whether stability engineering (state noise, pushforward)
  is warranted. Position noise is unchanged.
- Load-time validation: `aux_input` requires the transolver family, is
  rejected with `time_conditioned = true` (TC has no evolving state to
  consume) and with `frames_per_call != 1` (state feedback through k-frame
  bundles is out of scope), and requires `history_frames >= 0` as usual.

### Rollout — the three modes (the accumulation-isolation design)

For an `aux_input` run, evaluation reports all three:

1. **Teacher-forced one-step** (existing `one_step_*` sweeps): state input is
   ground truth at every step. Pure information value at one step — the
   pipeline-scale replication of the Stage-1 probe.
2. **Oracle-state rollout**: positions roll out (self-fed) but the state
   input is read from ground truth at every step. Isolates the information
   value of state input at rollout horizon with the state-feedback
   accumulation channel *severed*.
3. **Self-fed rollout**: the model's own predicted state feeds back — the
   honest simulator mode.

The pre-registerable decomposition: `(b)-vs-(c@2)` = information effect;
`(c@2)-vs-(c@3)` = accumulation through the state channel alone. Metrics for
mode 2 are recorded under `rollout_oracle_*` keys beside the standard
(mode-3) `rollout_*` keys in the same `metrics-*.json`; mode 3 writes the
rollout artifacts as usual.

Mechanics: `CaseBoundSimulator.bind_case` gains an optional `gt_aux`
trajectory `(T, P, C)`; the transolver keeps its own step counter (anchored
at the window frame count like the pointer, `frames_per_call = 1` only) and
an `aux_feedback` switch (`"self"` | `"oracle"`). Self-fed mode seeds from
the ground-truth state at the last seed frame (initial conditions are
known); `reset_rollout` clears the cached state. One-step sweeps run in
oracle mode by definition.

### Scope

Transolver family only (the experiment vehicle); MGN/GeoFLARE, other
benchmarks, and any stability engineering are out of scope. Launching the
state-helpfulness pair — (b) `aux_input=false` + state outputs vs (c)
`aux_input=true`, seed-matched on the AR recipe — is a separate flag-first
proposal once this surface lands.

## Alternatives considered

- **Feed state at every window frame** (a state history): more input, more
  ways to leak; the Markov claim (C3) is about the last state, and Stage-1
  tested exactly that. One frame, matching the probe.
- **Inject state noise now**: presumes the accumulation problem before
  measuring it; contradicts the measure-first directive. Deferred.
- **Implement on all mesh families**: triples the surface for no additional
  answer to the current question. Deferred.
- **Skip oracle mode** (one-step + self-fed only): loses the decomposition —
  a self-fed failure could not be attributed between information and
  accumulation. Oracle mode is the maintainer's isolation requirement.

## Consequences

- The (b)/(c) pair becomes runnable as configs; its readout cleanly
  separates "state carries information at pipeline scale" from "feedback is
  unstable without stability work".
- **Surface changed**: `config.py` (`aux_input` + validation),
  `datasets/particle.py` (`input_aux` sample key + collate),
  `models/common/simulator_base.py` (optional `gt_aux` bind),
  `models/transolver/simulator.py` (input width, feature part, feedback
  modes, counter), `cli/train.py` (trainer threading; evaluate's dual
  rollout + oracle one-step), tests. **Not touched**: TC paths, the other
  families, ADR-0059 artifacts/metrics at `aux_input = false` (byte-identity
  required), cards/protocols/registries.

## Relationship to other ADRs

- **Complements ADR-0059** (output side); together they close the loop the
  Stage-1 probe tested out-of-pipeline.
- **ADR-0051/0053 respected**: `frames_per_call = 1` required;
  `history_frames` orthogonal and unchanged.
- **ADR-0057 pattern**: off-by-default knob with byte-identity when off.
- The stability question (state noise / pushforward), if mode-3 results
  demand it, gets its own ADR.

## Implementation notes (2026-09-04, post-review amendment)

Three deltas from the multi-agent implementation review, folded in before
the fleet:

- **Kinematic rows of the fed-back state are ground truth.** In self-fed
  mode the cached state overwrites kinematic rows with the GT aux at the
  just-predicted frame — the exact analog of the rollout loop's GT position
  override. Those rows receive no aux training signal (their loss is
  masked; the codebase documents their decoder output as meaningless), so
  feeding the decoder's untargeted output back would contaminate mode 3
  with a train/rollout distribution shift unrelated to state-channel
  accumulation — corrupting precisely the mode-2/mode-3 decomposition this
  ADR exists to protect, and the in-training (self-fed) checkpoint
  selection with it.
- **The state counter fails loud on overrun** (RuntimeError naming
  ``reset_rollout()``), restoring parity with the position pointer's
  tripwire on kinematic-free benchmarks (wave-1D). Like that tripwire, a
  first stale-but-in-bounds read after a missing reset is undetectable in
  principle; the overrun is caught on the next call.
- **Mode-2 (oracle) records the four field metrics only**; QoIs come from
  the canonical self-fed rollout (the oracle pass no longer computes ones
  it would discard). The Surface-changed list additionally includes the 49
  transolver TOMLs/fixtures gaining the explicit ``aux_input = false`` line
  (strict schema, ADR-0057 precedent) and the ADR index row; the
  GeoFLARE-side gt_aux threading briefly introduced by a broad edit was
  removed (scope stays Transolver-only).

## Narrowing note (2026-09-08, ADR-0062)

The `aux_input` + `time_conditioned` rejection recorded above is narrowed
by ADR-0062: the combination is accepted **iff** `flow_map = true` (the
anchored flow map consumes the anchor state through the TC formulation).
Plain time-conditioning still consumes no evolving state, and the
rejection stands whenever `flow_map = false`. An adjustment, not a
reversal — recorded per the index-README amendment mechanism.
