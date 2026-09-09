# 0063 — Anchor-interface contraction training (flow map)

**Status**: Proposed
**Type**: Durable
**Date**: 2026-09-10

## Context

The two ADR-0062 fleets (18 runs) plus the budget fleet reduced the
flow-map question to a single mechanism. The information is there:
at 66-epoch coverage the Δt≤15 operator reaches oracle-anchored pooled
aux 14.3 at m=5 (cash +0.70 vs the TC control's 25.0) — cashing even
half of it beats TC outright, displacement included. What blocks it is
the anchor hand-off: self-anchored the same model sits at 30.1
(budget-fleet branch 2: coverage fixes within-segment accuracy, anchor
feedback binds). Budget, Δt-range, conditioning scalars, aux-input
noise, and architecture were each eliminated by dedicated arms.

The phase-0 attribution probe
(`scratch/2026-09-10-anchor-attribution-results.md`, eval-only hybrid
anchors on the best operator, sanity-matched to the fleet numbers)
split the hand-off damage BY FIELD, roughly additively:

- **Displacement collapse is ~100% the kinematic channel**: GT anchor
  kinematics alone restores oracle-level displacement (0.076 → 0.0059);
  GT aux does nothing for it.
- **~78% of the state-field gap is the fed-back aux channels**
  (30.30 → 17.82 with GT aux; GT kinematics alone reaches only 25.10).
- The free eval-time repair (least-squares velocity smoothing) is dead
  (~5%) — the fix must be trained, not filtered.

The model has never been *trained* to consume an imperfect anchor on
either channel: anchors are teacher-forced GT at train time, with
ADR-0061 noise on the aux block only (measured inert under
re-anchoring, FM-N0 ≈ FM-FULL). This ADR adds the two training
mechanisms the attribution motivates — one per damage pathway, plus
their joint form.

## Decision

**Add two off-by-default, independently-ablatable contraction knobs to
the flow-map training path; select by fleet (pre-registered separately)
with the oracle-vs-self gap at m ∈ {5, 15} as the target.**

### Knob 1 — `TransolverConfig.flow_map_pushforward: bool = False` (primary)

Two-hop training chains on the ANCHOR interface — the flow-map-native
pushforward (ADR-0051/0061 lineage), training on the model's own joint
anchor-error distribution rather than a noise proxy:

1. Sample a chain `(t₀, t₁, t₂)`: anchor `t₀` uniform over valid
   anchors; hop 1 `t₁ − t₀ ∈ [2, Δt_max]` (≥ 2 so BOTH frames of the
   constructed anchor pair are predictions — the dominant rollout
   regime); hop 2 `t₂ − t₁ ∈ [1, Δt_max]`; uniform over valid triples.
2. **Step A** (teacher-forced): from the GT anchor at `t₀`, TWO queries
   predict the full state at `t₁ − 1` and `t₁` — exactly the pair a
   rollout re-anchor consumes (the second query exists to form the FD
   velocity).
3. **Re-anchor on the predictions, DETACHED** (the pushforward trick —
   no gradient through step A): anchor displacement/velocity from the
   two predicted positions, anchor aux from the predicted state at
   `t₁`, with the house clamps (kinematic rows of positions and aux
   overwritten with GT — those rows are prescribed/untargeted, exactly
   as the rollout feeds them).
4. **Step B**: from the self-anchor, predict the state at `t₂`. Loss =
   mean over the THREE query losses (steps A keep the teacher-forced
   signal; step B trains contraction of the joint anchor error —
   both-steps loss, the ADR-0061 one-change convention). Normalizers
   accumulate on the clean step-A queries only.

~3× forward cost per optimizer step. Requires `flow_map = true`;
composable with knob 2 (the fleet tests each alone and the pair).

### Knob 2 — `TransolverConfig.flow_map_anchor_noise_std: float = 0.0` (comparator)

GNS-style Gaussian noise on the anchor KINEMATICS at train time:
independent noise on both frames of the anchor position pair (so
displacement AND the FD velocity are corrupted consistently with how
rollout errors enter), absolute scale in working units (mm — positions
are single-unit, the house `noise_std` convention; the ADR-0061
batch-std-relative form exists for the mixed-unit aux block and is not
duplicated here). Kinematic ROWS stay clean; targets stay clean GT.
Per the attribution this knob alone is worth at most TC-parity on the
state fields plus the displacement fix — it is the cheap comparator
and the kinematic half of the joint arm, not the headline mechanism.
Scale is treated as decisive (ADR-0049 precedent): the fleet sweeps two
scales rather than trusting one. Requires `flow_map = true`; `0.0`
(default) byte-identical.

### Explicitly not in this ADR

- Aux-channel-only mechanisms (a separate aux pushforward, projection
  in the hand-off): knob 1 already contracts the aux channel jointly;
  a dedicated aux device is priced only if the fleet shows kinematics
  fixed but aux stuck.
- Δt-distribution reweighting / curriculum (the dilution levers): a
  separate, secondary axis — irrelevant until the hand-off survives.
- Eval-time anchor filtering: measured dead (smooth3, phase 0).
- Any change to the ADR-0062 eval surface, metrics keys, or schemes.

## Alternatives considered

- **Noise on the anchor aux instead of chains**: already exists
  (ADR-0061 `aux_input_noise_std`, on in every fleet arm at 0.15) and
  is measured inert under re-anchoring — the aux damage is structured
  model error, which Gaussian noise evidently does not rehearse.
  Chains train on the actual error distribution.
- **Loss on step B only** (pure pushforward): rejected as in ADR-0061 —
  halving the teacher-forced signal is a second simultaneous change.
- **Chains with hop 1 = 1** (anchor pair mixing one GT frame): trains
  the easier, rarer hand-off case; excluded from sampling (recorded —
  revisit only if m=1 becomes a target).
- **Longer chains (3+ hops)**: closer to full unrolling (BPTT-adjacent
  cost and instability); two hops is the minimal form that trains
  contraction, per the ADR-0061 state-chain precedent.
- **Jump straight to the joint arm**: rejected — separate arms first;
  the pair of knobs is cheap and the fleet decides (house rule).

## Consequences

- The repair question becomes a config-level fleet on the dt15 recipe
  (Δt≤15, the promising-m regime), pre-registered in
  `scratch/2026-09-10-anchor-contraction-fleet-prereg.md`: success =
  self-anchored ≤ 25.0 pooled at m ∈ {5, 15} (break-even with TC) /
  ≤ 17.4 at m=5 (decisive), with the ADR-0061-style information guard
  (oracle within ~10% of the no-repair control) and the displacement
  guard at any claimed m.
- **Surface changed** (on acceptance): `config.py` (two fields +
  validation: both require `flow_map = true`, noise ≥ 0; mandatory
  two-key migration across all transolver TOMLs, ADR-0057/0061/0062
  precedent), `datasets/particle.py` (chain sampling — additive
  `FlowMapChainDataset`; existing datasets untouched), `cli/train.py`
  (chain branch in `_train_transolver_fm`: two step-A queries, detached
  raw-unit anchor reconstruction via the target-normalizer inverse,
  kinematic clamps, step-B forward; anchor-pair noise injection beside
  the existing aux-noise call), tests (byte-identity off; chain
  end-to-end smoke; clamp and detach checks; noise determinism).
  **Not touched**: `models/transolver/simulator.py` predict paths
  (`forward_train_tc` already accepts arbitrary anchor tensors — the
  chain is a trainer-side composition), `eval/rollout.py`, metrics
  keys, other families, schemes.
- Cost: pushforward arms train ~2.5–3× slower per step; the fleet
  (~10 runs ≈ 50 A100-h) is priced in the prereg.

## Relationship to other ADRs

- **ADR-0062**: implements its recorded follow-up ("anchor kinematics
  stay noise-free in v1 — the follow-up knob if the anchor-feedback
  branch fires"; budget-fleet branch 2 fired). The flow-map surface,
  eval modes, and clamps are reused unchanged.
- **ADR-0061**: the two-mechanism / fleet-selected shape, the
  both-steps loss, the detached-chain seam, and the info-guard pattern
  are inherited; its `aux_input_noise_std` remains active and
  orthogonal (aux-block noise), its `aux_input_pushforward` remains
  rejected under `flow_map` (AR-loop semantics).
- **ADR-0049**: noise-scale decisiveness — two scales fleet-swept.
- **ADR-0051**: the pushforward lineage (bundle-seam → state-channel →
  anchor-interface).
