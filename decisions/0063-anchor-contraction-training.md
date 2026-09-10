# 0063 — Anchor-interface contraction training (flow map)

**Status**: Accepted (maintainer, in-session 2026-09-10)
**Type**: Durable
**Date**: 2026-09-10

## Context

The ADR-0062 scheme fleet (12 runs) and the budget fleet (6 runs)
reduced the flow-map question to a single mechanism. The information is
there: at 66-epoch coverage the Δt≤15 operator reaches oracle-anchored
pooled aux 14.3 at m=5 (cash +0.70 vs the TC control's 25.0) — cashing
even half of it beats TC outright, displacement included. What blocks
it is the anchor hand-off: self-anchored the same model sits at 30.1
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
their joint form. The relevant house prior is acknowledged up front:
in the AR stability fleet the state-channel pushforward LOST to noise
("dominated loser", 2026-09-04 results). The flow-map interface differs
in the way that matters: there, Gaussian noise sufficed because it
rehearsed the drift; here, noise on the aux block is measured inert
(FM-N0 ≈ FM-FULL) — the aux damage is *structured* model error that
only training on the model's own predictions can rehearse.

## Decision

**Add two off-by-default, independently-ablatable contraction knobs to
the flow-map training path; select by fleet (pre-registered separately)
with the oracle-vs-self gap at m ∈ {5, 15} as the target.**

### Knob 1 — `TransolverConfig.flow_map_pushforward: bool = False` (primary)

Two-hop training chains on the ANCHOR interface — the flow-map-native
pushforward (ADR-0051/0061 lineage), training on the model's own joint
anchor-error distribution rather than a noise proxy:

1. Sample a chain `(t₀, t₁, t₂)` **uniformly over valid triples** (the
   `FlowMapPairDataset` enumeration convention; late-trajectory anchors
   carry fewer triples — end-skew recorded as the v1 choice): hop 1
   `t₁ − t₀ ∈ [2, Δt_max]` (≥ 2 so BOTH frames of the constructed
   anchor pair are predictions — the dominant rollout regime); hop 2
   `t₂ − t₁ ∈ [1, Δt_max]`. Hops are NOT pinned to a deployment
   interval — recorded rationale: it keeps training m-agnostic (the
   ADR-0062 anti-specialization stance) and the hop-1 spread doubles as
   an error-magnitude curriculum (larger hop 1 → dirtier step-B
   anchor). A hop-matched variant (`hop1 ∈ {m−1, m, m+1}`) is the
   pre-registered escalation, not a silent default.
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
   signal; step B trains contraction — the ADR-0061 both-steps
   convention, at 2:1 clean:dirty here because step A needs two queries
   to form the pair). Normalizers accumulate on the clean step-A
   queries only.

**Known limitation, recorded**: the chain rehearses FIRST-GENERATION
anchor error only (step A's anchor is clean GT), while deployment at
m=5 compounds 28 generations. This is the standard pushforward
compromise; the fleet prereg carries a generation-count diagnostic
(fractional gap closure at m=15 vs m=5, oracle-normalized), and a
3-hop chain is the pre-registered escalation **if closure decreases
with hand-off count** — not rejected outright.

~3× forward cost per optimizer step. Requires `flow_map = true` and a
capped `flow_map_max_dt ≥ 2` (hop 1 spans two frames; uncapped chains
would enumerate O(T³) triples — rejected at config load); composable
with knob 2 (below; the fleet tests each alone and the pair).

### Knob 2 — structured anchor-kinematic noise (comparator):
### `flow_map_anchor_noise_pos: float = 0.0` and
### `flow_map_anchor_noise_vel: float = 0.0`

Gaussian noise on the anchor KINEMATICS at train time, with the
component structure DICTATED BY MEASUREMENT rather than assumed. The
phase-0 error-structure measurement (existing self-anchored artifacts;
`scratch/stage3/anchor_subchannel.json`) shows the two anchor-pair
frames' errors are almost perfectly correlated (pair correlation
0.991): the per-hand-off position error is ~0.66 mm while the
FD-velocity error is only ~0.029 mm/frame (~27% of the 0.105 mm/frame
motion signal) — the correlated components cancel in the difference.
Naive independent per-frame noise therefore mis-rehearses the interface
by ~30× (position-matched independent noise corrupts the velocity
channel 30× more than reality). Hence two orthogonal components, both
absolute working-unit (mm) scales, per node:

- `..._pos` (common-mode): ONE draw added to BOTH pair frames —
  corrupts the anchor displacement, cancels exactly in the FD velocity.
- `..._vel` (differential): a draw added to the `t₀−1` frame only —
  corrupts the FD velocity by its std, leaves the anchor position
  untouched.

Kinematic ROWS stay clean; targets stay clean GT. The fleet doses both
components at the measured magnitudes and sweeps the overall scale with
the structure fixed (ADR-0049 scale-decisiveness; a third scale is the
pre-registered follow-up before any wrong-mechanism conclusion). Per
the attribution this knob alone is worth at most TC-parity on the
state fields plus the displacement fix — the cheap comparator and the
kinematic half of the joint arm, not the headline mechanism. Both
require `flow_map = true`; `0.0` (defaults) byte-identical.

### Composition (the joint arm)

With both knobs on, the kinematic noise applies **wherever a GT anchor
is consumed** — i.e. the chain's step-A anchor; the step-B self-anchor
stays raw (its corruption is the real thing the chain exists to
rehearse; stacking synthetic noise on it would blur attribution). If
the higher noise scale dominates the lower in the comparator pair, a
joint arm re-paired at the winning scale is the first follow-up before
any composition conclusion (the 2026-09-04 mispairing lesson).

### Explicitly not in this ADR

- Aux-channel-only mechanisms (a separate aux-side chain, hand-off
  projection): knob 1 already contracts the aux channel jointly; a
  dedicated aux device is priced only if the fleet shows kinematics
  fixed but aux stuck (and conversely, a kinematics escalation if the
  mirror signature fires — both routed by the prereg branches).
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
- **Longer chains (3+ hops)**: BPTT-adjacent cost; two hops is the
  minimal contraction form (ADR-0061 precedent). NOT rejected outright:
  the pre-registered escalation if the generation-count diagnostic
  fires (see Knob 1's known limitation).
- **Jump straight to the joint arm**: rejected — separate arms first;
  the pair of knobs is cheap and the fleet decides (house rule).

## Consequences

- The repair question becomes a config-level fleet on the dt15 recipe
  (Δt≤15, the promising-m regime), pre-registered in
  `scratch/2026-09-10-anchor-contraction-fleet-prereg.md`: success =
  self-anchored ≤ 25.0 pooled at either m ∈ {5, 15} (break-even with
  TC) / ≤ 17.4 at m=5 (decisive), with the displacement guard at any
  claimed m and a one-sided information guard (oracle must not degrade
  materially vs the control; improvement is reported, not tripped).
- **Surface changed** (on acceptance): `config.py` (three fields +
  validation: all require `flow_map = true`, noise ≥ 0; mandatory
  three-key migration across the 96 transolver TOMLs,
  ADR-0057/0061/0062 precedent), `models/transolver/simulator.py` (one additive public
  helper, `train_output_state`, inverting the target normalizer to the
  raw displacement + aux blocks — the ADR-0061 `train_output_aux`
  precedent; the chain needs the displacement slice and reaching into
  the private normalizer from the trainer is not acceptable),
  `datasets/particle.py` (chain sampling — additive
  `FlowMapChainDataset`; existing datasets untouched), `cli/train.py`
  (chain branch in `_train_transolver_fm`: two step-A queries, detached
  raw-unit anchor reconstruction via `train_output_state`, kinematic
  clamps, step-B forward; anchor-pair noise injection beside the
  existing aux-noise call), tests (byte-identity off; chain end-to-end
  smoke; clamp and detach checks; noise determinism). **Not touched**:
  simulator predict paths (`forward_train_tc` already accepts arbitrary
  anchor tensors — the chain is a trainer-side composition),
  `eval/rollout.py`, metrics keys, other families, schemes.
- Cost: pushforward arms train ~2.5–3× slower per step (~6 h per
  100k-step run at the measured ~2.1 h base); the fleet (~10 runs
  ≈ 35–40 A100-h) is priced in the prereg.

## Relationship to other ADRs

- **ADR-0062**: implements its recorded follow-up ("anchor kinematics
  stay noise-free in v1 — the follow-up knob if the anchor-feedback
  branch fires"; budget-fleet branch 2 fired). The flow-map surface,
  eval modes, and clamps are reused unchanged.
- **ADR-0061**: the two-mechanism / fleet-selected shape, the
  both-steps loss, the detached-chain seam, the `train_output_*` helper
  precedent, and the info-guard pattern are inherited; its
  `aux_input_noise_std` remains active and orthogonal (aux-block
  noise), its `aux_input_pushforward` remains rejected under
  `flow_map` (AR-loop semantics). Its fleet verdict (PF dominated by
  noise, AR interface) is the acknowledged prior this ADR bets against
  on stated grounds (structured vs rehearsable error).
- **ADR-0049**: noise-scale decisiveness — measurement-derived scales,
  swept.
- **ADR-0051**: the pushforward lineage (bundle-seam → state-channel →
  anchor-interface).

## Amendment (2026-09-10, maintainer-directed): generation curriculum

The repair fleet's branch-2 verdict (break-even at m ∈ {5, 15}; gap
closure ~0.30 FLAT in m) left one regime untreated: m = 1, where 144
hand-offs compound error generations the one-generation chain never
rehearses (residual gap 16.4 vs ~9.5 at m=5) and where the prize is
largest (oracle 13.4). On the maintainer's direction (GraphCast-style
horizon curriculum), the chain gains a GENERATION dimension:
`flow_map_pushforward_generations = G` deepens it to up to G successive
DETACHED re-anchor events (each generation's anchor rebuilt from the
previous generation's predicted pair with the house clamps; only
generation-1 queries are clean and warm the normalizers), annealed in
equal phases (`g(step) = min(G, 1 + floor(step·G/training_steps))`).
Deliberate deviations from GraphCast, recorded: detached between
generations (the ADR-0051/0061/0063 lineage — no BPTT through the
chain; BPTT is the later escalation if detached saturates), and a
generation curriculum rather than a step curriculum (the flow map's
feedback unit is the hand-off, not the frame). At a curriculum level
below G the final dirty query targets the next pair's first frame (a
valid ≤ max_dt offset), so every phase trains 2g+1 queries. `G = 1`
(default) is the plain two-hop chain, byte-identical. The G-deep chain
family is combinatorial, so the dataset enumerates anchors and draws
hops per access (torch RNG, deterministic under the run seed with the
in-process loader) — a recorded departure from the enumerated-index
convention. Fleet: `scratch/2026-09-10-anchor-contraction-fleet-
prereg.md` addendum (PFKN-700K budget arms + PFKN-CURR curriculum arms,
m=1 promoted to a primary readout for the curriculum).
