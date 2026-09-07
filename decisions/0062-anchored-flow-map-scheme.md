# 0062 — Anchored flow map: state-anchored time-conditioned prediction (Transolver)

**Status**: Proposed
**Type**: Durable
**Date**: 2026-09-08

## Context

The complete-state thread has measured both ends of a design axis and
found the prize sits strictly between them.

A state-anchored **jump operator** maps `(state at t₀, Δt) → state at
t₀ + Δt`. Re-anchoring every `m` frames spans a family in **feedback
frequency**: `m = 1` is the AR-like limit (an anchor every frame), `m ≥
horizon` the TC-like limit (one anchor, no feedback). The registered AR
and TC schemes are *not* literal endpoints of this family — they differ
from it in target parameterisation (increment vs absolute field), history
window (5 velocities vs one anchor), noise recipe, and training
distribution — but they are the measured proxies for its two ends, and
the measurements point inward:

- **The anchor state is worth 2.6× and the TC formulation can absorb
  it.** The TC oracle probe (`scratch/2026-09-05-tc-oracle-probe-results
  .md`, branch `probe/tc-state-input`) conditioned TC on the GT state at
  `t−1`, teacher-forced: pooled aux RMSE 25.0 → 9.8 (interp) / 30.6 →
  11.9 (extrap), matching the AR oracle band (9.2–12.1 across the
  stability arms) — the information effect is scheme-independent.
  Displacement *improved* (0.0125 → 0.0080 interp), below the registered
  TC row (0.009383/0.01637). And `s_xy` — at 0.94–1.19 in every
  *self-fed* AR arm — measured 0.24 at the probe ceiling. (Oracle-mode
  per-channel metrics were never recorded — ADR-0060 logs four pooled
  oracle metrics only — but the recorded pooled oracle RMSE caps oracle
  `s_xy` below ~0.6, so state helps shear under AR too; the probe is the
  only *direct* measurement of shear learnability with clean state, and
  it sits under the TC formulation.)
- **Per-frame feedback destroys the prize.** The raw (noise-free)
  state-feedback arms land at 35–51 pooled interp self-fed (up to 85
  extrap) — worse than the output-only arm (27.7–28.5 / 28.9–29.9) —
  against the 9.2–12.1 oracle ceiling
  (`runs/taylor-transolver-ar-statefb-s{1,2,3}` vs `-state-s{1,2,3}`).
  The adopted ADR-0061 noise (0.15, arm `nhigh`) flattens the drift
  (self-fed 28.9–31.0 / 32.2–34.1) but does not cash the ceiling.
- The supporting story transfers: within-model stress composition is free
  (vm-route fleet), the ADR-0061 stability toolkit and ADR-0060's
  oracle-mode evaluation design carry over unchanged.

The untested interior — re-anchor every `m` frames, so the 145-frame
scored rollout has `⌈145/m⌉` anchor segments and therefore
`⌈145/m⌉ − 1` self-produced feedback events (vs 144 for per-frame AR;
at `m ≥ 145` there are zero, the single anchor being the GT seed) — is
the **anchored flow map**. ADR-0061 explicitly set "k-bundle or flow-map
feedback-frequency changes" aside as scope-heavy; the probe's decisive
ceiling (branch 1 of its prereg) is the new evidence that reopens it.

## Decision

**Add an off-by-default anchored-flow-map mode to the time-conditioned
Transolver: the model conditions on a self-produced anchor state (complete
state + kinematics at frame `t₀`) and the offset `Δt`, is trained on
uniformly sampled `(t₀, Δt)` pairs, and is evaluated by re-anchoring every
`m` frames in self-anchored and oracle-anchored modes across a swept `m`.**

### The anchor contract

The anchor at frame `t₀` is the model's only inter-segment interface. It
carries, as node features appended to the TC input:

- **Anchor displacement** `x[t₀] − reference_coords` — `(P, dim)`.
- **Anchor velocity** `x[t₀] − x[t₀−1]` — `(P, dim)`, the finite
  difference of the two frames at the anchor (the AR families' `time_diff`
  convention, un-divided by dt). FD rather than the archives'
  `response/node/velocity` because it is the only form the model can
  *self-produce* at a re-anchor point — predicted positions exist,
  predicted solver-velocity does not — keeping train and rollout inputs
  identical in kind. (The stored solver velocity is verified present in
  the Taylor archives; a solver-velocity variant would need velocity as a
  predicted output channel and is deferred.)
- **Anchor state** `aux[t₀]` — `(P, C)`, the run's ADR-0059 channel block
  (the 6-channel deviator/peeq/energy/density selection for the fleet).
- **Anchor time** `t₀ / (time_ref − 1)` — `(P, 1)` broadcast scalar,
  present iff `flow_map_anchor_time = true` (below).

Query-side conditioning: the additive time embedding
(`timestep_embedding` + `time_fc`, ADR-0054) receives
**`Δt_norm = (t − t₀) / (time_ref − 1)`** instead of absolute `t_norm`.
The remaining TC features are unchanged: node-type one-hot, reference
coords, scripted-BC-at-query (`kinematic_bc`), optional impact-velocity
scalar. Targets are unchanged: absolute `[displacement, aux]` at the
query frame, through the existing target normalizer. All anchor features
pass through the existing online node normalizer (its width grows;
`flow_map = false` remains byte-identical).

**The Markov ablation is a feature-drop, not a mechanism**: setting
`flow_map_anchor_time = false` and `impact_velocity_feature = false`
conditions the model on `(anchor state, Δt, static geometry)` alone. If
the complete state is closed (claim C3), nothing else is needed; the gap
between this arm and the full arm *measures* the value of the two
conditioning scalars given the anchor state. (It does not, by itself,
speak to the AR history window — neither arm carries one; the anchor has
a single FD velocity where AR has five.)

### Training

One sample = one `(t₀, t)` pair, drawn uniformly over all valid pairs:
`t₀ ∈ [input_frames − 1, T − 2]`, `t ∈ [t₀ + 1, T − 1]`, further capped
at `t ≤ t₀ + Δt_max` when `flow_map_max_dt > 0` (0 = no cap).
Uniform-over-pairs skews the `Δt` marginal toward short offsets
(linearly, as availability does); recorded as the v1 choice —
reweighting is a later knob, not a silent default. Anchors start at the
last seed frame (`input_frames − 1`, ADR-0035: the seed is the rollout
init), matching the deployment anchor `t₀ = 5`; the FD velocity uses
`t₀ − 1 ≥ 4`, inside the seed.

Anchor-state noise: `aux_input_noise_std` applies to the anchor `aux`
block exactly as in ADR-0061 (batch-std-relative, per-channel, kinematic
*rows* clean, clean targets). The AR lesson says noise-free self-feeding
collapses; the fleet sweeps 0 vs the adopted 0.15 rather than assuming.
Anchor *kinematics* stay noise-free in v1 — recorded as the follow-up
knob if the fleet's anchor-feedback branch fires (the sharpest risk is
the FD anchor velocity: the difference of two independently queried
predictions one 2 µs frame apart can carry a much larger relative error
than either position, and it feeds the stress/peeq channels — the
displacement guard alone would not surface this).

**In-training validation**: the val pass runs the self-anchored
re-anchoring rollout (the honest mode, with the house clamps) at the
canonical interval, and model-best selection stays on val position RMSE
— the house convention, now measured in the deployable mode.

### Evaluation — re-anchoring rollout, two modes per interval

For each `m` in `flow_map_eval_intervals`: seed the anchor from GT at the
last seed frame, query `Δt = 1 … m` within each segment, then re-anchor at
the segment end and continue (the last segment truncates at the horizon).
Two modes:

1. **Self-anchored** (deployable): the new anchor is the model's own
   predicted positions at the two frames at the boundary and its predicted
   aux there — with the ADR-0060 house clamps (kinematic rows of the fed
   aux are GT; scripted positions are GT via the existing override).
2. **Oracle-anchored**: anchors read from GT (positions and aux). Isolates
   within-segment accuracy — the `Δt`-generalization curve — with anchor
   drift severed. At `m ≥ horizon` the two modes coincide (no feedback
   events), so that sweep point reads purely as the `Δt`-generalization
   endpoint.

The `(oracle − self)` gap at each `m` is the anchor-feedback damage; its
shape in `m` is the flow-map bet, measured. Oracle-anchored `m = 1` is
the closest mainline analogue of the TC oracle probe — the same question
(is anchor-state information cashable under the TC formulation) but a
*different measurement*: the flow-map model adds GT anchor kinematics and
anchor time, conditions on `Δt` rather than absolute `t`, and trains with
only ~1.4% of its samples at `Δt = 1`. It is a consistency readout
against the probe's 9.8/11.9, not a reproduction; plumbing per se is
certified by implementation tests (below), not by that number.

Metrics are recorded per interval under suffixed keys (`rollout_m{m}_*`,
`rollout_oracle_m{m}_*`); the **canonical interval**
(`flow_map_canonical_interval`) additionally writes the standard
`rollout_*` keys, artifacts, and QoIs, so cross-run tooling reads
flow-map runs unchanged. `one_step_*` stays undefined (TC convention).

### Config surface

On `TransolverConfig` (all inert at defaults; byte-identity required):

- `flow_map: bool = False` — requires `time_conditioned = true` and
  `aux_input = true`.
- `flow_map_anchor_time: bool = True` — anchor-time scalar feature.
- `flow_map_max_dt: int = 0` — training `Δt` cap (0 = no cap).
- `flow_map_eval_intervals: tuple[int, ...] = ()` — required non-empty
  when `flow_map = true`; positive, strictly increasing.
- `flow_map_canonical_interval: int = 0` — the interval that writes the
  standard `rollout_*` keys/artifacts/QoIs; 0 = the first (smallest)
  listed interval, otherwise must be a member of
  `flow_map_eval_intervals`.

Validation deltas: the ADR-0060 `aux_input` + `time_conditioned`
rejection is **narrowed** — the combination is accepted iff
`flow_map = true` (plain TC never consumes evolving state; the probe
branch stays an unmerged probe). The narrowing applies to *both* mainline
guards: the config-load rejection and the simulator constructor's
defense-in-depth `ValueError` (which is kept, not deleted as the probe
branch did). `aux_input_pushforward` is rejected with `flow_map = true`
(the AR two-step chain has no meaning here — the flow map's own exposure
control is the anchor-noise knob + re-anchoring). Non-default values of
`flow_map_max_dt`, `flow_map_eval_intervals`, and
`flow_map_canonical_interval` are rejected unless `flow_map = true`;
`flow_map_anchor_time` (default true) is inert when `flow_map = false` —
any value accepted, so the mandatory default-valued migration passes.
The TC prerequisites (`history_frames = 0`, `frames_per_call = 1`) are
inherited unchanged.

### Scope

Transolver on Taylor 2D (the thread's measurement vehicle). Other
families, other benchmarks, registry/leaderboard status, projection
productization, and any restart-task benchmark design are out of scope.
The fleet itself is launched under a separate binding pre-registration
(`scratch/2026-09-08-flowmap-fleet-prereg.md`) — flag-first, as usual.

## Alternatives considered

- **k-bundle state feedback** (relax ADR-0060's `frames_per_call = 1`):
  feeds state through the AR increment formulation — where self-fed
  `s_xy` never left ~1.0, while the absolute-field TC formulation
  measured 0.24 at the ceiling. Bundling keeps the increment
  parameterisation and adds a fixed-size multi-frame head (ADR-0051's
  k=16 seam artifacts). Rejected.
- **Per-m specialized training as the primary design** (train with
  `Δt ≤ m`, one model per m): a training per sweep point, and it entangles
  the training distribution with the re-anchoring interval. One broad-`Δt`
  model evaluated at every `m` isolates the feedback-frequency effect;
  matched-`Δt` arms remain in the fleet as the specialization check.
- **Solver velocity at the anchor** (`response/node/velocity`, present in
  the archives): breaks self-anchoring (the model cannot produce it) or
  forces velocity into the predicted state with a
  `v_pred`-vs-FD(`x_pred`) consistency problem. FD anchors now; the
  variant is a measurable follow-up.
- **Noise on by default at the adopted 0.15**: the stability-fleet verdict
  is an AR result; under TC the noise-free probe hit the ceiling. The
  fleet sweeps it (0 vs 0.15) instead of baking it in.
- **A new scheme enum replacing the `time_conditioned` bool**: a config
  migration across ~80 TOMLs (78 transolver configs at time of writing)
  for zero behavioural gain; the flow map *is* time-conditioning with an
  anchor. Composition of existing knobs, plus `flow_map`, keeps every old
  config valid.

## Consequences

- The prediction-scheme axis (ADR-0050/0051/0054) gains its interior:
  the oracle-vs-self curves in `m` measure the feedback-frequency
  trade-off directly — with the caveat that the registered AR and TC
  baselines remain formulation-distinct proxies for the ends, not points
  on the same curve.
- A flow-map model is a genuine **simulator** (it consumes and produces
  the complete state), unlocking the state-requiring task family
  (restart/re-anchor tasks) that TC cannot serve — while inheriting TC's
  within-segment parallelism (`⌈145/m⌉ − 1` sequential re-anchor events
  instead of 144).
- **Surface changed** (on acceptance): `config.py` (five fields +
  validation narrowing), `datasets/particle.py` (pair-sampling dataset —
  additive; `WindowDataset` untouched), `models/transolver/simulator.py`
  (anchor features + `Δt` conditioning + anchor cache/re-anchor +
  `bind_case` reuse of `gt_aux`; the constructor's aux_input+TC guard
  narrowed to `flow_map = false`, which requires threading `flow_map`
  into the constructor), `eval/rollout.py` (re-anchoring rollout with the
  two modes), `cli/train.py` (flow-map training loop; self-anchored val
  at the canonical interval; evaluate's per-interval recording),
  `decisions/0060-aux-input-channels.md` (a dated narrowing note
  appended, and its index row marked "narrowed by 0062"), every
  transolver config TOML (strict loader: five inert new keys across the
  78 transolver configs — mandatory migration, ADR-0057/0061 precedent),
  tests (byte-identity off; validation matrix incl. canonical-interval
  membership; anchor-feature construction — at `m = 1` oracle mode the
  features must use frames ≤ `t₀` only, the leakage tripwire; pair-index
  non-emptiness at `flow_map_max_dt = 0` and cap equality at 15;
  re-anchoring vs oracle divergence on synthetic data; Markov feature
  drop). **Not touched**: AR paths, other families, existing metrics
  keys, benchmarks/cards/registries.
- Costs: the pair-sampling index is O(T²/2) per trajectory (~10k pairs ×
  21 cases — an index, not data); evaluation multiplies by
  `2 × |intervals|` rollout passes (each ≈ one TC pass; ~2 A100-min per
  run per interval pair).
- The probe branch `probe/tc-state-input` is superseded as a measurement
  surface once the fleet's oracle-anchored `m = 1` readout is in and
  understood — not by a numeric reproduction (none is expected; see
  Evaluation). Retiring it freezes `runs/probe-tc-statein-s{1,2}` at
  their saved metrics and rollout artifacts: artifact-level rescoring
  stays possible, checkpoint re-evaluation does not (mainline rejects
  their `aux_input`+TC-without-`flow_map` config, and flow-map code
  cannot load probe checkpoints — different input surface).

## Relationship to other ADRs

- **ADR-0054**: the flow map is the TC mechanism with the anchor made
  explicit and movable; the TC scheme itself is unchanged and remains the
  registered baseline scheme.
- **ADR-0060**: the aux-input surface, oracle-mode evaluation design, and
  kinematic-row clamps are reused verbatim; its aux_input+TC rejection is
  narrowed to "TC without `flow_map`" via a dated note on 0060 (house
  amendment mechanism — an adjustment, not a reversal).
- **ADR-0061**: `aux_input_noise_std` transfers to the anchor state
  (same semantics); `aux_input_pushforward` does not (rejected with
  `flow_map`).
- **ADR-0059**: the anchor state block is the run's `train.aux_fields`
  selection; per-channel conventions unchanged.
- **ADR-0035/0053**: the seed window stays the rollout init and provides
  the first anchor; `history_frames = 0` inherited from TC.
- **ADR-0050/0051**: completes the prediction-scheme axis those ADRs
  opened; the k-frames bundling axis remains separate and mutually
  exclusive.
