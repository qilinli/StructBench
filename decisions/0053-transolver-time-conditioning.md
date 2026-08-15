# 0053 — Time-conditioned prediction scheme for Transolver (native structural baseline)

**Status**: Accepted (maintainer settled A/B/C in-session, 2026-08-15)
**Type**: Durable
**Date**: 2026-08-15

## Context

We are establishing faithful, native-scheme baselines for the two operator/GNN
families across DP, Taylor, and notch: **MGN = autoregressive** (its native
mesh-dynamics scheme), **Transolver = time-conditioned** (its native scheme for
*structural deformation*). The scheme choice was verified against the official
`thuml/Transolver` code, not inferred:

- **`exp_ns.py` (Navier-Stokes, fluid):** autoregressive with a sliding
  velocity-history window — `im = model(x, fx=fx); fx = cat((fx[..., step:], im))`.
  Predictions fed back. This is the *fluid* convention (= GNS/MeshGraphNets).
- **`exp_plas.py` (Plasticity, structural deformation — a die deforming an
  elastoplastic block):** **time-conditioned querying** —
  `for t in range(T): im = model(x, fx, T=tim[:, t:t+1])`, `fx` **fixed** across
  the loop (no feedback), **no velocity history**, per-timestep relative-L2 loss.

All three of our benchmarks are **structural solid mechanics** (bar impact,
beam impact, plate deformation) → the faithful Transolver template is
**Plasticity's time-conditioning, not NS's AR+history**. This corrects two prior
missteps: (a) our `frames_per_call=0` "one-shot" predicts *all frames in one
decoder pass* (`out_size=k·(dim+1)`) — this matches **no** official Transolver
scheme; (b) our AR + velocity-history Taylor/notch Transolver runs applied the
*fluid* convention to a structural problem. In the official code the
velocity-history window is a Navier-Stokes (fluid) construct; the structural
benchmark uses geometry + time, history-free.

## Decision

Implement a **time-conditioned** prediction scheme for the Transolver family,
faithful to `Transolver_Structured_Mesh_2D.py` (`Time_Input=True`).

### Faithful mechanism (official → ours)
| Official (thuml Plasticity) | Ours |
|---|---|
| `fx = cat((x, fx), -1); fx = preprocess(fx)` (coords + static input → n_hidden) | reference coords + node-type one-hot + **scalar impact velocity** (B, ADR-0051 B) → preprocess; **no velocity window** |
| `Time_emb = time_fc(timestep_embedding(T, n_hidden))` (sinusoidal + MLP Linear→SiLU→Linear) | identical: sinusoidal `timestep_embedding` + `time_fc` MLP |
| `fx = fx + Time_emb` (broadcast to N points, added before blocks) | identical additive injection before the Physics-Attention blocks |
| `for t: im = model(x, fx, T=t)`, `fx` fixed | per-t query, static input, **no AR feedback** |
| per-timestep relative L2, accumulate over t | per-frame position (+aux) loss over the scored frames |

### Settled decisions
1. **Static, history-free input.** Model sees the initial/reference geometry +
   node types + scalar impact velocity only. `velocity_history=false` and
   `frames_per_call=1` are **required** with time-conditioning (mutually
   exclusive schemes).
2. **Time normalisation.** Query time `t ∈ [0,1]` = frame index over the scored
   horizon (`linspace(0,1,T_scored)`), per benchmark (Taylor/notch/DP horizons
   differ). Matches the official `linspace`.
3. **Target = absolute state at t.** The model maps `(input, t) → state(t)`
   (position + aux at frame t), not a per-step delta — there is no AR step, so
   the GNS adjusted-next / noise machinery is inert (as at `k>1`).
4. **Config knob** `time_conditioned: bool` (default false = unchanged). Guards:
   requires `velocity_history=false`, `frames_per_call=1`; composes with
   `impact_velocity_feature`.
5. **Eval.** Query every scored frame independently; report per-frame position
   RMSE (fills the `rollout_position_rmse` slot but is computed *without*
   accumulation — a genuine independent-query error). `one_step_*` is undefined
   for TC and reported as N/A.

### A/B/C resolved (maintainer, 2026-08-15)
- **A → feed scripted-BC-at-t + override (option ii/iii).** At each query `t`,
  the scripted/kinematic nodes' GT positions (impactor / supports / actuator)
  are provided as an input channel AND their outputs are overridden post-hoc.
  Verified rationale: `exp_plas.py` feeds a **static** die + scalar `t` only
  because Plasticity's die-motion **schedule is shared across all cases** (the
  model learns die-state-at-t from `t` alone). Our loading is
  **velocity-parameterised** (not shared), so "static + t" would force the model
  to infer impactor ballistics — a task Plasticity never faced. Feeding the
  prescribed scripted BC at `t` is the faithful analog of the die-state
  Plasticity's model effectively has; scripted nodes are known BCs, so it is
  legitimate, not leakage. Per-benchmark: Taylor's wall is static (i≈ii); notch
  pin+supports and DP actuator move, where (ii) matters.
- **B → reference (rest) coords** as the static input geometry.
- **C → our standard position(+aux) RMSE loss** (registry comparability); the
  deviation from thuml's relative-L2 is noted, not adopted.

## Scope & budget
- Benchmarks: **DP (primary, where TC is most clearly native), Taylor, notch.**
- **DP capped at 3 days wall-clock** (maintainer directive). Both DP baselines
  (MGN-AR checkpoint + Transolver-TC) **budget-matched to the slower method's
  3-day step count** — measured once TC throughput is known (MGN-AR ≈ 24.7k
  steps/hr ⇒ ~1.75M in 72 h; the running blessing's current checkpoint is the
  DP MGN-AR baseline, no new MGN run).
- Configs: `transolver-timecond-iv-s{1,2}` per benchmark (TC + impact velocity).

## Alternatives considered
- **`frames_per_call=0` predict-all (one-shot).** Rejected as the Transolver
  baseline: matches no official scheme; fixed-size all-frames head; the ADR-0051
  Taylor k=T run is a 33-sample confounded one-shot. Retained only as an
  ablation of the prediction-scheme axis, not a native baseline.
- **AR + velocity-history (current Taylor/notch Transolver).** Rejected as the
  *faithful* baseline: it is the fluid (NS) convention; Transolver's structural
  benchmark is history-free time-conditioning. Retained as an ADR-0044 record,
  not the native baseline.

## Consequences
- The **DP Transolver baseline is redone as TC** (the completed AR
  `deforming-transolver-v03` used the wrong scheme for a structural/quasi-static
  task). Taylor/notch gain TC baselines; their AR runs are re-labelled
  fluid-convention comparisons, not the native baseline.
- New model modules (`timestep_embedding`, `time_fc`) and a TC simulator/eval
  path; `time_conditioned=false` keeps the existing behaviour byte-identical.

## Verification plan
- CPU smoke: TC forward/backward finite; `time_conditioned=false` byte-identical
  to current; two different `t` inputs give different outputs (time actually
  conditions); grad flows to `time_fc`.
- Faithfulness check: reproduce the additive-time-embedding shape flow of the
  official model on a toy grid.

## Status / next
Proposed. On acceptance: implement on `feat/native-baselines`, smoke, add
configs, then run (DP 3-day-capped, budget-matched).
