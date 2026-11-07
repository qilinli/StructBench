# Plan 3a: the two-site state in the Transolver flow map

*Build plan for part 3a of `2026-11-07-two-site-state-design.md` (ADR-0075,
Proposed). Written before the work on `feat/two-site-state-3a`; the full
suite is the gate. Scope narrowed from the design: the anchored flow map
only (ADR-0062, the simulator-class scheme); the AR scheme and the
pushforward chains (ADR-0063 knob 1) are follow-ups, rejected at config load
with the two-site decoder.*

## Tasks

1. **Shared mechanics** — `models/common/two_site.py`: torch cell weights
   and element-to-node projection (the loader's area-weighted mean),
   centroid strain and rotation increments of a four-node quad from the
   corner displacements between two configurations (mid-configuration
   gradients, Hughes–Winget rotation; axisymmetric adds the hoop strain), the
   J2 radial return with per-element law rows (shear and bulk moduli, the
   `(peeq, sigma_y)` knots with `np.interp` semantics, a three-step Newton
   solve exact for linear hardening), and the corner-mean head. Tests: the
   projection equals the loader's; uniaxial stretch of one element gives
   `E ε` below yield, yields at `sigma_y0` and hardens with the slope; a
   rigid rotation leaves the stress invariant; on random inputs von Mises
   stays at or below `sigma_y(peeq)`, `peeq` never decreases, the deviator
   trace is zero; gradients flow to the corner displacements.
2. **Data contract** — `CaseTrajectory.state (T, E, S)` and
   `state_channels`; the mesh loader fills them from the element block
   without projection when asked (`state_fields=`); `BenchmarkSpec` gains
   `state_site`, `state_channels`, `element_formulation`,
   `constitutive_law`; `benchmarks/constitutive.py` holds `ConstitutiveLaw`
   with validation. Trajectories load the element block only for runs whose
   model uses it.
3. **Config** — `TransolverConfig.two_site_decoder: "off" | "return_map" |
   "corner_mean"` (default off, byte-identical). Load-time rules: requires
   `flow_map`, a benchmark with `state_site = "element"`, `train.aux_fields`
   equal to the benchmark's `state_channels` (the nodal view is the
   projection of exactly the state block), the constitutive law for
   `return_map`; incompatible with `flow_map_pushforward`,
   `flow_map_structured_heads` and the hinge.
4. **Datasets and collate** — flow-map pair samples carry `input_state`
   (anchor) and `next_state` (query) when the trajectory has a state block;
   the mesh collate adds `cells` (node-offset), `cell_weights`,
   `n_elements_per_example` and per-element law rows.
5. **Simulator** — a state normalizer registered only when the knob is on;
   `bind_case(..., gt_state, law)`; `set_anchor(..., state)`;
   `forward_train_two_site` returning the nodal prediction (its aux slice
   is the projected element state) and the element state prediction;
   `predict_two_site_at` for evaluation; `predict_state_at` returns the
   projected view so the existing rollout and clamps work unchanged.
6. **Trainer** — `_train_transolver_fm` adds the element-state loss
   (per-channel z-scored, weighted by `w_aux`) beside the nodal loss, and
   binds state and law for validation.
7. **Rollout and evaluation** — `flow_map_rollout` carries the element
   state across re-anchors (self and oracle modes); `RolloutResult` gains
   `predicted_state` and state metrics; `evaluate` records
   `rollout_rel_l2_state`, `rollout_state_rmse`, their per-channel dicts
   (the keys follow the existing `rollout_*` naming; the design's shorter
   names are not used), per interval under the flow-map prefixes, and saves
   `predicted_state` in the rollout artifacts.
8. **Smoke and docs** — a two-site flow-map train-and-evaluate smoke on a
   synthetic quad mesh with a state block; `docs/ARCHITECTURE.md` ML
   data-flow paragraph; the plans index.

## Gate

Full suite green; the off-path byte-identity of every existing flow-map
test; on the synthetic smoke the `return_map` arm's violation rates read zero
at tolerance 1e-5 and the `corner_mean` arm's do not by construction.
