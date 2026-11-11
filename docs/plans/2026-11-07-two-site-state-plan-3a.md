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

## Note 2026-11-08: the return-map arm does not train; the cause is stiffness, not scale

The first by-path run of the `return_map` decoder trained at a loss of
10³–10⁴ from its first validation and a rollout position RMSE at the body's
own scale, while the `corner_mean` control differing in the decoder alone
trained normally. The suspected cause, the element state entering the loss
or the anchor hand-off in physical units beside normalized channels, is
ruled out: both decoders z-score the decoded physical state with the one
state normalizer in `forward_train_two_site`, and both carry the physical
state in double across `set_anchor`; `tests/models/transolver/
test_transolver_two_site.py` now pins both. With ground-truth kinematics
the return map reproduces the stored state (z-scored RMSE 0.02–0.3 per
channel at one frame).

The mechanism is the decoder's conditioning. The return map is the
material law applied to the strain increment of the *predicted*
displacements, so a displacement error `δ` on an element of edge `h` is a
strain error `δ/h`, a trial stress error `E δ/h`, and after z-scoring an
error of `E δ / (h · std σ)`. For a metal (`E ≈ 1.2 × 10⁵ MPa`, `std σ ≈
10² MPa`, fine elements) that factor is ~10³ per unit of relative
displacement error: an error of 1 % of an element edge already scores
~10³ in the state loss, and the state loss's gradient into the
displacement exceeds the position loss's by ~10⁷. The deviator is bounded
by the yield surface wherever the return is active, but the mean stress
`K tr Δε` is unbounded, so the three normal components carry the error.
Under gradient clipping the network is trained to match the volumetric
strain of its displacement field and not the positions, and the amplified
mean stress fed back through the anchor's projected aux takes the rollout
out of distribution. This is not a bug in the build; it is a property of
deriving stress from a surrogate's kinematics at `E / std σ ≈ 10³`, and it
needs a decision on the admissible arm's form before the arm is run again
(the candidates: the network predicts a trial state that the radial return
projects, keeping yield admissibility and peeq monotonicity by
construction; or the network predicts the mean stress and the return map
supplies the deviator and peeq). Recorded here as the finding of the
build; the decision is the maintainer's, on ADR-0075 clause 3.

Guard added with the finding: a benchmark loaded by path whose first
validation shows a position RMSE above the domain scale (the largest
bounding-box diagonal of the validation cases) or a train loss above 100
exits with one line (`TrainingDiverged`, CLI status 3). It exits rather than
warns because an arm under a scheduler runs unattended for hours after its
first validation; it is one-shot and by-path only, so shipped-benchmark runs
are unchanged.

## Note 2026-11-08 (2): the admissible arm in two forms

The maintainer's decision on the finding above (ADR-0075, note 2026-11-08):
the direct return map is not repaired; two further `two_site_decoder` forms
beside `corner_mean` are built, both reading no predicted kinematics, and
the results choose the final form.

- **`admissible_coords`** — `models.common.two_site.admissible_decode`. The
  network's five state channels are read in its OWN output space (the target
  normalizer's inverse plays no part), taken to the element as the corner
  mean, and decoded: `peeq = peeq_anchor + softplus(d)`; the deviator
  `σ_y(peeq) · tanh(|v|) / |v| · v` with `v` the raw 3-vector `(a, b, c)` in
  the trace-free basis `(a, b, −(a + b), c)` (the von Mises norm is
  1-homogeneous, so the composed von Mises is `σ_y(peeq) · tanh(|v|) <
  σ_y(peeq)`; smooth at `v → 0`); the mean stress `loc + scale · m`, with the
  location and scale taken from the state normalizer's three normal channels
  (`OnlineNormalizer.moments`, new) — a scale, not a statistic. `peeq_anchor`
  is the CARRIED anchor state's, never the noised anchor aux the network
  reads.
- **`projected_trial`** — `models.common.two_site.project_trial`. The
  network's five channels are read as raw-unit state (as `corner_mean`
  does), taken to the element as the corner mean, and projected: the plastic
  strain clamped to the anchor's (`max(peeq_trial, peeq_anchor)`), then the
  existing `radial_return` with a zero kinematic increment gives the plastic
  multiplier and the updated `peeq`; the trial deviator is scaled onto the
  surface at that `peeq` (`min(1, σ_y(peeq) / vm_trial)`), the mean stress
  passes through. The deviator is reassembled with the third diagonal
  component formed from the first two, so its trace is zero exactly.

The non-expansiveness of the projection is a theorem in the generalised-
stress metric of linear hardening, `|dev|²_F / (2G) + H peeq²`, where the
radial return is a closest-point projection onto a convex set; the test pins
the Jacobian's operator norm at or below one in that metric on random inputs
covering both branches (identity on elastic rows; one zero singular value,
the surface normal, on plastic rows). In raw units the norm is ~`H` MPa per
unit plastic strain through the hardening coupling, which is the law and not
a defect, and the test says so.

Wiring: `LAW_DECODERS = (return_map, admissible_coords, projected_trial)` in
both vocabularies (`benchmarks.constitutive`, `models.common.two_site`) gates
the law requirement and the five-channel layout at config load, at `bind_case`
and in the trainer; `_two_site_head` on the simulator picks the slice each
decoder reads; nothing else in the trainer, the rollout or the metrics
changes. Tests: the invariants for ANY head output (pure functions and the
simulator with a wild network), the Jacobian bound, the noised-aux test
(the plastic-strain floor is the carried state's), and the normalizer pins
and the synthetic smoke extended to all four decoders. The first GPU smoke
of each form on a by-path metal benchmark (the A3 pattern: 600 steps, a
validation at 300) trained at a loss below one from its first validation,
with a position RMSE at a few percent of the domain scale and falling by the
second — the divergence guard stayed silent where it had killed the direct
return map at the same step.

*2026-11-11 — the pushforward follow-up is built for the three head decoders
(ADR-0075 note of this date); the direct return map and the generation
curriculum remain rejected at config load with a two-site decoder.*
