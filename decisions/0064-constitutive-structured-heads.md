# 0064 — Constitutively-structured admissible heads (physics baked into training)

**Status**: Proposed
**Type**: Durable
**Date**: 2026-09-12

## Context

The maintainer's stated goal (2026-09-12) makes physics consistency and
accuracy CO-EQUAL: a methodology for structural response simulation
with **verified and validated physics outputs**, at accuracy beyond the
TC baseline. The measured state of both objectives on the flow-map
model of record (PFKN-100k, m=5):

- **Consistency**: the direct-vm baseline convention emits 10.3%
  physically impossible stresses, undetectably and unfixably; the
  complete-state route is self-auditing and EVAL-TIME enforceable at
  measured-zero cost (yield projection: D2 6.4% → 0 at vm 0.282 →
  0.282; monotone clamp: D3 41.3% → 0 at peeq 0.088 → 0.091).
  Enforcement is currently post-hoc — the model itself is trained
  unconstrained and routinely proposes inadmissible states. The
  maintainer's directive: bake the physics into TRAINING, not only
  evaluation.
- **Accuracy**: the residual oracle-vs-self gap (~9.5 pooled at m=5) is
  now measured **~100% aux-channel** (gap-analysis probe,
  `scratch/stage3/gap_analysis.json`: GT-aux anchors reach 15.55 ≈ the
  15.53 oracle, s_xy 0.79 → 0.46, composed vm 0.285 → **0.183 ≈ the
  registered direct head's 0.175**; GT kinematics buys 0.03 — the
  ADR-0063 kinematic repair is complete, post-repair anchor position
  error 0.66 → 0.097 mm). Every EVAL-TIME aux treatment is measured
  dead or marginal: in-loop projection neutral (25.00 vs 25.04),
  EMA-smoothed aux anchors flat-to-worse (25.07 / 25.55), TC-sourced
  anchors −3.4% (two-model cost), self-hierarchical anchors worse.
  **The aux hand-off must be repaired in training.**

Two more measured facts shape HOW:

- The house prior on soft losses is strongly negative for aux accuracy
  (F-011: no training-signal manipulation improved an aux field;
  WAUX3's loss re-weighting was rejected with the primary field
  paying), and in-manifold drift — not unphysicality — is the
  compounding driver (stability probes; projection closes ≤ 10%).
  Soft consistency penalties are therefore the WEAK bet and run only
  as the comparator.
- The per-channel hand-off error is grossly mis-matched to the uniform
  ADR-0061 noise dose (`scratch/stage3/ema_aux_probe.json` /
  probe printout): relative anchor errors (s_xx, s_yy, s_xy, peeq, E,
  rho) = (0.40, 0.38, **0.78**, 0.08, 0.10, 0.002) vs the flat 0.15 —
  s_xy under-rehearsed ~5×, peeq over-rehearsed ~2×. Re-dosing is a
  pure config arm (the knob is already per-channel).

## Decision

**Restructure the flow-map state head so admissibility holds BY
CONSTRUCTION, at training and deployment alike — the return-mapping
structure of computational plasticity as the decoder's hypothesis
class — with a soft-hinge comparator; select by fleet (pre-registered
separately) against both goals at once.**

### Knob 1 — `TransolverConfig.flow_map_structured_heads: bool = False` (primary)

The decoder's raw state slice is reinterpreted; the emitted state is:

- **peeq** `= peeq_anchor + softplus(raw_Δ)` — non-decreasing from the
  anchor by construction (D3 ≡ 0 across anchors and within segments;
  the flow-map anchor supplies the reference, which is why this head
  is flow-map-native).
- **deviator** `= r · σ_y(peeq) · u / ‖u‖_vm` where `r = sigmoid(raw_r)`
  ∈ [0, 1], `u` the raw 3-vector direction, and `‖u‖_vm` its von Mises
  norm — so the composed vm equals `r · σ_y(peeq)` EXACTLY and can
  never exceed the yield surface (D2 ≡ 0 by construction). The
  magnitude/direction decomposition is the structured-deviator
  parameterisation previously motivated for the s_xy channel
  (vm-route fleet, branch 4).
- **internal energy, density**: unconstrained (v1).

The hardening curve `σ_y(peeq)` enters as a fixed per-benchmark table
(the Taylor curve already used by every admissibility analysis),
supplied through the benchmark spec at build time. Loss unchanged in
form: structured raw outputs are passed through the existing target
normalizer so `w_pos`/`w_aux` semantics and all metrics are untouched.
Requires `flow_map = true` and the 6-channel state block layout
(deviatoric_stress_2d + effective_plastic_strain + internal_energy +
density — validated at load). `False` (default) byte-identical.

**What this buys per goal**: (V&V) the model *cannot emit* an
inadmissible state — projection becomes verification, not repair; the
guarantee holds inside the hand-off loop, in training, and at
deployment. (Accuracy) the aux hand-off error the residual gap is made
of gets a better-conditioned parameterisation exactly where it is
worst — s_xy through the bounded direction field, peeq through
increments — a hypothesis, priced by the fleet, not asserted.

### Knob 2 — `TransolverConfig.flow_map_consistency_hinge: float = 0.0` (comparator)

Soft admissibility penalties on the unstructured head, added to the
standard loss with weight λ: `relu(vm_comp − σ_y(peeq_pred))/σ_y0`
(yield hinge) + `relu(peeq_anchor − peeq_pred)/peeq_scale`
(irreversibility hinge — again anchored, flow-map-native). Two λ
scales fleet-swept (ADR-0049). Expected weaker per F-011; it exists so
"hard vs soft constraints" is measured, not assumed. Requires
`flow_map = true`; `0.0` byte-identical. Mutually exclusive with
knob 1 (a structured head has nothing to penalize).

### Explicitly not in this ADR (fleet-config arms, no new surface)

- **Aux-noise re-dose** at the measured per-channel hand-off structure
  (the existing per-channel `aux_input_noise_std`) with the kinematic
  dose reduced to its post-repair magnitudes.
- **Capacity arm** (hidden 256): the gap now tracks the oracle, which
  may be trunk-limited; existing knob.
- Eval-side projection/clamp productization for UNSTRUCTURED runs — a
  small separate follow-up (measured free); structured heads make it
  moot for their runs.

## Alternatives considered

- **Keep enforcing at eval only**: measured — projection/clamp are free
  correctors of outputs but neutral to accuracy, and they leave the
  training-time model proposing inadmissible states the chain then
  rehearses; contradicts the maintainer's directive.
- **Soft penalties as the primary**: against F-011 and the WAUX3
  rejection; demoted to comparator.
- **Structured heads for AR/TC too**: TC has no anchor peeq for the
  increment head (frame-0 anchoring would pin peeq to a full-history
  increment — a different design); AR is retired for this thread.
  Flow-map-scoped v1.
- **Constrain E/ρ too** (positivity etc.): tiny measured error
  (rel 0.10/0.002); complexity without a driver. Deferred.

## Consequences

- **Surface changed** (on acceptance): `config.py` (two knobs +
  validation incl. mutual exclusion and the state-layout check),
  benchmark spec (a `hardening_curve` hook, Taylor-only v1),
  `models/transolver/simulator.py` (structured decode in the TC/FM
  target path + the raw→normalized loss bridge; `train_output_state`
  and `set_anchor` unchanged — they consume raw states), `cli/train.py`
  (hinge terms in the fm loss), mandatory two-key migration across the
  ~110 transolver TOMLs, tests (byte-identity off; D2/D3 ≡ 0 on
  structured outputs as an IMPLEMENTATION check; hinge gradients;
  head-shape round-trip; chain smoke with structured heads).
- Structured runs' D2-own/D3 metrics double as implementation
  tripwires (any nonzero = bug, not physics).
- The fleet (pre-registered separately,
  `scratch/2026-09-12-structured-heads-fleet-prereg.md`) prices both
  pillars in one round against the measured door: GT-quality aux
  anchors reach 15.5 — the decisive bar (17.4) is INSIDE the door the
  residual analysis opened.

## Relationship to other ADRs

- **ADR-0063**: the chain + kinematic noise stay the base recipe (the
  kinematic channel is measured fully repaired); this ADR targets the
  remaining aux channel and converts eval-time physics enforcement
  into architecture.
- **ADR-0062**: scheme unchanged; the anchor interface is what makes
  both heads well-posed (anchored increments; state-complete anchors).
- **ADR-0059**: channel conventions unchanged; the structured head
  emits the same 6 channels in the same order and units.
- **F-011 / WAUX3**: the reason soft constraints are the comparator,
  not the bet.
