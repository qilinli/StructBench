# 0064 — Constitutively-structured admissible heads (physics baked into training)

**Status**: Accepted (maintainer, in-session 2026-09-12)
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
  ≈zero pooled cost (yield projection: D2 6.4% → 0 at vm 0.282 → 0.282;
  monotone clamp: D3 42.8% → 0 — seed mean, `scratch/stage3/
  consistency_comparison.json` — at peeq 0.088 → 0.091, i.e. +3.4% on
  that one channel; `scratch/stage3/handoff_dose_derivation.json`).
  Enforcement is currently post-hoc — the model itself is trained
  unconstrained and routinely proposes inadmissible states. The
  maintainer's directive: bake the physics into TRAINING, not only
  evaluation.
- **Accuracy**: the residual oracle-vs-self gap (~9.5 pooled at m=5)
  is measured **~100% aux-channel** (gap-analysis probe on pfkn-s1
  — SINGLE-SEED eval probe, noted per F-002;
  `scratch/stage3/gap_analysis.json`): GT-aux anchors reach 15.55 ≈
  the 15.53 oracle; s_xy 0.79 → 0.46; composed vm 0.285 → 0.183
  (within 4.6% of the registered direct head's 0.175). GT kinematics
  buys 0.03 — the ADR-0063 kinematic repair is complete (post-repair
  anchor position error 0.66 → 0.097 mm). Every EVAL-TIME aux
  treatment is measured dead or marginal: in-loop projection neutral
  (25.00 vs 25.04), EMA-smoothed aux anchors flat-to-worse
  (25.07 / 25.55), TC-sourced anchors −3.4% (two-model cost),
  self-hierarchical anchors worse. **The aux hand-off must be
  repaired in training.**

Two more measured facts shape HOW:

- The house prior on soft losses is strongly negative for aux accuracy
  (F-011; WAUX3 rejected with the primary field paying), and
  in-manifold drift — not unphysicality — is the compounding driver
  (stability probes; in-loop projection neutral). Soft consistency
  penalties are the WEAK bet and run only as the comparator.
- The per-channel hand-off error, measured in the noise knob's own
  batch-std-relative units (`handoff_dose_derivation.json`,
  `dose_vs_std`): (s_xx, s_yy, s_xy, peeq, E, rho) =
  (0.58, 0.55, 0.78, 0.10, 0.12, 0.01) vs the flat 0.15 — the shear
  channel under-rehearsed ~5×, the normal deviators ~4×. Re-dosing is
  a pure config arm (the knob is already per-channel).

## Decision

**Restructure the flow-map state head so yield admissibility and
anchor-relative irreversibility hold BY CONSTRUCTION, at training and
deployment alike — the return-mapping structure of computational
plasticity as the decoder's hypothesis class — with a soft-hinge
comparator; select by fleet (pre-registered separately) against both
goals at once.**

### Knob 1 — `TransolverConfig.flow_map_structured_heads: bool = False` (primary)

The decoder's raw state slice keeps its width (6) but is reinterpreted;
the emitted state is:

- **peeq** `= peeq_anchor + softplus(raw_Δ)` — never below the anchor,
  by construction. `peeq_anchor` is the FED anchor value exactly as the
  interface supplies it (ADR-0061-noised at chain step A, the detached
  prediction at step B, the fed-back prediction at deployment) — the
  train/deploy-consistent choice. Two recorded costs of the hard
  floor: (a) when noise or an over-shoot lifts the base above the GT
  target, the target increment is unrepresentable — the
  **unreachable-target fraction at the fleet doses is a binding
  readout**, and the across-hand-off ratchet is a known cost (the
  eval clamp's measured +3.4% on peeq is its existing estimate);
  (b) softplus is strictly positive, so exactly-elastic phases can
  only be approximated — a peeq-creep bias that compounds with
  hand-off count (an **elastic-subset increment readout** binds in the
  prereg; ReLU would give exact zeros at the cost of a dead-gradient
  zone — recorded alternative, not taken in v1).
- **deviator** `= σ_y(peeq) · v · tanh(‖v‖_vm) / ‖v‖_vm` with `v` the
  raw 3-vector and `‖·‖_vm` the plane-strain von Mises norm — smooth
  at `v → 0` (tanh(x)/x → 1; no ε-guard, no separate magnitude
  channel), and the composed vm equals `σ_y(peeq)·tanh(‖v‖_vm)`
  **≤ σ_y(peeq) by construction** (D2 ≡ 0). The direction/magnitude
  structure is the parameterisation previously motivated for s_xy
  (vm-route fleet, branch 4). Recorded trade: at-yield states need
  `tanh → 1` (saturation with decaying gradients over the dominant
  plastic regime) — the prereg's one-sided info guard is the check.
  *Sharpened by review: in float32 the decay is an EXACT dead zone
  (`1 − tanh²` underflows to 0 for `‖v‖_vm ≳ 9`), so a saturated
  at-yield particle's vm can move only through the σ_y(peeq)
  coupling — read the info guard with this in mind.*
- **internal energy, density**: unconstrained (v1 — their measured
  hand-off doses are 0.12/0.01; constraining them has no driver).

**What is and is NOT guaranteed** (scope, stated precisely): D2 ≡ 0
everywhere, and peeq monotone ACROSS THE ANCHOR CHAIN (every emission
≥ its anchor; hand-off-level D3 ≡ 0). Within-segment queries are
independent (ADR-0062), so frame-to-frame peeq between two queries of
the SAME anchor is not ordered by construction — trajectory-level
D3 ≡ 0 is completed by the measured-free eval clamp, which under this
head only ever corrects within-segment ripples bounded below by the
anchor. Any nonzero D2, or peeq below its anchor, IS an implementation
bug; within-segment D3 is a readout, not a tripwire. *Precision
(post-implementation review, 2026-09-12): the guarantees are exact in
real arithmetic; in float32 the RECOMPUTED vm/sigma_y ratio can exceed
1 by ~2 ulp under tanh saturation, and the training-side peeq floor is
normalize/inverse-round-trip-approximate (eval emissions are exact
decodes). The instrument tolerance (1.001) is the operative tripwire;
a strict-zero recomputed check would fire spuriously at the ulp
level.*

The hardening curve `σ_y(peeq)` enters as a fixed per-benchmark table
via a new benchmark-spec hook (Taylor-only v1) — **units MPa, knots
verbatim from the deck (including the non-monotone 251.1 → 250.9
knot), `np.interp` end-clamped semantics as the contract**; this hook
becomes the ONE authoritative table and the analysis scripts
(`consistency_comparison.py`, `gap_analysis_probe.py`,
`tools/state_probe`) migrate to it — retiring copies, not adding one.
Structured raw outputs pass through the existing target normalizer for
the loss, so the `w_pos`/`w_aux` loss FORM and every metric are
unchanged — noting honestly that the structure re-routes gradients
(σ_y′ couples deviator-channel error into the peeq head at a magnitude
comparable to peeq's own signal); that coupling is the mechanism, not
a side effect. One shared decode helper serves `forward_train_tc` AND
`predict_state_at` so train and eval cannot diverge. Requires
`flow_map = true` and the 6-channel state layout (validated at load).
`False` (default) byte-identical.

### Knob 2 — `TransolverConfig.flow_map_consistency_hinge: float = 0.0` (comparator)

Soft admissibility penalties on the unstructured head, weight λ:
`relu(vm_comp − σ_y(peeq_pred))/σ_y0` + `relu(peeq_anchor −
peeq_pred)/peeq_scale`, with `peeq_anchor` the same fed reference as
knob 1 (same rationale, same unreachable-target caveat). Two λ scales
fleet-swept; per the ADR-0049/0063 convention a third scale is owed
before any wrong-mechanism conclusion if both are null — accepted for
a comparator and recorded. Expected weaker per F-011 — it exists so
hard-vs-soft is measured, not assumed. Requires `flow_map = true`;
`0.0` byte-identical. Mutually exclusive with knob 1 (both hinges are
identically zero on structured outputs).

### Explicitly not in this ADR (fleet-config arms, no new surface)

- **Aux-noise re-dose** at the measured per-channel structure IN THE
  KNOB'S UNITS — [0.58, 0.55, 0.78, 0.10, 0.12, 0.01] — with the
  kinematic dose UNCHANGED (0.66/0.03): the measured dose–response
  says reducing it trips the displacement guard (KN-03 at 0.2 failed
  0.0169), and one-mechanism-per-arm is the house rule. Adverse prior
  stated up front: uniform aux noise was measured inert (FM-N0 ≈
  FM-FULL); the arm's hypothesis is *wrong structure, not wrong
  mechanism* — priced, not assumed.
- **Capacity arm** (hidden 256): the gap tracks the oracle, which may
  be trunk-limited; existing knob; measured cost precedent ~1.4×
  (ADR-0049 big arm).
- Eval-side projection/clamp productization for unstructured runs — a
  small separate follow-up (measured ≈free).

## Alternatives considered

- **Keep enforcing at eval only**: measured accuracy-neutral, leaves
  training rehearsing inadmissible states, contradicts the directive.
- **Soft penalties as primary**: against F-011/WAUX3; demoted to
  comparator.
- **`r·sigmoid` magnitude + ε-guarded unit direction**: needs the
  ε-guard (degenerate gradient at dev → 0, which is COMMON —
  pre-arrival particles), an extra raw channel, and a saturation
  cliff; the tanh form is smooth at zero at equal expressiveness.
- **ReLU increment for exact elastic zeros**: dead-gradient zone;
  recorded alternative, revisit if the elastic-creep readout binds.
- **Structured heads for TC/AR**: TC lacks an anchor reference for the
  increment head; AR is retired. Flow-map-scoped v1.

## Consequences

- **Surface changed** (on acceptance): `config.py` (two knobs +
  validation incl. mutual exclusion and the layout check), benchmark
  spec (`hardening_curve` hook, Taylor v1, single-source contract as
  above), `models/transolver/simulator.py` (one shared structured
  decode used by both the training forward and `predict_state_at`;
  `train_output_state`/`set_anchor` unchanged — raw-state consumers),
  `cli/train.py` (hinge terms in the fm loss), mandatory two-key
  migration across the ~110 transolver TOMLs, analysis-script
  migration to the hardening hook, tests (byte-identity off; D2 ≡ 0
  and peeq ≥ anchor on structured outputs as implementation checks;
  tanh-form smoothness at v = 0; hinge gradients; chain smoke).
- The structured arms' D2-own and hand-off-level D3 double as
  implementation tripwires; within-segment D3 and the two recorded
  cost readouts (unreachable-target fraction, elastic-subset
  increment) are diagnostics.
- The fleet (`scratch/2026-09-12-structured-heads-fleet-prereg.md`,
  BINDING) prices both pillars in one round; the decisive bar (17.4)
  sits inside the measured single-seed door (15.55), which the fleet's
  seed-matched arms will confirm or shrink honestly.

### Post-implementation review hardening (2026-09-12)

Three-agent adversarial review of the implementation (decode math /
trainer wiring / config-migration-prereg conformance); all
by-construction claims, anchor references, device placement, and
byte-identity checks verified. Hardened in response: the hinge gets
the SAME load-time guards as the structured heads (canonical-layout +
benchmark-curve presence — it slices the same channel indices);
`hardening_curve` is validated at spec construction (torch.bucketize
is silently wrong on unsorted knots); `load()` rejects a checkpoint
whose hardening buffers differ from the spec table (the
one-authoritative-table contract at the artifact boundary); the hinge
`peeq_scale` floor raised 1e-6 → 0.05 (an all-elastic batch would
otherwise amplify the irreversibility term ~1e5× into clip-saturating
spikes that distort exactly the hard-vs-soft comparison); a
one-table-tripwire test pins the spec knots. Recorded residuals, not
fixed: the five hinge insertion sites have no committed end-to-end CI
coverage (evidenced instead by the real-data micro-train and the
byte-gate; the CI smoke benchmark has no 6-channel state block), and
the GPU path is unexercised until the fleet's first minutes (buffers
and knots are constructed on-device; checked in review).

## Relationship to other ADRs

- **ADR-0063**: chains + kinematic noise stay the base recipe (that
  channel is measured repaired); this ADR targets the remaining aux
  channel and converts eval-time enforcement into architecture.
- **ADR-0062**: scheme unchanged; the anchor interface is what makes
  both heads well-posed (anchored increments, state-complete anchors).
- **ADR-0059**: channel conventions unchanged (same 6 channels, same
  order/units).
- **F-011 / WAUX3 / FM-N0**: the adverse priors this ADR's bets are
  structured around rather than against.
