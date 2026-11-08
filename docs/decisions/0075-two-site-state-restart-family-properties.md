# 0075 — The two-site state, the restart task family and the properties block (ADR-0065 follow-ups 1 and 3)

**Status**: Accepted (maintainer, 2026-11-07)
**Type**: Durable
**Date**: 2026-11-07
**Amends**: 0065 (clause 4; follow-ups 1 and 3), 0059, 0060, 0062, 0064, 0043 (clause 5), 0032, 0033, 0055; constrains 0019, 0025, 0026 (a task family is added; their tasks stand). The case schema (0012, 0013, 0042) is untouched.

**Your call**:
1. A mesh case's state has two sites, kinematics at the nodes and constitutive state at the integration points; the integration-point site is scored, checked and restarted from, the nodal projection (loader note, 2026-11-06) is a view. Reversing re-scores every element-site record.
2. Every registered result reports the properties block before its accuracy, and the restart family is a registered task family that only state-carrying simulators serve. Reversing re-renders the registries and withdraws the restart rows.
3. The constitutive state is carried by the simulator, not inside the network: the network reads only the projected view (maintainer, 2026-11-07). Reversing is a model-family change, cheap; a second site inside the network is the recorded alternative.

## Context

ADR-0065 left two follow-ups: a properties block with every result (clause 4) and a restart task family only state-carrying simulators can serve. The pieces exist, the contract does not. Abaqus-pipeline cases store stress and plastic strain per element; the loader projects them to nodes as an area-weighted mean, admissible by convexity but a smoothed proxy, so no arm predicts the solver's state, hands one back to a solver or restarts from a stored one. ADR-0064's heads bake one hardening table per benchmark; ADR-0062's flow map restarts from a nodal state; the verification kernels judge element state, the training path a projected one.

## Decision

1. **Sites.** A case's state is kinematics at the nodes and, where the solver stores it, constitutive state at the integration points: stress, equivalent plastic strain and the internal variables the material class names (ADR-0070). `CaseTrajectory` gains an element-site block read verbatim from `response/element`; the definition declares which channels live at which site.
2. **Authority.** Where a benchmark declares an element site, that site is scored, property-checked and restarted from. The nodal projection remains the cheap baseline and the like-for-like row: an element-state arm is also reported projected to nodes, beside arms that predict only there.
3. **Carrying.** The simulator carries the constitutive state between frames or anchors; the network predicts nodal kinematics and reads the element state only as its projected view (the ADR-0060 input surface). An admissible arm derives the element state by the material class's return map from the carried state, the kinematic increment and the case's own law; an unconstrained arm emits it from a head. ADR-0064's one-table hook becomes a per-case law supplied by the definition; a checkpoint stores no table.
4. **Task.** A benchmark's task is trajectory prediction from an initial state and the case parameters, with the prediction scheme declared; the seed frames (ADR-0035) are the kinematic part of that state. Existing tasks and registered rows stand.
5. **Restart family.** A restart task supplies the stored two-site state at an anchor frame and scores the prediction over a horizon; the definition declares anchors and horizons as it declares the scored span, fixed or per case (ADR-0039 note). The family is the restarts at each anchor, held-out interval pairs and the composition sweep (one interval in one, two or more compositions). A kinematics-only scheme may be reported from a kinematic anchor, marked so; it cannot restart from a stored state.
6. **Properties block.** Every registered result carries, where the benchmark defines them: constraint satisfaction at the state site (yield bound, plastic-strain monotonicity, deviator trace), state sufficiency where the benchmark has matched-initial-state cases, closure (teacher-forced one step against free rollout), error growth with horizon, and for restartable simulators the semigroup error. Each property has one kernel, the verification kernel where one exists (ADR-0066), with tolerances no tighter than the reference's own measured excess (ADR-0064 review note). No property gates. Accuracy is read within the set of results that report the block (ADR-0065 clause 4); registries and landing pages render the block before the accuracy columns.
7. **Metrics.** State-site accuracy keeps the ADR-0055 forms per channel; restart metrics are keyed by anchor and composition count. Keys, kernels and rendering: the plan.

## Alternatives considered

- **Nodal state only, projected.** Admissible but not the solver's state; no solver hand-back, no restart from a stored state. Kept as the baseline.
- **A second site inside the network**, element tokens with messages to their corner nodes. Deferred by the maintainer's call: it changes every family, and the decoder-only arm first measures whether the network needs it.
- **A new case-schema version.** Unneeded: the files already store element fields; only the ML-side trajectory and the protocol change.
- **Gating on properties.** Thresholds would precede any admissible arm on a mesh benchmark; reporting first, a gate by a later note.

## Consequences

- Surface: `datasets/canonical.py` (element block), `benchmarks/registry.py` (site declaration, per-case law, anchors and horizons), `eval/` (state-site metrics, restart rollouts, property kernels shared with `verification/`), `cli/train.py`, the results registries and landing pages, and the model families that opt in, Transolver first. Design, metric keys and build order: `docs/plans/2026-11-07-two-site-state-design.md`.
- Existing benchmarks declare no element site and are unchanged; their registered rows carry an empty properties block until re-evaluated.
- Follow-ups 2 and 4 of ADR-0065 remain open.

---

**Note (2026-11-08) — the admissible arm in two forms (clause 3).** The
first run of the direct return-map decoder on a metal did not train (plan
3a, note 2026-11-08): stress derived from a surrogate's *predicted*
kinematics amplifies a displacement error by `E / (h · std σ) ≈ 10³`, so the
state loss trains the volumetric strain of the displacement field and not the
positions. The maintainer's decision (2026-11-08, in session): the direct
return map is not repaired; the admissible arm is tested in two forms that
both abandon deriving stress from predicted kinematics, and the results
choose the final form.

1. **Admissible coordinates** (`two_site_decoder = "admissible_coords"`):
   the head predicts the element state in coordinates where every point is
   admissible — the equivalent plastic strain as the *carried* anchor value
   plus a softplus increment (monotone by construction); the deviator as a
   trace-free direction times a radius in `[0, 1)` times the yield stress of
   the case's own law at the updated plastic strain (on or inside the
   surface, trace-free, by construction); the mean stress a free channel. No
   constitutive call in the training loop: the element-site port of
   ADR-0064's structured heads.
2. **Trial state + projection** (`"projected_trial"`): the head predicts an
   unconstrained element state and the radial return projects it onto the
   admissible set inside the training graph — closed form, differentiable
   almost everywhere with the ReLU-like kink at the elastic/plastic switch,
   and non-expansive in the law's own metric (`|dev|²_F / 2G + H ε̄ᵖ²`), in
   which it is a closest-point projection.

Both forms read the carried anchor state and never the network's noised
anchor input, so training noise cannot ratchet the monotone plastic strain
(the 2026-09 structured-heads lesson); plan 3a's rejection of the
pushforward chains and the first-validation divergence guard apply unchanged.
Clause 3 is read accordingly: an admissible arm derives the element state
from the carried state and the case's own law — by the return map from the
kinematic increment (the direct form, kept as `return_map`), or from a head
read in admissible coordinates or projected by the return map. The
unconstrained arm (`corner_mean`) is unchanged. Invariants are pinned for any
head output in `tests/models/common/test_two_site.py` and
`tests/models/transolver/test_transolver_two_site.py`; the build is recorded
in plan 3a's note of the same date.
