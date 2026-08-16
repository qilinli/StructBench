# 0055 — Relative-L2 metric as the headline (literature comparability)

**Status**: Accepted (maintainer approved in-session, 2026-08-15); **amended
2026-08-15** — relative L2 lifted from *additional* metric to the **headline**
metric (see Amendment). The original "Decision" section below is the as-accepted
record; the Amendment supersedes decision 1's metric hierarchy.
**Type**: Durable
**Date**: 2026-08-15

## Context

The neural-operator / physics-ML literature reports errors almost universally as
**relative L2**: for a field `u` and prediction `û`, `‖u − û‖₂ / ‖u‖₂` (Wu et al.
Transolver, Li et al. FNO/Geo-FNO, GNOT, and the GeoTransolver/GeoFLARE crash
papers all use it). StructBench instead scores in **physical units** — position
RMSE (mm), aux RMSE (MPa / strain), and physical QoIs (deflection mm, cracked
fraction, peak stress MPa) — which is a deliberate engineering-relevance choice
(VISION) and underpins the blessed baselines and the **published MGN-DeformingPlate
gate (15.1 ×10⁻³ pooled RMSE)**.

The gap costs us cross-paper comparability: we cannot currently place our
Transolver/GeoFLARE numbers next to the *published* operator numbers in the same
metric — which directly weakens v0.3's "cross-method comparison" headline. This ADR
adds relative L2 as an **additional reported metric**, without displacing the
physical-unit metrics.

## Decision

1. **Add `rel_l2_displacement` and `rel_l2_aux` per split, ALONGSIDE** the existing
   `*_position_rmse`, `*_aux_rmse`, and `qoi_abs_error`. The physical-unit RMSEs +
   QoIs remain the engineering headline and the blessed/gate anchors — **unchanged**.
2. **`rel_l2_displacement` is computed on displacement, not absolute position — and
   named for it.** The L2 norm of absolute coordinates is dominated by the coordinate
   *origin* (a beam at x∈[−199,199] has a huge `‖u‖` from mere position), so relative
   L2 on absolute positions is origin noise, not error. Unlike `rollout_position_rmse`
   — whose *error* norm is origin-independent, so "position" is honest there — relative
   L2 divides by `‖u‖` and **must** use displacement `u = pos − pos_ref`; the metric
   name says `displacement` to make that explicit and pre-empt exactly the
   origin-confusion the metric avoids. Reference frame = the **frame-0** convention
   already used by the QoIs (`terminal_peak_displacement`, midspan deflection).
   *(Open decision A.)*
3. **`rel_l2_aux`** is computed on the raw aux field (strain / stress). For
   near-zero-background fields (strain is ~0 in undeformed regions, localised at
   cracks) relative L2 is legitimately dominated by the high-magnitude region —
   documented, not corrected.
4. **Formula & masking:** `rel_l2 = ‖û − u‖₂ / max(‖u‖₂, ε)` over the **scored**
   particles with the **same kinematic/scripted exclusion and scored horizon** as
   the RMSE metrics (ADR-0026/0035/0039). `ε` guards a near-static frame.
5. **Aggregation:** compute per-frame relative L2 over the scored horizon, then
   `rollout_rel_l2_* = mean over scored frames` — matching how `rollout_position_rmse`
   is aggregated (per-frame RMS → mean over frames), so the two read consistently.
   A `one_step_rel_l2_*` mirrors the one-step RMSEs for AR methods; it is **N/A for
   the time-conditioned scheme** (ADR-0054, no single-step notion), reported null
   like `one_step_position_rmse`. *(Open decision B: whether to also record a
   pooled-RMS variant to match papers that pool.)*
6. **Scope:** applies uniformly to all four benchmarks (Taylor, notch, wave-1D,
   DeformingPlate). Small addition in `eval/metrics.py` (a `relative_l2` alongside
   `position_rmse`/`field_rmse`) + `eval/rollout.py` (compute alongside RMSE) + the
   registry render (ADR-0046 surfaces every `metrics` key as a comparison column —
   which is the point here).

## Resolved decisions (maintainer approved, 2026-08-15)

- **A. Displacement reference frame → frame-0.** Consistent with the input prefix
  and the existing displacement QoIs (`terminal_peak_displacement`, midspan
  deflection); `u = pos − pos[0]`.
- **B. Pooling → per-frame-mean** for the leaderboard `rollout_rel_l2_*` (matches the
  RMSE leaderboard convention: per-frame relative L2 → mean over scored frames). A
  pooled/whole-sequence variant is NOT added now; it can be added later as a
  blessing-only / notes field if exact parity with a specific paper's table is
  needed (mirroring ADR-0043/0046's pooled-RMSE treatment).
- **C. Registry columns → yes.** `rel_l2_displacement` and `rel_l2_aux` are surfaced
  as leaderboard comparison columns — cross-paper comparability is the purpose.

## Alternatives considered

- **Replace RMSE with relative L2.** Rejected: breaks StructBench's physical-unit
  engineering identity (VISION), the blessed CGN/MGN numbers, and the published
  MGN-DP RMSE gate.
- **Relative L2 on absolute position.** Rejected: origin-dominated and meaningless
  (decision 2).
- **Do nothing.** Rejected: forfeits direct comparison to the published operator
  numbers, the exact gap the user flagged and the v0.3 cross-method headline needs.

## Consequences

- **Separable from the training loss.** ADR-0054 decision C kept the RMSE *loss*
  over thuml's relative-L2 loss for comparability; that is the training objective and
  is untouched. This ADR only adds an *evaluation* metric.
- **No change to blessed numbers, the DP RMSE gate, or the frozen splits.** Purely
  additive reporting.
- **Re-evaluation:** existing runs must be re-evaluated (a cheap rollout-eval pass)
  to populate the new keys, or the metric is forward-only for new runs.
- **Payoff:** our Transolver/GeoFLARE (and future crash-benchmark) numbers become
  directly comparable to the published operator relative-L2 tables.

## Amendment (2026-08-15): relative L2 is the headline metric

*Maintainer-directed in-session. Supersedes decision 1's "the physical-unit RMSEs +
QoIs remain the engineering headline" and converts the rejected "Replace RMSE with
relative L2" alternative into a **partial** acceptance — relative L2 leads, but RMSE
is retained, not removed.*

The as-accepted ADR added relative L2 *alongside* the RMSEs but kept RMSE as the
headline. That under-serves VISION's stated core purpose — "evaluation protocols that
let results be compared meaningfully across methods." The neural-operator literature
reports relative L2 near-universally; leading with it is what makes StructBench's
numbers directly readable next to published tables (the v0.3 cross-method goal). The
engineering-relevance half of VISION is preserved by *retaining* the physical units,
not by leading with them.

**Revised decisions (supersede decision 1 where they conflict):**

1. **`rollout_rel_l2_displacement` is the headline / ranking metric on every
   benchmark leaderboard** — the first column and the number quoted in the docs-index
   one-liner; `rollout_rel_l2_aux` beside it. Uniform across Taylor, notch, wave-1D,
   DeformingPlate.
2. **Physical-unit RMSE (`*_position_rmse`, `*_aux_rmse`) is retained as a secondary
   column group** — still rendered on every leaderboard and card, demoted *below* the
   relative-L2 headline group, never dropped. Keeps the engineering-relevance identity
   (a structural engineer still reads mm / MPa) and stays the substrate for the
   blessed anchors.
3. **QoIs unchanged** — the physical engineering quantities (deflection mm, cracked
   fraction, peak stress MPa) are a separate reporting axis, rendered as their own
   group after the trajectory-error groups; neither headline nor secondary-RMSE. They
   are **reported, not ranked** (maintainer, 2026-08-16): StructBench's primary audience
   is surrogate-modelling *algorithm developers*, for whom field accuracy (relative L2)
   is the comparison metric that ranks the leaderboard; the engineering-outcome axis is
   what the *downstream user* focuses on, so the QoIs stay a reported group rather than
   the ranking target. A future benchmark may still elect to co-headline a specific QoI
   if its scientific point demands it — that is a per-benchmark call, not the default.
4. **The blessing gate metric is per-source and orthogonal to the leaderboard
   headline.** The DeformingPlate MGN gate stays *pooled position RMSE* (ADR-0043) —
   there is no published MGN-DP relative-L2 number to validate blessing against. The
   operator baselines' published comparison anchor *is* relative L2. What leads the
   leaderboard does not move any benchmark's blessing criterion.
5. **Headline aggregation stays per-frame-mean** (decision B) so it reads
   consistently with the retained RMSE column. *Open note:* some operator papers
   report a pooled / whole-sequence relative L2; if exact table-parity with a specific
   paper is later needed, add the pooled variant as a blessing-only companion
   (mirroring ADR-0043/0046's pooled-RMSE treatment) rather than switching the
   leaderboard aggregation.

**Consequences.** A reporting-hierarchy change only: no metric *definition*, no
blessed number, no frozen split, and no blessing gate moves. The registry/card render
(ADR-0046) reorders into three groups — relative-L2 headline, RMSE secondary, QoI —
and degrades gracefully for runs whose relative-L2 keys are not yet populated (they
render RMSE-first until the pending re-eval fills the new keys).

## Follow-up amendment (2026-08-16): pooled space+time aggregation for the headline

*Maintainer-approved in-session after a code-grounded literature review of the
Transolver family (deep-research, 2026-08-16, 23/25 claims verified against primary
sources).* Decision **B above (per-frame-mean) is superseded for the headline.** The
first re-eval exposed the flaw: per-frame-mean divides a real error by a near-zero
single-frame reference norm, so it blows up on any field that starts at zero — Taylor
`rollout_rel_l2_aux = 5.7×10⁸` on every case, because von Mises stress is ~0 before
impact. The fix is the aggregation the Transolver family actually uses.

**What the Transolver family does (verified in code, not prose).** thuml's
`TestLoss.rel` (`utils/testloss.py`) flattens each sample with `reshape(N, -1)` —
folding space, channels, and **time** into one vector — then `‖pred−gt‖₂/‖gt‖₂`, mean
over the batch, **no epsilon**. For the time-dependent Plasticity/Navier–Stokes
benchmarks (`exp_plas.py`, `exp_ns.py`) the reported rollout number is `test_l2_full`:
the whole concatenated trajectory flattened to `[B, -1]` (pooled space+time per
sample), `/ntest`. The maintained Neural-Solver-Library reports only this pooled
number; GeoTransolver states it explicitly (`ε_L2 = Σⱼ‖x̃ⱼ−xⱼ‖₂ / Σⱼ‖xⱼ‖₂` over the
predicted *spatiotemporal* response). The crash sub-line (GeoFLARE, NVIDIA
PhysicsNeMo) **diverges** to a per-timestep error curve — which is why per-frame-mean
is retained as a secondary, not discarded.

**Revised decision B — the headline is pooled per trajectory, per quantity.**
`rollout_rel_l2_displacement` and `rollout_rel_l2_aux` are each computed by flattening
the *scored* rollout of that quantity — scored frames × scored particles × the
quantity's commensurate vector components — into one vector, `‖pred−gt‖₂/‖gt‖₂`, one
ratio per trajectory, then **mean over the split's trajectories** (= thuml's
batch-mean `/ntest`). The two quantities are **not merged**: displacement (mm) and aux
(MPa / strain) are incommensurable, so each is its own pooled ratio — pooling mm with
MPa would let the larger-magnitude field swamp the other. Within displacement the
`{x,y[,z]}` components *are* pooled (same unit). Same kinematic mask, scored horizon,
and frame-0 displacement reference as before. `eps = 1e-12` is a pure exact-zero guard
(degenerate fixtures only), **not** a scale knob — the pooled denominator is the whole
trajectory's field energy and cannot be driven near zero by an early ~0 frame.

**Per-frame-mean retained as a SECONDARY metric** (`rollout_rel_l2_*_perframe`) for
comparability with the GeoFLARE / PhysicsNeMo crash line (per-timestep error). It is
never the headline — it is the unguarded quantity that blows up on zero-start fields.

**Relative L2 is a ROLLOUT-only metric; one-step stays RMSE.** A second code-grounded
literature review (deep-research, 2026-08-16) established that the one-step
(teacher-forced) error is a standard reported diagnostic *only* in the
GNS/MeshGraphNets autoregressive lineage — where it is an MSE/RMSE (GNS
`one_step_position_mse`; MeshGraphNets "RMSE 1-step") — and that **no** published work
reports a one-step *relative* L2 (the neural operators report no one-step error at all,
only a whole-output field relative L2). A one-step relative L2 therefore has no
cross-paper comparability target and is N/A for the time-conditioned / one-shot schemes,
so it is **not computed**. One-step is reported as RMSE (`one_step_position_rmse`,
`one_step_aux_rmse`, matching the GNS/MGN convention); relative L2 is reported for the
rollout only.

## Status / next

Accepted + amended. Implemented (2026-08-15) on `feat/native-baselines`:
`relative_l2` in `eval/metrics.py`, threaded in `eval/rollout.py` (AR + TC paths,
commit 1c7ad7b); the ADR-0046 render reordered so relative L2 leads. Remaining: the
cheap re-eval of the current baselines to populate the new keys, then record the
values in each benchmark's registry.
