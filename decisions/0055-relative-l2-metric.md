# 0055 — Relative-L2 metric alongside physical-unit RMSE (literature comparability)

**Status**: Proposed (draft by Claude Code; maintainer finalises)
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

## Open decisions for the maintainer

- **A. Displacement reference frame:** frame-0 (the GT prefix start, consistent
  with the QoIs and the input window) **vs** `reference_coords` (rest geometry, a
  cleaner "deformation from undeformed"). *Recommend frame-0* for consistency with
  the existing displacement QoIs, but rest-coords is defensible.
- **B. Pooling:** leaderboard `rollout_rel_l2` = per-frame-mean (recommended, matches
  the RMSE leaderboard convention). Some papers report a single pooled/whole-sequence
  relative L2; if we want exact parity with a specific paper's table, add a pooled
  variant as a blessing-only / notes field (not a leaderboard column), mirroring the
  ADR-0043/0046 treatment of the pooled RMSE.
- **C. Registry columns:** confirm `rel_l2_*` should appear as leaderboard comparison
  columns (yes — cross-paper comparability is the purpose) rather than notes.

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

## Status / next

Proposed. On acceptance: implement `relative_l2` in `eval/metrics.py`, thread it in
`eval/rollout.py` alongside the RMSEs (AR + TC paths), extend the registry/card
render, re-eval the current baselines, and record the values.
