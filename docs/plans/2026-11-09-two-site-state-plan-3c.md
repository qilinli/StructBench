# Plan 3c: the properties block

*Build plan for part 3c of `2026-11-07-two-site-state-design.md` (ADR-0075,
Accepted 2026-11-07, clause 6; §7 of the design). Written 2026-11-09 before
the work on `feat/two-site-state-3c`; the full suite is the gate. Scope
agreed with the maintainer in session (2026-11-09): the block's columns ship
in v0.4 on every landing page; the numbers on the shipped benchmarks come
later from a re-evaluation of the blessed checkpoints on DUG, not from this
plan. Part 3b (the restart family) is not started; the semigroup property
is its row and waits for it.*

## What the block is

Every registered result carries, where the benchmark defines them, a group of
properties rendered **before** the accuracy columns (ADR-0074 clause 7;
ADR-0075 clause 6). Each property is one kernel, pure numpy over the rollout
record, calling the verification kernel where one exists so the reference
data and a prediction are judged by the same function. No property gates.

| Property | Key | Defined when | Kernel |
|---|---|---|---|
| Closure | `prop_closure_ratio` | always | sum over the scored span of the free-rollout position RMSE over the same sum of the teacher-forced one-step RMSE; undefined (`null`) for a time-conditioned run, which has no one-step sweep (ADR-0054) |
| Error growth | `prop_error_growth` | always | least-squares slope of `log(position RMSE)` against the predicted-frame index over the scored span, per frame; frames with zero error are dropped; fewer than two frames give `null` |
| Yield bound | `prop_yield_violation_rate` | `state_site == "element"` | fraction of (frame, element) in the scored span with `vm > (1 + tol) · σ_y(peeq)`, the yield stress by `verification.kernels.interp_yield_stress` from the case's own law (ADR-0075 clause 3) |
| Plastic-strain monotonicity | `prop_peeq_decrease_rate` | `state_site == "element"` | fraction of element steps in the scored span whose `peeq` falls by more than `tol`, the count from `verification.kernels.decrease_max` |
| State sufficiency | `prop_state_sufficiency` | `matched_initial_state_pairs` declared | pooled `‖X̂_a − X̂_b‖₂ / ‖X_a − X_b‖₂` over the scored span of a declared pair, positions; 1 is exact, 0 a model that reads no state |
| Deviator trace | `prop_trace_max` | *deferred* | see below |
| Semigroup | `semigroup_err` | *part 3b* | the restart family's row |

**Deferred: the deviator trace.** The design's `prop_trace_max` reads a
predicted deviator. The one element-site layout in the package stores four
stress components and `peeq` (`RETURN_MAP_STATE_CHANNELS`); its deviator is
derived, so its trace is zero for every arm by definition and the property
carries no information. The kernel is written when a state layout carries a
separately emitted deviator (CORRECTIONS 2026-09-21: implement a piece when a
dataset first exercises it). The block renders the properties the benchmark
defines, so its absence is within ADR-0075 clause 6; recorded as a dated note
on the design.

**Where the numbers travel** (this plan's call, not the design's): `prop_*`
keys ride inside the split-keyed `metrics` of a result and the `cases` and
`mean` of `metrics-<split>.json`, the way `gradient_*` keys do (ADR-0076), so
`BaselineResult` gains no field and legacy rows are unchanged. The renderer
partitions them into their own tier by prefix.

**Tolerances** are declared by the benchmark definition
(`BenchmarkSpec.property_tolerances`, keys `yield` and `peeq`), taken from the
reference data's own measured excess in its verification record and never
tighter (ADR-0064 review note); both are required when an element site is
declared, so an author reads the record before the block can fill.

## Tasks

1. **Kernels** — `eval/properties.py`: `closure_ratio(rollout_err,
   one_step_err) -> float`, `error_growth(err) -> float`,
   `yield_violation_rate(von_mises, peeq, knots_peeq, knots_sy, tol) ->
   float`, `peeq_decrease_rate(peeq, tol) -> float`,
   `state_sufficiency(pred_a, pred_b, true_a, true_b) -> float`,
   `von_mises4(stress) -> ndarray` (numpy twin of the torch helper, for the
   `(s11, s22, s33, s12)` layout), and `PROPERTY_KEYS`. Non-finite results
   are `nan` (`_json_safe` writes them `null`). `eval/__init__.py` exports
   them. Tests (`tests/eval/test_properties.py`): closure of equal series is
   1, of a doubled rollout 2, of a zero one-step `nan`; error growth of
   `exp(0.1 t)` is 0.1 to 1e-9, of a constant 0, of one frame `nan`; yield
   rate on a block at `0.9 σ_y` is 0, at `1.01 σ_y` with `tol = 0.02` is 0
   and with `tol = 0` is 1; peeq rate on a monotone block is 0 and counts a
   single drop of `2 tol` as `1 / n_steps`; sufficiency is 1 when the
   predictions equal the truth, 0 when both predictions coincide; the numpy
   `von_mises4` matches the torch one on random input.
2. **Spec surface** — `BenchmarkSpec.property_tolerances: Mapping[str,
   float] = {}` and `matched_initial_state_pairs: tuple[tuple[str, str],
   ...] = ()`; `defined_properties -> tuple[str, ...]` (the keys of the
   table above the benchmark defines, in table order). Validation: the
   tolerances need `state_site == "element"`, are non-negative, and both
   keys are required with an element site; a pair's two ids are distinct,
   known cases of the same split. Tests in `tests/benchmarks/test_registry.py`;
   the four shipped benchmarks define exactly `(prop_closure_ratio,
   prop_error_growth)`.
3. **Evaluate** — `cli/train.py::evaluate` writes `prop_closure_ratio` and
   `prop_error_growth` per case from the scored-span arrays it already has
   (`result.position_rmse[:n_scored]`, `one_step[:n_scored]`); for a two-site
   run with `predicted_state`, the yield and peeq rates over the scored
   frames with the case's law from `spec.constitutive_law(case_id)`; after
   the loop, `prop_state_sufficiency` for each declared pair whose two ids
   are both in `case_ids` (predicted positions of pair members kept in
   memory), written on both cases; `mean` carries each key by
   `_mean_over_cases`. Tests: `tests/cli/test_train_eval.py` asserts the two
   keys per case and in the mean and that no other key changed;
   `tests/cli/test_two_site_smoke.py` gains the tolerances on its fixture
   and asserts the law-reading decoders' yield rate is 0 and the peeq rate 0
   while the corner-mean arm's keys are present; the time-conditioned smoke
   asserts `prop_closure_ratio` is `None`.
4. **Renderer** — `_partition_metrics` returns `(props, rel_l2, rmse, qoi)`
   with `prop_*` first; a tier `Properties (ADR-0075; ↓ better for rates and
   growth)` rendered before the headline tier, whose columns are the
   benchmark's `defined_properties` on the in-distribution split (the
   `interp` the trimmed tiers use) *whether or not any result carries them*,
   plus any `prop_*` key a result does carry; empty cells `—`; one line under
   the tier: *An empty cell is a property the result has not yet reported;
   no property gates (ADR-0075 clause 6).* `_col_label` strips `prop_`;
   `_headline_metric` never picks a `prop_*` key. The registry preamble
   sentence names the block. Pages regenerated (`tools/gen_benchmark_docs.py`):
   the four landing pages gain the empty tier; the index is unchanged.
   Tests in `tests/benchmarks/test_render.py`: the tier precedes the
   headline tier on Taylor with its two columns and dash cells; a result
   carrying `prop_closure_ratio` renders the value; a `prop_*` key never
   lands in the RMSE tier; the committed pages are up to date.
5. **Scoring tool** — `tools/score_submission.py::score_case` adds
   `prop_error_growth` from the per-frame position RMSE of the submitted
   prediction (the one property a prediction file supports: closure needs a
   one-step sweep, the constraint properties an element-site submission,
   which no benchmark has). Test in `tests/tools/test_score_submission.py`.
6. **Docs** — `docs/ARCHITECTURE.md` (`eval/` paragraph: `properties.py`);
   the design's dated note (3c built; trace deferred; semigroup with 3b);
   `docs/plans/README.md` row; README Roadmap: the v0.4 bullet splits into
   the properties block (done) and the restart family (owed), the gradient
   bullet drops its stale branch pointer, the open-evaluation strand's
   `CONTRIBUTING.md` line is crossed out and its "renders first once 3c
   lands" sentence updated; the CLAUDE.md snapshot's "the restart family and
   the properties block are owed" reads "the restart family is owed".

## Gate

Full suite green. Every existing registered row and every metric key other
than `prop_*` is byte-identical before and after (the `metrics.json` of the
CGN tiny run in `test_train_eval.py` differs only by the two new keys per
case and in `mean`). On the two-site smoke the law-reading decoders'
`prop_yield_violation_rate` and `prop_peeq_decrease_rate` read 0 at the
fixture's declared tolerances. The four landing pages carry the properties
tier first with dash cells, and nothing else on them changes.
