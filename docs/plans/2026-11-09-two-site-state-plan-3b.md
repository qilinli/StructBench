# Plan 3b: the restart task family

*Build plan for part 3b of `2026-11-07-two-site-state-design.md` (ADR-0075,
Accepted 2026-11-07, clause 5; §6 of the design). Written 2026-11-09 before
the work on `feat/two-site-state-3b`, branched from `feat/two-site-state-3c`
(plan 3c, unmerged: 3b reads its spec surface and its properties block).
Scope agreed with the maintainer in session (2026-11-09): the declaration
surface and the machinery only; no shipped benchmark declares anchors, so
nothing new renders on the four landing pages until a definition by path
does, and the first restart rows appear on the Abaqus benchmark at
admission. The full suite is the gate.*

## What the family is

A restart task supplies the stored two-site state at an anchor frame and
scores the prediction over a horizon; the definition declares anchors and
horizons as it declares the scored span, fixed or per case (ADR-0075 clause
5). The family is the restarts at each anchor, the composition sweep (one
interval reached in one, two or more compositions), the semigroup error,
and the record of which horizons the run trained on, so a reader sees which
pairs are held out. A kinematics-only scheme is reported from the kinematic
anchor and marked so; it cannot restart from a stored state.

**A restart is the existing rollout on a window** (this plan's call): the
frames from `anchor − input_frames + 1` to `anchor + horizon`, re-bound to
the simulator, so the autoregressive and flow-map paths run unchanged and
score every predicted frame of the window. The window carries the element
state, so a two-site flow map restarts from the stored state (`restart_kind
= "state"`); every other scheme restarts from the kinematic anchor
(`"kinematic"`). The flow map's absolute anchor-time feature
(`flow_map_anchor_time`) must see the true anchor, so `flow_map_rollout`
gains `frame_offset` (default 0, byte-identical), added to the anchor's
normalised time only; the query offsets `Δt` are relative already.

**A time-conditioned run without the flow map is refused** (this plan's
call, recorded as a ruling): it carries nothing between frames (ADR-0054),
so there is nothing to restart; the mode exits with one line naming the
scheme.

**Keys** (per case and in the split mean; the four field metrics of the
rollout, plus the two state metrics on a two-site run):

| Row | Keys | Who |
|---|---|---|
| Restart at anchor `a`, horizon `h` | `restart_a{a}_h{h}_{rel_l2_displacement, rel_l2_aux, position_rmse, aux_rmse[, rel_l2_state, state_rmse]}` | every scheme; the flow map at its canonical interval |
| Composition of `T = h` in `k` steps | `compose_T{h}_k{k}_…` (same metrics) | the flow map only, self-anchored at interval `h / k`, for each declared `k` that divides `h` |
| Semigroup | `prop_semigroup_err` | the flow map only: the pooled relative L2 between the `k = 2` and `k = 1` predicted positions over the window, averaged over anchors and horizons; a property, so it joins the block |

A registry row gains `restart_kind` (`""`, `"state"`, `"kinematic"`),
rendered in a *Restart family* tier after the quantities of interest.

## Tasks

1. **Window** — `eval/restart.py`: `restart_window(trajectory, anchor,
   horizon, input_frames) -> CaseTrajectory` (positions, aux, time, state
   sliced to `[anchor − input_frames + 1, anchor + horizon]`, the rest
   shared), raising `ValueError` when the anchor lacks history or the
   horizon runs past the trajectory; `restart_keys(anchor, horizon,
   metrics)` and `compose_keys(horizon, k, metrics)`; `semigroup_error(
   pos_k1, pos_k2)` (pooled relative L2 of the predicted frames). Exports
   from `eval/__init__.py`. Tests (`tests/eval/test_restart.py`): the window's
   shapes and that frame `input_frames − 1` of the window is the anchor; the
   state block rides along; the two errors; the semigroup error of identical
   predictions is 0 and of a doubled displacement is the expected ratio.
2. **`frame_offset`** — `flow_map_rollout(..., frame_offset: int = 0)`,
   added to `anchor_t_norm` at the first anchor and at every re-anchor.
   Test (`tests/eval/test_rollout_two_site.py` or the flow-map rollout
   tests): with a recording simulator stub, the anchor times the rollout
   passes are shifted by the offset and nothing else changes; `frame_offset
   = 0` is byte-identical.
3. **Spec surface** — `BenchmarkSpec.restart_anchors: tuple[int, ...] |
   Callable[[str], tuple[int, ...]] | None = None`, `restart_horizons:
   tuple[int, ...] = ()`, `restart_compositions: tuple[int, ...] = (1, 2,
   4)`; `restart_anchors_for(case_id)` (validates each anchor in
   `[input_frames − 1, n_frames − min(horizons))`); `defines_restarts`.
   Validation at construction: horizons non-empty iff anchors declared,
   positive and strictly increasing; compositions positive, strictly
   increasing, starting at 1; fixed anchors checked against the card. The
   two-site smoke fixture declares `restart_anchors=(3,)`,
   `restart_horizons=(2,)`, `restart_compositions=(1, 2)`.
   `BaselineResult.restart_kind: str = ""`, validated against the three
   values. Tests in `tests/benchmarks/test_registry.py` and
   `tests/benchmarks/test_results.py`; the four shipped benchmarks define no
   restarts.
4. **`evaluate_restart`** — `cli/train.py`: `evaluate_restart(case_ids,
   data_root, out_dir, device, *, split_name, save_artifacts, checkpoint)`
   mirroring `evaluate`: rebuilds the run, refuses a benchmark without
   restarts and a time-conditioned non-flow-map run, and per case, anchor
   and horizon builds the window, re-binds a case-bound simulator to it
   (kinematic positions, aux, state and law of the window), resets, and
   runs `rollout` (autoregressive) or `flow_map_rollout` (canonical
   interval, `frame_offset = anchor − input_frames + 1`, self-anchored) with
   `qois=None`; for the flow map also each composition; writes the keys
   above, `prop_semigroup_err`, and `restart_kind`; the report carries
   `"restart_kind"`, `"anchors"` per case, `"horizons"`, `"compositions"`,
   `"train_intervals"` (the record's `flow_map_eval_intervals` where the run
   is a flow map, else `[]`), `"cases"` and `"mean"`; written to
   `metrics-restart-<split>.json` (checkpoint-suffixed like `evaluate`).
   `--mode restart` runs every eval split but `val`, like `rollout`. Tests:
   `tests/cli/test_train_eval.py` on the CGN tiny run with anchors declared
   on the local spec (kind `kinematic`, the four keys finite, no compose or
   semigroup keys); `tests/cli/test_two_site_smoke.py` (kind `state`,
   restart and compose keys finite, `prop_semigroup_err` finite and
   present in the mean); the time-conditioned smoke refuses with a
   `ValueError` naming the scheme.
5. **Renderer** — `_partition_metrics` returns `(props, rel_l2, rmse, qoi,
   restart)` with `restart_*` and `compose_*` keys in the last group; a tier
   `Restart family (ADR-0075 clause 5)` after the QoI tier on the
   in-distribution split, with a `Kind` column from `restart_kind` and
   `_col_label` reading `restart_a10_h20_rel_l2_displacement` as
   `interp·a10 h20 disp` and `compose_T20_k2_position_rmse` as
   `interp·T20 k2 pos (mm)`; no tier when no result carries the keys. Tests
   in `tests/benchmarks/test_render.py`; the committed pages are unchanged.
6. **Docs** — `docs/ARCHITECTURE.md` (`eval/` paragraph: `restart.py`); the
   design's dated note (3b built: the window, `frame_offset`, the refusal
   of time-conditioned runs, keys); `docs/plans/README.md` row; README
   Roadmap: the restart bullet done, the Inbox entry updated; the CLAUDE.md
   snapshot's "the restart family is owed" removed.

## Gate

Full suite green. `flow_map_rollout` at `frame_offset = 0` and every
`evaluate` key byte-identical. On the two-site smoke every law-reading
decoder's restart rows are finite, `prop_semigroup_err` is finite, and the
`k = 1` composition of horizon 2 equals the single-anchor flow map on the
window. The four landing pages do not change.
