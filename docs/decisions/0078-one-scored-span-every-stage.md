# 0078 — One declared scored span, read by every stage

**Status**: Accepted (maintainer, in-session 2026-11-11)
**Type**: Durable
**Date**: 2026-11-11
**Amends**: 0039 (§3: the long-horizon diagnostic is asked for; §4: the truncated recipe is the trainer's default for any declared span), 0054 (the time reference is recorded), 0032 (run-record and metrics-record fields); 0077 stands

**Your call**:
1. A reference run trains on the span the benchmark scores, by default, wherever a span is declared; `full` is the opt-out. Reversing restores a silent mismatch between what a run trains on and what it is scored on.
2. No registered row moves and none is re-run. The only shipped benchmark with a span is notch-impact, whose reference configs already cut training to it. Reversing would mean re-evaluating rows this record leaves alone.
3. The evaluator stops at the bound; the tail is an asked-for, labelled diagnostic. Reversing brings back rollouts and saved files that are mostly unscored frames.

## Context

ADR-0039 gave notch-impact a scored span and ruled (§4) that its reference recipe trains there. That reached one benchmark, through one integer (`train_frames`) in each config. A span may now be the case's own (ADR-0039 note, 2026-11-06), and only validation and the test metrics followed it: the training pool, the statistics, the rollout length, the saved rollouts and the figures used the whole stored record. A per-case cut would also have silently changed the clock of time-conditioned models, which fell back to the shortest training trajectory.

## Decision

1. **One accessor.** `BenchmarkSpec.scored_frames_for(case)` is the only source of a case's bound. No stage re-derives it. A per-case bound is evaluated for every case of every split when the definition loads.
2. **Training span.** `[train] training_span` takes `scored` (default) or `full`. Under `scored` each training case is cut to its bound before anything else, so every sample of every loop has all its frames, inputs and targets, below it. `train_frames` stays an explicit fixed cut on top of either. On a benchmark that declares no span the two settings are the same. Recipe, not protocol (ADR-0039 §4).
3. **Statistics.** Normalization statistics, cached or fitted online, are computed on the pool as cut. The cache key covers the cut.
4. **Time reference.** One number per benchmark, which does not move with the training span: the fixed bound where there is one (ADR-0054, unchanged); the card's `n_frames` for a per-case bound; for a benchmark with no span, the rule it had. It is written to the run record (`run.time_ref_frames`) and the evaluator reads it from there. A record without the field is evaluated by the rule it was trained under.
5. **Evaluation.** By default a case is rolled out to its bound and no further, and the saved rollout holds those frames and carries the bound. `--tail` rolls on to the last stored frame and writes the errors and properties of the frames beyond the bound in an `unscored_tail` block, per case and in the split mean. The scored numbers are the same either way. `rollout_position_rmse_full` is null when the rollout stopped at the bound.
6. **Figures.** Panels are spaced over the scored span and animations end at the bound, read from the saved rollout; `--full` shows every saved frame.
7. **Records.** The run record states the span setting, the number of samples in the pool, the share of them inside the scored span and the passes the schedule makes over the pool (`run.training_pool`); the same goes to the log at start. A metrics record states `training_span`; a run from before the setting reads `full`. Records trained on different spans are not comparable, as records with different `scored_frames` are not.
8. **Data preparation.** The stored record is not cut (ADR-0039 §1). The timeline analysis reports, per case and pooled, the share of stored frames inside the declared span, and the benchmark page states the span and that share.
9. **Shipped benchmarks.** Taylor, the 1D wave and DeformingPlate declare no span: nothing changes for them. Notch-impact's reference configs set `train_frames = 250`, its bound, so their pool is the same under the new default; its registered rows were trained that way and stand. Notch-impact evaluations now stop at frame 250 unless `--tail` is given.

## Alternatives considered

- **Keep `full` as the default.** Changes no run by itself. Rejected by the maintainer: the mismatch was silent.
- **Move the time reference with the span.** The time feature would change meaning against every run made so far. Rejected by the maintainer.
- **Always roll to the end.** The tail is most of the compute and of the saved files on a short-span benchmark.
- **A margin of frames past the bound; sampling that weights cases equally.** Not taken. A long-span case contributes more samples than a short one while the metric is a mean over cases; a later note can add either.
- **A scored-share quantity in the data-check catalogue (ADR-0066).** Not taken: the share is a fact about a declaration, not a measurement of the reference data.

## Consequences

- To reproduce an earlier whole-record run on a benchmark with a span, set `training_span = "full"`.
- One-shot Transolver (`frames_per_call = 0`) needs one trajectory length and cannot train on a per-case span.
- A tool that pairs a saved rollout with the stored record must read the span the rollout carries. The pooled-RMSE gate runs on DeformingPlate only, which has no span.
