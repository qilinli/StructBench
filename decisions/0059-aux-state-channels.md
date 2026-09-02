# 0059 — Auxiliary state channels: `aux` generalises from `(T, P)` to `(T, P, C)`

**Status**: Proposed
**Type**: Durable
**Date**: 2026-09-02

## Context

Every benchmark task is currently "positions plus exactly one auxiliary
scalar". That assumption is load-bearing across the pipeline:

- `CaseTrajectory.aux` is `(T, P)`, and `load_case_trajectory` takes a single
  `aux_field: str` (SPH extractor name / mesh `response.node` key);
- `NormalizationStats.aux_mean` / `aux_std` are per-run scalars;
- the mesh-family simulators hard-code `out_size = dim + 1` (Transolver:
  `frames_per_call * (dim + 1)` per ADR-0051); CGN's `n_aux` width exists but
  the pipeline pins it to 1;
- `RolloutResult.predicted_aux` and the saved rollout `.npz` are `(T, P)`;
  `eval/metrics` scores "the" aux field (`aux_rmse`, `rollout_rel_l2_aux`,
  aux QoIs);
- the benchmark card declares one `aux_field` (validated against
  `available_aux_fields()`) with one unit/label on the rendered surfaces.

The complete-state thread (plan: Taylor-Simulator 2026-08-31) needs models
that predict a *state block*, not one scalar. The evidence gate for paying
this cost has been passed, in the pre-registered order:

- **Stage 0** (2026-09-01): all six registered Taylor baselines emit
  physically inadmissible von Mises on 11–17% of scored samples
  (vs GT ≈ 0); `peeq` monotonicity is unscoreable for every baseline because
  none predicts plastic state.
- **Stage 1** (2026-09-02, MP trunk): one-step increments of the hidden state
  are unpredictable from kinematics (deviator −39%, `peeq` −42% for the
  full-state arm, kinematic arm at the predict-zero floor; 2 seeds).
- **Stage 1b** (2026-09-02, native `TransolverNet` trunk): the deviator gap
  replicates on the reference operator family (−34%; composed von Mises
  −26%; 2 seeds) — the family can consume the state, not just coexist with
  it.

Training the Stage-3 variants (scalar-vm control, deviator head, full state,
structured state) as *pipeline* runs — same recipes, splits, evaluator, and
leaderboard comparability — requires the aux axis to carry channels.

## Decision

**Generalise the auxiliary target from one scalar to a declared channel
block: `aux` becomes `(T, P, C)`, with `C = 1` the byte-identical special
case.**

- `CaseTrajectory.aux: (T, P, C)`. `load_case_trajectory` accepts
  `aux_fields: Sequence[str]` (each name resolved exactly as today: SPH
  extractor registry / mesh `response.node` key); a bare `str` keeps working
  and means `C = 1`. New extractors for the Taylor state block (deviatoric
  components, `peeq`, internal energy, density) join the SPH registry —
  additions to `available_aux_fields()`, no change to existing ones.
- `NormalizationStats.aux_mean` / `aux_std` become shape-`(C,)`; statistics
  are per-channel (fields differ by orders of magnitude — the Stage-1 probe's
  per-channel target scaling, for the same reason).
- Simulator head widths derive from the declared block: `out_size = dim + C`
  (Transolver: `frames_per_call * (dim + C)`, composing with ADR-0051
  unchanged; CGN passes `n_aux = C` to the width it already owns).
- `eval/metrics` reports per-channel errors (`aux_rmse` / `rollout_rel_l2_aux`
  keyed by channel name) plus the existing pooled forms; at `C = 1` every
  current metric name, value, and registry/report surface is unchanged. Aux
  QoIs bind to a named channel (today's implicit binding made explicit).
- The benchmark card declares the channel list — name, unit, label per
  channel — with today's singular declaration the `C = 1` degenerate form.
  Card `version` does **not** bump for existing benchmarks: their protocol,
  splits, and scored task are untouched.
- **Artifacts:** rollout `.npz` keeps `predicted_aux` as `(T, P)` when
  `C = 1` (existing tooling and blessed artifacts stay readable) and writes
  `(T, P, C)` otherwise.

Multi-channel *tasks* are new configs/variants, never silent edits to
existing benchmarks: the four public leaderboards remain `C = 1` tasks.

## Migration — byte-identical at `C = 1`, no blessed-result invalidation

Same gate as ADR-0053: after the refactor, a `C = 1` reference run
(config load → normalization → training step under fixed seed → eval →
rollout artifact) must reproduce the pre-0059 path bit-for-bit, and the full
test suite covers the `str` → `[str]` config adapter, per-channel stats
round-trip, head-width derivations for all four families, and the `C = 1`
artifact shape. Legacy `config.json` records (`aux_field: str`) map forward
on read, as `velocity_history` did under ADR-0053.

## Consequences

- Stage-3 variants become ordinary grouped-config runs on the existing
  pipeline (`structbench-train`), directly comparable to the leaderboard
  rows; physical-admissibility diagnostics (yield surface, `peeq`
  monotonicity, tracelessness) read off standard rollout artifacts.
- **Surface changed:** `datasets/canonical.py` (trajectory + loader +
  extractor registry), `datasets/normalization.py`, the four simulators'
  width derivations, `eval/metrics.py` + `eval/rollout.py`,
  `benchmarks/registry.py` card fields and their render, configs/tests.
  **Not touched:** the case schema and HDF5 layout (ADR-0042 — the files
  already store every field), splits and protocols, results registries and
  blessed checkpoints, `tools/state_probe` (superseded as the training
  vehicle but kept as the probe of record).
- Estimated effort ~1 week including the byte-identity gate.

## Alternatives considered

- **Keep the state work off-pipeline** (grow `tools/state_probe` into a
  trainer): fast to start, but forks recipes/eval/protocol — results would
  not be leaderboard-comparable, and every diagnostic would need a parallel
  implementation. Rejected; the probe was scoped as an instrument, not a
  vehicle.
- **A separate parallel "state pipeline"** beside the aux axis: duplicates
  normalization, rollout, metrics, and configs for what is structurally the
  same target block. Rejected.
- **Hard-code the Taylor 10-field state** instead of generic `C`: bakes one
  benchmark's closure into the platform; wave/notch/DeformingPlate have
  different state vectors. Rejected — the card-declared channel list keeps
  the schema benchmark-owned.

## Relationship to other ADRs

- **ADR-0042 untouched**: canonical files already carry the per-element state
  fields; this ADR only widens what the *training* layer reads from them.
- **Composes with ADR-0051/0053/0054**: `frames_per_call`, `history_frames`,
  and time-conditioning are orthogonal axes; width derivations gain `C` in
  place of the literal 1.
- **ADR-0055 unchanged**: relative L2 stays the headline; per-channel forms
  extend it without renaming the `C = 1` metrics.
- **Enables** the Stage-3 variant fleet of the complete-state plan (V0
  renormalised control, deviator head, full state, structured heads), which
  will be proposed separately once this lands.
