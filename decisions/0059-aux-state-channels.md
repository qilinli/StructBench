# 0059 — Auxiliary state channels: `aux` generalises from `(T, P)` to `(T, P, C)`

**Status**: Accepted (maintainer, in-session 2026-09-02)
**Type**: Durable
**Date**: 2026-09-02 (revised same day after a multi-agent surface audit)

## Context

Every benchmark task is currently "positions plus exactly one auxiliary
scalar". That assumption is load-bearing across the pipeline:

- `CaseTrajectory.aux` is `(T, P)`, and `load_case_trajectory` takes a single
  `aux_field: str` (SPH extractor name / mesh `response.node` key);
- `NormalizationStats.aux_mean` / `aux_std` are per-run scalars;
- the mesh-family simulators hard-code `out_size = dim + 1` (Transolver:
  `frames_per_call * (dim + 1)` per ADR-0051); CGN's `n_aux` width exists but
  `cli/train.py` pins it to 1, and every family's training loop scores the
  aux loss as a single trailing channel (`pred[..., -1]`);
- the sample/collate contract (`datasets/particle.py`) fixes `next_aux` at
  `(P,)` per frame, and the ADR-0047 wall augmentation
  (`datasets/sph_mesh.py`) appends a 2-D zero block to `aux`;
- `RolloutResult.predicted_aux` and the saved rollout `.npz` are `(T, P)`;
  `eval/metrics` scores "the" aux field (`aux_rmse`, `rollout_rel_l2_aux`,
  aux QoIs); the viz CLI renders `predicted_aux` as a per-particle scalar
  fringe, and `benchmarks/timeline.py` reduces `aux` to one peak-mean column;
- the aux field is declared twice: `BenchmarkSpec.aux_field`
  (`benchmarks/registry.py`, validated against `available_aux_fields()` and
  what the trainer actually loads with) and `BenchmarkCard.aux_field` /
  `aux_unit` (`benchmarks/card.py`, unvalidated descriptive metadata for the
  rendered pages) — kept equal only by convention, with no cross-check;
- the aux training knobs are scalar: CGN's `aux_transform` /
  `aux_transform_scale` (one asinh knee) and `TrainConfig.aux_tail_weight`
  (one relu tail weight, the blessed notch h250 knob).

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

Declaration and selection:

- The benchmark's canonical channel list lives where the two aux
  declarations live today: `BenchmarkSpec` (validated, per channel, against
  `available_aux_fields()` on the SPH path) and `BenchmarkCard` (name, unit,
  label per channel, for the rendered surfaces). A new spec ↔ card
  consistency check closes today's unchecked duplication. All four public
  benchmarks declare `C = 1`; the first declared channel is the **headline
  channel**.
- Variant tasks select their channels in the *run config*: a new optional
  `train.aux_fields` (list of extractor names, default = the spec's list),
  recorded by `resolved_config_dict` into `config.json`. This is how the
  Stage-3 fleet trains state blocks without touching any card, and it gives
  the current env-var aux-swap (`_env_aux_field_override`, used by the E-X
  strain-swap experiments) a recorded, first-class home; the env var is
  retired for future runs.
- `load_case_trajectory` keeps its `aux_field` keyword and widens its type to
  `str | Sequence[str]` — no rename, existing call sites (timeline, viz, the
  blessing tool) stay valid. A bare string means `C = 1`. New SPH extractors
  for the Taylor state block (deviatoric components, `peeq`, internal energy,
  density) join `available_aux_fields()`; existing extractors are untouched.
- `spec.aux_field` and `card.aux_unit` remain readable as singular
  headline-channel accessors, so read-only consumers keep working unchanged.

Widths, losses, and knobs:

- `NormalizationStats.aux_mean` / `aux_std` become shape-`(C,)`; statistics
  are per-channel (fields differ by orders of magnitude — the Stage-1
  probe's per-channel target scaling, for the same reason).
- Simulator head widths derive from the declared block: `out_size = dim + C`
  (Transolver: `frames_per_call * (dim + C)`, composing with ADR-0051
  unchanged; the `cli/train.py` CGN call site passes `n_aux = C` to the
  width CGN already owns). The training losses score the trailing `C`
  channels as a block (per-channel z-scored, mean-reduced), replacing every
  `pred[..., -1]` scalar slice; the sample/collate contract carries
  `next_aux` as `(P, C)` (and `(P, m, C)` for `frames_per_call > 1`).
- The aux knobs generalise per-channel with scalar shorthand: `aux_transform`
  / `aux_transform_scale` and `aux_tail_weight` accept either today's scalar
  (applied to every channel — byte-identical at `C = 1`) or a per-channel
  list. The relu-based tail weight is only meaningful for non-negative
  channels; signed channels (deviator components) declare `0` there. Stats
  remain computed in transformed space with the transform in the cache key,
  now per-channel.

Reporting and artifacts:

- `eval/metrics` reports per-channel errors (`aux_rmse` /
  `rollout_rel_l2_aux` keyed by channel name) plus the existing pooled
  forms; at `C = 1` every current metric name, value, and registry/report
  surface is unchanged. Aux QoIs bind to a named channel (today's implicit
  binding made explicit); `benchmarks/timeline.py`'s peak-mean-aux column
  binds to the headline channel the same way. The `metrics.json`
  `aux_field` / `aux_unit` keys become lists, with the singular keys retained
  at `C = 1`.
- Rollout `.npz` keeps `predicted_aux` as `(T, P)` when `C = 1` (existing
  tooling and blessed artifacts stay readable) and writes `(T, P, C)`
  otherwise. The viz CLI gains a channel selector (by name, default the
  headline channel) — fringe rendering stays per-scalar.

Multi-channel *tasks* are new configs/variants, never silent edits to
existing benchmarks: the four public leaderboards remain `C = 1` tasks.

## Migration — byte-identical at `C = 1`, no blessed-result invalidation

Same gate as ADR-0053: after the refactor, `C = 1` reference runs must
reproduce the pre-0059 path bit-for-bit — including a Taylor **mesh-family**
config, so the wall-augmentation path (`sph_mesh.py`) and the ADR-0051
bundle split are exercised, not just CGN. The suite covers the
`str | Sequence[str]` loader shorthand, per-channel stats round-trip (with
transforms), head-width derivations for all four families, the `C = 1`
artifact shape, the spec ↔ card consistency check, and timeline/viz smoke on
a `C = 1` card. Existing `config.json` records carry no aux declaration
(the field has always been spec-resolved at train time), so no legacy-record
adapter is needed; records written after 0059 include the resolved
`aux_fields` list.

## Consequences

- Stage-3 variants become ordinary grouped-config runs on the existing
  pipeline (`structbench-train`), directly comparable to the leaderboard
  rows; physical-admissibility diagnostics (yield surface, `peeq`
  monotonicity, tracelessness) read off standard rollout artifacts.
- **Surface changed:** `datasets/canonical.py` (trajectory + loader +
  extractor registry), `datasets/normalization.py` (per-channel stats),
  `datasets/particle.py` (sample/collate contract), `datasets/sph_mesh.py`
  (wall zero-block gains the channel axis), `config.py`
  (`train.aux_fields`, per-channel knob types, `resolved_config_dict`),
  `cli/train.py` (family loss loops, `n_aux` call site, bundle split,
  aux-field threading + env-override retirement, `metrics.json` keys),
  `models/{mgn,transolver,geoflare}/simulator.py` (width derivations, target
  assembly, per-step output splits, TC path), `eval/metrics.py` +
  `eval/rollout.py`, `benchmarks/registry.py` (spec channel list +
  validation + consistency check), `benchmarks/card.py` + the per-benchmark
  `benchmarks/*/card.py` instantiations + `benchmarks/render.py`,
  `benchmarks/timeline.py` (headline-channel binding), `viz/__main__.py`
  (channel selector), configs/tests.
  **Not touched:** the case schema and HDF5 layout (ADR-0042 — the files
  already store every field), splits and protocols, results registries and
  blessed checkpoints, `viz/fringe.py`'s per-scalar rendering contract,
  `tools/state_probe` (superseded as the training vehicle but kept as the
  probe of record).
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
  different state vectors. Rejected — the declared channel list keeps the
  schema benchmark-owned.
- **Declare variant channels on new benchmark cards** instead of a run-config
  field: forces a card (and landing page) per experimental state block,
  conflating the public protocol surface with private variant fleets.
  Rejected — cards stay canonical; configs carry the variants.

## Relationship to other ADRs

- **ADR-0042 untouched**: canonical files already carry the per-element state
  fields; this ADR only widens what the *training* layer reads from them.
- **Composes with ADR-0051/0053/0054**: `frames_per_call`, `history_frames`,
  and time-conditioning are orthogonal axes; width derivations gain `C` in
  place of the literal 1, and the k-frame aux bundle carries `(P, k, C)`.
- **ADR-0055 unchanged**: relative L2 stays the headline; per-channel forms
  extend it without renaming the `C = 1` metrics.
- **ADR-0038's knobs** (`aux_transform`, tail weight) keep their semantics,
  gaining per-channel form with scalar shorthand.
- **Enables** the Stage-3 variant fleet of the complete-state plan (V0
  renormalised control, deviator head, full state, structured heads), which
  will be proposed separately once this lands.
