# 0032 — Grouped run configuration and benchmark-protocol governance

**Status**: Proposed
**Type**: Durable
**Date**: 2026-07-05

## Context

Three pressures converged in the 2026-07-05 session (hyperparameter audit and
protocol discussion; working notes in `scratch/2026-07-05-*.md`):

1. **The flat TOML config is unsafe at scale.** Unknown keys are silently
   dropped (a typo trains the default); sparse configs inherit dataclass
   defaults that disagree with the accepted recipe — at the time of writing,
   `wave_1d.toml` and the notch configs silently inherit `lr_init = 1e-3`,
   a value ADR-0028 explicitly rejected, and the notch configs inherit
   `max_neighbors = 48`, far under the ~113 physical neighbours at their
   15 mm radius. Nothing distinguishes model architecture from optimization
   schedule from orchestration in one flat namespace.
2. **Multiple baselines on multiple benchmarks are coming** (ADR-0015's
   portfolio; the MS-GNS spec). A config format must dispatch model families
   without rewriting the trainer.
3. **Benchmark protocol was implicit and tunable by accident.** How many
   ground-truth frames seed a rollout (`init` = model window, historically),
   how far the rollout runs, and at what temporal resolution predictions are
   scored — these define the *task*, and the maintainer's own prior work
   showed how easily they get tuned to flatter a particular baseline. A
   ground-truth timeline analysis of the Taylor data (2026-07-05) made the
   stakes concrete (all 33 cases; evidence table committed at
   `docs/timelines/taylor_impact_2d.md`): the rod is in free flight until
   first wall contact near frame 7, so the historical init = 11 hands every
   model the shock onset — up to 10.6% of the kinetic energy is already
   dissipated — while init = 3 and 6 give away 0.0% in every case; 99%
   displacement settlement occurs as late as 296 µs of the 300 µs horizon,
   so the full horizon is dynamically active; and the particle-mean von Mises stress peaks mid-trajectory
   (191 MPa at 44 µs in T-20-80-150, relaxing to 133 MPa), a single-instant
   feature that plain RMSE at native times underweights.

## Decision

### 1. Grouped run configuration

Run configs are TOML files with sections mirroring ownership:

```toml
[run]        # orchestration
benchmark = "taylor_impact_2d"
seed = 0

[model]      # baseline architecture — dispatched by family
family = "gns"
window = 11
# ... every field of the family's config class, explicitly ...

[train]      # optimization schedule and loss weights
batch_size = 32
# ... every TrainConfig field except benchmark/seed, explicitly ...

[protocol]   # OPTIONAL — research override only (see §4)
init_frames = 11
```

Validation is strict: unknown sections or keys are errors; `[model]` and
`[train]` must list **every** field (no silent defaults — dataclass defaults
remain only for programmatic/test construction); `benchmark` and `seed`
belong to `[run]` and are rejected elsewhere. Top-level (flat) keys are an
error with a migration hint. Files live at
`configs/<benchmark>/<family>.toml` (+ `<family>_smoke.toml`).

### 2. Model-family registry

`[model].family` resolves through a registry (`structbench.config`,
today `{"gns": GNSConfig}`) to a config class and, in the trainer, to a
builder. New baselines register a family; the trainer does not change.

### 3. Resolved run record

`config.json` mirrors the nested sections, adds the git commit, and records
the protocol actually used:

```json
{"run": {"benchmark": ..., "seed": ..., "commit": ...},
 "model": {"family": "gns", ...},
 "train": {...},
 "protocol": {"init_frames": 3, "horizon": "full",
              "eval_times": "native", "standard": true},
 "n_particle_types": ..., "data_root": ...}
```

`evaluate()` keeps reading pre-0032 run directories (old flat shape; their
recorded `window` serves as their init), so fleet history stays evaluable.

### 4. Protocol is benchmark-owned, with rationale

`init_frames`, `horizon`, and `eval_times` are **benchmark protocol** — task
definition, not tunables. They live on the benchmark card, pinned by that
benchmark's ADR; changing them is a benchmark version bump. Rules:

- Models may consume **at most `init_frames`** ground-truth frames at rollout
  start. Internal history windows longer than the observed prefix must be
  warm-started by the model (e.g. constant-velocity backfill); the scored
  span is frames `[init_frames, end]` for every model regardless of how many
  frames it consumed.
- `horizon = full`: metrics and QoIs run to the (trimmed) end of trajectory.
- `eval_times = native`: predictions are scored at the solver's output
  times. Internal time-stepping (striding) is a model choice; training-time
  windowing over the TRAIN split is model recipe and unconstrained.
- A `[protocol]` section in a run config is a **research override**: the run
  records `protocol.standard = false` and is ineligible for official card
  metrics. This is the sanctioned path for protocol-sensitivity studies.

### 5. Mandatory ground-truth timeline analysis

Before a benchmark's protocol values are pinned (and whenever its data
changes), a standardized GT characterization **must** be run and its results
recorded as the card's `protocol_rationale`, rendered alongside the protocol
values in the generated benchmark docs. The analysis
(`python -m structbench.benchmarks.timeline`) reports, per case and in
aggregate: kinetic-energy dissipation milestones (50/90/99%), displacement
settlement time, tail activity (late-window mean acceleration vs peak),
KE fraction dissipated within candidate init prefixes, and peak temporal
features of the aux field (value and time). Protocol values without a
recorded rationale are a review error.

### 6. Taylor 2D protocol (amends the ADR-0019 evaluation protocol)

- **`init_frames = 3`** — the second-order minimum (two velocities → one
  acceleration), per the maintainer's F=ma rationale. Measured: 0.0% of KE
  dissipated within a 3- or 6-frame prefix in all 33 cases; first contact
  ≈ frame 7; the historical init = 11 gave away up to 10.6%.
- **`horizon = full`** (151 trimmed frames, 300 µs) — 99% settlement as
  late as 296 µs; the last fifth of the horizon still carries 1.6–8.1% of
  peak mean acceleration (elastic ringing). Nothing is dead time.
- **`eval_times = native`** (2 µs output interval).
- **QoIs gain `peak_von_mises` and `t_peak_von_mises`** (peak of the
  particle-mean von Mises field and its time): a mid-trajectory,
  single-instant feature (191 MPa at 44 µs in the T-20-80-150 ground truth)
  that penalizes temporally coarse surrogates which nail end states but blur
  transients. `final_length` and `mushroom_width` are unchanged.

### 7. Wave / notch protocol values are provisional

`init_frames = 3` provisionally, with `protocol_rationale` marked pending:
their timeline analyses require the ingested datasets (on the ingestion
machine). Running the analysis and pinning the values is a v0.2 gate. The
wave benchmark needs particular care that init stays below the wave's
arrival at the first gauge (the `arrival_time` QoI).

## Alternatives considered

- **Flat config plus a lint tool**: validation bolted on the side rots; the
  flat namespace still cannot dispatch model families.
- **Protocol values in every run config**: invites exactly the
  baseline-favoring protocol tuning this ADR exists to prevent.
- **Grandfather init = 11 for v0.1**: keeps fleet numbers comparable, but
  ships a deployment-dishonest task (the solver would have to produce 22 µs
  of the answer first) and hands models the hardest physics; the ADR-0031
  retrain was already going to reset absolute numbers.
- **Stride as a protocol constant**: dissolved instead — native-time scoring
  plus peak QoIs make temporal resolution a scored property of the model
  rather than a task parameter.

## Consequences

- Absolute metrics under init = 3 are not comparable to the 2026-07-03/04
  fleet (init = 11). The fleet's *relative* conclusions (capacity dominant,
  w_aux load-bearing, noise 0.02) stand; the final baseline run under the
  new protocol re-establishes absolute numbers.
- The task gets harder at the onset: models must predict first contact from
  the wall-distance feature rather than observe it. This is the task being
  honest, accepted deliberately.
- Sparse-config landmines are surfaced by the strict migration: wave/notch
  `lr_init` becomes an explicit 1e-4 (ADR-0028's rate); notch
  `max_neighbors = 48` is kept but flagged — it must be re-sized per
  ADR-0028's "never binds physically" rule before any notch training.
- `GNSConfig`/`TrainConfig` move to `structbench.config` (re-exported from
  `cli.train`); the lenient `from_toml` loaders are removed.
- Old flat TOMLs are deleted; HPC job scripts and README point at the new
  paths; the fleet ablation script's key-rewrite mechanism carries over
  (section keys remain unique file-wide).
- `rollout()` gains `init_frames` decoupled from the model window with
  constant-velocity warm-start — which also enables protocol-sensitivity
  re-evaluation of existing checkpoints without retraining.
