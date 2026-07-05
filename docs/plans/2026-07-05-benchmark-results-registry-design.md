# Design: per-benchmark results registry + enriched archive README

**Date**: 2026-07-05
**Status**: Approved (maintainer, in-session)
**Scope**: `src/structbench/benchmarks/` (new `results.py`, registry wiring,
renderer), per-module `RESULTS`, tests, `docs/benchmarks.md` + archive-README
regeneration, ADR-0033 draft, roadmap update.

---

## Problem

The archive README generated into each hosted dataset folder (ADR-0027,
`render_archive_readme`) is a thin dataset view: no task statement, no
evaluation criteria, and nowhere for baseline results to land when the DUG
runs finish. The maintainer wants each benchmark's README to carry dataset
info, evaluation criteria, and (once trained) baseline results — without
hand-editing generated files.

## Decisions (settled in-session)

1. **Official baseline results live in a per-benchmark results registry**,
   not on the card (cards stay pure task definition per ADR-0032) and not
   hand-pasted (generated views must not drift).
2. The archive README gains **Task / Evaluation criteria / Numbers to beat /
   Using this archive** sections; `docs/benchmarks.md` renders the same
   results from the same registry.
3. An **empty registry renders an explicit placeholder** ("No official
   baseline yet …") — shipped archives are honest rather than silent.
4. **ADR-0033** records the governance: results enter only via the registry,
   require run commit + date, and never bump the benchmark version.

## Components

### `src/structbench/benchmarks/results.py` (new)

```python
@dataclass(frozen=True)
class BaselineResult:
    family: str            # model-family key, e.g. "gns" (ADR-0032 registry)
    label: str             # display name, e.g. "GNS baseline"
    run_commit: str        # git commit of the blessed training run
    run_date: str          # YYYY-MM-DD
    metrics: Mapping[str, Mapping[str, float]]
    #   split name -> {metric name -> value}; split names must be a subset
    #   of the owning card's splits (validated where spec wires them)
    checkpoint: str | None = None   # pointer/URL once checkpoints publish
    notes: str = ""
```

Validation in `__post_init__`: non-empty `family`, `label`, `run_commit`,
`run_date`, and `metrics`. Split-name validation against the card happens in
the registry wiring (the card is not visible from the dataclass).

### Per-module `RESULTS`

Each benchmark module (`taylor_impact_2d`, `wave_propagation_1d`,
`notch_beam_2d_bend`, `notch_beam_2d_impact`) exports
`RESULTS: tuple[BaselineResult, ...] = ()` — empty today. Recording a
baseline after a blessed run = adding one entry transcribed from the run's
`metrics-*.json` (traceable via its `config.json` + commit) and
regenerating the views.

### Registry wiring

`BenchmarkSpec` gains `results: tuple[BaselineResult, ...]` (default `()`),
populated by each module next to `CARD`. `get_benchmark`/`available_benchmarks`
unchanged. At spec construction, validate every result's metric split names
against `card.splits` (raise `ValueError` on unknown split).

### Renderer changes (`render.py`)

`render_archive_readme(spec)` — after the existing dataset facts, append:

1. `## Task` — `c.task`; aux target `c.aux_field` (`c.aux_unit`); one fixed
   sentence on the autoregressive rollout setup.
2. `## Evaluation criteria` — protocol line (init `c.init_frames` frames,
   horizon `c.horizon`, scored at `c.eval_times` output times) + the
   recorded `c.protocol_rationale`; the platform-standard metrics (one-step
   and full-rollout position RMSE in mm; aux RMSE in `c.aux_unit`); QoIs
   `c.qois`.
3. `## Numbers to beat` — if `spec.results` is empty:
   "*No official baseline yet — the reference run's metrics land here.*"
   Otherwise one table per result: rows = splits (card order), columns =
   union of metric names in stable order; header line carries label, family,
   run date, and commit (+ checkpoint when set); `notes` as a trailing line.
4. `## Using this archive` — `pip install structbench`, then
   `structbench-train --mode train --config configs/<benchmark>/gns.toml
   --data-root <path to this folder> --out runs/<benchmark>-gns`
   (benchmark name from `spec.name`), and the existing card.json/loader
   pointers move here.

`_section(spec)` (docs/benchmarks.md) — after the QoIs line, add a
**Baseline** line: either the placeholder or, per result,
"`label` (family, run_date, commit): headline metrics" — compact single
line per result, with the full table living in the archive README.

### ADR-0033 (draft, Durable)

"Official baseline results live in per-benchmark results registries":
context (results need a home that is neither the task-defining card nor
hand-edited generated output), decision (registry + generated views only;
entries require run commit/date; results never bump benchmark version —
protocol changes do, ADR-0032), alternatives (card fields, hand-maintained
sections — both rejected in-session), consequences (blessing a run is a
small commit + regeneration; leaderboard growth is additive).

## Testing

- `tests/benchmarks/test_results.py`: `BaselineResult` validation (empty
  fields raise); spec wiring rejects unknown split names.
- `tests/benchmarks/test_render.py` additions: archive README with empty
  results carries the placeholder; with a fabricated result renders the
  table (label, commit, split rows, metric columns); docs section renders
  the Baseline line both ways.
- Existing drift test covers `docs/benchmarks.md` via regeneration;
  `gen_benchmark_docs.py --check` green after regen.
- Full suite green; ruff/format clean.

## Regeneration

`tools/gen_benchmark_docs.py` (unchanged mechanics): regenerate
`docs/benchmarks.md` and run `--archive <name> --out
../data/StructBench/canonical/<name>` for all four benchmarks
(metadata-size writes; no OneDrive hydration).

## Out of scope

Checksums/manifest (Zenodo supplies md5s at publish); citation entries
(release action); root-README result tables (root stays platform-level);
recording any actual result (no blessed runs exist yet).

## Roadmap effect

Inbox "per-benchmark README" item closes; the v0.1 release item's "baseline
metrics recorded (per-benchmark README)" now has its mechanism (registry
entry + regeneration after the DUG retrain).
