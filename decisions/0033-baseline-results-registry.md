# 0033 — Official baseline results live in per-benchmark results registries

**Status**: Accepted
**Type**: Durable
**Date**: 2026-07-05

## Context

The per-benchmark archive README (ADR-0027's generated view, enriched with
task and evaluation-criteria sections in this change) needs a place for
official baseline numbers once the DUG training runs land. Two existing
homes were considered and rejected: the benchmark card is pure task
definition under ADR-0032 (protocol changes bump the benchmark version —
a better baseline must not), and hand-editing generated files breaks the
no-drift guarantee generated views exist for. Results also need provenance:
a number without its run's commit is unverifiable against the
run-directory contract.

## Decision

Each benchmark module records blessed baselines as
`RESULTS: tuple[BaselineResult, ...]` (in `benchmarks/results.py`), wired
into its `SPEC`. A `BaselineResult` carries `family` (model-family key,
ADR-0032), `label`, `run_commit`, `run_date`, `metrics`
(split → metric → value, physical units), optional `checkpoint` pointer,
and `notes`. Validation: required text fields non-empty; at least one
split with at least one metric; split names must exist in the benchmark's
splits (checked at spec construction).

Results render **only** through the generated views: a "Numbers to beat"
table per result in the archive README, and a compact one-line summary in
`docs/benchmarks.md`. An empty registry renders an explicit
"no official baseline yet" placeholder, so shipped archives are honest
rather than silent.

Blessing a run = one small commit (transcribe the run's `metrics-*.json`
into a `BaselineResult`, traceable via the run's `config.json` and
recorded commit) plus regeneration. Adding or revising results never bumps
the benchmark version; only protocol changes do (ADR-0032 §4). Runs with
`protocol.standard = false` are ineligible (ADR-0032).

## Alternatives considered

- **Results as card fields**: one object, but muddies the card's
  task-definition role and its version semantics.
- **Hand-maintained README sections**: generated views could silently
  disagree with the repo; rejected on the no-drift principle.
- **A separate leaderboard file outside the package**: loses the typed
  validation against split names and the single import path the renderers
  already use.

## Consequences

- The v0.1 release step "baseline metrics recorded" becomes mechanical:
  registry entry + `tools/gen_benchmark_docs.py` regeneration.
- Community submissions have an obvious shape to grow into (a result entry
  with provenance), though third-party entries remain out of scope until
  the leaderboard validator (roadmap) exists.
- `render_archive_readme` now takes the registry name (for the
  grouped-config usage snippet); `BenchmarkSpec` gains a `results` field.
- The archive README carries task, evaluation criteria (protocol +
  rationale per ADR-0032, metrics, QoIs), results, and usage — the
  "per-benchmark README" the roadmap called for; dataset info was already
  there.
