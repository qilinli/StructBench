# 0025 — Benchmark cards: typed per-benchmark metadata with generated views

**Status**: Proposed
**Type**: Durable
**Date**: 2026-07-03

## Context

With v0.2 the platform grows from one benchmark to four. Two audiences
need to understand the nature of each dataset quickly: structural
engineers (which solver, which discretisation — SPH/FEM/coupled, which
material models, erosion or not, what loading) and ML researchers (how
many cases, particles, frames, which fields, what task, split sizes,
size on disk). Today that information has no home: the Taylor
benchmark's contract is code constants (`benchmark.py`), and the
data-side README was a one-off scratch script.

The design question raised by the maintainer: per-datafolder metadata or
a combined index? Both are needed and neither should be the source of
truth — hand-maintained copies drift.

## Decision

1. **One canonical card per benchmark, as code**: each benchmark module
   ships a `card.py` defining one instance of a typed `BenchmarkCard`
   dataclass (the type lives in `benchmarks/card.py`). Fields cover:
   - *identity*: name, version, one-line description, provenance
     (paper reference, who ran the simulations, when), data license;
   - *physics*: solver and version, discretisation
     (`Literal["SPH", "FEM", "coupled"]`), material models, erosion
     (bool), loading description, unit system, geometry summary;
   - *ml*: case count, split sizes, particles-per-case range, frame
     count, output interval, available fields, task type, auxiliary
     targets, QoI names, size on disk.

2. **Derivable numbers are computed, not declared.** Split sizes and
   case counts are computed from the module's own split constants
   (`len(TRAIN)`, …) — the card and the benchmark cannot disagree.
   Physics facts, which cannot be derived, are declared. Stats that
   live only in the data (particle counts, frames, size on disk) are
   validated by a test that runs when the data root is present and
   skips otherwise.

3. **Two generated views** (a small script under `tools/`):
   - `docs/benchmarks.md`: the combined cross-benchmark comparison
     table — the "which benchmark do I want" overview — plus a summary
     row in the repository README;
   - a per-dataset-archive `README.md` and `card.json` (stdlib
     `json`), shipped with the hosted data so a downloaded dataset is
     self-describing regardless of where hosting lands (ADR-0021 §5).

4. **Taylor gets its card retroactively in v0.2**, so the index
   launches complete.

## Alternatives considered

- **TOML card** (stdlib `tomllib` read). The initially drafted design;
  rejected on review: derived numbers become hand-copied literals
  needing a drift test, a runtime schema validator must be written and
  maintained where mypy checks the dataclass for free, and its one
  advantage — shipping verbatim with the data — dissolves because the
  archive needs a generated README anyway, alongside which a generated
  `card.json` is free.

- **YAML card.** Requires a new dependency (PyYAML) against the
  dependency policy, for no capability TOML/dataclasses lack. If
  hosting lands on Hugging Face, its YAML-frontmatter card is simply
  another generated view.

- **JSON as the source.** No comments; provenance notes and field
  explanations need them.

- **Hand-written per-folder READMEs plus a hand-maintained index
  table.** The status quo trajectory; rejected as guaranteed drift.

## Consequences

- New convention: every benchmark module contains `card.py`; the
  benchmark contract remains split across `benchmark.py` (task
  constants) and `card.py` (descriptive metadata), both typed and
  mypy-checked.

- A generation script and a card-vs-data validation test enter the
  repo; regeneration of `docs/benchmarks.md` becomes part of the
  benchmark-addition checklist (and a CI check when CI exists).

- Community contributors describe a new benchmark by filling in a
  typed dataclass with IDE support, not by learning an ad-hoc format.

- The card is Python-only at source; any non-Python consumer reads the
  generated `card.json`. This is judged acceptable because every
  source-level consumer (pipeline, docs generator, tests) is Python.
