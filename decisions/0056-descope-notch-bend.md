# 0056 — Descope notch-bend: the notch-beam benchmark narrows to notch-impact

**Status**: Proposed (agent draft; maintainer finalises)
**Type**: Durable
**Date**: 2026-08-17

## Context

ADR-0024 (v0.2 scope) and ADR-0026 (the notch-beam 2D benchmark pair) shipped
two notched-concrete-beam benchmarks that share the same specimen, material,
and notch geometry and differ only in the loading mode:

- **notch-beam 2D bend** — quasi-static 3-point bend.
- **notch-beam 2D impact** — drop-weight impact (high strain-rate).

`notch_beam_2d_bend` has been parked since ADR-0024's amendment: no blessed
baseline, no scheduled release. In practice the two are **very similar** — same
beam, same notch, same concrete-fracture physics — so carrying both on the
public surface adds little cross-benchmark diversity. The impact variant is the
more distinctive and demanding of the two (high-rate, the substrate's focus)
and is the one that carries a blessed CGN baseline (ADR-0039 horizon).

Separately, the public surface is being tightened (README + `docs/benchmarks`).
The docs are generated from the registry (`available_benchmarks()`), and a
drift test (`tests/benchmarks/test_render.py`) asserts docs == registered
benchmarks — so a benchmark cannot leave the public docs without leaving the
registry.

## Decision

**Descope `notch_beam_2d_bend` from the public benchmark set.** The notch-beam
"pair" of ADR-0026 narrows to **notch-impact only**.

1. Remove `notch_beam_2d_bend` from the registry (`_MODULES` in
   `benchmarks/registry.py`), so it leaves `available_benchmarks()` and is
   delisted from the README table, `docs/benchmarks.md`, and the per-benchmark
   pages (regenerated).
2. **Keep the code parked, not deleted.** The `benchmarks/notch_beam_2d_bend`
   module and `configs/notch_beam_2d_bend/` remain in the tree, re-registerable
   by restoring the single `_MODULES` line. This is a delisting, not a deletion.
3. Decouple the render empty-state tests from notch-bend: they now synthesize a
   result-less bare spec (`replace(spec, results=(), card=…)`) rather than
   relying on a specific benchmark having no baseline.

This amends **ADR-0024** (the v0.2 wave + notch-beam scope) and **ADR-0026**
(the notch-beam pair): the pair is now a single public benchmark.

## Alternatives considered

- **Keep both.** Rejected: redundant — same specimen/material/geometry, minimal
  added diversity, and bend has no baseline or plan.
- **Delete the module + configs outright.** Rejected: the code works and is
  cheap to keep; parking (delist, retain) preserves the option to revive it
  without re-implementing, at the cost of some dead code.
- **Hide from docs while keeping it registered** (a docs-exclusion flag).
  Rejected: the drift test ties `docs/benchmarks` to `available_benchmarks()`,
  so deregistration is the clean path; an exclusion flag would be a second,
  parallel notion of "public" to keep in sync.

## Consequences

- The public benchmark set is **four**: Taylor 2D, wave-1D, notch-impact,
  DeformingPlate — each with a blessed baseline.
- `notch_beam_2d_bend` module + configs remain as parked (unreachable) code;
  restoring the `_MODULES` entry re-lists it.
- ADR-0026's "pair" framing is superseded for the public set; the historical
  decisions (0024/0026) are retained as the record, not deleted.
- The render empty-state tests no longer depend on any benchmark being
  result-less (a synthetic bare spec covers those paths).
