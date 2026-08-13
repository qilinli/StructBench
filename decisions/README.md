# Decisions

This folder holds the project's Architecture Decision Records (ADRs). Each ADR captures one decision, its context, the alternatives considered, and its consequences. Together, they form the project's decision history.

---

## Format

Each ADR is one markdown file with the following structure:

```
# NNNN — Title

**Status**: Accepted | Proposed | Superseded by NNNN
**Type**: Durable | Ephemeral
**Date**: YYYY-MM-DD

## Context
What problem or question prompted this decision.

## Decision
What was decided.

## Alternatives considered
What else was on the table, and why not.

## Consequences
What becomes easier, harder, or constrained as a result.
```

### Filenames

`NNNN-kebab-case-title.md`, where NNNN is a zero-padded sequential number (0001, 0002, ...). Numbers are never reused, even when decisions are superseded.

### Status

- **Proposed** — drafted, not yet approved by the human.
- **Accepted** — current, active decision.
- **Superseded by NNNN** — replaced by a later ADR. The superseded ADR is kept for history; the new one references it.

### Type

- **Durable** — effectively permanent. Revising requires a new superseding ADR with explicit reasoning. Small adjustments that do not reverse the decision (timing slips, parking a sub-item) may instead be recorded as a dated amendment note appended to the ADR and reflected in the index Status column *(maintainer, 2026-08-06)*; supersession remains required for genuine reversals.
- **Ephemeral** — expected to change as the project evolves. Can be updated in place with a dated note appended to the ADR; supersession is not required.

---

## Index

| # | Title | Type | Status |
|---|-------|------|--------|
| 0001 | Adopt harness engineering methodology | Durable | Accepted |
| 0002 | Project name is StructBench | Durable | Accepted |
| 0003 | v0.1 anchor problem is impact on RC beams | Durable | Superseded by 0015 |
| 0004 | Platform is solver-agnostic; LS-DYNA for v0.1 data generation | Durable | Accepted |
| 0005 | ADR format and decision-log structure | Durable | Accepted |
| 0006 | Three-tier authority model for Claude Code | Durable | Accepted (amended by 0023) |
| 0007 | CORRECTIONS.md mechanism for small corrections | Durable | Accepted |
| 0008 | Principle/mechanism separation between HARNESS and CLAUDE | Durable | Accepted |
| 0009 | Session-start reading list | Ephemeral | Accepted |
| 0010 | FEM solver code lives outside the importable package | Durable | Accepted |
| 0011 | Case vocabulary for the data record | Durable | Accepted |
| 0012 | Case schema field-level structure | Durable | Accepted |
| 0013 | HDF5 persistence layout for the case schema | Durable | Accepted |
| 0014 | StructBench is the substrate layer of a broader research program | Durable | Accepted |
| 0015 | v0.1 ships existing LS-DYNA datasets as benchmarks with prior-paper GNN baselines (supersedes 0003) | Durable | Accepted (amended by 0021, 0024) |
| 0016 | LS-DYNA d3plot is the canonical ingestion path; general adapter on lasso-python | Durable | Accepted |
| 0017 | Relationship to NVIDIA PhysicsNeMo: independent substrate, opt-in model-edge interop | Durable | Accepted |
| 0018 | PyTorch + PyG are hard runtime dependencies of the ML layer | Durable | Accepted |
| 0019 | v0.1 Taylor 2D benchmark: autoregressive surrogate task, split, and eval protocol | Durable | Accepted (amended by 0032, 0035) |
| 0020 | Native radius_graph; no graph-backend binary dependency | Durable | Accepted |
| 0021 | v0.1 narrows to Taylor 2D; portfolio spreads across releases (amends 0015) | Durable | Accepted |
| 0022 | FEM-convention visualization harness (`viz/`, matplotlib as optional extra) | Durable | Accepted |
| 0023 | Git authority: `main` moves on explicit in-session instruction (amends 0006) | Durable | Accepted |
| 0024 | v0.2 ships the 1D wave and notch-beam benchmarks; RC beam moves to v0.3 | Durable | Accepted (amended 2026-08-06: notch-bend parked; v0.3 scope superseded by 0041, 2026-08-07) |
| 0025 | Wave 1D benchmark: task, split, and eval protocol | Durable | Accepted |
| 0026 | Notch-beam 2D benchmark pair: two benchmarks, tasks, splits, eval | Durable | Accepted (amended by ADR-0029) |
| 0027 | Benchmark cards: typed per-benchmark metadata with generated views | Durable | Accepted (amended by 0032) |
| 0028 | GNS baseline training-recipe rework after the first full run | Ephemeral | Accepted |
| 0029 | Notch-beam aux is max principal strain, not K&C damage (amends 0026) | Durable | Accepted (amended in place 2026-08-06: 0.01 threshold declared, provisional flag resolved) |
| 0030 | Concrete-Beam decks are kg-mm-ms; canonical data patched in place | Durable | Accepted |
| 0031 | Data archive layout: canonical/raw mirrors named by benchmark | Durable | Accepted (amended by 0037) |
| 0032 | Grouped run configuration and benchmark-protocol governance (amends 0019, 0027) | Durable | Accepted (amended by 0035) |
| 0033 | Official baseline results live in per-benchmark results registries | Durable | Accepted (amended by 0037; extended by 0046) |
| 0034 | The reference baseline is CGN (Concrete Graph Network, Li et al. 2023) | Durable | Accepted |
| 0035 | The model input window is the rollout init; no history backfill (amends 0019, 0032) | Durable | Accepted |
| 0036 | Per-benchmark landing pages: one generated docs page per benchmark (extends 0027) | Durable | Accepted (extended by 0046) |
| 0037 | Blessed runs archive: `models/` mirror and registry checkpoint pointers (amends 0031, 0033) | Durable | Accepted |
| 0038 | Auxiliary-channel training knobs: target-space transform and tail weight | Durable | Accepted |
| 0039 | Notch-impact scored horizon: 250 µs evaluation window, matched baseline recipe | Durable | Accepted |
| 0040 | Dataset hosting: maintainer's OneDrive stays the master; archives shared on request | Ephemeral | Accepted |
| 0041 | v0.3 pivots to a public multi-method benchmark: DeformingPlate with native MGN/Transolver/GeoFLARE (supersedes ADR-0024's v0.3 scope) | Durable | Accepted (amends 0034; corrected in place 2026-08-07 re schema, see 0042) |
| 0042 | Schema 0.2.0 adds per-node fields; nodal-FE ingestion via download-and-convert (deforming_plate) | Durable | Accepted (corrects 0041) |
| 0043 | DeformingPlate benchmark protocol: task, split, eval, and the MGN blessing gate | Durable | Accepted (narrowed by 0046) |
| 0044 | Transolver provisional adaptation: native Physics-Attention on the DeformingPlate rollout | Durable | Accepted |
| 0045 | GeoFLARE provisional adaptation: native GALE_FA (GeoTransolver + FLARE) on the DeformingPlate rollout | Durable | Accepted |
| 0046 | Provisional results and the method-comparison table (closes ADR-0041 clause 4) | Durable | Accepted |
| 0047 | Taylor 2D multi-method extension: native MGN/Transolver/GeoFLARE on the SPH benchmark | Durable | Accepted |
| 0048 | Notch-impact multi-method extension: native MGN/Transolver/GeoFLARE on the notched-beam SPH benchmark | Durable | Accepted |
| 0049 | Taylor native recipe repair: noise rescale, velocity history, MGN stretch gate | Durable | Accepted |

---

## Adding a new ADR

1. Claim the next available number by checking the highest NNNN in use.
2. Create `NNNN-kebab-case-title.md` using the format above.
3. Draft the ADR. Claude Code may draft; the human finalises.
4. Add a row to the index in this README.
5. Commit with a message like `docs: add ADR-NNNN on <title>`.
