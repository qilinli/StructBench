# 0031 — Data archive layout: canonical/raw mirrors named by benchmark

**Status**: Accepted
**Type**: Durable
**Date**: 2026-07-05

## Context

Before v0.2's hosting preparation, the out-of-repo data tree (`../data/`,
OneDrive) was organized by simulation campaign: canonical HDF5 sets lived in
three `h5_canonical/` folders under `2D-Copper-Bar-Taylor-Impact/` and
`Concrete-Beam/{1DWavePropagation,2DNotchBeam}/`, mixed with superseded
exports (sgnn-era `npz/`, a pre-canonical `h5/`), exploration scripts, and
the raw d3plot trees. Nothing about that layout said which files a hosted
benchmark archive would contain, and the two notch benchmarks shared one
undifferentiated folder. Dataset hosting (platform choice) remains an open
question, but the folder that will be uploaded should already exist in its
final shape.

## Decision

One consolidated tree at `../data/StructBench/` with two mirrors:

- **`canonical/<benchmark>/`** — one folder per registry name
  (`taylor_impact_2d`, `wave_propagation_1d`, `notch_beam_2d_bend`,
  `notch_beam_2d_impact`), each a self-contained, uploadable archive: flat
  `<case_id>.h5` files plus generated `README.md` + `card.json`
  (`tools/gen_benchmark_docs.py --archive`) and `LICENSE-CC-BY-4.0.txt`.
  The notch pair splits exactly here — the frozen split lists partition all
  221 cases (111 bend / 110 impact, probes 3/2, no overlap) — so each
  archive's file count equals its card's `n_cases`.
- **`raw/<name>/`** — the LS-DYNA binaries, benchmark-aligned names. Raw
  notch stays one pair-folder (`notch_beam_2d/`): its campaign subfolders
  (`ConstantVelocity` → bend, `InitialVelocity` → impact,
  `2DGeneralizibility` → probes) already encode the mapping, and the
  converter reads all three from one root. `rc_beam/` holds the v0.3 raw
  sweep (formerly `Concrete_simulation_constantV1-16`); campaign-level
  provenance documents sit at the `raw/` root.

The campaign wrappers (`2D-Copper-Bar-Taylor-Impact/`, `Concrete-Beam/`)
were dissolved; superseded exports, old scripts, and regenerable
normalization caches were deleted (recoverable via the OneDrive recycle bin
for ~30 days). Converter defaults, `patch_units.py` roots, the card-data
test env vars (`STRUCTBENCH_NOTCH_{BEND,IMPACT}_DATA_ROOT`), and the DUG
staging docs follow the new layout; `size_gb` is measured and set on all
four cards (2.4 / 0.23 / 24.1 / 24.9 GB).

## Alternatives considered

- **Leave data in campaign layout, build a hosting subset by copying**:
  rejected — duplicates tens of GB inside OneDrive and leaves two divergent
  copies to keep honest.
- **Keep raw in place, group only canonical**: rejected by the maintainer —
  the point of the exercise was everything StructBench-related in one tree.
- **Split raw notch by benchmark too**: rejected — raw campaigns map
  many-to-many onto benchmarks (one notch campaign feeds two benchmarks;
  `rc_beam` feeds none yet), and only `canonical/` is ever uploaded.
- **One shared canonical notch folder (as the loader's shared root was)**:
  rejected — two cards cannot both write `README.md`/`card.json` into one
  archive folder, and per-archive file counts would not match either card.

## Consequences

- Hosting becomes "upload `canonical/<benchmark>` somewhere and add the
  link" — the artifact is already assembled, whatever platform wins the
  open hosting question.
- `--data-root` for training points at `canonical/<benchmark>`; the two
  notch benchmarks now use different roots (env vars split accordingly).
- The `derived/` normalization cache regenerates inside each archive folder
  on first local training run; hosted copies stay clean (the cache write
  degrades gracefully on read-only roots).
- Older documents (ADR-0025/0026 §context, the 2026-07-03 plans, the
  pre-2026-07-05 `patch_units.py` docstring) reference the campaign-layout
  paths; per ADR-0009's precedent they describe the layout at their time of
  writing and are not rewritten.
- The private raw tree and the uploadable archives now live under one
  parent, so future backup/sync decisions can treat `StructBench/` as one
  unit.
