# Design: hosting-aligned reorganization of the data archive

**Date**: 2026-07-05
**Status**: Approved (maintainer, in-session)
**Scope**: the out-of-repo OneDrive data tree (`../data/`), plus a repo
companion commit (converter/patch/test/docs path updates, archive-view
generation, `size_gb` measurement, ADR-0031 draft).

---

## Problem

The canonical HDF5 sets that the four benchmarks consume live in three
campaign-organized locations (`2D-Copper-Bar-Taylor-Impact/h5_canonical`,
`Concrete-Beam/{1DWavePropagation,2DNotchBeam}/h5_canonical`), mixed with
superseded exports (npz, pre-canonical h5), old exploration scripts, and raw
LS-DYNA binaries. The maintainer wants everything StructBench-related grouped
under one tree whose shape already *is* the future hosting structure — no
duplicate "hosting subset", h5 only for the archives — and stale artifacts
deleted. Hosting itself (upload, platform choice) is explicitly not yet
happening.

## Decisions (settled in-session)

1. **In-place reorganization; no duplicates.** The canonical h5 sets move;
   they are not copied.
2. **One consolidated tree** `../data/StructBench/` with symmetric mirrors
   `canonical/` (hosting-ready archives) and `raw/` (LS-DYNA binaries).
3. **Archive folders are named exactly after benchmark registry names** and
   each is a self-contained uploadable unit: flat `<case_id>.h5` + generated
   `README.md` + `card.json` + `LICENSE-CC-BY-4.0.txt`.
4. **The notch pair splits at the canonical level** (the 5 probes partition
   3 bend / 2 impact — the frozen split lists partition all 221 files with
   zero overlap), but **raw notch stays one pair-folder** (`notch_beam_2d/`)
   because the campaign subfolders already encode the bend/impact mapping
   and the converter reads all three from one root.
5. **The `Concrete-Beam` wrapper dissolves**; raw folders take
   benchmark-aligned names; `rc_beam/` anticipates the v0.3 benchmark.
6. **Stale artifacts are deleted** (list below; OneDrive recycle bin is the
   safety net).

## Target structure

```
../data/StructBench/
  canonical/
    taylor_impact_2d/        34 h5   2.40 GB   T-20-<geom>-<vel>.h5 (+ Convergence)
    wave_propagation_1d/     16 h5   0.23 GB   W1D-<L>-<v>.h5
    notch_beam_2d_bend/     111 h5  ~24 GB     NB-B-* + 3 probe cases
    notch_beam_2d_impact/   110 h5  ~25 GB     NB-I-* + 2 probe cases
    (each folder: + README.md, card.json, LICENSE-CC-BY-4.0.txt)
  raw/
    taylor_impact_2d/lsdyna/20<geom>/<vel>/    (internal layout unchanged)
    wave_propagation_1d/<L>_<v>/               (16 run dirs)
    notch_beam_2d/{ConstantVelocity,InitialVelocity,2DGeneralizibility}/
    rc_beam/                                    (= Concrete_simulation_constantV1-16 contents)
    simulation_specification.xlsx
    GNN-Concrete-15022023.docx
```

Untouched: every non-StructBench dataset folder at the `../data` root
(BLEVE, FGN, Z24 bridge, …) and `Segmental_Beam` (parked candidate; joins
`raw/` only if/when it becomes a benchmark).

## Move map (exact)

| # | Source (under `../data/`) | Destination (under `../data/StructBench/`) |
|---|---|---|
| M1 | `2D-Copper-Bar-Taylor-Impact/h5_canonical/*.h5` (34) | `canonical/taylor_impact_2d/` |
| M2 | `Concrete-Beam/1DWavePropagation/h5_canonical/*.h5` (16) | `canonical/wave_propagation_1d/` |
| M3 | `Concrete-Beam/2DNotchBeam/h5_canonical/<id>.h5` for ids in the bend split lists (111) | `canonical/notch_beam_2d_bend/` |
| M4 | same, ids in the impact split lists (110) | `canonical/notch_beam_2d_impact/` |
| M5 | `2D-Copper-Bar-Taylor-Impact/lsdyna/` | `raw/taylor_impact_2d/lsdyna/` |
| M6 | `Concrete-Beam/1DWavePropagation/<L>_<v>/` (16 dirs) | `raw/wave_propagation_1d/` |
| M7 | `Concrete-Beam/2DNotchBeam/{ConstantVelocity,InitialVelocity,2DGeneralizibility}/` | `raw/notch_beam_2d/` |
| M8 | `Concrete-Beam/Concrete_simulation_constantV1-16/` | `raw/rc_beam/` (rename) |
| M9 | `Concrete-Beam/simulation_specification.xlsx`, `Concrete-Beam/GNN-Concrete-15022023.docx` | `raw/` |
| C1 | `2D-Copper-Bar-Taylor-Impact/LICENSE-CC-BY-4.0.txt` | copied into each of the four `canonical/*/` folders |

M3/M4 membership comes from the package's own frozen split lists
(`get_benchmark(name)` → split mappings), never from filename pattern
guessing. All moves of cloud-only placeholders are metadata renames — no
hydration, no re-upload; OneDrive needs sync-settle time for the big trees.

## Delete list (approved 2026-07-05)

| # | Item | Size | Why |
|---|---|---|---|
| D1 | `2D-Copper-Bar-Taylor-Impact/npz/` | 0.72 GB / 34 files | sgnn-era exports, superseded by canonical h5 |
| D2 | `2D-Copper-Bar-Taylor-Impact/h5/` | 0.29 GB / 33 files | pre-canonical schema (displacement + von Mises only) |
| D3 | `2D-Copper-Bar-Taylor-Impact/README.md` | tiny | documents the old `h5/` set; superseded by generated archive README (after C1) |
| D4 | `Concrete-Beam/.ipynb_checkpoints/`, `extract_run_time.ipynb`, `pg.py`, `read_2D_I.py` | tiny | junk / old exploration scripts |
| D5 | `…/1DWavePropagation/h5_canonical/derived/`, `…/2DNotchBeam/h5_canonical/derived/` | small | regenerable normalization caches, possibly stale post-ADR-0030 |
| D6 | the then-empty `2D-Copper-Bar-Taylor-Impact/` and `Concrete-Beam/` shells | — | dissolved by M1–M9 |

Deletion order: D1–D5 before shells; D6 only after verifying the shells are
empty. OneDrive recycle bin retains deletions ~30 days.

## Gate step: ADR-0030 spot-check (before any move)

Hydrate exactly two files — one wave (`~14 MB`) and one notch (`~230 MB`) —
and read `metadata.attrs['source_units']` with h5py:

- both `kg-mm-ms` → patch was run; proceed.
- any `g-mm-ms` → **stop and surface the decision**: running
  `patch_units.py` over all 237 files means downloading + re-uploading
  ~49 GB through OneDrive. The maintainer decides now-or-later; the reorg
  itself can proceed either way (patch script default roots get updated in
  the companion commit regardless).

Taylor is out of ADR-0030 scope (different deck family; its
`source_units='g-mm-ms'` is believed correct — separate Inbox sanity item).

## Repo companion commit

| File | Change |
|---|---|
| `data_generation/lsdyna/2D-Copper-Bar-Taylor-Impact/convert.py` | default data root → `StructBench/raw/taylor_impact_2d`; default out → `StructBench/canonical/taylor_impact_2d` |
| `data_generation/lsdyna/1DWavePropagation/convert.py` | root → `raw/wave_propagation_1d`; out → `canonical/wave_propagation_1d` |
| `data_generation/lsdyna/2DNotchBeam/convert.py` | root → `raw/notch_beam_2d`; output routed per benchmark (bend/impact dirs) using the frozen probe assignment |
| `data_generation/lsdyna/Concrete-Beam-unit-patch/patch_units.py` | default roots → the three Concrete-Beam-derived canonical dirs (wave, bend, impact) |
| `tests/benchmarks/test_card_data.py` | `STRUCTBENCH_NOTCH_DATA_ROOT` splits into `STRUCTBENCH_NOTCH_BEND_DATA_ROOT` + `STRUCTBENCH_NOTCH_IMPACT_DATA_ROOT` |
| `hpc/dug/README.md` | staging/rsync source paths updated |
| four `card.py` files | `size_gb` set from measured folder sizes; `docs/benchmarks.md` regenerated |
| archive views | `tools/gen_benchmark_docs.py --archive <name> --out <canonical/<name>>` run for all four |
| `decisions/0031-*.md` (draft) | data-archive layout convention (canonical/raw mirrors, benchmark-named archives); human finalises |
| roadmap (README § Roadmap) | v0.2 "Archive packaging" sub-items updated to reflect what this closes |

## Verification

- File-count and name-set checks per canonical folder (34/16/111/110),
  matched against the package split lists — metadata only, no hydration.
- Full `pytest -q` green.
- Env-gated card-data tests run for **taylor** (already local) and **wave**
  (0.23 GB hydration — acceptable); **notch validated by name-set only** (a
  content pass would hydrate ~49 GB — deferred to the DUG staging, which
  downloads it anyway).
- `gen_benchmark_docs.py --check` green after card changes + regeneration.
- Shells confirmed empty (shallow listing) before D6.

## Out of scope

Hosting upload / platform choice (open roadmap question); the full ADR-0030
patch run (own gate above); writing ADR-0030 itself (separate roadmap item);
per-benchmark README content expansion — eval criteria + baseline results
(separate Inbox item; this design ships the current generated README);
`Segmental_Beam`; renaming `data_generation/lsdyna/` script folders.

## Constraints

- Never scan the `../data` tree recursively; shallow listings of specific
  known paths only (CORRECTIONS.md 2026-06-29). Move/delete operations on
  cloud-only placeholders are metadata operations and safe.
- The agent memory note `reference-lsdyna-data-archive` is updated after
  execution so future sessions know the new layout.
