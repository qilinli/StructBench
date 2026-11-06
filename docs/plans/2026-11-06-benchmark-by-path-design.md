# A benchmark definition loaded by path

*Design note, 2026-11-06. Not an ADR: by the bar in `docs/decisions/README.md`
every part is either a dated note on a record it changes (ADR-0032, ADR-0039)
or implementation, and no judgment here needs the maintainer's context
(CORRECTIONS 2026-11-06). Written before the build; a historical record once
executed.*

## Why

The training entry resolves a benchmark by registry name, and the registry
holds only the shipped benchmarks inside the package. A dataset generated
with `structbench-datagen` therefore cannot be trained on until a module for
it is written into the package, which would put the dataset's name, splits
and constants into the public tree before admission (CORRECTIONS
2026-08-12; WORKFLOW publication note). ADR-0071 already made the *dataset*
a definition passed by path. This note makes the *benchmark* the second
such definition, so a data-producing user can train on their dataset before
it is admitted, and the maintainer's first Abaqus dataset is the first user.

## What a benchmark is today

One frozen `BenchmarkSpec` built by a package module: the card with the
protocol constants, the split lists, the auxiliary channel selection, the
QoI functions, a boundary feature function, node-type roles, one hardening
table, one scored-horizon bound, a per-case loading scalar and the results
registry. The training entry and the evaluator read it at about
twenty-five points; the run config names it under `[run].benchmark`
(ADR-0032) and the run record stores that string.

## Design

1. **Loading by path.** `[run].benchmark` accepts a directory as well as a
   registry name. The directory holds `benchmark.py` exporting a
   `BenchmarkSpec` named `SPEC`; it is Python, not TOML, because the spec
   carries functions. The importer is the one `structbench.datagen` already
   uses for `problem.py`, moved to `core/definition_module.py` so both the
   dataset and the benchmark definition load the same way: no bytecode
   written beside the definition, the directory importable only while the
   module loads, the siblings it imported unloaded afterwards. The config
   loader rewrites the field to the directory's absolute path, the spec
   records the directory and the LF-normalised sha256 of `benchmark.py` and
   its siblings (`BenchmarkSpec.source`), and every run record carries that
   block under `run.definition`. A path-loaded benchmark is unadmitted: no
   registry row, no landing page, no results registry. The protocol checks
   the config loader runs for a registered benchmark (`input_frames` against
   the card, the aux layout, the hardening table's presence) run for a
   definition too.
2. **A per-case scored horizon.** `scored_frames` (ADR-0039) also accepts a
   function of the case id; `scored_frames_for(case_id)` is what every
   rollout reads, `scored_frames_fixed` what the time-conditioned query's
   reference reads (it falls back to the recipe truncation or the
   trajectory length when the bound is per case, so the physical clock is
   the same for every case), and a metrics record says `"per_case"`.
3. **Element fields reach the models by nodal projection.** The mesh loader
   reads an auxiliary name as a node field (ADR-0043) or, failing that, as
   a registered field computed from the case's one element block and
   projected onto the nodes by an area-weighted mean of the adjacent
   elements (tetrahedron volume for 3D four-node cells). A positive-weighted
   mean of admissible integration-point states is admissible under a convex
   yield function and a hardening law linear in the plastic strain, and
   monotonicity and tracelessness survive it, so the projected reference
   satisfies the constraints the element state does. It is a smoothed proxy
   of that state; the two-site state, with kinematics at nodes and the
   constitutive state at the integration points, is ADR-0065's
   task-definition follow-up and is not built here. A case without node
   types has one type. `stress_axisymmetric` joins the registry: the four
   in-plane Voigt components in the working stress unit.
4. **Quadrilaterals.** `cells_to_edges` kept every vertex pair of a cell,
   exact for triangles and tetrahedra; a 2D four-node cell now contributes
   its four sides and not its diagonals, which no shipped benchmark's mesh
   is affected by (Taylor's lattice mesh is triangles, DeformingPlate is
   tetrahedra).

## Not in this note

The per-case hardening curve. The structured heads bake one table into the
model at construction and the checkpoint loader compares it; making it per
case changes ADR-0064's model code and belongs with the two-site state.

## Records

Dated notes on ADR-0032 (the benchmark field and the unadmitted status) and
ADR-0039 (the per-case bound); `docs/ARCHITECTURE.md`'s `benchmarks/` and ML
data-flow paragraphs; a closing section of `docs/DATA_GENERATION.md` on
training before admission; this note's row in `docs/plans/README.md`.
