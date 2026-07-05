# ARCHITECTURE.md

*The technical spine of StructBench: package layout, module responsibilities, and the case schema. Decisions in this document are durable unless explicitly marked otherwise.*

---

## Purpose

This document describes how StructBench is structured at the code level: what modules exist, what each is responsible for, how they relate to each other, and what data structures they share. Architectural changes are recorded by editing this document and creating a corresponding ADR.

For coding conventions (style, testing, documentation expectations), see PRINCIPLES.md. For the philosophy behind these structural choices, see HARNESS.md.

---

## Package layout

```
src/structbench/
├── core/          # case schema, graph utilities, I/O primitives
├── benchmarks/    # benchmark problem definitions
├── models/        # reference ML models
├── datasets/      # data loaders and dataset registry
├── eval/          # metrics and evaluation protocols
├── viz/           # FEM-style visualization of physics fields
└── cli/           # command-line interfaces

# Reserved namespaces (declared but not yet implemented)
├── deploy/        # asset onboarding and deployment workflows (post-v0.1)
├── vision/        # computer-vision-based damage detection (post-v0.1)
└── sensing/       # sensor-stream anomaly detection (post-v0.1)
```

The `src/`-layout is used (rather than placing the package directly at repo root) to avoid common Python packaging pitfalls and to make the distinction between source code and other repo content explicit.

Reserved namespaces are declared here so the long-term shape of the package is visible from day one, but they are not created on disk until a real implementation begins. Creating empty namespaces speculatively is forbidden; they are added when their first real content lands.

---

## Module responsibilities

### `core/`

Holds the foundational data structures and primitives that other modules depend on. The case schema lives here, along with graph manipulation utilities, file I/O for the canonical HDF5 format, custom exceptions, and any logic that is genuinely cross-cutting.

This module has no upstream dependencies within the package. Every other module may depend on `core/`; `core/` may not depend on any of them.

`core/` also stays free of the ML stack: it imports neither `torch` nor `torch_geometric`. Those are hard runtime dependencies of the package (ADR-0018) but are used only in the ML-facing layers (`datasets/`, `models/`, `eval/`). Keeping the substrate a pure `numpy`/`h5py` layer preserves the option to re-split the ML stack out later without touching `core/`.

### `benchmarks/`

Contains the definitions of benchmark problems. Each benchmark is a self-contained submodule that specifies: the problem statement, the parametric space, the data generation protocol, the train/val/test split, and references to its associated dataset. Benchmarks are resolved by name through a registry (`get_benchmark`, ADR-0024); each module ships a typed `BenchmarkCard` (ADR-0027) from which `docs/benchmarks.md` and per-archive metadata are generated.

A benchmark module describes *what* the problem is. It does not include the data itself (that lives outside the repo, in a versioned data archive) or the models that solve it (those live in `models/`).

### `models/`

Reference ML models that establish baselines on the benchmarks. This is where the data-driven approaches live — GNN surrogates, foundation models, anomaly detectors, and any other ML method shipped as part of the platform. Each model is a self-contained submodule with a defined training protocol, hyperparameter defaults, and a published checkpoint.

Models in this module are reference implementations. They are not the only models that can be evaluated on a benchmark — external contributions are evaluated through the same protocols without being added here.

ML touch points outside this module: data preparation (turning canonical-format data into model-ready tensors) lives in `datasets/`; evaluation logic lives in `eval/`. Models consume tensors and produce predictions; they do not know how to load files or compute metrics.

### `datasets/`

Data loading and dataset management. Provides the abstractions that turn an HDF5 file (or a remote dataset reference) into objects models can consume. Handles caching, version pinning, and integrity checks.

The schema for what's *inside* the data files lives in `core/`. The mechanics of loading and serving that data live here.

**ML data flow.** A canonical case (strict SI, HDF5) is loaded and converted to a `CaseTrajectory` — positions in mm plus one auxiliary target field selected by name (`aux_field`, e.g. von Mises stress for Taylor), per the owning benchmark's spec (ADR-0019, ADR-0027) — so that ported GNS hyperparameters transfer without rescaling. The trajectory is then split into overlapping windows (`WindowDataset`) and normalised via velocity/acceleration statistics (`compute_stats`). The windowed, normalised samples feed the GNS simulator (`models/gns`), whose predictions are compared to ground truth by `eval.rollout` — a full autoregressive rollout returning a `RolloutResult` with per-step and cumulative RMSE plus the benchmark's QoI values and errors, and a teacher-forced `one_step_position_rmse` that isolates single-step accuracy (ADR-0019 §5). The benchmark module is responsible for supplying the train/val/test split and for encoding the boundary-condition feature that conditions each particle's neighbourhood message.

### `eval/`

Metrics and evaluation protocols. Each benchmark declares its own evaluation metrics; this module implements them in a model-agnostic way. A leaderboard submission validator and cross-benchmark evaluation utilities are planned here (see the Roadmap section of README.md) but do not exist yet.

`eval/` depends on `core/` (for the case schema) and `datasets/` (for ground truth access). It does not depend on `models/` — evaluation is a property of the benchmark, not the model.

### `viz/`

FEM-postprocessor-style visualization of particle physics fields (ADR-0022). Any figure that shows a physics quantity — von Mises stress, plastic strain, pressure — renders through this module, following the conventions structural engineers know from LS-PrePost and Abaqus/CAE: the jet rainbow color code (blue = low, red = high), a fringe bar with evenly spaced labelled levels, physical units in the working frame. A field registry (`FIELDS`) carries each quantity's label, unit, and tick format so figures stay consistent across runs and documents.

`viz/` depends on `core/` (reading canonical cases) and `datasets/` (working-frame conversions). It does not depend on `models/`, `benchmarks/`, or `eval/` — it plots arrays, not models. Its matplotlib dependency is the optional `viz` extra, never a hard runtime dependency: importing `structbench.viz` without matplotlib succeeds, and plotting calls raise with the install instruction.

### `cli/`

Command-line entry points. Thin wrappers around functionality in the other modules. The CLI exposes the user-facing operations: running benchmarks, evaluating predictions, listing available datasets and models.

`cli/` depends on most other modules but is depended on by none. It is the outermost layer. (`viz/` additionally carries its own `__main__` so `python -m structbench.viz` can regenerate a run's standard figures without a console-script entry.)

---

## Interface discipline

Cross-module imports follow a moderate-coupling rule: each module exposes a public API through its `__init__.py`, and other modules import only from that public surface.

- Anything a module wants to expose to the rest of the package is re-exported in its `__init__.py`.
- Symbols whose names start with `_` are private and cannot be imported across module boundaries, even if Python permits it.
- Within a module, internal files may import from each other freely; the discipline applies only at module boundaries.

This is enforceable by convention, not by tooling. Violations are caught during review or, later, by linting rules that flag forbidden imports.

The rule's purpose is to make refactoring tractable: a change to a private helper inside one module should never break another module. If a refactor requires touching multiple modules' internals, the design has leaked something it shouldn't have, and the public API should be reconsidered before the refactor proceeds.

---

## Dependency graph

Allowed import directions between modules:

```
                    cli/
                     │
       ┌─────────────┼─────────────┬─────────────┐
       ▼             ▼             ▼             ▼
  benchmarks/     models/        eval/         viz/
       │             │             │             │
       └─────────────┴──────┬──────┴─────────────┘
                            ▼
                        datasets/
                            │
                            ▼
                         core/
```

Rules:

- `core/` has no upstream dependencies within the package.
- `datasets/` depends only on `core/`.
- `benchmarks/`, `models/`, `eval/`, and `viz/` may depend on `core/` and `datasets/`. They do not depend on each other — a model is not coupled to a specific benchmark, a benchmark is not coupled to a specific model, and visualization plots arrays rather than models.
- `cli/` may depend on any other module. It is the assembly point.
- Reserved namespaces (`deploy/`, `vision/`, `sensing/`) will be placed in this graph when implemented; their position is a future architectural decision.

Cycles are not permitted. If a proposed dependency would create a cycle, the design is wrong and must be reconsidered.

---

## Position of the FEM solver

StructBench treats the FEM solver as an external data source rather than as a package component. The package consumes data in a canonical format (the case schema, persisted as HDF5); how that data was originally produced — by which solver, with what input deck, on what compute resource — is upstream of the package's concerns.

Solver-related code is split across two locations:

- **`data_generation/`** at repo root holds the solver-specific scripts: input deck templates, parameter sweep configurations, Pawsey job submission scripts, and any glue code needed to orchestrate batch simulations. For v0.1, this folder contains LS-DYNA-specific content. As contributions arrive from groups using other solvers (Kratos, OpenSees, Abaqus), each solver's content lives in its own subfolder here. None of this is importable as part of `structbench`; users who only consume datasets never touch it.
- **`core/io/`** inside the package holds the readers and writers for the canonical HDF5 format, and (when needed) adapters that convert raw solver outputs into the canonical format. These adapters are the bridge: they let data produced by any solver be consumed by the rest of the package uniformly.

This separation enforces the solver-agnostic posture committed to in ADR-0004. The package depends on no solver. Contributions from other solvers integrate via output adapters in `core/io/`, not via package modifications.

A third repo-root folder follows the same non-importable-glue pattern: **`hpc/`** holds cluster job scripts for training runs (SLURM decks, environment setup — one subfolder per cluster, e.g. `hpc/dug/`). It is deliberately *not* named `deploy/`: that name is reserved for the future `src/structbench/deploy/` namespace (asset onboarding and deployment workflows), which is an entirely different concern.

---

## Case schema

The case schema is the central data structure that all modules read or write — it represents one record (a specimen under a scenario, with the resulting response) in a form that is common to data generation, surrogate training, and evaluation. The vocabulary used here is fixed in ADR-0011.

Designing the schema well is one of the highest-stakes architectural decisions in the project. A well-designed schema enables modules to compose cleanly and accommodates future scope expansion (multi-modal SHM, deployment workflows). A poorly-designed schema forces every downstream component to work around its limitations.

The schema's design is treated as its own focused exercise, separate from the rest of this document. The conceptual model and field-level structure below are settled (ADR-0011 and ADR-0012), as is the HDF5 persistence layout — group spelling, dtypes, attribute conventions (ADR-0013).

### Conceptual model

A **case** is one record — one file, one ML data example. Conceptually it has three parts:

- **Specimen** — the structure being studied: geometry, topology, materials, boundary conditions, sensor placements where present.
- **Scenario** — the loading or event applied: impact, blast, observed earthquake/wind, and so on.
- **Response** — the resulting temporal evolution: per-node, per-element, and global state under that scenario, plus any sensor readings.

`case = specimen + scenario + response` is **vocabulary, not layout**. The schema's actual groups follow the data's natural structure, not the conceptual split — solver inputs interleave specimen and scenario, and forcing them apart in storage would be an artificial cut. The vocabulary exists to keep documentation and discussion consistent.

Inside `response`, the temporal axis uses two further terms:

- **Frame** — a single time slice of the response (one image in the "video" of state evolution).
- **Transition** — a pair of consecutive frames `(frame_t, frame_{t+1})`, the natural unit for auto-regressive ML training.

The word **asset** is reserved for the physical-structure / deployment meaning (see `deploy/`). A case that came from real-world observation may carry an `asset_id` field linking it to the physical structure it was observed on; many such cases on the same asset link via that field.

### Field-level structure

A case file contains the following kinds of data. The specimen / scenario / response triple is documentation only; the file does not separate them structurally. ADR-0012 carries the full rationale and validity rules.

**Geometry and topology**
- `nodes` — `coords` and `node_id`.
- `elements/<type>` — for each element type present (`sph`, `solid`, `beam`, `shell`, `discrete`, …): `connectivity` (0-indexed into `nodes`), `element_id`, `part_id`.
- `parts` — links elements to a section and a material.
- `sections` — cross-section, shell-thickness, or SPH parameters.

**Materials**
- `materials` — hybrid representation: `canonical_model` from a small enum (populated when a clean mapping exists) plus `source_model` and `source_params` verbatim from the solver.

**Constraints, loading, IC**
- `boundary_conditions` — constraints.
- `loading` — applied loads, body forces, contacts, rigid walls.
- `initial_conditions` — optional; preserved IC spec from the source. The actual t=0 state is at `response/` frame 0.
- `time_curves` — named `(t, value)` curves.
- `sets` — named node / element sets.

**Response**
- `response/time/t` — single time array, one global time axis.
- `response/node` — `displacement`, `velocity`, `acceleration` of shape `(n_frames, n_nodes, dim)`.
- `response/element/<type>` — `stress`, `strain`, `damage`, …; tensor fields in Voigt-symmetric components.
- `response/global` — per-frame scalars (energies, contact force, reactions).
- `response/sensor` — slot reserved (SHM scope).

**Sensors**
- `sensors` — slot reserved (SHM scope).

**Metadata**
- `case_id`, `schema_version`, `units_convention` (= `"SI"`), `dimension` (`2` or `3`).
- `provenance` — solver name / version / generation date when solver-generated.
- `source_units` — original convention when not natively SI.
- `source_deck` — verbatim source blob, optional.
- `asset_id`, `dataset_id` — workflow-level identifiers, optional.

**Cross-cutting conventions**

- *Units*: strict SI canonical; adapters convert on write.
- *Identity*: 0-indexed sequential connectivity; original solver IDs preserved in `*_id` columns.
- *Time*: single `response/time/t` array; one global axis.
- *t=0 state*: frame 0 of `response/`, not duplicated in a separate group.
- *Tensors*: Voigt-symmetric (6 components in 3D, 4 in 2D).

**Validity tiers**

| Tier | Required content |
|---|---|
| Always required | `nodes`, at least one `elements/<type>`, `materials`, `metadata` (`case_id`, `schema_version`, `units_convention`, `dimension`); `provenance` when solver-generated |
| Required when applicable | `parts`, `sections`, `boundary_conditions`, `loading`, `time_curves`, `sets`, `initial_conditions`; `response` (with `time/t` and `node/displacement`) when the case has been simulated |
| Always optional | `sensors`, `response/sensor`, `response/global`, `response/element/*`, `metadata/source_deck`, `metadata/asset_id`, `metadata/dataset_id` |

A case file with no `response` group is a valid "stub" — specimen + scenario specified, simulation not yet run.

### HDF5 layout

The field set above is persisted as a single HDF5 file per case (ADR-0013), read and written through `h5py`. Salient points:

- **Paths** are lowercase `snake_case` matching the field names: `/metadata`, `/nodes`, `/elements/<type>`, `/parts`, `/sections`, `/materials`, `/boundary_conditions`, `/loading`, `/initial_conditions`, `/time_curves`, `/sets/{node,element}/<id>`, `/response/{time,node,element,global,sensor}`, `/sensors`.
- **Attributes vs datasets**: small scalars (the `/metadata` fields, `provenance`) are HDF5 attributes; arrays and anything possibly large (notably `/metadata/source_deck`) are datasets.
- **Dtypes**: float64 for geometry and the time axis, float32 for bulk response fields, int64 for ids and connectivity, variable-length UTF-8 for strings.
- **Compression**: response arrays and the source deck are gzip-compressed (level 4) and chunked along the frame axis, so transitions can be streamed without loading whole arrays.
- **Heterogeneous solver-native data**: `materials`/`sections` `source_params` and `metadata/source_deck` are stored as JSON strings; solver sub-models (EOS, hourglass) nest inside the owning material's `source_params`.
- **Version**: the initial `schema_version` is `"0.1.0"`; additive changes bump the minor version, structural changes the major version (with a superseding ADR).

ADR-0013 carries the full layout, dtype, and convention rationale.
