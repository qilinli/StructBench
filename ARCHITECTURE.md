# ARCHITECTURE.md

*The technical spine of StructBench: package layout, module responsibilities, and the asset model schema. Decisions in this document are durable unless explicitly marked otherwise.*

---

## Purpose

This document describes how StructBench is structured at the code level: what modules exist, what each is responsible for, how they relate to each other, and what data structures they share. Architectural changes are recorded by editing this document and creating a corresponding ADR.

For coding conventions (style, testing, documentation expectations), see PRINCIPLES.md. For the philosophy behind these structural choices, see HARNESS.md.

---

## Package layout

```
src/structbench/
├── core/          # asset model, graph utilities, I/O primitives
├── benchmarks/    # benchmark problem definitions
├── models/        # reference ML models
├── datasets/      # data loaders and dataset registry
├── eval/          # metrics and evaluation protocols
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

Holds the foundational data structures and primitives that other modules depend on. The asset model schema lives here, along with graph manipulation utilities, file I/O for the canonical HDF5 format, custom exceptions, and any logic that is genuinely cross-cutting.

This module has no upstream dependencies within the package. Every other module may depend on `core/`; `core/` may not depend on any of them.

### `benchmarks/`

Contains the definitions of benchmark problems. Each benchmark is a self-contained submodule that specifies: the problem statement, the parametric space, the data generation protocol, the train/val/test split, and references to its associated dataset.

A benchmark module describes *what* the problem is. It does not include the data itself (that lives outside the repo, in a versioned data archive) or the models that solve it (those live in `models/`).

### `models/`

Reference ML models that establish baselines on the benchmarks. This is where the data-driven approaches live — GNN surrogates, foundation models, anomaly detectors, and any other ML method shipped as part of the platform. Each model is a self-contained submodule with a defined training protocol, hyperparameter defaults, and a published checkpoint.

Models in this module are reference implementations. They are not the only models that can be evaluated on a benchmark — external contributions are evaluated through the same protocols without being added here.

ML touch points outside this module: data preparation (turning canonical-format data into model-ready tensors) lives in `datasets/`; evaluation logic lives in `eval/`. Models consume tensors and produce predictions; they do not know how to load files or compute metrics.

### `datasets/`

Data loading and dataset management. Provides the abstractions that turn an HDF5 file (or a remote dataset reference) into objects models can consume. Handles caching, version pinning, and integrity checks.

The schema for what's *inside* the data files lives in `core/`. The mechanics of loading and serving that data live here.

### `eval/`

Metrics and evaluation protocols. Each benchmark declares its own evaluation metrics; this module implements them in a model-agnostic way. Also contains the leaderboard submission validator and any cross-benchmark evaluation utilities.

`eval/` depends on `core/` (for the asset model) and `datasets/` (for ground truth access). It does not depend on `models/` — evaluation is a property of the benchmark, not the model.

### `cli/`

Command-line entry points. Thin wrappers around functionality in the other modules. The CLI exposes the user-facing operations: running benchmarks, evaluating predictions, listing available datasets and models.

`cli/` depends on most other modules but is depended on by none. It is the outermost layer.

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
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
  benchmarks/     models/        eval/
       │             │             │
       └─────────────┼─────────────┘
                     ▼
                 datasets/
                     │
                     ▼
                  core/
```

Rules:

- `core/` has no upstream dependencies within the package.
- `datasets/` depends only on `core/`.
- `benchmarks/`, `models/`, and `eval/` may depend on `core/` and `datasets/`. They do not depend on each other — a model is not coupled to a specific benchmark, and a benchmark is not coupled to a specific model.
- `cli/` may depend on any other module. It is the assembly point.
- Reserved namespaces (`deploy/`, `vision/`, `sensing/`) will be placed in this graph when implemented; their position is a future architectural decision.

Cycles are not permitted. If a proposed dependency would create a cycle, the design is wrong and must be reconsidered.

---

## Position of the FEM solver

StructBench treats the FEM solver as an external data source rather than as a package component. The package consumes data in a canonical format (the asset model schema, persisted as HDF5); how that data was originally produced — by which solver, with what input deck, on what compute resource — is upstream of the package's concerns.

Solver-related code therefore sits in two places, neither of which is inside the importable package proper:

- **`data_generation/`** at repo root holds the solver-specific scripts: input deck templates, parameter sweep configurations, Pawsey job submission scripts, and any glue code needed to orchestrate batch simulations. For v0.1, this folder contains LS-DYNA-specific content. As contributions arrive from groups using other solvers (Kratos, OpenSees, Abaqus), each solver's content lives in its own subfolder here. None of this is importable as part of `structbench`; users who only consume datasets never touch it.
- **`core/io/`** inside the package holds the readers and writers for the canonical HDF5 format, and (when needed) adapters that convert raw solver outputs into the canonical format. These adapters are the bridge: they let data produced by any solver be consumed by the rest of the package uniformly.

This separation enforces the solver-agnostic posture committed to in ADR-0004. The package depends on no solver. Contributions from other solvers integrate via output adapters in `core/io/`, not via package modifications.

---

## Asset model schema

The asset model schema is the central data structure that all modules read or write — it represents a structure (geometry, topology, materials, sensors, state) in a form that is common to data generation, surrogate training, and evaluation.

Designing the schema well is one of the highest-stakes architectural decisions in the project. A well-designed schema enables modules to compose cleanly and accommodates future scope expansion (multi-modal SHM, deployment workflows). A poorly-designed schema forces every downstream component to work around its limitations.

The schema's design is treated as its own focused exercise, separate from the rest of this document. When the schema is settled, its specification — including the conceptual model, field-level definitions, type requirements, and HDF5 layout — will be added below this section.

*(Schema specification: to be added.)*
