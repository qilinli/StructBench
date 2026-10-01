# 0010 — FEM solver code lives outside the importable package

**Status**: Accepted
**Type**: Durable
**Date**: 2026-04-24

## Context

The platform requires FEM simulations to produce ground truth data for benchmarks. ADR-0004 commits the platform to being solver-agnostic — users consume datasets and reference models without needing any specific solver installed. This raises the structural question of where solver-related code lives in the repository.

Three options were considered: (a) inside the package as a first-class module like `simulation/`, (b) outside the package entirely as data-generation scripts at repo root, (c) a hybrid in which the package contains thin solver-output adapters while solver-specific scripts live outside.

## Decision

Solver-related code is split across two locations:

- **`data_generation/`** at repo root holds solver-specific scripts: input deck templates, parameter sweep configurations, Pawsey job submission scripts, and any glue code needed to orchestrate batch simulations. Each solver gets its own subfolder as contributions arrive (e.g., `data_generation/lsdyna/`, `data_generation/kratos/`). This folder is not part of `pip install structbench`; users who consume only datasets never touch it.
- **`core/io/`** inside the package holds the canonical HDF5 format readers/writers and solver-output adapters that convert raw solver outputs into the canonical format. These adapters are the interoperability bridge between any solver and the rest of the package.

For v0.1, `data_generation/` contains LS-DYNA-specific content. The package itself contains no solver code.

## Alternatives considered

- **Solver code as a first-class package module** (e.g., `src/structbench/simulation/`): rejected. Bundling solver code into the importable package would force every user to install solver-related dependencies they don't need, and would create awkward per-solver conditional imports as the platform expands beyond LS-DYNA. This conflicts with the solver-agnostic posture in ADR-0004.
- **Everything outside the package, including format adapters**: rejected. The canonical-format readers/writers and the solver-output adapters are genuinely package functionality — they implement the data abstraction that all other modules depend on. Putting them outside the package would force `datasets/` to do format-conversion work, breaking the clean module boundaries in ARCHITECTURE.md.
- **A thin solver-abstraction layer inside the package** (with concrete solvers as plugins): rejected as premature. A multi-solver abstraction needs at least two concrete solvers to be designed correctly; designing it now for v0.1 with only LS-DYNA in scope would over-fit to one solver's idiosyncrasies. The hybrid above is the simpler structural answer that defers the abstraction question until it has evidence to be designed against.

## Consequences

- The package can be installed and used without any solver dependency, matching the user-facing meaning of "open" in ADR-0004.
- Users who want to generate new data run scripts from `data_generation/` directly. These scripts are not pip-installed code; they are run from a checkout of the repository.
- Adding support for a new solver means adding a new subfolder under `data_generation/` and (if needed) a new output adapter in `core/io/`. It does not require changes to `benchmarks/`, `models/`, `eval/`, or other consumer modules.
- The repository's top-level structure now explicitly includes `data_generation/` as a non-package directory alongside `src/`, `tests/`, `docs/`, etc.
- Pawsey orchestration and other compute-cluster glue is excluded from the package, which keeps the package's dependencies minimal and its concerns focused.

---

**Amendment (2026-10-01, maintainer; ADR-0073 D15).** The package may hold the
solver-side code of its own data-generation pipeline under
`structbench.datagen.<solver>/` (today the Abaqus deck writers and the ODB
exporter, ADR-0071), so long as `import structbench` and every consumer path
need no solver installed; the exporter runs under the solver's own Python and
imports nothing from the package. "The package itself contains no solver code"
above narrows to: the package *depends on* no solver. `data_generation/` keeps
the per-dataset glue of the datasets that predate the pipeline.

## Amendment (2026-10-01): the folder is `tools/ingest/`

`data_generation/` at the repo root is renamed `tools/ingest/`, its
`<solver>/<dataset>/` layout unchanged (maintainer, 2026-10-01). The name
this ADR gave the folder described the role it was to have -- decks,
sweeps, job submission -- and that role is the package's `datagen`
pipeline since ADR-0071; what the folder actually holds is the glue that
turned four datasets' *existing* solver output into canonical cases (the
three LS-DYNA sweeps and the MeshGraphNets `deforming_plate` download),
which is ingestion in ADR-0016's and ADR-0042's sense. Two folders named
data generation, one of which generates nothing, were confusing. Like
`tools/hpc/`, these are scripts that are run but not shipped. The posture
of this ADR stands: the package contains no solver code, the glue is not
importable, and a dataset the pipeline produces never gains a folder
here. ADR bodies that spell the old path are left as written.

*(Drafted by the agent on the maintainer's instruction; the maintainer
finalises.)*
