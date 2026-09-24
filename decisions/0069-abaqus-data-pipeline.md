# 0069 — The Abaqus data-generation pipeline

**Status**: Proposed
**Type**: Durable
**Date**: 2026-09-24

## Context

The README Roadmap names agent-driven data generation with Abaqus as the first
solver. ADR-0068 admitted Abaqus and its two text readers but left open the
intermediate format, the adapter and every generation step. The design is
`docs/plans/2026-09-24-abaqus-data-pipeline-design.md`.

## Decision

1. **Four stages, each an idempotent command**: generator, runner, extractor,
   validator. State lives in the case folder, and each stage does only the
   missing work. Claude Code operates the stages; compute and deletions wait
   for the maintainer.
2. **Shared code lives in `data_generation/abaqus/`, and a dataset is a
   `sweep.toml` plus a `model.py`**, passed by path. A dataset may live in a
   private repository until admission (CORRECTIONS 2026-08-12); on admission
   it moves to `data_generation/abaqus/<name>/`. Provenance records the
   commits of both repositories.
3. **The intermediate is `abaqus-npz/1`.** This settles ADR-0068's open format
   question. Measured 2026-09-24: the Abaqus 2025 interpreter is Python 3.10.5
   with numpy 1.22.4 and scipy 1.11.1, and has no h5py, so an HDF5
   intermediate would need a package the interpreter lacks.
4. **scipy enters as the optional extra `datagen`**, for scrambled Sobol
   sampling. `structbench` never imports it; this follows the `data` extra of
   ADR-0058.
5. **The package side is `core/io/abaqus.py`**, which turns the npz plus the
   deck into a `Case` and mirrors `lsdyna_to_case`.
   **`datacheck measure --declaration <file>`** lets a sweep that is not a
   registered benchmark be measured. Both are built in plan 2.
6. **Runs execute in a local work root** and are archived afterwards to the
   ADR-0031 tree (`raw/<name>/abaqus/<case_id>/` and `canonical/<name>/`). A
   solver never writes inside the OneDrive tree.
7. **Validator verdicts**: definitional rows get pass/fail. Energy and other
   solution-verification indicators are measured only, unchanged from the
   2026-09-21 maintainer decision.

## Alternatives considered

- **One driver that runs all stages per case.** Rejected: a failure mid-case
  forces a full redo, or the driver grows back the skip logic that separate
  stages give for free.
- **A workflow engine (Snakemake, doit).** Rejected: it adds a dependency and a
  DSL for only four stages.
- **An HDF5 intermediate.** Rejected by measurement (clause 3).
- **scipy as inline PEP 723 script metadata.** Rejected: it splits the
  environment, so the sampler tests could not run in the project env.
- **Dataset definitions in the public repository from the start.** Rejected:
  they are study designs until publication.

## Consequences

- A new Abaqus dataset needs only a `sweep.toml` and a `model.py`.
- `odb_export.py` must stay parseable as Python 3.10, and a test pins that.
- ADR-0068's open question on the intermediate's format is settled.
- Not decided here: the material class for isotropic tabulated plasticity,
  which gets its own ADR in plan 2; and any schema field for axisymmetry,
  which is added only if a consumer needs one.

## Note (2026-09-24): a declared feasibility limit

The first dataset's pilots found a region of its parameter box that a pure
Lagrangian mesh cannot follow: runs abort on excessive element distortion,
which is physical, not an hourglass artifact. Rather than let those cases fail
after the fact, which would drop points non-randomly from one corner, or
reshape the box, a dataset's `model.py` may define `feasible(params)`. The
generator applies it to Sobol-sampled points only, skipping infeasible ones
the way a region `exclude` does, so prefixes stay nested. Explicit points
bypass it. The limit is a property of the solver setup. It is measured on
pilot and probe runs, declared before production, and recorded with the
dataset. It is never adjusted afterwards to admit or drop production cases.
