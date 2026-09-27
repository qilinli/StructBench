# Data generation with StructBench

*A stub. The full guide arrives with parts two and three of ADR-0071.*

StructBench generates reference data through `structbench-datagen`. A dataset is
a definition in a directory of its own, which may be private:

- `dataset.toml` — what the dataset is: identity, declaration, constants, the
  sampled box, splits, mesh levels, the pilot cases, quantities of interest,
  ODB retention.
- `problem.py` — how one case is built and read: `input_deck`, `feasible`,
  `mesh`, `qoi`.

Start from the template and check it before anything runs:

    structbench-datagen new my_dataset
    structbench-datagen check my_dataset

Then the stages, in order: `generate`, `run`, `export`, `convert`, `validate`,
`archive`. The shipped example (`structbench/datagen/examples/abaqus_conformance`)
is the single-rod conformance case described in `docs/datagen/abaqus-conformance.md`.
The design is `docs/plans/2026-09-27-abaqus-datagen-platform-design.md`.
