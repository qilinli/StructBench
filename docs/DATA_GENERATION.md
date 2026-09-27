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

## What the preparation stage establishes

Before a production sweep, the preflight (part two of ADR-0071) probes the
pilot cases for three resolutions, and the data card declares each with the
number behind it:

- **space** — the mesh: nested levels, the quantities of interest extrapolated
  across them, the field metrics; the production level is chosen against them;
- **time** — the solver increment (the energy account must hold at more than
  one stable-increment scale) and the stored frame interval (a finer export
  must agree with the stored one at the common instants);
- **duration** — the stored horizon against the response's settling time: when
  the quantities of interest stop changing, how much of the horizon lies after
  that, and which part of the trajectory is scored.

Until the stage exists, do these three by hand on the pilot split and record
them in the data card; the questions do not change when the tooling arrives.
