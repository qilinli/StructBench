# Data generation with StructBench

*A stub. The full guide arrives with parts two and three of ADR-0071.*

StructBench generates reference data through `structbench-datagen`. A dataset is
a definition in a directory of its own, which may be private:

- `dataset.toml` — what the dataset is: identity, declaration, constants, the
  sampled box, splits, mesh levels (and their `symmetry`, `axisymmetric` or
  `planar`, which chooses the volume weights `converge` uses), the pilot
  cases, quantities of interest, ODB retention.
- `problem.py` — how one case is built and read: `input_deck`, `feasible`,
  `mesh`, `qoi`. It may import sibling modules of its own directory.

Start from the template and check it before anything runs:

    structbench-datagen new my_dataset
    structbench-datagen check my_dataset

Then the stages, in order: `generate`, `run`, `export`, `convert`, `verify`
(the ADR-0066 instrument over the sweep), `converge` (below), `archive`. The shipped example (`structbench/datagen/examples/abaqus_conformance`)
is the single-rod conformance case described in `docs/datagen/abaqus-conformance.md`.
The design is `docs/plans/2026-09-27-abaqus-datagen-platform-design.md`.

`verify` asks whether each run is internally sound (the ADR-0066 instrument).
Whether the *setup* reproduces a physical experiment is a different question,
answered by `structbench-validate` (ADR-0072): it measures canonical cases and
a shipped reference-experiment set with the same functions and writes a record
of the deviations — no threshold, no verdict. `structbench-validate --list`
names the sets; `src/structbench/validation/references/taylor_copper.md` is
the first one's provenance.

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

## Convergence across mesh levels

`structbench-datagen converge --dataset <dir> --sweep <runs>/<name> [--root <other run root> ...]`
pairs every production run with the probe runs at the other `[levels]` of the
same parameters (across run roots, so a production that moved to a finer mesh
keeps its older runs), extrapolates each quantity of interest from three
levels in constant ratio (Richardson, with the observed order and a status
that says when there is none), and measures every stored field of each
coarser level against the finest with the headline relative L2, restricted
with the `[levels].symmetry` volume weights. It writes `convergence.json`
(byte-stable, case ids only) and `convergence.md` under `<sweep>/converge/`.
The dataset supplies only `[levels]` and `qoi()`; the engine is
`structbench.verification.convergence`.

## Before production

[`datagen/abaqus-lessons.md`](datagen/abaqus-lessons.md) is what the first
Abaqus dataset taught, written as rules with the reason behind each and what
the pipeline now does about it, and it ends in a pre-production checklist.
Read it before the first run of a new model.
