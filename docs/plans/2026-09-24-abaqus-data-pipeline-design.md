# Design: the Abaqus data-generation pipeline

**Date**: 2026-09-24
**Status**: Draft for maintainer review (decisions below taken in-session 2026-09-24)
**Scope**: new shared scripts under `data_generation/abaqus/`; new
`core/io/abaqus.py`; extensions to `core/io/abaqus_run.py`, `verification/`
and `cli/datacheck.py`; `data_generation/abaqus/STANDARD_INPUT_BLOCK.md`;
`data_generation/README.md`; one optional-dependency extra. Builds on ADR-0068
(Proposed), which is normative wherever the two overlap.

---

## Problem

The README Roadmap names agent-driven data generation, with Abaqus as the
first solver: sweep, deck generation, submission, run evidence to the ADR-0066
requirement, and conversion to the canonical schema, so that data generation is
something the platform can audit and a contributor can repeat. ADR-0068 admitted
Abaqus and built its two text readers, but only against one Abaqus/Standard job
with `C3D` elements. The rest does not exist: no deck generation, no runner, no
`.odb` extractor, no adapter to the canonical schema, and the readers refuse
2D element codes and have never seen an Abaqus/Explicit run.

## Decisions (maintainer, in-session 2026-09-24)

1. **Four stages — generator, runner, extractor, validator.** Each stage is a
   versioned, idempotent command. Claude Code operates them: it runs a stage,
   reads its summary, and proposes the next step. Anything with real compute
   cost, and every deletion, waits for the maintainer (CLAUDE.md, flag-first).
2. **Shared code vs dataset code.** Everything solver-generic lives once in
   `data_generation/abaqus/`. A dataset contributes exactly two files: a
   `sweep.toml` (what to sample) and a `model.py` (what to build). Shared code
   gains only features some dataset exercises (CORRECTIONS 2026-09-21).
3. **Dataset definitions may live outside the repository** and are passed by
   path (`--dataset <dir>`). They stay private until the dataset is admitted as
   a benchmark (CORRECTIONS 2026-08-12), and on admission move to
   `data_generation/abaqus/<name>/`. The private location is itself a git
   repository, because provenance records its commit.
4. **First dataset: a 2D axisymmetric explicit impact sweep** (one deformable
   part, one elastic–plastic material, an analytical rigid wall). Later
   datasets extend the shared code with what they need (for example contact
   between parts, rigid-body output, further material models); nothing for
   them is built now.
5. **Validator verdicts.** Rows that are right or wrong by definition get
   pass/fail: termination, solver errors, finite fields, the sampling clock,
   the initial state against the input, the state variable's monotonicity,
   and the yield bound. Energy and other solution-verification indicators are
   **measured only**, with no threshold. This is the maintainer's 2026-09-21
   decision applied unchanged.
6. **Where runs live.** Jobs execute in a local work root (default
   `C:\structbench-runs\<name>\` on the maintainer's machine) and never inside
   the OneDrive tree while the solver is writing. A finished case is then
   archived to the data tree of ADR-0031: `<data-root>/raw/<name>/abaqus/<case_id>/`
   for the run folder and `<data-root>/canonical/<name>/` for the `.h5`. Both
   roots are arguments, not constants.
7. **The intermediate format is `.npz`.** This answers ADR-0068's open format
   question by measurement: the Abaqus 2025 interpreter is Python 3.10.5 with
   numpy 1.22.4 and scipy 1.11.1, and **no h5py**.
8. **scipy enters as an optional extra, `datagen`**, for scrambled Sobol
   sampling (`scipy.stats.qmc.Sobol`). It is never imported by the package,
   following the `data` extra's precedent (ADR-0058). This is a flag-first
   dependency and needs its ADR.

## Layout

```
data_generation/abaqus/
  sampling.py      Sobol over named ranges; regions; per-split seeds; variants
  deck.py          keyword writers; structured quad mesh; the standard output block
  generate.py      CLI: sweep.toml + model.py -> case folders + provenance
  run_jobs.py      CLI: run decks, N at a time; run.json + run_log.csv
  odb_export.py    CLI under `abaqus python`: .odb -> <case_id>.npz (abaqus-npz/1)
  convert.py       CLI: .npz + deck -> canonical .h5 via core/io/abaqus.py
  validate.py      CLI: run evidence -> datacheck measure/judge -> record + report
  archive.py       CLI: move finished cases to the data tree; prune .odb files
<dataset dir>/     (private until admission)
  sweep.toml
  model.py         build(params, variant) -> deck text
```

**State lives in the case folder.** No stage keeps its own database, so any
stage can be re-run after a crash, a reboot or a stop:

| File present in `<work-root>/<name>/<case_id>/` | Stage done |
|---|---|
| `<case_id>.inp`, `provenance.json` | generated |
| `run.json` (plus Abaqus's `.sta`, `.msg`, `.dat`, `.odb`) | run |
| `<case_id>.npz` | exported |
| `<work-root>/<name>/canonical/<case_id>.h5` | converted |
| `<work-root>/<name>/datacheck/` record covering the case | validated |

## Stage 1 — Generator

`sweep.toml` holds the dataset name, the unit system (`t-mm-s`), the frame
interval and count, fixed constants, variables as `[low, high]` ranges, named
regions (boxes over variables), and splits. A split sets `n`, `seed`, and
`exclude` or `within` regions. When a dataset needs them, it can also set
`extra` (more sampled variables), `categorical` (values cycled by index) and
`variants` (each sampled point expands to one case per variant). Every value
is in the deck's unit system. SI conversion happens exactly once, in the
extractor. The file also carries a `[declaration]` block for the validator
(Stage 4).

`sampling.py` draws 1024 scrambled Sobol points per split with that split's
seed and takes the first `n` in order, so the first *k* cases of a split are
nested subsets. An `exclude` split filters the draw out of the named regions.
A `within` split draws **inside** the region's box (region bounds for the
variables the region names, full ranges for the rest), because a small box
filtered from a global draw would leave too few points. The sampler raises an
error if fewer than `n` points remain.

`model.py` returns deck text built from `deck.py`: node and element blocks for
a structured quad mesh, material, section, boundary and initial conditions,
the step, and the standard output block (field output on a fixed time interval
with time marks; every energy history term on the same interval). **Every
keyword choice is a hypothesis until the conformance run (Stage 4) confirms
it**, and each is checked against the Keywords Reference in the local
installation's documentation. Nothing is recommended from memory
(`STANDARD_INPUT_BLOCK.md`).

Output per case: `<case_id>.inp`, plus `provenance.json` recording the
parameters, split, Sobol index, seed, variant, the `sweep.toml` hash, the
`.inp` SHA-256, and the commits (with dirty flags) of both the repository and
the dataset directory. The generator also writes a sweep-level
`manifest.csv`. Re-running it produces byte-identical decks and skips existing
cases. If a case's `.inp` would change, it refuses unless given `--force`.

## Stage 2 — Runner

`run_jobs.py --sweep <dir> [--split|--cases|--limit] [--workers 6]
[--timeout s] [--abaqus <exe>] [--dry-run]`. Standard library only; nothing
dataset-specific. It runs `abaqus job=<id> input=<id>.inp double=both cpus=1
interactive` in each case folder. When the process exits, it writes
`run.json` (command, Abaqus version token, UTC start and end, wall time, return
code, status) and appends a row to `run_log.csv`.

- **Status has one source**: `read_abaqus_run_evidence`, which the validator
  also uses. The runner adds only `timeout` and `launch_error`. A completion
  message the reader does not recognise reads `none`, never success.
- **Done** means `run.json` exists. **Interrupted** means a `.lck` file with
  no `run.json` and no runner holding the sweep: its files move to
  `attempts/<n>/` and the case is rerun. **Failed** cases are rerun only with
  `--retry-failed`, which moves the old attempt aside the same way. Nothing is
  deleted.
- A `.runner.lock` holding the runner's process ID allows one runner per sweep.
  A timed-out job is stopped with `abaqus terminate`, and its process tree is
  killed if that fails.
- It prints one line per finished job and a closing summary (counts by status,
  slowest cases, failures with their reasons). It exits non-zero on any
  failure.

## Stage 3 — Extractor

**`odb_export.py`** runs under `abaqus python` and imports only the standard
library, numpy and `odbAccess`. It processes cases whose `run.json` reports
normal completion and that have no `.npz` yet, so it can run alongside the
runner. It dumps whatever the ODB holds, with no dataset knowledge:
- node labels and reference coordinates;
- element labels, types and connectivity;
- part, instance, material and section **names** (ids are minted by
  `mint_ids`, never by the exporter);
- every frame's time;
- every field output as `(T, n, k)` with its labels, component labels and
  position, read through `bulkDataBlocks`;
- every history output as time–value pairs.

The layout is flat keys (`field/S/data`, `history/<region>/<var>`, …) plus a
manifest: format version `abaqus-npz/1`, the ODB's SHA-256, the Abaqus
release, and the precision. The layout is documented in the module docstring
and in `STANDARD_INPUT_BLOCK.md` (ADR-0068 clause 6).

**`core/io/abaqus.py`** (new public API) reads and validates an `abaqus-npz/1`
file, failing closed on an unknown version. `abaqus_export_to_case(npz_path,
deck_path, *, source_units, dimension, case_id=None, dataset_id=None) -> Case`
mirrors `lsdyna_to_case` and reuses its unit table and field names:

| Abaqus output | Canonical field |
|---|---|
| U, V, A | `response/node/{displacement, velocity, acceleration}` |
| S (S11, S22, S33, S12) | `response/element/solid/stress`, six-component Voigt `(xx, yy, zz, xy, yz, zx)`. For plane and axisymmetric elements `yz` and `zx` are stored as exact zeros, which is the element's own result |
| PEEQ | `response/element/solid/effective_plastic_strain` |
| ALLKE, ALLIE, ETOTAL | `response/global/{kinetic_energy, internal_energy, total_energy}`, copied as the solver wrote them |
| ALLAE, ALLPD, ALLVD, ALLWK | `response/global/{hourglass_energy, plastic_dissipation, viscous_dissipation, external_work}` |

History terms enter `response/global` only if their time points equal the
frame times. Otherwise the adapter refuses rather than interpolate.
Provenance is `solver_name="Abaqus"` with the release as version. The `.inp`
is stored verbatim as `source_deck`, with `source_units` set. Axisymmetry has
no schema field: it is recoverable from the element code in the stored deck,
and a schema field is proposed only if a consumer needs one.

**`convert.py`** converts each case that has an `.npz` and no `.h5`, writing
it with the existing HDF5 writer.

## Stage 4 — Validator

Existing catalogue rows cover almost all of Decision 5. Most of the work is
making them apply to Abaqus/Explicit:

| Check | Row(s) | Verdict | Work |
|---|---|---|---|
| Normal completion, end time reached | `terminated_normally`, `reached_end_time` | pass/fail | reader: Explicit `.sta`/`.msg` wording |
| No solver errors | `solver_error_count` | pass/fail | reader: Explicit `.msg` |
| Finite fields | `nonfinite_count` | pass/fail | none |
| Frames on the declared interval | `sampling_clock_consistent`, `time_axis_monotone` | pass/fail | build `sampling_clock_consistent` (open in ADR-0066) |
| Initial state matches the input | `initial_state_matches_input` | pass/fail | **new row**; reader parses `*INITIAL CONDITIONS` (velocity, hardening) |
| State variable never decreases | `state_variable_min`, `state_variable_decrease_max` | pass/fail | new material class |
| Yield bound | `yield_ratio_max` | pass/fail | new material class; yield table read per case from the input's `*PLASTIC` |
| Hourglass, energy balance, time-step drop | `zero_energy_mode_*`, `total_energy_change_final`, `energy_residual_final`, `timestep_min_ratio` | measured only | ledger from the `.npz` history; time-step series from the Explicit `.sta` |
| Late plastic-dissipation growth | `plastic_dissipation_late_growth` | measured only | **new row** |

Other changes:
- **Readers** (`core/io/abaqus_run.py`): element codes `CAX*` and `CPE*`; the
  `*DYNAMIC, EXPLICIT` period as end time; the `*PLASTIC` table; initial
  conditions; output requests.
- **Material class** `elastic_plastic_isotropic`: Mises yield, a tabulated
  σ_y(ε̄ᵖ), monotone plastic strain, no equation of state. Its own short ADR,
  following ADR-0067.
- **`datacheck measure --declaration <sweep.toml>`**, an alternative to
  `--benchmark` that builds `DeclaredFacts` from the `[declaration]` block, so
  a sweep that is not a registered benchmark can be measured. This is a CLI
  change.
- **Conformance.** The first smoke runs are ADR-0068's conformance run. What
  they establish (energy output, `PRESELECT` contents, integration-point
  output, the Explicit `.sta` series, element-code semantics) is written into
  `STANDARD_INPUT_BLOCK.md` with the file it came from. The
  `input_requests_required_evidence` row then checks decks against it.

**`validate.py`** collects run evidence (from the logs, and the ledger from the
`.npz`), runs `datacheck measure --declaration` and `judge`, and writes the
record and report into `<work-root>/<name>/datacheck/`, **not**
`docs/datachecks/`, because the dataset is private until admission. It prints
failing rows by case and the spread of each measured-only row, and exits
non-zero on any `fail`.

## Housekeeping — archive and prune

`archive.py` moves validated cases to the data tree (Decision 6). It writes
only to known paths and never scans OneDrive (CORRECTIONS 2026-06-29). ODB
retention is declared in `sweep.toml` (`[retention]`: a seeded random
fraction plus named cases). `--prune-odb` deletes the other `.odb` files only
after a dry-run listing and an explicit confirmation flag.

## Testing

No test needs Abaqus, network access or large files (PRINCIPLES.md). Scripts
are loaded by path from `tests/tools/`, as the LS-DYNA glue tests are, using
test file names that are unique across the suite.

- **Sampler**: determinism for a given seed, region filtering, nested
  prefixes, and the error when a region leaves too few points.
- **Deck writers and reader, against each other**: decks written by `deck.py`
  are parsed back by `read_abaqus_input_facts`, which must recover element
  codes, the material table, initial conditions, end time and output requests.
  One small golden deck is kept as a text snapshot.
- **Runner**: a fake `abaqus` executable (a Python script that writes canned
  `.sta`/`.msg` text, sleeps, fails or hangs), passed via `--abaqus`, tests
  concurrency, skip-if-done, interruption, `attempts/`, timeout and the
  sweep lock.
- **Adapter**: a synthetic `abaqus-npz/1` file converted to a `Case`,
  covering units, the 4-to-6 Voigt mapping, clock-mismatch refusal and
  unknown-version refusal.
- **Readers and rows**: Explicit text fixtures copied from the first smoke run
  with the licence line replaced by invented text; synthetic cases for the two
  new rows.
- **Exporter**: its logic stays thin. One env-gated real-run test
  (`STRUCTBENCH_ABAQUS_RUN_DIR`) skips when the variable is unset, like the
  existing real-data tests.

## Paperwork

- ADR: the Abaqus data-generation pipeline. It covers the `datagen` extra, the
  `abaqus-npz/1` format, `core/io/abaqus.py`, and `datacheck --declaration`,
  and amends ADR-0068's "not decided" list.
- ADR: the `elastic_plastic_isotropic` material class.
- ADR-0066 dated note: `initial_state_matches_input` and
  `plastic_dissipation_late_growth`.
- `data_generation/README.md`: the shared-code pattern for Abaqus.

## Out of scope

Features only a later dataset needs; benchmark admission (cards, frozen splits, landing pages, published records,
Hugging Face); mesh-convergence analysis (convergence sweeps are ordinary
sweeps through this pipeline, and their analysis comes later); a schema field
for axisymmetry; any ratified reference level; cluster submission.

## Open points (settled by the conformance run, not by recall)

1. The exact keywords for the standard output block under Abaqus/Explicit, and
   whether history output shares the field clock under time marks.
2. The wording of the Explicit `.sta` and `.msg` files, and whether the `.sta`
   gives a usable stable-time-increment series.
3. The precision of field data in the ODB under `double=both`.
4. Whether `odbAccess` checks out a licence token (ADR-0068 open point 6).
