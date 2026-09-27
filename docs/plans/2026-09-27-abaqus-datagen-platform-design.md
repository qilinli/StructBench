# The Abaqus data-generation pipeline as a StructBench capability — design

**Date**: 2026-09-27. **Status**: draft for the maintainer's review; the ADR is
`decisions/0071-datagen-platform.md` (Proposed). Supersedes the *scope* of
`2026-09-24-abaqus-data-pipeline-design.md` (four stages built for one
dataset); that document's stage designs stand where this one does not change
them.

## Problem

ADR-0069's pipeline exists and works: it generated, ran, exported, converted,
verified and dry-archived one 500-case dataset end to end, about 2,900 jobs
in all, with no pipeline failure. It is nevertheless a set of scripts under the
repository root, imported through `sys.path`, with no user-facing entry point,
no documentation beyond the plans, and no template that tells a new dataset
what to supply. The dataset that exercised it needed four production versions,
and each cause is now known:

| Version | Cause | Would the pipeline catch it today? |
|---|---|---|
| v1 → v2 | the severity limit was chosen after production; the deck did not request the one energy term the ledger needs (ALLPW) | the deck writer now hardcodes ALLPW, but the check that the input requests the evidence is LS-DYNA-only and reads `unsupported` on Abaqus |
| v2 → v3 | the mesh was chosen up front and the convergence set run after production | the convergence tools exist only in the private dataset repository, and nothing requires them before a sweep |
| v3 → v4 | kinematic contact created energy in 5 runs with no solver warning | yes: `plastic_dissipation_excess_max` (ADR-0066 energy-account note) |
| the penalty detour | a fix was checked on energy but not across meshes | nothing requires the mesh pair to be re-run after a solver setting changes |
| the disk stop | superseded versions accumulated; the runner had no free-space check and died on a write error | no |

Three of the five are failures of protocol that a person had to remember. The
platform goal is that a user who arrives at StructBench with a solver and a
problem finds the whole pipeline, a template that says what to fill in, and a
gate that refuses to start a production sweep until the things we learnt the
hard way have been checked.

## Principles

1. **A dataset is a definition, not code that owns the pipeline.** Everything
   generic lives in StructBench; the dataset supplies a `dataset.toml` and a
   `problem.py` that satisfy a stated contract, plus the prose only its author
   can write.
2. **Every solver keyword goes through the writer library.** No dataset edits
   deck text by string replacement (two options of dataset A were set that
   way; both become writer arguments). Every writer is a hypothesis until a
   conformance run confirms it, and the confirmation is recorded.
3. **Verification before production, mechanised.** The pre-production checks
   are a stage with an output and a stamp, not a checklist.
4. **Fail closed, record everything, delete nothing by itself.** The runner
   stops cleanly rather than crashing; provenance carries commits and hashes;
   overrides are recorded in the provenance they override.
5. **Nothing private in the package.** Dataset definitions may be private and
   live outside the repository; the package ships one public example.

## The dataset definition: the template

`structbench-datagen new <dir> --solver abaqus` writes a directory a user
fills in. Every field carries a comment saying what it means, whether it is
required, and an example. `structbench-datagen check <dir>` validates it
against the contract before anything runs.

### `dataset.toml` (what the dataset is)

| Table | Fields | Required | Meaning |
|---|---|---|---|
| `[dataset]` | `name`, `case_prefix`, `units`, `solver` | yes | identity; `units` is a `unit_factors` label; `solver = "abaqus"` |
| `[declaration]` | `unit_system`, `discretisation`, `erosion`, `fields`, optional `material_class`, `units_anchors` | yes | what a benchmark card would declare; `validate` measures against it |
| `[fixed]` | any constants | yes, may be empty | passed to `build()` with the sampled values |
| `[variables]` | `name = [low, high]` | yes | the sampled box |
| `[regions.<r>]` | sub-boxes | no | for `exclude` and `within` |
| `[splits.<s>]` | `n` + `seed`, or `points`; `exclude` / `within`; `extra`; `categorical`; `variants`; `probe = true` | yes, at least one | as today; `probe` marks a split the preflight gate does not guard |
| `[limits]` | parameters `feasible()` reads | no | the declared feasibility or severity limit, stated where it is applied |
| `[levels]` | `refine_key = "refine"`, `production = "2"`, `pilot = ["1", "2", "4"]`, `symmetry = "axisymmetric"` (or `planar`; plan 2a) | yes | the mesh-level convention the preflight uses; `mesh()` must nest across them |
| `[pilot]` | `split`, `fine_cases`, `min_free_gb`, `accepted_gaps`; with defaults (plan 2b): `increment_key = "dt_scale"`, `increment_factors = [0.5]`, `frame_key = "frame_interval"`, `frame_count_key = "n_intervals"`, `frame_factor = 0.5`, `frame_tolerance = 0.05`, `settling_margin = 0.25`, optional `contact_force_global` | yes | the preflight's targets: the pilot split, which pilots also run at the finest level, the disk margin, the verification rows the dataset accepts as known gaps, and the time and duration probes |
| `[qoi]` | `names`, `units`, `tolerance` (one relative number per name, default 0.01; plan 2b) | yes | the keys `qoi()` returns, for `converge`, the preflight's targets and the card |
| `[retention]` | `odb_fraction`, `odb_seed`, `odb_cases` | no | as today |

Reserved parameter keys: `refine` (the mesh level) and `variant`. A dataset
may add categoricals for solver options only if a writer takes them
(`contact`, `dt_scale` become `contact_pair(mechanical_constraint=)` and
`explicit_step(scale_factor=)`).

### `problem.py` (how one case is built and read)

```python
def input_deck(params: dict, variant: str | None) -> str # required
def feasible(params: dict) -> bool                       # optional, default True
def mesh(params: dict) -> deck.QuadMesh                  # required for preflight
def qoi(case: structbench.core.Case) -> dict[str, float] # required for converge and card
```

`input_deck` writes the deck through `structbench.datagen.abaqus.deck` only and
must be a pure function of its arguments (byte-identical output for identical
input). `mesh` returns the mesh `input_deck` uses, so that `preflight` and
`converge` can verify that level *k* nests level 1. `qoi` reads only a
canonical `Case`, so the same function scores a surrogate's prediction.

### What `check` verifies

Required tables and fields present and typed; the unit label known; every
split well-formed (a sampled split has a seed, an explicit one has points that
cover the box); `[levels].production` among the pilot levels; `[pilot].split`
exists and is a `probe`; `[qoi].names` equal the keys `qoi()` returns on a
synthetic case; `input_deck` is byte-stable; `mesh` nests across the pilot levels;
the declaration's field names are canonical names. It runs no solver.

### The scaffold also writes

`README.md` with the stage sequence and the exact commands; `DATA_CARD.md` with the
sections the generator fills marked as such and the author's sections left as
placeholders; a `pilot` split of three explicit points to be replaced.

## Stages

One console entry point, `structbench-datagen`, with a sub-command per stage.
Exit codes: 0 clean, 1 findings, 2 refused by a gate, 3 stopped by the
environment (disk, solver missing).

| Stage | Role | New or changed |
|---|---|---|
| `new` | scaffold a definition | new |
| `check` | validate a definition against the contract | new |
| `preflight` | the gate (below) | built (plan 2b, 2026-09-28): the case set from the pilot points, ten steps as pure functions on records, `report.md` and `stamp.json` in `<work-root>/<name>/preflight/`; the settled `[pilot]` fields are in the row above and in `docs/DATA_GENERATION.md` |
| `generate` | decks and provenance for chosen splits | built (plan 2b): refuses splits that are not probes without a passing stamp for the current `dataset.toml` and `problem.py`; `--no-preflight` is allowed and recorded as `{"skipped": true}` in every such case's provenance (`abaqus-provenance/3`) |
| `run` | the job runner | built (plan 2b): free-space check before each launch (`--min-free-gb`, default the stamp's `[pilot].min_free_gb`), a clean stop with exit 3 (running jobs finish, held cases stay pending), the estimate from the stamp printed before the first launch |
| `follow` | export and convert finished cases while a run proceeds | built (plan 2b) |
| `export` | ODB to `abaqus-npz/1` under Abaqus's Python | unchanged; ships inside the package as a data file the CLI locates and hands to `abaqus python` |
| `convert` | npz to canonical cases | unchanged |
| `verify` | the ADR-0066 instrument over the sweep | renamed from `validate` (ADR-0072, 2026-09-27) |
| `converge` | mesh-level comparison of QoIs and fields | built (plan 2a, 2026-09-27): the engine in `verification/convergence.py`, the pairing and the record in `datagen/converge.py` |
| `archive` | copy to the data tree, retain ODBs, redact | unchanged |
| `card` | render the generated sections of `DATA_CARD.md` | new |

### `preflight`

Runs on `[pilot].split`, in order; each step writes a verdict and its numbers
to `preflight/report.md`; the stage fails at the first failing step.

1. **Conformance.** One pilot at the production level with the solver's
   complete energy output requested (`*ENERGY OUTPUT, VARIABLE=ALL`): the
   ledger identity closes to float32 with the terms the standard deck
   requests, and no other term is non-zero. The deck of every pilot passes
   `input_requests_required_evidence` (extended to Abaqus, below).
2. **Deck regression.** Every pilot deck rebuilds byte for byte; no reserved
   key is set outside the writers.
3. **Feasibility.** Every pilot completes at the production level; the limit
   `feasible()` applies is printed with the pilots' values of it.
4. **Resolution — space, time, duration.** Three probes, each a measured
   number the card declares (ADR-0071 note 2026-09-27):
   - *Space.* Every pilot runs at each `[levels].pilot` level except the
     finest, and `[pilot].fine_cases` at the finest; `converge` reports the
     QoIs' observed order and extrapolated error and the fields' pooled
     relative L2 against the finest level. Passes when the production level's
     QoIs are within the dataset's stated target of the extrapolated value;
     the field numbers are reported for the maintainer to weigh and go into
     the card.
   - *Time.* (a) The solver increment: the pilots at the production level run
     again at a second stable-increment scale (proposed `[pilot].increment_scales`,
     default `[1.0, 0.5]`), and step 5's energy account must pass at both; a
     QoI that depends on the scale beyond its stated tolerance fails the step.
     (b) The frame interval: one pilot is exported at half the stored frame
     interval (proposed `[pilot].frame_probe`), and the stored fields, globals
     and QoIs at the common instants must agree to a stated tolerance; the
     fastest feature of the response (the shortest rise time among the
     globals) is reported in frames.
   - *Duration.* On every pilot the settling time is measured — the last frame
     at which any QoI changes by more than its stated tolerance, and, where the
     problem names a contact-force global, the frame it drops below 10⁻³ of
     its peak — and the share of the horizon after settling is reported.
     Passes when the slowest pilot settles within the horizon with the stated
     margin; the card declares the stored horizon, the settling range and the
     scored horizon (fixed, or per case from the named global). A tail kept
     beyond settling is the bed for post-settling property tests, not for
     accuracy scoring.
5. **Energy account.** On every pilot at every level: `plastic_dissipation_excess_max`
   passes, and `energy_gain_max` is below its reference level (below). A
   change to any solver setting re-arms the whole preflight because the stamp
   covers `problem.py`.
6. **Budget.** Median and maximum wall time per level from the pilots; the
   production sweep's core-hours and disk (ODB, npz, h5, side files) estimated
   from them; free space compared with the estimate plus `min_free_gb`.
7. **Verification.** `verify` over the pilots: no `fail` except rows named
   in `[pilot].accepted_gaps`, each of which the report lists with the
   dataset's stated reason.

The stamp (`preflight/stamp.json`) records the sha256 of `dataset.toml` and
`problem.py`, the StructBench commit, the date, and the per-step verdicts.
`generate` compares the hashes: a production split is generated only against
a passing stamp for the current definition.

### `converge`

The generic engine moves from the private repository into
`structbench.verification.convergence` (solution verification belongs beside
ADR-0066's instrument): matching nested nodes, restriction of element fields
onto the coarser mesh with axisymmetric or planar volume weights, Richardson
extrapolation and Roache's GCI with the observed-order statuses (monotone,
oscillatory, diverging, flat), pooled relative L2 per field (ADR-0055), and
the breakdown that separates a stress difference into contact-phase,
post-release, time-shift and block-averaged parts. The dataset supplies only
`qoi()` and `[levels]`. The CLI pairs runs across roots by their parameters,
so a production run supplies its own level.

### `card`

Renders from records only: the declaration, the verification summary
(verdict counts, findings, measured spreads, the accepted gaps and their
reasons), the convergence tables, QoI ranges, run statistics, storage,
retention and provenance. Author sections stay untouched between renders.

## Package layout

```
src/structbench/datagen/            solver-agnostic: sampling, generate, run, follow,
                                    preflight, archive, card, template/, cli
src/structbench/datagen/abaqus/     deck writers, odb_export.py (standalone,
                                    Python 3.10, imports nothing from the package)
src/structbench/datagen/examples/   the public example definition (below)
src/structbench/verification/convergence.py
docs/DATA_GENERATION.md             the user guide
docs/datagen/abaqus-conformance.md  today's STANDARD_INPUT_BLOCK.md
```

The readers and the adapter stay where they are (`core/io/abaqus.py`,
`core/io/abaqus_run.py`). `data_generation/` at the repository root keeps
only glue that is not importable (the LS-DYNA collectors); `ARCHITECTURE.md`'s
statement that nothing under it is importable is amended by the ADR, and the
layering becomes `core ← datasets ← verification ← {eval, benchmarks, datagen} ← cli`.
The `datagen` extra (ADR-0069) remains the install path for scipy.

## Verification additions

1. `input_requests_required_evidence` for Abaqus decks: the `*ENERGY OUTPUT`
   list against the ledger terms including ALLPW, `*OUTPUT, HISTORY` present
   on the field clock, `TIME MARKS=YES`, the reaction-force request when a
   rigid body exists.
2. The three hourglass rows (`zero_energy_mode_*`) measured from the stored
   `global/hourglass_energy` where a case stores it; they read `unsupported`
   today although the global exists.
3. Energy gain: ratify the sourced reference level already in the catalogue
   (Belytschko, Liu & Moran's 1 %, B-BLM-1, provisional) for `energy_gain_max`
   and `energy_loss_max` as an *indicator*: exceedance is `review`, a
   person's call, as ADR-0066 clause 7 intends. Not a requirement: a tolerance
   tight enough to matter would be fitted to our own runs, which clause 7
   forbids, and explicit schemes gain small amounts legitimately. The level
   would have marked the v3 defect `review` on its first validation.

Each is a change to the public verification module and gets its own ADR-0066
note.

## Documentation and the public example

`docs/DATA_GENERATION.md`: defining a dataset from the template, the stages
and their commands, the preflight protocol and what each step protects
against, the conformance block, practices (the public form of the private
lessons: settle filters and output requests before production, decide which
energy checks are requirements, re-run the mesh pair after any setting
change, budget the disk per version, verify on real output).

The public example, `structbench/datagen/examples/abaqus_conformance/`: the
single-rod conformance case the conformance document already describes, with
a three-point pilot split. It is what `new` copies its worked example from,
what the docs walk through, and what the env-gated acceptance test runs. It
carries none of any private dataset's box, splits or constants.

## Testing

- Unit tests per stage with synthetic decks, as today (the existing tests
  move with the code).
- `preflight`, `generate`'s gate, `run`'s budget and stop, and `follow` are
  tested against a stub solver: `run --abaqus <stub>` invokes a script that
  writes a canned `.sta`, `.msg`, `.dat` and an `npz` from a fixture, so the
  whole chain runs without Abaqus and without `odbAccess`.
- `converge` is tested on analytic fields with known orders (the private
  tests move).
- One acceptance test, env-gated on `STRUCTBENCH_ABAQUS`, runs the public
  example end to end on a machine with the solver.

## Migration of dataset A (no re-run)

Its `sweep.toml` becomes `dataset.toml` and gains `[levels]`, `[pilot]`,
`[qoi]` and `[limits]`; its `model.py` becomes `problem.py`, renames `build`
to `input_deck` and `rod_mesh` to `mesh`, gains `qoi` from the private
`qoi.py`, and replaces its two string edits with writer arguments. The 500
production decks and the 20 convergence decks must rebuild byte for byte
against the sha256 in their provenance; that regression is the acceptance
test of the migration. The private `qoi.py` and `convergence.py` retire in
favour of the package.

## Phasing

1. **Platform** (this design): package move and CLI; writer arguments for the
   two options; `converge` and the `qoi` hook; template, `new`, `check`;
   `preflight` and the gate; `run` hardening and `follow`; the three
   verification additions; docs and the example; `card`; A's migration.
   Roughly eight to twelve sessions, each ending green and mergeable.
2. **Dataset B's solver side**: combined-hardening material, point mass,
   self-contact, base and trigger writers; backstress and plastic-strain
   fields in the canonical schema; a combined-hardening material class and its
   constitutive row. Each is an ADR of its own.
3. **Admission**: `card` feeding ADR-0065's card fields and compliance table.

## Out of scope

LS-DYNA generation (only readers exist and no generation is planned);
cluster submission; any change to dataset A's data.

## Open points

- Whether `generate`'s gate should also cover `probe` splits above a size
  (say 50 cases). Proposed: no; probes are how the pilots are found.
- The stub solver's fidelity: it cannot exercise `odbAccess`; the acceptance
  test does.
- The finest pilot level's cost for slow datasets (B's runs are minutes to an
  hour each); `[pilot].fine_cases` exists so that only a few pay it.
- The `[pilot]` fields for the time and duration probes were settled by plan
  2b (2026-09-28), all with defaults: `increment_key = "dt_scale"` and
  `increment_factors = [0.5]` relative to the production value; `frame_key`,
  `frame_count_key`, `frame_factor = 0.5` (0 disables the probe);
  `frame_tolerance = 0.05`; `settling_margin = 0.25`; an optional
  `contact_force_global`; and `[qoi].tolerance`, one relative number per
  quantity (0.01 each), as the single target of the space, increment and
  settling rules. The frame probe's pass criterion became the
  midpoint-interpolation error of the finer export (the resolution of the
  stored clock), with the common-instant agreement reported. Whether the
  scored horizon is a card field per benchmark (ADR-0039) or per case is an
  admission question the card format must answer before the first Abaqus
  dataset is admitted.
