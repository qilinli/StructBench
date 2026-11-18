# Data generation with StructBench

*The guide to `structbench-datagen` (ADR-0071). One stage the design names is
not built yet: `card`, which will render the generated parts of
`DATA_CARD.md`; its plan is `docs/plans/2026-11-18-datagen-card-plan.md`.*

StructBench generates reference data through `structbench-datagen`. A dataset
is a *definition* in a directory of its own, passed by path and never code in
this repository, so it may be private. The command takes the definition
through eleven stages, from a template to an archived, verified sweep, and a
gate between them refuses production until the pre-production checks have
passed.

Read this page in order the first time. The
[walkthrough](#a-walkthrough-without-a-solver) runs every stage on the shipped
example with no solver installed, so each stage's output can be seen once
before a licence or a cluster is involved.

## Install

    pip install -e ".[datagen]"      # from a checkout; adds scipy

The solver is not a dependency. Today the one solver the stages drive is
Abaqus/Explicit (ADR-0068), found on the path or named with `--abaqus`; only
`preflight`, `run`, `export` and `follow` need it.

## The definition

A definition is two files:

- `dataset.toml` — what the dataset is.
- `problem.py` — how one case is built and read. It may import sibling
  modules of its own directory.

`structbench-datagen new <dir>` writes both from the shipped example, with a
`README.md` and a `DATA_CARD.md`, and names the dataset after the directory.
**The directory must be a git repository with a commit** before `preflight`
or `generate` will run: every case's provenance records the commit of the
definition it was generated from, and both stages refuse with "commit the
definition first" otherwise.

### `dataset.toml`

The example's file
(`src/structbench/datagen/examples/abaqus_conformance/dataset.toml`) carries a
comment on every key; it is the reference for spelling. The tables:

| Table | Fields | Required | Meaning |
|---|---|---|---|
| `[dataset]` | `name`, `case_prefix`, `units`, `solver` | yes | identity: the sweep folder and canonical dataset id, the prefix of every case id, a unit label `structbench.core.io.unit_factors` knows (the deck's own units; canonical files are written in SI), and `solver = "abaqus"` |
| `[declaration]` | `unit_system` (required), `discretisation`, `erosion`, `fields`, `material_family` | yes | what a benchmark card would declare; `verify` measures the runs against it |
| `[fixed]` | any constants | yes, may be empty | handed to every case with the sampled values |
| `[variables]` | `name = [low, high]` | yes | the sampled box |
| `[regions.<r>]` | sub-boxes | no | named boxes a split may `exclude` or sample `within` |
| `[splits.<s>]` | `n` and `seed`; or `points`; or `grid`; with `exclude` / `within`, `extra`, `categorical`, `variants`, `probe`, `withheld` | at least one | the case sets (below) |
| `[limits]` | the parameters `feasible()` reads | no | the declared feasibility or severity limit, stated where it is applied |
| `[levels]` | `refine_key`, `production`, `pilot`, `symmetry` | yes | the mesh levels: the reserved parameter the pipeline sets, the level production runs at, the levels the preflight runs (coarse to fine; three for an extrapolation), and the volume weights `converge` uses (`axisymmetric` or `planar`) |
| `[pilot]` | `split`, `fine_cases`, `min_free_gb`, `accepted_gaps`, and the probe fields | yes | the preflight's targets ([below](#the-pilot-fields)) |
| `[qoi]` | `names`, `units`, `tolerance` | yes | the quantities of interest `qoi()` returns, and one relative tolerance each (default 0.01) |
| `[retention]` | `odb_fraction`, `odb_seed`, `odb_cases` | no | which solver databases `archive` keeps (default about 5 %) |

A split takes one of three forms. A *sampled* split draws `n` points from a
seeded, scrambled Sobol sequence, so the first *k* cases of a split are a
nested subset. An *explicit* split lists `points`. A *grid* split,
`grid = { L = [1.2, 1.8], v0 = [12.0, 18.0] }`, is the full factor grid of the
listed levels, in list order, with no `n`, `seed` or `points` (string levels
are categorical; `check` refuses a grid over 64 cases): the public `test` set
of open evaluation is such a grid (ADR-0074). Two flags change how a split is
treated:

- `probe = true` marks a pilot or probe split. The gate does not guard it, and
  its cases are generated without the refine key unless the preflight sets it.
- `withheld = true` marks a split whose reference outputs stay out of the
  public archive, the `blind` set: `archive` writes its cases to the canonical
  tree with inputs and seed frames only, and the full cases to
  `<data-root>/withheld/<name>/<blind tag>/`, so it needs `--blind-tag` and
  `--seed-frames` (the benchmark protocol's `input_frames`; `datagen` sits
  below `benchmarks` and does not read it from a `benchmark.py`). `train` and
  `val` are never withheld.

Case ids are `<prefix>-<split>-<index>[-<variant>]`. The refine key (and
`variant`) are reserved parameter names.

### `problem.py`

```python
def input_deck(params: dict, variant: str | None) -> str   # required
def mesh(params: dict) -> deck.QuadMesh                    # required
def qoi(case: structbench.core.Case) -> dict[str, float]   # required
def feasible(params: dict) -> bool                         # optional
```

- `input_deck` writes the solver input through the writer library
  (`structbench.datagen.abaqus.deck`) and never edits deck text. It must be a
  pure function of its arguments: the same parameters give the same bytes, in
  this interpreter and in the next.
- `mesh` returns the mesh `input_deck` uses at the level in
  `params[refine_key]`. Each level must nest in the next finer one (every
  coarser node is a finer node), because `converge` compares fields node for
  node.
- `qoi` reads a canonical case and nothing else, so the same function scores a
  surrogate's prediction. It must accept a one-frame case: the preflight
  evaluates it on every prefix of a trajectory.
- `feasible` filters sampled points the way `exclude` does; explicit points
  bypass it.

## The stages

| Stage | Needs a solver | Reads | Writes |
|---|---|---|---|
| `new <dir>` | no | the shipped example | the four files of a definition |
| `check <dir>` | no | the definition | nothing; prints each contract problem in a sentence |
| `preflight` | yes | the definition | `<runs>/<name>/preflight/`: its own cases, `report.md`, `stamp.json` |
| `generate` | no | the definition, the stamp | per case `<id>.inp` and `provenance.json`; `manifest.csv` |
| `run` | yes | the case folders | per case the solver's files, `run.json`, `runner.log`; `run_log.csv` |
| `follow` | yes | a sweep that is running | exports and canonical files as cases finish |
| `export` | yes | each case's solver database | `<id>.npz`, the `abaqus-npz/1` intermediate |
| `convert` | no | `<id>.npz`, `<id>.inp`, `provenance.json`, `run.json` | `canonical/<id>.h5` |
| `verify` | no | the canonical cases, the run files, `[declaration]` | `datacheck/`: `run_evidence.json`, `measurements.json`, `report.md` |
| `converge` | no | canonical cases at several levels, `qoi()` | `converge/`: `convergence.json`, `convergence.md` |
| `archive` | no | `datacheck/measurements.json`, the case folders | the data tree: `raw/<name>/abaqus/<id>/`, `canonical/<name>/<id>.h5` |

Every stage answers `--help`. Exit codes mean the same thing everywhere:

| Code | Meaning |
|---|---|
| 0 | clean |
| 1 | findings: a job did not complete, a case failed to convert, a verification row fails, a preflight step did not pass |
| 2 | refused: the definition does not meet the contract, the gate is closed, an argument is wrong; nothing was changed |
| 3 | stopped by the environment: the solver executable was not found, or the free space fell below the margin; run it again when the environment is right, and it resumes |

Every stage resumes: `generate` skips an identical deck, `run` a finished
case, `export` an exported one, `convert` a converted one. Nothing is
overwritten: a deck that differs is a conflict, a deck whose case already ran
is locked, an interrupted attempt moves to `attempts/<n>/`.

### What is on disk

    <runs>/<name>/                       the sweep
      manifest.csv                       every planned case and its parameters
      run_log.csv                        one row per launch
      <case id>/
        <case id>.inp                    the deck
        provenance.json                  parameters, hashes, commits, the stamp it was generated against
        run.json                         status, times, the command, what the solver's files say
        <case id>.sta .msg .dat .odb     the solver's own files
        <case id>.npz                    the export
        runner.log                       the solver's console; never archived (it holds licence text)
        attempts/<n>/                    an interrupted or retried attempt
      canonical/<case id>.h5             the canonical case
      datacheck/                         verify's record and report
      converge/                          converge's record and report
      preflight/                         a sweep of its own, with stamp.json and report.md

Two limits to know before designing a dataset. `convert` writes
two-dimensional cases only (the element shapes `converge` compares are in its
section below). `verify` writes one record per sweep: `verify --split`
replaces the record of the splits verified before, and `archive` takes its
case list from that record, so verify the whole sweep before archiving it.

## A walkthrough without a solver

The tests drive every stage with a stand-in for the solver,
`tests/datagen/fake_solver.py`: a script that answers the solver's command
lines and writes an analytic response for the shipped single-rod example. It
is reached from the command line through two options, `--solver-args` (on
`preflight` and `run`) and `--exporter-args` (on `preflight`, `export` and
`follow`), which place a script after the executable.

**Its numbers are synthetic.** The response is a formula chosen so that a
sound preflight passes; nothing below is a simulation, and nothing it writes
is reference data. A run made this way is recognisable afterwards: its
`run.json` records the command, and the command names the script. Use a
scratch directory, and delete it when done.

From a checkout, with the `datagen` extra installed (PowerShell: put the path
in `$FAKE` the same way):

    FAKE=$(pwd)/tests/datagen/fake_solver.py
    mkdir /tmp/walk && cd /tmp/walk

    structbench-datagen new rod
    git -C rod init -q && git -C rod add . && git -C rod commit -q -m "the definition"
    structbench-datagen check rod

`check` answers `ok: rod: 2 splits, 1 variables, pilot 'pilot' at levels
['1', '2', '4']`. Break something in `dataset.toml` (misspell a `[pilot]`
key) and run it again to see a contract problem named.

    structbench-datagen preflight --dataset rod --work-root runs \
        --abaqus python --solver-args $FAKE --exporter-args $FAKE

Twelve cases are planned from the three pilot points: each pilot at levels 1
and 2 and one at level 4, the three pilots at half the solver increment, one
pilot exported at half the frame interval, and one conformance run. The four
cases at the production level run first, then the rest; ten verdicts print as
the steps are judged, ending in `passed -> runs/rod/preflight`. It takes two
to three minutes. Open `runs/rod/preflight/report.md`: it is rendered from
`stamp.json` alone, and the stamp binds the verdicts to the hashes of the
definition. Set `FAKE_SOLVER_SETTLE=slow` and run it in a second work root to
see a step fail (the response never settles, so `duration` does) and the exit
code become 1.

    structbench-datagen generate --dataset rod --work-root runs

`written=11`: the eight `train` cases and the three pilots. Without the
passing stamp this is refused with exit 2, naming the split and the reason;
change a tolerance in `dataset.toml` and try it, then restore the file.
`provenance.json` of a `train` case names the stamp it was generated against.

    structbench-datagen run --sweep runs/rod --abaqus python --solver-args $FAKE --workers 2
    structbench-datagen export --sweep runs/rod --abaqus python --exporter-args $FAKE
    structbench-datagen convert --sweep runs/rod

`run` prints one line per case and a summary; each case folder gains the
solver's files and `run.json`. (`export` has nothing to do here: the stand-in
writes the export with the run.) `convert` answers `written=11 skipped=0
failed=0` and the canonical cases are in `runs/rod/canonical/`.

    structbench-datagen verify --sweep runs/rod --dataset rod

This exits 1, and that is the lesson of the step: one row fails on every
case, `solver_identity_complete`, because the stand-in's run record does not carry
the solver's full identity (version and precision). `runs/rod/datacheck/report.md` says what was checked,
what was found, what is measured without a criterion, and what could not be
checked. The example's `[pilot].accepted_gaps` names that row, which is why
the preflight passed with it: an accepted gap is the preflight's notion,
listed with its verdicts in the stamp, and `verify` reports every fail
whatever a definition accepts.

    structbench-datagen converge --dataset rod --sweep runs/rod/preflight --anchor pilot

The preflight's cases are the ones run at several levels, so the comparison is
made there; `--anchor pilot` lets the pilot split's production-level run stand
for production. `convergence.md` has, per quantity of interest, the value at
each level, the observed order, the extrapolated value and the error at the
production level, then the field errors of each coarser level against the
finest. The stand-in's response converges at second order by construction,
and the record says so.

    structbench-datagen archive --sweep runs/rod --dataset rod --data-root tree
    structbench-datagen archive --sweep runs/rod --dataset rod --data-root tree --yes

The first prints the plan and writes nothing; the second copies each file
through a `.partial` name and verifies it. `tree/raw/rod/abaqus/<id>/` holds
the run record and `tree/canonical/rod/` the cases. Run it a third time:
every file is verified and none copied.

That is every stage but `follow`, which does `export` and `convert` beside a
running sweep. With a real solver the commands are the same without the two
stand-in options.

## The stages one by one

### `new` and `check`

    structbench-datagen new <dir>
    structbench-datagen check <dir>

`new` refuses a directory that is not empty. `check` runs every contract
check that needs no solver: the tables and their types; the hooks; everything
`generate` would refuse (case ids that break the solver's job-name rule, a
parameter both fixed and sampled, an unknown region); a deck that is
byte-stable, in a fresh interpreter with another hash seed too; mesh levels
that nest; `qoi()` returning the declared names, on a one-frame case too; the
`[declaration]` as `verify` will read it; and `[pilot]` entries that name
something that exists (a fine case that is a pilot, an accepted gap that is a
catalogue row, an accepted increment change on a pilot). `preflight` runs
`check` first and refuses on any finding.

### `generate`

    structbench-datagen generate --dataset <dir> --work-root <runs>
        [--split NAME ...] [--force] [--dry-run] [--no-preflight]

Writes each planned case's deck and provenance into `<runs>/<name>/`.
`--dry-run` prints the count per split. A split that is not a probe is
generated only against a passing stamp for the current definition (the gate,
[below](#the-preflight)); `--no-preflight` generates anyway and records the
omission in every such case's provenance. `--force` replaces a deck that
differs, never one whose case has run.

### `run`

    structbench-datagen run --sweep <runs>/<name>
        [--split NAME ...] [--cases ID ...] [--limit N] [--workers 6]
        [--timeout S] [--retry-failed] [--abaqus EXE] [--dry-run]
        [--min-free-gb GB] [--solver-args ARG ...]

Runs the decks, `--workers` at a time, each job on one core. A case is done
once its `run.json` exists; its status comes from the solver's own files, read
by the reader `verify` uses, so "did it finish" has one answer everywhere. A
second runner on the same sweep is refused by a lock. The budget is
[below](#the-runners-budget).

### `export` and `convert`

    structbench-datagen export  --sweep <runs>/<name> [--cases ID ...] [--abaqus EXE]
    structbench-datagen convert --sweep <runs>/<name> [--split NAME ...] [--out DIR]

An Abaqus database is readable only by Abaqus's own Python, so the path is two
processes (ADR-0068): `export` hands the packaged exporter to `abaqus python`,
which writes `<id>.npz` in the solver's units and labels; `convert` reads that
and the deck and writes the canonical case in SI. A case that fails to convert
is reported with its reason and is never half-written.

### `verify`

    structbench-datagen verify --sweep <runs>/<name> --dataset <dir>
        [--split NAME ...] [--data-root DIR]

Runs the reference-data verification instrument (ADR-0066) over the sweep: it
collects each run's evidence, measures every canonical case against the
catalogue of quantities and the `[declaration]`, judges the measurements, and
writes the record and the report. A case that ran and has no canonical file is
measured from its deck and run record alone, so it is reported and not lost.
Exit 1 on any failing row or any case without a canonical file.

`verify` asks whether each run is internally sound. Whether the *setup*
reproduces a physical experiment is a different question, answered by
`structbench-validate` (ADR-0072): it measures canonical cases and a shipped
reference-experiment set with the same functions and writes a record of the
deviations, with no threshold and no verdict. `structbench-validate --list`
names the sets; `src/structbench/validation/references/taylor_copper.md` is
the first one's provenance.

### `archive`

    structbench-datagen archive --sweep <runs>/<name> --dataset <dir> --data-root <tree>
        [--split NAME ...] [--yes] [--prune-odb] [--blind-tag TAG] [--seed-frames N]

Copies the verified cases into the data tree. Without `--yes` it prints the
plan and writes nothing. Nothing is ever overwritten: a file already at its
destination is skipped when identical, and a different one stops that case.
The solver's printed files are copied with their licence lines removed, and
`runner.log` is never copied. `[retention]` chooses which solver databases are
kept; `--prune-odb --yes` then deletes the local ones that are not, only for
archived cases whose export names that database's hash.

`preflight`, the runner's budget, `follow` and `converge` have sections of
their own below.

## The shipped examples

Three definitions ship in `src/structbench/datagen/examples/`:

- `abaqus_conformance` — one rod striking a frictionless rigid wall, the
  conformance case of `docs/datagen/abaqus-conformance.md` and the template
  `new` copies;
- `abaqus_cyclic_element` — a block of a combined-hardening material under one
  cycle of axial strain, with an initial plastic strain and backstress (one
  element at level 1, seconds to run);
- `abaqus_drop_mass` — a cylinder upset by a dropped rigid plate whose
  reference node carries a point mass, on a mesh whose quads are not
  rectangles.

The design is `docs/plans/2026-09-27-abaqus-datagen-platform-design.md`.

## The preflight

    structbench-datagen preflight --dataset <dir> --work-root <runs>
        [--abaqus EXE] [--workers N] [--timeout S] [--rejudge]
        [--solver-args ARG ...] [--exporter-args ARG ...]

The preparation stage. It plans a case set from the `[pilot].split` points —
every pilot at every `[levels].pilot` level except the finest (the
`fine_cases` there too), the pilots at the production level with the solver
increment scaled by each `increment_factors` entry, one pilot exported at
`frame_factor` times the stored frame interval, and one conformance run with
every energy term requested — and drives `run`, `export`, `convert`,
`verify` and `converge` over it in `<runs>/<name>/preflight/`, a sweep of its
own (case ids carry `-L<level>`, `-T<value>`, `-F`, `-E`). The pilots at the
production level and the conformance run go first; if they do not complete
or the ledger does not close, nothing more is launched. Ten steps turn the
records into verdicts, in ADR-0066's vocabulary:

| Step | Passes when |
|---|---|
| `deck_regression` | every preflight deck rebuilds byte for byte in a fresh interpreter, and each probe key (`refine`, `increment_key`, `frame_key`) changes the deck it is meant to change |
| `feasibility` | every pilot completes at the production level (the `[limits]` and `feasible()` values are listed) |
| `conformance` | every preflight deck asks for the evidence the instrument requires (`input_requests_required_evidence`, [the Abaqus requirement](datagen/abaqus-conformance.md#what-the-instrument-requires-of-an-abaqus-input)), judged on the decks before anything is written, so a deck that could never supply its evidence launches nothing (`fail`), nor does one the platform cannot tell about (`not_assessable`); and on the run with `*ENERGY OUTPUT, VARIABLE=ALL`, the ledger identity closes with the standard terms to 1e-5 and no other term is non-zero |
| `space` | each quantity of interest at the production level is within its `[qoi].tolerance` of the value extrapolated from three levels, whose two refinement ratios may differ (with no observed order: `review` when the quantity moves across the levels by no more than its tolerance, which a person may accept in `[pilot].accepted_reviews` as `space.<name>` with a reason, and `fail` when it moves more; the field errors are reported for the maintainer to weigh) |
| `increment` | each quantity of interest changes by less than its tolerance between the production increment and the scaled ones, except the changes the dataset accepts for one pilot and one quantity in `[pilot].accepted_increment`, each with its reason; and no energy row of those runs fails, which no acceptance covers (the summary names the rows the instrument could not assess) |
| `frame` | every `1 / frame_factor`-th frame of the finer export predicts the frames between by linear interpolation to `frame_tolerance`, for every stored field except those `[pilot].frame_reported` names, each reported with its error and the dataset's reason (the common-instant agreement with the stored clock and the shortest rise time among the globals are reported too); at least one node field must be judged, or the step is `not_assessable` |
| `duration` | the slowest pilot settles — every quantity of interest within its tolerance of its final value — by `(1 − settling_margin)` of the horizon; where `contact_force_global` is named, the frame at which it drops below 10⁻³ of its peak is reported per pilot |
| `energy` | no energy row of the instrument fails on any preflight case |
| `budget` | the free space covers what production still has to generate — the production cases not yet completed in the sweep, sized by their own meshes at the pilots' median bytes per node — plus `min_free_gb`; nothing left passes. The wall estimate (the pilots' seconds per element × the remaining elements) ignores increments and severity; wall time per level is reported |
| `verification` | no row of the instrument fails on any preflight case, except the rows named in `accepted_gaps`, which are listed with their verdicts |

`report.md` and `stamp.json` are written beside the cases. The stamp
(`preflight-stamp/1`) records the sha256 of both definition files and of the
sibling modules `problem.py` imported, the StructBench version and commit,
the date, every verdict with its numbers, the budget and the case ids. It
passes when every step is `pass` or `not_applicable`, or is a `review` whose
every key the dataset accepts in `[pilot].accepted_reviews`; an acceptance
never rescues a `fail` or a `not_assessable`, and the stamp lists the accepted
reviews with their reasons, the unaccepted ones, and the acceptances that
never occurred. The increment changes a dataset accepts are listed the same
way: `accepted_increment` (each entry with its measured change at every scaled
increment, the tolerance and the reason), `unaccepted_increment` (the changes
beyond tolerance that no entry names, which fail the step) and
`unused_accepted_increment` (entries the runs do not exceed). A value that is not finite never passes: it makes the step
`not_assessable` (or, in the solver's energy history, `fail`). **`generate` refuses a split that is not a probe
unless a passing stamp exists for the current `dataset.toml`, `problem.py`
and siblings**; `--no-preflight` generates anyway and writes
`"preflight": {"skipped": true}` into every such case's provenance, where a
stamp is otherwise recorded by its hash and date. Probe splits are never
gated. Because the preflight evaluates `qoi()` on every prefix of a
trajectory, `qoi()` must accept a one-frame case; `check` exercises that.

When only the judging changed — a tolerance, `frame_reported`,
`accepted_reviews`, `accepted_increment`, `qoi()`, the declaration, or the
platform's own judging (a release that extrapolates a ladder it used not to) —
`preflight --rejudge`
re-evaluates every step on the runs already made, with no solver. It accepts
them as the same evidence only if the case set is the planned one, every case
has run, every deck on disk equals the current definition's byte for byte and
every case's units are the current units; otherwise it names the first case
that is not and exits 2. The stamp then carries the current hashes, which the
gate reads, and `runs_generated_under`, the hashes the runs were made under.

Running the preflight again resumes: `run` skips finished cases, `export`
exported ones, `convert` converted ones. A preflight folder that holds runs of
another definition — either file or a sibling changed, whether or not the
decks did — is refused (exit 2) and nothing in it is deleted; move it aside.
Exit codes: 0 passed, 1 a step failed or needs review (the stamp says which),
2 refused, 3 stopped by the environment (the free space fell below
`min_free_gb`, or the solver executable was not found; `run`, `export` and
`follow` answer a missing solver with 3 too).

### The `[pilot]` fields

| Field | Default | Meaning |
|---|---|---|
| `split` | required | the probe split whose `points` are the pilots |
| `fine_cases` | required | plain case ids that also run at the finest level (three levels are needed for an extrapolation; their two ratios need not be equal) |
| `min_free_gb` | required | the free-space margin: the budget step's, and the runner's before every launch |
| `accepted_gaps` | required | verification rows the dataset accepts as known gaps |
| `increment_key` | `"dt_scale"` | the `[fixed]` key the increment probe scales; it must be a constant there — a split that samples, pins or points it (a probe split included) is refused, and without it in `[fixed]` the step is `not_assessable` |
| `increment_factors` | `[0.5]` | the production value in `[fixed]` times each factor; `[]` disables the probe |
| `frame_key`, `frame_count_key` | `"frame_interval"`, `"n_intervals"` | the stored clock's keys in `[fixed]`, constants like the increment key; the count divided by `frame_factor` must be whole |
| `frame_factor` | `0.5` | the frame probe's interval as a fraction of the stored one; `1 / frame_factor` must be whole (0.5, 0.25, 0.2, …) so the stored clock is a stride of the probe's; `0` disables the probe (`not_applicable`) |
| `frame_tolerance` | `0.05` | the interpolation error the stored clock may leave in any judged field |
| `settling_margin` | `0.25` | the share of the horizon that must lie after the slowest pilot's settling |
| `contact_force_global` | none | a stored global (the `global/` prefix may be dropped) whose separation frame the duration step reports |
| `frame_reported` | `{}` | stored fields the frame step reports rather than judges, `key = "reason"` (keys `node/<f>`, `<block>/<f>`, `global/<g>`); empty judges every stored field, and the example declares acceleration |
| `accepted_reviews` | `{}` | reviews a person accepts, `"<step>.<name>" = "reason"` (e.g. `space.final_length`); they rescue a review, never a fail or a missing measurement |
| `accepted_increment` | `{}` | changes beyond tolerance at the scaled increment that a person accepts, `"<pilot case id>.<quantity of interest>" = "reason"` (e.g. `"ACX-pilot-0001.final_length"`): a sensitivity of the problem and not of the increment, such as the timing of a late second instability. The case is the pilot's plain id (`check` refuses any other), the quantity one of `[qoi].names`. An entry covers that pilot and quantity at every factor and nothing else: a change not listed still fails, a missing or non-finite value and a failing energy row are never covered, and an entry the runs do not exceed is reported as unused |

`[qoi].tolerance` is one relative number per quantity of interest (default
0.01 each): the target for space, the increment step and the settling rule.

## What the preparation stage establishes

The preflight probes the pilot cases for three resolutions, and the data card
declares each with the number behind it:

- **space** — the mesh: nested levels, the quantities of interest extrapolated
  across them, the field metrics; the production level is chosen against them
  (the `space` step, on `converge`'s record);
- **time** — the solver increment (the energy account must hold at more than
  one stable-increment scale, and the quantities of interest must not move:
  the `increment` step) and the stored frame interval (a finer export must be
  predictable from the stored clock: the `frame` step);
- **duration** — the stored horizon against the response's settling time: when
  the quantities of interest stop changing, when contact ends, how much of the
  horizon lies after that, and which part of the trajectory is scored (the
  `duration` step).

The questions are the same when done by hand; the stage makes the answers
reproducible and binds them to the definition that produced them.

## The runner's budget

`run` reads the sweep's stamp when it has one: before the first launch it
prints the chosen cases' estimated hours and disk against the free space, and
before every launch it compares the free space with the margin
(`--min-free-gb`, default the stamp's `min_free_gb`). Below the margin it
launches nothing more; running jobs finish and are recorded, held cases stay
pending, and the exit code is 3. Without a stamp or a flag, nothing is checked.

## Following a run

    structbench-datagen follow --sweep <runs>/<name> [--interval 60] [--once] [--split NAME ...]

Beside a running sweep: each round hands the cases whose run completed and
that have no export to the exporter, then converts what has an export and no
canonical file. It stops when the runner's lock is gone and nothing is left,
after one round with `--once`, or when the runner is gone and the exporter
made no progress. Exit 1 when an export or a conversion failed.

## Convergence across mesh levels

`structbench-datagen converge --dataset <dir> --sweep <runs>/<name> [--root <other run root> ...] [--anchor SPLIT] [--cases ID ...]`
pairs every production run with the probe runs at the other `[levels]` of the
same parameters (across run roots, so a production that moved to a finer mesh
keeps its older runs), extrapolates each quantity of interest from the three
finest levels (Richardson, with the observed order and a status that says
when there is none; when the two refinement ratios differ, the order is solved
from the relation for unequal ratios of Celik et al. 2008, J. Fluids Eng. 130,
078001, the record carries `coarse_ratio` beside `ratio`, and a constant
ratio keeps the closed form and its record unchanged), and measures every stored field of each
coarser level against the finest with the headline relative L2, restricted
with the `[levels].symmetry` volume weights. It writes `convergence.json`
(byte-stable, case ids only) and `convergence.md` under `<sweep>/converge/`.
Groups that have runs from probe splits only are counted in a note, not
compared, unless `--anchor` names the probe split whose run at the production
level stands for production (the preflight names its pilot split); `--cases`
restricts the pairing to the ids listed. The dataset supplies only
`[levels]` and `qoi()`; the engine is `structbench.verification.convergence`.
Level labels are refinement factors proportional to 1 / h, larger meaning
finer, and `[levels].pilot` lists them coarse to fine; Richardson
extrapolation needs the three finest to be positive numbers in a constant
ratio (word labels get field errors only). The restriction of element fields
supports meshes of axis-aligned rectangular quads; other elements are refused
by name, per case.

## Before production

[`datagen/abaqus-lessons.md`](datagen/abaqus-lessons.md) is what the first
Abaqus dataset taught, written as rules with the reason behind each and what
the pipeline now does about it, and it ends in a pre-production checklist.
Read it before the first run of a new model.

## Training on a dataset before admission

A dataset that has passed `verify` and `archive` can be trained on with
`structbench-train` before it is admitted, and without any of its names
entering the public repository. Beside the dataset definition, write a
*benchmark definition*: a directory holding `benchmark.py` that exports a
`BenchmarkSpec` named `SPEC` — the card, the split lists, the auxiliary
channels, the QoIs, the boundary feature, and the scored horizon, which may
be a function of the case id. The run config then names that directory
instead of a registry key:

    [run]
    benchmark = "/path/to/my_dataset/benchmark"

and `--data-root` points at the canonical cases. The config loader checks the
model's `input_frames` against the definition's card as it does for a shipped
benchmark, and every run's `config.json` records the definition's absolute
path and the sha256 of its files, so a result is tied to the definition that
produced it. Such a benchmark is unadmitted: it has no registry row, no
landing page and no results registry; admission is a separate step. On a
mesh case an auxiliary name may be a registered field computed from the
element block (stress, plastic strain) and projected onto the nodes by an
area-weighted mean; `benchmark.py` may import sibling modules of its own
directory, which are hashed with it.
