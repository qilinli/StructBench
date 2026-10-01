# Data generation with StructBench

*ADR-0071 is built through part three (a), with the Abaqus verification rows
(2026-09-28). Part three still owes the energy-gain indicator's ratification,
the card generator and this guide in full; until then this page is the working
reference.*

StructBench generates reference data through `structbench-datagen`. A dataset is
a definition in a directory of its own, which may be private:

- `dataset.toml` — what the dataset is: identity, declaration, constants, the
  sampled box, splits, mesh levels (and their `symmetry`, `axisymmetric` or
  `planar`, which chooses the volume weights `converge` uses), the pilot
  cases and the probes the preflight runs on them, quantities of interest and
  their tolerances, ODB retention.
- `problem.py` — how one case is built and read: `input_deck`, `feasible`,
  `mesh`, `qoi`. It may import sibling modules of its own directory.

Start from the template and check it before anything runs:

    structbench-datagen new my_dataset
    structbench-datagen check my_dataset

Then the stages, in order: `preflight` (below), `generate`, `run` (with
`follow` alongside it), `export`, `convert`, `verify` (the ADR-0066 instrument
over the sweep), `converge`, `archive`. The shipped example
(`structbench/datagen/examples/abaqus_conformance`) is the single-rod
conformance case described in `docs/datagen/abaqus-conformance.md`. The design
is `docs/plans/2026-09-27-abaqus-datagen-platform-design.md`.

`verify` asks whether each run is internally sound (the ADR-0066 instrument).
Whether the *setup* reproduces a physical experiment is a different question,
answered by `structbench-validate` (ADR-0072): it measures canonical cases and
a shipped reference-experiment set with the same functions and writes a record
of the deviations — no threshold, no verdict. `structbench-validate --list`
names the sets; `src/structbench/validation/references/taylor_copper.md` is
the first one's provenance.

## The preflight

    structbench-datagen preflight --dataset <dir> --work-root <runs>
        [--abaqus EXE] [--workers N] [--timeout S]

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
| `space` | each quantity of interest at the production level is within its `[qoi].tolerance` of the value extrapolated from three levels (with no observed order: `review` when the quantity moves across the levels by no more than its tolerance, which a person may accept in `[pilot].accepted_reviews` as `space.<name>` with a reason, and `fail` when it moves more; the field errors are reported for the maintainer to weigh) |
| `increment` | each quantity of interest changes by less than its tolerance between the production increment and the scaled ones, and no energy row of those runs fails (the summary names the rows the instrument could not assess) |
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
never occurred. A value that is not finite never passes: it makes the step
`not_assessable` (or, in the solver's energy history, `fail`). **`generate` refuses a split that is not a probe
unless a passing stamp exists for the current `dataset.toml`, `problem.py`
and siblings**; `--no-preflight` generates anyway and writes
`"preflight": {"skipped": true}` into every such case's provenance, where a
stamp is otherwise recorded by its hash and date. Probe splits are never
gated. Because the preflight evaluates `qoi()` on every prefix of a
trajectory, `qoi()` must accept a one-frame case; `check` exercises that.

When only the judging changed — a tolerance, `frame_reported`,
`accepted_reviews`, `qoi()`, the declaration — `preflight --rejudge`
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
`min_free_gb`).

### The `[pilot]` fields

| Field | Default | Meaning |
|---|---|---|
| `split` | required | the probe split whose `points` are the pilots |
| `fine_cases` | required | plain case ids that also run at the finest level (three levels are needed for an extrapolation) |
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
keeps its older runs), extrapolates each quantity of interest from three
levels in constant ratio (Richardson, with the observed order and a status
that says when there is none), and measures every stored field of each
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
