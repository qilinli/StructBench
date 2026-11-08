# Plan: open evaluation (ADR-0074)

*The one plan ADR-0074's Consequences call for. Written 2026-11-08 before
the work. The record holds the judgments (two scored sets, computed
coverage, two tiers without a rank, publication whole); this plan fixes the
mechanics the record delegated: the prediction format, set sizes, the
rotation cadence, the rate limit, the coverage kernel, the contract's
`withheld` splits and `grid` shorthand, the container contract, the steward
processes, the registry fields and their rendering, and `CONTRIBUTING.md`.
Where a number below is a choice rather than a consequence, it says so and
why; changing one is a dated note here, not a record. Card and registry
fields are additive and flag-first (ADR-0074 Consequences): the commit that
adds them names the public-API change. Platform implementation only.*

## 1. Sets

Public names on every benchmark: `train`, `val`, `test`, `blind`
(ADR-0074 clause 2). Legacy names (`test_interp`, `test_extrap`, `probe`)
stand in `cases.csv` and the loaders and map for rendering: every split
that is neither `train` nor `val` is a *scored set*, and the page labels it
by its coverage, not its name.

**Platform-generated datasets** (ADR-0074 clause 3; the dataset contract,
ADR-0071):

- `train` — the author's design; the contract's Sobol split with its
  declared `n` and seed. The card's `parameter_space.box` is the sampled
  box of this split.
- `test` — a small full factor grid inside the box, two levels per numeric
  parameter at 25 % and 75 % of each range (`2^k` cases; three levels,
  `3^k`, where `k ≤ 3`), declared with the `grid` shorthand (§6). Public
  outputs. The same cases are the preflight's pilot anchors where the
  dataset chooses so, which is what "anchors the data standard's
  verification runs" means: one set of cases carries both the public score
  and the convergence evidence.
- `blind` — random: a scrambled Sobol split with a seed the platform holds
  and never publishes, drawn inside the box extended by the declared
  `margin` (§3), filtered to lie off every training point by at least half
  the training set's median nearest-neighbour spacing (§4) so no blind case
  is a near-duplicate of a public one. Its reference outputs are
  `withheld` (§6).

**Sizes** (a choice): `test` is the grid; `blind` is 32 cases for a first
set, or 10 % of `train` when that is larger, rounded up to a power of two.
Why: 32 is enough to bin by coverage (within/beyond × three distance bins,
§4) with several cases per bin, and small enough that regenerating a blind
set costs a day of solver time, not a week.

**Legacy LS-DYNA benchmarks** get a blind set when solver access allows
(ADR-0074 Consequences): off-grid velocities (and widths, for notch-impact)
inside the box plus a margin's worth beyond, 12 cases each as a first set,
produced through the same contract once the LS-DYNA decks are a dataset
definition. DeformingPlate never (no generator).

## 2. Blind tags, rotation and cadence

A blind set carries a tag, `blind-v1`, `blind-v2`, … on the card; it never
bumps the benchmark version (ADR-0074 clause 1). Scores keep the tag they
were scored under. **Cadence** (a choice): a new tag at most once a year, cut
with the first code release after twelve months of the current tag, or at
once on suspected leakage (a result closer than the noise floor that the
submitter cannot reproduce in a container, or the withheld outputs found
in public). On rotation the retired set's outputs are published and its
cases join `test` with a note saying which tag they were blind under.

## 3. Margin

Each benchmark declares a `margin` per numeric parameter as a fraction of
its box width (card `parameter_space.margin`; contract `[splits.blind].
margin`), default 0.1 (a choice: one grid step of a ten-level design, far
enough to be *beyond* and near enough to be a design someone would run).
Blind cases are drawn in the box extended by the margin on both sides where
physically admissible (the dataset's `feasible` limit still applies), so a
blind set reads both within and beyond coverage by construction. A margin
of 0 means an interior-only blind set, declared as such.

## 4. The coverage kernel

Per scored case, from its parameters (ADR-0074 clause 2); built as
`eval/coverage.py`, pure functions over arrays, in this plan's first task:

- **Coordinates.** The numeric parameters of the training design, each
  divided by its box width, so the box is the unit cube. A parameter the
  design holds fixed (zero width) is dropped from the distance; a scored
  case that moves it is *beyond*.
- **Within / beyond.** *Within* when every numeric parameter lies in its
  box, inclusive, to a relative tolerance of 1e-9; else *beyond*.
- **Distance.** The Euclidean distance in unit-cube coordinates from the
  case to its nearest training point, divided by the training set's median
  nearest-neighbour distance in the same coordinates. On a full grid with
  equal normalised steps this is the distance in steps; on Taylor (velocity
  step 1/9 of its box, length step 1/2) the unit is the velocity step, an
  interior test velocity reads 1.0, the 200 m/s extrapolation reads 1.0 and
  *beyond*, and a 70 mm bar at a training velocity's midpoint would read
  2.5 (a quarter of the length range, in velocity steps).
- **Bins** (a choice, for rendering): `≤ 1`, `1–2`, `> 2` units, crossed
  with within/beyond: six bins, labelled "within, ≤ 1" and so on. The
  headline is pooled per set per bin; an empty bin renders nothing.
- **Novelty.** An unseen categorical level or a changed constant is
  declared per case in the definition (`novelty = true` on a split, or per
  point) and in `cases.csv`; it is not computed. A novel case is shown in
  its own column, never pooled with the bins.

Parameters per case come from the benchmark: `BenchmarkSpec.case_parameters`
(a callable from case id to the numeric and categorical parameters; the
shipped benchmarks' id parsers move into the package from
`tools/build_hf_bundle.py`, where they already exist) or, for a pipeline
dataset, the definition's points.

## 5. Scoring by submission

**Prediction format.** One canonical-format HDF5 per blind case (the case
schema, ADR-0012/0013): `metadata.case_id` and `schema_version`,
`response/time/t` over the scored span `[input_frames, T)`, and the scored
fields — `response/node/displacement` and the benchmark's aux field(s) at
the stored site (nodal; or the element block where the benchmark declares
an element site, ADR-0075) — in strict SI, float32, the same shapes as the
reference. Nothing else is read. A `submission.toml` beside them carries
the provenance ADR-0074 clause 4 lists: `method` (label), `family`,
`scheme`, `division` (`closed` | `open`), `commit`, `config_sha256`,
`seeds`, `checkpoint_sha256`, `writeup` (URL or DOI), `data` (the open
division's declared training data), and for the verified tier the
`container` digest (§7). The platform computes the metrics (ADR-0055 forms
and the QoIs), the properties block where the benchmark defines it
(ADR-0075 part 3c), and the coverage bins; it returns scores only.

**Rate limit** (a choice): three submissions per method per blind tag per
thirty days, a method being the `(submitter, method)` pair; a fourth is
held until the window passes. Why: enough to fix a packaging mistake, too
few to tune on the blind set.

**Regeneration guard** (ADR-0074 clause 5): where the benchmark's card
records a measured noise floor for a scored field (the compliance row
`noise_floor` is `pass` and `ground_truth.uncertainty` names the floor),
a submission whose pooled relative L2 on that field is below the floor is
flagged, the scores are returned with the flag, and the steward may ask for
a container before the result enters the registry. The guard reads no
properties (reference baselines satisfy them by construction, ADR-0064).
Taylor today: von Mises relative L2 below 0.12 is flagged; displacement has
no measured floor and is not guarded.

## 6. The dataset contract

Two additions to `dataset.toml` (ADR-0071 says the contract changes by
ADR; ADR-0074 made the decision and this plan the shape):

- `[splits.<name>] withheld = true` — the split's reference outputs are
  not exported to the public archive. `archive` writes each case with its
  inputs only: metadata, nodes, elements, materials, the solver input and
  the seed frames `[0, input_frames)` of every response field; the full
  cases go to the private archive under the blind tag. `check` refuses
  `withheld` on `train` or `val`.
- `[splits.<name>] grid = { param = [levels, …], … }` — the split is the
  full factor grid of the listed levels (the Cartesian product, in list
  order); it takes no `n`, `seed` or `points`. Categorical levels are
  allowed. `check` refuses a grid over more than 64 cases.

`cases.csv` gains the columns `coverage` (`within` | `beyond` | `novel`
for scored cases, blank for train and val), `distance` and `blind_tag`;
parameter columns stay as they are.

## 7. Tiers and the container contract

*Self-reported* by default. *Verified* when the platform executed the
submitter's container (ADR-0074 clause 6): an OCI image whose entry point
is `structbench-submit predict --cases <dir> --out <dir>`, reading the
published blind inputs from `<dir>` and writing predictions in the format
of §5 to `<out>`, with no network access, within a budget of one GPU-hour
or twenty-four CPU-hours per benchmark (a choice; the blessed CGN rolls the
Taylor test sets out in minutes on one GPU). The result is verified when
the container's blind outputs reproduce the submitted ones to a pooled
relative L2 of 1e-3 per field and its public-test metrics reproduce the
submitted ones to the same tolerance; otherwise it stays self-reported with
a note. Where a benchmark has no blind set yet, the verified tier runs the
public protocol alone and the row says so (clause 6). *Blessed* remains the
maintainer's bar for platform-run reference baselines (ADR-0046).

## 8. Registry fields and rendering

`BaselineResult` (additive, flag-first; `tier` and `division` default to
the platform-run values so every existing row is unchanged):

| field | values | default |
|---|---|---|
| `tier` | `platform` (the maintainer ran it; blessed or provisional as today), `verified`, `self_reported` | `platform` |
| `division` | `closed`, `open` | `closed` |
| `blind_tag` | the tag scored under, or empty | `""` |
| `submitter` | who submitted an external result | `""` |
| `writeup` | URL or DOI | `""` |
| `data` | the open division's declared training data | `""` |
| `case_metrics` | case id → metric → value, for the scored sets | `{}` |

`metrics` keeps its split-keyed shape; a `blind` key is valid once the
benchmark declares a `blind` split (its published inputs are case ids like
any other). The renderer (ADR-0074 clause 7, with ADR-0075 part 3c deciding
the order against the properties block): per benchmark, one *results
registry* section in registration order, no rank; the properties block
first where reported; the headline relative L2 per set per coverage bin,
computed from `case_metrics` and the coverage kernel at render time;
then tier, division, blind tag and provenance; everything else collapsed as
a diagnostic. The word *leaderboard* leaves the renderer and the pages in
the same change (the wording sweep).

## 9. Steward processes

Automated where a process can be, on explicit instruction where the
decision log says so (ADR-0074 clause 9; ADR-0023):

1. **Intake** — a submission is a pull request to a `submissions/<benchmark>/
   <method>/` folder holding `submission.toml` and the prediction files, or
   an archive link when the files are large; a template is in
   `CONTRIBUTING.md`. Automated: format check, rate limit, provenance
   completeness.
2. **Scoring** — `tools/score_submission.py` reads the withheld outputs from
   the private archive, computes the metrics, the properties block and the
   coverage bins, runs the regeneration guard, and writes `scores.json`
   back to the submission folder. Automated; the withheld outputs never
   leave the maintainer's machine or DUG.
3. **Flagging** — the guard and the rate limit write their flags into
   `scores.json`; a flagged result waits for triage.
4. **Triage** — the maintainer reads flags, asks for a container where the
   guard fired, and runs the verified tier on DUG (`tools/verify_submission.py`
   drives the container and compares). Human.
5. **Registration** — a result enters `results.py` by a commit the
   maintainer instructs in session, with its tier, division and tag; the
   pages regenerate. On explicit instruction (ADR-0023).
6. **Rotation** — a new blind tag per §2; the retired set is published; the
   card, `cases.csv` and the archive update together. On explicit
   instruction.

## 10. `CONTRIBUTING.md`

Sections, in this order: what a submission is and the two divisions; the
prediction format and `submission.toml` (§5); the tiers and the container
contract (§7); the admission list — every contract-admitted benchmark with
its sets, blind tag, margin and noise floors — generated from the cards so
it cannot drift; challenges (a time-boxed event may rank for its duration
and names its own rules; its results enter the registry unranked after);
and the rule that a result changing task, split or protocol is another
benchmark (ADR-0074 Consequences), with the path to propose one.

## Not in this plan

The properties block's kernels and order (ADR-0075 plan 3c); the gradient
family's rows (ADR-0076: `gradient_*` keys ride in `metrics` like any
other); the first blind set's generation (an Abaqus dataset at admission,
the maintainer's); hosting of the private archive beyond the maintainer's
storage and DUG (ADR-0040).

## Build plan

1. **Coverage kernel** — `eval/coverage.py`: `Coverage`, `coverage()`,
   the bins; tests on the Taylor grid (the numbers in §4), a Sobol cloud,
   a fixed axis, inclusive edges. *Built with this plan.*
2. **Case parameters** — `BenchmarkSpec.case_parameters` from the id
   parsers in `tools/build_hf_bundle.py`; the four benchmarks declare it;
   `cases.csv` gains `coverage` and `distance`; the Taylor page shows the
   scored sets' coverage.
3. **Registry fields and rendering** — the table of §8, the `blind` split
   declaration, the results-registry renderer with bins and tiers, the
   wording sweep; pages regenerated; drift tests.
4. **Contract** — `withheld` and `grid` in `dataset.toml`, `check` rules,
   `archive` writing inputs only for withheld splits, `cases.csv` columns;
   the template documents both; `docs/DATA_GENERATION.md`.
5. **Scoring and the guard** — `tools/score_submission.py`, the prediction
   reader, the guard against the card's floor, `scores.json`; tests on a
   synthetic submission.
6. **Container contract** — `tools/verify_submission.py` and the
   reproduction comparison; documented, exercised on the blessed CGN as a
   container when the maintainer builds one.
7. **`CONTRIBUTING.md`** and the intake template; the admission list
   generated from the cards.

Gate for each task: full suite green; no existing registered row or page
changes except where the task says so.
