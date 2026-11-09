# Contributing a result to StructBench

StructBench scores learned surrogates of structural response on benchmark
problems with reference data of declared uncertainty, and records the results
in a per-benchmark *results registry*: in registration order, never ranked,
each result read beside the coverage of the cases it was scored on and, where
reported, its physics properties (ADR-0065, ADR-0074). This page is how an
external result enters that registry. The mechanics it describes are fixed by
the open-evaluation plan (`docs/plans/2026-11-08-open-evaluation-plan.md`);
the judgments behind them are ADR-0074.

## What a submission is

A submission is a set of predictions for the cases of one benchmark's scored
sets, with the provenance that lets a reader weigh it. Every benchmark has
two kinds of scored set: `test`, whose reference outputs are public, and
`blind`, whose inputs are public and whose reference outputs the platform
withholds (the legacy names `test_interp`, `test_extrap` and `probe` are
`test` sets). The platform scores the predictions, computes the coverage of
every case from its design parameters, and returns the scores.

Two divisions:

- **closed** — the model was trained on the benchmark's published `train`
  set and nothing else;
- **open** — any data, declared in the submission.

Three tiers, which the registry shows beside every result:

- **platform** — the maintainer ran it (the blessed and provisional reference
  baselines);
- **self-reported** — the default for an external result;
- **verified** — the platform executed the submitter's container and it
  reproduced the submitted predictions (below).

## The prediction format

One canonical-format HDF5 file per predicted case, named `<case_id>.h5`,
built the way the published case is (the case schema, ADR-0012/0013; the
`structbench.core` writer) and holding a `response` over the **scored span
of the working trajectory**: frames `[input_frames, T')` of the case as
`structbench.datasets.load_case_trajectory` reads it, with `response/time/t`
equal to those times. It carries:

- `response/node/displacement`, `(n, N, d)` in metres, every node in the
  published order;
- the benchmark's auxiliary field as a scalar per particle in SI units —
  `response/element/sph/<aux_field>` `(n, P)` for an SPH benchmark,
  `response/node/<aux_field>` `(n, N, 1)` for a mesh benchmark.

Nothing else is read. `input_frames`, `aux_field` and the scored horizon are
on each benchmark's page. Beside the files, `submission.toml`:

```toml
method = "MyModel"                 # the registry label
family = "transolver"              # the model family, or "other"
scheme = "time-conditioned"        # autoregressive | time-conditioned | one-shot | ...
division = "closed"                # closed | open
submitter = "Name, affiliation"
commit = "abc1234"                 # the code the predictions came from
config_sha256 = "..."              # of the run configuration
seeds = [1]
checkpoint_sha256 = "..."
writeup = "https://..."            # a paper, report or DOI
# data = "..."                     # the open division's training data
# container = "ghcr.io/you/model@sha256:..."   # for the verified tier
```

Scoring is `tools/score_submission.py`; it reports the registry's metrics
per case (pooled relative L2 of displacement and of the auxiliary field,
RMSEs, the quantities of interest), the mean per scored set and per coverage
bin, and any flags. Two flags exist. A set whose mean relative L2 on a field
lies below the floor the benchmark card measured for that field is flagged
as closer to the reference than its own noise (ADR-0074 clause 5); the
steward may ask for a container before the result is registered. A fourth
submission of the same method to the same blind tag within thirty days is
held until the window passes.

## The verified tier and the container contract

A result becomes *verified* when the platform runs the submitter's container
and it reproduces the submission. The container is an OCI image whose entry
point is

```bash
structbench-submit predict --cases /cases --out /out
```

reading the published inputs from `/cases` (mounted read-only), writing
prediction files in the format above to `/out`, with no network access, and
finishing within the budget the steward grants (one GPU-hour or twenty-four
CPU-hours per benchmark). The result is verified when the container's
predictions reproduce the submitted ones to a pooled relative L2 of 1e-3 per
response field on every case and its public-set metrics reproduce the
submitted ones to the same tolerance (`tools/verify_submission.py`).
Otherwise the result stays self-reported, with a note saying where it
departed. *Blessed* is a different thing: the maintainer's bar for the
platform's own reference baselines (ADR-0046).

## How to submit

1. Open a pull request adding `submissions/<benchmark>/<method>/` with
   `submission.toml` and the prediction files (or a link to an archive of
   them when they are large).
2. The platform checks the format, the provenance and the rate limit, scores
   the predictions against the withheld outputs, and writes `scores.json`
   back to the folder. The withheld outputs never leave the steward's
   machine.
3. Flagged results wait for the steward; a container request follows a floor
   flag.
4. The maintainer registers the result in the benchmark's results registry
   with its tier, division and blind tag, and the pages regenerate. Rows
   enter on the maintainer's explicit instruction (ADR-0023).

## Admission list

Every registered benchmark, with what a submitter needs to know: the scored
sets read by their coverage (within or beyond the training design's box, and
the distance to the nearest training point in units of the design's spacing),
whether a blind set exists, the margin blind cases may lie beyond the box,
the measured noise floors the guard reads, and how far the reference data
meets the data standard. The table is generated from the benchmark cards.

<!-- admission-list:start (generated by tools/gen_benchmark_docs.py from the benchmark cards; do not edit by hand) -->

| Benchmark | Scored sets (coverage) | Blind set | Margin | Measured floors (relative L2) | Standard (ADR-0073 D14) |
|---|---|---|---|---|---|
| [Wave1D-Propagation](docs/benchmarks/wave_propagation_1d.md) | `test_interp` (within, 1–2: 1, within, ≤ 1: 1) | missing | none declared | none measured | 0 of 4 required items; grandfathered |
| [Taylor2D-Impact](docs/benchmarks/taylor_impact_2d.md) | `test_interp` (within, ≤ 1: 6); `test_extrap` (beyond, ≤ 1: 3) | missing | none declared | `von_mises_stress` 0.12 | 2 of 4 required items; grandfathered |
| [NotchBeam2D-Impact](docs/benchmarks/notch_beam_2d_impact.md) | `test_interp` (within, ≤ 1: 12); `probe` (novel: 2) | missing | none declared | none measured | 0 of 4 required items; grandfathered |
| [DeformingPlate](docs/benchmarks/deforming_plate.md) | `test` — no coverage: the design's parameters are not published | never | none declared | none measured | 0 of 4 required items; grandfathered |
<!-- admission-list:end -->

A blind set reads *missing* until solver access produces one (the LS-DYNA
benchmarks), *never* where no generator exists (DeformingPlate), and *pass*
with its tag once published. A blind tag never bumps a benchmark version;
when a tag rotates, the retired set's outputs are published and its cases
join `test`, and scores keep the tag they were scored under.

## Challenges

A challenge is a time-boxed event on one or more benchmarks that may rank
its entries for its duration and names its own rules. When it ends, its
results enter the registries unranked, like any other result.

## A different task is a different benchmark

A result that changes the task, the split or the protocol of a benchmark —
another horizon, another seed length, another scored set — is not a result
on that benchmark. Propose it as a new benchmark: a decision record under
`docs/decisions/` (see that folder's README for what earns one and the
format), or, for a dataset generated through `structbench-datagen`, a
dataset definition that meets the data standard (ADR-0073 D14) and is
admitted whole (ADR-0074 clause 8).

## Contributing code

Code contributions follow `docs/PRINCIPLES.md` (conventions), `docs/
ARCHITECTURE.md` (layering) and the decision log. Open a pull request on a
feature branch; the full test suite is the gate. The maintainer merges.
