# StructBench

**Verification and validation benchmarks for learned surrogates of
structural response.** Can a data-driven simulator be trusted? Reference
data with declared uncertainty, a fixed task and split, evaluation protocols
that test physical consistency alongside accuracy in physical units, and
reference baselines you can rerun — for structural response across loading
regimes, from quasi-static contact to impact and fracture.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![Release](https://img.shields.io/github/v/release/qilinli/StructBench)](https://github.com/qilinli/StructBench/releases)

> **Status: four benchmarks, four model families; cross-method comparison
> tables on Taylor, notch-impact, and DeformingPlate (wave-1D stays
> CGN-only). Reporting verification properties — constraint satisfaction,
> state sufficiency, closure, error growth with horizon — alongside accuracy
> is the next protocol step (ADR-0065).** What exists is real and tested;
> what doesn't is on the [roadmap](#roadmap).

![Taylor bar rollout: ground truth vs CGN prediction, copper bar mushrooming against a rigid wall, colored by von Mises stress](assets/taylor_rollout.gif)

*A 2D copper bar striking a rigid wall at 150 m/s — LS-DYNA SPH ground truth
(left) vs the CGN baseline's prediction (right), colored by von Mises stress.
See the [Taylor2D-Impact benchmark page](docs/benchmarks/taylor_impact_2d.md)
for the full problem, data, and registered results.*

## Benchmarks

| Benchmark | Problem | Cases | Data |
|---|---|---|---|
| Wave1D-Propagation | elastic wave in a bar (entry tier) | 16 | [Hugging Face](https://huggingface.co/datasets/StructBench/wave-propagation-1d) |
| Taylor2D-Impact | copper bar impact (SPH, plasticity) | 33 | [Hugging Face](https://huggingface.co/datasets/StructBench/taylor-impact-2d) |
| NotchBeam2D-Impact | notched concrete beam, drop-weight impact | 110 | [Hugging Face](https://huggingface.co/datasets/StructBench/notch-beam-2d-impact) |
| DeformingPlate | hyperelastic 3D plate + rigid actuator (MeshGraphNets, quasi-static) | 1200 | [public source](tools/ingest/meshgraphnets/deforming_plate/) |

Ordered by constitutive regime: linear elastic → elastoplastic → concrete
fracture → 3D hyperelastic contact.
Full cards (solver, materials, splits, QoIs): [docs/benchmarks.md](docs/benchmarks.md).
Every benchmark fixes its task, split, and evaluation protocol in an ADR —
changing any of them is a new benchmark version. The headline metric is a
pooled space+time **relative L2** — the convention the neural-operator
literature reports, so StructBench numbers read directly against published
tables — with physical-unit RMSE (mm, MPa) and engineering quantities of
interest retained alongside it (ADR-0055).

## Reference models

Four model families, implemented natively in `src/structbench/models/` (no
compiled graph extensions, no PhysicsNeMo runtime dependency), spanning the
two dominant simulator paradigms — all trained and scored under one pipeline
and one protocol:

| Family | Paradigm | Status |
|---|---|---|
| CGN — Concrete Graph Network (Li et al. 2023 lineage) | autoregressive graph network | **blessed** baseline: wave-1D, Taylor, notch-impact |
| MeshGraphNets (Pfaff et al., 2021) | autoregressive graph network | **blessed**: DeformingPlate — reproduces the published error band; provisional: Taylor, notch-impact |
| Transolver (Wu et al., 2024; + Transolver++ variant, off by default) | attention operator, time-conditioned (ADR-0054) | provisional: Taylor, notch-impact, DeformingPlate |
| GeoFLARE (NVIDIA, 2025) | attention operator, time-conditioned | provisional: DeformingPlate |

*Blessed* means the result passed its benchmark's acceptance bar and anchors
the results registry (ADR-0033/0046; the DeformingPlate
published-band gate: ADR-0043); *provisional* marks best-effort native ports
with no published number to reproduce (ADR-0046). Per-benchmark
method-comparison tables live on the generated benchmark pages
([docs/benchmarks/](docs/benchmarks/)).

## Why

Learned surrogates of structural simulation have become cheap to build and
remain expensive to trust. A model can reproduce a simulation to a few
percent and still emit stresses that are physically impossible, sensitivities
that point the wrong way, or rollouts that collapse outside the training
cases — and the reference data it learned from carries discretisation and
constitutive uncertainty of its own. Finite element analysis has verification
and validation standards for exactly this; learned models have none, and
benchmarks borrowed from machine learning measure accuracy alone.

The everyday symptoms are familiar to anyone who has trained on solver
output:

- **Every paper ships its own post-processing** — one-off scripts that pull
  just the fields that paper needed out of solver binaries, in whatever units
  the deck happened to use. The next project starts from zero.
- **Evaluations don't reproduce** — undocumented splits, normalized-unit
  metrics, and a different meaning of "rollout error" in every codebase.
- **The install is the first experiment that fails** — most GNS-style
  codebases need compiled graph extensions matched to your exact
  torch + CUDA + OS combination.

Speed is why surrogates are wanted — explicit solvers cost minutes to days
per run, and design sweeps, probabilistic assessment, and inverse problems
want thousands — but trust is what decides whether they get used. StructBench
exists so that *can this surrogate be trusted?* has a standard answer:
reproducible data with quantified uncertainty, evaluation protocols that
report physical consistency alongside accuracy, and reference baselines you
can rerun.

## Quickstart

```bash
git clone https://github.com/qilinli/StructBench
cd StructBench
pip install -e .
```

Installs from wheels on Linux, macOS, and Windows, CPU or CUDA. **No compiled
graph dependencies**: a native pure-torch `radius_graph` replaces
`torch-cluster`/`pyg-lib` (`torch_geometric` is used for `MessagePassing`
only) — no C++ build step, no CUDA-version matching dance. If you have fought
GNS codebases on a cluster or on Windows, you know why this matters.

```bash
# Train the CGN baseline (Concrete Graph Network, Li et al. 2023)
structbench-train --mode train --config configs/taylor_impact_2d/cgn.toml \
    --data-root /path/to/StructBench/canonical/taylor_impact_2d --out runs/taylor-cgn

# Validate, then roll out on the test splits (architecture is rebuilt from
# the run directory's own record — no --config needed, or accepted)
structbench-train --mode valid   --data-root /path/to/StructBench/canonical/taylor_impact_2d --out runs/taylor-cgn
structbench-train --mode rollout --data-root /path/to/StructBench/canonical/taylor_impact_2d --out runs/taylor-cgn
```

### Reproduce a registered result from its public checkpoint

The blessed checkpoints are public on Hugging Face, one model repository per
benchmark with one bundle per registered row (the checkpoint, its run record,
normalisation statistics and checksums). Two downloads and one command
reproduce a row; the smallest is wave-1D (a 6 MB bundle, 0.23 GB of data, a
few minutes on a CPU):

```bash
hf download StructBench/wave-propagation-1d --revision v0.4.0 --local-dir bundles
hf download StructBench/wave-propagation-1d --repo-type dataset --local-dir data/wave_propagation_1d
(cd bundles/cgn-48046ea && sha256sum -c SHA256SUMS)
structbench-train --mode rollout --out bundles/cgn-48046ea --data-root data/wave_propagation_1d
```

The command rebuilds the model from the bundle's own record, rolls out every
scored case and prints the row's metrics. Checked on 2026-11-11 from the
public files alone, on a CPU: displacement relative L2 0.3506 against the
registered 0.3507, axial stress 0.9004 against 0.9025. How closely a
re-evaluation should agree is stated in each landing page's results registry.

Where a benchmark declares a scored span (ADR-0078), training draws from that
span by default (`[train] training_span = "scored"`; `"full"` uses the whole
stored record) and a rollout stops at the span's end; add `--tail` to
`valid`/`rollout` to roll on to the last stored frame and report the frames
beyond the span under `unscored_tail`.

Configs are grouped per benchmark (ADR-0032): swap
`configs/taylor_impact_2d/cgn.toml` for `configs/wave_propagation_1d/cgn.toml`
or `configs/notch_beam_2d_impact/cgn.toml` to train against a different
benchmark — or swap the model family within a benchmark, e.g.
`configs/deforming_plate/{mgn,transolver,geoflare}.toml` for the 3D
DeformingPlate benchmark (ADR-0041; operator adaptations ADR-0044/0045).

## Repository layout

```
src/structbench/
  core/            # case schema, schema validation, HDF5 I/O, solver adapters
  datasets/        # canonical readers, windowing, normalization
  verification/    # reference-data verification: quantity catalogue, measures,
                   #   criteria, generated report (ADR-0066)
  validation/      # validation against experiments: reference sets, shared
                   #   measures, the deviation record; structbench-validate (ADR-0072)
  datagen/         # data generation from a dataset definition; structbench-datagen
                   #   (ADR-0071; the abaqus/ subpackage holds the deck writers and exporter)
  benchmarks/      # one module per benchmark: split + protocol + QoIs
  models/          # model families: cgn, mgn, transolver, geoflare (+ shared common/)
  eval/            # rollout driver, metrics
  viz/             # physics-quantity figures, FEM-postprocessor style (ADR-0022)
  cli/             # structbench-train, datacheck
configs/           # grouped TOML run configs, configs/<benchmark>/<family>.toml (ADR-0032)
tools/             # doc generation, the pooled-RMSE blessing aggregator, dev scripts;
                   #   hpc/ (cluster launch scripts, DUG SLURM),
                   #   ingest/ (converters for the datasets that predate datagen)
docs/              # benchmark cards, architecture, harness, corrections;
                   #   decisions/ (architecture decision records, ADRs),
                   #   datachecks/ (published verification records),
                   #   timelines/ (ground-truth evidence behind protocol values),
                   #   plans/ (designs, the verification source dossier, implementation plans)
tests/             # deterministic CPU-only test suite
assets/            # figures embedded in the docs + landing pages
```

## Roadmap

<!-- Living todo list (the single planning home; ROADMAP.md is retired).
     Conventions: done = [x] + strikethrough + (date); ad-hoc additions land
     in Inbox and get triaged into a milestone; when a milestone ships, its
     crossed-out block may be compressed to one line. Reasoning lives in
     docs/decisions/, not here. Platform work only: the maintainer's research
     plans stay outside the repository (ADR-0065). -->

*Last revised: 2026-11-11.*

### Shipped

- [x] ~~**v0.1** (2026-07-09, `v0.1.0`) — substrate proof: canonical schema +
      HDF5 I/O, LS-DYNA adapter, Taylor2D-Impact + blessed CGN baseline
      (ADRs 0019/0021/0033/0034).~~
- [x] ~~**v0.2** (2026-08-06, `v0.2.0`) — Wave1D-Propagation + the notch-beam
      pair (notch-bend since descoped, ADR-0056), with cards, grouped configs,
      and results registries (ADRs 0024–0039); CGN blessed on wave-1d and
      notch-impact; hosting = OneDrive-on-request (ADR-0040).~~
- [x] ~~**v0.3** (2026-08-27, `v0.3.0`) — `DeformingPlate` multi-method
      benchmark on public data (ADR-0041: cross-method comparison is the
      headline): blessed MGN reproducing the published result, Transolver +
      GeoFLARE (+ off-by-default Transolver++) provisional, ranked
      cross-method leaderboards, the prediction-scheme axis, relative-L2
      headline metric (ADRs 0041–0057).~~
- [x] ~~**v0.4** (2026-11-10, `v0.4.0`) — trust before accuracy (ADR-0065):
      a registered result reports its properties before its accuracy (the
      properties block, ADR-0075; 21 of the 24 rows carry numbers), the
      restart family as a declaration surface, the gradient family with its
      first rows (ADR-0076), open evaluation (ADR-0074: coverage on every
      scored case, unranked results registries in two tiers,
      `CONTRIBUTING.md`), the card's V&V fields and compliance table,
      reference-data verification (ADR-0066), the data-generation platform
      (ADRs 0068–0071), validation against experiments (ADR-0072), and the
      blessed checkpoints public on Hugging Face (ADRs 0058–0076).~~

### Next release — v0.5, the citable platform

*Nothing new: what v0.4 claims is made true and stable, and the release is
the first archived with a DOI. The arc from here to v1.0, with each release's
closing criteria, is
[`docs/plans/2026-11-11-release-outline-to-v1.md`](docs/plans/2026-11-11-release-outline-to-v1.md);
a release's criteria become its checklist here when it is picked up.*

- [x] ~~`docs/VISION.md` carries the ADR-0065 identity (maintainer, out of
      session, 2026-11-12).~~
- [x] ~~`CITATION.cff`, the BibTeX entry below and the dataset cards' BibTeX
      carry the ADR-0065 identity (2026-11-11).~~
- [x] ~~The tree is fit to archive (2026-11-11): the source dossier is an
      index of claims, sources and locators without the sources' own
      sentences; no record names a dataset that is not public.~~
- [x] ~~Every registered row reports its properties or states why not
      (2026-11-11: the three rows without properties say why in their
      notes; re-running them is in the Inbox).~~
- [x] ~~The repeatability tolerance of an evaluation is declared
      (2026-11-11: on every landing page's results registry — four
      significant figures, 2% for the graph-network families on a GPU, the
      maintainer's decision of 2026-11-10).~~
- [x] ~~The gradient family has rows from a second model family
      (2026-11-11: the Taylor autoregressive Transolver; the time-conditioned
      rows wait on the protocol gap in the Inbox).~~
- [x] ~~Per-case coverage figures on the landing pages (2026-11-12: the
      headline by coverage bin on Taylor, notch-impact and wave-1D, 16 of 18
      rows; DeformingPlate declares no case parameters and has none; the
      two rows without values are in the Inbox).~~
- [x] ~~A registered row reproduced from its public checkpoint by the
      documented command alone (2026-11-11: wave-1D CGN, displacement
      relative L2 0.3506 against 0.3507 registered; the procedure is in
      Quickstart).~~
- [ ] The release archived with a DOI.

### In progress

- **Reference-data verification** (ADR-0066) — `structbench.verification`
  measures a run against a catalogue of numerical-health, conservation,
  constitutive, units and integrity quantities, and judges the measurements
  against platform criteria. Published records:
  [Taylor](docs/datachecks/taylor_impact_2d.md),
  [notch-impact](docs/datachecks/notch_beam_2d_impact.md).
  - [ ] One conformance run with the standard LS-DYNA input block
        ([`docs/datagen/lsdyna-conformance.md`](docs/datagen/lsdyna-conformance.md)),
        and the two-grid difference on the convergence case.
  - [ ] The kinetic-energy closure's tolerance and the remaining
        sampling-clock rows; E6, E7 and the remaining E9 rows, which wait on
        a run that writes their files.
- **Data generation platform** (ADR-0068/0069/0071, Accepted) —
  `structbench-datagen`: the Abaqus pipeline as a package capability, with a
  preflight gate before production. Guide:
  [`docs/DATA_GENERATION.md`](docs/DATA_GENERATION.md).
  - [ ] The card generator and the full guide (the energy-gain indicator
        stays unratified by decision, ADR-0073 D9).
  - [ ] Minors deferred from part one's whole-branch review (moved here from
        CLAUDE.md 2026-09-29, not re-checked): `check` nests each level
        against the first only · the QoI-name check is order-sensitive ·
        `[pilot].fine_cases` and `accepted_gaps` are not checked against
        planned case ids or catalogue rows · `new .` derives an empty name ·
        `[declaration]` is validated only by `verify`, not by `check` ·
        `provenance.json` is written CRLF on Windows · a stale module path in
        `core/io/abaqus.py` · an old collect command in the conformance
        document, and `sampling.py`'s docstring still says `sweep.toml` · a
        missing scipy raises an ImportError instead of an install hint, and
        exit codes 2 and 3 are used inconsistently across stages · the
        private dataset's `problem.py` still carries literal keyword text for
        probe splits.
- **Validation against experiments** (ADR-0072, Accepted 2026-10-01) — built
  2026-09-27: `structbench.validation`, `structbench-validate`, and the first
  reference set, `taylor_copper`. Reports deviations, never a verdict.
  - [ ] A `BenchmarkCard` field and landing-page link, with the first public
        record.
  - [ ] An LS-DYNA Taylor record, which needs runs at the experiments'
        conditions (maintainer's call).
  - [ ] A figure in the reference companion.
  - [ ] A `/2` record format for time-resolved measurements.
- **Open evaluation** (ADR-0074, Accepted 2026-11-08) — two scored sets, `test`
  (public outputs) and `blind` (outputs withheld by the platform), each case read
  by coverage computed from its parameters (within the declared box or beyond
  it, distance to the nearest training point); external results in two tiers,
  self-reported and verified, in a results registry without a rank.
  - [ ] The first blind set and the first external result — with the first
        contract dataset admitted (maintainer, 2026-11-09).
- **The decision log** (ADR-0073, 2026-10-01) — the maintainer's review of
  all 71 records; its decisions are applied.
  - [ ] Still from the review: stale text in the older records (the
        maintainer's triage register names them).

### Inbox — untriaged, add freely

<!-- Completed inbox items are removed at each release; their record lives in
     git history and the ADRs they cite. -->

- [ ] ADR-0075, the two-site state — owed: the AR scheme and the
      pushforward chains under the two-site decoder; the final form of the
      admissible arm (`admissible_coords` or `projected_trial`, chosen by
      their results; the `return_map` decoder does not train on a metal, ADR-0075
      note 2026-11-08); the first restart declaration, with the first
      admitted contract dataset.
- [ ] Three registered rows report no properties (2026-11-10): Taylor MGN
      and notch-impact MGN did not reproduce their registered relative L2
      within 2%, and the DeformingPlate GeoFLARE time-conditioned run was
      not found, so that row is marked unverified.
- [ ] GPU rollouts of the graph-network families (CGN, MGN) are not bitwise
      repeatable (CUDA scatter-add): two evaluations of one checkpoint differ
      by up to about 1.5% in rollout relative L2. Deterministic kernels in
      the evaluation path are untested.
- [x] ~~The gradient family has one registered model (Taylor CGN); a second
      family's rows would show whether it separates models (2026-11-11: it
      does — the Taylor autoregressive Transolver reads relative L2 0.38 and
      0.47 with every sign right, against CGN's 2.5 and 2.4).~~
- [ ] Two rows carry no per-case values (2026-11-12). The notch-impact
      CGN row: its run is not in the local archive with per-case relative
      L2 (the Taylor CGN row was filled the same day from a CPU
      re-evaluation of the registered checkpoint, within 0.5% of the
      registered headline on both sets). The notch-impact GeoFLARE row: a CPU
      re-evaluation of the registered checkpoint read 0.05324 on
      `test_interp` against the registered 0.05257 (1.3% apart), outside
      the four figures the registry states for that family, so its values
      were not entered. Diagnosed the same day: same checkpoint and scored
      span as the archived GPU evaluation; one-step error agrees to 0.1%
      and rollout position RMSE differs by up to 7% on a case (0.8% in the
      mean), while the Transolver checkpoint on the same benchmark agrees
      to seven figures and GeoFLARE on Taylor to four. The likely source (not isolated) is
      GeoFLARE's capped neighbour queries on the particle set, amplified
      by the rollout. The repeatability statement names four figures for
      this family; whether it joins the graph-network families at 2% on
      particle benchmarks is the maintainer's.
- [ ] The gradient family does not reach a time-conditioned model
      (2026-11-11): the protocol perturbs the seed frames, and a
      time-conditioned model takes its loading scalar from the case's nominal
      value, so its sensitivity reads exactly zero (relative L2 1, sign
      agreement 0) whatever the model has learned. Measured on the registered
      Taylor time-conditioned Transolver; the row is not registered. The
      protocol has to perturb the loading scalar with the seed (ADR-0076
      clause 2).
- [ ] Time-conditioned evaluation takes its time reference from each case's
      own length and training from the shortest training trajectory; they
      differ on trajectories of unequal length unless the run records its
      reference.
- [ ] Registry upkeep: `run_commit` of the Taylor Transolver and GeoFLARE
      autoregressive rows (registered `7ef3bf2`, run records `6d65193`); the
      five-decimal table format on property values of very different
      magnitude; the private archive bundles of the notch-impact CGN and
      DeformingPlate MGN, never assembled (ADR-0037).

### Later (each becomes an ADR/spec when picked up)

- **Benchmarks to add** — four, each becoming its own ADR when picked up:
  - *A component-tier benchmark* — the first above specimen scale (every
    shipped card reads `structural scale: specimen`). ADR-0065 follow-up 4
    named it as metal and for v0.4; the maintainer parked it on 2026-11-08:
    the case for it — why a component, why metal, why before the gradient
    benchmark has shown what the properties block can say — is not made, and
    it enters a release only once it is. If picked up, it would be generated
    through `structbench-datagen` so it meets the data standard at admission
    (ADR-0073 D14) with a blind set from the start (ADR-0074)
  - *RC beam bending test* — flexural loading, not the impact regime the
    parked RC beam item assumed (erosion was that one's gate, ADR-0024/0041);
    a bending test may not need erosion at all, which is the first thing to
    settle
  - *Crash — an assembly-tier question* (ADR-0065 follow-up 4, 2026-11-08;
    was "v0.4 candidate") — the question is not whether public crash data
    exists but whether the platform can say anything trustworthy about an
    assembly: the properties block reporting on the shipped benchmarks comes
    first, and the scale infrastructure (cell-list `radius_graph`, TB-scale
    hosting) and the data (CarCrashNet's release, or open-licence LS-DYNA
    data generated through the pipeline) remain its gates; its methods
    already ship (ADR-0041)
  - *Segmented beam* — parked since ADR-0015
- **An agent driving data generation** — `structbench-datagen` (In
  progress, above) is the pipeline; what remains is an agent that runs it end
  to end for a dataset definition — preflight, production, verification,
  convergence, archive — and reports what the records say, so that a
  contributor can repeat a dataset without the maintainer's hands on it
- **Surrogates inside engineering workflows (long run)** — once a
  surrogate's trust can be established, placing it in the workflows that
  need it: protective-structure design and optimisation, and structural
  health monitoring, where real-time sensor data calibrates the surrogate
  against the structure it stands for. The case schema's `asset_id` already
  links a case to a physical structure (ADR-0011); nothing else is reserved
  for this in the package, and each piece becomes an ADR when picked up
- Training: resume support · part-id→embedding remap · ADR-0028 Phase-2
  ablations
- **Surrogate V&V as a reported axis** — a prediction is judged on three
  independent questions, not one ranking: accuracy against the reference,
  physics-consistency (does it violate laws it cannot legitimately violate,
  however close it is), and fitness for a downstream use (design
  optimisation, digital twin). ADR-0075 decided the first piece, the
  properties block, and its part 3c (the block in the registries and landing
  pages) and part 3b (the restart family, with the semigroup row) are built
  (2026-11-09), with numbers on 21 of the 24 shipped rows (2026-11-10); see
  ADR-0065's 2026-09-23 note for why the three do not collapse
- Eval: submission validator for the results registry · per-region probe metrics ·
  convergence check · cross-benchmark utilities
- Data & scale: second aux target (plastic
  strain) · cell-list `radius_graph` when any
  ≥10⁶-node dataset lands · other solvers (Kratos, OpenSees, OpenRadioss) ·
  SHM expansion · deployment tools · packaging extras · PhysicsNeMo interop

Rationale for every item lives in [`docs/decisions/`](docs/decisions/).

## Citation

If you use StructBench, please cite it
([CITATION.cff](CITATION.cff)):

```bibtex
@software{li_structbench_2026,
  author  = {Li, Qilin},
  title   = {{StructBench}: Verification and validation benchmarks for
             learned surrogates of structural response},
  year    = {2026},
  version = {0.4.0},
  license = {Apache-2.0},
  url     = {https://github.com/qilinli/StructBench}
}
```

## License

[Apache 2.0](LICENSE). The Transolver family's network module derives
from two MIT-licensed reference implementations (Transolver and Transolver++,
THUML @ Tsinghua University) and carries their copyright and permission
notices in its header (ADR-0073 D4).
