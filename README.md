# StructBench

**Standardized benchmarks for machine learning on structural simulation.**
A task definition, a fixed split, metrics in physical units, and a reference
baseline to beat — for structures under dynamic and extreme loading.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

> **Status: pre-release (v0.1 imminent).** What exists is real and tested;
> what doesn't is on the [roadmap](#roadmap).

![Taylor bar rollout: ground truth vs CGN prediction, copper bar mushrooming against a rigid wall, colored by von Mises stress](assets/taylor_rollout.gif)

*A 2D copper bar striking a rigid wall at 150 m/s — LS-DYNA SPH ground truth
(left) vs the CGN baseline's prediction (right), colored by von Mises stress.
See the [Taylor2D-Impact benchmark page](docs/benchmarks/taylor_impact_2d.md)
for the full problem, data, and numbers to beat.*

## Benchmarks

| Benchmark | Problem | Cases |
|---|---|---|
| Taylor2D-Impact | copper bar impact (SPH, plasticity) | 33 |
| Wave1D-Propagation | elastic wave in a bar (entry tier) | 16 |
| NotchBeam2D-Bend | notched concrete beam, 3-point bend | 111 |
| NotchBeam2D-Impact | notched concrete beam, drop-weight impact | 110 |

Full cards (solver, materials, splits, QoIs): [docs/benchmarks.md](docs/benchmarks.md).
Every benchmark fixes its task, split, and evaluation protocol in an ADR —
changing any of them is a new benchmark version — and all metrics are
reported in physical units (mm, MPa), never dimensionless scores.

## Why

If you have trained ML surrogates on structural simulation data, you know the
routine:

- **Every paper ships its own post-processing** — one-off scripts that pull
  just the fields that paper needed out of solver binaries, in whatever units
  the deck happened to use. The next project starts from zero.
- **Evaluations don't reproduce** — undocumented splits, normalized-unit
  metrics, and a different meaning of "rollout error" in every codebase.
- **The install is the first experiment that fails** — most GNS-style
  codebases need compiled graph extensions matched to your exact
  torch + CUDA + OS combination.

Underneath sits a question the field keeps circling: *can a learned simulator
reproduce the full elasto-plastic response of a structure under impact, fast
enough to be useful?* Explicit solvers cost minutes to days per run; design
sweeps, probabilistic assessment, and inverse problems want thousands of runs.
StructBench exists so answers to that question can be compared: standardized
benchmarks, honest evaluation, reference baselines you can rerun.

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

Configs are grouped per benchmark (ADR-0032): swap
`configs/taylor_impact_2d/cgn.toml` for `configs/wave_propagation_1d/cgn.toml`,
`configs/notch_beam_2d_bend/cgn.toml`, or `configs/notch_beam_2d_impact/cgn.toml`
to train against a different benchmark.

**Data availability:** each benchmark ships as a self-contained canonical
archive — a `canonical/<benchmark>/` folder of `<case_id>.h5` files with a
generated `README.md`, `card.json`, and CC BY 4.0 license — and `--data-root`
points at that folder. Hosting is being finalised for the v0.1 release; until
then, the adapter can ingest your own LS-DYNA output.

## How the pieces fit

```mermaid
flowchart LR
    A[LS-DYNA d3plot] -->|extract everything| B[(canonical HDF5<br/>strict SI)]
    B --> C[benchmark<br/>task + split + protocol]
    C --> D[structbench-train]
    D --> E[run dir<br/>self-contained record]
    E --> F[metrics JSON<br/>+ rollout artifacts]
```

The LS-DYNA adapter (built on `lasso-python`) follows an **extract-everything
policy**: positions, velocities, full stress and strain tensors, plastic
strain, energies, erosion state — all of it lands in the HDF5 whether or not
the current task uses it. You never re-run post-processing because a reviewer
asked for stress instead of displacement. The format is solver-agnostic by
design — the canonical schema is the contract, not LS-DYNA — and it has
already ingested a second dataset family (a concrete-beam SPH case) unchanged.
Sibling adapters (Kratos, OpenSees, OpenRadioss, …) are the intended path for
other solvers.

**Reproducibility contract.** Every run directory is self-contained —
`config.json` (fully resolved), `normalization_stats.npz`, `model-*.pt`
checkpoints — and evaluation rebuilds the exact architecture from the run's
own record, never from whatever the current code default happens to be.
Metrics land as `metrics-<split>.json` plus per-case predicted-trajectory
`.npz` files: a run directory is the complete, portable evidence for its
numbers. The repo carries a deterministic CPU-only test suite, is mypy- and
ruff-clean, and pins its environment with a `uv` lockfile.

## Repository layout

```
src/structbench/
  core/            # case schema, validation, HDF5 I/O, LS-DYNA adapter
  datasets/        # canonical readers, windowing, normalization
  benchmarks/      # one module per benchmark: split + protocol + QoIs
  models/cgn/      # CGN reference baseline (Li et al. 2023; native radius_graph)
  eval/            # rollout driver, metrics
  cli/             # structbench-train
configs/           # grouped TOML run configs, configs/<benchmark>/<family>.toml (ADR-0032)
decisions/         # architecture decision records
```

## Roadmap

<!-- Living todo list (the single planning home; ROADMAP.md is retired).
     Conventions: done = [x] + strikethrough + (date); ad-hoc additions land
     in Inbox and get triaged into a milestone; when a milestone ships, its
     crossed-out block may be compressed to one line. Reasoning lives in
     decisions/, not here. Substrate-layer work only (ADR-0014). -->

*Last revised: 2026-07-09.*

### v0.1 — Taylor 2D substrate proof

- [x] ~~Canonical case format + round-trip-tested HDF5 I/O (ADR-0011..0013)~~
- [x] ~~General LS-DYNA adapter on lasso-python (ADR-0016)~~
- [x] ~~Taylor 2D benchmark: fixed split, eval protocol, QoIs (ADR-0019)~~
- [x] ~~Config-driven pipeline `structbench-train` (train/valid/rollout)~~
- [x] ~~`radius_graph` batch-partition fix: 50.9 s → 0.22 s per batch~~ (2026-07-02)
- [x] ~~Public GitHub repository~~ (2026-07-02)
- [x] ~~First full baseline run → training-recipe rework (ADR-0028)~~ (2026-07-03)
- [x] ~~Trained CGN baseline with the ADR-0028 recipe (DUG A100; baseline
      named CGN per ADR-0034)~~ (2026-07-08)
  - [x] ~~full retrain: 4 seeds x 100k steps, about 22.4 h each on one
        A100-80GB~~ (2026-07-08)
  - [x] ~~checkpoint + recorded ADR-0019 metrics: seed s1 blessed into the
        ADR-0033 results registry~~ (2026-07-09; publishing the checkpoint
        itself remains a later item)
- [ ] Release (human action): tag the version (`0.1.0`). Baseline metrics
      (per-benchmark README + ADR-0033 results registry) and the
      prediction-vs-truth hero GIF (`assets/taylor_rollout.gif`) are in place;
      public dataset hosting is no longer a v0.1 gate (deferred, see Later)

### v0.2 — wave-1d + notch-beam pair

- [x] ~~Ingestion: 16 wave runs + 221 notch-beam cases to canonical HDF5~~ (2026-07-04)
- [x] ~~Three benchmark modules: frozen splits + QoIs (ADR-0025/0026)~~ (2026-07-03)
- [x] ~~Benchmark cards + generated views (ADR-0027), Taylor retrofitted~~ (2026-07-03)
- [x] ~~Benchmark-selection registry in `structbench-train`~~ (2026-07-03)
- [x] ~~Notch aux → max principal strain; damaged→cracked fraction (ADR-0029)~~ (2026-07-04)
- [x] ~~Data archive reorganized to the hosting layout:
      `StructBench/{canonical,raw}` mirrors (ADR-0031)~~ (2026-07-05)
- [x] ~~ADR-0030 unit-fix follow-through: patch confirmed on all 237 files,
      converters + cards corrected, ADR written + indexed~~ (2026-07-05)
- [ ] Three trained CGN baselines (checkpoint + metrics each)
  - [ ] `wave_propagation_1d`
  - [ ] `notch_beam_2d_bend`
  - [ ] `notch_beam_2d_impact`
- [ ] Validate the provisional `cracked_fraction` threshold 0.01 (ADR-0029;
      version bump if revised)
- [x] ~~Archive packaging: measure `size_gb` per benchmark (2.4 / 0.23 /
      24.1 / 24.9), generate per-archive README + card.json~~ (2026-07-05)

### Inbox — untriaged, add freely

- [x] ~~per-benchmark landing pages~~ (2026-07-09, ADR-0036): one generated
      page per benchmark at `docs/benchmarks/<name>.md`, built by
      `render_benchmark_page` (`benchmarks/render.py`, driven by
      `tools/gen_benchmark_docs.py` with a `--check` drift guard) from the
      card + results registry; narrative and figures live in the card's new
      `overview`/`figures` fields. The open questions resolved differently than
      sketched here: venue is `docs/benchmarks/` (not top-level `benchmarks/`),
      no `--landing` mode, no handwritten `intro.md`. Taylor page authored and
      tuned; the other three render without narrative until authored
- [x] ~~qualitative comparison figures in `viz/`: truth-vs-prediction von
      Mises fringe panels at 2–3 time instants~~ (2026-07-06): per-benchmark
      eval artifact via `python -m structbench.viz` (`compare_rollout` in
      `viz/fringe.py`, resolving each benchmark's aux field — `9b53b19`); on
      the Taylor page as `assets/taylor_vms_interp_170.png` (`61c3ad3`).
      ADR-0019 review note 2026-07-05
- [ ] deformed-contour overlay figure in `viz/`: truth-vs-prediction outlines
      on shared axes — the second half of that ADR-0019 review-note item; not
      yet built (only side-by-side panels exist)
- [x] ~~mypy fails on numpy 2.5 stubs (`type` statement needs py3.12
      target)~~ (resolved 2026-07-05: floor raised to Python 3.12 —
      numpy ≥ 2.5 requires it, so the 3.11 floor was untestable; mypy
      green again)
- [ ] DUG remote data dir is `data/taylor_impact`; rename to
      `taylor_impact_2d` (archive name) and update `train_taylor.slurm`,
      `ablate_taylor.slurm`, and `hpc/dug/README.md` together, between job
      fleets (still pending 2026-07-09 — all three read `.../data/taylor_impact`;
      the 07-08 retrain fleet has run, so this is due before the next v0.2 fleet)
- [x] ~~per-benchmark README: dataset info, evaluation criteria, and
      baseline results~~ (2026-07-05, ADR-0033: archive README gains
      Task/Evaluation/Numbers-to-beat/Usage sections; results live in
      per-module registries, rendered only via generated views — blessing
      a DUG run is now a ten-line registry entry + regeneration)
- [x] ~~`lr_init` code default still 1e-3; ADR-0028's 1e-4 lives only in
      the TOML~~ (resolved by ADR-0032: every config lists `lr_init`
      explicitly under strict validation, `--config` is required in train
      mode, and dataclass defaults are sanctioned as test-only, 2026-07-05)
- [x] ~~confirm the Taylor deck genuinely is g-mm-ms~~ (verified against
      `scratch/Taylor.k`: RO/G/EOS-C physical only under g-mm-ms; recorded
      in ADR-0030, 2026-07-05)
- [x] ~~reconcile ADR-0012's "4 Voigt components in 2D" prose
      (CORRECTIONS.md item)~~ (2026-07-06, `8c6d364`): ADR-0012's
      tensor-component line now records that the full 6-component Voigt layout
      is stored verbatim for all case dimensions; CORRECTIONS entry promoted

### Later (each becomes an ADR/spec when picked up)

- **v0.3 — RC beam benchmark**: erosion, twice (numerically for the FEM
  data; structurally for the surrogate — particles vanishing mid-rollout)
- Segmented beam benchmark (parked) · multi-scale CGN second Taylor baseline (spec
  Proposed)
- Training: resume support (optimizer state + `--resume`) ·
  part-id→embedding remap · ADR-0028 Phase-2 ablations (noise_std, aux
  head, capacity, stress-history)
- Eval: leaderboard submission validator · cross-benchmark utilities ·
  per-region probe metrics · convergence check
- Checkpoint-publishing workflow · second aux target (effective plastic
  strain)
- Public dataset hosting (parked): Zenodo direction agreed 2026-07-05 — one
  record per benchmark, versions ↔ record DOIs, OneDrive stays the private
  master — but no near-term plan to publish; picked up when it is
- Data-generation autonomy (deck templating or a Python-native solver)
- Scale: cell-list `radius_graph` backend when a ≥10⁶-node dataset lands
- Other solvers (Kratos, OpenSees, OpenRadioss) · SHM expansion ·
  deployment tools · packaging extras · PhysicsNeMo interop

Rationale for every item lives in [`decisions/`](decisions/).

## How this project is run

StructBench is co-developed by its maintainer and an AI agent under an
explicit written harness: a decision log of ADRs, tiered agent authority,
and a corrections log — [HARNESS.md](docs/HARNESS.md) explains the philosophy.
Agent-assisted research needs the same auditability we demand of the
benchmarks themselves; whatever you think of the arrangement, the side effect
is useful to you as a reader: the *why* behind every choice in this repo is
written down.

## Limitations, stated plainly

Small datasets by learned-simulator standards — tens to low hundreds of cases
per benchmark, testing protocol rigor and rollout stability, not web-scale
generalization. 1D/2D problems only, no erosion yet (that is v0.3's open
problem), no experimental validation data. If you need any of those today,
this repo is not it yet; if you want a clean, reproducible number to beat on
a real solid-mechanics rollout task, it is.

## License

[Apache 2.0](LICENSE). A citation entry will accompany the first release.
