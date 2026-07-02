# StructBench

**Standardized benchmarks for machine learning on structural simulation.**
A task definition, a fixed split, metrics in physical units, and a reference
baseline to beat — for structures under dynamic and extreme loading.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> **Status: pre-release (v0.1 imminent).** One benchmark, one baseline, a small
> dataset, no paper. What exists is real and tested; what doesn't is on the
> [roadmap](ROADMAP.md).

![Taylor bar rollout: copper bar mushrooming against a rigid wall, colored by von Mises stress](assets/taylor_rollout.gif)

*A 2D copper bar striking a rigid wall at 200 m/s — LS-DYNA SPH ground truth,
colored by von Mises stress. The prediction-vs-truth comparison replaces this
once the baseline training run completes.*

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

## The Taylor 2D benchmark (v0.1)

Taylor bar impact: a metal bar fired against a rigid wall mushrooms
plastically at the impact face. Small enough to simulate cheaply, yet it
exercises what makes extreme-loading simulation hard — large deformation,
contact, plasticity, stress waves. Classic, compact, and brutal on rollout
stability: a clean first target for learned simulators.

| | |
|---|---|
| **Data** | 34 LS-DYNA SPH simulations: 33 benchmark cases (velocities 100–200 m/s × 3 geometries) + 1 held-aside mesh-convergence case. 4,800–8,000 particles × ~152 frames per case, canonical HDF5, strict SI, everything extracted (stress/strain tensors, plastic strain, energies, erosion state) |
| **Task** | Autoregressive next-step surrogate: from an 11-frame position history, predict per-particle acceleration (Euler-integrated) + von Mises stress; roll out the full trajectory |
| **Baseline** | Single-scale GNS (encode–process–decode, Sanchez-Gonzalez et al. 2020): 5 message-passing steps, hidden width 64, von Mises auxiliary head, wall-distance feature |
| **Protocol** | [ADR-0019](decisions/0019-taylor-2d-benchmark-definition.md) — task, split, and metrics are fixed; changing them is a new benchmark version |

### The split is immutable

| split | velocities (m/s) | cases |
|---|---|---|
| train | 100, 110, 120, 140, 160, 180, 190 (×3 geometries) | 21 |
| val | 150 | 3 |
| **test — interpolation (headline)** | **130, 170** | **6** |
| test — extrapolation (reported separately) | 200 | 3 |

"Generalizes to unseen velocities" and "extrapolates past the training
envelope" are different claims, and the benchmark refuses to blur them:
interpolation is the headline number, extrapolation a separate, harder probe.

### Metrics — physical units, no dimensionless scores

- **Position RMSE (mm)**: one-step and full rollout
- **von Mises RMSE (MPa)** over the rollout
- **Quantities of interest**: final bar length and mushroom-width errors —
  the numbers a structural engineer would actually check

Every evaluation persists its metrics as JSON plus per-case rollout `.npz`
artifacts — results are inspectable, not just printable.

### Numbers to beat

*The reference GNS training run is staged for a single A100; this table is
filled from that run's artifacts before release.*

| model | one-step pos. RMSE (mm) ↓ | rollout pos. RMSE (mm) ↓ | von Mises RMSE (MPa) ↓ | QoI errors |
|---|---|---|---|---|
| GNS baseline — interpolation | *TBD* | *TBD* | *TBD* | *TBD* |
| GNS baseline — extrapolation | *TBD* | *TBD* | *TBD* | *TBD* |

## Quickstart

```bash
git clone https://github.com/<org>/StructBench
cd StructBench
pip install -e .
```

Installs from wheels on Linux, macOS, and Windows, CPU or CUDA. **No compiled
graph dependencies**: a native pure-torch `radius_graph` replaces
`torch-cluster`/`pyg-lib` (`torch_geometric` is used for `MessagePassing`
only) — no C++ build step, no CUDA-version matching dance. If you have fought
GNS codebases on a cluster or on Windows, you know why this matters.

```bash
# Train the GNS baseline
structbench-train --mode train --config configs/taylor_2d.toml \
    --data-root /path/to/taylor_2d_h5 --out runs/taylor-gns

# Validate, then roll out on the test splits (architecture is rebuilt from
# the run directory's own record — no --config needed, or accepted)
structbench-train --mode valid   --data-root /path/to/taylor_2d_h5 --out runs/taylor-gns
structbench-train --mode rollout --data-root /path/to/taylor_2d_h5 --out runs/taylor-gns
```

**Data availability:** the canonical HDF5 dataset ships with the v0.1 release
(hosting being finalised). Until then, the adapter can ingest your own LS-DYNA
output.

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
numbers. The repo carries 66 deterministic CPU-only tests, is mypy- and
ruff-clean, and pins its environment with a `uv` lockfile.

## Repository layout

```
src/structbench/
  core/            # case schema, validation, HDF5 I/O, LS-DYNA adapter
  datasets/        # canonical readers, windowing, normalization
  benchmarks/      # taylor_impact_2d: split + protocol + QoIs
  models/gns/      # reference GNS (native radius_graph, no compiled deps)
  eval/            # rollout driver, metrics
  cli/             # structbench-train
configs/           # TOML training configs
decisions/         # 21 architecture decision records
```

## Roadmap

- **v0.1** — Taylor 2D end to end: data, baseline checkpoint, recorded
  metrics, public release.
- **v0.2+** — RC beam and segmented-beam benchmarks (data exists from prior
  published work); a multi-scale GNS second baseline (spec drafted); plastic
  strain as a second target.
- **Later** — more solver adapters, larger-scale graph backends, multi-modal
  SHM benchmarks, deployment tooling.

Sequencing in [ROADMAP.md](ROADMAP.md); reasoning in [`decisions/`](decisions/).

## How this project is run

StructBench is co-developed by its maintainer and an AI agent under an
explicit written harness: a decision log (21 ADRs), tiered agent authority,
and a corrections log — [HARNESS.md](HARNESS.md) explains the philosophy.
Agent-assisted research needs the same auditability we demand of the
benchmarks themselves; whatever you think of the arrangement, the side effect
is useful to you as a reader: the *why* behind every choice in this repo is
written down.

## Limitations, stated plainly

One benchmark. 34 cases is small by learned-simulator standards — this tests
protocol rigor and rollout stability, not web-scale generalization. 2D SPH
only, single material model, no experimental validation data yet. If you need
any of those today, this repo is not it yet; if you want a clean, reproducible
number to beat on a real solid-mechanics rollout task, it is.

## License

[Apache 2.0](LICENSE). A citation entry will accompany the first release.
