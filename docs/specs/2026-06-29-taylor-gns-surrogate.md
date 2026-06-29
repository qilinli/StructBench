# Design spec — Taylor 2D autoregressive GNS surrogate

*Status: approved; feeds the implementation plan. Implements ADR-0018 (ML stack
deps) and ADR-0019 (Taylor 2D benchmark), both Accepted.*

---

## Goal

An end-to-end path on the canonical Taylor data: **load canonical HDF5 → train a
GNS surrogate → evaluate by autoregressive rollout**, porting the user's `sgnn`
single-scale model faithfully into the package. Lights up four namespaces
(`benchmarks/`, `datasets/`, `models/`, `eval/`) plus a training entry in `cli/`.

Non-goals (this slice): multi-scale GNN; hyperparameter search; the other v0.1
datasets; published-checkpoint release; `effective_plastic_strain` as a second
auxiliary target (data supports it; deferred).

## Module & file layout

```
src/structbench/
  core/                         # unchanged; stays torch-free (ADR-0018)
  benchmarks/
    __init__.py
    taylor_impact_2d/
      __init__.py
      benchmark.py              # split (case-id lists), wall geometry, QoI defs
  datasets/
    __init__.py
    canonical.py                # HDF5 case -> CaseTrajectory (numpy)
    particle.py                 # torch Dataset: windows, targets, batching
    normalization.py            # vel/acc stats (compute over train, apply)
  models/
    __init__.py
    gns/
      __init__.py
      graph_network.py          # EncodeProcessDecode (ported sgnn)
      simulator.py              # LearnedSimulator (ported sgnn, generalised)
  eval/
    __init__.py
    rollout.py                  # autoregressive rollout
    metrics.py                  # position/vonMises RMSE, QoIs
  cli/
    __init__.py
    train.py                    # config-driven train/valid/rollout entry
```

Config is typed dataclasses (`GNSConfig`, `TrainConfig`) with defaults matching
`sgnn/single_scale/config.yaml`, optionally loaded from a TOML/YAML file.

## Data flow

```
canonical HDF5 (SI, per case)
  └─ datasets.canonical.load_case_trajectory  ──► CaseTrajectory (numpy, mm)
       positions(T, P, 2)  particle_type(P)  von_mises(T, P)  time(T)
  └─ datasets.particle.WindowDataset (train)  ──► (pos_seq[L], next_pos, next_vm, ptype)
  └─ datasets.particle.trajectories (eval)    ──► full CaseTrajectory batches
  └─ datasets.normalization.compute(train)    ──► vel/acc mean,std
models.gns.LearnedSimulator(stats, boundary_fn, n_aux=1)
  └─ predict_accelerations(...) [train]   predict_positions(...) [rollout]
eval.rollout.rollout(sim, trajectory, L)    ──► predicted trajectory + per-step metrics
```

## `datasets/`

**`canonical.load_case_trajectory(h5_path, *, length_scale=1e3) -> CaseTrajectory`**
Reads one canonical case and returns a numpy `CaseTrajectory`:
- `positions` `(T, P, dim)`: `coords[sph] + displacement[t, sph]`, scaled by
  `length_scale` (SI m → mm by default, so the ported hyperparameters — radius
  0.6, wall x=−2, noise 0.02 — transfer unchanged; SI remains the storage truth).
- `particle_type` `(P,)`: from `elements/sph/part_id`.
- `von_mises` `(T, P)`: from `response/element/sph/stress` (Voigt → von Mises),
  in MPa (`stress_Pa / 1e6`) to match the prior model's scale.
- `time` `(T,)`.
- **SPH particles only** — viz-shell nodes excluded via the sph connectivity
  index (ADR-0019 §3).

**`particle.WindowDataset(cases, window=L, ...)`** — a `torch.utils.data.Dataset`
of training samples. For each case and each valid `t`, a sample is
`(position_seq (P, L, dim), particle_type (P,), next_position (P, dim),
next_von_mises (P,))`. A custom collate concatenates particles across the batch
and tracks `n_particles_per_example` (matching the sgnn graph batching).

**`normalization.compute_stats(train_cases) -> NormalizationStats`** — velocity
(first difference) and acceleration (second difference) mean/std over the train
split, in mm/frame and mm/frame². Combined with `noise_std` at model build time
(`std ← sqrt(std² + noise_std²)`), exactly as the sgnn does. Cached to disk
keyed by the split so it is computed once. The auxiliary (von Mises) field is
also normalized: its scalar mean/std are pooled over the raw values (no finite
difference) of the train split. The decoder predicts the auxiliary channel in
normalized space, the training target is normalized to match
(`next_aux_norm = (next_aux − mean) / std`), and `predict_positions`
de-normalizes the output back to MPa for rollout/metrics. This keeps the
auxiliary loss O(1) and balanced against the position loss — a deliberate
improvement over the SGNN's raw-MPa auxiliary loss, which (with `w_aux = 1`)
over-weighted stress. The auxiliary stats carry no `noise_std` inflation (the
auxiliary target has no input noise).

## `models/gns/`

Ported from `sgnn/single_scale`, generalised so the module carries no
Taylor-specific knowledge:

- **`graph_network.EncodeProcessDecode`** — encoder (node/edge MLPs + LayerNorm)
  → processor (M `InteractionNetwork` message-passing steps, residual + LayerNorm,
  `add` message aggregation) → decoder MLP. Output width = `dim + n_aux`.
  Ported essentially verbatim.

- **`simulator.LearnedSimulator`** — builds the graph with `radius_graph`
  (connectivity radius, `max_num_neighbors`), normalizes the velocity history,
  predicts normalized acceleration + `n_aux` scalars, integrates via Euler.
  **Generalisation:** the wall-distance feature is removed from the model and
  replaced by an injected `boundary_feature_fn(positions) -> Tensor | None`
  callable supplied at construction. The model concatenates whatever node
  features the callable returns; with `None` it is a vanilla GNS. `n_aux` is a
  constructor argument (Taylor uses 1).

The Taylor benchmark constructs the simulator with a wall-distance
`boundary_feature_fn` for the rigidwall plane and `n_aux=1`.

## `eval/`

- **`rollout.rollout(simulator, trajectory, window, device) -> RolloutResult`** —
  seed with the first `L` ground-truth frames, predict step by step to the end,
  returns predicted positions/von Mises and per-step + cumulative metrics.
  Mirrors the sgnn `evaluate.rollout`.
- **`metrics`** — `position_rmse` (one-step, full-rollout), `von_mises_rmse`,
  and QoIs `final_length`, `mushroom_width` with their errors (ADR-0019 §5).

## `benchmarks/taylor_impact_2d/`

`benchmark.py` holds the immutable split (ADR-0019 §4) as case-id lists
(`TRAIN`, `VAL`, `TEST_INTERP`, `TEST_EXTRAP`, `HELD_ASIDE`), the wall geometry
(`x = −2 mm` plane → the `boundary_feature_fn`), the auxiliary-field name
(`von_mises_stress`), and the QoI definitions. Resolves a `dataset_root` to the
`h5_canonical/` case files.

## `cli/train.py`

Config-driven entry with `train` / `valid` / `rollout` modes (like the sgnn).
Training loop: GNS random-walk position noise, Adam, exponential LR decay,
dual MSE loss `w_pos·‖Δacc‖² + w_aux·(Δvm)²`, periodic validation rollout,
checkpoint-best. Device auto-selects CUDA. Defaults match the sgnn config
(L=11, radius 0.6, hidden 64, 5 MP steps, noise 0.02, lr 1e-3, decay 0.1/30k).

## Dependencies & records

- ADR-0018: add `torch`, `torch_geometric` to PRINCIPLES.md approved list and
  `pyproject.toml` `[project.dependencies]` (loose lower bounds; pinned in the
  `uv` lockfile). `core/` stays torch-free.
- ADR-0019: the benchmark task, split, and protocol.
- ARCHITECTURE.md: flesh out `benchmarks/`, `datasets/`, `models/`, `eval/`
  responsibilities and add the case→graph data-flow note + the torch-free-core
  rule.

## Testing

Data-free, deterministic, CPU-only (PRINCIPLES.md):
- `datasets`: windowing, SI→mm scaling, Voigt→von Mises, viz-node exclusion,
  normalization stats — on tiny synthetic `CaseTrajectory`/HDF5 fixtures.
- `models`: forward-pass and rollout **shape/finiteness** on a small synthetic
  graph (a few particles, 1–2 MP steps); `boundary_feature_fn` injection;
  `n_aux` plumbing.
- `eval`: rollout length/shape and metric values on a trivial constant-motion
  trajectory with a known answer.
- Manual (not in suite): a short smoke-train on 1–2 cases for a few hundred
  steps on GPU to confirm loss decreases and a rollout runs.

## Resolved choices

1. **Config**: typed `@dataclass` configs (`GNSConfig`, `TrainConfig`) are the
   source of truth; an optional `TrainConfig.from_toml(path)` loads overrides
   via stdlib `tomllib` (3.11 floor). No YAML dependency.
2. **Normalization-stats cache**: a small `.npz` written next to the split under
   the dataset root (e.g. `<dataset_root>/derived/norm_<split-hash>.npz`),
   computed once and reused; recomputed if the train case-id list changes.
3. **Run outputs**: training writes checkpoints, the resolved config, and
   rollout artifacts under a `--out` directory (default `./runs/<run-name>/`,
   gitignored). Publishing a baseline checkpoint to a registry is out of scope
   for this slice.
