# MGN Training Path + Family Dispatch Implementation Plan (checkpoint ①-c2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `structbench-train --mode train --config configs/deforming_plate/mgn.toml` a working command: `MGNConfig` registered in `MODEL_FAMILIES`, `cli/train.py` dispatching per family (CGN path byte-identical), the MGN train step (NORMAL-masked equal-weight L2 on normalized velocity+stress, noise 3e-3 with γ=1.0 correction, normalizer warmup), mesh-aware windowing collate, family-aware validation/evaluation with `bind_case`/`reset_rollout`, smoke configs, and an end-to-end CPU train→validate smoke gate. Includes the ledgered carry-forwards: the **shared `_graph_features` extraction** (the final review's train/eval divergence seam — Task 1, first), the Quickstart template fix, and the mesh-branch time-dtype idiom.

**Architecture:** Task 1 refactors `MeshSimulator` so ONE private feature builder serves both inference (`predict_positions`) and a new training entry (`forward_train`), with a parity test pinning them equal. Task 2 gives windowed samples a trajectory identity (`traj_idx`) and adds a model-owned mesh collate (per-trajectory static edge/reference tensors, node-offset batching). Task 3 registers `MGNConfig`. Task 4 branches `cli/train.py`: an early `if family == "mgn"` fork into `_train_mgn` (its own loop; no `compute_stats` — MGN's normalizers live in the checkpoint, so **MGN checkpoints are self-contained**) and a family branch in `evaluate()` that reads the record's family BEFORE the stats-file guard and binds+resets per case (`_validate` stays CGN-only and untouched — `_train_mgn` owns its val loop). Task 4 is split into 4a (evaluate side) and 4b (train side). Task 5 lands configs + the two small carry-forwards + docs regen. Task 6 is the gate: a real `train()` call on a tiny synthetic mesh benchmark, through validation, on CPU, in pytest.

**Tech Stack:** Python 3.12+, torch. No new dependencies; no TensorFlow; no torch_geometric in mgn code.

**Plan ①-c2 of the v0.3 build order** (after ①-c1, merged at `dbc4653`). The 10M-step blessing run itself is maintainer-scheduled compute (ADR-0043 §8) — this plan makes it *runnable*, it does not run it.

## Global Constraints

- Python floor **3.12**; ruff line length **88** + `ruff format`; mypy `disallow_untyped_defs = true`; NumPy-style docstrings on every public API; `_`-prefix symbols private across module boundaries (package-internal tests may exercise them).
- **No new dependencies. CGN behaviour byte-identical** — the CGN train/eval path must not change observably; the full pre-existing suite (**283 passed / 6 skipped**) is the regression gate before every commit.
- **ADR-0043 recipe values (training side):** training noise `N(0, 3e-3)` added to the last input frame's world positions of **NORMAL nodes only**, with **γ = 1.0** target correction (targets computed from the *noisy* position, so the model learns to correct the noise: `v_target = next_pos − x_noisy`); loss = single L2 on the concatenated normalized (velocity, stress) output, **NORMAL-masked** (§9a declared choice; `w_pos`/`w_aux` act as sub-term multipliers and are **1.0 in the reference config**); scripted (OBSTACLE=1) nodes receive the next-step velocity *input* `next_pos − x_last` (their rows are never noised — noise is NORMAL-only); normalizer warmup = `accumulate=True` for the first `normalizer_warmup_steps` (default 1000) training steps, on feature AND target normalizers; batch size 2; Adam, lr 1e-4 → 1e-6 (the existing `TrainConfig` exponential schedule + `LR_SCHEDULE_FLOOR = 1e-6`); 10M steps in the reference config, tiny in smoke. **Two declared interpretations** (record in the training ledger when the blessing run is configured): the ADR's "1000-step normalizer warmup" is implemented as accumulate-during-the-first-N-steps — the official framework instead accumulates to the normalizer's built-in 1e6-call cap (recoverable by setting the knob huge); and the reused `TrainConfig` schedule's knee is `0.4*training_steps` (~1e-6 by 10M) rather than pinning 1e-6 at 5M — the anneal depth is what is reproduced, not the knee.
- **Verified cli/config facts this plan builds on (68/68-confirmed grounding, 2026-08-08):** `MODEL_FAMILIES: dict[str, type] = {"cgn": CGNConfig, "gns": CGNConfig}` (config.py:179); the strict loader dispatches `[model].family` through it and requires the TOML `[model]` keys to match the config dataclass's fields; the `aux_transform` check uses `getattr(model, "aux_transform", "none")` so a config class without that field no-ops gracefully; `train(spec, cgn, train_cfg, data_root, out_dir, device, *, family="cgn")` records family but never dispatches on it; the ADR-0035 window check compares `model.input_frames` to `spec.card.input_frames`; `evaluate()` reconstructs `CGNConfig(**...)` unconditionally (train.py:766) and reuses ONE simulator across its per-case loop (train.py:804), calling rollout → one_step_position_rmse → one_step_aux_rmse per case; `_validate` iterates rollout over val trajectories on the live training simulator; `main()` has no family flag (family comes from the TOML in train mode, from `config.json` in valid/rollout); checkpoints follow `model-best-<step>.pt`/`model-final-<step>.pt` with a highest-step glob loader; `collate_samples` concatenates a fixed explicit key set (an extra sample key is ignored).
- Tests: pytest, synthetic-only, deterministic; conda env interpreter (PowerShell): `& "C:\Users\272766h\AppData\Local\miniconda3\envs\structbench\python.exe" -m pytest <target> -v`. Full suite before each commit; ruff check on `src tests`; ruff format --check + mypy on **changed files only** (repo-wide baselines carry the two pre-existing hygiene items — `datasets/normalization.py` mypy, `notch_beam_2d_impact/__init__.py` format — NOT this plan's).
- After the render/docs change (Task 5): regenerate docs (`python tools/gen_benchmark_docs.py`) — the drift test binds.
- Branch: **`feat/mgn-training`** off `main` (@ `dbc4653`). Never commit to `main`; merge/push are human calls (ADR-0023).
- Commits: Conventional Commits; append to every commit message, after a blank line:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01HHpG1wUFUfb2Q31Hp948YX`

## File Structure

```
src/structbench/models/mgn/
  simulator.py       # MODIFY: extract _graph_features; add forward_train
  collate.py         # CREATE: MeshStatic, mesh_static_from_trajectory, collate_mesh_samples
  __init__.py        # MODIFY: export the collate names
src/structbench/datasets/
  particle.py        # MODIFY: WindowDataset samples gain "traj_idx" (additive)
  canonical.py       # MODIFY (Task 5): mesh-branch time dtype idiom (carry-forward)
src/structbench/config.py            # MODIFY: MGNConfig + MODEL_FAMILIES["mgn"]
src/structbench/cli/train.py         # MODIFY: family dispatch (train + evaluate; _validate untouched)
src/structbench/benchmarks/render.py # MODIFY (Task 5): Quickstart family wording (carry-forward)
configs/deforming_plate/
  mgn.toml           # CREATE: the §8 reference config
  mgn_smoke.toml     # CREATE: tiny CPU smoke config
docs/benchmarks.md, docs/benchmarks/*.md  # REGENERATE (Task 5)
tests/models/mgn/
  test_mgn_simulator.py  # MODIFY: forward_train + feature-parity tests
  test_mgn_collate.py    # CREATE
tests/test_config.py (or the file that tests load_run_config — bind to real name)  # MODIFY: mgn config-load tests
tests/cli/test_mgn_train_smoke.py    # CREATE: the end-to-end gate
```

---

### Task 1: Extract `_graph_features`; add `forward_train` (+ the train/eval parity test)

**Files:**
- Modify: `src/structbench/models/mgn/simulator.py`
- Test: `tests/models/mgn/test_mgn_simulator.py`

**Interfaces:**
- Consumes: the existing `MeshSimulator` internals (Tasks 1–4 of ①-c1).
- Produces (exact signatures later tasks rely on):

```python
def _graph_features(
    self,
    x_last: Tensor,            # (P, dim) current world positions (mm)
    one_hot: Tensor,           # (P, node_type_size) float32
    scripted_velocity: Tensor, # (P, dim) — zeros off scripted rows
    mesh_edge_index: Tensor,   # (2, Em) int64
    reference_coords: Tensor,  # (P, dim)
    n_particles_per_example: Tensor | None,  # (B,) int64; None == single example
    *,
    accumulate: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """(node_f_norm, mesh_edge_index, mesh_ef_norm, world_edge_index, world_ef_norm).

    **Batched world-edge partition (load-bearing):** when
    ``n_particles_per_example`` is given, world edges are computed PER EXAMPLE —
    slice the positions per sample, run ``world_edges`` on each slice, add the
    node offset, concatenate — never one whole-tensor radius query over the
    collated batch (that would invent cross-example edges between unrelated
    cases). ``None`` (inference) is the single-example fast path.
    """


def forward_train(
    self,
    x_last: Tensor,          # (P, dim) — noise ALREADY applied by the caller
    next_positions: Tensor,  # (P, dim) ground-truth next frame
    next_aux: Tensor,        # (P,) ground-truth stress (working frame, MPa)
    particle_types: Tensor,  # (P,) int64
    mesh_edge_index: Tensor, # (2, Em) int64 (batched with node offsets is fine)
    reference_coords: Tensor,# (P, dim)
    n_particles_per_example: Tensor,  # (B,) int64 — the collate's per-example counts
    *,
    accumulate: bool,
) -> tuple[Tensor, Tensor]:
    """Return (pred_norm (P, dim+1), target_norm (P, dim+1)).

    Builds one-hot + the scripted-velocity input (``next_positions - x_last``
    on scripted rows, zeros elsewhere), runs the shared feature builder (its
    ``accumulate`` flag flows into ``_graph_features``, so the node/mesh-edge/
    world-edge normalizers accumulate alongside the target normalizer) and
    the network, and normalizes the target ``cat([next_positions - x_last,
    next_aux[:, None]], dim=1)`` via the target normalizer (with
    ``accumulate``). γ = 1.0 falls out of the construction: the caller noises
    ``x_last``, so the velocity target is measured from the noisy position.
    """
```

  `predict_positions` is refactored to call `_graph_features` (with `accumulate=False`, `n_particles_per_example=None`, scripted velocity from the bound GT) — its behaviour must be byte-identical (the existing 7 simulator tests + integration test are the regression net). Also add a constructor invariant: `set(scripted_types) <= set(kinematic_types)` else `ValueError` (the NORMAL-only noise mask silently relies on it) — with a one-line test.

- [ ] **Step 1: Write the failing tests** (append to `tests/models/mgn/test_mgn_simulator.py`)

```python
def test_forward_train_shapes_and_target_semantics():
    torch.manual_seed(0)
    sim = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=0.5)
    P = 5
    x = torch.rand(P, 3)
    nxt = x + 0.1
    aux = torch.rand(P)
    types = torch.tensor([0, 0, 1, 3, 0], dtype=torch.int64)
    mesh = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
    ref = torch.rand(P, 3)
    pred, target = sim.forward_train(
        x, nxt, aux, types, mesh, ref, torch.tensor([P]), accumulate=False
    )
    assert pred.shape == (P, 4) and target.shape == (P, 4)
    # untrained target normalizer is identity: target == [v_target | stress]
    torch.testing.assert_close(target[:, :3], nxt - x)
    torch.testing.assert_close(target[:, 3], aux)


def test_train_and_eval_paths_build_identical_features():
    """The final-review seam: train and eval must share ONE feature builder."""
    sim, gt, types = _bound_sim()
    x_last = gt[1]  # last frame of the first window
    # eval-path features: capture what predict_positions feeds the net
    captured: list[torch.Tensor] = []
    hook = sim._net.register_forward_pre_hook(
        lambda mod, args: captured.append(tuple(a.detach().clone() for a in args))
    )
    sim.reset_rollout()
    sim.predict_positions(
        gt[0:2].permute(1, 0, 2).contiguous(), torch.tensor([5]), types
    )
    hook.remove()
    eval_args = captured[0]  # ALL FIVE network inputs, not just node features
    # train-path features on the SAME inputs (no noise; scripted velocity in
    # training comes from next_positions == GT[2], identical to the bound GT)
    captured.clear()
    hook = sim._net.register_forward_pre_hook(
        lambda mod, args: captured.append(tuple(a.detach().clone() for a in args))
    )
    sim.forward_train(
        x_last,
        gt[2],
        torch.zeros(5),
        types,
        sim._mesh_edge_index,
        sim._reference_coords,
        torch.tensor([5]),
        accumulate=False,
    )
    hook.remove()
    for train_arg, eval_arg in zip(captured[0], eval_args, strict=True):
        torch.testing.assert_close(train_arg, eval_arg)


def test_forward_train_accumulates_normalizers_when_asked():
    torch.manual_seed(0)
    sim = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=0.5)
    P = 4
    args = (
        torch.rand(P, 3), torch.rand(P, 3) + 1.0, torch.rand(P),
        torch.zeros(P, dtype=torch.int64),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int64), torch.rand(P, 3),
    )
    norms = [
        sim._target_normalizer,
        sim._node_normalizer,
        sim._mesh_edge_normalizer,
        sim._world_edge_normalizer,
    ]  # bind to the REAL attribute names
    before = [int(n._n_accumulations) for n in norms]
    sim.forward_train(*args, torch.tensor([P]), accumulate=True)
    sim.forward_train(*args, torch.tensor([P]), accumulate=False)
    after = [int(n._n_accumulations) for n in norms]
    assert after == [b + 1 for b in before]  # ALL FOUR accumulate exactly once


def test_forward_train_never_builds_cross_example_world_edges():
    """Two collated examples whose nodes coincide ACROSS examples: batched
    world edges must stay within-example (the plan review's Critical — a
    whole-batch radius query would invent cross-case edges)."""
    torch.manual_seed(0)
    sim = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=100.0)
    P = 2  # per example; positions overlap across examples deliberately
    x = torch.zeros(2 * P, 3)  # ALL nodes coincide -> max cross-example risk
    nxt = x + 0.1
    aux = torch.zeros(2 * P)
    types = torch.zeros(2 * P, dtype=torch.int64)
    mesh = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int64)
    ref = torch.zeros(2 * P, 3)
    captured: list[tuple] = []
    hook = sim._net.register_forward_pre_hook(
        lambda mod, args: captured.append(tuple(a.detach().clone() for a in args))
    )
    sim.forward_train(
        x, nxt, aux, types, mesh, ref, torch.tensor([P, P]), accumulate=False
    )
    hook.remove()
    world_ei = captured[0][3]  # (2, Ew) — bind index to the real net arg order
    example_of = torch.tensor([0, 0, 1, 1])
    assert (example_of[world_ei[0]] == example_of[world_ei[1]]).all()
```

(Bind `sim._mesh_edge_index`/`sim._reference_coords`/`sim._net`/`sim._target_normalizer` to the REAL private attribute names in the current `simulator.py` — read it first; adjust the test spellings, not the semantics. If the bound-attribute names differ, use the real ones.)

- [ ] **Step 2: Run to verify they fail** (`forward_train` missing; parity test fails at attribute or call).

- [ ] **Step 3: Refactor + implement.** Extract the feature-assembly block of `predict_positions` into `_graph_features` verbatim (move-only for the shared part: one-hot/scripted handling stays in the callers since the sources differ; the normalize-features + world-edge recompute + mesh-edge feature math is the shared body). Implement `forward_train` per the docstring above. `predict_positions` behaviour unchanged.

- [ ] **Step 4: Run the whole mgn suite** — new tests PASS and all pre-existing ①-c1 tests still pass unchanged: `python -m pytest tests/models/mgn -v`.

- [ ] **Step 5: Full gates + commit** — `feat(mgn): shared _graph_features + forward_train (train/eval parity pinned)`

---

### Task 2: `traj_idx` in windowed samples + the mesh collate

**Files:**
- Modify: `src/structbench/datasets/particle.py` (`WindowDataset.__getitem__` gains `"traj_idx": <trajectory index>` — the sample CONTRACT is additive; internally the `_index` tuples gain the trajectory index, the only clean implementation)
- Create: `src/structbench/models/mgn/collate.py`
- Modify: `src/structbench/models/mgn/__init__.py` (export `MeshStatic`, `mesh_static_from_trajectory`, `collate_mesh_samples`)
- Test: `tests/models/mgn/test_mgn_collate.py`; extend the existing WindowDataset test file (bind to its real name under `tests/datasets/`) with a `traj_idx` assertion + a CGN-collate-unaffected assertion.

**Interfaces:**
- Consumes: `CaseTrajectory` (with `cells`/`reference_coords` from ①-b), `cells_to_edges` (①-c1).
- Produces:

```python
@dataclass(frozen=True)
class MeshStatic:
    mesh_edge_index: Tensor   # (2, Em) int64
    reference_coords: Tensor  # (P, dim) float32


def mesh_static_from_trajectory(traj: CaseTrajectory) -> MeshStatic:
    """cells → bidirectional edges (cells_to_edges); ValueError if traj.cells
    or traj.reference_coords is None (not a mesh benchmark)."""


def collate_mesh_samples(
    batch: list[dict], statics: Sequence[MeshStatic]
) -> dict:
    """collate_samples' keys PLUS 'mesh_edge_index' (2, ΣEm — node-offset
    adjusted) and 'reference_coords' (ΣP, dim). statics is indexed by each
    sample's 'traj_idx'. Node offsets are the cumulative particle counts of
    the batch's samples, in batch order."""
```

- [ ] **Step 1: Write the failing tests.** Collate test (hand-checked offsets): two fake samples with P=3 and P=2 from two statics whose edge sets are `[[0,1],[1,0]]` and `[[0,1],[1,0]]` → collated `mesh_edge_index` equals `[[0,1,3,4],[1,0,4,3]]`; `reference_coords` is the row-concat; all `collate_samples` keys present and equal to `collate_samples(batch)`'s output. WindowDataset test: a two-trajectory dataset's samples carry `traj_idx` 0/1 matching their source; `collate_samples` output is unchanged by the new key.

- [ ] **Step 2: RED** → **Step 3: implement** (`collate_mesh_samples` calls the existing `collate_samples(batch)` for the shared keys, then appends the two mesh keys — no duplication of the concat logic) → **Step 4: GREEN + datasets suite regression** → **Step 5: full gates + commit** — `feat(mgn): trajectory-indexed windows + mesh collate with node offsets`

---

### Task 3: `MGNConfig` + `MODEL_FAMILIES["mgn"]`

**Files:**
- Modify: `src/structbench/config.py`
- Test: the existing config test file (bind to its real name; likely `tests/test_config.py`)

**Interfaces:**
- Produces:

```python
@dataclass
class MGNConfig:
    """MGN family hyperparameters (ADR-0043 §8; ADR-0041 v0.3)."""

    input_frames: int = 2
    dim: int = 3
    hidden_dim: int = 128
    message_passing_steps: int = 15
    nmlp_layers: int = 2
    node_type_size: int = 9
    world_edge_radius: float = 30.0  # working frame (mm); provisional — Task 8
    noise_std: float = 0.003
    normalizer_warmup_steps: int = 1000
```

  and `MODEL_FAMILIES` gains `"mgn": MGNConfig`. The strict loader needs NO other change (verified: family dispatch is table-driven; the `aux_transform` check `getattr`s with a default and no-ops for MGNConfig).

- [ ] **Step 1: Failing tests** — a TOML snippet with `[model] family="mgn"` + all `MGNConfig` fields loads to `ResolvedRunConfig` with `family == "mgn"` and an `MGNConfig` instance; a wrong `input_frames` (≠ the deforming_plate card's 2) raises the ADR-0035 `ConfigError`; an unknown `[model]` key raises. (Mirror the file's existing CGN test style — read it first.)
- [ ] **Step 2: RED** → **Step 3: implement** → **Step 4: GREEN + config suite** → **Step 5: full gates + commit** — `feat(config): MGNConfig registered as the mgn model family`

---

### Task 4a: `evaluate()` family branch (+ `_model_config_from_record`)

**Files:**
- Modify: `src/structbench/cli/train.py` (evaluate side only)
- Test: extend the existing cli test file (`tests/cli/test_train_eval.py` — bind to the real name)

**Interfaces (pinned by the review against the real file):**
- Extract `_model_config_from_record(record) -> CGNConfig | MGNConfig` — reads `record["model"]["family"]` and constructs the right dataclass from the model section (drop the `family` key before `**`).
- In `evaluate()`: read the record family **immediately after `_resolve_run_spec` (train.py:761) and BEFORE the stats-file guard** — the `normalization_stats.npz` existence check at train.py:762-764 and `NormalizationStats.load` at 768 move into (or are gated on) the **cgn arm**, along with `build_simulator` + `_bind_boundary_feature` (770-776). The mgn arm: `MGNConfig` from the record, `build_mgn_simulator(...)` (Task 4b defines it; for 4a introduce the builder with the simulator-construction mapping and no train-loop dependencies), `.load(checkpoint)`, **no stats file at all** (MGN checkpoints are self-contained — normalizer buffers live in the state_dict, verified in ①-c1). Update `evaluate()`'s docstring: the FileNotFoundError-for-missing-stats promise (756-757) becomes cgn-conditional.
- Per-case loop (train.py:804): for the mgn family, after `load_case_trajectory`, `bind_case(torch.from_numpy(t.cells), torch.from_numpy(t.reference_coords), torch.from_numpy(t.particle_type), torch.from_numpy(t.positions))` and `reset_rollout()` before **each** of the three eval calls (rollout at 808, one_step_position_rmse at 817, one_step_aux_rmse at 824 — three resets). Guard: `cells`/`reference_coords` None → `ValueError` naming the benchmark as non-mesh.
- The shared tail after reconstruction is union-safe (`input_frames` exists on both config classes — verified).

- [ ] **Step 1: Read `evaluate()` in full** (train.py ~750-920) and bind the anchors above to the real lines.
- [ ] **Step 2: Failing unit test** — build a record via `resolved_config_dict(family="mgn", model=MGNConfig(), train=<minimal TrainConfig>, ...)` (mirror the real call site's kwargs), round-trip `read_run_record`, assert `_model_config_from_record` returns an `MGNConfig` with the right fields; same assertion for a cgn record returning `CGNConfig` (regression).
- [ ] **Step 3: RED** → **Step 4: implement** (extraction + branch + docstring) → **Step 5: GREEN; run `tests/cli` + FULL suite** (CGN byte-identical) → **Step 6: gates + commit** — `feat(cli): family-aware evaluate — mgn runs need no stats file`

---

### Task 4b: `train()` dispatch + `_train_mgn`

**Files:**
- Modify: `src/structbench/cli/train.py` (train side)
- Test: none new here — Task 6's smoke is this task's gate (write it next; if you must sanity-run earlier, drive `_train_mgn` directly with a 2-case in-memory spec).

**Interfaces (pinned; deviations are findings):**
- **Branch placement:** after the val-truncation block (train.py:438) and **before `cached_compute_stats` (442)** — the data-prep above (trajectory loading, `train_frames`/`scored_frames` truncation) is family-agnostic and stays shared. The branch passes the loaded data: `if family == "mgn": return _train_mgn(spec, model_cfg, train_cfg, train_trajs, val_trajs, out_dir, device)`.
- **One sanctioned CGN-path addition:** immediately after the mgn early-return, a single narrowing line (`assert isinstance(model_cfg, CGNConfig)` or `typing.cast`) so mypy accepts the CGN body below — this is the ONLY permitted change to the CGN arm's body.
- **LR schedule:** no helper exists — the CGN schedule is 3 inline lines (train.py:568-574: `lr_new = train_cfg.lr_init * train_cfg.lr_decay ** (step / train_cfg.lr_decay_steps) + LR_SCHEDULE_FLOOR` + param-group assignment). **Sanctioned move-only extraction:** `_lr_at(step, train_cfg) -> float`, called by BOTH loops — a named, behaviour-preserving exception to "CGN untouched" (the full CGN test suite is the proof of preservation).
- `build_mgn_simulator(mgn: MGNConfig, *, kinematic_types: tuple[int, ...], device: str) -> MeshSimulator` (introduced in 4a): kwargs map exactly `dim=dim`, `latent=hidden_dim`, `mp_steps=message_passing_steps`, `n_hidden=nmlp_layers`, `node_type_size=node_type_size`, `world_edge_radius=world_edge_radius`, `kinematic_types=kinematic_types`, `device=device`; `scripted_types` stays the class default `(1,)`.
- `_train_mgn` body:
  1. Guard: every trajectory has `cells`/`reference_coords` (else `ValueError`: non-mesh benchmark).
  2. `statics = [mesh_static_from_trajectory(t) for t in train_trajs]`; simulator via `build_mgn_simulator(..., kinematic_types=spec.kinematic_types, device=device)`; `sim.to(device)`.
  3. DataLoader: `WindowDataset(train_trajs, mgn.input_frames)`, `shuffle=True`, `batch_size=train_cfg.batch_size`, `collate_fn=functools.partial(collate_mesh_samples, statics=statics)`.
  4. Per step (tensors `.to(device)`): `x_last = batch["position_seq"][:, -1]`; noise `torch.randn_like(x_last) * mgn.noise_std` masked to rows NOT in `spec.kinematic_types`; `x_noisy = x_last + noise`; `pred, target = sim.forward_train(x_noisy, batch["next_position"], batch["next_aux"], batch["particle_type"], batch["mesh_edge_index"], batch["reference_coords"], batch["n_particles_per_example"], accumulate=(step < mgn.normalizer_warmup_steps))`; per-node squared error `w_pos * ||dv||^2 + w_aux * ds^2`, meaned over non-kinematic rows; Adam + `_lr_at`; backward/step.
  5. `val_every`: `sim.eval()` + `torch.no_grad()`; for each val trajectory `t`: `bind_case(...)` (tensors `.to(device)`), `reset_rollout()`, `rollout(sim, t, mgn.input_frames, device, kinematic_types=spec.kinematic_types, scored_frames=spec.scored_frames)`; track mean position RMSE; `model-best-<step>.pt` on improvement; back to `sim.train()`. **`model-final-<step>.pt` ONLY when no validation ever improved (`best_ckpt is None`) — mirroring train.py:625-628 exactly; an unconditional final save would out-step model-best in `_find_checkpoint`'s highest-step glob and evaluate() would silently score the wrong weights.**
  6. `config.json` via `resolved_config_dict(family="mgn", model=mgn, train=train_cfg, ...)` mirroring the CGN call site's exact keywords (`n_particle_types=mgn.node_type_size`).
- `_validate` is NOT touched (CGN-only; typed to LearnedSimulator).

- [ ] **Step 1: Read `train()` in full** (train.py ~311-660); bind the anchors (438/442 seam, 568-574 schedule, 625-628 final-save semantics, the config.json call site) to the real lines; hoisting is NOT expected — if the real seam differs, stop and report before improvising.
- [ ] **Step 2: Implement** (branch + narrowing line + `_lr_at` extraction + `_train_mgn`).
- [ ] **Step 3: Full suite** — the CGN regression IS this step's test (283+ passing, byte-identical behaviour); `_train_mgn` itself is gated by Task 6.
- [ ] **Step 4: gates + commit** — `feat(cli): mgn train dispatch — _train_mgn loop with inline validation`


### Task 5: Configs + carry-forwards + docs regen

**Files:**
- Create: `configs/deforming_plate/mgn.toml` (reference: `[run] benchmark="deforming_plate"`, seed; `[model] family="mgn"` + every `MGNConfig` field explicit (strict loader); `[train]` batch_size=2, lr_init=1e-4, training_steps=10_000_000, `w_pos=1.0`, `w_aux=1.0`, remaining `[train]` keys mirroring an existing benchmark's cgn.toml structure — bind to one and keep MGN-neutral values)
- Create: `configs/deforming_plate/mgn_smoke.toml` — **every `[model]` and non-derived `[train]` key explicit** (the strict loader defaults nothing), mirroring `mgn.toml`, with these overrides: hidden_dim=16, message_passing_steps=2, normalizer_warmup_steps=5, world_edge_radius=50.0, training_steps=50, batch_size=2, val_every=25
- Modify: `src/structbench/benchmarks/render.py` — **the landing-page Quickstart is ALREADY family-generic** (`spec.results[0].family` with a `"cgn"` no-results fallback, render.py:373-382) — leave it alone. The real ①-b carry-forward is the **dataset-archive "Using this archive" block**: an unconditional `configs/{name}/cgn.toml` hardcode at render.py:302-303 plus SPH-only wording (`sph/stress` Voigt note, 307-308) that is wrong for mesh benchmarks. Fix THAT block: reuse the same results-derived family fallback, and make the stress-derivation sentence conditional on the card's discretisation (mesh benchmarks read the nodal field directly).
- Modify: `src/structbench/datasets/canonical.py` — mesh-branch `time=response.time[:n].copy()` → `np.asarray(case.response.time[:n], dtype=np.float64)` (the SPH idiom; ①-b carry-forward).
- Regenerate: `docs/benchmarks.md` + `docs/benchmarks/*.md` (`python tools/gen_benchmark_docs.py`; then `--check` green).
- Test: config-load test for BOTH new TOMLs (they must pass the strict loader + the card equality check — add to the Task 3 test file).

- [ ] Steps: failing config-load tests → RED → write configs + the two small code edits + render change → regenerate docs → GREEN → full gates (incl. drift check) → commit — `feat(mgn): deforming_plate mgn configs; quickstart + loader carry-forwards`

---

### Task 6: End-to-end CPU smoke gate

**Files:**
- Create: `tests/cli/test_mgn_train_smoke.py`

**Interfaces:** consumes everything above. The gate: a REAL `train(...)` call, family `"mgn"`, on a tiny synthetic mesh benchmark, through at least one validation pass, on CPU, deterministic, < ~60 s.

- [ ] **Step 1: Write the gate**

```python
"""End-to-end MGN training smoke: train() -> validation -> checkpoint."""

import numpy as np
import torch

from structbench.benchmarks.card import BenchmarkCard
from structbench.benchmarks.registry import BenchmarkSpec
from structbench.cli.train import train
from structbench.config import MGNConfig, TrainConfig
from structbench.core import write_case
from structbench.core.io.meshgraphnets import build_deforming_plate_case
from structbench.eval import peak_nodal_aux, terminal_peak_displacement


def _mini_spec(case_ids: dict[str, list[str]]) -> BenchmarkSpec:
    qois = {
        "peak_vm_stress": peak_nodal_aux(exclude_types=(1, 3)),
        "terminal_peak_deflection": terminal_peak_displacement(exclude_types=(1, 3)),
    }
    card = BenchmarkCard(
        name="MgnSmoke",
        version="0.0",
        description="synthetic smoke benchmark",
        provenance="synthetic (test fixture)",
        data_license="n/a (synthetic test data)",
        solver="COMSOL",
        discretisation="FEM",
        materials=("synthetic",),
        loading="synthetic actuator",
        erosion=False,
        source_units="kg-m-s",
        geometry="synthetic tets",
        n_cases=sum(len(v) for v in case_ids.values()),
        splits={k: len(v) for k, v in case_ids.items()},
        task="smoke",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=tuple(qois),
        fields=("node/displacement", "node/von_mises_stress"),
        particles_per_case="6-6",
        n_frames=8,
        output_dt_ms=1.0,
        input_frames=2,
        protocol_rationale="synthetic smoke fixture; not a benchmark",
    )
    return BenchmarkSpec(
        card=card,
        splits={k: tuple(v) for k, v in case_ids.items()},
        eval_splits=("val",),
        aux_field="von_mises_stress",
        qois=qois,
        boundary_feature_fn=None,
        dataset_id="mgn-smoke",
        kinematic_types=(1, 3),
    )


def _write_cases(root, ids):
    rng = np.random.default_rng(11)
    for cid in ids:
        P, T = 6, 8
        w0 = rng.random((P, 3)).astype(np.float32)
        drift = rng.random((T, P, 3)).astype(np.float32) * 0.01
        arrays = {
            "cells": np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32),
            "node_type": np.array([0, 0, 0, 0, 1, 3], dtype=np.int32),
            "mesh_pos": w0.copy(),
            "world_pos": (w0[None] + np.cumsum(drift, axis=0)).astype(np.float32),
            "stress": rng.random((T, P, 1)).astype(np.float32),
        }
        case = build_deforming_plate_case(
            arrays, source_units="kg-m-s", case_id=cid
        )
        write_case(case, root / f"{cid}.h5")


def test_mgn_train_smoke(tmp_path):
    torch.manual_seed(0)
    ids = {
        "train": [f"train_{i:04d}" for i in range(4)],
        "val": ["val_0000"],
    }
    spec = _mini_spec(ids)
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_cases(data_root, [c for v in ids.values() for c in v])

    mgn = MGNConfig(
        hidden_dim=8,
        message_passing_steps=1,
        world_edge_radius=50.0,
        normalizer_warmup_steps=3,
    )
    tcfg = TrainConfig(
        benchmark="MgnSmoke", batch_size=2, training_steps=12, val_every=6,
        # remaining TrainConfig fields: bind to the real dataclass and give
        # small neutral values (lr_init=1e-3, w_pos=1.0, w_aux=1.0, seed=0, ...)
    )
    out = tmp_path / "run"
    train(spec, mgn, tcfg, data_root, out, "cpu", family="mgn")

    assert (out / "config.json").exists()
    ckpts = list(out.glob("model-*.pt"))
    assert ckpts, "no checkpoint written"
    # a validation pass genuinely ran: best starts at inf, so the first val
    # always writes model-best-<step>.pt (model-final alone == dead val loop)
    assert any(p.name.startswith("model-best-") for p in ckpts), "no val pass ran"
    # normalizers actually warmed up: reload and check accumulation happened
    from structbench.models.mgn import MeshSimulator

    sim = MeshSimulator(
        latent=8, mp_steps=1, world_edge_radius=50.0
    )
    sim.load(sorted(ckpts)[-1])
    assert int(sim._target_normalizer._n_accumulations) > 0
    assert int(sim._node_normalizer._n_accumulations) > 0  # feature warmup too


def test_mgn_evaluate_smoke(tmp_path, monkeypatch):
    """evaluate() on an MGN run dir: no stats file, bind/reset per case.

    Reuses the train smoke's artifacts by re-running the tiny train, then
    monkeypatching the registry lookup so evaluate() resolves the synthetic
    spec (precedent: tests/cli/test_train_eval.py's unregistered-benchmark
    monkeypatch, ~line 709 — bind to the real helper).
    """
    import structbench.cli.train as cli_train

    # ... build spec/data/run exactly as test_mgn_train_smoke (factor a helper) ...
    # monkeypatch.setattr(cli_train, "get_benchmark", lambda name: spec)
    # cli_train.evaluate(out_dir=out, data_root=data_root, split="val", ...)
    #   -- bind evaluate()'s REAL signature; assert metrics-val.json written and
    #      no normalization_stats.npz was ever required.
```

(Factor the spec/data/train setup into a module-level helper both tests share;
bind `evaluate()`'s real signature and the metrics filename from the code. The
evaluate smoke is REQUIRED — it is the only end-to-end coverage of Task 4a's
no-stats branch and per-case bind/reset.)

(Bind `TrainConfig`'s full required field set and `train`'s exact call shape to the real code — Task 4's report documents both. The test must construct everything the strict dataclasses require; no defaults invented. If `train()` requires `tcfg.benchmark == spec.card.name`, keep them equal as shown.)

- [ ] **Step 2: Run it** — if it fails, the failure is a REAL integration gap: fix in the owning module (smallest change), rerun that module's tests, and document the gap in the report. Do not weaken the gate.
- [ ] **Step 3: Full gates** (full suite, ruff, scoped format/mypy) → **commit** — `test(mgn): end-to-end train->validate->checkpoint smoke on synthetic mesh data`

---

## Deliberately OUT of this plan

- The 10M-step blessing run and the pooled-RMSE blessing evaluation harness (ADR-0043 §8 gate) — scheduled by the maintainer once Task 8 (units) lands; the pooled-convention aggregator is a small follow-up once real data exists.
- Transolver (②) and GeoFLARE (③); the per-method `provisional` registry flag + comparison-table rendering (its own small plan).
- Task 8 (human): units measurement → `SOURCE_UNITS`, `world_edge_radius`, card numerics.
- Parked ①-c1 minors (normalizer CPU syncs; orientation-blind fixture; `particle_types` arg; tripwire at `t==T+1`) — unchanged.

## Self-Review

**Recipe coverage (training side, ADR-0043 §8/§9a):** NORMAL-only noise 3e-3 with γ=1.0-by-construction (Task 4 §5 + `forward_train` docstring) ✓; equal-weight NORMAL-masked L2 with `w_pos`/`w_aux`=1.0 in the reference config ✓; scripted next-step velocity input in training from the window's own next frame ✓ (parity test pins train==eval features); normalizer warmup as accumulate-during-first-N-steps on features AND targets ✓ (Tasks 1/4, asserted in the smoke via reloaded state); batch 2 / lr 1e-4→1e-6 / 10M steps in `mgn.toml` ✓; self-contained MGN checkpoints (no stats file) ✓.

**The seam:** Task 1 is the extraction the final review demanded, WITH a parity test over ALL FIVE network inputs and a cross-example world-edge test — the train/eval divergence class AND the batched-radius-query class are structurally closed before the train loop exists. The evaluate() no-stats branch is end-to-end covered by the Task 6 evaluate smoke.

**Bind-steps:** Tasks 4–6 carry read-the-real-file steps for the cli internals; every such step states the full semantics here, and the verified grounding pins the load-bearing anchors (branch placement after the shared guards; the `CGNConfig(**...)` reconstruction site; the per-case loop; checkpoint naming; `collate_samples`' tolerance of extra keys).

**Type consistency:** `forward_train`'s signature is identical in Tasks 1, 4 §5, and the parity/smoke tests; `collate_mesh_samples(batch, statics)` consistent between Tasks 2 and 4 §4; `MGNConfig` field names consistent between Tasks 3, 4 §3/§7, and both TOMLs; `build_mgn_simulator` kwargs map `hidden_dim→latent`, `message_passing_steps→mp_steps`, `nmlp_layers→n_hidden` exactly once (Task 4 §3) and the smoke constructs the reload simulator with the same mapping.
