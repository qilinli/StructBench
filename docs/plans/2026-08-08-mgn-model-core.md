# MGN Model Core Implementation Plan (checkpoint ①-c1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A native `models/mgn/` implementing the ADR-0043 §8 MeshGraphNet recipe — mesh + world edge sets, online accumulating normalizers, first-order integration, scripted-actuator input — whose **untrained** simulator satisfies `eval`'s `_SimulatorLike` protocol end to end: `eval.rollout` and both `one_step_*` functions produce correctly-shaped results on a synthetic mesh case. **No training path in this plan** — `MGNConfig`, `MODEL_FAMILIES`, `cli/train.py` dispatch, the loss/noise recipe, windowing/collate extensions, and smoke configs are plan ①-c2.

**Architecture:** Four new modules under `src/structbench/models/mgn/`, self-contained (no imports from `models/cgn` — the world-edge search is implemented natively here, mirroring ADR-0020's "graph construction lives with the model that uses it"): `mesh_ops.py` (cells→unique bidirectional mesh edges; radius world edges excluding mesh-connected pairs), `normalizers.py` (online accumulating feature normalizers — required because MGN normalizes *edge* features, which the precomputed `compute_stats` pipeline cannot provide), `network.py` (encode-process-decode over two edge sets with residual edge+node updates), `simulator.py` (`MeshSimulator`: per-case binding of static mesh data + GT kinematic trajectory, step pointer with a ground-truth tripwire, first-order Euler integration, `_SimulatorLike`-conformant `predict_positions`).

**Tech Stack:** Python 3.12+, torch (+ `torch_geometric.nn.MessagePassing` optional — see Task 3; prefer plain scatter via `index_add_` to keep MGN dependency-light). No TensorFlow. No new dependencies.

**Plan ①-c1 of the v0.3 build order** (after ①-a ingestion and ①-b benchmark, both merged at `5d23b17`). ①-c2 (training path + family dispatch) follows.

## Global Constraints

- Python floor **3.12**; ruff line length **88** + `ruff format`; mypy `disallow_untyped_defs = true`; NumPy-style docstrings on every public API; `_`-prefix symbols private across module boundaries.
- **No new dependencies; no TensorFlow; no imports from `models/cgn`** (families stay independent; world-edge search implemented natively in `mesh_ops.py`).
- **ADR-0043 §8 recipe values (verified against the paper/meta.json — exact):** node features = **one-hot node type of width 9** (`NodeType.SIZE` in the source framework; codes present are {0,1,3}) plus, for scripted kinematic nodes only, the **next-step world-space velocity** (zeros for all other nodes) → `nnode_in = 9 + dim`; mesh-edge features `(u_ij, |u_ij|, x_ij, |x_ij|)` → width `2*dim + 2` (u from the **reference/mesh-space** coordinates, x from current world positions); world-edge features `(x_ij, |x_ij|)` → width `dim + 1`; world edges connect non-mesh-connected pairs within **r_W** — the simulator takes r_W in the *working frame* as a constructor parameter and the caller converts: working-frame value = `0.03 × f_length(source_units) × 1e3`, which is **30.0 mm only if the source is metre-native** (PROVISIONAL until Task 8 measures the units; the ①-c2 config is the source of truth); outputs = `(velocity, stress)` → width `dim + 1`; **first-order** integration `x_{t+1} = x_t + v̂`; 15 message-passing steps, latent 128, two-hidden-layer ReLU MLPs with LayerNorm on every MLP output **except the decoder's**.
- **Eval-protocol facts this plan builds on (verified 2026-08-08, 68/68 claims confirmed):** `predict_positions(position_sequence (P,F,dim), nparticles_per_example (1,), particle_types (P,)) -> (next_positions (P,dim), aux (P,n_aux))` with aux de-normalized, is the ONLY method `eval/` calls; no per-case context is passed — static data must be bound on the simulator between cases; at every predict call (rollout AND teacher-forced) the window's kinematic rows hold GT for frames `[t-F, t-1]` and frame `t`'s GT is not derivable from the arguments; `eval` consumes only `aux[:, 0]`.
- Working frame: positions/reference_coords in **mm**, aux (stress) in **MPa** — as produced by `load_case_trajectory`. All simulator math is working-frame.
- Tests: pytest, **synthetic-only**, deterministic (`torch.manual_seed` / `np.random.default_rng`); mesh fixtures built via `build_deforming_plate_case` → `write_case` → `load_case_trajectory` (the ①-b round-trip), or direct tensors for unit tests. RUN with the structbench conda env interpreter (PowerShell): `& "C:\Users\272766h\AppData\Local\miniconda3\envs\structbench\python.exe" -m pytest <target> -v`. Full suite (`python -m pytest -q`, baseline **260 passed / 6 skipped**) before each commit; ruff check / ruff format --check / **mypy on the new/changed mgn files only** — the repo-wide `mypy src` baseline carries 4 pre-existing `no-untyped-def` errors in `datasets/normalization.py:22,43` (and one ruff-format drift in `notch_beam_2d_impact/__init__.py`) that are NOT this plan's to fix; do not touch them.
- Branch: **`feat/mgn-model-core`** off `main` (@ `5d23b17`). Never commit to `main`; merge/push are human calls (ADR-0023).
- Commits: Conventional Commits; append to every commit message, after a blank line:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01HHpG1wUFUfb2Q31Hp948YX`

## File Structure

```
src/structbench/models/mgn/
  __init__.py        # CREATE: exports MeshSimulator, MGNet
  mesh_ops.py        # CREATE: cells_to_edges, world_edges
  normalizers.py     # CREATE: OnlineNormalizer (nn.Module, buffer-backed)
  network.py         # CREATE: build_mlp, MGNet (two-edge-set encode-process-decode)
  simulator.py       # CREATE: MeshSimulator (bind_case/reset_rollout/predict_positions)
tests/models/mgn/
  test_mgn_mesh_ops.py     # CREATE
  test_mgn_normalizers.py  # CREATE
  test_mgn_network.py      # CREATE
  test_mgn_simulator.py    # CREATE (incl. the eval-protocol integration test)
```

(Verified: `tests/models/` exists and contains `tests/models/cgn/`; there are NO `__init__.py` files anywhere under `tests/` and no conftest. Mirror the per-family convention with a plain `tests/models/mgn/` directory. Keep the `test_mgn_` basename prefix — without test-package `__init__.py` files pytest requires unique basenames repo-wide, and `tests/models/cgn/test_simulator.py` already exists.)

---

### Task 1: `mesh_ops.py` — mesh edges from cells; radius world edges with mesh exclusion

**Files:**
- Create: `src/structbench/models/mgn/__init__.py` (minimal: docstring only for now)
- Create: `src/structbench/models/mgn/mesh_ops.py`
- Test: `tests/models/mgn/test_mgn_mesh_ops.py`

**Interfaces:**
- Consumes: nothing (pure torch).
- Produces:
  - `cells_to_edges(cells: Tensor) -> Tensor` — `(n_cells, nodes_per_cell)` int64 → `(2, E)` int64 **bidirectional** unique edge index (both `(i,j)` and `(j,i)`; no self-loops; no duplicates). For a tetra, each cell contributes its 6 undirected vertex pairs.
  - `world_edges(positions: Tensor, radius: float, mesh_edge_index: Tensor) -> Tensor` — `(P, dim)` float → `(2, E_w)` int64: pairs with `|x_i - x_j| < radius`, excluding self-pairs AND pairs present in `mesh_edge_index` (either direction), bidirectional output. Brute-force `torch.cdist` in chunks is fine at deforming-plate scale (~1.3k nodes).

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/mgn/test_mgn_mesh_ops.py
import torch

from structbench.models.mgn.mesh_ops import cells_to_edges, world_edges


def _edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(s), int(r)) for s, r in edge_index.t()}


def test_cells_to_edges_single_tet():
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    e = cells_to_edges(cells)
    assert e.shape[0] == 2 and e.dtype == torch.int64
    es = _edge_set(e)
    # 6 undirected pairs x 2 directions, no self loops, no dupes
    assert len(es) == 12 and e.shape[1] == 12
    assert (0, 1) in es and (1, 0) in es and (2, 3) in es
    assert all(s != r for s, r in es)


def test_cells_to_edges_shared_face_dedup():
    # two tets sharing face (1,2,3): union of pairs, each counted once
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
    es = _edge_set(cells_to_edges(cells))
    # undirected pairs: {01,02,03,12,13,23} U {12,13,23,14,24,34} = 9 pairs
    assert len(es) == 18


def test_world_edges_radius_and_mesh_exclusion():
    #  nodes: 0-(0,0,0), 1-(1,0,0), 2-(10,0,0); mesh edge 0-1
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    mesh = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    w = world_edges(pos, radius=2.0, mesh_edge_index=mesh)
    ws = _edge_set(w)
    # 0-1 within radius but mesh-connected -> excluded; 2 is far from both
    assert ws == set()
    # without the mesh edge, 0-1 appears (both directions)
    w2 = world_edges(pos, radius=2.0, mesh_edge_index=torch.empty(2, 0, dtype=torch.int64))
    assert _edge_set(w2) == {(0, 1), (1, 0)}


def test_world_edges_one_directional_mesh_index_excludes_both_directions():
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mesh_one_dir = torch.tensor([[0], [1]], dtype=torch.int64)  # only 0->1
    w = world_edges(pos, radius=2.0, mesh_edge_index=mesh_one_dir)
    assert _edge_set(w) == set()  # 1->0 excluded too (symmetrized keys)


def test_world_edges_chunked_equals_unchunked(monkeypatch):
    import structbench.models.mgn.mesh_ops as mesh_ops

    torch.manual_seed(0)
    pos = torch.rand(7, 3)
    empty = torch.empty(2, 0, dtype=torch.int64)
    full = _edge_set(world_edges(pos, radius=0.6, mesh_edge_index=empty))
    monkeypatch.setattr(mesh_ops, "_QUERY_CHUNK", 2)  # force multi-chunk path
    chunked = _edge_set(world_edges(pos, radius=0.6, mesh_edge_index=empty))
    assert chunked == full  # row-offset arithmetic across chunks is correct
```

- [ ] **Step 2: Run to verify they fail** — `python -m pytest tests/models/mgn/test_mgn_mesh_ops.py -v`. Expected: FAIL (module does not exist; create `tests/models/` first if absent).

- [ ] **Step 3: Implement**

```python
# src/structbench/models/mgn/mesh_ops.py
"""Mesh-graph construction for the MGN baseline (ADR-0043 §8).

Graph construction lives with the model that uses it (ADR-0020 precedent).
"""

from __future__ import annotations

import torch
from torch import Tensor

_QUERY_CHUNK = 2048


def cells_to_edges(cells: Tensor) -> Tensor:
    """Unique bidirectional edge index from element connectivity.

    Parameters
    ----------
    cells:
        ``(n_cells, nodes_per_cell)`` int64 connectivity (0-indexed).

    Returns
    -------
    Tensor
        ``(2, E)`` int64 edge index containing every vertex pair of every
        cell in both directions, deduplicated, without self-loops.
    """
    k = cells.shape[1]
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    src = torch.cat([cells[:, a] for a, b in pairs])
    dst = torch.cat([cells[:, b] for a, b in pairs])
    und = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])  # (2, 2*n)
    und = und[:, und[0] != und[1]]
    return torch.unique(und, dim=1)


def world_edges(positions: Tensor, radius: float, mesh_edge_index: Tensor) -> Tensor:
    """Radius neighbourhood edges excluding mesh-connected pairs.

    Parameters
    ----------
    positions:
        ``(P, dim)`` world positions (working frame).
    radius:
        World-edge radius in the same frame as ``positions``.
    mesh_edge_index:
        ``(2, E)`` mesh edges whose pairs (either direction) are excluded.

    Returns
    -------
    Tensor
        ``(2, E_w)`` int64 bidirectional world-edge index (no self-loops).
    """
    n = positions.shape[0]
    rows: list[Tensor] = []
    cols: list[Tensor] = []
    for start in range(0, n, _QUERY_CHUNK):
        chunk = positions[start : start + _QUERY_CHUNK]
        dist = torch.cdist(chunk, positions)
        r, c = torch.nonzero(dist < radius, as_tuple=True)
        rows.append(r + start)
        cols.append(c)
    src, dst = torch.cat(rows), torch.cat(cols)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    # exclude mesh-connected pairs via a collision-free pair key; symmetrize the
    # mesh keys so the "either direction" contract holds even for a
    # one-directional mesh_edge_index
    key = src * n + dst
    m0, m1 = mesh_edge_index[0], mesh_edge_index[1]
    mesh_key = torch.cat([m0 * n + m1, m1 * n + m0])
    keep = ~torch.isin(key, mesh_key)
    return torch.stack([src[keep], dst[keep]]).to(torch.int64)
```

- [ ] **Step 4: Run to verify they pass**, then ruff/mypy on the new files.

- [ ] **Step 5: Commit** — `feat(mgn): mesh_ops — cells_to_edges + radius world_edges with mesh exclusion`

---

### Task 2: `normalizers.py` — online accumulating feature normalizer

**Files:**
- Create: `src/structbench/models/mgn/normalizers.py`
- Test: `tests/models/mgn/test_mgn_normalizers.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `class OnlineNormalizer(nn.Module)` — `__init__(size: int, max_accumulations: int = 10**6, std_epsilon: float = 1e-8)`; `forward(x: Tensor, accumulate: bool = False) -> Tensor` (normalize `(N, size)`; when `accumulate=True` and under the cap, update running sums first); `inverse(x: Tensor) -> Tensor`; buffer-backed (`_count`, `_sum`, `_sum_sq`, `_n_accumulations`, all `register_buffer(..., persistent=True)`) so `state_dict` round-trips through save/load. **Semantics pinned:** `_n_accumulations` increments by **1 per `accumulate=True` call** (counts batches/steps, mirroring the source framework's Normalizer); `_count` separately accumulates **sample rows** (`x.shape[0]`). `std = sqrt(clamp(_sum_sq/_count − mean², min=0))` then `max(std, std_epsilon)` — the clamp guards float cancellation on near-constant features. With zero accumulations: mean 0, std 1 (identity) — an untrained simulator must run.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/mgn/test_mgn_normalizers.py
import torch

from structbench.models.mgn.normalizers import OnlineNormalizer


def test_identity_before_any_accumulation():
    n = OnlineNormalizer(size=3)
    x = torch.randn(5, 3)
    torch.testing.assert_close(n(x), x)
    torch.testing.assert_close(n.inverse(x), x)


def test_converges_to_moments_and_inverts():
    torch.manual_seed(0)
    n = OnlineNormalizer(size=2)
    data = torch.randn(1000, 2) * torch.tensor([3.0, 0.5]) + torch.tensor([1.0, -2.0])
    n(data, accumulate=True)
    out = n(data)
    assert abs(float(out.mean())) < 0.05
    assert abs(float(out.std()) - 1.0) < 0.05
    torch.testing.assert_close(n.inverse(n(data)), data, rtol=1e-4, atol=1e-4)


def test_accumulation_cap_counts_calls_not_samples():
    # cap = 2 CALLS; each call carries 10 samples. Third call must be ignored.
    n = OnlineNormalizer(size=1, max_accumulations=2)
    n(torch.zeros(10, 1), accumulate=True)          # call 1: counts
    n(torch.ones(10, 1), accumulate=True)           # call 2: counts (20 samples ok)
    assert int(n._count) == 20                      # samples accumulated across 2 calls
    mean_before = (n._sum / n._count).clone()
    n(torch.full((10, 1), 100.0), accumulate=True)  # call 3: over cap, ignored
    assert int(n._count) == 20
    torch.testing.assert_close(n._sum / n._count, mean_before)


def test_constant_feature_column_stays_finite():
    n = OnlineNormalizer(size=2)
    x = torch.cat([torch.full((50, 1), 3.0), torch.randn(50, 1)], dim=1)
    n(x, accumulate=True)
    out = n(x)  # constant column: variance clamps to 0 -> std_epsilon, no NaN
    assert torch.isfinite(out).all()


def test_state_dict_roundtrip():
    n = OnlineNormalizer(size=2)
    n(torch.randn(50, 2) + 5.0, accumulate=True)
    m = OnlineNormalizer(size=2)
    m.load_state_dict(n.state_dict())
    x = torch.randn(4, 2)
    torch.testing.assert_close(n(x), m(x))
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement** per the pinned interface semantics: buffers via `register_buffer(..., persistent=True)`; on `accumulate=True` and `_n_accumulations < max_accumulations`, add `x.sum(0)` to `_sum`, `(x*x).sum(0)` to `_sum_sq`, `x.shape[0]` to `_count`, and increment `_n_accumulations` by 1 (per call); `forward` computes `(x - mean) / std` with `mean = _sum/_count`, `std = max(sqrt(clamp(_sum_sq/_count - mean**2, min=0)), std_epsilon)`; with `_count == 0`, mean 0 / std 1 (identity).

- [ ] **Step 4: Run to verify they pass**; ruff/mypy; full `tests/models`.

- [ ] **Step 5: Commit** — `feat(mgn): online accumulating feature normalizer`

---

### Task 3: `network.py` — two-edge-set encode-process-decode

**Files:**
- Create: `src/structbench/models/mgn/network.py`
- Test: `tests/models/mgn/test_mgn_network.py`

**Interfaces:**
- Consumes: nothing (pure torch; use `index_add_`/gather for message passing — do NOT depend on `torch_geometric` here, keeping MGN's core PyG-free).
- Produces:
  - `build_mlp(in_size: int, hidden: int, n_hidden: int, out_size: int, *, layer_norm: bool) -> nn.Sequential` — ReLU MLP with `n_hidden` hidden layers; `LayerNorm(out_size)` appended iff `layer_norm`.
  - `class MGNet(nn.Module)` — `__init__(node_in: int, mesh_edge_in: int, world_edge_in: int, out_size: int, latent: int = 128, mp_steps: int = 15, n_hidden: int = 2)`. `forward(node_feats (P, node_in), mesh_edge_index (2,Em), mesh_edge_feats (Em, mesh_edge_in), world_edge_index (2,Ew), world_edge_feats (Ew, world_edge_in)) -> (P, out_size)`. Encoders (one per feature kind, LayerNorm on); `mp_steps` processor blocks, each updating mesh-edge latents, world-edge latents, then node latents (edge MLP on `[e, v_src, v_dst]`; node MLP on `[v, sum(mesh msgs), sum(world msgs)]`; **residual** adds on all three, per the MGN architecture); decoder MLP (LayerNorm OFF). Empty world-edge set (`Ew == 0`) must work (zero message contribution).

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/mgn/test_mgn_network.py
import torch

from structbench.models.mgn.network import MGNet, build_mlp


def _net(**kw):
    torch.manual_seed(0)
    defaults = dict(
        node_in=12, mesh_edge_in=8, world_edge_in=4, out_size=4,
        latent=16, mp_steps=2, n_hidden=2,
    )
    defaults.update(kw)
    return MGNet(**defaults)


def test_forward_shapes():
    net = _net()
    P, Em, Ew = 5, 12, 4
    out = net(
        torch.randn(P, 12),
        torch.randint(0, P, (2, Em)), torch.randn(Em, 8),
        torch.randint(0, P, (2, Ew)), torch.randn(Ew, 4),
    )
    assert out.shape == (P, 4)


def test_forward_empty_world_edges():
    net = _net()
    P, Em = 5, 12
    out = net(
        torch.randn(P, 12),
        torch.randint(0, P, (2, Em)), torch.randn(Em, 8),
        torch.empty(2, 0, dtype=torch.int64), torch.empty(0, 4),
    )
    assert out.shape == (P, 4)


def test_message_passing_propagates_information():
    # node 0's input feature must influence node 1's output via the edge 0->1
    net = _net(mp_steps=1)
    nf = torch.zeros(2, 12)
    mesh = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    ef = torch.zeros(2, 8)
    base = net(nf, mesh, ef, torch.empty(2, 0, dtype=torch.int64), torch.empty(0, 4))
    nf2 = nf.clone(); nf2[0, 0] = 10.0
    pert = net(nf2, mesh, ef, torch.empty(2, 0, dtype=torch.int64), torch.empty(0, 4))
    assert not torch.allclose(base[1], pert[1])


def test_build_mlp_layernorm_toggle():
    with_ln = build_mlp(3, 8, 2, 5, layer_norm=True)
    without = build_mlp(3, 8, 2, 5, layer_norm=False)
    assert isinstance(list(with_ln.children())[-1], torch.nn.LayerNorm)
    assert not isinstance(list(without.children())[-1], torch.nn.LayerNorm)
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement.** Message convention: `edge_index[0] = sender`, `edge_index[1] = receiver`; per-edge update `e' = e + MLP([e, v_sender, v_receiver])`; node aggregation `sum` via `torch.zeros(P, latent).index_add_(0, receiver, messages)`; node update `v' = v + MLP([v, agg_mesh, agg_world])`. Keep every sub-MLP `latent`-wide.

- [ ] **Step 4: Run to verify they pass**; ruff/mypy.

- [ ] **Step 5: Commit** — `feat(mgn): MGNet two-edge-set encode-process-decode`

---

### Task 4: `simulator.py` — `MeshSimulator` with per-case binding and the GT tripwire

**Files:**
- Create: `src/structbench/models/mgn/simulator.py`
- Modify: `src/structbench/models/mgn/__init__.py` (export `MeshSimulator`, `MGNet`)
- Test: `tests/models/mgn/test_mgn_simulator.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces `class MeshSimulator(nn.Module)`:
  - `__init__(dim: int = 3, latent: int = 128, mp_steps: int = 15, n_hidden: int = 2, node_type_size: int = 9, kinematic_types: tuple[int, ...] = (1, 3), scripted_types: tuple[int, ...] = (1,), world_edge_radius: float = 30.0, device: str = "cpu")` — builds `MGNet(node_in=node_type_size + dim, mesh_edge_in=2*dim + 2, world_edge_in=dim + 1, out_size=dim + 1, ...)` plus four `OnlineNormalizer`s: nodes (`node_type_size + dim`), mesh edges, world edges, targets (`dim + 1`).
  - `bind_case(cells: Tensor, reference_coords: Tensor, particle_types: Tensor, kinematic_positions: Tensor) -> None` — caches `cells_to_edges(cells)`, mesh-space `u_ij` gather indices, one-hot node types, the kinematic mask, and the full GT positions `(T, P, dim)` restricted use: only kinematic rows are ever read. Resets the step pointer.
  - `reset_rollout() -> None` — pointer back to `None` (re-anchors on the next predict call).
  - `predict_positions(current_positions (P, F, dim), nparticles_per_example, particle_types) -> (next_positions (P, dim), aux (P, 1))`:
    1. `x_t = current_positions[:, -1]` (last window frame).
    2. **Anchor/advance the pointer — deterministic, not search-based:** on the first call after `bind_case`/`reset_rollout`, set `t = F` where `F = current_positions.shape[1]` — both eval entry points verifiably start at the trajectory head (rollout seeds frames `[0, F)` and loops from `t = F`, rollout.py:139/145; the one-step functions stride all-GT windows from `t = F`, rollout.py:255-256), so the first window's last frame is always GT frame `F-1`. Subsequent calls advance `t` by 1. **Tripwire (verification only, never anchoring):** at every call verify `x_t[kin] ≈ GT[t-1][kin]` (atol 1e-4); on mismatch raise `RuntimeError` whose message names both likely causes ("call reset_rollout() before each eval pass, and ensure bind_case matches the trajectory being evaluated"). This is exact even when kinematic rows are stationary across frames (a HANDLE node never moves; the actuator may pause) — the failure mode a search-based anchor would have. With no kinematic nodes, pointer logic is skipped and the scripted-velocity input is zeros.
    3. Node features: `[one_hot(types, 9) | scripted_velocity]` where `scripted_velocity = GT[t][scripted] - x_t[scripted]` on scripted rows, zeros elsewhere (if `t` is beyond the bound trajectory, zeros — final-frame guard).
    4. Edge features: mesh `(u_ij, |u_ij|, x_ij, |x_ij|)` from bound reference coords + `x_t`; world edges recomputed from `x_t` each call via `world_edges(x_t, world_edge_radius, mesh_edge_index)`.
    5. Normalize features (no accumulation at inference), run `MGNet`; **apply `target_normalizer.inverse()` to the full `(P, dim+1)` network output FIRST, then slice** columns `[:dim]` (velocity) and `[dim:]` (stress, kept `(P,1)` for aux) — slicing before inverse would broadcast against the 4-wide std buffers; integrate `x_{t+1} = x_t + v̂`, return `(x_{t+1}, σ̂ (P,1))`.
  - `save(path)` / `load(path)` — plain `state_dict` round-trip (normalizer buffers included).
  - Class docstring MUST document the statefulness contract loudly: bind per case, reset before *each* eval pass (rollout / one-step-position / one-step-aux), tripwire semantics.

- [ ] **Step 1: Write the failing tests** (unit level; the eval integration is Task 5)

```python
# tests/models/mgn/test_mgn_simulator.py
import numpy as np
import pytest
import torch

from structbench.models.mgn import MeshSimulator


def _bound_sim(T=6, P=5, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sim = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=0.5)
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
    ref = torch.tensor(rng.random((P, 3)), dtype=torch.float32)
    types = torch.tensor([0, 0, 1, 3, 0], dtype=torch.int64)
    gt = torch.tensor(rng.random((T, P, 3)), dtype=torch.float32).cumsum(0)
    sim.bind_case(cells, ref, types, gt)
    return sim, gt, types


def test_predict_shapes_and_pointer_advance():
    sim, gt, types = _bound_sim()
    npp = torch.tensor([5])
    win = gt[0:2].permute(1, 0, 2).contiguous()   # frames [0,1] -> predict frame 2
    nxt, aux = sim.predict_positions(win, npp, types)
    assert nxt.shape == (5, 3) and aux.shape == (5, 1)
    # next call must accept the GT-overwritten window for frame 3
    win2 = torch.stack([gt[1], gt[2]], dim=1)     # (P, 2, dim), kin rows GT
    nxt2, _ = sim.predict_positions(win2, npp, types)
    assert nxt2.shape == (5, 3)


def test_tripwire_fires_on_desynced_window():
    sim, gt, types = _bound_sim()
    npp = torch.tensor([5])
    sim.predict_positions(gt[0:2].permute(1, 0, 2).contiguous(), npp, types)
    stale = gt[0:2].permute(1, 0, 2).contiguous()  # same window again: t desync
    with pytest.raises(RuntimeError, match="reset_rollout"):
        sim.predict_positions(stale, npp, types)
    sim.reset_rollout()
    nxt, _ = sim.predict_positions(stale, npp, types)  # re-anchors cleanly
    assert nxt.shape == (5, 3)


def test_tripwire_fires_on_foreign_window_at_first_call():
    # Anchoring is deterministic (t = F), so a window from a DIFFERENT
    # trajectory must trip the GT check on the very first call.
    sim, gt, types = _bound_sim()
    sim.reset_rollout()
    foreign = torch.rand(5, 2, 3) + 50.0  # nowhere near the bound GT
    with pytest.raises(RuntimeError, match="bind_case"):
        sim.predict_positions(foreign, torch.tensor([5]), types)


def test_no_kinematic_nodes_runs_stateless():
    torch.manual_seed(0)
    sim = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=0.5)
    P = 4
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    ref = torch.rand(P, 3)
    types = torch.zeros(P, dtype=torch.int64)  # all NORMAL
    gt = torch.rand(3, P, 3)
    sim.bind_case(cells, ref, types, gt)
    for _ in range(4):  # arbitrary call count, no tripwire without kin rows
        nxt, aux = sim.predict_positions(torch.rand(P, 2, 3), torch.tensor([P]), types)
    assert nxt.shape == (P, 3) and aux.shape == (P, 1)


def test_save_load_roundtrip(tmp_path):
    sim, gt, types = _bound_sim()
    p = tmp_path / "mgn.pt"
    sim.save(p)
    sim2 = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=0.5)
    sim2.load(p)
    for (k1, v1), (k2, v2) in zip(
        sim.state_dict().items(), sim2.state_dict().items()
    ):
        assert k1 == k2
        torch.testing.assert_close(v1, v2)
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement `MeshSimulator`** per the interface above. Care points: the pointer is anchored deterministically (`t = F` on the first call after bind/reset — never by searching the GT); the tripwire compares ONLY kinematic rows; scripted-velocity uses `GT[t]` (the frame being predicted); the final predictable frame guard returns zeros for scripted velocity when `t >= T`; `register_buffer` vs plain attributes — bound case tensors are **plain attributes** (not part of `state_dict`; a checkpoint is case-independent).

- [ ] **Step 4: Run to verify they pass**; ruff/mypy on all `models/mgn` files.

- [ ] **Step 5: Commit** — `feat(mgn): MeshSimulator — per-case binding, GT tripwire, first-order integration`

---

### Task 5: eval-protocol integration test + full gates

**Files:**
- Test: `tests/models/mgn/test_mgn_simulator.py` (append the integration test)

**Interfaces:**
- Consumes: everything above + `build_deforming_plate_case`, `write_case`, `load_case_trajectory`, `eval.rollout` (called WITH `qois=get_benchmark("deforming_plate").qois` so the ADR-0043 QoI path is exercised against a mesh trajectory), `eval.one_step_position_rmse`, `eval.one_step_aux_rmse`.

- [ ] **Step 1: Write the failing integration test**

```python
def test_untrained_mgn_through_eval_rollout(tmp_path):
    """The ①-c1 deliverable gate: an UNTRAINED MeshSimulator satisfies the
    eval protocol end to end on a synthetic mesh case (ADR-0043)."""
    import numpy as np
    from structbench.core import write_case
    from structbench.core.io.meshgraphnets import build_deforming_plate_case
    from structbench.datasets import load_case_trajectory
    from structbench.eval import one_step_aux_rmse, one_step_position_rmse, rollout

    rng = np.random.default_rng(7)
    P, T = 6, 8
    world0 = rng.random((P, 3)).astype(np.float32)
    drift = rng.random((T, P, 3)).astype(np.float32) * 0.01
    arrays = {
        "cells": np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32),
        "node_type": np.array([0, 0, 0, 0, 1, 3], dtype=np.int32),
        "mesh_pos": world0.copy(),
        "world_pos": (world0[None] + np.cumsum(drift, axis=0)).astype(np.float32),
        "stress": rng.random((T, P, 1)).astype(np.float32),
    }
    case = build_deforming_plate_case(arrays, source_units="kg-m-s", case_id="mgn-it")
    path = tmp_path / "mgn-it.h5"
    write_case(case, path)
    traj = load_case_trajectory(path, aux_field="von_mises_stress")

    torch.manual_seed(0)
    sim = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=50.0)
    sim.bind_case(
        torch.from_numpy(traj.cells),
        torch.from_numpy(traj.reference_coords),
        torch.from_numpy(traj.particle_type),
        torch.from_numpy(traj.positions),
    )

    from structbench.benchmarks import get_benchmark

    spec = get_benchmark("deforming_plate")
    sim.reset_rollout()
    result = rollout(
        sim, traj, input_frames=2, kinematic_types=(1, 3), qois=spec.qois
    )
    assert result.predicted_positions.shape == (T, P, 3)
    assert result.predicted_aux.shape == (T, P)
    assert result.position_rmse.shape == (T - 2,)
    # the ADR-0043 QoIs evaluate on MGN output: populated, finite
    assert set(result.qoi_pred) == {"peak_vm_stress", "terminal_peak_deflection"}
    assert all(np.isfinite(v) for v in result.qoi_pred.values())
    assert all(np.isfinite(v) for v in result.qoi_error.values())
    # kinematic rows are GT-prescribed in the output
    np.testing.assert_allclose(
        result.predicted_positions[:, 4], traj.positions[:, 4], rtol=1e-5
    )

    sim.reset_rollout()
    one_pos = one_step_position_rmse(sim, traj, input_frames=2, kinematic_types=(1, 3))
    assert one_pos.shape == (T - 2,)

    sim.reset_rollout()
    one_aux = one_step_aux_rmse(sim, traj, input_frames=2, kinematic_types=(1, 3))
    assert one_aux.shape == (T - 2,)
```

- [ ] **Step 2: Run to verify it fails** (before Task 4 is complete it cannot run; after Task 4 it should pass immediately — if it fails, the failure is a real protocol gap: fix in the owning module, report which).

- [ ] **Step 3: Full gates** — `python -m pytest -q` (baseline 260/6 + all new `tests/models/mgn` tests), `ruff check src tests`, then `ruff format --check` and `mypy` **on the new `src/structbench/models/mgn` and `tests/models/mgn` files only** (the repo-wide baselines carry the two pre-existing hygiene items named in Global Constraints — not this plan's).

- [ ] **Step 4: Commit** — `test(mgn): untrained MeshSimulator satisfies the eval protocol end to end`

---

## Deliberately OUT of this plan (①-c2 and later)

- `MGNConfig`, `MODEL_FAMILIES["mgn"]`, `cli/train.py` family dispatch (build/evaluate/reconstruction), the `reset_rollout()` call sites in `evaluate()`/`_validate` loops.
- Training: velocity-space loss (NORMAL-masked, §9a declared choice), noise 3e-3 with γ=1.0 target correction, normalizer accumulation schedule (1000-step warmup), optimizer/schedule, checkpoint naming, `WindowDataset`/collate extensions (`traj_idx` key + MGN collate with node/edge offsets), smoke configs.
- Carry-forwards ledgered in ①-b: generated Quickstart template family fix; mesh-branch time-dtype idiom; `available_aux_fields` SPH-only validation debt.
- Task 8 (human): real-data download + units; the blessing run (compute scheduling is the maintainer's, ADR-0043 §8).

## Self-Review

**Recipe coverage (ADR-0043 §8):** one-hot(9)+scripted-velocity node features ✓ (Task 4); mesh-edge `(u,|u|,x,|x|)` + world-edge `(x,|x|)` widths ✓ (Tasks 3–4); r_W as working-frame parameter with the 0.03→30 mm conversion documented ✓ (constraints + Task 4 default); first-order integration ✓; 15 MP / latent 128 defaults with test-scale overrides ✓; LayerNorm-except-decoder ✓ (Task 3); online normalizers ✓ (Task 2); undirected world edges excluding mesh pairs, no type filtering ✓ (Task 1). Training-side recipe items are explicitly ①-c2.

**Protocol facts honoured:** predict-only interface, per-case binding between trajectories, both-eval-passes-start-at-head fact → **deterministic anchor `t = F`** with the GT match demoted to a pure tripwire (Task 4 — exact even for stationary kinematic rows), reset-per-eval-pass documented and exercised (Task 5), `aux[:,0]` consumption (aux returned `(P,1)`), the ADR-0043 QoIs exercised through `rollout(qois=...)` in the integration gate.

**Placeholder scan:** none — every step carries runnable code; Task 3 Step 3's prose spec is backed by the exact update equations and the four tests that pin them.

**Type consistency:** `cells_to_edges`/`world_edges` signatures used identically in Tasks 1 and 4; `OnlineNormalizer(size, ...)` sizes match the four instantiations in Task 4; `MGNet` constructor/forward match between Tasks 3 and 4; `MeshSimulator(latent=8, mp_steps=1, world_edge_radius=...)` consistent across all Task 4–5 tests.
