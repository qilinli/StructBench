# Transolver Provisional (ADR-0041 step ②) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A native, provisional Transolver family (`models/transolver`) trainable and evaluable on the DeformingPlate benchmark via `structbench-train`, sharing the MGN task harness (noise, loss, rollout, artifacts) with a faithfully ported Physics-Attention backbone.

**Architecture:** Physics-Attention transformer (Wu et al., ICML 2024) ported pure-torch from the official `thuml/Transolver` irregular-mesh variant, wrapped in a `TransolverSimulator` that mirrors `MeshSimulator`'s stateful rollout contract. The method-agnostic state machinery (bind/reset/tripwire/scripted-velocity/save-load) is first extracted from `MeshSimulator` into a shared base class in `models/common/` (move-only; MGN behaviour bit-identical). Ragged-N batching is solved by per-example segment computation — mathematically identical to thuml's batch=1, no padding, no masks.

**Tech Stack:** Python 3.12, PyTorch only (no einops, no timm, no PyG for this family), pytest, ruff, mypy. Research basis: `scratch/2026-08-08-transolver-grounding.md` (147 claims, 136 confirmed; §-references below point there).

## Global Constraints

- **Working frame:** mm / MPa (`datasets/canonical.py` scales m→mm ×1e3, Pa→MPa ×1e-6). All model I/O is in this frame.
- **Protocol pins (ADR-0043):** `input_frames = 2` (card-enforced by `load_run_config`); `kinematic_types = (1, 3)`; `scripted_types = (1,)` (OBSTACLE only); noise `N(0, 0.003)` NORMAL-masked with γ=1-by-construction (noise added to `x_last` by the caller; targets measured from the noisy position); output layout `(P, dim+1)` velocity-then-aux; first-order Euler; loss = NORMAL-masked `w_pos·‖Δv‖² + w_aux·Δaux²` mean.
- **MGN must not change behaviour.** Task 1's extraction is move-only: the full existing test suite (309+ tests incl. `tests/models/mgn/`, `tests/cli/test_mgn_train_smoke.py`) must pass unmodified (import-path edits only if a test imports a moved private name). MGN's public API (`MeshSimulator` constructor, `bind_case`, `reset_rollout`, `predict_positions`, `forward_train` signatures) is frozen.
- **Faithful-to-released-code** (ADR-0044 draft, Task 8): temperature learnable per-head init 0.5 **unclamped**; placeholder added **unconditionally**; `mlp_ratio=1`, GELU, dropout 0.0; `in_project_slice` gets `orthogonal_` init then is **overwritten** by the global `trunc_normal_(std=0.02)` pass (thuml ordering quirk — replicate, document); no `Time_Input` path.
- **No new dependencies.** The one einops call in the reference is a head-merge `rearrange('b h n d -> b n (h d)')` = `permute+reshape`; `timm`'s `trunc_normal_` = `torch.nn.init.trunc_normal_`.
- **Attribution:** module docstrings in `models/transolver/` credit "Wu et al., ICML 2024 (arXiv:2402.02366); reference implementation github.com/thuml/Transolver, MIT License, Copyright (c) 2024 THUML @ Tsinghua University".
- **Conventions:** ruff (line-length 88, E/F/I/UP/B), mypy strict (`disallow_untyped_defs`), NumPy-style docstrings citing governing ADRs, TDD, one commit per task minimum. Format-check before every commit: `ruff format --check . && ruff check . && mypy src`.
- **Tests are synthetic-only** — never require `STRUCTBENCH*_DATA_ROOT` env vars; follow the fixture style of `tests/cli/test_mgn_train_smoke.py` (tiny synthetic mesh case).

## Design decisions (closures of grounding-doc §8 gaps — Task 8 turns these into ADR-0044)

| Gap | Decision |
|---|---|
| 1, 2 — featurization | Per-node input = `[one_hot(node_type, 9), scripted_velocity (3), x_t (3), reference_coords (3)]` = **18 channels**. Information-parity with MGN: same one-hot + scripted-velocity features; geometry that MGN gets via edges enters as absolute current + reference coordinates (Transolver's native convention — Elasticity inputs are raw coords, grounding §4.2). Contact must be learned from coordinates; vanilla Transolver has no proximity mechanism — an accepted, documented off-native cost (provisional). No history velocity: source task is h=0 (grounding §5.4/C21); `x_last = position_seq[:, -1]` exactly as MGN. |
| 3 — hyperparameters | Elasticity irregular-mesh reference (grounding §4.1): L=8, C=128, heads=8, M=64, mlp_ratio=1, dropout=0.0, GELU. Only irregular-mesh config published; deviation risk ledgered in ADR-0044. |
| 4 — ragged-N batching | **Per-example segment computation** keyed on `n_particles_per_example`: slice-weight computation is pointwise; aggregation (Eq 2), token attention (Eq 3), and deslice (Eq 4) run per contiguous example segment in a Python loop (B ≤ 8, M = 64 — loop cost negligible). Mathematically identical to thuml batch=1 per example; reuses `collate_mesh_samples` unchanged; killer test = batched forward ≡ per-example forwards. |
| 5 — output | `(P, dim+1)` velocity-then-stress, forward-Euler, target normalizer — identical to MGN (grounding c15/c32/c34), maximizing harness reuse. Crash paper's acceleration+Verlet rejected: deforming_plate is quasi-static; MGN's own recipe uses velocity. |
| 6 — stabilization | GNS noise σ=3e-3 NORMAL-masked, γ=1-by-construction — the field's revealed preference on deforming_plate (grounding §5.3/C24); reuses the `_train_mgn` noise block verbatim. AR-RT/BPTT rejected for v0.3 (new training-loop machinery, out of provisional scope). |
| 7 — done | This plan is done when: all tasks review-clean, full suite green, smoke config trains+evaluates end-to-end. The eventual training run is maintainer compute; its number is recorded **provisional, no numeric gate** (no published vanilla-Transolver deforming_plate number exists — grounding §5.1/C18). |
| 8 — optimizer | Method-native recipe: **AdamW(weight_decay=1e-5) + global-norm grad-clip 0.1 + cosine anneal** `lr_init → LR_SCHEDULE_FLOOR` over `training_steps` (steps-port of Elasticity's per-epoch CosineAnnealingLR). `weight_decay`/`max_grad_norm` live on **`TransolverConfig`** (precedent: `mgn.noise_std`, `mgn.normalizer_warmup_steps` — family-recipe knobs on the model config; zero churn to `TrainConfig`/existing TOMLs, which the strict loader would otherwise all break). Reference budget matched to MGN for comparability: **batch 2, 10M steps** (budget only). `lr_init = 1e-3` is Transolver's own Elasticity-reference LR — deliberately NOT matched to MGN's 1e-4: each method keeps its native optimizer recipe (AdamW+cosine vs Adam+exponential), so the comparison is same-task/same-data/same-budget, not same-optimizer. `[train] lr_decay` is present-but-unused for this family (schema is family-uniform) — commented in the config. |
| 9 — licence | Clean-room reimplementation from the grounding doc's math, not a code copy; MIT attribution in docstrings regardless (Global Constraints). NOTICE-file question routed to maintainer at plan presentation (repo-root file = flag-first). |
| 10, 12 — simulator/shared scaffolding | Extract `CaseBoundSimulator` (state machinery only) into `models/common/simulator_base.py`; `MeshSimulator` and `TransolverSimulator` subclass it; `cli/train.py:1173`'s `isinstance(MeshSimulator)` gate widens to `CaseBoundSimulator`. `predict_positions`/`forward_train` stay per-family (~40 lines each; the risky shared semantics is the state machinery). No new dataset/collate abstractions: `WindowDataset` + `collate_mesh_samples` already serve (Transolver ignores `mesh_edge_index`, uses `reference_coords`). GeoFLARE (step ③) reuses the base. |
| 11 — starting state | Step ① is code-complete and merged (main @ f812dab); canonical archive exists (1,200 cases). MGN *blessing run* is pending maintainer compute — not a code dependency; maintainer directed step ② to proceed. |
| 13 — comparison statistic | The §5 leaderboard per-step-mean metrics that `evaluate()` already emits for every family. `tools/blessing_pooled_rmse.py` runs on any run dir (informational for Transolver). Nothing to build here; comparison table is a separate plan. |
| 14 — einops/init | Pure torch; faithful-to-released-code init (see Global Constraints). |

## File Structure

- Create: `src/structbench/models/common/__init__.py`, `src/structbench/models/common/simulator_base.py`
- Create: `src/structbench/models/transolver/__init__.py`, `network.py`, `simulator.py`
- Modify: `src/structbench/config.py` (TransolverConfig + registry), `src/structbench/cli/train.py` (dispatch, `_train_transolver`, `_lr_at_cosine`, `build_transolver_simulator` — lives here beside `build_mgn_simulator` — evaluate arm, isinstance gate, and the type-annotation widenings listed in Tasks 5/6), `src/structbench/models/mgn/simulator.py` (inherit base)
- Create: `configs/deforming_plate/transolver.toml`, `configs/deforming_plate/transolver_smoke.toml`
- Create: `decisions/0044-transolver-provisional-adaptation.md` (draft; maintainer finalises)
- Modify: `docs/ARCHITECTURE.md` (models/ section: mgn + transolver + common)
- Tests: `tests/models/common/test_simulator_base.py`, `tests/models/transolver/test_network.py`, `tests/models/transolver/test_simulator.py`, `tests/cli/test_transolver_train_smoke.py`, additions to `tests/cli/test_train_config.py`

---

### Task 1: Extract `CaseBoundSimulator` base (move-only) and widen the evaluate() gate

**Files:**
- Create: `src/structbench/models/common/__init__.py`, `src/structbench/models/common/simulator_base.py`
- Modify: `src/structbench/models/mgn/simulator.py`, `src/structbench/cli/train.py` (line ~1173)
- Test: `tests/models/common/test_simulator_base.py` (new, minimal — the real gate is the existing mgn suite)

**Interfaces:**
- Produces: `class CaseBoundSimulator(torch.nn.Module)` with the constructor state `(dim, node_type_size, kinematic_types, scripted_types, device)` (including the `scripted_types ⊆ kinematic_types` ValueError), the bind-state attributes, and these methods moved **verbatim** from `MeshSimulator`:
  - `bind_case(self, cells, reference_coords, particle_types, kinematic_positions) -> None` — everything `MeshSimulator.bind_case` (simulator.py:171-209) does today EXCEPT `mesh_edge_index = cells_to_edges(cells)`; at its end calls `self._on_bind_case(cells)` (hook, default no-op) so MGN builds edges there.
  - `reset_rollout(self)` (211-213), the pointer/tripwire logic (277-292), the scripted-velocity-at-eval computation (294-302 → a protected helper `_eval_scripted_velocity(self, x_t) -> Tensor` returning the `(P, dim)` feature and advancing nothing), the training-time scripted-velocity helper `_train_scripted_velocity(self, x_last, next_positions, particle_types) -> Tensor` (545-549 pattern), and `save`/`load` (580-598).
- Consumes: nothing new. `MeshSimulator(CaseBoundSimulator)` keeps its exact public API and all 13 existing simulator tests.

**Steps:**

- [ ] **Step 1: Write the base-class test** — `tests/models/common/test_simulator_base.py`:

```python
"""CaseBoundSimulator base contract (extracted from MeshSimulator; ADR-0044)."""

import pytest
import torch

from structbench.models.common import CaseBoundSimulator
from structbench.models.mgn import MeshSimulator


def test_mesh_simulator_is_case_bound() -> None:
    assert issubclass(MeshSimulator, CaseBoundSimulator)


def test_scripted_subset_validation() -> None:
    with pytest.raises(ValueError, match="scripted_types"):
        CaseBoundSimulator(
            dim=3, node_type_size=9, kinematic_types=(3,), scripted_types=(1,)
        )


def test_reset_rollout_clears_pointer() -> None:
    sim = CaseBoundSimulator(
        dim=3, node_type_size=9, kinematic_types=(1, 3), scripted_types=(1,)
    )
    sim._t = 7
    sim.reset_rollout()
    assert sim._t is None
```

- [ ] **Step 2: Run it to make sure it fails** (`pytest tests/models/common/ -v` → import error).
- [ ] **Step 3: Create `models/common/simulator_base.py`** by MOVING the listed code from `models/mgn/simulator.py`. Preserve the statefulness-contract docstring (simulator.py:1-60) at the base module, adjusted to name both families; keep ADR-0043 citations. `MeshSimulator` becomes a subclass: its `__init__` calls `super().__init__(...)` then builds `MGNet` + the 4 normalizers; `_on_bind_case(cells)` builds `self._mesh_edge_index`. `predict_positions`/`forward_train`/`_graph_features` stay in `MeshSimulator`, now calling the inherited helpers instead of inline logic. `models/common/__init__.py` exports `CaseBoundSimulator`.
- [ ] **Step 4: Widen the gate** — `cli/train.py:1173`: `isinstance(simulator, MeshSimulator)` → `isinstance(simulator, CaseBoundSimulator)` (import from `structbench.models.common`).
- [ ] **Step 5: Run the FULL suite** (`pytest` → all pass, zero mgn-test edits beyond imports of private names, if any). Run `ruff format --check . && ruff check . && mypy src`.
- [ ] **Step 6: Commit** — `refactor(models): extract CaseBoundSimulator state base from MeshSimulator (move-only)`.

### Task 2: `TransolverConfig` + registry

**Files:**
- Modify: `src/structbench/config.py` (dataclass near MGNConfig; `MODEL_FAMILIES` line 226)
- Test: additions to `tests/cli/test_train_config.py`

**Interfaces:**
- Produces: `TransolverConfig` consumed by Tasks 4-7:

```python
@dataclass
class TransolverConfig:
    """Native Transolver family (ADR-0041 step ②; recipe pins in ADR-0044).

    ``weight_decay``/``max_grad_norm`` are family-recipe knobs and live here
    rather than on TrainConfig (precedent: ``MGNConfig.noise_std``), keeping
    the strict ``[train]`` schema family-uniform.
    """

    input_frames: int = 2
    dim: int = 3
    hidden_dim: int = 128
    n_layers: int = 8
    n_heads: int = 8
    slice_num: int = 64
    mlp_ratio: int = 1
    dropout: float = 0.0
    node_type_size: int = 9
    noise_std: float = 0.003
    normalizer_warmup_steps: int = 1000
    weight_decay: float = 1e-5
    max_grad_norm: float = 0.1
```

- `MODEL_FAMILIES = {"cgn": CGNConfig, "gns": CGNConfig, "mgn": MGNConfig, "transolver": TransolverConfig}`.

**Steps:**

- [ ] **Step 1: Write failing tests** in `tests/cli/test_train_config.py`, mirroring the existing mgn config tests: (a) a valid transolver TOML loads with `family == "transolver"` and round-trips every field; (b) an unknown `[model]` key raises `ConfigError`; (c) a missing `[model]` key raises; (d) `input_frames != 2` on deforming_plate raises the ADR-0035 message.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** (dataclass + registry entry + docstring note on the `"transolver"` family in the `MODEL_FAMILIES` comment).
- [ ] **Step 4: Run tests + lint gates. Commit** — `feat(config): TransolverConfig + transolver family registration`.

### Task 3: Physics-Attention network (`models/transolver/network.py`)

**Files:**
- Create: `src/structbench/models/transolver/__init__.py`, `src/structbench/models/transolver/network.py`
- Test: `tests/models/transolver/test_network.py`

**Interfaces:**
- Produces (consumed by Task 4):
  - `build_mlp_2layer(in_size, hidden, out_size) -> nn.Sequential` — `Linear(in, hidden) → GELU → Linear(hidden, out)` (thuml MLP n_layers=0 shape).
  - `PhysicsAttentionIrregularMesh(dim, heads, dim_head, slice_num, dropout=0.0)` — forward `(x: (P, dim), n_particles_per_example: Tensor | None) -> (P, dim)`.
  - `TransolverNet(node_in, out_size, hidden_dim=128, n_layers=8, n_heads=8, slice_num=64, mlp_ratio=1, dropout=0.0)` — forward `(node_feats: (P, node_in), n_particles_per_example: Tensor | None) -> (P, out_size)`.
- Math per grounding §2-§3 (Eqs 1-4, pre-LN Eq 6, `fx_mid`-vs-`x_mid` subtlety, eps 1e-5, temperature, placeholder).

**Steps:**

- [ ] **Step 1: Write failing tests** — `tests/models/transolver/test_network.py`:

```python
"""TransolverNet / Physics-Attention (Wu et al. 2024; ADR-0044).

Reference math: Eqs (1)-(4) and pre-LN block Eq (6) of arXiv:2402.02366;
implementation details follow thuml/Transolver's irregular-mesh variant.
"""

import torch

from structbench.models.transolver.network import (
    PhysicsAttentionIrregularMesh,
    TransolverNet,
)


def test_forward_shape_single_example() -> None:
    net = TransolverNet(
        node_in=7, out_size=4, hidden_dim=16, n_layers=2, n_heads=2, slice_num=4
    )
    out = net(torch.randn(11, 7), None)
    assert out.shape == (11, 4)


def test_batched_forward_matches_per_example() -> None:
    # THE ragged-batching correctness test: segment computation must equal
    # running each example alone (thuml batch=1 semantics). eval() + dropout=0
    # make both paths deterministic; atol loosened above the float32 GEMM
    # tiling noise floor (different matrix sizes accumulate differently).
    torch.manual_seed(0)
    net = TransolverNet(
        node_in=7, out_size=4, hidden_dim=16, n_layers=2, n_heads=2, slice_num=4
    )
    net.eval()
    a, b = torch.randn(11, 7), torch.randn(5, 7)
    with torch.no_grad():
        batched = net(torch.cat([a, b]), torch.tensor([11, 5]))
        singles = torch.cat([net(a, None), net(b, None)])
    assert torch.allclose(batched, singles, atol=1e-5)


def test_slice_weights_softmax_over_slices() -> None:
    torch.manual_seed(0)
    attn = PhysicsAttentionIrregularMesh(dim=16, heads=2, dim_head=8, slice_num=4)
    w = attn._slice_weights(torch.randn(11, 16))  # (P, H, M)
    assert w.shape == (11, 2, 4)
    assert torch.allclose(w.sum(dim=-1), torch.ones(11, 2), atol=1e-6)
    # Non-degeneracy at init: weights must VARY across nodes (a constant
    # projection would silently give average pooling — the Transolver++
    # collapse risk, ADR-0044 ledgered; deeper non-collapse is NOT tested).
    assert w.std(dim=0).max() > 1e-4


def test_temperature_learnable_init() -> None:
    attn = PhysicsAttentionIrregularMesh(dim=16, heads=2, dim_head=8, slice_num=4)
    assert attn.temperature.requires_grad
    assert attn.temperature.shape == (2, 1)  # per-head, broadcasts vs (P, H, M)
    assert torch.allclose(attn.temperature, torch.full_like(attn.temperature, 0.5))


def test_reference_parameter_count() -> None:
    # Pins the reference architecture (L=8, C=128, H=8, M=64, ratio=1,
    # node_in=18, out=4) against the structural formula from grounding §2-§3.
    net = TransolverNet(node_in=18, out_size=4)
    n = sum(p.numel() for p in net.parameters())
    c, h, dh, m, node_in, out = 128, 8, 16, 64, 18, 4
    preprocess = (node_in + 1) * 2 * c + (2 * c + 1) * c
    attn = 2 * ((c + 1) * c) + (dh + 1) * m + 3 * dh * dh + (c + 1) * c + h
    block = 2 * 2 * c + attn + 2 * ((c + 1) * c)  # ln_1+ln_2, attn, mlp(ratio=1)
    last_extra = 2 * c + (c + 1) * out  # ln_3 + mlp2
    assert n == preprocess + c + 8 * block + last_extra  # +c = placeholder


def test_trunc_normal_init_applied() -> None:
    # Faithful to released code: the global trunc_normal_(std=0.02) + zero-bias
    # pass runs LAST, overwriting the orthogonal in_project_slice init (thuml
    # initialize_weights() ordering quirk, ADR-0044). Assertions must be
    # DISCRIMINATING — PyTorch's default Linear init has nonzero uniform bias
    # and weight std ~0.14 at these widths, so each check below fails if the
    # init pass is omitted.
    net = TransolverNet(node_in=18, out_size=4)
    lin = net.preprocess[0]
    assert isinstance(lin, torch.nn.Linear)
    assert torch.all(lin.bias == 0.0)
    assert abs(float(lin.weight.std()) - 0.02) < 0.006
    # The overwrite happened: slice projection is trunc_normal, NOT orthonormal
    # (orthogonal_ on the (slice_num=64, dim_head=16) weight gives Wᵀ W = I₁₆).
    w = net.blocks[0].attn.in_project_slice.weight
    assert abs(float(w.std()) - 0.02) < 0.006
    assert not torch.allclose(w.T @ w, torch.eye(w.shape[1]), atol=0.1)
    for mod in net.modules():  # secondary: LayerNorm resets
        if isinstance(mod, torch.nn.LayerNorm):
            assert torch.all(mod.weight == 1.0) and torch.all(mod.bias == 0.0)
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement `network.py`.** Core of the attention forward (per-example loop; `_slice_weights` factored out for testability):

```python
def _segments(total: int, n_per: Tensor | None) -> list[tuple[int, int]]:
    if n_per is None:
        return [(0, total)]
    ends = torch.cumsum(n_per, dim=0).tolist()
    starts = [0, *ends[:-1]]
    return list(zip(starts, ends, strict=True))

# inside PhysicsAttentionIrregularMesh:
def _slice_weights(self, x: Tensor) -> Tensor:
    # Eq (1): pointwise — safe on the flat (P, C) tensor.
    x_mid = self.in_project_x(x).reshape(-1, self.heads, self.dim_head)
    logits = self.in_project_slice(x_mid) / self.temperature  # (P, H, M)
    return torch.softmax(logits, dim=-1)

def forward(self, x: Tensor, n_particles_per_example: Tensor | None) -> Tensor:
    fx_mid = self.in_project_fx(x).reshape(-1, self.heads, self.dim_head)
    w = self._slice_weights(x)                       # (P, H, M)
    outs = []
    for start, end in _segments(x.shape[0], n_particles_per_example):
        w_e, fx_e = w[start:end], fx_mid[start:end]
        norm = w_e.sum(dim=0)                        # (H, M)
        token = torch.einsum("nhd,nhm->hmd", fx_e, w_e)
        token = token / (norm + 1e-5).unsqueeze(-1)  # Eq (2), eps per thuml
        q, k, v = self.to_q(token), self.to_k(token), self.to_v(token)
        dots = q @ k.transpose(-1, -2) * self.scale  # (H, M, M), Eq (3)
        token_out = self.attn_dropout(torch.softmax(dots, dim=-1)) @ v
        out_e = torch.einsum("hmd,nhm->nhd", token_out, w_e)  # Eq (4)
        outs.append(out_e.reshape(end - start, self.heads * self.dim_head))
    return self.to_out(torch.cat(outs, dim=0))
```

  `__init__` per grounding §3.2: `in_project_x`/`in_project_fx = Linear(dim, heads*dim_head)`; `in_project_slice = Linear(dim_head, slice_num)` with `torch.nn.init.orthogonal_(weight)`; `to_q/to_k/to_v = Linear(dim_head, dim_head, bias=False)`; `to_out = Sequential(Linear(inner, dim), Dropout(dropout))`; `temperature = nn.Parameter(torch.full((heads, 1), 0.5))` (broadcasts against `(P, H, M)`); `scale = dim_head ** -0.5`; `attn_dropout = nn.Dropout(dropout)`.

  `TransolverBlock(hidden_dim, heads, slice_num, mlp_ratio, dropout, last_layer, out_size)`: `ln_1`, `attn`, `ln_2`, `mlp = build_mlp_2layer(hidden, hidden * mlp_ratio, hidden)`; if `last_layer`: `ln_3 = LayerNorm(hidden)`, `mlp2 = Linear(hidden, out_size)`. Forward: `fx = attn(ln_1(fx), n_per) + fx; fx = mlp(ln_2(fx)) + fx; return mlp2(ln_3(fx)) if last_layer else fx` (no residual on the head — grounding §3.3).

  `TransolverNet`: `preprocess = build_mlp_2layer(node_in, hidden_dim * 2, hidden_dim)`; `placeholder = nn.Parameter((1 / hidden_dim) * torch.rand(hidden_dim))`; blocks list with `last_layer = (i == n_layers - 1)`; forward: `fx = preprocess(node_feats) + self.placeholder` (unconditional — irregular-mesh behaviour, grounding §3.3), then the block loop. `_initialize_weights()` at the END of `__init__`: `self.apply(...)` — `Linear → trunc_normal_(weight, std=0.02)` + zero bias; `LayerNorm → weight 1, bias 0`. This overwrites the orthogonal slice-projection init; keep the `orthogonal_` call anyway and document the quirk (faithful-to-released-code, ADR-0044).

  `__init__.py`: `__all__ = ["PhysicsAttentionIrregularMesh", "TransolverNet"]` — ONLY the names Task 3 defines (listing `TransolverSimulator` now would be ruff F822); Task 4 adds its import + `__all__` entry.
- [ ] **Step 4: Run tests + lint gates. Commit** — `feat(models): native Transolver network — Physics-Attention with segment-exact ragged batching`.

### Task 4: `TransolverSimulator` (`models/transolver/simulator.py`)

**Files:**
- Create: `src/structbench/models/transolver/simulator.py`; extend `__init__.py`
- Test: `tests/models/transolver/test_simulator.py`

**Interfaces:**
- Consumes: `CaseBoundSimulator` (Task 1), `TransolverNet` (Task 3), `OnlineNormalizer` (`models/mgn/normalizers.py` — import as-is), the OnlineNormalizer call convention used in `models/mgn/simulator.py` (accumulate flag + `.inverse`).
- Produces:

```python
class TransolverSimulator(CaseBoundSimulator):
    def __init__(
        self,
        dim: int = 3,
        hidden_dim: int = 128,
        n_layers: int = 8,
        n_heads: int = 8,
        slice_num: int = 64,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
        node_type_size: int = 9,
        kinematic_types: tuple[int, ...] = (1, 3),
        scripted_types: tuple[int, ...] = (1,),
        device: str | torch.device = "cpu",
    ) -> None: ...
    # node_in = node_type_size + 3 * dim  (one-hot, scripted vel, x_t, ref coords)
    # net = TransolverNet(node_in, out_size=dim + 1, ...)
    # _node_normalizer = OnlineNormalizer(node_in); _target_normalizer = OnlineNormalizer(dim + 1)

    def predict_positions(
        self, current_positions, nparticles_per_example, particle_types
    ) -> tuple[Tensor, Tensor]: ...   # (P, dim), (P, 1) — _SimulatorLike protocol

    def forward_train(
        self,
        x_last: Tensor,               # (P, dim) — ALREADY noised by the caller
        next_positions: Tensor,       # (P, dim)
        next_aux: Tensor,             # (P,)
        particle_types: Tensor,       # (P,)
        reference_coords: Tensor,     # (P, dim) from collate_mesh_samples
        n_particles_per_example: Tensor,  # (B,)
        *,
        accumulate: bool,
    ) -> tuple[Tensor, Tensor]: ...   # (pred_norm, target_norm), both (P, dim+1)
```

**Steps:**

- [ ] **Step 1: Write failing tests** — mirror the structure of `tests/models/mgn/test_mgn_simulator.py` (read it first; note the `test_mgn_*.py` naming), covering: (a) `predict_positions` output shapes on a bound synthetic case; (b) tripwire: perturbed kinematic rows raise `RuntimeError` mentioning `reset_rollout`; (c) calling a second eval pass without `reset_rollout` raises; (d) scripted-velocity feature: assert on `_features` directly — the scripted rows of the scripted-velocity slice equal the GT deltas, zero elsewhere. Do NOT zero all net weights to get a no-op forward: the learnable temperature divides slice logits, so zeroing it produces NaN; if a zero-output forward is wanted, zero only the last block's `mlp2` and leave everything else at init; (e) `forward_train` target = `cat([next_positions - x_last, next_aux[:, None]], 1)` normalized (γ=1: measured from the given noisy `x_last`); (f) save/load round-trip restores params AND normalizer buffers, and works before any `bind_case` (self-contained checkpoint, no stats file); (g) constructor rejects `scripted_types ⊄ kinematic_types` (inherited).
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.** `_features(one_hot, scripted_velocity, x_t, reference_coords) = cat([one_hot, scripted_velocity, x_t, reference_coords], -1)` — `one_hot` is an EXPLICIT parameter (mirrors MGN's `_graph_features` signature): `predict_positions` supplies the bound `self._node_type_onehot`; `forward_train` builds it from its `particle_types` argument (bind_case is never called on the training path — batched collated data, no bound case). `predict_positions`: base tripwire → `_eval_scripted_velocity` → features (bound `reference_coords`) → `_node_normalizer(feats, accumulate=False)` → `net(feats, None)` → `_target_normalizer.inverse` on the FULL `(P, dim+1)` **before** slicing (denorm-before-slice, grounding c32) → `velocity, stress = out[:, :dim], out[:, dim:]` → `next = x_t + velocity`. `forward_train`: `_train_scripted_velocity` (base helper) → features (batch `reference_coords`) → normalizer with `accumulate` → `net(feats, n_particles_per_example)`; target normalized with `accumulate`. Device handling mirrors MGN (`.to(device)` in the builder, tensors moved by callers).
- [ ] **Step 4: Run tests + lint gates. Commit** — `feat(models): TransolverSimulator — stateful rollout wrapper on the shared base`.

### Task 5: Training path — dispatch, `_train_transolver`, cosine schedule

**Files:**
- Modify: `src/structbench/cli/train.py`
- Test: extend `tests/cli/test_train_eval.py` (unit-level); the end-to-end gate is Task 7

**Interfaces:**
- Consumes: `_train_mgn` (731-953) as the structural template; `mesh_static_from_trajectory` + `collate_mesh_samples` (reused verbatim); `resolved_config_dict("transolver", ...)`; `LR_SCHEDULE_FLOOR`.
- Produces:
  - `_lr_at_cosine(step: int, train_cfg: TrainConfig) -> float`:

```python
def _lr_at_cosine(step: int, train_cfg: TrainConfig) -> float:
    """Cosine anneal lr_init → LR_SCHEDULE_FLOOR over training_steps (ADR-0044).

    Steps-port of the Transolver reference's per-epoch CosineAnnealingLR
    (grounding §4.1); the exponential `_lr_at` stays the CGN/MGN schedule.
    """
    frac = min(step / max(1, train_cfg.training_steps), 1.0)
    span = train_cfg.lr_init - LR_SCHEDULE_FLOOR
    return LR_SCHEDULE_FLOOR + span * 0.5 * (1.0 + math.cos(math.pi * frac))
```

  - `build_transolver_simulator(cfg: TransolverConfig, *, kinematic_types, device) -> TransolverSimulator` (mirror of `build_mgn_simulator`, 217-255; `scripted_types` left at class default `(1,)`).
  - `_train_transolver(spec, cfg, train_cfg, train_trajs, val_trajs, out_dir, device, data_root) -> Path | None` — clone `_train_mgn`'s structure with these deltas ONLY: `sim = build_transolver_simulator(...)`; `resolved_config_dict("transolver", cfg, ...)`; optimizer `torch.optim.AdamW(sim.parameters(), lr=train_cfg.lr_init, weight_decay=cfg.weight_decay)`; per-step lr from `_lr_at_cosine`; after `loss.backward()`: `if cfg.max_grad_norm > 0: torch.nn.utils.clip_grad_norm_(sim.parameters(), cfg.max_grad_norm)`; `forward_train` call passes `(x_noisy, next_position, next_aux, particle_type, reference_coords, n_particles_per_example, accumulate=...)` (no `mesh_edge_index`); **drop the `mesh_edge_index = batch["mesh_edge_index"].to(device)` unpack** (line 866 in the template — unused here would be ruff F841); keep unpacking `position_seq`, `particle_type`, `next_position`, `next_aux`, `reference_coords`, `n_particles_per_example`. Noise block (870-874), loss block (887-898), warmup accumulate, val-rollout loop (909-945), best-ckpt selection, periodic `ckpt-<step>.pt` — copied unchanged.
  - `train()` dispatch (529-540): restructure the binary `if family == "mgn": ... else: <cgn>` into `if family == "mgn": return _train_mgn(...)` / `if family == "transolver": return _train_transolver(...)` / fall-through CGN path.
  - **Type widenings (mypy-strict gate fails without them):** `train()`'s `model_cfg: CGNConfig | MGNConfig` (train.py:391) gains `| TransolverConfig`, with `assert isinstance(model_cfg, TransolverConfig)` narrowing in the new dispatch arm (mirror of the mgn assert at :530); import `TransolverSimulator`/`TransolverConfig` at the top of train.py.

**Steps:**

- [ ] **Step 1: Write failing unit tests**: (a) `_lr_at_cosine(0) == lr_init` (within float eps), `_lr_at_cosine(training_steps) == LR_SCHEDULE_FLOOR + ~0`, monotone decreasing on samples; (b) `build_transolver_simulator` returns a `TransolverSimulator` with `kinematic_types` from the spec and net sized per config; (c) **training-wiring test** (guards the hand-cloned loop — a paste that forgets the AdamW/cosine/clip swaps passes everything else): run `_train_transolver` for a handful of steps at smoke sizes on the tiny synthetic fixture, with `monkeypatch` capturing (i) `torch.optim.AdamW` construction kwargs — assert `weight_decay == cfg.weight_decay` (this also proves AdamW, not Adam), and (ii) `torch.nn.utils.clip_grad_norm_` calls — assert called with `cfg.max_grad_norm`; after the run assert `optimizer.param_groups[0]["lr"]` equals `_lr_at_cosine(last_step, train_cfg)` (proves the cosine schedule is wired and `[train].lr_decay` unused), and assert at least one model parameter changed from its pre-run value (gradient flow — the smoke never checks this).
- [ ] **Step 2: Run to verify failure. Implement. Run.**
- [ ] **Step 3: Lint gates. Commit** — `feat(cli): transolver training path — AdamW + cosine anneal + grad clip, shared mesh harness`.

### Task 6: `evaluate()` transolver arm

**Files:**
- Modify: `src/structbench/cli/train.py` (`evaluate()`, 1130-1153)
- Test: extend `tests/cli/test_train_eval.py`

**Interfaces:**
- Consumes: `_model_config_from_record` (generic via `MODEL_FAMILIES`), `build_transolver_simulator` (Task 5), the widened `CaseBoundSimulator` gate (Task 1).
- Produces: `evaluate()` handles `family == "transolver"`: build simulator from record + checkpoint, **no** `normalization_stats.npz` expected (self-contained checkpoint, same as mgn — grounding c23); per-case `bind_case` + the 3 `reset_rollout` calls flow through the existing gate unchanged.
- **Type widenings (mypy-strict gate fails without them):** the local at train.py:1135 becomes `simulator: LearnedSimulator | MeshSimulator | TransolverSimulator` — the CONCRETE union, NOT `CaseBoundSimulator`: `rollout(simulator, ...)` (train.py:1202) needs the `_SimulatorLike` protocol's `predict_positions`, which the base class deliberately lacks (Design decision 10 keeps it per-family). `_model_config_from_record`'s return annotation (train.py:1021) gains `| TransolverConfig`; the new arm narrows with `assert isinstance(model_cfg, TransolverConfig)` (mirror of the mgn assert at :1137).

**Steps:**

- [ ] **Step 1: Write failing tests** mirroring the mgn cases in `tests/cli/test_train_eval.py`: (a) a transolver `config.json` record round-trips through `_model_config_from_record` into a `TransolverConfig`; (b) evaluate on a saved tiny transolver checkpoint does NOT require a stats file (assert it is absent yet evaluation proceeds — reuse the mgn test's fixture pattern).
- [ ] **Step 2: Run to verify failure. Implement** — extend the family branch: `if family == "mgn": ... elif family == "transolver": ... else: <stats path>`.
- [ ] **Step 3: Run tests + lint gates. Commit** — `feat(cli): evaluate() transolver arm — self-contained checkpoints, shared bind/reset flow`.

### Task 7: Configs + end-to-end smoke

**Files:**
- Create: `configs/deforming_plate/transolver.toml`, `configs/deforming_plate/transolver_smoke.toml`
- Test: `tests/cli/test_transolver_train_smoke.py`

**Interfaces:** consumes everything above; produces the runnable reference config.

**Steps:**

- [ ] **Step 1: Write the reference config** `configs/deforming_plate/transolver.toml`:

```toml
# DeformingPlate — Transolver PROVISIONAL reference config (ADR-0041 ②, ADR-0044).
# No published vanilla-Transolver rollout number exists for this task; the
# recipe is declared, not reproduced. Budget matched to the MGN reference
# (batch 2, 10M steps) for a same-task/same-data/same-budget comparison.
# Not yet run.

[run]
benchmark = "deforming_plate"
seed = 1

[model]
family = "transolver"
input_frames = 2
dim = 3
hidden_dim = 128
n_layers = 8
n_heads = 8
slice_num = 64
mlp_ratio = 1
dropout = 0.0
node_type_size = 9
noise_std = 0.003
normalizer_warmup_steps = 1000
weight_decay = 1e-5      # AdamW, thuml reference (grounding §4.1)
max_grad_norm = 0.1      # global-norm clip, thuml reference

[train]
batch_size = 2
lr_init = 1e-3
lr_decay = 0.1           # UNUSED by the transolver family (cosine anneal,
                         # ADR-0044); present because [train] is family-uniform.
training_steps = 10_000_000
val_every = 50_000
w_pos = 1.0
w_aux = 1.0
aux_tail_weight = 0.0
train_frames = 0
```

  and `transolver_smoke.toml` (`# NOT a baseline` header; `hidden_dim = 16, n_layers = 2, n_heads = 2, slice_num = 8, normalizer_warmup_steps = 5, training_steps = 50, val_every = 25, seed = 0`, rest as reference).
- [ ] **Step 2: Write the end-to-end smoke test** `tests/cli/test_transolver_train_smoke.py` — copy the fixture + flow of `tests/cli/test_mgn_train_smoke.py` (tiny synthetic mesh cases, real `train()` with `family="transolver"` at smoke sizes, then real `evaluate()`), asserting: checkpoints exist (`model-*.pt`, periodic `ckpt-*.pt` if steps cross the interval), `config.json` records `family = "transolver"`, `metrics-val.json` written with finite numbers, no `normalization_stats.npz`. Keep runtime in the same budget as the mgn smoke (~≤60 s CPU).
- [ ] **Step 3: Run it (it should pass against the already-implemented stack; if it fails, the failure is a real integration bug — fix in place).** Also load-check both TOMLs via `load_run_config` in a small test (reference config loads; smoke config loads).
- [ ] **Step 4: Full suite + lint gates. Commit** — `feat(configs): deforming_plate transolver reference + smoke configs; end-to-end smoke test`.

### Task 8: ADR-0044 draft + docs

**Files:**
- Create: `decisions/0044-transolver-provisional-adaptation.md` (status **Proposed** — the maintainer finalises)
- Modify: `decisions/README.md` (index row), `docs/ARCHITECTURE.md` (models/ section: `cgn/`, `mgn/`, `transolver/`, `common/`)

**Steps:**

- [ ] **Step 1: Draft ADR-0044** from this plan's Design-decisions table, structured per `decisions/README.md` conventions (Context / Decision / Alternatives / Consequences). It must record: the 18-channel featurization and its information-parity argument; `(P, dim+1)` velocity+stress output with Euler; NORMAL-masked noise 3e-3 γ=1; segment-exact ragged batching (≡ thuml batch=1); OnlineNormalizer in place of thuml's precomputed `UnitTransformer` stats (harness-consistent deviation); AdamW + wd 1e-5 + clip 0.1 + cosine-to-floor as `TransolverConfig` knobs and WHY they are not TrainConfig fields; the matched 10M/batch-2 budget declaration; faithful-to-released-code fidelity target (trunc_normal overwrite quirk, unclamped temperature 0.5, unconditional placeholder, mlp_ratio 1, GELU, dropout 0, no Time_Input); pure-torch no-einops/timm; provisional = no numeric gate, §5 leaderboard comparison statistic, pooled tool informational; the `models/common` `CaseBoundSimulator` extraction; the two documented fidelity risks (slice-weight collapse — Transolver++ frames it as a defect of the original, grounding §7.3; contact learned from absolute coordinates without any proximity mechanism); MIT attribution convention (docstring-level; NOTICE file routed to maintainer).
- [ ] **Step 2: Update `decisions/README.md`** (0044 row, Proposed) and `docs/ARCHITECTURE.md`'s models/ layout + prose (also correcting the stale cgn-only wording — grounding c62).
- [ ] **Step 3: Lint/docs gates as applicable. Commit** — `docs: ADR-0044 draft (Transolver provisional adaptation) + ARCHITECTURE models/ refresh`.

---

## Self-review notes

- Every task's exact values trace to `scratch/2026-08-08-transolver-grounding.md` (§2-§6) — implementers should treat that file as read-only reference when a number seems surprising.
- Type consistency: `TransolverNet(node_in, out_size, ...)` ↔ Task 4's `node_in = node_type_size + 3 * dim`, `out_size = dim + 1`; `forward_train` arg order matches `_train_transolver`'s call in Task 5; `n_particles_per_example: Tensor | None` everywhere.
- The Task 3 param-count formula assumes bias=True on all Linears except `to_q/to_k/to_v` (bias=False) — matching grounding §3.2 exactly.
- Deliberately NOT in scope: `BaselineResult.provisional` + comparison table (separate plan); any training run (maintainer compute); GeoFLARE (step ③); 3D viz (deferred per ADR-0041).
