# Multi-scale GNS (MS-GNS) Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** Proposed — implements `docs/specs/2026-07-02-taylor-multiscale-gns.md`; do not start until the spec is approved.

**Goal:** Port the prior-paper multi-scale GNS as the second Taylor 2D baseline, selectable via `model = "msgns"` in the training TOML, evaluated by the unchanged ADR-0019 protocol.

**Reference source** (read-only, outside the repo): `../code/sgnn/sgnn/multi_scale/{multi_scale_graph.py, multi_scale_gnn.py, multi_scale_simulator.py, multi_scale_train.py, multi_scale_config.yaml, static_graph_data_loader.py}`.

## Global Constraints

Same as the single-scale plan (`docs/plans/2026-06-29-taylor-gns-surrogate.md` Global Constraints): py 3.11+, ruff 88 cols, mypy on public APIs, NumPy docstrings with units/shapes, CPU-only deterministic tests, stdlib logging, mm/MPa working frame, `.venv` tooling, Conventional Commits on the feature branch. Additionally:

- **No new dependencies.** Graph build reuses `models/gns/graph_ops.radius_graph` (ADR-0020); message passing uses `torch_geometric.nn.MessagePassing` only.
- **Port defects are fixed, not copied** — the spec's deviations 2 (batching) and 3 (checkpoint gating) are normative.
- **TDD per task**: failing test first, watch it fail, minimal code, whole suite green before commit.

## File Structure

```
src/structbench/models/gns/__init__.py       # + re-export radius_graph
src/structbench/models/msgns/__init__.py     # NEW: re-exports MultiScaleSimulator, MSGNSConfig home is cli
src/structbench/models/msgns/hierarchy.py    # NEW: subsample + edge build, per-example offsets
src/structbench/models/msgns/network.py      # NEW: encoders, G2M/M2M/M2G, head
src/structbench/models/msgns/simulator.py    # NEW: MultiScaleSimulator
src/structbench/eval/__init__.py             # + export SimulatorLike
src/structbench/eval/rollout.py              # _SimulatorLike -> public SimulatorLike (alias kept)
src/structbench/cli/train.py                 # MSGNSConfig, model key, builder registry, config.json "model"
configs/taylor_2d_msgns.toml                 # NEW: full config (reference defaults)
configs/taylor_2d_msgns_smoke.toml           # NEW: CPU smoke
tests/models/msgns/test_hierarchy.py         # NEW
tests/models/msgns/test_network.py           # NEW
tests/models/msgns/test_simulator.py         # NEW
tests/cli/test_model_dispatch.py             # NEW
```

## Tasks

### Task 1 — Public surfaces: `radius_graph` re-export + `SimulatorLike`
- [ ] Failing test: import `radius_graph` from `structbench.models.gns` and `SimulatorLike` from `structbench.eval`.
- [ ] Re-export `radius_graph` in `models/gns/__init__.py`; rename `_SimulatorLike` → `SimulatorLike` in `eval/rollout.py` (keep a private alias for compat), export from `eval/__init__.py`.
- [ ] Retype `cli/train.py` `_validate`/`evaluate` hints from `LearnedSimulator` to `SimulatorLike`. mypy green.

### Task 2 — `hierarchy.py`: per-example multi-scale hierarchy
- [ ] Failing tests on a hand-checkable 4×4 unit-spacing grid: stride-2 subsampling keeps the 2×2 corner subset; spacing doubles per scale; g2m edges all target mesh nodes; m2g edges all source mesh nodes; m2m radius/edge set matches a hand computation; self-loops present; `max_neighbors` respected.
- [ ] Failing **batching regression test**: two concatenated examples (offset via `n_particles_per_example`) produce no cross-example edges, and every example's nodes appear as receivers.
- [ ] Implement `build_hierarchy(positions, n_particles_per_example, cfg) -> Hierarchy` (edge sets as `(2, E)` receiver-first tensors, global indices, per-example offsets), using `models/gns` `radius_graph` with a per-example batch vector. Document the regular-grid assumption and the receiver-first orientation.

### Task 3 — `network.py`: MS-GNN blocks
- [ ] Failing tests: forward pass on a tiny hierarchy returns `(P, dim + n_aux)`; finite outputs; G2M/M2M/M2G blocks respect edge direction (a node with no incoming m2g edge keeps its encoder latent, etc.); parameter count sanity vs config.
- [ ] Port encoders + blocks + head from `multi_scale_gnn.py` (MessagePassing `aggr="add"`, residual + LayerNorm as in the reference), config-driven widths.

### Task 4 — `simulator.py`: `MultiScaleSimulator`
- [ ] Failing tests: satisfies `SimulatorLike` (shapes `(P, dim)` + `(P, 1)`); aux de-normalized in `predict_positions`, normalized in `predict_accelerations`; noise path matches the single-scale contract; `save`/`load` round-trip; batch of two windows trains without cross-example leakage (reuse Task 2's regression fixture through the full model).
- [ ] Port `multi_scale_simulator.py` onto the package contract: stats dict with noise-inflated velocity/acceleration std, injected `boundary_feature_fn`, Euler integration, hierarchy built per call from the window's earliest frame (spec deviation 2).

### Task 5 — CLI integration
- [ ] Failing tests: TOML `model = "msgns"` dispatches to the msgns builder; default remains `"gns"`; `config.json` records `"model"` + the msgns block; `evaluate()` on a run dir with `"model": "msgns"` rebuilds the right simulator; old config.json without a `model` key still evaluates as gns.
- [ ] Implement `MSGNSConfig` (window=11, num_scales=2, subsample_stride=2, radius_multiplier=2.0, grid_spacing=0.5, hidden_dim=128, message_passing_steps=10, particle_type_embedding_size=9, noise_std=0.02, dim=2, max_neighbors=24) + `from_toml`; builder registry; `_write_resolved_config` extension; `evaluate()` dispatch.
- [ ] Add `configs/taylor_2d_msgns.toml` + smoke variant.

### Task 6 — Whole-branch verification & docs
- [ ] Full suite + ruff + mypy green; CPU smoke-train (synthetic fixture) for both models.
- [ ] ARCHITECTURE.md `models/` section: mention the two baselines; data-flow note stays accurate.
- [ ] Whole-branch adversarial review (workflow) before the final commit; fix confirmed findings.
- [ ] Update `docs/specs/2026-07-02-taylor-multiscale-gns.md` status → Implemented.

## Self-review notes

- The per-window hierarchy rebuild (spec deviation 2) is the only semantic departure with modelling consequences; it is flagged in the spec's deviations for human review.
- Real-data validation and full training are out of scope; the DUG run for either baseline is a separate, parked step.
- `time_diff` is imported by `cli/train.py` from `models.gns.simulator` (not on the public surface) — pre-existing wrinkle, left alone here.
