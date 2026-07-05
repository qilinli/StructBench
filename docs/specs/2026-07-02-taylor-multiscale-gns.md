# Design spec — Multi-scale GNS (MS-GNS) second baseline for Taylor 2D

> **ADR-0032 note (2026-07-05).** The flat-config mechanisms this plan references (`from_toml`, `_write_resolved_config`, a flat `configs/taylor_2d_msgns.toml`) were replaced by the grouped run config: register `MSGNSConfig` in `structbench.config.MODEL_FAMILIES` and select it via `[model] family = "msgns"`; the trainer records the nested `config.json` via `resolved_config_dict`.
*Status: **Proposed** — drafted by Claude Code from a four-reader analysis of the
reference implementation; awaiting human approval before implementation.
Implements ADR-0015 §2 (prior-paper GNNs ship as baselines); constrained by
ADR-0018 (torch/PyG only in the ML layer), ADR-0019 (benchmark protocol is
model-agnostic and unchanged), ADR-0020 (native radius_graph, no compiled
graph deps).*

---

## Goal

Port the user's prior-paper **multi-scale GNS** (`../code/sgnn/sgnn/multi_scale/`,
read-only reference) into `src/structbench/models/msgns/` as the **second
reference baseline** for the Taylor 2D benchmark. The model slots behind the
same simulator protocol the eval layer already defines, is trained/evaluated by
the same `structbench-train` CLI on the same canonical data, and is selected by
a `model` key in the TOML config. No benchmark, schema, or dependency changes.

Non-goals (this slice): running the full training (parked with the DUG run);
resume support (parked; the reference's resume is broken anyway); other
datasets; generalising beyond one auxiliary channel; transferring reference
checkpoint weights (impossible — see deviation 1).

## Reference architecture (what is ported)

GraphCast-style **encode–process–decode over a static two-level hierarchy**:

- **Coarse scales by strided subsampling.** Every `stride`-th unique x and y
  coordinate of the regular SPH grid is kept, so coarse nodes are a strict
  subset of fine particles addressed by global indices; level spacing grows by
  `stride` per scale. Pure torch; assumes a tensor-product regular grid (true
  for the Taylor SPH layouts).
- **Edges.** One radius graph over the full fine grid
  (`r = radius_multiplier × grid_spacing`, self-loops, ≤ `max_neighbors`)
  filtered by mesh membership into **grid→mesh** (target is a mesh node) and
  **mesh→grid** (source is a mesh node); plus one radius graph per mesh scale
  (`r = spacing_s × radius_multiplier`) whose edges are concatenated into a
  single **mesh→mesh** set. The fine scale has no intra-scale edges — fine
  interaction flows g2m → m2m → m2g.
- **Network.** All latents live in one per-grid-node tensor. Node/edge encoders
  (node features: normalized velocity window + boundary feature + type
  embedding; edge features: relative displacement / r + norm, per edge type),
  one G2M block, `message_passing_steps` M2M blocks, one M2G block, prediction
  head → normalized acceleration + one auxiliary channel. All blocks are
  `torch_geometric.nn.MessagePassing(aggr="add")` — allowed under ADR-0020.
- **Simulator wrapper.** Same contract as the single-scale port: velocity
  normalization with noise-inflated std, random-walk input noise during
  training, Euler integration, auxiliary head.

Reference defaults (from `multi_scale_config.yaml`): hidden 128, 10 m2m steps,
window 11, `num_scales` 2, stride 2 (named `window_size` there),
`radius_multiplier` 2.0, noise 0.02, lr 1e-3 with 0.1/15k decay.

## Deliberate deviations from the reference

1. **Native `radius_graph` (ADR-0020).** The reference's only compiled-op use
   is PyG's torch-cluster-backed `radius_graph` at two call sites. The port
   reuses `models/gns/graph_ops.radius_graph`. Note its edge orientation is the
   transpose of PyG's (row 0 = receiver) — membership filters adapt — and its
   neighbor truncation keeps nearest rather than arbitrary. Consequence:
   reference weights are not loadable; the baseline is retrained (planned
   anyway).
2. **Batch-correct hierarchy.** The reference attaches only the *first*
   example's static graph to a batch and applies global-index edges with no
   per-example offset — in any batch > 1, only example 0 exchanges messages (a
   real training defect, confirmed at `multi_scale_collate_fn` /
   `MultiScaleGNN.forward`). The port builds the hierarchy **per example inside
   the simulator** from the earliest frame of the input window, offsetting
   indices per example — batching is correct and `WindowDataset`/
   `collate_samples` are reused unchanged. Consequence: topology is
   *per-window quasi-static* (refreshed as the window slides) rather than
   frozen at t = 0. The frozen-at-t0 variant cannot be expressed behind the
   case-blind simulator protocol without leaking trajectory identity, and the
   per-window form handles the large mushrooming deformation at least as
   sensibly. A regression test encodes the corrected batching behavior.
3. **Working checkpoint selection.** The reference gates "best" on dict keys
   its validator never returns, so only the first validation ever saves. The
   port uses the package's existing `_validate` (mean rollout RMSE over VAL,
   physical units), identical to the single-scale baseline.
4. **Auxiliary channel named and shaped correctly.** von Mises stress (MPa),
   returned `(P, 1)`; de-normalized in `predict_positions`, normalized in
   `predict_accelerations` — the package contract. (The reference returns 1-D
   and calls it "strain"; ADR-0019 records the mislabel.)
5. **No hard-coded geometry.** `grid_spacing` (0.5 mm) and `max_neighbors` (24)
   become `MSGNSConfig` fields with the reference values as defaults. The
   regular-grid assumption is documented on the hierarchy builder.
6. **No resume**, matching the package (and sidestepping the reference's
   triply-broken implementation). Resume is a separate, parked item.

## Module layout

```
src/structbench/models/msgns/
  __init__.py         # re-exports MultiScaleSimulator
  hierarchy.py        # per-example subsample + g2m/m2g/m2m edge build (pure torch)
  network.py          # encoders, G2M/M2M/M2G blocks, prediction head
  simulator.py        # MultiScaleSimulator: protocol impl, save/load
tests/models/msgns/   # mirrors, CPU-only synthetic fixtures
```

## CLI & config integration

- Top-level **`model = "gns" | "msgns"`** TOML key, default `"gns"` (existing
  configs stay valid — unknown keys are already ignored by the dataclass
  loaders). New `MSGNSConfig` dataclass with `from_toml`, same cherry-pick
  pattern.
- A name→builder registry in `cli/train.py`; both builders bind the Taylor
  `wall_distance_feature` in `cli/` (models must not import `benchmarks/`).
- `config.json` gains `"model": <name>` plus the model's config block;
  `evaluate()` dispatches on `config.get("model", "gns")`, keeping old run
  dirs valid.
- **Interface discipline fixes (small, included):**
  `models/gns/__init__.py` re-exports `radius_graph` so `msgns` imports a
  public surface; `eval/` exports the simulator protocol (`SimulatorLike`) and
  `cli/train.py`'s `LearnedSimulator`-typed hints are retyped to it.
- New configs: `configs/taylor_2d_msgns.toml` (full) and a CPU smoke variant.

## Testing

Mirrors the single-scale slice (deterministic, CPU-only, tiny synthetic grids):
hierarchy subsampling and edge filters on a 4×4 grid with hand-checked
neighbor sets; global-index mapping; **the batching regression test** (a
two-example batch has no cross-example edges and both examples' nodes receive
messages); network forward shape/finiteness; simulator protocol compliance
(`predict_positions`/`predict_accelerations` shapes, aux de-normalization);
CLI model dispatch + `config.json` round-trip; end-to-end smoke via the
existing synthetic-case fixtures.
