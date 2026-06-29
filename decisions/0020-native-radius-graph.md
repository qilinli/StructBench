# 0020 — Native radius_graph; no graph-backend binary dependency

**Status**: Accepted
**Type**: Durable
**Date**: 2026-06-29

## Context

ADR-0018 added `torch` and `torch_geometric` as runtime dependencies, on the
assumption that `torch_geometric` provides the neighbour-graph construction the
GNS needs. In practice `torch_geometric.nn.radius_graph` does not compute the
graph itself — it dispatches to a compiled backend (`pyg-lib` or
`torch-cluster`). Those backends ship as per-`torch`-version binary wheels, and
**no wheel exists for the project's installed torch (2.12) on Windows** (the
newest with a Windows `torch-cluster` wheel is torch 2.8). The options were to
pin torch back to ≤2.8 to obtain a wheel, build the backend from source (needs a
C++ toolchain; fragile per machine), or stop depending on the backend. The
maintainer wants to stay on current torch (2.12), and a fragile binary
dependency is a poor fit for an open platform meant to `pip install` across
platforms.

## Decision

1. **StructBench provides its own `radius_graph`** in
   `models/gns/graph_ops.py`, in pure `torch`. The GNS simulator uses it
   instead of `torch_geometric.nn.radius_graph`. No `torch-cluster` or
   `pyg-lib` dependency is added.

2. **Current implementation is a memory-bounded O(N²) neighbour search**
   (query nodes processed in chunks, so peak memory is O(chunk·N)). It is
   correct and adequate for the datasets in scope (Taylor: ≤ ~2.6×10⁴
   particles per case; comfortably to ~10⁵).

3. **The interface is the seam for scaling.** `radius_graph(pos, r, batch, *,
   max_num_neighbors, loop)` matches the convention the simulator consumes. A
   spatial-grid / cell-list backend (O(N), scales to ≥10⁶ nodes) is swapped in
   **behind this same signature** when a dataset actually needs it. It is not
   built now (YAGNI); the O(N²)-compute ceiling is documented at the function.

4. **`torch_geometric` remains a dependency** (ADR-0018 stands): it is still
   used for `MessagePassing` and related primitives. This ADR refines *how the
   neighbour graph is built*, not whether PyG is a dependency.

## Alternatives considered

- **Add `torch-cluster` / `pyg-lib`.** Rejected: no wheel for torch 2.12, so it
  forces either pinning torch back (below) or a from-source build (fragile,
  needs a compiler); and a binary graph backend undermines cross-platform
  `pip install` for an open platform.

- **Pin torch to ≤2.8 to get a backend wheel.** Rejected: the maintainer wants
  to track current torch; pinning the whole stack backward to obtain one
  compiled op is a poor trade.

- **Build the spatial-grid backend now.** Deferred, not rejected: it is the
  right implementation for ≥10⁶-node datasets, but none exist in scope yet.
  Building it speculatively is premature; the interface makes it a one-function
  swap later.

## Consequences

- StructBench installs and runs the GNS on plain `torch` (CPU or CUDA), no
  binary graph backend, on any platform — including the project's torch-2.12
  environment.

- The current `radius_graph` is O(N²) in compute; it is fine to ~10⁵ nodes and
  documented as such. Datasets at the 10⁶-node scale will need the deferred
  grid backend before training is practical — a known, signposted limit, not a
  silent cliff.

- Particle models beyond the GNS reuse the same `radius_graph` op, so the
  eventual grid upgrade benefits all of them at once.

- This ADR does not supersede ADR-0018; it records a consequence of it.
