"""Dynamic k-capped radius graph for the hybrid MP-Transolver branch (Design A).

The local message-passing branch of the hybrid Transolver (a middle-block
residual, see :class:`~structbench.models.transolver.network.LocalMessagePassing`)
runs over a *dynamic* radius graph rebuilt each step from the current
positions. This module builds that graph.

Why a bespoke builder rather than reusing MGN's ``world_edges`` or GeoFLARE's
``ball_query`` verbatim (the G3-P3 critique, §4b):

* :func:`structbench.models.mgn.mesh_ops.world_edges` returns an **uncapped**
  ``(2, E)`` index — every pair within ``radius``. Under contact/compaction
  ``E`` grows super-linearly and blew up MGN's world-edge activations at
  ~760 k steps (the ``mgn-worldedge-oom-fix`` episode). The hybrid branch
  needs a **bounded** ``E``.
* :func:`structbench.models.geoflare.geo_ops.ball_query` *is* k-capped, but it
  returns padded neighbour **coordinates** ``(P, k, dim)``, not an
  ``edge_index`` — its neighbour indices are internal and never surfaced.

So this module writes the missing piece: a **k-capped radius edge-index**
(top-``max_neighbors`` nearest within ``radius`` per node, so ``E <= P *
max_neighbors``) plus MGN-style relative-position edge features
``[x_ij, |x_ij|]``. Edge direction follows the project convention
(``edge_index[0]`` sender, ``edge_index[1]`` receiver; a message on edge
``(j, i)`` is aggregated into receiver ``i``): each node ``i`` receives from
its nearest neighbours ``j``. Graphs are built **per example** (partitioned by
``n_particles_per_example``) so no edge crosses an example boundary — the
same invariant that keeps MGN's ``index_add_`` aggregation segment-safe.

Graph construction lives with the model that uses it (ADR-0020 precedent,
``models/mgn/mesh_ops.py``, ``models/geoflare/geo_ops.py``).
"""

from __future__ import annotations

import torch
from torch import Tensor

_QUERY_CHUNK = 2048


def _segments(total: int, n_per: Tensor | None) -> list[tuple[int, int]]:
    """Per-example ``[start, end)`` ranges over a flat ``(total, ...)`` tensor.

    Mirrors :func:`structbench.models.transolver.network._segments` but is
    kept local so this module has no dependency on ``network`` (which depends
    on *it*).
    """
    if n_per is None:
        return [(0, total)]
    ends = torch.cumsum(n_per, dim=0).tolist()
    starts = [0, *ends[:-1]]
    return list(zip(starts, ends, strict=True))


def _capped_radius_edges_single(
    positions: Tensor, radius: float, max_neighbors: int
) -> Tensor:
    """k-capped radius edge index for ONE example.

    For each query node, the ``max_neighbors`` nearest OTHER nodes within
    ``radius`` (a hard cutoff applied to the top-k selection, mirroring
    :func:`~structbench.models.geoflare.geo_ops.ball_query`'s semantics) become
    its incoming edges. Self-matches (distance 0) are dropped, so a node's own
    row never appears as one of its neighbours.

    Parameters
    ----------
    positions:
        ``(P, dim)`` world positions for one example.
    radius:
        Distance cutoff (same units as ``positions``); a neighbour at exactly
        ``radius`` is kept (``<=``).
    max_neighbors:
        Per-node incoming-edge cap ``k``; ``E <= P * k``.

    Returns
    -------
    Tensor
        ``(2, E)`` int64 edge index (row 0 sender/neighbour, row 1
        receiver/query).
    """
    n = positions.shape[0]
    if n <= 1:
        return positions.new_zeros((2, 0), dtype=torch.int64)
    # +1 slot for the self-match (distance 0) that we then drop, so up to
    # max_neighbors true neighbours survive per node.
    eff_k = min(max_neighbors + 1, n)
    senders: list[Tensor] = []
    receivers: list[Tensor] = []
    for start in range(0, n, _QUERY_CHUNK):
        chunk = positions[start : start + _QUERY_CHUNK]
        c = chunk.shape[0]
        dist = torch.cdist(chunk, positions)  # (c, n)
        # topk(largest=False) returns ascending — nearest first, self first.
        top_dist, top_idx = torch.topk(dist, eff_k, dim=-1, largest=False)
        q = torch.arange(start, start + c, device=positions.device).unsqueeze(1)
        keep = (top_dist <= radius) & (top_idx != q)  # (c, eff_k) bool
        receivers.append(q.expand(-1, eff_k)[keep])
        senders.append(top_idx[keep])
    sender = torch.cat(senders)
    receiver = torch.cat(receivers)
    return torch.stack([sender, receiver]).to(torch.int64)


def build_radius_graph(
    positions: Tensor,
    n_per: Tensor | None,
    radius: float,
    max_neighbors: int,
) -> tuple[Tensor, Tensor]:
    """Per-example k-capped radius graph plus MGN-style edge features.

    Built ONCE per step in the simulator (from ``x_t`` on the eval path,
    ``x_last`` on the train path) and reused across every hybrid block, exactly
    as MGN builds its world-edge index once per forward and reuses it across
    all message-passing steps.

    Parameters
    ----------
    positions:
        ``(P, dim)`` world positions, examples concatenated along dim 0.
    n_per:
        ``(B,)`` per-example node counts, or ``None`` for a single example
        (the bound-case eval path).
    radius:
        Radius cutoff in the same working frame as ``positions``.
    max_neighbors:
        Per-node incoming-edge cap.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(edge_index (2, E) int64, edge_feats (E, dim + 1) float)`` where
        ``edge_feats = [x_ij, |x_ij|]`` and ``x_ij = positions[sender] -
        positions[receiver]`` — the MGN world-edge feature convention.
    """
    parts: list[Tensor] = []
    for start, end in _segments(positions.shape[0], n_per):
        local = _capped_radius_edges_single(positions[start:end], radius, max_neighbors)
        parts.append(local + start)  # re-offset to global node indices
    if parts:
        edge_index = torch.cat(parts, dim=1)
    else:
        edge_index = positions.new_zeros((2, 0), dtype=torch.int64)

    sender, receiver = edge_index[0], edge_index[1]
    x_ij = positions[sender] - positions[receiver]
    x_norm = torch.linalg.norm(x_ij, dim=-1, keepdim=True)
    edge_feats = torch.cat([x_ij, x_norm], dim=-1)
    return edge_index, edge_feats
