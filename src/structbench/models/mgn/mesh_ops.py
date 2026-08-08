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
