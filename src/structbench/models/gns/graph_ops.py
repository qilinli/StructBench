"""Native, dependency-free graph construction for the GNS simulator.

This module provides a pure-``torch`` :func:`radius_graph` that replaces
``torch_geometric.nn.radius_graph``.  The PyG operator requires a compiled
binary backend (``pyg-lib`` / ``torch-cluster``) that is not available for the
torch build this project pins; building the neighbour graph in plain ``torch``
keeps StructBench installable from wheels on both CPU and CUDA without any extra
dependency (see Task 7).

The implementation is brute-force *within each example*: nodes are grouped by
their ``batch`` id and pairwise distances are materialised only inside each
example, so compute scales as ``O(sum_i n_i**2)`` over the per-example node
counts ``n_i`` — not ``O(N_total**2)`` over the concatenated batch.  Since
cross-example edges are forbidden by contract, distances across examples are
never computed at all (computing and masking them cost ~224x at training batch
32, measured 2026-07-02).  Memory is bounded by processing each example's query
nodes in chunks (see :data:`_QUERY_CHUNK_SIZE`), giving a peak of
``O(chunk * max_i n_i)``.  For per-example node counts beyond roughly ``1e6``
the quadratic compute becomes the bottleneck; the remedy is to swap in a
spatial-grid / cell-list backend *behind this exact interface* so callers are
unaffected.  That acceleration is intentionally not built yet (YAGNI): the v0.1
benchmark cases are far smaller, and a premature spatial index would add
complexity with no present payoff.
"""

import torch
from torch import Tensor

# Number of query nodes processed per distance-matrix block.  Caps peak memory
# at ``O(_QUERY_CHUNK_SIZE * n_i)`` where ``n_i`` is the example's node count.
_QUERY_CHUNK_SIZE = 4096


def radius_graph(
    pos: Tensor,
    r: float,
    batch: Tensor,
    *,
    max_num_neighbors: int = 20,
    loop: bool = True,
) -> Tensor:
    """Build a radius neighbourhood graph in pure ``torch``.

    For every node ``i`` this lists the nodes ``j`` that lie in the same example
    (identical ``batch`` value) and within Euclidean distance ``r`` of ``i``,
    i.e. ``||pos_i - pos_j|| <= r``.  At most ``max_num_neighbors`` neighbours
    are kept per ``i``; when more candidates qualify, the *nearest* ones are
    retained.  A self-loop ``(i, i)`` is included when ``loop`` is ``True``.

    The returned orientation is ``edge_index[0] = query node i`` (receiver) and
    ``edge_index[1] = neighbour j`` (sender), as consumed by
    :class:`~structbench.models.gns.simulator.LearnedSimulator`.  Note that
    ``torch_geometric.nn.radius_graph`` returns the *transpose* of this
    convention, and under ``max_num_neighbors`` truncation the two are not
    identical (nearest-neighbour selection depends on the query direction).
    PyG-trained weights should therefore **not** be loaded into a model trained
    with this operator, and vice versa.

    Parameters
    ----------
    pos : Tensor
        Node positions with shape ``(n_nodes, n_dim)`` in the caller's working
        units (millimetres for the ML layer; the operator is unit-agnostic).
    r : float
        Connectivity radius in the same units as ``pos``.  Neighbours satisfy
        ``distance <= r`` (inclusive).
    batch : Tensor
        Integer example id per node, shape ``(n_nodes,)``.  Nodes only connect
        to other nodes sharing their ``batch`` value, so distinct examples in a
        batch never cross-connect.
    max_num_neighbors : int, optional
        Maximum number of neighbours kept per central node, counting the
        self-loop when ``loop`` is ``True`` (default ``20``).  When more nodes
        fall within ``r`` the nearest ``max_num_neighbors`` are kept.
    loop : bool, optional
        Whether to include the self-loop ``(i, i)`` (default ``True``).

    Returns
    -------
    Tensor
        ``edge_index`` of shape ``(2, n_edges)`` and dtype ``torch.long`` on the
        same device as ``pos``.  Row 0 lists central nodes ``i``; row 1 lists
        the corresponding neighbours ``j``.

    Notes
    -----
    Nodes are grouped by ``batch`` id and searched within their example only —
    cross-example distances are never computed (they can never be edges).
    Compute is ``O(sum_i n_i**2)`` over per-example node counts; peak memory is
    bounded to ``O(_QUERY_CHUNK_SIZE * max_i n_i)`` by chunking each example's
    query nodes.  For per-example counts beyond roughly ``1e6`` replace this
    brute-force backend with a spatial-grid / cell-list implementation behind
    the same signature.  Edge ordering is unspecified; consumers must not rely
    on it.
    """
    if pos.dim() != 2:
        raise ValueError(
            f"Expected pos with 2 dimensions, got shape {tuple(pos.shape)}"
        )

    n_nodes = pos.shape[0]
    device = pos.device
    if n_nodes == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    row_blocks: list[Tensor] = []
    col_blocks: list[Tensor] = []

    for example_id in torch.unique(batch):
        # Global node indices of this example; all search happens inside it.
        (example_idx,) = torch.where(batch == example_id)
        example_pos = pos[example_idx]
        n_example = example_idx.numel()
        # Cannot keep more neighbours than the example has nodes.
        keep = min(max_num_neighbors, n_example)

        for start in range(0, n_example, _QUERY_CHUNK_SIZE):
            end = min(start + _QUERY_CHUNK_SIZE, n_example)
            chunk = end - start

            # Pairwise distances from this block of query nodes to the
            # example's nodes: shape (chunk, n_example) — the O(chunk * n_i)
            # term. Other examples are never touched.
            dist = torch.cdist(example_pos[start:end], example_pos)
            within = dist <= r

            if not loop:
                local = torch.arange(chunk, device=device)
                within[local, torch.arange(start, end, device=device)] = False

            # Push invalid candidates to +inf so the nearest valid neighbours
            # are the smallest entries; take the closest ``keep`` per query.
            masked = torch.where(within, dist, torch.full_like(dist, float("inf")))
            nearest_dist, nearest_col = torch.topk(masked, k=keep, dim=1, largest=False)

            # Finite distances mark genuine neighbours; +inf padding is
            # discarded, so query nodes with fewer than ``keep`` neighbours
            # contribute fewer edges (and isolated nodes contribute none).
            valid = torch.isfinite(nearest_dist)
            rows = example_idx[start:end].unsqueeze(1).expand(-1, keep)
            row_blocks.append(rows[valid])
            col_blocks.append(example_idx[nearest_col[valid]])

    return torch.stack([torch.cat(row_blocks), torch.cat(col_blocks)], dim=0)
