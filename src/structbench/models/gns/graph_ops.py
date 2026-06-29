"""Native, dependency-free graph construction for the GNS simulator.

This module provides a pure-``torch`` :func:`radius_graph` that replaces
``torch_geometric.nn.radius_graph``.  The PyG operator requires a compiled
binary backend (``pyg-lib`` / ``torch-cluster``) that is not available for the
torch build this project pins; building the neighbour graph in plain ``torch``
keeps StructBench installable from wheels on both CPU and CUDA without any extra
dependency (see Task 7).

The implementation is brute-force: it materialises pairwise distances between a
chunk of query nodes and *all* nodes, so compute scales as ``O(N**2)`` in the
number of nodes ``N``.  Memory is bounded by processing the query nodes in
chunks (see :data:`_QUERY_CHUNK_SIZE`), giving a peak of ``O(chunk * N)`` rather
than ``O(N**2)``.  For datasets beyond roughly ``1e6`` nodes this quadratic
compute becomes the bottleneck; the remedy is to swap in a spatial-grid /
cell-list backend *behind this exact interface* so callers are unaffected.  That
acceleration is intentionally not built yet (YAGNI): the v0.1 benchmark cases
are far smaller, and a premature spatial index would add complexity with no
present payoff.
"""

import torch
from torch import Tensor

# Number of query nodes processed per distance-matrix block.  Caps peak memory
# at ``O(_QUERY_CHUNK_SIZE * N)`` regardless of the total node count ``N``.
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
    Compute is ``O(n_nodes**2)`` because distances to all nodes are evaluated
    for every query node; peak memory is bounded to ``O(_QUERY_CHUNK_SIZE *
    n_nodes)`` by chunking the query nodes.  For ``n_nodes`` beyond roughly
    ``1e6`` replace this brute-force backend with a spatial-grid / cell-list
    implementation behind the same signature.  Edge ordering is unspecified;
    consumers must not rely on it.
    """
    if pos.dim() != 2:
        raise ValueError(
            f"Expected pos with 2 dimensions, got shape {tuple(pos.shape)}"
        )

    n_nodes = pos.shape[0]
    device = pos.device
    if n_nodes == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # Cannot keep more neighbours than there are nodes.
    keep = min(max_num_neighbors, n_nodes)

    row_blocks: list[Tensor] = []
    col_blocks: list[Tensor] = []

    for start in range(0, n_nodes, _QUERY_CHUNK_SIZE):
        end = min(start + _QUERY_CHUNK_SIZE, n_nodes)
        chunk = end - start

        # Pairwise Euclidean distances from this block of query nodes to all
        # nodes: shape (chunk, n_nodes).  This is the O(chunk * n_nodes) term.
        dist = torch.cdist(pos[start:end], pos)

        # A query may only connect to nodes in the same example.
        same_example = batch[start:end].unsqueeze(1) == batch.unsqueeze(0)
        within = (dist <= r) & same_example

        if not loop:
            local = torch.arange(chunk, device=device)
            within[local, torch.arange(start, end, device=device)] = False

        # Push invalid candidates to +inf so the nearest valid neighbours are
        # the smallest entries; take the closest ``keep`` per query node.
        masked = torch.where(within, dist, torch.full_like(dist, float("inf")))
        nearest_dist, nearest_col = torch.topk(masked, k=keep, dim=1, largest=False)

        # Finite distances mark genuine neighbours; +inf padding is discarded,
        # so query nodes with fewer than ``keep`` neighbours contribute fewer
        # edges (and isolated nodes contribute none).
        valid = torch.isfinite(nearest_dist)
        rows = torch.arange(start, end, device=device).unsqueeze(1).expand(-1, keep)
        row_blocks.append(rows[valid])
        col_blocks.append(nearest_col[valid])

    return torch.stack([torch.cat(row_blocks), torch.cat(col_blocks)], dim=0)
