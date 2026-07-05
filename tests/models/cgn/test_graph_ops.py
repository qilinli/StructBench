import torch

from structbench.models.cgn.graph_ops import radius_graph


def _edge_set(edge_index):
    """Return the graph's directed edges as a set of ``(i, j)`` int tuples."""
    return {(int(i), int(j)) for i, j in edge_index.t().tolist()}


def test_radius_graph_edges_self_loops_and_batch_separation():
    # Two examples on a line. Nodes 0 and 4 occupy the same position but live
    # in different examples, so they must never connect. Nodes 3 and 5 are
    # isolated within their example. r = 1.5 so only unit-spaced neighbours
    # connect.
    pos = torch.tensor(
        [
            [0.0, 0.0],  # 0  example 0
            [1.0, 0.0],  # 1  example 0
            [2.0, 0.0],  # 2  example 0
            [5.0, 5.0],  # 3  example 0 (isolated)
            [0.0, 0.0],  # 4  example 1 (same xy as node 0)
            [10.0, 10.0],  # 5  example 1 (isolated)
        ]
    )
    batch = torch.tensor([0, 0, 0, 0, 1, 1])

    edge_index = radius_graph(pos, r=1.5, batch=batch, max_num_neighbors=20, loop=True)

    expected = {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
    }
    edges = _edge_set(edge_index)
    assert edges == expected
    assert edge_index.dtype == torch.long
    assert edge_index.shape[0] == 2
    # Self-loops are present for every node when loop=True.
    assert all((node, node) in edges for node in range(6))
    # Batch separation: identical-position nodes 0 and 4 do not cross-connect.
    assert (0, 4) not in edges
    assert (4, 0) not in edges


def test_radius_graph_loop_false_drops_self_edges():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 5.0]])
    batch = torch.zeros(4, dtype=torch.long)

    edge_index = radius_graph(pos, r=1.5, batch=batch, loop=False)

    edges = _edge_set(edge_index)
    # Only the unit-spaced pairs survive; the isolated node 3 yields nothing.
    assert edges == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert all(i != j for i, j in edges)


def test_radius_graph_batched_equals_per_example_union():
    # The batched call must produce exactly the union of per-example calls
    # (with indices offset back to global) — the invariant that lets the
    # implementation compute distances per example instead of across the
    # whole concatenated batch. Examples differ in size, and one batch id is
    # non-contiguous to pin the general contract, not just collate's layout.
    torch.manual_seed(0)
    examples = [torch.rand(7, 2), torch.rand(3, 2), torch.rand(5, 2)]
    pos = torch.cat(examples)
    batch = torch.tensor([0] * 7 + [2] * 3 + [1] * 5)  # ids not consecutive

    batched = _edge_set(radius_graph(pos, r=0.4, batch=batch, max_num_neighbors=4))

    expected = set()
    offset = 0
    for example in examples:
        zeros = torch.zeros(example.shape[0], dtype=torch.long)
        local = radius_graph(example, r=0.4, batch=zeros, max_num_neighbors=4)
        expected |= {(i + offset, j + offset) for i, j in _edge_set(local)}
        offset += example.shape[0]

    assert batched == expected


def test_radius_graph_max_num_neighbors_keeps_nearest():
    # Node 0 has three candidates within r (0.5, 1.0, 1.4 away). With a cap of
    # 2 it keeps the two nearest: itself (distance 0) and node 1 (distance 0.5).
    pos = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.4, 0.0]])
    batch = torch.zeros(4, dtype=torch.long)

    edge_index = radius_graph(pos, r=1.5, batch=batch, max_num_neighbors=2, loop=True)

    edges = _edge_set(edge_index)
    neighbours_of_0 = {j for i, j in edges if i == 0}
    assert neighbours_of_0 == {0, 1}
    # The cap holds for every central node.
    rows = edge_index[0].tolist()
    for node in set(rows):
        assert rows.count(node) <= 2
