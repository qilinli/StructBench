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
    w2 = world_edges(
        pos, radius=2.0, mesh_edge_index=torch.empty(2, 0, dtype=torch.int64)
    )
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
