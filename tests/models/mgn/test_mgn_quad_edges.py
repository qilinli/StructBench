"""A 2D four-node cell is a quadrilateral: its sides are edges, not its
diagonals (2026-11-06). Without ``dim`` every vertex pair is kept, which is
what triangles and tetrahedra need and what every earlier benchmark used."""

import torch

from structbench.models.mgn.mesh_ops import cells_to_edges


def _edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(s), int(r)) for s, r in edge_index.t()}


def test_quad_in_2d_has_sides_only():
    quad = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    es = _edge_set(cells_to_edges(quad, dim=2))
    assert len(es) == 8
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        assert (a, b) in es and (b, a) in es
    assert (0, 2) not in es and (1, 3) not in es


def test_four_node_cell_in_3d_is_a_tetrahedron():
    tet = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    assert len(_edge_set(cells_to_edges(tet, dim=3))) == 12
    assert len(_edge_set(cells_to_edges(tet))) == 12  # dim unknown: unchanged


def test_two_quads_sharing_a_side_dedup():
    cells = torch.tensor([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=torch.int64)
    es = _edge_set(cells_to_edges(cells, dim=2))
    # 4 + 4 sides, the shared side (1, 4) counted once: 7 undirected pairs
    assert len(es) == 14


def test_triangles_in_2d_keep_every_pair():
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    assert len(_edge_set(cells_to_edges(tri, dim=2))) == 6
