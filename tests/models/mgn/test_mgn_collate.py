import numpy as np
import pytest
import torch

from structbench.datasets.canonical import CaseTrajectory
from structbench.datasets.particle import collate_samples
from structbench.models.mgn.collate import (
    MeshStatic,
    collate_mesh_samples,
    mesh_static_from_trajectory,
)


def _sample(n_particles: int, traj_idx: int) -> dict:
    return {
        "position_seq": torch.zeros(n_particles, 3, 2),
        "particle_type": torch.ones(n_particles, dtype=torch.int64),
        "next_position": torch.zeros(n_particles, 2),
        "next_aux": torch.zeros(n_particles),
        "n_particles": n_particles,
        "traj_idx": traj_idx,
    }


def _static(n_particles: int) -> MeshStatic:
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    coords = torch.arange(n_particles * 2, dtype=torch.float32).reshape(n_particles, 2)
    return MeshStatic(mesh_edge_index=edges, reference_coords=coords)


def test_collate_mesh_samples_offsets_edges_by_batch_order():
    batch = [_sample(3, traj_idx=0), _sample(2, traj_idx=1)]
    statics = [_static(3), _static(2)]

    out = collate_mesh_samples(batch, statics)

    torch.testing.assert_close(
        out["mesh_edge_index"],
        torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]], dtype=torch.int64),
    )


def test_collate_mesh_samples_reference_coords_is_row_concat():
    batch = [_sample(3, traj_idx=0), _sample(2, traj_idx=1)]
    statics = [_static(3), _static(2)]

    out = collate_mesh_samples(batch, statics)

    expected = torch.cat([statics[0].reference_coords, statics[1].reference_coords])
    torch.testing.assert_close(out["reference_coords"], expected)
    assert out["reference_coords"].shape == (5, 2)


def test_collate_mesh_samples_includes_all_collate_samples_keys():
    batch = [_sample(3, traj_idx=0), _sample(2, traj_idx=1)]
    statics = [_static(3), _static(2)]

    out = collate_mesh_samples(batch, statics)
    base = collate_samples(batch)

    assert base.keys() <= out.keys()
    for key, value in base.items():
        torch.testing.assert_close(out[key], value)


def test_collate_mesh_samples_statics_indexed_by_traj_idx_not_position():
    # traj_idx order is reversed relative to statics list and batch position,
    # and the two statics have distinguishable edge sets so an implementation
    # that (incorrectly) indexes statics by batch position instead of
    # traj_idx produces a different (and here, out-of-range-for-P=2) result.
    static_traj0 = MeshStatic(
        mesh_edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        reference_coords=torch.zeros(2, 2),
    )
    static_traj1 = MeshStatic(
        mesh_edge_index=torch.tensor([[0, 2], [2, 0]], dtype=torch.int64),
        reference_coords=torch.zeros(3, 2),
    )
    statics = [static_traj0, static_traj1]  # statics[traj_idx]

    batch = [_sample(3, traj_idx=1), _sample(2, traj_idx=0)]

    out = collate_mesh_samples(batch, statics)

    # batch[0] (P=3, traj_idx=1) -> statics[1] edges [[0,2],[2,0]], offset 0.
    # batch[1] (P=2, traj_idx=0) -> statics[0] edges [[0,1],[1,0]], offset 3.
    torch.testing.assert_close(
        out["mesh_edge_index"],
        torch.tensor([[0, 2, 3, 4], [2, 0, 4, 3]], dtype=torch.int64),
    )


def _traj_with_mesh(case_id: str, n_particles: int, dim: int = 2) -> CaseTrajectory:
    T = 2
    pos = np.zeros((T, n_particles, dim), dtype=np.float32)
    ptype = np.ones(n_particles, dtype=np.int64)
    aux = np.zeros((T, n_particles), dtype=np.float32)
    time = np.arange(T, dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64) if n_particles >= 3 else None
    ref = np.zeros((n_particles, dim), dtype=np.float32)
    return CaseTrajectory(
        case_id, pos, ptype, aux, time, cells=cells, reference_coords=ref
    )


def test_mesh_static_from_trajectory_builds_edges_from_cells():
    traj = _traj_with_mesh("a", n_particles=3)
    static = mesh_static_from_trajectory(traj)
    assert static.mesh_edge_index.dtype == torch.int64
    assert static.mesh_edge_index.shape[0] == 2
    # cells [[0,1,2]] -> pairs (0,1),(0,2),(1,2) bidirectional -> 6 edges
    assert static.mesh_edge_index.shape[1] == 6
    torch.testing.assert_close(
        static.reference_coords, torch.from_numpy(traj.reference_coords)
    )


def test_mesh_static_from_trajectory_raises_without_cells():
    traj = _traj_with_mesh("a", n_particles=3)
    traj.cells = None
    with pytest.raises(ValueError):
        mesh_static_from_trajectory(traj)


def test_mesh_static_from_trajectory_raises_without_reference_coords():
    traj = _traj_with_mesh("a", n_particles=3)
    traj.reference_coords = None
    with pytest.raises(ValueError):
        mesh_static_from_trajectory(traj)
