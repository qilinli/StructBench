import numpy as np
import torch
from torch.utils.data import DataLoader

from structbench.datasets.canonical import CaseTrajectory
from structbench.datasets.particle import WindowDataset, collate_samples


def _traj(case_id, P, T=6):
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pos[:, :, 0] = np.arange(T)[:, None]  # moves +1 mm/frame in x
    vm = np.zeros((T, P), dtype=np.float32)
    return CaseTrajectory(
        case_id, pos, np.ones(P, np.int64), vm, np.arange(T, dtype=np.float64)
    )


def test_window_dataset_sample_shapes_and_target():
    ds = WindowDataset([_traj("a", P=5)], input_frames=3)
    # T=6, input_frames=3 -> next index from 3..5 -> 3 samples
    assert len(ds) == 3
    s = ds[0]
    assert s["position_seq"].shape == (5, 3, 2)
    assert s["next_position"].shape == (5, 2)
    # frame 3 position is x=3 for all particles
    torch.testing.assert_close(s["next_position"][:, 0], torch.full((5,), 3.0))


def test_collate_concatenates_particles():
    ds = WindowDataset([_traj("a", 5), _traj("b", 4)], input_frames=3)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_samples, shuffle=False)
    batch = next(iter(loader))
    # two examples with 5 and 4 particles -> 9 rows
    assert batch["position_seq"].shape == (9, 3, 2)
    torch.testing.assert_close(batch["n_particles_per_example"], torch.tensor([5, 4]))


def test_window_dataset_sample_carries_traj_idx():
    ds = WindowDataset([_traj("a", 5), _traj("b", 4)], input_frames=3)
    # t-major, traj-minor interleaving (see WindowDataset docstring): index 0
    # is trajectory "a"'s first sample, index 1 is trajectory "b"'s first
    # sample.
    assert ds[0]["traj_idx"] == 0
    assert ds[1]["traj_idx"] == 1
    # every sample from "a" carries traj_idx 0, every sample from "b" carries 1
    traj_idxs = [ds[i]["traj_idx"] for i in range(len(ds))]
    assert traj_idxs == [0, 1] * (len(ds) // 2)


def test_collate_samples_output_unaffected_by_traj_idx_key():
    # The CGN training/eval path calls collate_samples directly; traj_idx
    # (added for the MGN mesh collate) must not change its output.
    ds = WindowDataset([_traj("a", 5), _traj("b", 4)], input_frames=3)
    batch = [ds[0], ds[1]]
    out = collate_samples(batch)
    assert set(out.keys()) == {
        "position_seq",
        "particle_type",
        "next_position",
        "next_aux",
        "n_particles_per_example",
    }
    assert "traj_idx" not in out
