import numpy as np
import pytest
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
        "input_aux",  # ADR-0060 state-feedback input (additive)
        "n_particles_per_example",
    }
    assert "traj_idx" not in out


# --- ADR-0050/0051 k-frames-per-call: k-frame target span ---


def test_window_dataset_kframe_target_shapes_and_count():
    # T=6, input_frames=3, k=2 -> starts t in {3,4} (t+k<=6) -> 2 samples;
    # target span positions[t:t+2] -> (P, k, dim) / (P, k).
    ds = WindowDataset([_traj("a", P=5)], input_frames=3, target_frames=2)
    assert len(ds) == 2
    s = ds[0]
    assert s["position_seq"].shape == (5, 3, 2)
    assert s["next_position"].shape == (5, 2, 2)  # (P, k, dim)
    assert s["next_aux"].shape == (5, 2, 1)  # (P, k, C)
    # frames 3,4 have x = 3,4 for every particle
    torch.testing.assert_close(
        s["next_position"][:, :, 0], torch.tensor([[3.0, 4.0]] * 5)
    )


def test_window_dataset_oneshot_is_one_sample_per_trajectory():
    # k = T - input_frames covers the whole horizon in one window.
    ds = WindowDataset([_traj("a", 5), _traj("b", 4)], input_frames=3, target_frames=3)
    assert len(ds) == 2  # exactly one per trajectory
    s = ds[0]
    assert s["next_position"].shape == (5, 3, 2)
    torch.testing.assert_close(
        s["next_position"][:, :, 0], torch.tensor([[3.0, 4.0, 5.0]] * 5)
    )


def test_window_dataset_k1_default_matches_pre_0051():
    # The default (k=1) must yield the exact pre-0051 sample count and a 2-D
    # single-frame target, so every k=1 family is byte-identical.
    default = WindowDataset([_traj("a", 5)], input_frames=3)
    explicit = WindowDataset([_traj("a", 5)], input_frames=3, target_frames=1)
    assert len(default) == len(explicit) == 3
    assert default[0]["next_position"].shape == (5, 2)  # (P, dim), not (P, 1, dim)


# --- ADR-0062: FlowMapPairDataset -----------------------------------------


def _pair_traj(T: int = 8, P: int = 4, dim: int = 2, C: int = 2) -> CaseTrajectory:
    rng = np.random.default_rng(3)
    return CaseTrajectory(
        case_id="fm",
        positions=rng.random((T, P, dim)).astype(np.float32),
        particle_type=np.zeros(P, dtype=np.int64),
        aux=rng.random((T, P, C)).astype(np.float32),
        time=np.arange(T, dtype=np.float64),
    )


def test_flowmap_pairs_full_span():
    from structbench.datasets import FlowMapPairDataset

    # T=8, input_frames=2 -> t0 in [1, 6], t in [t0+1, 7]
    tr = _pair_traj()
    ds = FlowMapPairDataset([tr], input_frames=2, max_dt=0)
    assert len(ds) == 6 + 5 + 4 + 3 + 2 + 1
    dts = [
        int(ds[i]["target_frame"]) - int(ds[i]["anchor_frame"]) for i in range(len(ds))
    ]
    # The uncapped index reaches the full horizon offset (T-1) - t0_min = 6.
    assert max(dts) == 6
    assert min(dts) == 1


def test_flowmap_pairs_max_dt_cap():
    from structbench.datasets import FlowMapPairDataset

    tr = _pair_traj()
    ds = FlowMapPairDataset([tr], input_frames=2, max_dt=3)
    # per t0 = 1..6: min(t0+3, 7) - t0 = 3, 3, 3, 3, 2, 1
    assert len(ds) == 3 + 3 + 3 + 3 + 2 + 1
    assert (
        max(
            int(ds[i]["target_frame"]) - int(ds[i]["anchor_frame"])
            for i in range(len(ds))
        )
        == 3
    )
    with pytest.raises(ValueError, match="max_dt"):
        FlowMapPairDataset([tr], input_frames=2, max_dt=-1)


def test_flowmap_sample_contract():
    from structbench.datasets import FlowMapPairDataset

    tr = _pair_traj()
    ds = FlowMapPairDataset([tr], input_frames=2, max_dt=0)
    s = ds[0]
    t0, t = int(s["anchor_frame"]), int(s["target_frame"])
    assert t0 == 1 and t == 2  # first indexed pair
    # position_seq is the ANCHOR PAIR (t0-1, t0), particle-major.
    assert s["position_seq"].shape == (4, 2, 2)
    np.testing.assert_array_equal(
        s["position_seq"].numpy(),
        np.transpose(tr.positions[t0 - 1 : t0 + 1], (1, 0, 2)),
    )
    # input_aux carries the ANCHOR aux (ADR-0062 reuse of the ADR-0060 key).
    np.testing.assert_array_equal(s["input_aux"].numpy(), tr.aux[t0])
    np.testing.assert_array_equal(s["next_position"].numpy(), tr.positions[t])
    np.testing.assert_array_equal(s["next_aux"].numpy(), tr.aux[t])


# --- ADR-0063: FlowMapChainDataset -----------------------------------------


def test_flowmap_chain_index_and_contract():
    from structbench.datasets import FlowMapChainDataset

    tr = _pair_traj()  # T=8, P=4
    ds = FlowMapChainDataset([tr], input_frames=2, max_dt=3)
    # t0 in [1,4]; t1 in [t0+2, min(t0+3, 6)]; t2 in [t1+1, min(t1+3, 7)]:
    # t0=1 -> 6, t0=2 -> 5, t0=3 -> 3, t0=4 -> 1 chains.
    assert len(ds) == 15
    s = ds[0]
    t0, t1, t2 = int(s["anchor_frame"]), int(s["chain_frame"]), int(s["target_frame"])
    assert (t0, t1, t2) == (1, 3, 4)
    assert 2 <= t1 - t0 <= 3 and 1 <= t2 - t1 <= 3
    # anchor pair (t0-1, t0); targets stacked (t1-1, t1, t2).
    np.testing.assert_array_equal(
        s["position_seq"].numpy(),
        np.transpose(tr.positions[t0 - 1 : t0 + 1], (1, 0, 2)),
    )
    assert s["next_position"].shape == (4, 3, 2)
    assert s["next_aux"].shape == (4, 3, 2)
    for j, f in enumerate((t1 - 1, t1, t2)):
        np.testing.assert_array_equal(s["next_position"][:, j].numpy(), tr.positions[f])
        np.testing.assert_array_equal(s["next_aux"][:, j].numpy(), tr.aux[f])
    np.testing.assert_array_equal(s["input_aux"].numpy(), tr.aux[t0])
    with pytest.raises(ValueError, match="max_dt"):
        FlowMapChainDataset([tr], input_frames=2, max_dt=1)


def test_flowmap_chain_generations():
    from structbench.datasets import FlowMapChainDataset

    tr = _pair_traj(T=20)
    ds = FlowMapChainDataset([tr], input_frames=2, max_dt=4, generations=3)
    # anchors t0 in [1, T-2-2G] = [1, 12] -> 12 entries (anchor-enumerated).
    assert len(ds) == 12
    torch.manual_seed(0)
    s = ds[0]
    G = 3
    assert s["next_position"].shape == (4, 2 * G + 1, 2)
    assert s["chain_frames"].shape == (G,)
    t0 = int(s["anchor_frame"])
    chain = [t0] + [int(f) for f in s["chain_frames"]] + [int(s["target_frame"])]
    # pair hops in [2, max_dt]; final hop in [1, max_dt]; strictly inside T.
    for a, b in zip(chain[:-2], chain[1:-1], strict=True):
        assert 2 <= b - a <= 4
    assert 1 <= chain[-1] - chain[-2] <= 4
    assert chain[-1] <= 19
    # targets are the (t_i-1, t_i) pairs then t_final, in order.
    frames = [f for t in chain[1:-1] for f in (t - 1, t)] + [chain[-1]]
    for j, f in enumerate(frames):
        np.testing.assert_array_equal(s["next_position"][:, j].numpy(), tr.positions[f])
    # gens=1 path unchanged (enumerated triples).
    ds1 = FlowMapChainDataset([tr], input_frames=2, max_dt=4, generations=1)
    assert "chain_frame" in ds1[0] and "chain_frames" not in ds1[0]
    with pytest.raises(ValueError, match="generations"):
        FlowMapChainDataset([tr], input_frames=2, max_dt=4, generations=0)
