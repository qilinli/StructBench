import numpy as np
import pytest
import torch

from structbench.models.mgn import MeshSimulator


def _bound_sim(T=6, P=5, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sim = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=0.5)
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
    ref = torch.tensor(rng.random((P, 3)), dtype=torch.float32)
    types = torch.tensor([0, 0, 1, 3, 0], dtype=torch.int64)
    gt = torch.tensor(rng.random((T, P, 3)), dtype=torch.float32).cumsum(0)
    sim.bind_case(cells, ref, types, gt)
    return sim, gt, types


def test_predict_shapes_and_pointer_advance():
    sim, gt, types = _bound_sim()
    npp = torch.tensor([5])
    win = gt[0:2].permute(1, 0, 2).contiguous()  # frames [0,1] -> predict frame 2
    nxt, aux = sim.predict_positions(win, npp, types)
    assert nxt.shape == (5, 3) and aux.shape == (5, 1)
    # next call must accept the GT-overwritten window for frame 3
    win2 = torch.stack([gt[1], gt[2]], dim=1)  # (P, 2, dim), kin rows GT
    nxt2, _ = sim.predict_positions(win2, npp, types)
    assert nxt2.shape == (5, 3)


def test_tripwire_fires_on_desynced_window():
    sim, gt, types = _bound_sim()
    npp = torch.tensor([5])
    sim.predict_positions(gt[0:2].permute(1, 0, 2).contiguous(), npp, types)
    stale = gt[0:2].permute(1, 0, 2).contiguous()  # same window again: t desync
    with pytest.raises(RuntimeError, match="reset_rollout"):
        sim.predict_positions(stale, npp, types)
    sim.reset_rollout()
    nxt, _ = sim.predict_positions(stale, npp, types)  # re-anchors cleanly
    assert nxt.shape == (5, 3)


def test_tripwire_fires_on_foreign_window_at_first_call():
    # Anchoring is deterministic (t = F), so a window from a DIFFERENT
    # trajectory must trip the GT check on the very first call.
    sim, gt, types = _bound_sim()
    sim.reset_rollout()
    foreign = torch.rand(5, 2, 3) + 50.0  # nowhere near the bound GT
    with pytest.raises(RuntimeError, match="bind_case"):
        sim.predict_positions(foreign, torch.tensor([5]), types)


def test_no_kinematic_nodes_runs_stateless():
    torch.manual_seed(0)
    sim = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=0.5)
    P = 4
    cells = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    ref = torch.rand(P, 3)
    types = torch.zeros(P, dtype=torch.int64)  # all NORMAL
    gt = torch.rand(3, P, 3)
    sim.bind_case(cells, ref, types, gt)
    for _ in range(4):  # arbitrary call count, no tripwire without kin rows
        nxt, aux = sim.predict_positions(torch.rand(P, 2, 3), torch.tensor([P]), types)
    assert nxt.shape == (P, 3) and aux.shape == (P, 1)


def test_scripted_velocity_zeroes_at_final_bound_frame():
    # Drive the pointer through t = F, F+1, ..., T (the last bound GT frame)
    # with correctly-slid GT windows so the tripwire passes at every call,
    # exercising the final-frame guard: at t == T there is no GT[t] to read
    # for scripted velocity, so that node-feature block must be zero (not a
    # crash). One call further (t == T + 1) is NOT exercised here: the
    # tripwire's `gt_positions[t - 1]` read is unconditional, so that call
    # would compute t - 1 == T, an out-of-range index into the T-frame
    # gt_positions tensor -- an IndexError outside the brief's pinned
    # final-frame-guard contract (which covers only the scripted-velocity
    # read, not the tripwire).
    sim, gt, types = _bound_sim(T=4, P=5)
    npp = torch.tensor([5])
    nts, dim = sim._node_type_size, sim._dim

    captured: list[torch.Tensor] = []
    handle = sim._node_normalizer.register_forward_pre_hook(
        lambda module, args: captured.append(args[0])
    )
    try:
        win1 = gt[0:2].permute(1, 0, 2).contiguous()  # last frame gt[1] -> t=2
        sim.predict_positions(win1, npp, types)
        win2 = torch.stack([gt[1], gt[2]], dim=1)  # last frame gt[2] -> t=3
        sim.predict_positions(win2, npp, types)
        win3 = torch.stack([gt[2], gt[3]], dim=1)  # last frame gt[3] -> t=4==T
        nxt3, aux3 = sim.predict_positions(win3, npp, types)
    finally:
        handle.remove()

    assert nxt3.shape == (5, 3) and aux3.shape == (5, 1)
    scripted_velocity_block = captured[-1][:, nts : nts + dim]
    assert torch.equal(
        scripted_velocity_block, torch.zeros_like(scripted_velocity_block)
    )


def test_save_load_roundtrip(tmp_path):
    sim, gt, types = _bound_sim()
    p = tmp_path / "mgn.pt"
    sim.save(p)
    sim2 = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=0.5)
    sim2.load(p)
    for (k1, v1), (k2, v2) in zip(
        sim.state_dict().items(), sim2.state_dict().items(), strict=True
    ):
        assert k1 == k2
        torch.testing.assert_close(v1, v2)
