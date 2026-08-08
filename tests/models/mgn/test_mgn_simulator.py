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


def test_untrained_mgn_through_eval_rollout(tmp_path):
    """The ①-c1 deliverable gate: an UNTRAINED MeshSimulator satisfies the
    eval protocol end to end on a synthetic mesh case (ADR-0043)."""
    import numpy as np

    from structbench.core import write_case
    from structbench.core.io.meshgraphnets import build_deforming_plate_case
    from structbench.datasets import load_case_trajectory
    from structbench.eval import one_step_aux_rmse, one_step_position_rmse, rollout

    rng = np.random.default_rng(7)
    P, T = 6, 8
    world0 = rng.random((P, 3)).astype(np.float32)
    drift = rng.random((T, P, 3)).astype(np.float32) * 0.01
    arrays = {
        "cells": np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32),
        "node_type": np.array([0, 0, 0, 0, 1, 3], dtype=np.int32),
        "mesh_pos": world0.copy(),
        "world_pos": (world0[None] + np.cumsum(drift, axis=0)).astype(np.float32),
        "stress": rng.random((T, P, 1)).astype(np.float32),
    }
    case = build_deforming_plate_case(arrays, source_units="kg-m-s", case_id="mgn-it")
    path = tmp_path / "mgn-it.h5"
    write_case(case, path)
    traj = load_case_trajectory(path, aux_field="von_mises_stress")

    torch.manual_seed(0)
    sim = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=50.0)
    sim.bind_case(
        torch.from_numpy(traj.cells),
        torch.from_numpy(traj.reference_coords),
        torch.from_numpy(traj.particle_type),
        torch.from_numpy(traj.positions),
    )

    from structbench.benchmarks import get_benchmark

    spec = get_benchmark("deforming_plate")
    sim.reset_rollout()
    result = rollout(sim, traj, input_frames=2, kinematic_types=(1, 3), qois=spec.qois)
    assert result.predicted_positions.shape == (T, P, 3)
    assert result.predicted_aux.shape == (T, P)
    assert result.position_rmse.shape == (T - 2,)
    # the ADR-0043 QoIs evaluate on MGN output: populated, finite
    assert set(result.qoi_pred) == {"peak_vm_stress", "terminal_peak_deflection"}
    assert all(np.isfinite(v) for v in result.qoi_pred.values())
    assert all(np.isfinite(v) for v in result.qoi_error.values())
    # kinematic rows are GT-prescribed in the output
    np.testing.assert_allclose(
        result.predicted_positions[:, 4], traj.positions[:, 4], rtol=1e-5
    )

    sim.reset_rollout()
    one_pos = one_step_position_rmse(sim, traj, input_frames=2, kinematic_types=(1, 3))
    assert one_pos.shape == (T - 2,)

    sim.reset_rollout()
    one_aux = one_step_aux_rmse(sim, traj, input_frames=2, kinematic_types=(1, 3))
    assert one_aux.shape == (T - 2,)


def test_scripted_types_must_be_subset_of_kinematic_types():
    with pytest.raises(ValueError):
        MeshSimulator(scripted_types=(5,), kinematic_types=(1, 3))


def test_forward_train_shapes_and_target_semantics():
    torch.manual_seed(0)
    sim = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=0.5)
    P = 5
    x = torch.rand(P, 3)
    nxt = x + 0.1
    aux = torch.rand(P)
    types = torch.tensor([0, 0, 1, 3, 0], dtype=torch.int64)
    mesh = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
    ref = torch.rand(P, 3)
    pred, target = sim.forward_train(
        x, nxt, aux, types, mesh, ref, torch.tensor([P]), accumulate=False
    )
    assert pred.shape == (P, 4) and target.shape == (P, 4)
    # untrained target normalizer is identity: target == [v_target | stress]
    torch.testing.assert_close(target[:, :3], nxt - x)
    torch.testing.assert_close(target[:, 3], aux)


def test_train_and_eval_paths_build_identical_features():
    """The final-review seam: train and eval must share ONE feature builder."""
    sim, gt, types = _bound_sim()
    x_last = gt[1]  # last frame of the first window
    # eval-path features: capture what predict_positions feeds the net
    captured: list[torch.Tensor] = []
    hook = sim._net.register_forward_pre_hook(
        lambda mod, args: captured.append(tuple(a.detach().clone() for a in args))
    )
    sim.reset_rollout()
    sim.predict_positions(
        gt[0:2].permute(1, 0, 2).contiguous(), torch.tensor([5]), types
    )
    hook.remove()
    eval_args = captured[0]  # ALL FIVE network inputs, not just node features
    # train-path features on the SAME inputs (no noise; scripted velocity in
    # training comes from next_positions == GT[2], identical to the bound GT)
    captured.clear()
    hook = sim._net.register_forward_pre_hook(
        lambda mod, args: captured.append(tuple(a.detach().clone() for a in args))
    )
    sim.forward_train(
        x_last,
        gt[2],
        torch.zeros(5),
        types,
        sim._mesh_edge_index,
        sim._reference_coords,
        torch.tensor([5]),
        accumulate=False,
    )
    hook.remove()
    for train_arg, eval_arg in zip(captured[0], eval_args, strict=True):
        torch.testing.assert_close(train_arg, eval_arg)


def test_forward_train_accumulates_normalizers_when_asked():
    torch.manual_seed(0)
    sim = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=0.5)
    P = 4
    args = (
        torch.rand(P, 3),
        torch.rand(P, 3) + 1.0,
        torch.rand(P),
        torch.zeros(P, dtype=torch.int64),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        torch.rand(P, 3),
    )
    norms = [
        sim._target_normalizer,
        sim._node_normalizer,
        sim._mesh_edge_normalizer,
        sim._world_edge_normalizer,
    ]  # bind to the REAL attribute names
    before = [int(n._n_accumulations) for n in norms]
    sim.forward_train(*args, torch.tensor([P]), accumulate=True)
    sim.forward_train(*args, torch.tensor([P]), accumulate=False)
    after = [int(n._n_accumulations) for n in norms]
    assert after == [b + 1 for b in before]  # ALL FOUR accumulate exactly once


def test_forward_train_never_builds_cross_example_world_edges():
    """Two collated examples whose nodes coincide ACROSS examples: batched
    world edges must stay within-example (the plan review's Critical — a
    whole-batch radius query would invent cross-case edges)."""
    torch.manual_seed(0)
    sim = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=100.0)
    P = 2  # per example; positions overlap across examples deliberately
    x = torch.zeros(2 * P, 3)  # ALL nodes coincide -> max cross-example risk
    nxt = x + 0.1
    aux = torch.zeros(2 * P)
    types = torch.zeros(2 * P, dtype=torch.int64)
    mesh = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int64)
    ref = torch.zeros(2 * P, 3)
    captured: list[tuple] = []
    hook = sim._net.register_forward_pre_hook(
        lambda mod, args: captured.append(tuple(a.detach().clone() for a in args))
    )
    sim.forward_train(
        x, nxt, aux, types, mesh, ref, torch.tensor([P, P]), accumulate=False
    )
    hook.remove()
    world_ei = captured[0][3]  # (2, Ew) — bind index to the real net arg order
    example_of = torch.tensor([0, 0, 1, 1])
    assert (example_of[world_ei[0]] == example_of[world_ei[1]]).all()
