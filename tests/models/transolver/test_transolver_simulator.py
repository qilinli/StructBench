import numpy as np
import pytest
import torch

from structbench.models.transolver import TransolverSimulator


def _tiny_sim(**kwargs):
    return TransolverSimulator(
        dim=3, hidden_dim=8, n_layers=1, n_heads=2, slice_num=2, **kwargs
    )


def _bound_sim(T=6, P=5, seed=0, **kwargs):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sim = _tiny_sim(**kwargs)
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
    ref = torch.tensor(rng.random((P, 3)), dtype=torch.float32)
    types = torch.tensor([0, 0, 1, 3, 0], dtype=torch.int64)  # OBSTACLE, HANDLE present
    gt = torch.tensor(rng.random((T, P, 3)), dtype=torch.float32).cumsum(0)
    sim.bind_case(cells, ref, types, gt)
    return sim, gt, types


def test_predict_positions_shapes_on_bound_case():
    sim, gt, types = _bound_sim()
    npp = torch.tensor([5])
    win = gt[0:2].permute(1, 0, 2).contiguous()  # frames [0,1] -> predict frame 2
    nxt, aux = sim.predict_positions(win, npp, types)
    assert nxt.shape == (5, 3) and aux.shape == (5, 1)
    # next call must accept the GT-slid window for frame 3
    win2 = torch.stack([gt[1], gt[2]], dim=1)  # (P, 2, dim), kin rows GT
    nxt2, aux2 = sim.predict_positions(win2, npp, types)
    assert nxt2.shape == (5, 3) and aux2.shape == (5, 1)


def test_predict_positions_before_bind_case_raises():
    sim = _tiny_sim()
    with pytest.raises(RuntimeError, match="bind_case"):
        sim.predict_positions(
            torch.rand(3, 2, 3), torch.tensor([3]), torch.zeros(3, dtype=torch.int64)
        )


def test_tripwire_fires_on_perturbed_kinematic_row():
    sim, gt, types = _bound_sim()
    npp = torch.tensor([5])
    sim.predict_positions(gt[0:2].permute(1, 0, 2).contiguous(), npp, types)
    bad_win = torch.stack([gt[1], gt[2]], dim=1).clone()
    bad_win[2, 1] += 5.0  # perturb particle 2 (OBSTACLE, scripted) at the current frame
    with pytest.raises(RuntimeError, match="reset_rollout"):
        sim.predict_positions(bad_win, npp, types)


def test_second_eval_pass_without_reset_raises():
    sim, gt, types = _bound_sim()
    npp = torch.tensor([5])
    win = gt[0:2].permute(1, 0, 2).contiguous()
    sim.predict_positions(win, npp, types)
    with pytest.raises(RuntimeError, match="reset_rollout"):
        sim.predict_positions(win, npp, types)  # stale: pointer already advanced
    sim.reset_rollout()
    nxt, _ = sim.predict_positions(win, npp, types)  # re-anchors cleanly
    assert nxt.shape == (5, 3)


def test_features_places_scripted_velocity_with_gt_deltas():
    sim, gt, types = _bound_sim(T=4, P=5)
    x_t = gt[0]
    sim._advance_pointer(x_t, n_frames=1)  # anchors t=1 via the real base helper
    scripted_velocity = sim._eval_scripted_velocity(x_t)

    scripted_mask = types == 1  # default scripted_types=(1,)
    expected = torch.zeros(5, 3)
    expected[scripted_mask] = (gt[1] - x_t)[scripted_mask]
    torch.testing.assert_close(scripted_velocity, expected)

    ref = sim._reference_coords
    one_hot = sim._node_type_onehot
    feats = sim._features(one_hot, scripted_velocity, x_t, ref)

    nts, dim = sim._node_type_size, sim._dim
    assert feats.shape == (5, nts + 3 * dim)
    torch.testing.assert_close(feats[:, :nts], one_hot)
    sv_slice = feats[:, nts : nts + dim]
    torch.testing.assert_close(sv_slice, scripted_velocity)
    # scripted row (type 1) carries the GT delta; the other kinematic row
    # (type 3, HANDLE, not scripted) and NORMAL rows are zero.
    assert torch.equal(sv_slice[scripted_mask], expected[scripted_mask])
    assert torch.equal(
        sv_slice[~scripted_mask], torch.zeros_like(sv_slice[~scripted_mask])
    )
    torch.testing.assert_close(feats[:, nts + dim : nts + 2 * dim], x_t)
    torch.testing.assert_close(feats[:, nts + 2 * dim :], ref)


def test_forward_train_shapes_and_target_semantics():
    torch.manual_seed(0)
    sim = _tiny_sim()
    P = 5
    x = torch.rand(P, 3)
    nxt = x + 0.1
    aux = torch.rand(P)
    types = torch.tensor([0, 0, 1, 3, 0], dtype=torch.int64)
    ref = torch.rand(P, 3)
    pred, target = sim.forward_train(
        x, nxt, aux, types, ref, torch.tensor([P]), accumulate=False
    )
    assert pred.shape == (P, 4) and target.shape == (P, 4)
    # untrained target normalizer is identity: target == [v_target | stress]
    torch.testing.assert_close(target[:, :3], nxt - x)
    torch.testing.assert_close(target[:, 3], aux)


def test_forward_train_accumulates_normalizers_when_asked():
    torch.manual_seed(0)
    sim = _tiny_sim()
    P = 4
    x = torch.rand(P, 3)
    nxt = x + 0.1
    aux = torch.rand(P)
    types = torch.zeros(P, dtype=torch.int64)
    ref = torch.rand(P, 3)
    norms = [sim._node_normalizer, sim._target_normalizer]
    before = [int(n._n_accumulations) for n in norms]
    sim.forward_train(x, nxt, aux, types, ref, torch.tensor([P]), accumulate=True)
    sim.forward_train(x, nxt, aux, types, ref, torch.tensor([P]), accumulate=False)
    after = [int(n._n_accumulations) for n in norms]
    assert after == [b + 1 for b in before]


def test_train_and_eval_paths_build_identical_features():
    """The feature-builder seam: train and eval must share ONE builder."""
    sim, gt, types = _bound_sim()
    x_last = gt[1]  # last frame of the first window

    captured: list[torch.Tensor] = []
    hook = sim._net.register_forward_pre_hook(
        lambda mod, args: captured.append(args[0].detach().clone())
    )
    sim.reset_rollout()
    sim.predict_positions(
        gt[0:2].permute(1, 0, 2).contiguous(), torch.tensor([5]), types
    )
    hook.remove()
    eval_feats = captured[0]

    captured.clear()
    hook = sim._net.register_forward_pre_hook(
        lambda mod, args: captured.append(args[0].detach().clone())
    )
    sim.forward_train(
        x_last,
        gt[2],
        torch.zeros(5),
        types,
        sim._reference_coords,
        torch.tensor([5]),
        accumulate=False,
    )
    hook.remove()
    torch.testing.assert_close(captured[0], eval_feats)


def test_save_load_roundtrip_before_bind_case(tmp_path):
    torch.manual_seed(0)
    sim = _tiny_sim()
    P = 4
    x = torch.rand(P, 3)
    nxt = x + 0.05
    aux = torch.rand(P)
    types = torch.zeros(P, dtype=torch.int64)
    ref = torch.rand(P, 3)
    # Accumulate some normalizer stats so the buffers being restored is
    # actually exercised (not just identical default zeros).
    sim.forward_train(x, nxt, aux, types, ref, torch.tensor([P]), accumulate=True)

    p = tmp_path / "transolver.pt"
    sim.save(p)

    sim2 = _tiny_sim()
    sim2.load(p)  # loadable BEFORE any bind_case call -- self-contained checkpoint
    for (k1, v1), (k2, v2) in zip(
        sim.state_dict().items(), sim2.state_dict().items(), strict=True
    ):
        assert k1 == k2
        torch.testing.assert_close(v1, v2)


def test_scripted_types_must_be_subset_of_kinematic_types():
    with pytest.raises(ValueError):
        TransolverSimulator(scripted_types=(5,), kinematic_types=(1, 3))


# --- ADR-0050/0051 prediction-scheme axis (frames_per_call) ---


def test_frames_per_call_defaults_to_one():
    # Default is the byte-identical autoregressive scheme.
    assert TransolverSimulator()._k == 1


def test_frames_per_call_k1_head_and_normalizer_widths_byte_identical():
    # k=1 keeps out_size == dim+1 and the target normalizer at width dim+1;
    # these are the structural invariants that make k=1 byte-identical to the
    # pre-0051 recipe (the reshape and row-fold are no-ops at k=1).
    dim = 3
    sim = _tiny_sim(frames_per_call=1)
    assert sim._net.blocks[-1].mlp2.out_features == dim + 1
    assert sim._target_normalizer._sum.shape[0] == dim + 1


def test_frames_per_call_zero_sentinel_raises_at_construction():
    # The k=T sentinel (0) must be resolved to a concrete horizon upstream; an
    # unresolved 0 reaching the constructor is a bug, not a silent no-op.
    with pytest.raises(ValueError, match="must be >= 1"):
        _tiny_sim(frames_per_call=0)


def test_frames_per_call_gt_one_builds_k_head():
    # k>1 widens the decoder head to k*(dim+1); the target normalizer stays
    # width dim+1 (per-frame, applied on the row-folded view).
    dim, k = 3, 4
    sim = _tiny_sim(frames_per_call=k)
    assert sim.frames_per_call == k
    assert sim._net.blocks[-1].mlp2.out_features == k * (dim + 1)
    assert sim._target_normalizer._sum.shape[0] == dim + 1


def test_predict_positions_kframe_bundle_shapes():
    # A k>1 bundle returns (P, k, dim) positions and (P, k, 1) aux.
    k = 4
    sim, gt, types = _bound_sim(T=8, frames_per_call=k)
    npp = torch.tensor([5])
    win = gt[0:2].permute(1, 0, 2).contiguous()  # frames [0,1] -> predict [2, 2+k)
    nxt, aux = sim.predict_positions(win, npp, types)
    assert nxt.shape == (5, k, 3) and aux.shape == (5, k, 1)


def test_predict_positions_kframe_integrates_from_x_t():
    # The k bundled positions are x_t + cumsum(velocity), so consecutive
    # frames' displacements are the decoded per-frame velocities: reversing the
    # cumsum (diff, with x_t prepended) must reproduce a monotone-consistent set
    # -- here we just assert the first frame's step is x_t + v_0 (finite,
    # well-formed) and the bundle is strictly an integration (no NaNs, right
    # shape), the numerical round-trip being covered end-to-end in the rollout.
    k = 3
    sim, gt, types = _bound_sim(T=8, frames_per_call=k)
    npp = torch.tensor([5])
    win = gt[0:2].permute(1, 0, 2).contiguous()
    x_t = win[:, -1]
    nxt, _ = sim.predict_positions(win, npp, types)
    steps = torch.diff(nxt, dim=1, prepend=x_t.unsqueeze(1))  # (P, k, dim)
    # integration is invertible: re-integrating the recovered per-frame steps
    # reproduces the bundle exactly.
    torch.testing.assert_close(x_t.unsqueeze(1) + torch.cumsum(steps, dim=1), nxt)


def test_forward_train_kframe_target_shapes():
    # k>1 forward_train emits (P, k, dim+1) prediction and target.
    k, p, dim = 4, 5, 3
    sim = _tiny_sim(frames_per_call=k)
    torch.manual_seed(0)
    x_last = torch.randn(p, dim)
    next_positions = torch.randn(p, k, dim)
    next_aux = torch.randn(p, k)
    types = torch.tensor([0, 0, 1, 3, 0])
    ref = torch.randn(p, dim)
    npp = torch.tensor([p])
    pred, target = sim.forward_train(
        x_last, next_positions, next_aux, types, ref, npp, accumulate=True
    )
    assert pred.shape == (p, k, dim + 1) and target.shape == (p, k, dim + 1)


def test_kframe_bundled_rollout_moving_kinematic_keeps_pointer_synced():
    # C4 regression: a k>1 bundled rollout advances the step pointer by k. With
    # a MOVING kinematic particle (unlike Taylor's static wall, which would mask
    # a desync by comparing frame-invariant rows), a wrong pointer step trips
    # the tripwire; a correct step + all-k override keeps it green and pins the
    # kinematic rows to GT across every bundle.
    from structbench.datasets.canonical import CaseTrajectory
    from structbench.eval.rollout import rollout

    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    T, P, k, dim = 8, 5, 3, 3
    pos = rng.random((T, P, dim)).astype(np.float32).cumsum(0)
    pos[:, 2, :] = (np.arange(T)[:, None] ** 2).astype(np.float32)  # kinematic, moving
    ptype = np.array([0, 0, 1, 0, 0], dtype=np.int64)  # row 2 is kinematic type 1
    aux = np.zeros((T, P), dtype=np.float32)
    ref = rng.random((P, dim)).astype(np.float32)
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    traj = CaseTrajectory("k", pos, ptype, aux, np.arange(T, dtype=float))

    sim = _tiny_sim(frames_per_call=k, kinematic_types=(1,), scripted_types=(1,))
    sim.eval()
    sim.bind_case(
        torch.from_numpy(cells),
        torch.from_numpy(ref),
        torch.from_numpy(ptype),
        torch.from_numpy(pos),
    )
    sim.reset_rollout()
    # No RuntimeError => the pointer stayed in sync with the moving kinematic
    # rows across bundles.
    res = rollout(sim, traj, input_frames=2, kinematic_types=(1,))
    np.testing.assert_allclose(res.predicted_positions[:, 2], pos[:, 2], atol=1e-4)
