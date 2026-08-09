import numpy as np
import pytest
import torch

from structbench.models.geoflare import GeoFlareSimulator

# Relaxed radii/neighbors (standardized-coordinate units) so tiny synthetic
# meshes (P=4/5 points) reliably find at least one neighbour within the ball
# query -- the ADR-0045 defaults (0.05, 0.25) are tuned for ~1.3k-node
# DeformingPlate meshes and would zero-pad almost everywhere at this scale.
# The exact neighbour count is immaterial to these tests (shape/contract
# only); it just needs to not degenerate to a fully-zero-padded query.
_TINY_GEOMETRY = dict(radii=(0.5, 2.0), neighbors=(2, 3))


def _tiny_sim(**kwargs):
    return GeoFlareSimulator(
        dim=3,
        n_hidden=16,
        n_layers=2,
        n_heads=2,
        slice_num=4,
        n_hidden_local=4,
        **_TINY_GEOMETRY,
        **kwargs,
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

    p = tmp_path / "geoflare.pt"
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
        GeoFlareSimulator(scripted_types=(5,), kinematic_types=(1, 3))


def test_net_receives_raw_x_t_as_coords_not_a_feats_slice_or_standardized():
    """The GeoFLARE-specific coords pin: the net's coords arg is raw x_t.

    ``GeoFlareNet.forward`` takes coords as a SEPARATE, third positional
    argument (unlike ``TransolverNet``, which has no such argument) and
    threads them, unmodified, to its internal ``MultiScaleContext`` for
    per-example standardization. Two plausible bugs would silently corrupt
    this: (1) passing the feats tensor's OWN ``x_t`` slice (post node-
    normalization) instead of raw ``x_t``, or (2) pre-standardizing ``x_t``
    in the simulator before handing it to the net (double-standardizing,
    since ``MultiScaleContext`` already standardizes internally).

    ``OnlineNormalizer`` is the identity before any ``accumulate=True``
    call, so on a FRESH simulator ``feats[:, 12:15] == raw x_t`` exactly
    and mutation (1) would be invisible. Warm the node normalizer to
    non-identity running stats first so the feats slice and raw x_t
    provably diverge, making both mutations detectable.
    """
    sim, gt, types = _bound_sim()
    npp = torch.tensor([5])

    # Warm the node normalizer to non-identity stats with a few distinct
    # accumulate=True training calls.
    for i in range(3):
        x_i, nxt_i = gt[i], gt[i + 1]
        aux_i = torch.rand(5)
        sim.forward_train(
            x_i, nxt_i, aux_i, types, sim._reference_coords, npp, accumulate=True
        )
    sim.reset_rollout()

    captured_args: list[tuple[torch.Tensor, torch.Tensor]] = []
    hook = sim._net.register_forward_pre_hook(
        lambda mod, args: captured_args.append(
            (args[0].detach().clone(), args[1].detach().clone())
        )
    )
    win = gt[0:2].permute(1, 0, 2).contiguous()
    x_t = gt[1]  # the raw current-frame positions predict_positions will see
    sim.predict_positions(win, npp, types)
    hook.remove()

    feats_arg, coords_arg = captured_args[0]

    # Sanity: the normalizer has actually moved away from the identity, so
    # the feats slice at [12:15] (node_type_size + dim : node_type_size +
    # 2*dim, i.e. the x_t channel) now differs from raw x_t -- otherwise
    # mutation (1) below would be silently invisible.
    nts, dim = sim._node_type_size, sim._dim
    feats_x_t_slice = feats_arg[:, nts + dim : nts + 2 * dim]
    assert nts == 9 and dim == 3  # pins the brief's [12:15] slice indices
    assert not torch.allclose(feats_x_t_slice, x_t), (
        "normalizer warm-up did not move feats' x_t slice away from raw "
        "x_t -- the mutation-detection setup below would be vacuous"
    )

    # The real pin: the net's coords argument is raw x_t, matching neither
    # the (normalized, divergent) feats slice nor a standardized transform.
    torch.testing.assert_close(coords_arg, x_t)
    assert not torch.allclose(coords_arg, feats_x_t_slice)
