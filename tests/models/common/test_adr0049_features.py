"""ADR-0049 recipe repairs: velocity-history features and the MGN stretch gate.

Covers the three mesh-native families' shared velocity-history contract
(``history_velocities`` widens the node input; the feature is required when
enabled, rejected as absent, and the window must carry enough frames) and
MGN's mesh-edge stretch gate (edges past the threshold are dropped from the
mesh index and regain world-edge eligibility). Default-off construction is
asserted to keep the exact pre-ADR-0049 normalizer widths, so existing
checkpoints stay loadable.
"""

import numpy as np
import pytest
import torch

from structbench.cli.train import _mesh_family_noise
from structbench.models.geoflare import GeoFlareSimulator
from structbench.models.mgn import MeshSimulator
from structbench.models.transolver import TransolverSimulator


def _tiny_transolver(**kwargs):
    return TransolverSimulator(
        dim=3, hidden_dim=8, n_layers=1, n_heads=2, slice_num=2, **kwargs
    )


def _tiny_geoflare(**kwargs):
    return GeoFlareSimulator(
        dim=3,
        n_hidden=8,
        n_layers=1,
        n_heads=2,
        slice_num=2,
        n_hidden_local=4,
        **kwargs,
    )


def _tiny_mgn(**kwargs):
    return MeshSimulator(
        dim=3, latent=8, mp_steps=1, n_hidden=1, world_edge_radius=0.5, **kwargs
    )


_BUILDERS = [_tiny_transolver, _tiny_geoflare, _tiny_mgn]


def _bind(sim, T=6, P=5, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
    ref = torch.tensor(rng.random((P, 3)), dtype=torch.float32)
    types = torch.tensor([0, 0, 1, 3, 0], dtype=torch.int64)
    gt = torch.tensor(rng.random((T, P, 3)), dtype=torch.float32).cumsum(0)
    sim.bind_case(cells, ref, types, gt)
    return gt, types


@pytest.mark.parametrize("build", _BUILDERS)
def test_default_off_keeps_reference_node_width(build):
    plain = build()
    wide = build(history_velocities=2)
    plain_w = plain._node_normalizer._sum.shape[-1]
    wide_w = wide._node_normalizer._sum.shape[-1]
    assert wide_w == plain_w + 2 * 3
    # default-off state dict loads into a fresh default-off instance
    build().load_state_dict(plain.state_dict())


@pytest.mark.parametrize("build", _BUILDERS)
def test_velocity_history_rollout_and_window_checks(build):
    sim = build(history_velocities=2)
    gt, types = _bind(sim)
    npp = torch.tensor([5])
    win = gt[0:3].permute(1, 0, 2).contiguous()  # 3 frames -> 2 velocities
    nxt, aux = sim.predict_positions(win, npp, types)
    assert nxt.shape == (5, 3) and aux.shape == (5, 1)

    # a window too short for the requested history is rejected
    sim.reset_rollout()
    with pytest.raises(ValueError, match="window frames"):
        sim.predict_positions(gt[0:2].permute(1, 0, 2).contiguous(), npp, types)


@pytest.mark.parametrize("build", _BUILDERS)
def test_forward_train_requires_velocity_history_when_enabled(build):
    sim = build(history_velocities=2)
    rng = np.random.default_rng(1)
    P = 5
    x = torch.tensor(rng.random((P, 3)), dtype=torch.float32)
    nxt = x + 0.01
    aux = torch.zeros(P)
    types = torch.zeros(P, dtype=torch.int64)
    ref = torch.tensor(rng.random((P, 3)), dtype=torch.float32)
    npp = torch.tensor([P])
    vh = torch.tensor(rng.random((P, 6)), dtype=torch.float32)
    mesh_edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)

    def call(velocity_history):
        if isinstance(sim, MeshSimulator):
            return sim.forward_train(
                x,
                nxt,
                aux,
                types,
                mesh_edges,
                ref,
                npp,
                accumulate=False,
                velocity_history=velocity_history,
            )
        return sim.forward_train(
            x,
            nxt,
            aux,
            types,
            ref,
            npp,
            accumulate=False,
            velocity_history=velocity_history,
        )

    pred, target = call(vh)
    assert pred.shape == target.shape == (P, 4)
    with pytest.raises(ValueError, match="velocity_history"):
        call(None)


def test_mesh_family_noise_vh_target_is_clean_next_velocity():
    # GNS adjusted-next convention (ADR-0049): on the velocity-history path
    # the returned target satisfies next_target - x_noisy == next - x_clean
    # EXACTLY (the accumulated position noise cancels), so the model
    # de-noises the velocity, never the position offset.
    torch.manual_seed(3)
    P, F, dim = 7, 6, 2
    position_seq = torch.randn(P, F, dim).cumsum(1)
    next_position = position_seq[:, -1] + torch.randn(P, dim)
    is_kinematic = torch.tensor([False] * 5 + [True] * 2)

    x_noisy, vh, next_target = _mesh_family_noise(
        position_seq, next_position, is_kinematic, 0.02, F - 1
    )
    assert vh is not None and vh.shape == (P, (F - 1) * dim)
    torch.testing.assert_close(
        next_target - x_noisy, next_position - position_seq[:, -1]
    )
    # kinematic rows: no noise anywhere, target untouched
    torch.testing.assert_close(x_noisy[5:], position_seq[5:, -1])
    torch.testing.assert_close(next_target[5:], next_position[5:])
    # noise really was injected on free rows
    assert not torch.allclose(x_noisy[:5], position_seq[:5, -1])


def test_mesh_family_noise_intermediate_history_uses_last_k_velocities():
    # ADR-0053: history_frames = k with 0 < k < F-1 (decoupled from the
    # input_frames seed) builds exactly k velocities from the last k+1 frames,
    # with the random walk history-matched to that span; the GNS clean-velocity
    # target invariant still holds and kinematic rows stay clean.
    torch.manual_seed(5)
    P, F, dim = 6, 6, 2
    position_seq = torch.randn(P, F, dim).cumsum(1)
    next_position = position_seq[:, -1] + torch.randn(P, dim)
    is_kinematic = torch.tensor([False] * 4 + [True] * 2)

    k = 2  # strictly between Markovian (0) and the full window (F - 1 = 5)
    x_noisy, vh, next_target = _mesh_family_noise(
        position_seq, next_position, is_kinematic, 0.02, k
    )
    assert vh is not None and vh.shape == (P, k * dim)
    torch.testing.assert_close(
        next_target - x_noisy, next_position - position_seq[:, -1]
    )
    torch.testing.assert_close(x_noisy[4:], position_seq[4:, -1])
    torch.testing.assert_close(next_target[4:], next_position[4:])
    assert not torch.allclose(x_noisy[:4], position_seq[:4, -1])


def test_mesh_family_noise_reference_path_keeps_mgn_gamma_one():
    # single-frame path: next_position passes through UNCHANGED, so the
    # caller's target next - x_noisy measures from the noisy position
    # (MGN gamma = 1), and no velocity-history feature is built.
    torch.manual_seed(4)
    P, F, dim = 5, 6, 2
    position_seq = torch.randn(P, F, dim).cumsum(1)
    next_position = position_seq[:, -1] + torch.randn(P, dim)
    is_kinematic = torch.zeros(P, dtype=torch.bool)

    x_noisy, vh, next_target = _mesh_family_noise(
        position_seq, next_position, is_kinematic, 0.02, 0
    )
    assert vh is None
    assert next_target is next_position
    assert not torch.allclose(x_noisy, position_seq[:, -1])


def test_mgn_stretch_gate_drops_torn_edges_and_restores_world_edges():
    sim = _tiny_mgn(mesh_edge_max_stretch=2.0)
    P = 4
    # rest lattice: unit spacing along x
    ref = torch.tensor(
        [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]], dtype=torch.float32
    )
    # current: node 1 pulled 3 units from node 0 (stretch 3x > 2x -> torn),
    # nodes 2/3 keep rest spacing (stretch 1x -> kept)
    x = torch.tensor(
        [[0.0, 0, 0], [3.0, 0, 0], [3.2, 0, 0], [4.2, 0, 0]], dtype=torch.float32
    )
    mesh_edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)
    one_hot = torch.nn.functional.one_hot(
        torch.zeros(P, dtype=torch.int64), num_classes=9
    ).to(torch.float32)
    scripted = torch.zeros(P, 3)

    (_, gated_index, gated_feats, world_index, _) = sim._graph_features(
        x, one_hot, scripted, mesh_edges, ref, None, accumulate=False
    )
    kept = set(map(tuple, gated_index.t().tolist()))
    assert (0, 1) not in kept  # torn (3x)
    assert (1, 2) in kept and (2, 3) in kept
    assert gated_feats.shape[0] == gated_index.shape[1]
    world = set(map(tuple, world_index.t().tolist()))
    assert (1, 2) not in world  # kept mesh edge stays excluded from world

    # gate off: all three mesh edges survive
    plain = _tiny_mgn()
    (_, plain_index, _, _, _) = plain._graph_features(
        x, one_hot, scripted, mesh_edges, ref, None, accumulate=False
    )
    assert plain_index.shape[1] == 3


def test_mgn_stretch_gate_restores_world_edge_for_close_torn_pair():
    # a torn pair that is still inside the world radius: rest 1.0, current
    # 3.0 (stretch 3x > gate 2x) with world_edge_radius 5.0.
    sim = MeshSimulator(
        dim=3,
        latent=8,
        mp_steps=1,
        n_hidden=1,
        world_edge_radius=5.0,
        mesh_edge_max_stretch=2.0,
    )
    ref = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float32)
    x = torch.tensor([[0.0, 0, 0], [3.0, 0, 0]], dtype=torch.float32)
    mesh_edges = torch.tensor([[0], [1]], dtype=torch.int64)
    one_hot = torch.nn.functional.one_hot(
        torch.zeros(2, dtype=torch.int64), num_classes=9
    ).to(torch.float32)
    scripted = torch.zeros(2, 3)

    (_, gated_index, _, world_index, _) = sim._graph_features(
        x, one_hot, scripted, mesh_edges, ref, None, accumulate=False
    )
    assert gated_index.shape[1] == 0  # torn
    world = set(map(tuple, world_index.t().tolist()))
    assert (0, 1) in world and (1, 0) in world  # regained world-edge eligibility

    # without the gate the same pair is mesh-connected and world-excluded
    plain = MeshSimulator(
        dim=3, latent=8, mp_steps=1, n_hidden=1, world_edge_radius=5.0
    )
    (_, plain_index, _, plain_world, _) = plain._graph_features(
        x, one_hot, scripted, mesh_edges, ref, None, accumulate=False
    )
    assert plain_index.shape[1] == 1
    assert (0, 1) not in set(map(tuple, plain_world.t().tolist()))
