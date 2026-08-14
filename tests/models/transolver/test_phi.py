"""Physics-conditioned slicing (SRO): phi field + per-block conditioning."""

import pytest
import torch

from structbench.models.transolver.network import TransolverNet
from structbench.models.transolver.phi import knn_neighbor_indices, strain_rate_phi
from structbench.models.transolver.simulator import TransolverSimulator


def test_knn_indices_shape_and_nearest_first() -> None:
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    idx = knn_neighbor_indices(coords, k=2)  # returns k+1 = 3
    assert idx.shape == (4, 3)
    # each point's own index is its nearest (distance 0), first column.
    assert torch.equal(idx[:, 0], torch.arange(4))


def test_strain_rate_phi_shape_and_standardized() -> None:
    torch.manual_seed(0)
    x = torch.randn(20, 2)
    v = torch.randn(20, 2)
    phi = strain_rate_phi(x, v, None, k=4, clamp=4.0)
    assert phi.shape == (20, 1)
    # per-example standardized: ~zero mean, ~unit std (up to the eps floor).
    assert phi.mean().abs() < 1e-4
    assert abs(phi.std().item() - 1.0) < 0.1


def test_strain_rate_phi_tracks_local_velocity_gradient() -> None:
    # A lattice where one half is quiescent and the other half shears hard:
    # phi should be higher on the shearing side.
    xs = torch.linspace(0, 9, 10).unsqueeze(1).repeat(1, 2)
    x = xs.clone()
    v = torch.zeros(10, 2)
    v[5:, 0] = torch.linspace(0, 5, 5)  # rising velocity => local gradient
    phi = strain_rate_phi(x, v, None, k=2, clamp=10.0)
    assert phi[5:].mean() > phi[:5].mean()


def test_strain_rate_phi_batched_equals_per_example() -> None:
    torch.manual_seed(1)
    xa, va = torch.randn(11, 2), torch.randn(11, 2)
    xb, vb = torch.randn(7, 2), torch.randn(7, 2)
    batched = strain_rate_phi(
        torch.cat([xa, xb]), torch.cat([va, vb]), torch.tensor([11, 7]), k=4, clamp=4.0
    )
    singles = torch.cat(
        [strain_rate_phi(xa, va, None, 4, 4.0), strain_rate_phi(xb, vb, None, 4, 4.0)]
    )
    assert torch.allclose(batched, singles, atol=1e-6)


def test_persistent_net_batched_matches_per_example_with_phi() -> None:
    # THE invariant under conditioning: per-example phi + per-block bias must
    # keep the ragged-batch forward equal to per-example forwards.
    torch.manual_seed(0)
    net = TransolverNet(
        node_in=7,
        out_size=4,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        slice_num=4,
        phi_conditioned=True,
        phi_lambda_init=0.3,
    )
    net.eval()
    a, b = torch.randn(11, 7), torch.randn(5, 7)
    pa, pb = torch.randn(11, 1), torch.randn(5, 1)
    with torch.no_grad():
        batched = net(torch.cat([a, b]), torch.tensor([11, 5]), torch.cat([pa, pb]))
        singles = torch.cat([net(a, None, pa), net(b, None, pb)])
    assert torch.allclose(batched, singles, atol=1e-5)


def test_lambda_zero_makes_phi_a_noop() -> None:
    # phi_lambda_init=0.0 (the default) => the phi bias is gated to zero, so
    # passing phi is identical to not passing it (step-0 == vanilla behaviour).
    torch.manual_seed(0)
    net = TransolverNet(
        node_in=7,
        out_size=4,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        slice_num=4,
        phi_conditioned=True,
        phi_lambda_init=0.0,
    )
    net.eval()
    x = torch.randn(9, 7)
    phi = torch.randn(9, 1)
    with torch.no_grad():
        with_phi = net(x, None, phi)
        without = net(x, None, None)
    assert torch.allclose(with_phi, without, atol=1e-7)


def test_vanilla_net_ignores_phi() -> None:
    # A net built without phi_conditioned has no phi_g; a phi passed in is a
    # silent no-op (phi_bias stays None).
    torch.manual_seed(0)
    net = TransolverNet(node_in=7, out_size=4, hidden_dim=16, n_layers=2, n_heads=2)
    net.eval()
    x, phi = torch.randn(6, 7), torch.randn(6, 1)
    with torch.no_grad():
        assert torch.allclose(net(x, None, phi), net(x, None, None), atol=1e-7)


def test_simulator_feature_mode_widens_node_in() -> None:
    base = TransolverSimulator(
        dim=2,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        history_velocities=5,
        phi_mode="off",
    )
    feat = TransolverSimulator(
        dim=2,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        history_velocities=5,
        phi_mode="feature",
    )
    base_in = base._net.preprocess[0].in_features
    assert feat._net.preprocess[0].in_features == base_in + 1


def test_simulator_persistent_builds_gate() -> None:
    sim = TransolverSimulator(
        dim=2,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        history_velocities=5,
        phi_mode="persistent",
    )
    assert sim._net.phi_g is not None
    assert all(blk.attn.lam is not None for blk in sim._net.blocks)
    off = TransolverSimulator(
        dim=2,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        history_velocities=5,
        phi_mode="off",
    )
    assert off._net.phi_g is None
    assert all(blk.attn.lam is None for blk in off._net.blocks)


def test_simulator_phi_requires_velocity_history() -> None:
    with pytest.raises(ValueError, match="velocity history"):
        TransolverSimulator(dim=2, phi_mode="persistent", history_velocities=0)


def test_simulator_rejects_unknown_phi_mode() -> None:
    with pytest.raises(ValueError, match="phi_mode"):
        TransolverSimulator(dim=2, phi_mode="bogus", history_velocities=5)


def test_robust_smooth_phi_batched_equals_per_example() -> None:
    # The Cause-1 robustness knobs (spatial smooth + median/IQR) must preserve
    # the per-example invariant.
    torch.manual_seed(2)
    xa, va = torch.randn(13, 2), torch.randn(13, 2)
    xb, vb = torch.randn(9, 2), torch.randn(9, 2)
    kw = dict(smooth=True, robust=True)
    batched = strain_rate_phi(
        torch.cat([xa, xb]), torch.cat([va, vb]), torch.tensor([13, 9]), 4, 4.0, **kw
    )
    singles = torch.cat(
        [strain_rate_phi(xa, va, None, 4, 4.0, **kw),
         strain_rate_phi(xb, vb, None, 4, 4.0, **kw)]
    )
    assert torch.allclose(batched, singles, atol=1e-6)
    assert torch.isfinite(batched).all()


def test_robust_sim_builds_and_vel_smooth() -> None:
    sim = TransolverSimulator(dim=2, hidden_dim=16, n_layers=2, n_heads=2,
                              history_velocities=5, phi_mode="persistent",
                              phi_smooth=True, phi_robust=True, phi_vel_smooth=True)
    assert sim._phi_smooth and sim._phi_robust and sim._phi_vel_smooth
    # phi computes with velocity-smoothing over the 5-velocity window
    vh = torch.randn(11, 5 * 2)
    phi = sim._compute_phi(torch.randn(11, 2), vh, None)
    assert phi.shape == (11, 1) and torch.isfinite(phi).all()
