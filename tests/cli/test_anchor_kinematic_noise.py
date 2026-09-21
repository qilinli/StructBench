"""ADR-0063 knob 2: structured anchor-kinematic noise (unit-level)."""

import torch

from structbench.cli.train import _anchor_kinematic_noise

P, DIM = 5, 2


def _pair():
    g = torch.Generator().manual_seed(3)
    return torch.rand((P, 2, DIM), generator=g)


def test_zero_scales_are_identity():
    pair = _pair()
    kin = torch.zeros(P, dtype=torch.bool)
    assert _anchor_kinematic_noise(pair, 0.0, 0.0, kin) is pair


def test_common_mode_cancels_in_fd_velocity():
    torch.manual_seed(0)
    pair = _pair()
    kin = torch.zeros(P, dtype=torch.bool)
    noised = _anchor_kinematic_noise(pair, 0.5, 0.0, kin)
    # position corrupted...
    assert not torch.allclose(noised[:, 1], pair[:, 1])
    # ...but the FD velocity is untouched (same draw on both frames).
    torch.testing.assert_close(noised[:, 1] - noised[:, 0], pair[:, 1] - pair[:, 0])


def test_differential_hits_velocity_only():
    torch.manual_seed(0)
    pair = _pair()
    kin = torch.zeros(P, dtype=torch.bool)
    noised = _anchor_kinematic_noise(pair, 0.0, 0.05, kin)
    # anchor position (frame t0) untouched...
    torch.testing.assert_close(noised[:, 1], pair[:, 1])
    # ...FD velocity corrupted.
    assert not torch.allclose(noised[:, 1] - noised[:, 0], pair[:, 1] - pair[:, 0])


def test_kinematic_rows_stay_clean():
    torch.manual_seed(0)
    pair = _pair()
    kin = torch.tensor([True, False, False, False, True])
    noised = _anchor_kinematic_noise(pair, 0.5, 0.05, kin)
    torch.testing.assert_close(noised[kin], pair[kin])
    assert not torch.allclose(noised[~kin], pair[~kin])


def test_deterministic_under_seed():
    pair = _pair()
    kin = torch.zeros(P, dtype=torch.bool)
    torch.manual_seed(7)
    a = _anchor_kinematic_noise(pair, 0.3, 0.02, kin)
    torch.manual_seed(7)
    b = _anchor_kinematic_noise(pair, 0.3, 0.02, kin)
    torch.testing.assert_close(a, b)
