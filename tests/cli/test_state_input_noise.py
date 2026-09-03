"""ADR-0061 knob 1: the batch-std-relative state-input noise helper."""

import torch

from structbench.cli.train import _state_input_noise


def _free(n: int) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.bool)


def test_zero_or_none_knob_is_identity():
    x = torch.randn(64, 3)
    assert _state_input_noise(x, None, _free(64)) is x
    assert _state_input_noise(x, (0.0, 0.0, 0.0), _free(64)) is x


def test_noise_scales_per_channel_with_batch_std():
    torch.manual_seed(0)
    x = torch.randn(20_000, 2) * torch.tensor([1.0, 100.0])
    y = _state_input_noise(x, (0.5, 0.0), _free(20_000))
    d = y - x
    # channel 1 (knob 0) untouched; channel 0 perturbed at ~0.5x its std
    assert torch.all(d[:, 1] == 0)
    ratio = d[:, 0].std() / x[:, 0].std()
    assert 0.4 < float(ratio) < 0.6


def test_kinematic_rows_stay_clean():
    torch.manual_seed(0)
    x = torch.randn(64, 3)
    is_kin = torch.zeros(64, dtype=torch.bool)
    is_kin[::4] = True
    y = _state_input_noise(x, (0.5, 0.5, 0.5), is_kin)
    d = y - x
    # every kinematic row untouched; every free row perturbed (all-channel knob)
    assert torch.all(d[is_kin] == 0)
    assert torch.all(d[~is_kin].abs().sum(dim=-1) > 0)


def test_deterministic_under_seed():
    x = torch.randn(32, 2)
    torch.manual_seed(7)
    a = _state_input_noise(x, (0.1, 0.1), _free(32))
    torch.manual_seed(7)
    b = _state_input_noise(x, (0.1, 0.1), _free(32))
    torch.testing.assert_close(a, b)
