"""ADR-0051 B: benchmark loading_scalar (impact-velocity) extractors."""

import pytest

from structbench.benchmarks import get_benchmark


def test_taylor_loading_scalar_parses_every_case():
    spec = get_benchmark("taylor_impact_2d")
    assert spec.loading_scalar is not None
    # spot checks + every real case id parses to a finite positive velocity
    assert spec.loading_scalar("T-20-100-150") == 150.0
    assert spec.loading_scalar("T-20-60-200") == 200.0
    for split in ("train", "val", "test_interp", "test_extrap"):
        for cid in spec.splits[split]:
            v = spec.loading_scalar(cid)
            assert v > 0 and v == v  # finite, positive


def test_notch_loading_scalar_handles_both_id_formats():
    spec = get_benchmark("notch_beam_2d_impact")
    assert spec.loading_scalar is not None
    # main grid: velocity is the last '-' field
    assert spec.loading_scalar("NB-I-320-Bullet-a-120") == 120.0
    # off-grid probe: velocity is the V token
    assert spec.loading_scalar("S_100_800_V60_extrapolation") == 60.0
    assert spec.loading_scalar("S_80_400_V140_intrapolation") == 140.0
    for split in ("train", "val", "test_interp", "probe"):
        for cid in spec.splits[split]:
            assert spec.loading_scalar(cid) > 0


def test_notch_loading_scalar_rejects_unparseable_id():
    spec = get_benchmark("notch_beam_2d_impact")
    with pytest.raises(ValueError, match="cannot parse impact velocity"):
        spec.loading_scalar("S_100_800_noV_here")


def test_wave_loading_scalar_parses_every_case():
    spec = get_benchmark("wave_propagation_1d")
    assert spec.loading_scalar is not None
    # spot checks: initial axial velocity (mm/ms) is the last '-' field
    assert spec.loading_scalar("W1D-300-4") == 4.0
    assert spec.loading_scalar("W1D-500-8") == 8.0
    for split in ("train", "val", "test_interp"):
        for cid in spec.splits[split]:
            v = spec.loading_scalar(cid)
            assert v in (1.0, 2.0, 4.0, 8.0)  # the ADR-0025 sweep values


def test_deforming_plate_has_no_loading_scalar():
    # dp is actuator-driven, not impact-velocity: no scalar (a run requesting
    # impact_velocity_feature is rejected at train time).
    assert get_benchmark("deforming_plate").loading_scalar is None
