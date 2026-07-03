"""ADR-0025 wave-1d split: 12/2/2 interior holdout, no extrapolation."""

from structbench.benchmarks import get_benchmark
from structbench.benchmarks.wave_propagation_1d.benchmark import (
    TEST_INTERP,
    TRAIN,
    VAL,
)


def test_split_partitions_the_16_cases():
    all_ids = TRAIN + VAL + TEST_INTERP
    assert len(TRAIN) == 12 and len(VAL) == 2 and len(TEST_INTERP) == 2
    assert len(set(all_ids)) == 16


def test_split_cells_match_adr_0025():
    assert set(VAL) == {"W1D-300-2", "W1D-400-4"}
    assert set(TEST_INTERP) == {"W1D-300-4", "W1D-400-2"}
    assert all(
        c.startswith("W1D-200-")
        or c.startswith("W1D-500-")
        or c in {"W1D-300-1", "W1D-300-8", "W1D-400-1", "W1D-400-8"}
        for c in TRAIN
    )


def test_spec_resolves_with_no_extrapolation_split():
    spec = get_benchmark("wave_propagation_1d")
    assert spec.eval_splits == ("val", "test_interp")
    assert "test_extrap" not in spec.splits
    assert spec.aux_field == "axial_stress"
    assert spec.boundary_feature_fn is None
    assert set(spec.qois) == {
        "arrival_time_25",
        "arrival_time_50",
        "arrival_time_75",
        "peak_stress",
    }
