"""Case parameters and coverage on the registered benchmarks (ADR-0074
clause 2; the open-evaluation plan §4, task 2)."""

import pytest

from structbench.benchmarks import available_benchmarks, get_benchmark
from structbench.benchmarks.notch_beam_2d_impact.benchmark import (
    IMPACTOR_SHAPES,
)
from structbench.benchmarks.notch_beam_2d_impact.benchmark import (
    case_parameters as notch_parameters,
)
from structbench.benchmarks.taylor_impact_2d.benchmark import (
    case_parameters as taylor_parameters,
)
from structbench.benchmarks.wave_propagation_1d.benchmark import (
    case_parameters as wave_parameters,
)
from structbench.eval.coverage import coverage_bin


def test_parsers_key_the_cards_parameter_space():
    for name, parse in (
        ("taylor_impact_2d", taylor_parameters),
        ("wave_propagation_1d", wave_parameters),
        ("notch_beam_2d_impact", notch_parameters),
    ):
        spec = get_benchmark(name)
        box = spec.card.parameter_space.box
        for case_id in spec.splits["train"][:3]:
            params = parse(case_id)
            assert set(box) <= set(params), (name, case_id)
        assert spec.case_parameters is parse


def test_taylor_and_wave_parsers_read_their_ids():
    assert taylor_parameters("T-20-80-150") == {
        "bar_width_mm": 20.0,
        "bar_length_mm": 80.0,
        "impact_velocity_m_s": 150.0,
    }
    assert wave_parameters("W1D-300-4") == {
        "bar_length_mm": 300.0,
        "initial_velocity_m_s": 4.0,
    }
    with pytest.raises(ValueError):
        taylor_parameters("W1D-300-4")
    with pytest.raises(ValueError):
        wave_parameters("T-20-80-150")


def test_notch_parser_reads_grid_and_probe_ids():
    grid = notch_parameters("NB-I-480-Sphere-b-120")
    assert grid["beam_width_mm"] == 480.0 and grid["beam_height_mm"] == 80.0
    assert grid["impact_velocity_m_s"] == 120.0 and grid["impact_offset_mm"] == 0.0
    assert grid["impactor_shape"] == IMPACTOR_SHAPES["Sphere"] == "disk"
    assert grid["notch_position"] == "b"
    probe = notch_parameters("S_100_800_V60_extrapolation")
    assert probe["beam_height_mm"] == 100.0 and probe["beam_width_mm"] == 800.0
    assert probe["impact_velocity_m_s"] == 60.0
    assert probe["impact_offset_mm"] == pytest.approx(48.0)  # 6 % of the width
    assert probe["impactor_shape"] == "disk" and probe["notch_position"] == ""
    with pytest.raises(ValueError):
        notch_parameters("C-1")


def _bins(name, split):
    spec = get_benchmark(name)
    cov = spec.coverage()
    return sorted({coverage_bin(cov[c]) for c in spec.splits[split]})


def test_taylor_scored_sets_read_as_the_plan_says():
    assert _bins("taylor_impact_2d", "test_interp") == ["within, ≤ 1"]
    assert _bins("taylor_impact_2d", "test_extrap") == ["beyond, ≤ 1"]


def test_notch_probes_are_novel_and_its_holdouts_within():
    assert _bins("notch_beam_2d_impact", "probe") == ["novel"]
    assert _bins("notch_beam_2d_impact", "test_interp") == ["within, ≤ 1"]
    spec = get_benchmark("notch_beam_2d_impact")
    assert set(spec.novel_cases) == set(spec.splits["probe"])


def test_wave_holdouts_are_within():
    assert all(
        b.startswith("within") for b in _bins("wave_propagation_1d", "test_interp")
    )


def test_deforming_plate_has_no_coverage():
    spec = get_benchmark("deforming_plate")
    assert spec.case_parameters is None
    with pytest.raises(ValueError, match="coverage cannot be computed"):
        spec.coverage()


def test_scored_splits_are_everything_but_train_and_val():
    for name in available_benchmarks():
        spec = get_benchmark(name)
        assert "train" not in spec.scored_splits and "val" not in spec.scored_splits
        assert set(spec.scored_splits) | {"train", "val"} == set(spec.splits)
