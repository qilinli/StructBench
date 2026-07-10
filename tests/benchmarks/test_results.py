"""BaselineResult validation and spec wiring (ADR-0033)."""

from dataclasses import replace

import pytest

from structbench.benchmarks import available_benchmarks, get_benchmark
from structbench.benchmarks.results import BaselineResult


def _result(**overrides):
    kwargs = dict(
        family="cgn",
        label="CGN baseline",
        run_commit="abc1234",
        run_date="2026-07-05",
        metrics={"test_interp": {"rollout_pos_rmse_mm": 1.5}},
    )
    kwargs.update(overrides)
    return BaselineResult(**kwargs)


def test_valid_result_constructs_with_defaults():
    result = _result()
    assert result.checkpoint is None
    assert result.notes == ""
    assert result.metrics["test_interp"]["rollout_pos_rmse_mm"] == 1.5


@pytest.mark.parametrize("name", ["family", "label", "run_commit", "run_date"])
def test_blank_required_field_raises(name):
    with pytest.raises(ValueError, match=name):
        _result(**{name: "  "})


def test_empty_metrics_raise():
    with pytest.raises(ValueError, match="metrics"):
        _result(metrics={})


def test_empty_split_metrics_raise():
    with pytest.raises(ValueError, match="test_interp"):
        _result(metrics={"test_interp": {}})


def test_metrics_are_read_only():
    result = _result()
    with pytest.raises(TypeError):
        result.metrics["test_interp"]["rollout_pos_rmse_mm"] = 0.0


def test_taylor_and_wave_are_the_blessed_benchmarks():
    blessed = {n for n in available_benchmarks() if get_benchmark(n).results}
    assert blessed == {"taylor_impact_2d", "wave_propagation_1d"}


def test_taylor_baseline_is_the_cgn_reference_run():
    (result,) = get_benchmark("taylor_impact_2d").results
    assert result.family == "cgn"
    assert result.run_commit == "7be9d4b"
    # val selects the checkpoint, so only the held-out splits are numbers to beat.
    assert set(result.metrics) == {"test_interp", "test_extrap"}


def test_wave_baseline_is_the_cgn_reference_run():
    (result,) = get_benchmark("wave_propagation_1d").results
    assert result.family == "cgn"
    assert result.run_commit == "48046ea"
    # val selects the checkpoint; wave's only held-out split is test_interp.
    assert set(result.metrics) == {"test_interp"}


def test_spec_rejects_result_with_unknown_split():
    spec = get_benchmark("taylor_impact_2d")
    bad = _result(metrics={"nonexistent_split": {"rollout_pos_rmse_mm": 1.0}})
    with pytest.raises(ValueError, match="nonexistent_split"):
        replace(spec, results=(bad,))


def test_spec_accepts_result_with_known_splits():
    spec = get_benchmark("taylor_impact_2d")
    good = _result(
        metrics={
            "test_interp": {"rollout_pos_rmse_mm": 1.5},
            "test_extrap": {"rollout_pos_rmse_mm": 2.1},
        }
    )
    wired = replace(spec, results=(good,))
    assert wired.results[0].label == "CGN baseline"
