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


def test_provisional_defaults_to_false():
    assert _result().provisional is False


def test_provisional_flag_round_trips():
    assert _result(provisional=True).provisional is True


@pytest.mark.parametrize("name", ["family", "label", "run_commit", "run_date"])
def test_blank_required_field_raises_when_provisional(name):
    # provisional=True must not bypass any existing validation.
    with pytest.raises(ValueError, match=name):
        _result(provisional=True, **{name: "  "})


def test_provisional_result_still_validates_empty_metrics():
    with pytest.raises(ValueError, match="metrics"):
        _result(provisional=True, metrics={})


def test_provisional_result_still_validates_checkpoint_sha256():
    with pytest.raises(ValueError, match="requires checkpoint"):
        _result(provisional=True, checkpoint_sha256="0" * 64)


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


def test_checkpoint_sha256_requires_checkpoint():
    with pytest.raises(ValueError, match="requires checkpoint"):
        _result(checkpoint_sha256="0" * 64)


@pytest.mark.parametrize("digest", ["", "abc", "G" * 64, "A" * 64, "0" * 63])
def test_malformed_checkpoint_sha256_raises(digest):
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        _result(checkpoint="models/x/cgn-abc1234/m.pt", checkpoint_sha256=digest)


def test_checkpoint_pointer_with_digest_constructs():
    result = _result(checkpoint="models/x/cgn-abc1234/m.pt", checkpoint_sha256="0" * 64)
    assert result.checkpoint_sha256 == "0" * 64


def test_taylor_wave_and_notch_impact_are_the_blessed_benchmarks():
    # blessed_results excludes provisional entries; today every RESULTS
    # tuple is entirely blessed, so this must name the same set as before.
    blessed = {n for n in available_benchmarks() if get_benchmark(n).blessed_results}
    assert blessed == {
        "taylor_impact_2d",
        "wave_propagation_1d",
        "notch_beam_2d_impact",
    }


def test_taylor_baseline_is_the_cgn_reference_run():
    # Taylor now carries provisional native baselines (Transolver-TC, MGN)
    # alongside the blessed CGN reference; pick out the blessed one.
    results = get_benchmark("taylor_impact_2d").results
    cgn = next(r for r in results if r.family == "cgn")
    assert not cgn.provisional
    assert cgn.run_commit == "7be9d4b"
    # val selects the checkpoint, so only the held-out splits are numbers to beat.
    assert set(cgn.metrics) == {"test_interp", "test_extrap"}


def test_wave_baseline_is_the_cgn_reference_run():
    (result,) = get_benchmark("wave_propagation_1d").results
    assert result.family == "cgn"
    assert result.run_commit == "48046ea"
    # val selects the checkpoint; wave's only held-out split is test_interp.
    assert set(result.metrics) == {"test_interp"}


def test_notch_impact_baseline_is_the_cgn_reference_run():
    # notch now carries provisional native baselines (a pending MGN placeholder
    # and Transolver) alongside the blessed CGN reference; pick out the blessed.
    results = get_benchmark("notch_beam_2d_impact").results
    cgn = next(r for r in results if r.family == "cgn")
    assert not cgn.provisional
    assert cgn.run_commit == "5956d81"
    # val selects the checkpoint; test_interp and probe are the held-out splits.
    assert set(cgn.metrics) == {"test_interp", "probe"}


def test_blessed_entries_point_at_the_models_archive():
    # ADR-0037: blessed entries carry an archive-relative pointer + digest.
    # Provisional native baselines (ADR-0044/0045) are local runs, not archived,
    # so they are exempt — only the blessed entries must point at the archive.
    for name in ("taylor_impact_2d", "wave_propagation_1d", "notch_beam_2d_impact"):
        blessed = [r for r in get_benchmark(name).results if not r.provisional]
        assert blessed, name
        for result in blessed:
            assert result.checkpoint is not None
            assert result.checkpoint.startswith(
                f"models/{name}/{result.family}-{result.run_commit}/"
            )
            assert result.checkpoint_sha256 is not None


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
