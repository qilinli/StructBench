"""Benchmark registry resolution and spec invariants."""

import pytest

from structbench.benchmarks import (
    BenchmarkSpec,
    available_benchmarks,
    get_benchmark,
)


def test_taylor_is_registered():
    assert "taylor_impact_2d" in available_benchmarks()


def test_get_benchmark_resolves_taylor_spec():
    spec = get_benchmark("taylor_impact_2d")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.card.name == "Taylor2D-Impact"
    assert spec.eval_splits == ("val", "test_interp", "test_extrap")
    assert len(spec.splits["train"]) == 21
    assert spec.aux_field == "von_mises_stress"
    assert spec.boundary_feature_fn is not None
    assert spec.dataset_id == "2D-Copper-Bar-Taylor-Impact"


def test_unknown_benchmark_raises_with_available_names():
    with pytest.raises(KeyError, match="taylor_impact_2d"):
        get_benchmark("no_such_benchmark")


def test_spec_validates_card_split_sizes():
    spec = get_benchmark("taylor_impact_2d")
    bad_card_splits = dict(spec.card.splits)
    bad_card_splits["train"] += 1
    from dataclasses import replace

    with pytest.raises(ValueError, match="split"):
        replace(spec, card=replace(spec.card, n_cases=spec.card.n_cases + 1,
                                   splits=bad_card_splits))
