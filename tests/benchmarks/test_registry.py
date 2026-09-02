"""Benchmark registry resolution and spec invariants."""

from dataclasses import replace

import pytest

from structbench.benchmarks import (
    BenchmarkSpec,
    available_benchmarks,
    get_benchmark,
)
from structbench.benchmarks.results import BaselineResult


def _result(**overrides):
    kwargs = dict(
        family="cgn",
        label="baseline",
        run_commit="abc1234",
        run_date="2026-07-05",
        metrics={"test_interp": {"rollout_pos_rmse_mm": 1.5}},
    )
    kwargs.update(overrides)
    return BaselineResult(**kwargs)


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
        replace(
            spec,
            card=replace(
                spec.card, n_cases=spec.card.n_cases + 1, splits=bad_card_splits
            ),
        )


def test_spec_split_mappings_are_read_only():
    spec = get_benchmark("taylor_impact_2d")
    with pytest.raises(TypeError):
        spec.splits["val"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.qois["extra"] = len  # type: ignore[index]


def test_spec_kinematic_types_default_empty():
    spec = get_benchmark("wave_propagation_1d")
    assert spec.kinematic_types == ()


def test_taylor_pins_wall_kinematic_type():
    # ADR-0047: synthesized wall nodes are type 2, kinematic.
    spec = get_benchmark("taylor_impact_2d")
    assert spec.kinematic_types == (2,)
    assert spec.mesh_transform is not None


def test_notch_impact_pins_scored_frames():
    assert get_benchmark("notch_beam_2d_impact").scored_frames == 250


def test_spec_validates_scored_frames_bounds():
    spec = get_benchmark("notch_beam_2d_impact")
    with pytest.raises(ValueError, match="scored_frames"):
        replace(spec, scored_frames=spec.card.input_frames)  # not > input_frames
    with pytest.raises(ValueError, match="scored_frames"):
        replace(spec, scored_frames=spec.card.n_frames + 1)  # beyond trajectory
    assert replace(spec, scored_frames=None).scored_frames is None


def test_blessed_results_filters_provisional_and_preserves_order():
    spec = get_benchmark("taylor_impact_2d")
    provisional_a = _result(family="transolver", provisional=True)
    blessed = _result(family="cgn", provisional=False)
    provisional_b = _result(family="geoflare", provisional=True)
    wired = replace(spec, results=(provisional_a, blessed, provisional_b))
    assert wired.blessed_results == (blessed,)


def test_blessed_results_empty_when_all_provisional():
    spec = get_benchmark("taylor_impact_2d")
    wired = replace(spec, results=(_result(provisional=True),))
    assert wired.blessed_results == ()


def test_spec_rejects_duplicate_family_in_results():
    spec = get_benchmark("taylor_impact_2d")
    dup_a = _result(family="cgn", metrics={"test_interp": {"m": 1.0}})
    dup_b = _result(family="cgn", metrics={"test_extrap": {"m": 2.0}})
    with pytest.raises(ValueError) as exc_info:
        replace(spec, results=(dup_a, dup_b))
    message = str(exc_info.value)
    assert "cgn" in message
    assert spec.card.name in message


def test_spec_accepts_distinct_families_in_results():
    spec = get_benchmark("taylor_impact_2d")
    a = _result(family="cgn", metrics={"test_interp": {"m": 1.0}})
    b = _result(family="mgn", metrics={"test_extrap": {"m": 2.0}})
    wired = replace(spec, results=(a, b))
    assert {r.family for r in wired.results} == {"cgn", "mgn"}


def test_quickstart_family_defaults_to_cgn():
    spec = get_benchmark("taylor_impact_2d")
    assert spec.quickstart_family == "cgn"


def test_quickstart_family_blank_raises():
    spec = get_benchmark("taylor_impact_2d")
    with pytest.raises(ValueError, match="quickstart_family"):
        replace(spec, quickstart_family="  ")


def test_spec_card_aux_declaration_consistency_checked():
    """ADR-0059: card.aux_field and spec.aux_field must agree."""
    from dataclasses import replace

    spec = get_benchmark("taylor_impact_2d")
    with pytest.raises(ValueError, match="card.aux_field"):
        replace(spec, aux_field="axial_stress")
