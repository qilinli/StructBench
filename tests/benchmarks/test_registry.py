"""Benchmark registry resolution and spec invariants."""

from dataclasses import replace

import pytest

from structbench.benchmarks import (
    BenchmarkSpec,
    ConstitutiveLaw,
    available_benchmarks,
    get_benchmark,
)
from structbench.benchmarks.results import BaselineResult


@pytest.fixture
def two_site_spec():
    """Taylor's card re-declared as an element-site benchmark (plane-strain
    quads, the return-map layout), the smallest valid ADR-0075 declaration."""
    spec = get_benchmark("taylor_impact_2d")
    law = ConstitutiveLaw(124_000.0, 0.34, ((0.0, 10.0), (200.0, 10_200.0)))
    return replace(
        spec,
        state_site="element",
        state_channels=("stress_axisymmetric", "effective_plastic_strain"),
        element_formulation="plane_strain",
        constitutive_law=lambda cid: law,
        property_tolerances={"yield": 0.01, "peeq": 0.0},
    )


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


# --- ADR-0075 clause 6 (plan 3c): the properties the benchmark defines ---


def test_shipped_benchmarks_define_the_two_nodal_properties():
    for name in available_benchmarks():
        assert get_benchmark(name).defined_properties == (
            "prop_closure_ratio",
            "prop_error_growth",
        ), name


def test_property_tolerances_need_an_element_site():
    spec = get_benchmark("taylor_impact_2d")
    with pytest.raises(ValueError, match="property_tolerances"):
        replace(spec, property_tolerances={"yield": 0.01, "peeq": 0.0})


def test_element_site_requires_both_tolerances(two_site_spec):
    with pytest.raises(ValueError, match="property_tolerances"):
        replace(two_site_spec, property_tolerances={})
    with pytest.raises(ValueError, match="property_tolerances"):
        replace(two_site_spec, property_tolerances={"yield": 0.01})
    with pytest.raises(ValueError, match="non-negative"):
        replace(two_site_spec, property_tolerances={"yield": -0.01, "peeq": 0.0})
    with pytest.raises(ValueError, match="property_tolerances"):
        replace(
            two_site_spec, property_tolerances={"yield": 0.01, "peeq": 0.0, "x": 1.0}
        )


def test_element_site_defines_the_constraint_properties(two_site_spec):
    assert two_site_spec.defined_properties == (
        "prop_closure_ratio",
        "prop_error_growth",
        "prop_yield_violation_rate",
        "prop_peeq_decrease_rate",
    )
    assert two_site_spec.property_tolerances == {"yield": 0.01, "peeq": 0.0}


def test_matched_pairs_are_distinct_known_cases_of_one_split():
    spec = get_benchmark("taylor_impact_2d")
    a, b = spec.splits["test_interp"][:2]
    paired = replace(spec, matched_initial_state_pairs=((a, b),))
    assert paired.defined_properties[-1] == "prop_state_sufficiency"
    with pytest.raises(ValueError, match="matched_initial_state_pairs"):
        replace(spec, matched_initial_state_pairs=((a, a),))
    with pytest.raises(ValueError, match="matched_initial_state_pairs"):
        replace(spec, matched_initial_state_pairs=((a, "nope"),))
    with pytest.raises(ValueError, match="same split"):
        replace(spec, matched_initial_state_pairs=((a, spec.splits["train"][0]),))


# --- ADR-0075 clause 5 (plan 3b): the restart declaration ---


def test_shipped_benchmarks_declare_no_restarts():
    for name in available_benchmarks():
        spec = get_benchmark(name)
        assert spec.restart_anchors is None and not spec.defines_restarts, name
        assert spec.restart_horizons == ()
        assert spec.restart_compositions == (1, 2, 4)


def test_fixed_restart_anchors_are_validated_against_the_card():
    spec = get_benchmark("taylor_impact_2d")  # input_frames 6, n_frames per card
    declared = replace(spec, restart_anchors=(5, 20), restart_horizons=(10, 20))
    assert declared.defines_restarts
    assert declared.restart_anchors_for(spec.splits["test_interp"][0]) == (5, 20)
    with pytest.raises(ValueError, match="restart_anchors"):
        replace(spec, restart_anchors=(4,), restart_horizons=(10,))  # no history
    with pytest.raises(ValueError, match="restart_anchors"):
        replace(
            spec,
            restart_anchors=(spec.card.n_frames - 5,),
            restart_horizons=(10,),  # runs past the end
        )


def test_restart_horizons_and_compositions_are_validated():
    spec = get_benchmark("taylor_impact_2d")
    with pytest.raises(ValueError, match="restart_horizons"):
        replace(spec, restart_anchors=(5,))  # anchors without horizons
    with pytest.raises(ValueError, match="restart_horizons"):
        replace(spec, restart_horizons=(10,))  # horizons without anchors
    with pytest.raises(ValueError, match="restart_horizons"):
        replace(spec, restart_anchors=(5,), restart_horizons=(20, 10))
    with pytest.raises(ValueError, match="restart_compositions"):
        replace(
            spec,
            restart_anchors=(5,),
            restart_horizons=(10,),
            restart_compositions=(2, 4),
        )
    with pytest.raises(ValueError, match="restart_compositions"):
        replace(
            spec,
            restart_anchors=(5,),
            restart_horizons=(10,),
            restart_compositions=(1, 1),
        )


def test_per_case_restart_anchors_are_checked_when_read():
    spec = get_benchmark("taylor_impact_2d")
    case = spec.splits["test_interp"][0]
    declared = replace(
        spec,
        restart_anchors=lambda cid: (5, 30) if cid == case else (3,),
        restart_horizons=(10,),
    )
    assert declared.restart_anchors_for(case) == (5, 30)
    with pytest.raises(ValueError, match="restart_anchors"):
        declared.restart_anchors_for(spec.splits["test_interp"][1])


def test_restart_anchors_stay_inside_the_scored_horizon():
    # ADR-0039 pins notch-impact to 250 of its frames; a restart may not
    # score the frames the protocol excludes (design §3: anchors lie inside
    # [input_frames - 1, scored_bound)).
    spec = get_benchmark("notch_beam_2d_impact")
    bound = spec.scored_frames_fixed
    assert bound is not None and bound < spec.card.n_frames
    replace(spec, restart_anchors=(bound - 21,), restart_horizons=(20,))
    with pytest.raises(ValueError, match="restart_anchors"):
        replace(spec, restart_anchors=(bound - 20,), restart_horizons=(20,))
