"""The registry's ADR-0074 fields and their rendering (open-evaluation plan
§8, task 3): tiers, divisions, blind tags, per-case rows and the headline by
coverage bin; every existing row unchanged by the defaults."""

from dataclasses import replace

import pytest

from structbench.benchmarks import get_benchmark
from structbench.benchmarks.render import (
    _coverage_tier,
    _results_registry,
    render_benchmark_page,
)
from structbench.benchmarks.results import DIVISIONS, TIERS, BaselineResult


def _result(**overrides):
    kwargs = dict(
        family="cgn",
        label="CGN baseline",
        run_commit="abc1234",
        run_date="2026-11-08",
        metrics={"test_interp": {"rollout_rel_l2_disp": 0.02}},
    )
    kwargs.update(overrides)
    return BaselineResult(**kwargs)


def test_defaults_are_the_platform_run_values():
    r = _result()
    assert r.tier == "platform" and r.division == "closed"
    assert r.blind_tag == "" and r.submitter == "" and r.writeup == "" and r.data == ""
    assert dict(r.case_metrics) == {}
    assert TIERS == ("platform", "verified", "self_reported")
    assert DIVISIONS == ("closed", "open")


def test_fields_are_validated():
    with pytest.raises(ValueError, match="tier"):
        _result(tier="blessed")
    with pytest.raises(ValueError, match="division"):
        _result(division="semi")
    with pytest.raises(ValueError, match="declare its data"):
        _result(division="open")
    with pytest.raises(ValueError, match="submitter"):
        _result(tier="self_reported")
    with pytest.raises(ValueError, match="must not be empty"):
        _result(case_metrics={"T-20-80-130": {}})
    ok = _result(
        tier="verified",
        division="open",
        data="train + 200 own LS-DYNA runs",
        submitter="Someone",
        case_metrics={"T-20-80-130": {"rollout_rel_l2_disp": 0.03}},
    )
    assert ok.case_metrics["T-20-80-130"]["rollout_rel_l2_disp"] == 0.03


def test_spec_refuses_case_metrics_off_its_scored_cases():
    spec = get_benchmark("taylor_impact_2d")
    bad = _result(case_metrics={"T-20-80-100": {"rollout_rel_l2_disp": 0.1}})  # train
    with pytest.raises(ValueError, match="not scored cases"):
        replace(spec, results=(bad,))
    ok = _result(case_metrics={"T-20-80-130": {"rollout_rel_l2_disp": 0.1}})
    replace(spec, results=(ok,))


def test_tier_and_division_columns_appear_only_when_needed():
    spec = get_benchmark("taylor_impact_2d")
    platform_only = replace(spec, results=(_result(),))
    lines = _results_registry(platform_only)
    assert lines[0] == "## Results registry"
    assert any("registration order and are not ranked" in ln for ln in lines)
    assert not any(ln.startswith("| Method | Scheme | Tier") for ln in lines)
    external = _result(
        family="mgn",
        label="Someone's model",
        tier="self_reported",
        submitter="Someone",
        division="open",
        data="own data",
    )
    mixed = replace(spec, results=(_result(), external))
    lines = _results_registry(mixed)
    assert "| Method | Scheme | Tier | Division | interp·disp |" in lines
    assert "| CGN baseline | — | platform | closed | 0.02000 |" in lines
    assert "| Someone's model | — | self-reported | open | 0.02000 |" in lines


def test_baseline_details_carry_the_external_provenance():
    spec = get_benchmark("taylor_impact_2d")
    external = _result(
        label="Someone's model",
        tier="verified",
        submitter="Someone",
        blind_tag="blind-v1",
        writeup="https://example.org/paper",
    )
    page = render_benchmark_page(replace(spec, results=(external,)), "taylor_impact_2d")
    details = page.split("## Baseline details", 1)[1]
    assert (
        "verified result; scored under `blind-v1`; submitted by Someone; "
        "write-up <https://example.org/paper>."
    ) in details
    page = render_benchmark_page(
        replace(spec, results=(_result(),)), "taylor_impact_2d"
    )
    assert "platform result" not in page


def test_headline_by_coverage_bin_renders_from_case_metrics():
    spec = get_benchmark("taylor_impact_2d")
    interp = {c: {"rollout_rel_l2_disp": 0.01} for c in spec.splits["test_interp"]}
    extrap = {c: {"rollout_rel_l2_disp": 0.05} for c in spec.splits["test_extrap"]}
    with_cases = _result(case_metrics={**interp, **extrap})
    without = _result(label="No per-case rows", family="mgn", provisional=True)
    lines = _coverage_tier(replace(spec, results=(with_cases, without)))
    assert lines[0].startswith("_Headline by coverage bin")
    assert "| Method | Scheme | interp·within, ≤ 1 | extrap·beyond, ≤ 1 |" in lines
    assert "| CGN baseline | — | 0.01000 | 0.05000 |" in lines
    assert "| No per-case rows | — | — | — |" in lines
    # wired into the registry section
    section = _results_registry(replace(spec, results=(with_cases,)))
    assert any(ln.startswith("_Headline by coverage bin") for ln in section)
    # no per-case rows anywhere: no table
    assert _coverage_tier(replace(spec, results=(without,))) == []
    # a benchmark without coverage: no table even with rows
    dp = get_benchmark("deforming_plate")
    dp_rows = _result(
        metrics={"test": {"rollout_rel_l2_disp": 0.1}},
        case_metrics={dp.splits["test"][0]: {"rollout_rel_l2_disp": 0.1}},
    )
    assert _coverage_tier(replace(dp, results=(dp_rows,))) == []
