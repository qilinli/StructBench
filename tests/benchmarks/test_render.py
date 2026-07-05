"""Card renderers and the committed-index drift check."""

from dataclasses import replace
from pathlib import Path

from structbench.benchmarks import available_benchmarks, get_benchmark
from structbench.benchmarks.render import (
    card_json,
    render_archive_readme,
    render_index,
)
from structbench.benchmarks.results import BaselineResult

REPO_ROOT = Path(__file__).resolve().parents[2]


def _all_specs():
    return [get_benchmark(name) for name in available_benchmarks()]


def _fake_result():
    return BaselineResult(
        family="cgn",
        label="CGN baseline",
        run_commit="abc1234",
        run_date="2026-07-05",
        metrics={
            "test_interp": {
                "rollout_pos_rmse_mm": 1.5,
                "one_step_pos_rmse_mm": 0.004,
            },
            "test_extrap": {"rollout_pos_rmse_mm": 2.1},
        },
        notes="single A100, 100k steps",
    )


def test_index_contains_taylor_row_and_generation_marker():
    text = render_index(_all_specs())
    assert "do not edit by hand" in text
    assert "Taylor2D-Impact" in text
    assert "Wave1D-Propagation" in text
    assert "SPH" in text


def test_archive_readme_is_self_describing():
    spec = get_benchmark("taylor_impact_2d")
    text = render_archive_readme(spec, "taylor_impact_2d")
    assert "Taylor2D-Impact" in text
    assert "CC BY 4.0" in text
    assert "g-mm-ms" in text


def test_archive_readme_carries_task_eval_and_usage_sections():
    spec = get_benchmark("taylor_impact_2d")
    text = render_archive_readme(spec, "taylor_impact_2d")
    assert "## Task" in text
    assert "## Evaluation criteria" in text
    assert "## Numbers to beat" in text
    assert "## Using this archive" in text
    # protocol values + rationale from the card (ADR-0032)
    assert f"init {spec.card.init_frames} frames" in text
    assert spec.card.protocol_rationale[:40] in text
    # QoIs listed; runnable command names the grouped config
    assert spec.card.qois[0] in text
    assert "configs/taylor_impact_2d/cgn.toml" in text


def test_archive_readme_without_results_carries_placeholder():
    spec = get_benchmark("taylor_impact_2d")
    text = render_archive_readme(spec, "taylor_impact_2d")
    assert "No official baseline yet" in text


def test_archive_readme_renders_result_table():
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_fake_result(),))
    text = render_archive_readme(spec, "taylor_impact_2d")
    assert "No official baseline yet" not in text
    assert "CGN baseline" in text
    assert "abc1234" in text
    # split rows in card order, metric columns in first-seen order
    assert "| test_interp | 1.5 | 0.004 |" in text
    assert "| test_extrap | 2.1 |" in text
    assert "single A100, 100k steps" in text


def test_index_section_renders_baseline_line_both_ways():
    bare = get_benchmark("wave_propagation_1d")
    assert "no official baseline yet" in render_index([bare])
    with_result = replace(get_benchmark("taylor_impact_2d"), results=(_fake_result(),))
    text = render_index([with_result])
    assert "CGN baseline" in text
    assert "abc1234" in text


def test_card_json_round_trips():
    import json

    spec = get_benchmark("taylor_impact_2d")
    data = json.loads(card_json(spec.card))
    assert data["name"] == "Taylor2D-Impact"


def test_committed_index_is_up_to_date():
    committed = (REPO_ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    assert committed == render_index(_all_specs())
