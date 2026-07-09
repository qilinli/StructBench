"""Card renderers and the committed-index drift check."""

from dataclasses import replace
from pathlib import Path

from structbench.benchmarks import available_benchmarks, get_benchmark
from structbench.benchmarks.render import (
    card_json,
    render_archive_readme,
    render_benchmark_page,
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
    # protocol values + rationale from the card (ADR-0032, ADR-0035)
    assert f"{spec.card.input_frames} input frames" in text
    assert spec.card.protocol_rationale[:40] in text
    # QoIs listed; runnable command names the grouped config
    assert spec.card.qois[0] in text
    assert "configs/taylor_impact_2d/cgn.toml" in text


def test_archive_readme_without_results_carries_placeholder():
    # Taylor is blessed (ADR-0033); wave is still unblessed and covers the path.
    spec = get_benchmark("wave_propagation_1d")
    text = render_archive_readme(spec, "wave_propagation_1d")
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


# --- per-benchmark landing pages (ADR-0036) ---


def test_committed_benchmark_pages_are_up_to_date():
    for name in available_benchmarks():
        page = REPO_ROOT / "docs" / "benchmarks" / f"{name}.md"
        assert page.read_text(encoding="utf-8") == render_benchmark_page(
            get_benchmark(name), name
        ), name


def test_index_links_to_each_benchmark_page():
    text = render_index(_all_specs())
    for name in available_benchmarks():
        assert f"(benchmarks/{name}.md)" in text, name


def test_benchmark_page_embeds_overview_numbers_and_figures():
    spec = get_benchmark("taylor_impact_2d")
    text = render_benchmark_page(spec, "taylor_impact_2d")
    # narrative from the card
    assert spec.card.overview[:24] in text
    # each figure renders as a markdown image at a page-relative asset path
    assert spec.card.figures  # guard: taylor has figures
    for fig in spec.card.figures:
        assert f"(../../{fig.path})" in text
        assert fig.caption in text
    # the blessed baseline table + quickstart are present
    assert "## Numbers to beat" in text
    assert "CGN baseline" in text
    assert "## Quickstart" in text
    assert "configs/taylor_impact_2d/cgn.toml" in text


def test_benchmark_page_omits_absent_optional_sections():
    # wave has neither overview nor figures nor a baseline
    spec = get_benchmark("wave_propagation_1d")
    text = render_benchmark_page(spec, "wave_propagation_1d")
    assert "## Figures" not in text
    assert "## The problem" not in text
    assert "No official baseline yet" in text


def test_card_figure_paths_exist():
    for name in available_benchmarks():
        for fig in get_benchmark(name).card.figures:
            assert (REPO_ROOT / fig.path).is_file(), f"{name}: missing {fig.path}"


def test_numbers_to_beat_splits_qoi_into_its_own_table():
    # A result with both RMSE and QoI metrics renders two narrower tables.
    text = render_archive_readme(get_benchmark("taylor_impact_2d"), "taylor_impact_2d")
    assert "_Trajectory error (RMSE)_" in text
    assert "_Quantities of interest (MAE)_" in text
    # the QoI columns sit under the QoI subheading, not the RMSE one
    qoi_section = text.split("_Quantities of interest (MAE)_", 1)[1]
    assert "qoi_final_length_mae_mm" in qoi_section
    rmse_section = text.split("_Trajectory error (RMSE)_", 1)[1].split(
        "_Quantities of interest (MAE)_", 1
    )[0]
    assert "qoi_" not in rmse_section
    assert "rollout_pos_rmse_mm" in rmse_section


def test_single_metric_group_stays_one_unlabelled_table():
    # No qoi_ metrics -> one table, no subheadings (unchanged behaviour).
    result = BaselineResult(
        family="cgn",
        label="CGN baseline",
        run_commit="abc1234",
        run_date="2026-07-05",
        metrics={"test_interp": {"rollout_pos_rmse_mm": 1.5}},
    )
    spec = replace(get_benchmark("taylor_impact_2d"), results=(result,))
    text = render_archive_readme(spec, "taylor_impact_2d")
    assert "_Trajectory error (RMSE)_" not in text
    assert "| test_interp | 1.5 |" in text


def test_landing_page_folds_protocol_rationale_but_index_does_not():
    spec = get_benchmark("taylor_impact_2d")
    page = render_benchmark_page(spec, "taylor_impact_2d")
    # full rationale text is present, but inside a collapsed <details>, not an
    # inline "- Protocol rationale:" bullet
    assert "<details>" in page
    assert spec.card.protocol_rationale[:40] in page
    assert "- Protocol rationale:" not in page
    # the archive README keeps it inline (fold is page-only)
    archive = render_archive_readme(spec, "taylor_impact_2d")
    assert "- Protocol rationale:" in archive
    assert "<details>" not in archive
