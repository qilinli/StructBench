"""Card renderers and the committed-index drift check."""

import re
from dataclasses import replace
from pathlib import Path

from structbench.benchmarks import available_benchmarks, get_benchmark
from structbench.benchmarks.render import (
    _baseline_line,
    _method_comparison,
    _quickstart_family,
    card_json,
    render_archive_readme,
    render_benchmark_page,
    render_index,
)
from structbench.benchmarks.results import BaselineResult

REPO_ROOT = Path(__file__).resolve().parents[2]


def _all_specs():
    return [get_benchmark(name) for name in available_benchmarks()]


def _result(**overrides):
    """A minimal valid BaselineResult, for fixtures that just need a family
    slot filled (mirrors test_results.py / test_registry.py's local helper).
    """
    kwargs = dict(
        family="cgn",
        label="baseline",
        run_commit="abc1234",
        run_date="2026-07-05",
        metrics={"test_interp": {"rollout_pos_rmse_mm": 1.5}},
    )
    kwargs.update(overrides)
    return BaselineResult(**kwargs)


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
    # Taylor and wave are blessed (ADR-0033); notch-bend covers the path.
    spec = get_benchmark("notch_beam_2d_bend")
    text = render_archive_readme(spec, "notch_beam_2d_bend")
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
    bare = get_benchmark("notch_beam_2d_bend")
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
    # notch-bend has neither overview nor figures nor a baseline
    spec = get_benchmark("notch_beam_2d_bend")
    text = render_benchmark_page(spec, "notch_beam_2d_bend")
    assert "## Figures" not in text
    assert "## The problem" not in text
    assert "No official baseline yet" in text
    # without a blessed result the quickstart stays a plain quickstart
    assert "## Quickstart" in text
    assert "blessed baseline recipe" not in text


def test_blessed_page_quickstart_carries_the_reproduction_sentence():
    # A blessed result adds one sentence to the quickstart: the config is
    # the recipe verbatim, and the two eval modes regenerate the transcribed
    # metrics-<split>.json files (with the honesty caveats).
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_fake_result(),))
    text = render_benchmark_page(spec, "taylor_impact_2d")
    assert "## Quickstart" in text
    assert "blessed baseline recipe verbatim, seed included" in text
    assert "--mode valid" in text
    assert "--mode rollout" in text
    assert "`metrics-<split>.json`" in text
    # honesty caveats: recipe-level reproduction, exact artifact via digest
    assert "bit-identical" in text
    assert "SHA-256" in text


def test_blessed_page_quickstart_trains_the_blessed_family():
    # The quickstart command follows the blessed family, not a hardcoded cgn.
    mlp = replace(_fake_result(), family="mlp", label="MLP baseline")
    spec = replace(get_benchmark("taylor_impact_2d"), results=(mlp,))
    text = render_benchmark_page(spec, "taylor_impact_2d")
    assert "configs/taylor_impact_2d/mlp.toml" in text
    assert "runs/taylor_impact_2d-mlp" in text


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


def test_private_checkpoint_pointer_renders_with_marker():
    # ADR-0037: an archive-relative pointer carries the private-archive marker
    # in both generated views; a public URL renders unmarked.
    private = replace(
        _fake_result(),
        checkpoint="models/taylor_impact_2d/cgn-abc1234/model-best-096000.pt",
        checkpoint_sha256="0" * 64,
    )
    spec = replace(get_benchmark("taylor_impact_2d"), results=(private,))
    for text in (
        render_archive_readme(spec, "taylor_impact_2d"),
        render_benchmark_page(spec, "taylor_impact_2d"),
    ):
        assert "checkpoint: `models/taylor_impact_2d/cgn-abc1234/" in text
        assert "private archive; publication parked" in text

    published = replace(private, checkpoint="https://example.org/m.pt")
    spec = replace(get_benchmark("taylor_impact_2d"), results=(published,))
    text = render_archive_readme(spec, "taylor_impact_2d")
    assert "checkpoint: `https://example.org/m.pt`" in text
    assert "publication parked" not in text


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


# --- ADR-0055 (amended 2026-08-15): relative L2 is the headline metric ---


def _rel_l2_result():
    """A result carrying relative-L2 headline keys, RMSE, and a QoI — the shape
    a registry entry takes once the ADR-0055 amendment's re-eval populates the
    new keys (rel-L2 keys first, as the headline group)."""
    return BaselineResult(
        family="transolver",
        label="Transolver-TC",
        run_commit="abc1234",
        run_date="2026-08-15",
        provisional=True,
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 0.033,
                "rollout_rel_l2_aux": 0.211,
                "rollout_pos_rmse_mm": 1.5,
                "one_step_pos_rmse_mm": 0.004,
                "qoi_final_length_mae_mm": 0.2,
            },
        },
    )


def test_numbers_to_beat_leads_with_relative_l2_group():
    # rel-L2 headline group first, RMSE second, QoI last; rel-L2 keys sit under
    # their own heading, not lumped with the RMSE group.
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_rel_l2_result(),))
    text = render_archive_readme(spec, "taylor_impact_2d")
    rel_title = "_Trajectory error — relative L2 (headline)_"
    rmse_title = "_Trajectory error (RMSE)_"
    qoi_title = "_Quantities of interest (MAE)_"
    assert rel_title in text and rmse_title in text and qoi_title in text
    assert text.index(rel_title) < text.index(rmse_title) < text.index(qoi_title)
    rel_section = text.split(rel_title, 1)[1].split(rmse_title, 1)[0]
    assert "rollout_rel_l2_disp" in rel_section
    rmse_section = text.split(rmse_title, 1)[1].split(qoi_title, 1)[0]
    assert "rel_l2" not in rmse_section
    assert "rollout_pos_rmse_mm" in rmse_section


def test_method_comparison_orders_rel_l2_before_rmse_before_qoi():
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_rel_l2_result(),))
    lines = _method_comparison(spec)

    def row(metric: str) -> int:
        return next(i for i, ln in enumerate(lines) if f"· {metric} " in ln)

    assert row("rollout_rel_l2_disp") < row("rollout_pos_rmse_mm")
    assert row("rollout_pos_rmse_mm") < row("qoi_final_length_mae_mm")


def test_baseline_line_headline_is_relative_l2_when_present():
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_rel_l2_result(),))
    line = _baseline_line(spec)
    assert "rollout_rel_l2_disp" in line
    # the physical-unit RMSE is retained elsewhere but is NOT the quoted headline
    assert "rollout_pos_rmse_mm" not in line


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


# --- method comparison + provisional-aware Quickstart selection (ADR-0046) ---
# _method_comparison is unit-tested directly below, then its wiring into both
# renderers (immediately before "## Numbers to beat") is covered further down
# by the section-ordering and config-path-exists regression tests (Task 3).


def test_method_comparison_empty_state_is_verbatim():
    spec = get_benchmark("notch_beam_2d_bend")
    assert spec.results == ()
    lines = _method_comparison(spec)
    assert lines == [
        "## Method comparison",
        "",
        "*No results yet — method entries land here as runs are "
        "recorded (blessed or provisional).*",
    ]


def test_method_comparison_multi_family_table_and_footnote():
    # (iii): one blessed + two provisional, three families, ragged metrics.
    # test_interp is shared by all three but mgn/transolver miss keys cgn
    # carries there; transolver also misses test_extrap entirely.
    blessed = _result(
        family="cgn",
        label="CGN baseline",
        metrics={
            "test_interp": {
                "rollout_pos_rmse_mm": 1.5,
                "one_step_pos_rmse_mm": 0.004,
                "qoi_final_length_mae_mm": 0.2,
            },
            "test_extrap": {"rollout_pos_rmse_mm": 2.1},
        },
    )
    prov_mgn = _result(
        family="mgn",
        label="MGN candidate",
        provisional=True,
        metrics={
            "test_interp": {"rollout_pos_rmse_mm": 1.8},
            "test_extrap": {"rollout_pos_rmse_mm": 2.3},
        },
    )
    prov_transolver = _result(
        family="transolver",
        label="Transolver candidate",
        provisional=True,
        metrics={"test_interp": {"rollout_pos_rmse_mm": 1.9}},
    )
    spec = replace(
        get_benchmark("taylor_impact_2d"),
        results=(blessed, prov_mgn, prov_transolver),
    )
    lines = _method_comparison(spec)

    assert lines[0] == "## Method comparison"
    assert (
        "| Metric | **cgn** | **mgn** (provisional) | **transolver** (provisional) |"
        in lines
    )
    # first-seen order within the shared split (rmse before qoi); missing
    # cells (a key one method lacks that another carries) render "—"
    assert "| test_interp · rollout_pos_rmse_mm | 1.5 | 1.8 | 1.9 |" in lines
    assert "| test_interp · one_step_pos_rmse_mm | 0.004 | — | — |" in lines
    assert "| test_interp · qoi_final_length_mae_mm | 0.2 | — | — |" in lines
    # transolver has no test_extrap entry at all -> a whole-split "—"
    assert "| test_extrap · rollout_pos_rmse_mm | 2.1 | 2.3 | — |" in lines
    # footnote is the ONLY guard on this string — pinned verbatim
    assert (
        "*Provisional entries are best-effort implementations whose "
        "fidelity is not validated against published numbers "
        "(ADR-0044/0045) — never read them as blessed baselines.*"
    ) in lines


def test_method_comparison_no_footnote_or_tag_when_all_blessed():
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_fake_result(),))
    lines = _method_comparison(spec)
    text = "\n".join(lines)
    assert "| Metric | **cgn** |" in lines
    assert "(provisional)" not in text
    assert "Provisional entries" not in text


def test_method_comparison_provisional_only_tags_every_column():
    a = _result(family="transolver", provisional=True)
    b = _result(family="geoflare", provisional=True)
    spec = replace(get_benchmark("taylor_impact_2d"), results=(a, b))
    lines = _method_comparison(spec)
    assert (
        "| Metric | **transolver** (provisional) | **geoflare** (provisional) |"
        in lines
    )
    assert any("Provisional entries" in line for line in lines)


def test_benchmark_page_method_comparison_appears_before_numbers_to_beat():
    # Multi-result fixture: table + footnote render, section precedes
    # "## Numbers to beat" (ADR-0046 wiring).
    prov_mgn = replace(_fake_result(), family="mgn", provisional=True)
    spec = replace(
        get_benchmark("taylor_impact_2d"), results=(_fake_result(), prov_mgn)
    )
    text = render_benchmark_page(spec, "taylor_impact_2d")
    assert "## Method comparison" in text
    assert text.index("## Method comparison") < text.index("## Numbers to beat")
    assert "| Metric | **cgn** | **mgn** (provisional) |" in text
    assert "Provisional entries are best-effort implementations" in text
    # Numbers-to-beat detail blocks: both entries still render (per-split
    # tables + checkpoint pointer matter for provisional runs too), but only
    # the provisional heading carries the tag (final whole-branch review,
    # 2026-08-09). Exact-line match, not substring: an untagged heading is a
    # prefix of a tagged one, so `in text` alone wouldn't catch a regression
    # where the blessed heading got wrongly tagged too.
    lines = text.splitlines()
    assert "**CGN baseline** (cgn, 2026-07-05, commit `abc1234`)" in lines
    assert "**CGN baseline** (mgn, 2026-07-05, commit `abc1234`) (provisional)" in lines

    # Empty fixture: the empty-state line, same ordering.
    empty_spec = get_benchmark("notch_beam_2d_bend")
    text = render_benchmark_page(empty_spec, "notch_beam_2d_bend")
    assert "## Method comparison" in text
    assert text.index("## Method comparison") < text.index("## Numbers to beat")
    assert (
        "*No results yet — method entries land here as runs are "
        "recorded (blessed or provisional).*"
    ) in text


def test_archive_readme_method_comparison_appears_before_numbers_to_beat():
    prov_mgn = replace(_fake_result(), family="mgn", provisional=True)
    spec = replace(
        get_benchmark("taylor_impact_2d"), results=(_fake_result(), prov_mgn)
    )
    text = render_archive_readme(spec, "taylor_impact_2d")
    assert "## Method comparison" in text
    assert text.index("## Method comparison") < text.index("## Numbers to beat")
    assert "| Metric | **cgn** | **mgn** (provisional) |" in text
    assert "Provisional entries are best-effort implementations" in text
    # Same tagging rule as the landing page: both detail blocks render, only
    # the provisional heading is tagged (final whole-branch review, 2026-08-09).
    # Exact-line match, not substring — see the landing-page test for why.
    lines = text.splitlines()
    assert "**CGN baseline** (cgn, 2026-07-05, commit `abc1234`)" in lines
    assert "**CGN baseline** (mgn, 2026-07-05, commit `abc1234`) (provisional)" in lines

    empty_spec = get_benchmark("notch_beam_2d_bend")
    text = render_archive_readme(empty_spec, "notch_beam_2d_bend")
    assert "## Method comparison" in text
    assert text.index("## Method comparison") < text.index("## Numbers to beat")
    assert (
        "*No results yet — method entries land here as runs are "
        "recorded (blessed or provisional).*"
    ) in text


def test_quickstart_config_path_exists_for_every_benchmark():
    # Regression guard (ADR-0046): would have caught the original bug where
    # deforming_plate's Quickstart pointed at configs/deforming_plate/cgn.toml
    # -- a family with no committed grouped config on disk. For every
    # registered benchmark, the family the renderer actually selects must
    # have a real configs/<name>/<family>.toml on disk.
    for name in available_benchmarks():
        spec = get_benchmark(name)
        text = render_benchmark_page(spec, name)
        match = re.search(r"--config (configs/\S+\.toml)", text)
        assert match, f"{name}: no --config line in rendered Quickstart"
        config_path = match.group(1)
        assert (REPO_ROOT / config_path).is_file(), f"{name}: missing {config_path}"


def test_quickstart_family_prefers_blessed_over_declaration_order():
    # A provisional entry listed FIRST still loses to a later blessed one.
    provisional_first = _result(family="transolver", provisional=True)
    blessed_second = _result(family="mgn", provisional=False)
    spec = replace(
        get_benchmark("taylor_impact_2d"),
        results=(provisional_first, blessed_second),
    )
    assert _quickstart_family(spec) == ("mgn", "blessed")


def test_quickstart_family_falls_back_to_first_when_all_provisional():
    only_provisional = _result(family="geoflare", provisional=True)
    spec = replace(get_benchmark("taylor_impact_2d"), results=(only_provisional,))
    assert _quickstart_family(spec) == ("geoflare", "provisional")


def test_quickstart_family_falls_back_to_spec_default_when_empty():
    spec = replace(get_benchmark("notch_beam_2d_bend"), quickstart_family="mgn")
    assert spec.results == ()
    assert _quickstart_family(spec) == ("mgn", "default")


def test_quickstart_prose_blessed_variant():
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_fake_result(),))
    text = render_benchmark_page(spec, "taylor_impact_2d")
    assert "blessed baseline recipe verbatim" in text


def test_quickstart_prose_provisional_variant():
    provisional = replace(_fake_result(), family="mgn", provisional=True)
    spec = replace(get_benchmark("taylor_impact_2d"), results=(provisional,))
    text = render_benchmark_page(spec, "taylor_impact_2d")
    assert "the provisional mgn recipe" in text
    assert "blessed baseline recipe verbatim" not in text
    assert "configs/taylor_impact_2d/mgn.toml" in text


def test_quickstart_prose_absent_when_no_results():
    spec = get_benchmark("notch_beam_2d_bend")
    text = render_benchmark_page(spec, "notch_beam_2d_bend")
    assert "recipe" not in text


def test_baseline_line_tags_provisional_entries():
    blessed = _fake_result()
    provisional = replace(
        _fake_result(), family="mgn", label="MGN candidate", provisional=True
    )
    spec = replace(get_benchmark("taylor_impact_2d"), results=(blessed, provisional))
    parts = _baseline_line(spec).split("; ")
    assert "(provisional)" not in parts[0]
    assert "(provisional)" in parts[1]
    assert "MGN candidate" in parts[1]


def test_archive_readme_quickstart_family_selection_is_output_neutral():
    # _quickstart_family replaces the old hardcoded
    # `spec.results[0].family if spec.results else "cgn"`; every existing
    # benchmark must resolve to the identical family EXCEPT deforming_plate,
    # whose quickstart_family default flips cgn -> mgn in this same task
    # (Task 3): configs/deforming_plate/cgn.toml never existed on disk, so
    # the old hardcoded fallback was the exact bug the config-path-exists
    # regression guard now catches. Every other benchmark (no provisional
    # entries committed yet) stays byte-for-byte the same as before this task.
    for name in available_benchmarks():
        spec = get_benchmark(name)
        if name == "deforming_plate":
            old_family = "mgn"
        else:
            old_family = spec.results[0].family if spec.results else "cgn"
        text = render_archive_readme(spec, name)
        assert f"configs/{name}/{old_family}.toml" in text
        assert f"runs/{name}-{old_family}" in text
