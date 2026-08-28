"""Card renderers and the committed-index drift check."""

import re
from dataclasses import replace
from pathlib import Path

from structbench.benchmarks import available_benchmarks, get_benchmark
from structbench.benchmarks.render import (
    _baseline_line,
    _col_label,
    _leaderboard,
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


def _bare_spec():
    """A result-less spec with no overview or figures — the empty-state fixture
    for the render placeholder paths. Formerly the parked ``notch_beam_2d_bend``
    benchmark; it was descoped from the registry (ADR-0056), so the empty-state
    tests synthesize a bare spec instead of depending on a specific benchmark
    having no results.
    """
    spec = get_benchmark("taylor_impact_2d")
    return replace(spec, results=(), card=replace(spec.card, overview="", figures=()))


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


def test_archive_readme_is_a_dataset_page():
    # The archive README describes the data (files, layout, loading, splits)
    # and points at the benchmark page for protocol/baselines/leaderboard —
    # those tables are not duplicated into the archive.
    spec = get_benchmark("taylor_impact_2d")
    text = render_archive_readme(spec, "taylor_impact_2d")
    for heading in (
        "## Dataset summary",
        "## Files",
        "## HDF5 layout",
        "## Loading",
        "## Benchmark protocol",
    ):
        assert heading in text
    assert "## Leaderboard" not in text and "## Baseline details" not in text
    assert "## Splits" not in text  # split ids live in cases.csv, counts in the summary
    # SPH layout table + both loading paths
    assert "`response/element/sph/{stress, strain, strain_rate}` | (T, P, 6)" in text
    assert 'load_case_trajectory("<case_id>.h5", aux_field="von_mises_stress")' in text
    assert "import h5py" in text
    # protocol pointer: values from the card, the page link, the train command
    assert f"{spec.card.input_frames} input frames" in text
    assert spec.card.qois[0] in text
    assert "docs/benchmarks/taylor_impact_2d.md" in text
    assert "configs/taylor_impact_2d/cgn.toml" in text


def test_archive_readme_mesh_variant_describes_nodal_layout():
    text = render_archive_readme(get_benchmark("deforming_plate"), "deforming_plate")
    assert "`nodes/reference_coords`" in text
    assert "`response/node/von_mises_stress` | (T, N, 1)" in text
    assert "elements/sph" not in text
    assert "the mesh-aware loader reads it verbatim" in text


def test_benchmark_page_without_results_carries_placeholder():
    # A result-less bare spec exercises the no-baseline placeholder path.
    spec = _bare_spec()
    text = render_benchmark_page(spec, "taylor_impact_2d")
    assert "No official baseline yet" in text


def test_benchmark_page_renders_leaderboard_row():
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_fake_result(),))
    text = render_benchmark_page(spec, "taylor_impact_2d")
    assert "No official baseline yet" not in text
    assert "CGN baseline" in text
    assert "abc1234" in text
    # one leaderboard row: scheme unknown -> "—". The RMSE tier is trimmed to
    # test_interp and drops the one_step diagnostic columns, so the row carries
    # a single value under the one surviving column, `interp·pos (mm)`.
    assert "| CGN baseline | — | 1.50000 |" in text
    # notes render in the provenance block, not a metric table
    assert "single A100, 100k steps" in text


def test_index_section_renders_baseline_line_both_ways():
    bare = _bare_spec()
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
    # the blessed baseline leaderboard + details + quickstart are present
    assert "## Leaderboard" in text
    assert "## Baseline details" in text
    assert "| CGN |" in text  # the blessed method row in the leaderboard
    assert "## Quickstart" in text
    assert "configs/taylor_impact_2d/cgn.toml" in text


def test_benchmark_page_omits_absent_optional_sections():
    # a bare spec has neither overview nor figures nor a baseline
    spec = _bare_spec()
    text = render_benchmark_page(spec, "taylor_impact_2d")
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


def test_leaderboard_splits_metrics_into_three_tiers():
    # The blessed Taylor result carries rel-L2, RMSE, and QoI keys: three tiers,
    # in headline -> RMSE -> QoI order, each a separate labelled table.
    text = render_benchmark_page(get_benchmark("taylor_impact_2d"), "taylor_impact_2d")
    head = "_Headline — pooled relative L2 (↓ better)_"
    rmse = "_Trajectory error — RMSE_"
    qoi = "_Quantities of interest (MAE)_"
    assert head in text and rmse in text and qoi in text
    assert text.index(head) < text.index(rmse) < text.index(qoi)
    # QoI column labels sit under the QoI tier, not the RMSE one; units ride
    # in the parenthesised header suffix (mm/MPa/ms).
    qoi_section = text.split(qoi, 1)[1]
    assert "interp·final_length (mm)" in qoi_section
    rmse_section = text.split(rmse, 1)[1].split(qoi, 1)[0]
    assert "final_length" not in rmse_section
    assert "interp·pos (mm)" in rmse_section


def test_private_checkpoint_pointer_renders_with_marker():
    # ADR-0037: an archive-relative pointer carries the private-archive marker
    # in both generated views; a public URL renders unmarked.
    private = replace(
        _fake_result(),
        checkpoint="models/taylor_impact_2d/cgn-abc1234/model-best-096000.pt",
        checkpoint_sha256="0" * 64,
    )
    spec = replace(get_benchmark("taylor_impact_2d"), results=(private,))
    text = render_benchmark_page(spec, "taylor_impact_2d")
    assert "checkpoint: `models/taylor_impact_2d/cgn-abc1234/" in text
    assert "private archive; publication parked" in text

    published = replace(private, checkpoint="https://example.org/m.pt")
    spec = replace(get_benchmark("taylor_impact_2d"), results=(published,))
    text = render_benchmark_page(spec, "taylor_impact_2d")
    assert "checkpoint: `https://example.org/m.pt`" in text
    assert "publication parked" not in text


def test_single_metric_group_renders_only_its_tier():
    # Only RMSE metrics -> only the RMSE tier renders (still titled; the
    # leaderboard always names its tiers, no unlabelled fallback).
    result = BaselineResult(
        family="cgn",
        label="CGN baseline",
        run_commit="abc1234",
        run_date="2026-07-05",
        metrics={"test_interp": {"rollout_pos_rmse_mm": 1.5}},
    )
    spec = replace(get_benchmark("taylor_impact_2d"), results=(result,))
    text = render_benchmark_page(spec, "taylor_impact_2d")
    assert "_Trajectory error — RMSE_" in text
    assert "_Headline — pooled relative L2 (↓ better)_" not in text
    assert "_Quantities of interest (MAE)_" not in text
    assert "| CGN baseline | — | 1.50000 |" in text


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


def test_leaderboard_leads_with_relative_l2_tier():
    # Headline rel-L2 tier first, RMSE second, QoI last; rel-L2 columns sit
    # under their own heading, not lumped with the RMSE tier.
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_rel_l2_result(),))
    text = render_benchmark_page(spec, "taylor_impact_2d")
    head = "_Headline — pooled relative L2 (↓ better)_"
    rmse = "_Trajectory error — RMSE_"
    qoi = "_Quantities of interest (MAE)_"
    assert head in text and rmse in text and qoi in text
    assert text.index(head) < text.index(rmse) < text.index(qoi)
    head_section = text.split(head, 1)[1].split(rmse, 1)[0]
    assert "interp·disp" in head_section
    rmse_section = text.split(rmse, 1)[1].split(qoi, 1)[0]
    assert "·disp" not in rmse_section
    assert "interp·pos (mm)" in rmse_section


def test_leaderboard_orders_rel_l2_before_rmse_before_qoi():
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_rel_l2_result(),))
    lines = _leaderboard(spec)

    def tier(title: str) -> int:
        return next(i for i, ln in enumerate(lines) if ln == f"_{title}_")

    assert tier("Headline — pooled relative L2 (↓ better)") < tier(
        "Trajectory error — RMSE"
    )
    assert tier("Trajectory error — RMSE") < tier("Quantities of interest (MAE)")


def test_col_label_strips_prefixes_and_suffixes():
    # rel-L2 keys stay dimensionless; physical-unit keys gain a parenthesised
    # unit suffix (mm/MPa/ms) instead of the bare ``_mm``/``_mpa``/``_ms`` tail.
    assert _col_label("test_interp", "rollout_rel_l2_disp") == "interp·disp"
    assert _col_label("test_interp", "rollout_rel_l2_aux") == "interp·aux"
    assert _col_label("test_interp", "rollout_pos_rmse_mm") == "interp·pos (mm)"
    assert _col_label("test_interp", "rollout_vm_rmse_mpa") == "interp·vm (MPa)"
    assert _col_label("test_extrap", "one_step_pos_rmse_mm") == "extrap·1s·pos (mm)"
    assert (
        _col_label("test_interp", "qoi_final_length_mae_mm")
        == "interp·final_length (mm)"
    )
    assert _col_label("test_interp", "qoi_t_peak_vm_mae_ms") == "interp·t_peak_vm (ms)"
    # a non-``test_`` split keeps its name (notch's off-grid probe)
    assert _col_label("probe", "rollout_rel_l2_disp") == "probe·disp"


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
    # the archive README (a dataset page) carries the rationale in neither
    # form — it points at the benchmark page for the protocol
    archive = render_archive_readme(spec, "taylor_impact_2d")
    assert "Protocol rationale" not in archive
    assert "<details>" not in archive
    assert "docs/benchmarks/taylor_impact_2d.md" in archive


# --- leaderboard + provisional-aware Quickstart selection (ADR-0046, ADR-0055) ---
# _leaderboard is unit-tested directly below, then its wiring into both
# renderers (immediately before "## Baseline details") is covered further down
# by the section-ordering and config-path-exists regression tests (Task 3).


def test_leaderboard_empty_state_is_verbatim():
    spec = _bare_spec()
    assert spec.results == ()
    lines = _leaderboard(spec)
    assert lines == [
        "## Leaderboard",
        "",
        "*No results yet — method entries land here as runs are "
        "recorded (blessed or provisional).*",
    ]


def test_leaderboard_multi_family_rows_in_declaration_order():
    # one blessed + two provisional, three families, ragged metrics.
    # Rows follow registry declaration order (CGN, MGN, Transolver); no
    # provisional tag and no footnote appear. The RMSE tier is trimmed to
    # test_interp and drops the one_step columns (so a single value column),
    # ragged cells render "—".
    blessed = _result(
        family="cgn",
        label="CGN baseline",
        scheme="autoregressive",
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
    lines = _leaderboard(spec)
    text = "\n".join(lines)

    assert lines[0] == "## Leaderboard"
    # RMSE tier: header + one row per method in declaration order; the one_step
    # and test_extrap columns are trimmed to a single `interp·pos (mm)` column.
    assert "_Trajectory error — RMSE_" in lines
    assert "| Method | Scheme | interp·pos (mm) |" in lines
    assert "| CGN baseline | autoregressive | 1.50000 |" in lines
    assert "| MGN candidate | — | 1.80000 |" in lines
    assert "| Transolver candidate | — | 1.90000 |" in lines
    # rows appear in registry declaration order (CGN, then MGN, then Transolver)
    assert (
        text.index("| CGN baseline | autoregressive | 1.50000 |")
        < text.index("| MGN candidate | — | 1.80000 |")
        < text.index("| Transolver candidate | — | 1.90000 |")
    )
    # QoI tier: only cgn carries the QoI; the others render "—"
    assert "_Quantities of interest (MAE)_" in lines
    assert "| CGN baseline | autoregressive | 0.20000 |" in lines
    # no provisional marker or footnote anywhere in the leaderboard
    assert "*(provisional)*" not in text
    assert "Provisional entries are best-effort" not in text


def test_leaderboard_rows_follow_declaration_order_not_metric():
    # Rows are NOT ranked (ranking is gone): registry declaration order wins
    # even when the first-declared method has the worse headline number.
    worse = _result(
        family="mgn",
        label="MGN",
        provisional=True,
        metrics={"test_interp": {"rollout_rel_l2_disp": 0.9}},
    )
    better = _result(
        family="transolver",
        label="Transolver",
        provisional=True,
        metrics={"test_interp": {"rollout_rel_l2_disp": 0.1}},
    )
    spec = replace(get_benchmark("taylor_impact_2d"), results=(worse, better))
    lines = _leaderboard(spec)
    rows = [ln for ln in lines if ln.startswith(("| MGN", "| Transolver"))]
    assert rows[0].startswith("| MGN")  # declared first, despite worse 0.9
    assert rows[1].startswith("| Transolver")  # declared second, despite 0.1


def test_leaderboard_declaration_order_holds_with_ragged_metrics():
    # The single declaration order drives every tier: a headline-less entry
    # declared first still renders first in the RMSE tier (no reordering).
    without = _result(
        family="mgn",
        label="MGN",
        provisional=True,
        metrics={"test_interp": {"rollout_pos_rmse_mm": 0.1}},
    )
    with_rel = _result(
        family="transolver",
        label="Transolver",
        provisional=True,
        metrics={
            "test_interp": {"rollout_rel_l2_disp": 0.5, "rollout_pos_rmse_mm": 1.0}
        },
    )
    spec = replace(get_benchmark("taylor_impact_2d"), results=(without, with_rel))
    lines = _leaderboard(spec)
    start = lines.index("_Trajectory error — RMSE_")
    rows = [ln for ln in lines[start:] if ln.startswith(("| MGN", "| Transolver"))]
    assert rows[0].startswith("| MGN")  # declared first
    assert rows[1].startswith("| Transolver")  # declared second


def test_leaderboard_no_footnote_or_tag_when_all_blessed():
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_fake_result(),))
    lines = _leaderboard(spec)
    # the method row carries no provisional tag (the intro paragraph mentions
    # "*(provisional)*" for the legend, so guard the row line specifically)
    assert "| CGN baseline | — |" in "\n".join(lines)
    assert not any(
        ln.startswith("| CGN baseline") and "*(provisional)*" in ln for ln in lines
    )
    # no provisional footnote when every entry is blessed
    assert not any("Provisional entries are best-effort" in ln for ln in lines)


def test_leaderboard_all_provisional_carries_no_tag_or_footnote():
    # Even when every entry is provisional, rows carry no *(provisional)* tag
    # and no footnote is appended (the provisional field still exists, it is
    # just no longer surfaced in the leaderboard).
    a = _result(family="transolver", label="Transolver", provisional=True)
    b = _result(family="geoflare", label="GeoFLARE", provisional=True)
    spec = replace(get_benchmark("taylor_impact_2d"), results=(a, b))
    lines = _leaderboard(spec)
    text = "\n".join(lines)
    assert "| Transolver | — | 1.50000 |" in lines
    assert "| GeoFLARE | — | 1.50000 |" in lines
    # declaration order preserved
    assert text.index("| Transolver | — | 1.50000 |") < text.index("| GeoFLARE | — | 1.50000 |")
    assert "*(provisional)*" not in text
    assert "Provisional entries" not in text


def test_baseline_details_has_notes_but_no_metric_table():
    # The provenance block keeps the notes but no per-split metric table —
    # every number now lives in the leaderboard above it.
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_fake_result(),))
    text = render_benchmark_page(spec, "taylor_impact_2d")
    details = text.split("## Baseline details", 1)[1]
    assert "single A100, 100k steps" in details
    assert "| split |" not in details
    assert "| test_interp |" not in details


def test_benchmark_page_leaderboard_appears_before_baseline_details():
    # Multi-result fixture: rows render in declaration order with no provisional
    # tag and no footnote; the section precedes "## Baseline details".
    prov_mgn = replace(
        _fake_result(), family="mgn", label="MGN candidate", provisional=True
    )
    spec = replace(
        get_benchmark("taylor_impact_2d"), results=(_fake_result(), prov_mgn)
    )
    text = render_benchmark_page(spec, "taylor_impact_2d")
    assert "## Leaderboard" in text
    assert text.index("## Leaderboard") < text.index("## Baseline details")
    # RMSE tier trimmed to a single test_interp column; rows in declaration
    # order (blessed CGN first, provisional MGN second), neither tagged.
    assert "| CGN baseline | — | 1.50000 |" in text
    assert "| MGN candidate | — | 1.50000 |" in text
    assert text.index("| CGN baseline | — | 1.50000 |") < text.index(
        "| MGN candidate | — | 1.50000 |"
    )
    assert "*(provisional)*" not in text
    assert "Provisional entries are best-effort implementations" not in text
    # Baseline-details headings: both render, neither tagged. Exact-line match.
    lines = text.splitlines()
    assert "**CGN baseline** (cgn, 2026-07-05, commit `abc1234`)" in lines
    assert "**MGN candidate** (mgn, 2026-07-05, commit `abc1234`)" in lines

    # Empty fixture: the empty-state line, same ordering.
    empty_spec = _bare_spec()
    text = render_benchmark_page(empty_spec, "taylor_impact_2d")
    assert "## Leaderboard" in text
    assert text.index("## Leaderboard") < text.index("## Baseline details")
    assert (
        "*No results yet — method entries land here as runs are "
        "recorded (blessed or provisional).*"
    ) in text


def test_archive_readme_never_renders_result_tables():
    # Results (blessed or provisional, or none) never reach the dataset page:
    # the leaderboard and baseline details have one home, the benchmark page.
    prov_mgn = replace(
        _fake_result(), family="mgn", label="MGN candidate", provisional=True
    )
    spec = replace(
        get_benchmark("taylor_impact_2d"), results=(_fake_result(), prov_mgn)
    )
    for s in (spec, _bare_spec()):
        text = render_archive_readme(s, "taylor_impact_2d")
        assert "## Leaderboard" not in text
        assert "## Baseline details" not in text
        assert "| CGN baseline |" not in text
        assert "No results yet" not in text


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
    spec = replace(_bare_spec(), quickstart_family="mgn")
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
    spec = _bare_spec()
    text = render_benchmark_page(spec, "taylor_impact_2d")
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


def test_archive_readme_quickstart_family_follows_blessed_first():
    # The Quickstart config path follows _quickstart_family's selection
    # (ADR-0046: first BLESSED family in declaration order, else the first
    # provisional entry, else the spec default) — NOT raw declaration order.
    # Taylor now declares MGN (provisional) first, but the CGN blessed baseline
    # must still anchor the quickstart; the render must thread the same choice.
    for name in available_benchmarks():
        spec = get_benchmark(name)
        family, _ = _quickstart_family(spec)
        text = render_archive_readme(spec, name)
        assert f"configs/{name}/{family}.toml" in text
        assert f"runs/{name}-{family}" in text
    # Regression pin: taylor's provisional-first registry still resolves to cgn.
    taylor = get_benchmark("taylor_impact_2d")
    assert taylor.results[0].family == "mgn"  # provisional, declared first
    assert _quickstart_family(taylor)[0] == "cgn"  # blessed still wins


def test_references_section_lists_each_method():
    # ADR-0033: the landing page ends with a de-duplicated, method-tagged
    # citation list built from each result's ``reference`` field.
    text = render_benchmark_page(get_benchmark("taylor_impact_2d"), "taylor_impact_2d")
    assert "## References" in text
    assert "**CGN** — Li, Q." in text
    assert "**Transolver** — Wu, H." in text
    assert "**MGN** — Pfaff, T." in text
    # it is the final section (after the Quickstart)
    assert text.index("## References") > text.index("## Quickstart")


def test_references_omitted_without_citations():
    # a result-less spec has no References section.
    spec = _bare_spec()
    assert "## References" not in render_benchmark_page(spec, "taylor_impact_2d")


def test_pending_baseline_renders_training_placeholder_row():
    # A pending entry (still-training baseline, empty metrics) renders a
    # leaderboard row of "—" tagged (training), with a note; it never raises.
    pending = BaselineResult(
        family="mgn",
        label="MGN",
        scheme="autoregressive",
        provisional=True,
        pending=True,
        run_commit="abc1234",
        run_date="2026-08-16",
        metrics={},
        notes="run in progress",
    )
    real = _rel_l2_result()
    spec = replace(get_benchmark("taylor_impact_2d"), results=(pending, real))
    lines = _leaderboard(spec)
    # pending row first (declaration order), all metric cells "—"
    row = next(ln for ln in lines if ln.startswith("| MGN *(training)*"))
    assert set(c.strip() for c in row.split("|")[3:-1]) == {"—"}
    assert any("still in progress" in ln for ln in lines)
