"""BenchmarkCard invariants (ADR-0027)."""

import json

import pytest

from structbench.benchmarks.card import BenchmarkCard, BenchmarkFigure


def _kwargs(**overrides):
    base = dict(
        name="Demo-Bench",
        version="0.1",
        description="A demo benchmark.",
        provenance="Synthetic, for tests.",
        data_license="CC BY 4.0",
        solver="LS-DYNA",
        discretisation="SPH",
        materials=("*MAT_ELASTIC",),
        erosion=False,
        loading="rigid-wall impact",
        source_units="g-mm-ms",
        geometry="2D bar",
        n_cases=3,
        splits={"train": 2, "val": 1},
        task="autoregressive transition",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=("final_length",),
        fields=("positions", "stress"),
        particles_per_case="100",
        n_frames=10,
        output_dt_ms=0.1,
        input_frames=6,
        protocol_rationale="test-only rationale",
    )
    base.update(overrides)
    return base


def test_card_accepts_consistent_splits():
    card = BenchmarkCard(**_kwargs())
    assert card.n_cases == 3
    assert card.size_gb is None


def test_card_rejects_split_sum_mismatch():
    with pytest.raises(ValueError, match="n_cases"):
        BenchmarkCard(**_kwargs(n_cases=99))


def test_card_rejects_input_frames_below_two():
    # Needs >= 2 frames to form one velocity (ADR-0035).
    with pytest.raises(ValueError, match="input_frames must be >= 2"):
        BenchmarkCard(**_kwargs(input_frames=1))


def test_card_json_dict_serializes():
    card = BenchmarkCard(**_kwargs())
    payload = json.dumps(card.to_json_dict())
    assert "Demo-Bench" in payload


def test_card_defaults_to_no_overview_or_figures():
    card = BenchmarkCard(**_kwargs())
    assert card.overview == ""
    assert card.figures == ()


def test_card_json_dict_serializes_figures():
    # asdict recurses into nested BenchmarkFigure records (ADR-0036).
    card = BenchmarkCard(
        **_kwargs(
            overview="## Demo\n\nText.",
            figures=(BenchmarkFigure(path="assets/x.png", caption="a plot"),),
        )
    )
    payload = json.loads(json.dumps(card.to_json_dict()))
    assert payload["figures"][0]["path"] == "assets/x.png"
    assert payload["overview"].startswith("## Demo")


@pytest.mark.parametrize("bad", [{"path": "  "}, {"caption": " "}])
def test_figure_rejects_blank_path_or_caption(bad):
    kwargs = {"path": "assets/x.png", "caption": "ok"}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        BenchmarkFigure(**kwargs)


def test_figure_alt_defaults_blank():
    assert BenchmarkFigure(path="assets/x.png", caption="c").alt == ""
