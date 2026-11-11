"""A benchmark definition loaded from a directory outside the package.

The run config's benchmark field may name a directory holding
``benchmark.py`` that exports ``SPEC``; such a benchmark is unadmitted (no
registry row), and the spec carries where it came from (ADR-0032 note,
2026-11-06).
"""

import textwrap

import pytest

from structbench.benchmarks import (
    BenchmarkSpec,
    available_benchmarks,
    get_benchmark,
    resolve_benchmark,
)

DEFINITION = textwrap.dedent(
    '''
    """A tiny benchmark definition (test fixture)."""
    from structbench.benchmarks import BenchmarkSpec
    from structbench.benchmarks.card import BenchmarkCard
    import horizons

    SPLITS = {"train": ("a-0", "a-1"), "val": ("a-2",)}
    CARD = BenchmarkCard(
        name="ByPath",
        version="0.0",
        description="fixture",
        provenance="synthetic",
        data_license="n/a",
        solver="none",
        discretisation="FEM",
        materials=("synthetic",),
        erosion=False,
        loading="none",
        source_units="kg-m-s",
        geometry="synthetic",
        n_cases=3,
        splits={k: len(v) for k, v in SPLITS.items()},
        task="fixture",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=(),
        fields=("node/displacement",),
        particles_per_case="4",
        n_frames=8,
        output_dt_ms=1.0,
        input_frames=2,
        protocol_rationale="fixture",
    )
    SPEC = BenchmarkSpec(
        card=CARD,
        splits=SPLITS,
        eval_splits=("val",),
        aux_field="von_mises_stress",
        dataset_id="by-path",
        scored_frames=horizons.scored,
    )
    '''
)
HORIZONS = "def scored(case_id):\n    return 4 + int(case_id.rsplit('-', 1)[1])\n"


def _definition(tmp_path):
    d = tmp_path / "defn"
    d.mkdir()
    (d / "benchmark.py").write_text(DEFINITION, encoding="utf-8")
    (d / "horizons.py").write_text(HORIZONS, encoding="utf-8")
    return d


def test_get_benchmark_loads_a_definition_directory(tmp_path):
    d = _definition(tmp_path)
    spec = get_benchmark(str(d))
    assert isinstance(spec, BenchmarkSpec)
    assert spec.card.name == "ByPath"
    assert spec.source is not None
    assert spec.source.path == d.resolve()
    assert list(spec.source.files) == ["benchmark.py", "horizons.py"]
    assert all(len(h) == 64 for h in spec.source.files.values())
    # Unadmitted: the registry lists only the shipped benchmarks.
    assert "ByPath" not in available_benchmarks()
    assert str(d) not in available_benchmarks()


def test_resolve_benchmark_returns_none_for_neither(tmp_path):
    assert resolve_benchmark("no_such_benchmark") is None
    assert resolve_benchmark(str(tmp_path)) is None  # a directory, no benchmark.py
    assert resolve_benchmark("taylor_impact_2d") is not None


def test_get_benchmark_unknown_name_points_at_both_routes(tmp_path):
    with pytest.raises(KeyError, match="benchmark.py"):
        get_benchmark("no_such_benchmark")
    with pytest.raises(KeyError, match="benchmark.py"):
        get_benchmark(str(tmp_path))


def test_definition_must_export_a_spec(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "benchmark.py").write_text("SPEC = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SPEC"):
        get_benchmark(str(d))
    (d / "benchmark.py").write_text("X = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SPEC"):
        get_benchmark(str(d))


def test_scored_frames_may_depend_on_the_case(tmp_path):
    spec = get_benchmark(str(_definition(tmp_path)))
    assert callable(spec.scored_frames)
    assert spec.scored_frames_fixed is None
    assert spec.scored_frames_for("a-1") == 5
    assert spec.scored_frames_for("a-2") == 6
    notch = get_benchmark("notch_beam_2d_impact")
    assert notch.scored_frames_fixed == 250
    assert notch.scored_frames_for("anything") == 250
    taylor = get_benchmark("taylor_impact_2d")
    assert taylor.scored_frames_fixed is None
    assert taylor.scored_frames_for("anything") is None


def test_per_case_scored_frames_are_range_checked(tmp_path):
    spec = get_benchmark(str(_definition(tmp_path)))
    # 4 + 9 = 13 > n_frames = 8
    with pytest.raises(ValueError, match="scored_frames"):
        spec.scored_frames_for("a-9")


def test_a_per_case_bound_is_checked_for_every_case_when_the_definition_loads(
    tmp_path,
):
    """ADR-0078: a bound out of range for one case of one split fails the
    load, naming the case, not the first stage that reads that case."""
    d = _definition(tmp_path)
    # a-0 and a-1 stay in range (4, 5); the VAL case a-2 gets 9 > n_frames.
    (d / "horizons.py").write_text(
        "def scored(case_id):\n"
        "    return 9 if case_id == 'a-2' else 4 + int(case_id.rsplit('-', 1)[1])\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"scored_frames\('a-2'\)"):
        get_benchmark(str(d))


def test_scored_share_is_the_declared_span_over_the_stored_frames(tmp_path):
    """ADR-0078: the pooled share of stored frames inside the declared span,
    from the declaration alone; ``None`` without a span."""
    spec = get_benchmark(str(_definition(tmp_path)))
    # bounds 4, 5, 6 with input_frames 2 over three cases of 8 frames.
    assert spec.scored_share() == pytest.approx((2 + 3 + 4) / 24)
    notch = get_benchmark("notch_beam_2d_impact")
    assert notch.scored_share() == pytest.approx((250 - 6) / 502)
    assert get_benchmark("taylor_impact_2d").scored_share() is None
