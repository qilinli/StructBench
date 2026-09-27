"""converge: what the plan 2a review found -- the declared production level across
roots, zero QoIs, level order from JSON, word labels, exit codes."""

import json

import pytest
from conftest import MINIMAL_TOML, write_definition
from test_converge import BASE, PROBLEM, TOML, Sweep, _three_levels

pytest.importorskip("scipy")

from structbench.datagen import converge, definition  # noqa: E402


@pytest.fixture
def toy(tmp_path):
    ds = write_definition(tmp_path / "ds", toml=TOML, problem=PROBLEM)
    return ds, definition.load_definition(ds), definition.load_problem(ds)


def test_the_production_level_is_the_declared_one_even_with_an_older_production_root(
    tmp_path, toy
):
    _, defn, problem = toy
    old = Sweep(tmp_path / "old", problem).run("TOY-train-0000", "train", "1")
    new = Sweep(tmp_path / "new", problem)
    new.run("TOY-train-0000", "train", "2").run("TOY-conv-0000", "conv", "4")
    (entry,) = converge.converge(defn, problem, [new.root, old.root])["cases"]
    assert entry["production_level"] == "2"
    x = entry["extrapolation"]["length"]
    assert x["error_vs_extrapolated"]["2"] == pytest.approx(0.01, rel=1e-2)
    assert any("production" in n and "1" in n for n in entry["notes"])


def test_a_probe_standing_at_the_production_level_is_noted(tmp_path, toy):
    _, defn, problem = toy
    s = Sweep(tmp_path / "runs", problem)
    s.run("TOY-train-0000", "train", "2", canonical=False)  # not converted yet
    s.run("TOY-conv-0000", "conv", "1").run("TOY-conv-0001", "conv", "2")
    s.run("TOY-conv-0002", "conv", "4")
    (entry,) = converge.converge(defn, problem, [s.root])["cases"]
    assert entry["levels"]["2"] == "TOY-conv-0001" and entry["production_level"] == "2"
    assert any("probe" in n and "production level" in n for n in entry["notes"])


def test_a_zero_qoi_is_a_note_not_a_crash(tmp_path):
    zero = PROBLEM.replace(
        'return {"length": float(case.response.node["displacement"][-1, :, 0].max())}',
        'return {"length": 0.0}',
    )
    ds = write_definition(tmp_path / "ds", toml=TOML, problem=zero)
    defn, problem = definition.load_definition(ds), definition.load_problem(ds)
    s = _three_levels(tmp_path, problem)
    record = converge.converge(defn, problem, [s.root])
    (entry,) = record["cases"]
    assert entry["extrapolation"]["length"]["status"] == "flat"
    assert entry["extrapolation"]["length"]["coarsest_vs_finest"] is None
    converge.to_json(record)  # no NaN reaches the JSON


def test_markdown_from_the_json_keeps_the_numeric_level_order(tmp_path, toy):
    _, defn, problem = toy
    wide = MINIMAL_TOML.replace('production = "1"', 'production = "2"').replace(
        'pilot = ["1", "2"]', 'pilot = ["1", "2", "4", "8", "16"]'
    )
    wide = wide.replace(
        "[splits.pilot]",
        "[splits.conv]\npoints = [{ L = 1.0, v0 = 10.0 }]\nprobe = true\n\n"
        "[splits.pilot]",
    )
    ds = write_definition(tmp_path / "ds2", toml=wide, problem=PROBLEM)
    defn, problem = definition.load_definition(ds), definition.load_problem(ds)
    s = Sweep(tmp_path / "runs", problem).run("TOY-train-0000", "train", "2")
    for k, level in enumerate(("1", "4", "8", "16")):
        s.run(f"TOY-conv-{k:04d}", "conv", level)
    record = converge.converge(defn, problem, [s.root])
    assert record["level_order"] == ["1", "2", "4", "8", "16"]
    from_memory = converge.render_markdown(record)
    from_json = converge.render_markdown(json.loads(converge.to_json(record)))
    assert from_memory == from_json
    assert "| level 1 | level 2 | level 4 | level 8 |" in from_json


def test_word_labels_follow_the_pilot_listing_and_get_no_extrapolation(tmp_path):
    words = (
        MINIMAL_TOML.replace('production = "1"', 'production = "medium"')
        .replace('pilot = ["1", "2"]', 'pilot = ["coarse", "medium", "fine"]')
        .replace(
            "[splits.pilot]",
            "[splits.conv]\npoints = [{ L = 1.0, v0 = 10.0 }]\nprobe = true\n\n"
            "[splits.pilot]",
        )
    )
    problem = PROBLEM.replace(
        'k = int(params.get("refine", 1))',
        'w = params.get("refine", "coarse")\n'
        '    k = {"coarse": 1, "medium": 2, "fine": 4}[w]'
        " if isinstance(w, str) else int(w)",
    )
    ds = write_definition(tmp_path / "ds", toml=words, problem=problem)
    defn, prob = definition.load_definition(ds), definition.load_problem(ds)
    s = Sweep(tmp_path / "runs", prob)
    s.run("TOY-train-0000", "train", "medium", elements_across=2, h_level=2)
    s.run("TOY-conv-0000", "conv", "coarse", elements_across=1, h_level=1)
    s.run("TOY-conv-0001", "conv", "fine", elements_across=4, h_level=4)
    record = converge.converge(defn, prob, [s.root])
    (entry,) = record["cases"]
    assert record["level_order"] == ["coarse", "medium", "fine"]
    assert entry["extrapolation"]["length"] is None
    assert any("numeric" in n for n in entry["notes"])
    assert set(entry["fields"]["displacement"]) == {
        "coarse",
        "medium",
    }  # fine is the reference


def test_the_constant_ratio_test_looks_at_the_three_finest_levels(tmp_path, toy):
    _, defn, problem = toy
    wide = TOML.replace(
        'pilot = ["1", "2", "4"]', 'pilot = ["1", "3", "6", "12"]'
    ).replace('production = "2"', 'production = "3"')
    ds = write_definition(tmp_path / "ds3", toml=wide, problem=PROBLEM)
    defn, problem = definition.load_definition(ds), definition.load_problem(ds)
    s = Sweep(tmp_path / "runs", problem).run("TOY-train-0000", "train", "3")
    s.run("TOY-conv-0000", "conv", "1").run("TOY-conv-0001", "conv", "6")
    s.run("TOY-conv-0002", "conv", "12")
    (entry,) = converge.converge(defn, problem, [s.root])["cases"]
    x = entry["extrapolation"]["length"]
    assert (
        x is not None
        and x["levels"] == ["3", "6", "12"]
        and x["ratio"] == pytest.approx(2.0)
    )


def test_exit_codes_refuse_a_missing_sweep_and_report_refusals(tmp_path, toy, capsys):
    ds, _, problem = toy
    missing = tmp_path / "nowhere"
    assert converge.main(["--dataset", str(ds), "--sweep", str(missing)]) == 2
    assert not missing.exists()
    s = _three_levels(tmp_path, problem)
    other = {**BASE, "L": 2.0}
    s.run("TOY-train-0001", "train", "2", other).run(
        "TOY-conv-0002", "conv", "1", other
    )
    s.run(
        "TOY-conv-0003", "conv", "4", other, elements_across=7
    )  # refused: does not nest
    assert converge.main(["--dataset", str(ds), "--sweep", str(s.root)]) == 1
