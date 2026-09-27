"""converge: a probe split may anchor groups that have no production run (plan 2b)."""

import pytest
from conftest import write_definition
from test_converge import PROBLEM, TOML, Sweep

pytest.importorskip("scipy")

from structbench.datagen import converge, definition  # noqa: E402


@pytest.fixture
def toy(tmp_path):
    ds = write_definition(tmp_path / "ds", toml=TOML, problem=PROBLEM)
    return ds, definition.load_definition(ds), definition.load_problem(ds)


def _probe_only(tmp_path, problem):
    s = Sweep(tmp_path / "runs", problem)
    s.run("TOY-conv-0000-L1", "conv", "1").run("TOY-conv-0000-L2", "conv", "2")
    s.run("TOY-conv-0000-L4", "conv", "4")
    return s


def test_without_an_anchor_probe_only_groups_are_counted_not_compared(tmp_path, toy):
    _, defn, problem = toy
    record = converge.converge(defn, problem, [_probe_only(tmp_path, problem).root])
    assert record["cases"] == [] and record["anchor"] is None
    assert any("no production run" in n and "--anchor" in n for n in record["notes"])


def test_the_anchor_splits_production_level_run_stands_for_production(tmp_path, toy):
    _, defn, problem = toy
    root = _probe_only(tmp_path, problem).root
    record = converge.converge(defn, problem, [root], anchor="conv")
    (entry,) = record["cases"]
    assert entry["case_id"] == "TOY-conv-0000-L2"
    assert entry["production_level"] == "2"
    assert entry["extrapolation"]["length"]["status"] == "monotone"
    assert entry["notes"] == []
    assert record["anchor"] == "conv" and record["notes"] == []
    assert "conv" in converge.render_markdown(record)


def test_a_true_production_run_still_wins_over_the_anchor(tmp_path, toy):
    _, defn, problem = toy
    s = _probe_only(tmp_path, problem).run("TOY-train-0000", "train", "2")
    (entry,) = converge.converge(defn, problem, [s.root], anchor="conv")["cases"]
    assert entry["case_id"] == "TOY-train-0000"
    assert entry["levels"]["2"] == "TOY-train-0000"


def test_an_anchor_run_at_another_level_does_not_anchor(tmp_path, toy):
    _, defn, problem = toy
    s = Sweep(tmp_path / "runs", problem)
    s.run("TOY-conv-0000-L1", "conv", "1").run("TOY-conv-0000-L4", "conv", "4")
    record = converge.converge(defn, problem, [s.root], anchor="conv")
    assert record["cases"] == []
    assert any("no production run" in n for n in record["notes"])


def test_cases_restricts_the_pairing(tmp_path, toy):
    _, defn, problem = toy
    root = _probe_only(tmp_path, problem).root
    record = converge.converge(
        defn,
        problem,
        [root],
        anchor="conv",
        cases={"TOY-conv-0000-L1", "TOY-conv-0000-L2"},
    )
    (entry,) = record["cases"]
    assert set(entry["levels"]) == {"1", "2"}
    assert entry["extrapolation"]["length"] is None


def test_the_cli_takes_the_anchor(tmp_path, toy):
    ds, _, problem = toy
    root = _probe_only(tmp_path, problem).root
    args = ["--dataset", str(ds), "--sweep", str(root)]
    assert converge.main(args) == 1  # counted, not compared: a finding
    assert converge.main([*args, "--anchor", "conv"]) == 0
    assert converge.main([*args, "--anchor", "nowhere"]) == 2
