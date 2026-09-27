"""The runner's budget (plan 2b): the free-space margin, the clean stop, the estimate line."""  # noqa: E501

import json
import sys

import pytest
from test_run import FAKE, _case

from structbench.datagen import run as run_jobs


@pytest.fixture
def abaqus(tmp_path):
    fake = tmp_path / "fake_abaqus.py"
    fake.write_text(FAKE, encoding="utf-8")
    return [sys.executable, str(fake)]


def test_a_low_disk_stops_launching_and_leaves_pending_cases_pending(
    tmp_path, abaqus, monkeypatch
):
    sweep = tmp_path / "sweep"
    for k in range(3):
        _case(sweep, f"J-ok-{k}")
    free = iter([100.0, 0.1, 0.1])  # the first launch sees room, the next does not
    monkeypatch.setattr(run_jobs, "free_gb", lambda path: next(free))
    lines: list[str] = []
    results = run_jobs.run_sweep(
        sweep, abaqus, workers=1, min_free_gb=5.0, echo=lines.append
    )
    assert sorted(r.status for r in results) == [
        "completed",
        "not_launched",
        "not_launched",
    ]
    states = sorted(
        run_jobs.case_state(d) for d in sorted(sweep.iterdir()) if d.is_dir()
    )
    assert states == ["done", "pending", "pending"]
    assert not (sweep / run_jobs.LOCK_NAME).exists()
    rows = (sweep / "run_log.csv").read_text().splitlines()
    assert len(rows) == 2  # the header and the one run
    held = [line for line in lines if "below 5" in line and "nothing more" in line]
    assert len(held) == 1


def test_without_a_margin_nothing_is_checked(tmp_path, abaqus, monkeypatch):
    sweep = tmp_path / "sweep"
    _case(sweep, "J-ok-0")
    monkeypatch.setattr(run_jobs, "free_gb", lambda path: 0.1)
    results = run_jobs.run_sweep(sweep, abaqus, workers=1, echo=lambda s: None)
    assert [r.status for r in results] == ["completed"]


def test_main_exits_three_on_a_disk_stop_and_reads_the_margin_from_the_stamp(
    tmp_path, monkeypatch, capsys
):
    sweep = tmp_path / "sweep"
    _case(sweep, "J-ok-0")
    (sweep / "preflight").mkdir()
    budget = {
        "min_free_gb": 5.0,
        "wall_s_median": 30.0,
        "bytes_per_case": 2e8,
        "production_cases": 10,
    }
    (sweep / "preflight" / "stamp.json").write_text(json.dumps({"budget": budget}))
    monkeypatch.setattr(run_jobs, "free_gb", lambda path: 0.1)
    args = ["--sweep", str(sweep), "--workers", "1", "--abaqus", sys.executable]
    assert run_jobs.main(args) == 3
    out = capsys.readouterr().out
    assert "estimate from the preflight" in out and "margin 5 GB" in out
    assert "below 5" in out and "FAILED" not in out and "not_launched=1" in out
    assert run_jobs.case_state(sweep / "J-ok-0") == "pending"


def test_an_explicit_margin_overrides_the_stamp(tmp_path, monkeypatch, capsys):
    sweep = tmp_path / "sweep"
    _case(sweep, "J-ok-0")
    (sweep / "preflight").mkdir()
    stamp = {"budget": {"min_free_gb": 500.0, "wall_s_median": 1.0}}
    (sweep / "preflight" / "stamp.json").write_text(json.dumps(stamp))
    monkeypatch.setattr(run_jobs, "free_gb", lambda path: 10.0)
    args = ["--sweep", str(sweep), "--abaqus", sys.executable, "--min-free-gb", "1"]
    # the margin passes; the "solver" (python) rejects the job, which is exit 1
    assert run_jobs.main(args) == 1
    assert "not_launched" not in capsys.readouterr().out


def test_estimate_line_reads_the_budget():
    budget = {"min_free_gb": 20.0, "wall_s_median": 36.0, "bytes_per_case": 5e7}
    line = run_jobs.estimate_line({"budget": budget}, 500, 6, 170.3)
    assert "500 cases" in line and "0.8 h at 6 workers" in line
    assert "25.0 GB" in line and "170.3 GB" in line and "margin 20 GB" in line
