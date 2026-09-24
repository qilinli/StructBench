"""Tests for the Abaqus runner (ADR-0069) against a fake solver. No Abaqus."""

import csv
import json
import sys
import time

import abaqus_paths  # noqa: F401
import pytest
import run_jobs

FAKE = r"""
import pathlib, sys, time
args = sys.argv[1:]
if args[0] == "terminate":
    sys.exit(0)
job = next(a.split("=", 1)[1] for a in args if a.startswith("job="))
here = pathlib.Path.cwd()
(here / f"{job}.lck").write_text("")
mode = job.split("-")[1]
done = " THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n"
counts = "      0 ERROR MESSAGES\n      0 WARNING MESSAGES\n"
fatal = " THE PROGRAM HAS DISCOVERED     1 FATAL ERRORS\n"
if mode == "hang":
    time.sleep(60)
elif mode == "ok":
    (here / f"{job}.sta").write_text(" Abaqus/Explicit 2025\n" + done)
    (here / f"{job}.msg").write_text(" Abaqus 2025\n" + counts)
    (here / f"{job}.dat").write_text(" Abaqus 2025\n")
elif mode == "reject":
    (here / f"{job}.dat").write_text(" Abaqus 2025\n" + fatal)
(here / f"{job}.lck").unlink()
sys.exit(0 if mode == "ok" else 1)
"""


@pytest.fixture
def abaqus(tmp_path):
    fake = tmp_path / "fake_abaqus.py"
    fake.write_text(FAKE, encoding="utf-8")
    return [sys.executable, str(fake)]


def _case(sweep, case_id, split="s"):
    d = sweep / case_id
    d.mkdir(parents=True)
    (d / f"{case_id}.inp").write_text("*HEADING\n", encoding="utf-8")
    (d / "provenance.json").write_text(json.dumps({"units": "t-mm-s", "split": split}))
    return d


def test_completed_runs_record_and_skip_on_rerun(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    for k in range(3):
        _case(sweep, f"C-ok-{k}")
    results = run_jobs.run_sweep(sweep, abaqus, workers=2, echo=lambda s: None)
    assert sorted(r.status for r in results) == ["completed"] * 3
    record = json.loads((sweep / "C-ok-0/run.json").read_text())
    assert record["abaqus_version"] == "2025" and record["n_errors"] == 0
    assert record["command"][0].startswith("python")  # the name only, never a path
    rows = list(csv.DictReader((sweep / "run_log.csv").open(encoding="utf-8")))
    assert len(rows) == 3
    assert run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None) == []


def test_rejected_job_is_error_and_retried_only_on_request(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    _case(sweep, "C-reject-0")
    assert run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None)[0].status == "error"
    assert run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None) == []
    again = run_jobs.run_sweep(sweep, abaqus, retry_failed=True, echo=lambda s: None)
    assert again[0].status == "error"
    assert (sweep / "C-reject-0/attempts/1/run.json").is_file()


def test_interrupted_case_is_moved_aside_and_rerun(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    d = _case(sweep, "C-ok-9")
    (d / "C-ok-9.lck").write_text("")
    (d / "C-ok-9.sta").write_text("partial")
    assert run_jobs.case_state(d) == "interrupted"
    result = run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None)[0]
    assert result.status == "completed"
    assert (d / "attempts/1/C-ok-9.sta").read_text() == "partial"


def test_timeout_stops_the_job(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    _case(sweep, "C-hang-0")
    result = run_jobs.run_sweep(sweep, abaqus, timeout=2, echo=lambda s: None)[0]
    assert result.status == "timeout" and result.wall_s < 30


def test_split_filter_and_limit(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    _case(sweep, "C-ok-0", split="a")
    _case(sweep, "C-ok-1", split="b")
    _case(sweep, "C-ok-2", split="b")
    out = run_jobs.run_sweep(sweep, abaqus, splits=["b"], limit=1, echo=lambda s: None)
    assert [r.case_id for r in out] == ["C-ok-1"]


def test_a_held_lock_refuses_a_second_runner(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    _case(sweep, "C-ok-0")
    (sweep / ".runner.lock").write_text("123")
    with pytest.raises(RuntimeError, match=r"\.runner\.lock"):
        run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None)


class _Stop(Exception):
    """Stands in for Ctrl+C or any failure in the result loop."""


def _raise(line):
    raise _Stop


def test_an_interruption_cancels_queued_jobs_and_releases_the_lock(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    for case_id in ("A-ok-0", "B-hang-1", "B-hang-2"):
        _case(sweep, case_id)
    started = time.monotonic()
    with pytest.raises(_Stop):
        run_jobs.run_sweep(sweep, abaqus, workers=1, echo=_raise)
    assert time.monotonic() - started < 30  # the hanging job did not run out
    assert not (sweep / "B-hang-2" / "runner.log").exists()  # never launched
    assert run_jobs.case_state(sweep / "B-hang-1") in ("pending", "interrupted")
    assert not (sweep / ".runner.lock").exists()


def test_an_interruption_stops_a_running_job_and_it_reruns_by_default(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    _case(sweep, "A-ok-0")
    hung = _case(sweep, "B-hang-1")
    started = time.monotonic()
    with pytest.raises(_Stop):
        run_jobs.run_sweep(sweep, abaqus, workers=2, echo=_raise)
    assert time.monotonic() - started < 30
    assert json.loads((hung / "run.json").read_text())["status"] == "stopped"
    assert run_jobs.case_state(hung) == "interrupted"


def test_bad_provenance_units_are_refused_before_any_launch(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    d = _case(sweep, "C-ok-0")
    (d / "provenance.json").write_text(json.dumps({"units": "t-mm-sec", "split": "s"}))
    with pytest.raises(ValueError, match="t-mm-sec"):
        run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None)
    assert not (d / "runner.log").exists()
    assert not (sweep / ".runner.lock").exists()


def test_a_reader_failure_still_records_the_run(tmp_path, abaqus, monkeypatch):
    sweep = tmp_path / "sweep"
    _case(sweep, "C-ok-0")

    def broken(**kwargs):
        raise RuntimeError("reader bug")

    monkeypatch.setattr(run_jobs, "read_abaqus_run_evidence", broken)
    result = run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None)[0]
    record = json.loads((sweep / "C-ok-0/run.json").read_text())
    assert result.status == record["status"] == "unrecognised"
    assert "RuntimeError" in record["error"]


def test_whitespace_in_the_sweep_path_is_refused(tmp_path, abaqus):
    sweep = tmp_path / "One Drive" / "sweep"
    _case(sweep, "C-ok-0")
    with pytest.raises(ValueError, match="whitespace"):
        run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None)
