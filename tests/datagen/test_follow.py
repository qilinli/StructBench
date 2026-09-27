"""follow (plan 2b): export and convert finished cases while a run proceeds."""

import json
import sys
from pathlib import Path

import pytest
from test_convert import _case

from structbench.datagen import follow
from structbench.datagen import run as run_jobs

_FIXTURE = Path(__file__).resolve().parents[1] / "core" / "test_abaqus_adapter.py"
#: In the exporter's place: writes the adapter fixture's export for each case named.
STUB = f"""
import importlib.util, sys
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("fx", {str(_FIXTURE)!r})
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)
args = sys.argv[1:]
sweep = Path(args[args.index("--sweep") + 1])
cases = args[args.index("--cases") + 1:] if "--cases" in args else []
for cid in cases:
    np.savez(sweep / cid / (cid + ".npz"), **fx._with(fx._arrays(), ALLCD=0.0))
sys.exit(0)
"""


@pytest.fixture
def stub(tmp_path):
    path = tmp_path / "stub_exporter.py"
    path.write_text(STUB, encoding="utf-8")
    return [str(path)]


def _finish(folder: Path) -> None:
    run = {"status": "completed", "end_utc": "2026-09-24T16:07:59+00:00"}
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")


def _quiet(**kw):
    return dict(echo=lambda s: None, sleep=lambda s: None, **kw)


def test_follow_exports_then_converts_and_stops_when_the_runner_is_gone(tmp_path, stub):
    sweep = tmp_path / "toy_sweep"
    _case(sweep, "T-0000", npz=False)
    _case(sweep, "T-0001", npz=False)
    (sweep / "T-0001" / "run.json").unlink()  # still running
    (sweep / run_jobs.LOCK_NAME).write_text("1")
    events: list[float] = []

    def sleep(seconds):
        events.append(seconds)
        if len(events) == 1:
            _finish(sweep / "T-0001")  # its run.json appears
        else:
            (sweep / run_jobs.LOCK_NAME).unlink()  # the runner is gone

    lines: list[str] = []
    report = follow.follow(
        sweep,
        sys.executable,
        interval=7.0,
        exporter_args=stub,
        echo=lines.append,
        sleep=sleep,
    )
    assert report.exported == ["T-0000", "T-0001"]
    assert report.converted == ["T-0000", "T-0001"]
    assert report.failed == {} and report.rounds == 3 and events == [7.0, 7.0]
    assert (sweep / "canonical" / "T-0001.h5").is_file()
    assert len(lines) == 3 and lines[0].startswith("round 1:")


def test_once_does_one_round_and_reports_conversion_failures(tmp_path, stub):
    sweep = tmp_path / "toy_sweep"
    _case(sweep, "T-0000", npz=False)
    bad = _case(sweep, "T-0001", npz=False)
    deck = (bad / "T-0001.inp").read_text(encoding="utf-8")
    deck = deck.replace("*MATERIAL, NAME=M\n", "*INCLUDE, INPUT=m.inp\n")
    (bad / "T-0001.inp").write_text(deck, encoding="utf-8")
    report = follow.follow(
        sweep, sys.executable, once=True, exporter_args=stub, **_quiet()
    )
    assert report.rounds == 1 and report.exported == ["T-0000", "T-0001"]
    assert report.converted == ["T-0000"]
    assert "SchemaError" in report.failed["T-0001"]
    args = ["--sweep", str(sweep), "--abaqus", sys.executable, "--once"]
    assert follow.main([*args, "--exporter-args", *stub]) == 1


def test_an_exporter_failure_is_reported_and_follow_stops_without_a_runner(tmp_path):
    sweep = tmp_path / "toy_sweep"
    _case(sweep, "T-0000", npz=False)
    boom = tmp_path / "boom.py"
    boom.write_text("import sys; sys.exit(3)\n", encoding="utf-8")
    report = follow.follow(sweep, sys.executable, exporter_args=[str(boom)], **_quiet())
    assert report.exported == [] and report.converted == []
    assert "returned 3" in report.failed["export"]
    assert report.rounds == 1  # no runner, no progress: it does not spin


def test_split_filter_and_nothing_to_do(tmp_path, stub):
    sweep = tmp_path / "toy_sweep"
    _case(sweep, "T-0000", npz=False, split="train")
    _case(sweep, "T-0001", npz=False, split="probe")
    first = follow.follow(
        sweep,
        sys.executable,
        once=True,
        splits=["train"],
        exporter_args=stub,
        **_quiet(),
    )
    assert first.exported == ["T-0000"] and first.converted == ["T-0000"]
    again = follow.follow(
        sweep,
        sys.executable,
        once=True,
        splits=["train"],
        exporter_args=stub,
        **_quiet(),
    )
    assert again.exported == [] and again.converted == [] and again.rounds == 1
    assert not (sweep / "T-0001" / "T-0001.npz").exists()


def test_main_without_a_solver_exits_two(tmp_path, capsys):
    args = ["--sweep", str(tmp_path), "--abaqus", "no-such-solver-xyz", "--once"]
    assert follow.main(args) == 2
    assert "no-such-solver-xyz" in capsys.readouterr().err
