"""The preflight end to end against the fake solver (plan 2b, Task 7)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")

from structbench.datagen import definition, generate, preflight, template  # noqa: E402

FAKE = str(Path(__file__).resolve().with_name("fake_solver.py"))

PILOTS = """\
points = [
  { v0 = 1.0e5 },
  { v0 = 2.0e5 },
]
"""


def _git(ds: Path, message: str) -> None:
    git = ["git", "-C", str(ds), "-c", "user.name=t", "-c", "user.email=t@t"]
    if not (ds / ".git").exists():
        subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", message, "--allow-empty"], check=True)


def _dataset(tmp_path: Path) -> Path:
    """The shipped example, shrunk: two pilots, three levels, 21 frames."""
    ds = tmp_path / "ds"
    template.scaffold(ds, "e2e")
    toml = (ds / "dataset.toml").read_text(encoding="utf-8")
    toml = toml.replace("L = 40.0", "L = 10.0").replace(
        "n_intervals = 400", "n_intervals = 20"
    )
    head, _, tail = toml.partition("[splits.pilot]")
    _, _, rest = tail.partition("probe = true\n")
    toml = head + "[splits.pilot]\n" + PILOTS + "probe = true\n" + rest
    toml = (
        toml.replace('production = "1"', 'production = "2"')
        .replace('fine_cases = ["ACX-pilot-0001"]', 'fine_cases = ["ACX-pilot-0000"]')
        .replace("min_free_gb = 20.0", "min_free_gb = 0.5")
    )
    # the example itself names contact_force_global and the probe defaults
    assert 'contact_force_global = "reaction_force_2_reference_node"' in toml
    (ds / "dataset.toml").write_bytes(toml.encode("utf-8"))
    _git(ds, "e2e")
    return ds


def _preflight(ds: Path, work: Path) -> int:
    args = ["--dataset", str(ds), "--work-root", str(work), "--abaqus", sys.executable]
    args += ["--workers", "2", "--solver-args", FAKE, "--exporter-args", FAKE]
    return preflight.main(args)


def _stamp(work: Path) -> dict:
    return json.loads((work / "e2e" / "preflight" / "stamp.json").read_text("utf-8"))


def test_the_preflight_passes_on_the_fake_solver_and_the_gate_opens(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    rc = _preflight(ds, work)
    pre = work / "e2e" / "preflight"
    stamp = _stamp(work)
    verdicts = {name: s["verdict"] for name, s in stamp["steps"].items()}
    assert verdicts == dict.fromkeys(preflight.STEPS, "pass"), (
        pre / "report.md"
    ).read_text(encoding="utf-8")
    assert rc == 0 and stamp["passed"] is True
    assert sorted(stamp["cases"]["level"]) == [
        "ACX-pilot-0000-L1",
        "ACX-pilot-0000-L2",
        "ACX-pilot-0000-L4",
        "ACX-pilot-0001-L1",
        "ACX-pilot-0001-L2",
    ]
    assert stamp["cases"]["increment"] == [
        "ACX-pilot-0000-T0p25",
        "ACX-pilot-0001-T0p25",
    ]
    assert stamp["cases"]["frame"] == ["ACX-pilot-0000-F"]
    assert stamp["cases"]["conformance"] == ["ACX-pilot-0000-E"]
    assert (pre / "converge" / "convergence.json").is_file()
    assert (pre / "datacheck" / "report.md").is_file()
    assert (pre / "canonical" / "ACX-pilot-0000-F.h5").is_file()
    space = stamp["steps"]["space"]["detail"]["cases"]["ACX-pilot-0000-L2"]
    assert space["final_length"]["status"] == "monotone"
    assert space["final_length"]["error_at_production"] < 0.01
    duration = stamp["steps"]["duration"]["detail"]["pilots"]["ACX-pilot-0000-L2"]
    assert duration["separation_frame"] == 5 and duration["settling_frame"] <= 6
    budget = stamp["budget"]
    assert budget["production_cases"] == 8 and budget["bytes_per_case"] > 0
    # the gate opens for the production split, and the provenance names the stamp
    assert (
        generate.main(
            ["--dataset", str(ds), "--work-root", str(work), "--split", "train"]
        )
        == 0
    )
    prov = json.loads((work / "e2e" / "ACX-train-0000" / "provenance.json").read_text())
    assert prov["preflight"]["stamp_sha256"] == definition.file_sha256(
        pre / "stamp.json"
    )
    # the preflight cases carry their role and no gate record
    e = json.loads((pre / "ACX-pilot-0000-E" / "provenance.json").read_text())
    assert e["probe"]["role"] == "conformance" and e["preflight"] is None
    assert (
        "VARIABLE=ALL"
        in (pre / "ACX-pilot-0000-E" / "ACX-pilot-0000-E.inp").read_text()
    )


def test_a_second_run_resumes_and_rewrites_the_same_verdicts(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    assert _preflight(ds, work) == 0
    pre = work / "e2e" / "preflight"
    first = _stamp(work)
    records = {p.parent.name: p.read_bytes() for p in pre.glob("*/run.json")}
    (pre / "stamp.json").unlink()
    assert _preflight(ds, work) == 0
    second = _stamp(work)
    assert {p.parent.name: p.read_bytes() for p in pre.glob("*/run.json")} == records
    assert {n: s["verdict"] for n, s in second["steps"].items()} == {
        n: s["verdict"] for n, s in first["steps"].items()
    }


def test_a_changed_definition_is_refused_and_nothing_is_deleted(tmp_path, capsys):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    assert _preflight(ds, work) == 0
    pre = work / "e2e" / "preflight"
    before = sorted(p.parent.name for p in pre.glob("*/run.json"))
    problem = (ds / "problem.py").read_text(encoding="utf-8")
    problem = problem.replace("rod on a rigid wall", "rod on a rigid wall, revised")
    (ds / "problem.py").write_bytes(problem.encode("utf-8"))
    _git(ds, "revised heading")
    assert _preflight(ds, work) == 2
    assert "move it aside" in capsys.readouterr().err
    assert sorted(p.parent.name for p in pre.glob("*/run.json")) == before
    assert (
        pre / "stamp.json"
    ).is_file()  # the earlier stamp stands, for the old hashes
    assert (
        generate.main(
            ["--dataset", str(ds), "--work-root", str(work), "--split", "train"]
        )
        == 2
    )


def test_a_failing_step_writes_the_stamp_as_not_passed_and_the_gate_stays_shut(
    tmp_path, monkeypatch, capsys
):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    monkeypatch.setitem(os.environ, "FAKE_SOLVER_SETTLE", "slow")
    assert _preflight(ds, work) == 1
    stamp = _stamp(work)
    assert stamp["passed"] is False
    assert stamp["steps"]["duration"]["verdict"] == "fail"
    assert stamp["steps"]["space"]["verdict"] == "pass"
    report = (work / "e2e" / "preflight" / "report.md").read_text(encoding="utf-8")
    assert "**Not passed.**" in report and "duration (fail)" in report
    rc = generate.main(
        ["--dataset", str(ds), "--work-root", str(work), "--split", "train"]
    )
    assert rc == 2 and "did not pass" in capsys.readouterr().err


def test_changed_definition_same_decks_refused_and_sibling_shuts_gate(tmp_path, capsys):
    """Review findings 1 and 4."""
    ds, work = _dataset(tmp_path), tmp_path / "work"
    (ds / "helpers.py").write_bytes(b"SCALE = 1.0\n")
    problem = (ds / "problem.py").read_text(encoding="utf-8")
    problem = problem.replace(
        "from structbench.datagen.abaqus import deck",
        "from structbench.datagen.abaqus import deck\n"
        "from helpers import SCALE  # noqa: F401",
    )
    (ds / "problem.py").write_bytes(problem.encode("utf-8"))
    _git(ds, "with a sibling")
    assert _preflight(ds, work) == 0
    pre = work / "e2e" / "preflight"
    records = {p.parent.name: p.read_bytes() for p in pre.glob("*/run.json")}
    toml = (ds / "dataset.toml").read_text(encoding="utf-8")
    looser = toml.replace("tolerance = [0.01, 0.01]", "tolerance = [0.5, 0.5]")
    assert looser != toml
    (ds / "dataset.toml").write_bytes(looser.encode("utf-8"))
    _git(ds, "looser tolerances, same decks")
    assert _preflight(ds, work) == 2
    assert "another definition" in capsys.readouterr().err
    assert {p.parent.name: p.read_bytes() for p in pre.glob("*/run.json")} == records
    (ds / "dataset.toml").write_bytes(toml.encode("utf-8"))
    (ds / "helpers.py").write_bytes(b"SCALE = 2.0\n")
    _git(ds, "a sibling changed")
    rc = generate.main(
        ["--dataset", str(ds), "--work-root", str(work), "--split", "train"]
    )
    assert rc == 2 and "sibling" in capsys.readouterr().err
    assert _preflight(ds, work) == 2


def test_when_the_conformance_run_has_no_export_nothing_more_is_launched(
    tmp_path, monkeypatch
):
    """Review finding 11: fail-fast on a step that did not pass, not only on fail."""
    ds, work = _dataset(tmp_path), tmp_path / "work"
    monkeypatch.setitem(os.environ, "FAKE_SOLVER_NO_EXPORT", "1")
    assert _preflight(ds, work) == 1
    pre = work / "e2e" / "preflight"
    ran = sorted(p.parent.name for p in pre.glob("*/run.json"))
    assert ran == ["ACX-pilot-0000-E", "ACX-pilot-0000-L2", "ACX-pilot-0001-L2"]
    stamp = _stamp(work)
    assert stamp["steps"]["conformance"]["verdict"] == "not_assessable"
    assert stamp["steps"]["space"]["verdict"] == "not_assessable"
    assert stamp["passed"] is False
