"""structbench-datagen: one entry point, one sub-command per stage."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")

from structbench.datagen import cli  # noqa: E402

STAGES = [
    "new",
    "check",
    "preflight",
    "generate",
    "run",
    "follow",
    "export",
    "convert",
    "verify",
    "converge",
    "archive",
    "card",
]


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_answers_help(stage, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main([stage, "--help"])
    assert stop.value.code == 0
    assert stage in capsys.readouterr().out


def test_the_old_stage_name_points_to_verify(capsys):
    # ADR-0072: validate means comparison with experiment; the stage is verify
    assert cli.main(["validate", "--help"]) == 2
    err = capsys.readouterr().err
    assert "verify" in err and "ADR-0072" in err


def test_an_unknown_stage_is_refused(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["frobnicate"])
    assert stop.value.code == 2


def test_new_then_check_round_trip(tmp_path):
    assert cli.main(["new", str(tmp_path / "d")]) == 0
    assert cli.main(["check", str(tmp_path / "d")]) == 0


def test_export_builds_the_abaqus_python_command(tmp_path):
    command = cli.export_command("abaqus", tmp_path, ["A-1", "A-2"])
    assert command[:2] == ["abaqus", "python"]
    assert Path(command[2]).name == "odb_export.py" and Path(command[2]).is_file()
    assert command[3:] == ["--sweep", str(tmp_path), "--cases", "A-1", "A-2"]


def test_a_missing_solver_is_the_environments_stop_not_a_refusal(tmp_path, capsys):
    # the design's exit codes: 2 refused by a gate, 3 stopped by the environment
    args = ["export", "--sweep", str(tmp_path), "--abaqus", "no-such-solver-xyz"]
    rc = cli.main(args)
    assert rc == 3
    assert "no-such-solver-xyz" in capsys.readouterr().err
    args = ["run", "--sweep", str(tmp_path), "--abaqus", "no-such-solver-xyz"]
    assert cli.main(args) == 3


def test_without_scipy_the_command_names_the_extra_instead_of_a_traceback():
    import subprocess

    code = (
        "import sys; sys.modules['scipy'] = None\nfrom structbench.datagen import cli"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 1
    assert "structbench[datagen]" in proc.stderr and "Traceback" not in proc.stderr


def test_new_in_the_current_directory_takes_the_directorys_name(
    tmp_path, monkeypatch, capsys
):
    target = tmp_path / "my_set"
    target.mkdir()
    monkeypatch.chdir(target)
    assert cli.main(["new", "."]) == 0
    toml = (target / "dataset.toml").read_text(encoding="utf-8")
    assert 'name = "my_set"' in toml
    assert cli.main(["check", "."]) == 0


def test_export_runs_a_stub_solver(tmp_path):
    stub = tmp_path / "stub.py"
    stub.write_text("import sys; print(' '.join(sys.argv[1:])); sys.exit(0)\n")
    rc = cli.main(
        ["export", "--sweep", str(tmp_path), "--abaqus", sys.executable]
        + ["--exporter-args", str(stub)]
    )
    assert rc == 0
