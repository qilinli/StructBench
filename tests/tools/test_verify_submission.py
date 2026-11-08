"""tools/verify_submission.py (ADR-0074 clause 6; open-evaluation plan §7):
the container command, the budget, and the reproduction comparisons, with
the container runner replaced by a stand-in."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from structbench.core import (
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
    write_case,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "verify_submission", REPO_ROOT / "tools" / "verify_submission.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_submission"] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _prediction(case_id: str, scale: float = 1.0) -> Case:
    t = np.arange(3, dtype=float) * 1e-6
    disp = (
        np.ones((3, 2, 2), dtype=np.float32) * np.arange(1, 4)[:, None, None]
    ) * scale
    return Case(
        metadata=Metadata(case_id=case_id, dimension=2),
        nodes=Nodes(coords=np.zeros((2, 2)), node_id=np.array([1, 2])),
        elements={
            "sph": ElementBlock(
                connectivity=np.array([[0], [1]]),
                element_id=np.array([1, 2]),
                part_id=np.array([1, 1]),
            )
        },
        materials=[Material(1, "M", {}, None)],
        response=Response(
            time=t,
            node={"displacement": disp.astype(np.float32)},
            element={"sph": {"von_mises_stress": np.ones((3, 2), np.float32) * scale}},
        ),
    )


def _write(folder: Path, scale: float = 1.0) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for cid in ("T-20-80-130", "T-20-80-170"):
        write_case(_prediction(cid, scale), folder / f"{cid}.h5")
    return folder


def test_container_command_follows_the_contract(tmp_path):
    cmd = tool.container_command("ghcr.io/x/y:1", tmp_path / "c", tmp_path / "o")
    assert cmd[:5] == ["docker", "run", "--rm", "--network", "none"]
    assert cmd[-6:] == [
        "structbench-submit",
        "predict",
        "--cases",
        "/cases",
        "--out",
        "/out",
    ]
    assert any(v.endswith(":/cases:ro") for v in cmd)


def test_run_container_records_the_run_and_a_timeout(tmp_path):
    def ok(command, **kwargs):
        assert kwargs["timeout"] == 12.5
        return SimpleNamespace(returncode=0, stdout="done", stderr="")

    run = tool.run_container(
        "img", tmp_path / "c", tmp_path / "o", budget_seconds=12.5, runner=ok
    )
    assert (
        run["returncode"] == 0 and not run["timed_out"] and "done" in run["output_tail"]
    )
    assert (tmp_path / "o").is_dir()

    def slow(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    run = tool.run_container(
        "img", tmp_path / "c", tmp_path / "o2", budget_seconds=1.0, runner=slow
    )
    assert run["timed_out"] and run["returncode"] is None


def test_compare_predictions_verifies_a_reproduction_and_not_a_departure(tmp_path):
    submitted = _write(tmp_path / "submitted")
    same = _write(tmp_path / "same")
    result = tool.compare_predictions(submitted, same)
    assert result["verified"] is True
    assert all(v == 0.0 for row in result["cases"].values() for v in row.values())
    off = _write(tmp_path / "off", scale=1.01)
    result = tool.compare_predictions(submitted, off)
    assert result["verified"] is False
    assert result["cases"]["T-20-80-130"]["node/displacement"] > tool.TOLERANCE
    (tmp_path / "off" / "T-20-80-170.h5").unlink()
    result = tool.compare_predictions(submitted, tmp_path / "off")
    assert result["cases"]["T-20-80-170"] == {"status": "missing"}
    assert tool.compare_predictions(tmp_path / "empty", same)["verified"] is False


def test_compare_metrics_reads_the_two_score_files():
    a = {"mean": {"rollout_rel_l2_disp": 0.0200, "rollout_rel_l2_aux": 0.2}}
    b = {"mean": {"rollout_rel_l2_disp": 0.0200001, "rollout_rel_l2_aux": 0.2}}
    assert tool.compare_metrics(a, b)["verified"] is True
    c = {"mean": {"rollout_rel_l2_disp": 0.021, "rollout_rel_l2_aux": 0.2}}
    assert tool.compare_metrics(a, c)["verified"] is False
    d = {"mean": {"rollout_rel_l2_disp": 0.02}}
    out = tool.compare_metrics(a, d)
    assert out["metrics"]["rollout_rel_l2_aux"] == "missing" and not out["verified"]


def test_main_runs_the_stand_in_and_writes_the_verification(tmp_path):
    submitted = _write(tmp_path / "submitted")
    cases = tmp_path / "cases"
    cases.mkdir()

    def copying_runner(command, **kwargs):
        # the stand-in "container" writes the submitted files to the /out mount
        mount = next(arg for arg in command if arg.endswith(":/out"))
        out = Path(mount[: -len(":/out")])
        for p in submitted.glob("*.h5"):
            shutil.copy(p, out / p.name)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    out = tmp_path / "verification.json"
    rc = tool.main(
        [
            "--image",
            "img",
            "--cases",
            str(cases),
            "--submission",
            str(submitted),
            "--out",
            str(out),
            "--work",
            str(tmp_path / "work"),
        ],
        runner=copying_runner,
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0 and report["verified"] is True
    assert report["predictions"]["cases"]["T-20-80-130"]["node/displacement"] == 0.0
    # dry run prints the command and writes nothing
    rc = tool.main(
        [
            "--image",
            "img",
            "--cases",
            str(cases),
            "--submission",
            str(submitted),
            "--out",
            str(tmp_path / "x.json"),
            "--dry-run",
        ]
    )
    assert rc == 0 and not (tmp_path / "x.json").exists()
