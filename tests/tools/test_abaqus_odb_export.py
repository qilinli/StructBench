"""Tests for the ODB exporter's pure helpers (ADR-0069). No Abaqus needed,
except the env-gated end-to-end test."""

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import abaqus_paths
import numpy as np
import odb_export
import pytest


def test_parses_as_python_3_10():
    source = (abaqus_paths.ABAQUS_DIR / "odb_export.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 10))
    assert "\nimport odbAccess" not in source and "\nfrom odbAccess" not in source


def _block(data, nodes=(), elements=(), ips=()):
    return SimpleNamespace(
        data=np.asarray(data),
        nodeLabels=np.asarray(nodes),
        elementLabels=np.asarray(elements),
        integrationPoints=np.asarray(ips),
    )


def test_stack_blocks_nodal_and_scalar_element():
    nodal = odb_export.stack_blocks(
        [_block([[1.0, 2.0]], nodes=[1]), _block([[3.0, 4.0]], nodes=[2])]
    )
    assert nodal["label_kind"] == "node" and nodal["labels"].tolist() == [1, 2]
    assert nodal["data"].shape == (2, 2) and nodal["integration_points"] is None
    scalar = odb_export.stack_blocks([_block([0.1, 0.2], elements=[7, 8], ips=[1, 1])])
    assert scalar["label_kind"] == "element" and scalar["data"].shape == (2, 1)


def test_stack_blocks_reads_none_as_absent_labels():
    # A real ODB gives None, not an empty array, for the labels a block lacks.
    nodal = SimpleNamespace(
        data=np.ones((2, 2), np.float32),
        nodeLabels=np.array([1, 2], np.int32),
        elementLabels=None,
        integrationPoints=None,
    )
    element = SimpleNamespace(
        data=np.ones((3, 4), np.float32),
        nodeLabels=None,
        elementLabels=np.array([7, 8, 9], np.int32),
        integrationPoints=np.array([1, 1, 1], np.int32),
    )
    assert odb_export.stack_blocks([nodal])["label_kind"] == "node"
    stacked = odb_export.stack_blocks([element])
    assert stacked["label_kind"] == "element"
    assert stacked["labels"].tolist() == [7, 8, 9]


def test_time_series_refuses_changing_labels():
    a = odb_export.stack_blocks([_block([[1.0]], nodes=[1])])
    b = odb_export.stack_blocks([_block([[1.0]], nodes=[2])])
    assert odb_export.time_series([a, a])["data"].shape == (2, 1, 1)
    with pytest.raises(ValueError, match="labels change"):
        odb_export.time_series([a, b])


def test_selects_only_completed_unexported_cases(tmp_path):
    for name, status, exported in [
        ("A", "completed", False),
        ("B", "error", False),
        ("C", "completed", True),
    ]:
        d = tmp_path / name
        d.mkdir()
        (d / "run.json").write_text(json.dumps({"status": status}))
        if exported:
            (d / f"{name}.npz").write_bytes(b"")
    (tmp_path / "D").mkdir()  # never ran
    assert [p.name for p in odb_export.selected_cases(tmp_path)] == ["A"]


@pytest.mark.skipif(
    "STRUCTBENCH_ABAQUS_RUN_DIR" not in os.environ,
    reason="needs a finished Abaqus case folder",
)
def test_exports_a_real_odb(tmp_path):
    source = Path(os.environ["STRUCTBENCH_ABAQUS_RUN_DIR"])
    case = tmp_path / source.name
    case.mkdir()
    shutil.copy2(source / f"{source.name}.odb", case)
    (case / "run.json").write_text(json.dumps({"status": "completed"}))
    exe = shutil.which("abaqus")
    assert exe is not None
    script = str(abaqus_paths.ABAQUS_DIR / "odb_export.py")
    subprocess.run([exe, "python", script, "--sweep", str(tmp_path)], check=True)
    with np.load(case / f"{source.name}.npz", allow_pickle=False) as npz:
        manifest = json.loads(str(npz["manifest"]))
        assert manifest["format"] == "abaqus-npz/1"
        times = [k for k in npz.files if k.endswith("/frame_times")]
        assert times and npz[times[0]][0] == 0.0
        assert any(k.startswith("history/") for k in npz.files)
