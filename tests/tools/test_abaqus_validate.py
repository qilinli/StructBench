"""Tests for validate.py: evidence, measurement and verdicts for a sweep (Task 8)."""

import importlib.util
import json
from pathlib import Path

import abaqus_paths  # noqa: F401
import convert
import numpy as np
import validate

_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FIXTURE = _load(
    "abaqus_adapter_fixture", _HERE.parent / "core" / "test_abaqus_adapter.py"
)
_COLLECT = _load("abaqus_collect_fixture", _HERE / "test_abaqus_collect.py")

_TOML = """
[dataset]
name = "toy_sweep"

[declaration]
unit_system = "t-mm-s"
fields = [
    "node/displacement", "node/velocity", "node/acceleration",
    "solid/stress", "solid/effective_plastic_strain", "global/kinetic_energy",
]
discretisation = "FEM"
erosion = false
"""
_ABORTED_STA = _COLLECT._STA.replace(
    "THE ANALYSIS HAS COMPLETED SUCCESSFULLY", "THE ANALYSIS HAS NOT BEEN COMPLETED"
)


def _sweep(tmp_path: Path) -> tuple[Path, Path]:
    sweep = tmp_path / "toy_sweep"
    _COLLECT._case(sweep, "T-0000")
    _COLLECT._case(sweep, "T-0001", status="failed", npz=False)
    for folder in (sweep / "T-0000", sweep / "T-0001"):
        (folder / f"{folder.name}.inp").write_text(
            _FIXTURE._ADAPTER_DECK, encoding="utf-8"
        )
    (sweep / "T-0001" / "T-0001.sta").write_text(_ABORTED_STA, encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "sweep.toml").write_text(_TOML, encoding="utf-8")
    convert.convert_sweep(sweep)
    return sweep, dataset


def test_an_aborted_run_is_reported_as_a_failure(tmp_path, capsys):
    """Review Focus 1: convert.py skips it, validate.py must not lose it."""
    sweep, dataset = _sweep(tmp_path)
    assert validate.main(["--sweep", str(sweep), "--dataset", str(dataset)]) == 1
    out = sweep / "datacheck"
    record = json.loads((out / "measurements.json").read_text(encoding="utf-8"))
    assert [c["case_id"] for c in record["cases"]] == ["T-0000", "T-0001"]
    aborted = next(c for c in record["cases"] if c["case_id"] == "T-0001")
    row = next(
        m for m in aborted["measurements"] if m["quantity"] == "terminated_normally"
    )
    assert row["value"] == 0.0
    assert (out / "report.md").is_file() and (out / "run_evidence.json").is_file()
    printed = capsys.readouterr().out
    assert "T-0001" in printed and "terminated_normally" in printed
    assert "measured only" in printed


def test_splits_narrow_the_cases(tmp_path):
    sweep, dataset = _sweep(tmp_path)
    for case in ("T-0000", "T-0001"):
        prov = sweep / case / "provenance.json"
        data = json.loads(prov.read_text(encoding="utf-8"))
        data["split"] = "keep" if case == "T-0000" else "other"
        prov.write_text(json.dumps(data), encoding="utf-8")
    validate.main(["--sweep", str(sweep), "--dataset", str(dataset), "--split", "keep"])
    record = json.loads(
        (sweep / "datacheck" / "measurements.json").read_text(encoding="utf-8")
    )
    assert [c["case_id"] for c in record["cases"]] == ["T-0000"]
    assert np.isfinite(
        next(
            m["value"]
            for m in record["cases"][0]["measurements"]
            if m["quantity"] == "time_axis_monotone"
        )
    )
