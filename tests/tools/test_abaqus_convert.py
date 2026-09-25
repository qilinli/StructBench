"""Tests for convert.py: a sweep's exports -> canonical HDF5 cases (plan 2, Task 5)."""

import importlib.util
import json
from pathlib import Path

import abaqus_paths  # noqa: F401
import convert
import numpy as np

from structbench.core.io import read_case
from structbench.core.validation import validate

# The adapter's synthetic export and deck, loaded by path: test folders are
# not packages, and one fixture keeps the two tests describing one layout.
_SPEC = importlib.util.spec_from_file_location(
    "abaqus_adapter_fixture",
    Path(__file__).resolve().parents[1] / "core" / "test_abaqus_adapter.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE)


def _case(sweep: Path, name: str, *, status="completed", split="train", npz=True):
    folder = sweep / name
    folder.mkdir(parents=True)
    run = {"status": status, "end_utc": "2026-09-24T16:07:59+00:00"}
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    prov = {"units": "t-mm-s", "split": split, "format": "abaqus-provenance/1"}
    (folder / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    (folder / f"{name}.inp").write_text(_FIXTURE._ADAPTER_DECK, encoding="utf-8")
    if npz:
        np.savez(folder / f"{name}.npz", **_FIXTURE._arrays())
    return folder


def test_completed_exports_become_valid_cases(tmp_path):
    sweep = tmp_path / "toy_sweep"
    _case(sweep, "T-0000")
    _case(sweep, "T-0001", status="failed")
    _case(sweep, "T-0002", npz=False)
    report = convert.convert_sweep(sweep)
    assert (report.written, report.skipped, report.failed) == (["T-0000"], 2, {})
    case = read_case(sweep / "canonical" / "T-0000.h5")
    validate(case)
    assert case.response.time.shape == (2,)
    assert case.metadata.dataset_id == "toy_sweep"
    assert case.metadata.provenance.generation_date == "2026-09-24"
    assert case.metadata.source_units == "t-mm-s"


def test_a_second_run_writes_nothing(tmp_path):
    sweep = tmp_path / "toy_sweep"
    _case(sweep, "T-0000")
    convert.convert_sweep(sweep)
    again = convert.convert_sweep(sweep)
    assert (again.written, again.skipped) == ([], 1)


def test_a_bad_export_is_reported_not_fatal(tmp_path):
    sweep = tmp_path / "toy_sweep"
    _case(sweep, "T-0000")
    bad = _case(sweep, "T-0001", npz=False)
    arrays = _FIXTURE._arrays()
    arrays["manifest"] = np.array(json.dumps({"format": "other/9"}))
    np.savez(bad / "T-0001.npz", **arrays)
    report = convert.convert_sweep(sweep)
    assert report.written == ["T-0000"]
    assert "abaqus-npz" in report.failed["T-0001"]
    assert not (sweep / "canonical" / "T-0001.h5").exists()


def test_splits_and_out_dir(tmp_path):
    sweep = tmp_path / "toy_sweep"
    _case(sweep, "T-0000", split="train")
    _case(sweep, "T-0001", split="test")
    out = tmp_path / "elsewhere"
    report = convert.convert_sweep(sweep, splits=["test"], out=out)
    assert report.written == ["T-0001"]
    assert [p.name for p in out.iterdir()] == ["T-0001.h5"]


def test_main_prints_the_counts(tmp_path, capsys):
    sweep = tmp_path / "toy_sweep"
    _case(sweep, "T-0000")
    assert convert.main(["--sweep", str(sweep)]) == 0
    assert "written=1 skipped=0 failed=0" in capsys.readouterr().out


def test_a_case_the_schema_refuses_does_not_stop_the_sweep(tmp_path):
    """Review (final) I3: SchemaError is not a ValueError."""
    sweep = tmp_path / "toy_sweep"
    bad = _case(sweep, "T-0000")
    _case(sweep, "T-0001")
    deck = _FIXTURE._ADAPTER_DECK.replace(
        "*MATERIAL, NAME=M\n", "*INCLUDE, INPUT=m.inp\n"
    )
    (bad / "T-0000.inp").write_text(deck, encoding="utf-8")
    report = convert.convert_sweep(sweep)
    assert report.written == ["T-0001"]
    assert "SchemaError" in report.failed["T-0000"]
