"""Tests for collect_run_evidence.py: a sweep's run records -> one JSON (Task 6).

Invented text only. The printed file carries an invented licence line; the
record written must not.
"""

import importlib.util
import json
from pathlib import Path

import abaqus_paths  # noqa: F401
import collect_run_evidence as collect
import numpy as np

from structbench.core.io import load_run_evidence

_SPEC = importlib.util.spec_from_file_location(
    "abaqus_adapter_fixture",
    Path(__file__).resolve().parents[1] / "core" / "test_abaqus_adapter.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE)

_STA = (
    "Abaqus/Explicit 2025                             DATE 01-Jan-2026  TIME 00:00:00\n"
    "Double precision package and explicit executables will be used in this analysis.\n"
    "  STEP  TOTAL      STEP      CPU       STABLE       CRITICAL    KINETIC    TOTAL\n"
    "INCREMENT     TIME      TIME      TIME     INCREMENT     ELEMENT     ENERGY     ENERGY\n"  # noqa: E501 - the real fixed-width format
    "        0  0.000E+00 0.000E+00  00:00:00 5.00000E-08          12  1.000E+04  1.000E+04\n"  # noqa: E501 - the real fixed-width format
    "       40  5.000E-04 5.000E-04  00:00:01 3.00000E-08           7  9.000E+03  9.990E+03\n"  # noqa: E501 - the real fixed-width format
    "\n  THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n"
)
_MSG = "\n STEP 1  ORIGIN 0.0000\n"
_DAT = (
    "   Abaqus 2025\n"
    " Licensed to: An Invented Customer on host build-box-17, seat 4242\n"
    " ***WARNING: THE PARAMETER HOURGLASS ON THE *SECTION CONTROLS\n"
)


def _case(sweep: Path, name: str, *, split="train", npz=True, status="completed"):
    folder = sweep / name
    folder.mkdir(parents=True)
    run = {"status": status, "end_utc": "2026-09-24T16:07:59+00:00"}
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    prov = {"units": "t-mm-s", "split": split}
    (folder / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    (folder / f"{name}.sta").write_text(_STA, encoding="utf-8")
    (folder / f"{name}.msg").write_text(_MSG, encoding="utf-8")
    (folder / f"{name}.dat").write_text(_DAT, encoding="utf-8")
    if npz:
        np.savez(
            folder / f"{name}.npz", **_FIXTURE._with(_FIXTURE._arrays(), ALLCD=0.0)
        )


def test_each_run_becomes_a_whitelisted_record(tmp_path):
    sweep = tmp_path / "toy_sweep"
    _case(sweep, "T-0000")
    _case(sweep, "T-0001", npz=False)
    out = tmp_path / "evidence.json"
    assert collect.main(["--sweep", str(sweep), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "build-box-17" not in text and "4242" not in text and "Invented" not in text
    records = load_run_evidence(text)
    assert set(records) == {"T-0000", "T-0001"}
    done = records["T-0000"]
    assert done.termination[0].status == "normal"
    assert done.identity.version == "2025" and done.identity.precision == "double"
    assert done.ledger is not None and "kinetic" in done.ledger.terms
    assert records["T-0001"].ledger is None  # no export, no ledger


def test_an_aborted_run_is_recorded_too(tmp_path):
    """Review Focus 1: a failed case must not vanish from the evidence."""
    sweep = tmp_path / "toy_sweep"
    _case(sweep, "T-0000", status="failed", npz=False)
    records = collect.collect_sweep(sweep)
    assert set(records) == {"T-0000"}


def test_splits_select_cases(tmp_path):
    sweep = tmp_path / "toy_sweep"
    _case(sweep, "T-0000", split="train", npz=False)
    _case(sweep, "T-0001", split="test", npz=False)
    assert set(collect.collect_sweep(sweep, splits=["test"])) == {"T-0001"}
