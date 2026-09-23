"""Tests for the notch-beam run-evidence glue (ADR-0066 clause 3).

The glue builds paths from case ids, opens one small text file per run, and
writes one whitelisted record. These runs kept no global-statistics file, so
every record comes out with no energy ledger. Invented text only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from structbench.core.io import load_run_evidence

_TOOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "data_generation"
    / "lsdyna"
    / "2DNotchBeam"
    / "collect_run_evidence.py"
)
_spec = importlib.util.spec_from_file_location("collect_notch_run_evidence", _TOOL_PATH)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

_MESSAGES = "\n".join(
    [
        " Licensed to: An Invented Customer on host build-box-17",
        " Input file: Q:\\secret\\folder\\Beam1.k",
        "     |  Revision: R12.1-190-gadfcdf9018                |",
        "                         ls-dyna mpp.190 d           date 01/01/2022",
        " *** termination time reached ***",
        " N o r m a l    t e r m i n a t i o n                   01/01/22 10:00:01",
        " Problem cycle      =      1637",
    ]
)


def test_paths_are_built_from_case_ids_not_discovered(tmp_path: Path) -> None:
    grid = tmp_path / "InitialVelocity" / "Sphere" / "80480" / "Ab120"
    assert tool.run_dir(tmp_path, "NB-I-480-Sphere-b-120") == grid
    probe = tmp_path / "2DGeneralizibility" / "S_80_400_V140_intrapolation"
    assert tool.run_dir(tmp_path, "S_80_400_V140_intrapolation") == probe
    for bogus in ("../../etc", "NB-B-320-Aa-8", "S_80_400/../../etc"):
        with pytest.raises(ValueError, match="case id"):
            tool.run_dir(tmp_path, bogus)


def test_the_written_record_holds_nothing_private(tmp_path: Path) -> None:
    run = tmp_path / "InitialVelocity" / "Bullet" / "80320" / "Aa40"
    run.mkdir(parents=True)
    (run / "mes0000").write_text(_MESSAGES, encoding="utf-8")
    out = tmp_path / "out" / "evidence.json"
    code = tool.main(
        [
            "--data-root",
            str(tmp_path),
            "--case",
            "NB-I-320-Bullet-a-40",
            "--case",
            "NB-I-320-Bullet-a-80",  # no such run folder: skipped, not fatal
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    for leak in ("Invented", "build-box", "secret", "Q:", "01/01/22"):
        assert leak not in text
    records = load_run_evidence(text)
    assert list(records) == ["NB-I-320-Bullet-a-40"]
    record = records["NB-I-320-Bullet-a-40"]
    assert record.termination is not None and record.termination[0].n_steps == 1637
    assert record.identity is not None
    assert record.identity.revision == "R12.1-190-gadfcdf9018"


def test_these_runs_keep_no_energy_ledger(tmp_path: Path) -> None:
    """The decks never requested ``*DATABASE_GLSTAT``; E5 is simply absent."""
    run = tmp_path / "InitialVelocity" / "Bullet" / "80320" / "Aa40"
    run.mkdir(parents=True)
    (run / "mes0000").write_text(_MESSAGES, encoding="utf-8")
    records = tool.collect(tmp_path, ["NB-I-320-Bullet-a-40"])
    assert records["NB-I-320-Bullet-a-40"].ledger is None


def test_a_missing_data_root_is_a_usage_error(tmp_path: Path) -> None:
    out = str(tmp_path / "x.json")
    assert tool.main(["--data-root", str(tmp_path / "absent"), "--out", out]) == 2
