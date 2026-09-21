"""Tests for the Taylor run-evidence glue (ADR-0066 clause 3).

The glue builds paths from case ids, opens two small text files per run, and
writes one whitelisted record. Invented text only.
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
    / "2D-Copper-Bar-Taylor-Impact"
    / "collect_run_evidence.py"
)
_spec = importlib.util.spec_from_file_location("collect_run_evidence", _TOOL_PATH)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

_MESSAGES = "\n".join(
    [
        " Licensed to: An Invented Customer on host build-box-17",
        " Input file: Q:\\secret\\folder\\Taylor.k",
        "                         ls-dyna mpp.123456 d           date 01/01/2020",
        " *** termination time reached ***",
        " N o r m a l    t e r m i n a t i o n                   01/01/20 10:00:01",
        " Problem cycle      =        52",
    ]
)


def test_paths_are_built_from_case_ids_not_discovered(tmp_path: Path) -> None:
    assert tool.run_dir(tmp_path, "T-20-60-100") == tmp_path / "lsdyna" / "2060" / "100"
    expected = tmp_path / "lsdyna" / "2080" / "Convergence"
    assert tool.run_dir(tmp_path, "T-20-80-Convergence") == expected
    with pytest.raises(ValueError, match="case id"):
        tool.run_dir(tmp_path, "../../etc")


def test_the_written_record_holds_nothing_private(tmp_path: Path) -> None:
    run = tmp_path / "lsdyna" / "2060" / "100"
    run.mkdir(parents=True)
    (run / "mes0000").write_text(_MESSAGES, encoding="utf-8")
    out = tmp_path / "out" / "evidence.json"
    code = tool.main(
        [
            "--data-root",
            str(tmp_path),
            "--case",
            "T-20-60-100",
            "--case",
            "T-20-60-110",  # no such run folder: skipped, not fatal
            "--out",
            str(out),
        ]
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    for leak in ("Invented", "build-box", "secret", "Q:", "01/01/20"):
        assert leak not in text
    records = load_run_evidence(text)
    assert list(records) == ["T-20-60-100"]
    record = records["T-20-60-100"]
    assert record.termination is not None and record.termination[0].n_steps == 52
    assert record.ledger is None  # no statistics file in this run folder


def test_a_missing_data_root_is_a_usage_error(tmp_path: Path) -> None:
    out = str(tmp_path / "x.json")
    assert tool.main(["--data-root", str(tmp_path / "absent"), "--out", out]) == 2
