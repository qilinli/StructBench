"""Tests for the whitelisted run-evidence record as JSON (ADR-0066 clause 3)."""

from __future__ import annotations

import json

import pytest

from structbench.core import (
    EnergyLedger,
    RunEvidence,
    SolverIdentity,
    TerminationRecord,
)
from structbench.core.io import (
    RUN_EVIDENCE_SCHEMA,
    dump_run_evidence,
    load_run_evidence,
)


def _full() -> RunEvidence:
    return RunEvidence(
        SolverIdentity("ls-dyna", "mpp.123456", "123999", "double", "mpp:4"),
        (TerminationRecord("normal", 4.002e-6, 52, "end_time"),),
        0,
        2,
        ((0.0, 2.0e-6), (8.0e-8, 7.6e-8)),
        EnergyLedger(
            (0.0, 2.0e-6),
            {
                "kinetic": (2.0, 1.5),
                "internal": (0.0, 0.52),
                "external_work": (0.0, 0.0),
            },
            {"kinetic": 1, "internal": 1},
            (2.0, 2.02),
        ),
        frozenset({"termination_criterion"}),
    )


def test_the_record_round_trips_and_is_byte_stable() -> None:
    records = {"B-2": _full(), "A-1": RunEvidence()}
    text = dump_run_evidence(records)
    assert load_run_evidence(text) == records
    assert dump_run_evidence(load_run_evidence(text)) == text
    assert list(json.loads(text)["runs"]) == ["A-1", "B-2"]
    assert text.endswith("\n") and "\r" not in text


def test_an_empty_run_is_all_nulls() -> None:
    raw = json.loads(dump_run_evidence({"A-1": RunEvidence()}))["runs"]["A-1"]
    assert raw == {
        "identity": None,
        "termination": None,
        "n_errors": None,
        "n_warnings": None,
        "timestep": None,
        "ledger": None,
        "unparsable": [],
    }


def test_another_schema_is_refused() -> None:
    text = dump_run_evidence({}).replace(RUN_EVIDENCE_SCHEMA, "something/9")
    with pytest.raises(ValueError, match="schema"):
        load_run_evidence(text)


def test_a_hand_edited_record_cannot_carry_free_text() -> None:
    raw = json.loads(dump_run_evidence({"A-1": _full()}))
    raw["runs"]["A-1"]["identity"]["version"] = "C:\\Users\\someone\\run 7"
    with pytest.raises(ValueError, match="version"):
        load_run_evidence(json.dumps(raw))
    raw["runs"]["A-1"]["identity"]["version"] = "mpp.123456"
    raw["runs"]["A-1"]["unparsable"] = ["see host build-box-17"]
    with pytest.raises(ValueError, match="tokens"):
        load_run_evidence(json.dumps(raw))
