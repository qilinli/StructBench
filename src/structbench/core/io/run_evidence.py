"""The whitelisted run-evidence record as JSON (ADR-0066 clause 3).

Per-dataset glue reads each run's text files through the solver's extractor
and writes *one* record with this module; the verification CLI reads that
record and never a run directory. The record is built from ``RunEvidence``
alone, so it can hold numbers, enum values and version tokens — nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..evidence import EnergyLedger, RunEvidence, SolverIdentity, TerminationRecord

__all__ = ["RUN_EVIDENCE_SCHEMA", "dump_run_evidence", "load_run_evidence"]

RUN_EVIDENCE_SCHEMA = "structbench.run_evidence/1"


def _dump_one(run: RunEvidence) -> dict[str, Any]:
    identity, ledger = run.identity, run.ledger
    return {
        "identity": None
        if identity is None
        else {
            "name": identity.name,
            "version": identity.version,
            "revision": identity.revision,
            "precision": identity.precision,
            "parallel_layout": identity.parallel_layout,
        },
        "termination": None
        if run.termination is None
        else [
            {
                "status": t.status,
                "final_time": t.final_time,
                "n_steps": t.n_steps,
                "criterion": t.criterion,
            }
            for t in run.termination
        ],
        "n_errors": run.n_errors,
        "n_warnings": run.n_warnings,
        "timestep": None
        if run.timestep is None
        else {"time": list(run.timestep[0]), "step": list(run.timestep[1])},
        "ledger": None
        if ledger is None
        else {
            "time": list(ledger.time),
            "terms": {name: list(series) for name, series in ledger.terms.items()},
            "identity": dict(ledger.identity),
            "solver_total": None
            if ledger.solver_total is None
            else list(ledger.solver_total),
            "origin": ledger.origin,
        },
        "unparsable": sorted(run.unparsable),
    }


def _load_one(raw: Mapping[str, Any]) -> RunEvidence:
    identity, ledger, steps = raw["identity"], raw["ledger"], raw["timestep"]
    return RunEvidence(
        identity=None if identity is None else SolverIdentity(**identity),
        termination=None
        if raw["termination"] is None
        else tuple(TerminationRecord(**t) for t in raw["termination"]),
        n_errors=raw["n_errors"],
        n_warnings=raw["n_warnings"],
        timestep=None
        if steps is None
        else (tuple(steps["time"]), tuple(steps["step"])),
        ledger=None
        if ledger is None
        else EnergyLedger(
            tuple(ledger["time"]),
            {name: tuple(series) for name, series in ledger["terms"].items()},
            ledger["identity"],
            None if ledger["solver_total"] is None else tuple(ledger["solver_total"]),
            ledger["origin"],
        ),
        unparsable=frozenset(raw["unparsable"]),
    )


def dump_run_evidence(records: Mapping[str, RunEvidence]) -> str:
    """Serialise run evidence keyed by case id; byte-stable for equal input."""
    payload = {
        "schema": RUN_EVIDENCE_SCHEMA,
        "runs": {case_id: _dump_one(records[case_id]) for case_id in sorted(records)},
    }
    return json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"


def load_run_evidence(text: str) -> dict[str, RunEvidence]:
    """Read a record written by :func:`dump_run_evidence`.

    Every value passes through the record types' own validation again, so a
    hand-edited file cannot smuggle free text into a report.

    Raises
    ------
    ValueError
        If the schema id is not this module's, or a record is invalid.
    """
    raw = json.loads(text)
    if raw.get("schema") != RUN_EVIDENCE_SCHEMA:
        raise ValueError(
            f"expected schema {RUN_EVIDENCE_SCHEMA!r}, got {raw.get('schema')!r}"
        )
    return {case_id: _load_one(run) for case_id, run in raw["runs"].items()}
