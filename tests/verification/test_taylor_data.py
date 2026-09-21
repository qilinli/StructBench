"""Stage-1 acceptance on the real Taylor archive (ADR-0066) — runs only with data.

Set ``STRUCTBENCH_DATA_ROOT`` to the ``taylor_impact_2d`` canonical folder
(the convention of ``tests/benchmarks/test_card_data.py``). The expected
numbers are those ``tools/state_probe`` recorded for ``T-20-60-100`` before
this instrument existed: the instrument must reproduce them.

Set ``STRUCTBENCH_TAYLOR_RUN_DIR`` to the *raw run folder* of ``T-20-100-100``
for the stage-2 facts, hand-read from that run's text files before the reader
existed. Raw run folders are private; only numbers are asserted here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from structbench.benchmarks import get_benchmark
from structbench.cli.datacheck import measure_dataset
from structbench.core import read_input_facts, read_run_evidence
from structbench.verification import Verdict
from structbench.verification.criteria import judge
from structbench.verification.measures import measure_case

_ROOT = os.environ.get("STRUCTBENCH_DATA_ROOT")
_CASE = "T-20-60-100"
_RUN_DIR = os.environ.get("STRUCTBENCH_TAYLOR_RUN_DIR")


def test_the_instrument_reproduces_the_probe_on_one_taylor_case() -> None:
    if _ROOT is None or not (Path(_ROOT) / f"{_CASE}.h5").is_file():
        pytest.skip("STRUCTBENCH_DATA_ROOT not set, or the case is not present")
    record = measure_dataset(get_benchmark("taylor_impact_2d"), Path(_ROOT), [_CASE])
    (case,) = record.cases
    rows = {m.quantity: m for m in case.measurements}

    assert case.run_traits == {"explicit", "dim2"}
    ratio = rows["yield_ratio_max"]
    assert ratio.value == pytest.approx(1.000353, abs=1e-6)
    assert ratio.n_samples == 152 * 4800  # every stored frame, every particle
    backward = rows["state_variable_decrease_max"]
    assert (backward.value, backward.detail["decreasing_steps"]) == (0.0, 0)
    assert backward.n_samples == 151 * 4800
    assert rows["terminal_artifact_frames"].value == 1.0  # 152 stored, 151 valid
    assert rows["reached_end_time"].value == pytest.approx(1.0, abs=1e-3)
    assert rows["nonfinite_count"].value == 0.0
    # findings the archive is known to carry
    assert rows["yield_table_monotone"].value == 1.0  # one knot dips by 0.2 MPa
    assert rows["yield_table_monotone"].detail["largest_drop"] == pytest.approx(2.0e5)
    assert rows["elements_without_input_part"].value == 1.0  # a shell no part owns

    (report,) = judge(record).cases
    verdicts = {r.quantity: r.verdict for r in report.results}
    assert verdicts["yield_ratio_max"] is Verdict.NOT_ASSESSABLE  # no criterion yet
    assert verdicts["energy_gain_max"] is Verdict.NOT_ASSESSABLE  # no ledger supplied
    assert verdicts["implicit_convergence"] is Verdict.NOT_APPLICABLE
    assert Verdict.REVIEW not in verdicts.values()


def test_the_run_record_of_one_taylor_run_reproduces_the_hand_read_facts() -> None:
    if _RUN_DIR is None or not (Path(_RUN_DIR) / "glstat").is_file():
        pytest.skip("STRUCTBENCH_TAYLOR_RUN_DIR not set, or it holds no glstat")
    folder = Path(_RUN_DIR)

    def text(name: str) -> str:
        return (folder / name).read_text(encoding="utf-8", errors="replace")

    run = read_run_evidence(
        messages_text=text("mes0000"),
        global_statistics_text=text("glstat"),
        source_units="g-mm-ms",
    )
    assert run.unparsable == frozenset()
    assert run.termination is not None
    (segment,) = run.termination
    assert (segment.status, segment.n_steps, segment.criterion) == (
        "normal",
        3918,
        "end_time",
    )
    assert run.timestep is not None
    assert min(run.timestep[1]) == pytest.approx(7.46e-8, rel=1e-3)  # 7.46e-5 ms
    assert max(run.timestep[1]) == pytest.approx(7.78e-8, rel=1e-3)
    assert run.ledger is not None
    assert run.ledger.terms["rigid_surface"][-1] == pytest.approx(0.311, rel=1e-3)

    facts = read_input_facts(text("Taylor.k"), source_units="g-mm-ms")
    case = measure_case(None, facts, None, case_id="T-20-100-100", run=run)
    rows = {m.quantity: m.value for m in case.measurements}
    assert rows["total_energy_change_final"] == pytest.approx(0.0702, abs=1e-4)
    # smaller: the non-kinetic bucket has outgrown the initial total
    assert rows["energy_residual_final"] == pytest.approx(0.0657, abs=1e-4)
    assert rows["energy_gain_max"] == pytest.approx(0.0893, abs=1e-4)
    assert rows["energy_loss_max"] == 0.0
    assert (rows["terminated_normally"], rows["solver_error_count"]) == (1.0, 0.0)
