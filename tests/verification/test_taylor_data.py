"""Stage-1 acceptance on the real Taylor archive (ADR-0066) — runs only with data.

Set ``STRUCTBENCH_DATA_ROOT`` to the ``taylor_impact_2d`` canonical folder
(the convention of ``tests/benchmarks/test_card_data.py``). The expected
numbers are those ``tools/state_probe`` recorded for ``T-20-60-100`` before
this instrument existed: the instrument must reproduce them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from structbench.benchmarks import get_benchmark
from structbench.cli.datacheck import measure_dataset
from structbench.verification import Verdict
from structbench.verification.criteria import judge

_ROOT = os.environ.get("STRUCTBENCH_DATA_ROOT")
_CASE = "T-20-60-100"


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
