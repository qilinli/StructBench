"""Tests for ``python -m structbench.cli.datacheck`` (ADR-0066).

A tiny synthetic particle case carrying its own invented solver input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from structbench.benchmarks.card import BenchmarkCard
from structbench.benchmarks.registry import BenchmarkSpec
from structbench.cli.datacheck import declared_from_spec, main, measure_dataset
from structbench.core import (
    AbsenceReason,
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
    write_case,
)
from structbench.verification.report import from_json, to_json

T, P = 3, 2


def _row(*values: object) -> str:
    return "".join(f"{v!s:>10}" for v in values)


_DECK = "\n".join(
    [
        "*KEYWORD",
        "*CONTROL_SPH",
        _row(1, 0, "1.0E20", 2, 150, 0),
        "*CONTROL_TERMINATION",
        _row(0.2, 0, 0.0, 0.0, "1.0E8"),
        "*PART",
        "an invented bar",
        _row(1, 7, 2, 0, 0),
        "*SECTION_SPH",
        _row(7, 1.2, 1.0, 1.0),
        "*MAT_ELASTIC_PLASTIC_HYDRO",
        _row(2, 0.0027, 26000.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        _row(0.0, 1.0, "", "", "", "", "", ""),
        _row("", "", "", "", "", "", "", ""),
        _row(100.0, 200.0, "", "", "", "", "", ""),
        _row("", "", "", "", "", "", "", ""),
        "*END",
    ]
)


def _spec() -> BenchmarkSpec:
    card = BenchmarkCard(
        name="DatacheckSmoke",
        version="0.0",
        description="synthetic fixture",
        provenance="synthetic (test fixture)",
        data_license="n/a (synthetic test data)",
        solver="invented",
        discretisation="SPH",
        materials=("synthetic",),
        loading="none",
        erosion=False,
        source_units="g-mm-ms (as stored with the case)",
        geometry="two particles",
        n_cases=2,
        splits={"train": 1, "val": 1},
        task="smoke",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=(),
        fields=("node/displacement", "sph/density", "sph/effective_plastic_strain"),
        particles_per_case="2-2",
        n_frames=T,
        output_dt_ms=0.1,
        input_frames=2,
        protocol_rationale="synthetic fixture; not a benchmark",
    )
    return BenchmarkSpec(
        card=card,
        splits={"train": ("B-1",), "val": ("A-1",)},
        eval_splits=("val",),
        aux_field="von_mises_stress",
        dataset_id="datacheck-smoke",
        hardening_curve=((0.0, 1.0), (100.0, 200.0)),
    )


def _write(root: Path, case_id: str) -> None:
    case = Case(
        metadata=Metadata(
            case_id=case_id, dimension=2, source_units="g-mm-ms", source_deck=_DECK
        ),
        nodes=Nodes(
            coords=np.array([[0.0, 0.0], [1e-3, 0.0]]),
            node_id=np.arange(1, P + 1, dtype=np.int64),
        ),
        elements={
            "sph": ElementBlock(
                connectivity=np.arange(P, dtype=np.int64).reshape(P, 1),
                element_id=np.arange(1, P + 1, dtype=np.int64),
                part_id=np.ones(P, dtype=np.int64),
            )
        },
        materials=[Material(2, "MAT_ELASTIC_PLASTIC_HYDRO", {"data": [[1]]}, None)],
        response=Response(
            time=np.array([0.0, 1.0e-4, 2.0e-4]),
            node={"displacement": np.zeros((T, P, 2), dtype=np.float32)},
            element={
                "sph": {
                    "density": np.full((T, P), 2700.0, dtype=np.float32),
                    "effective_plastic_strain": np.zeros((T, P), dtype=np.float32),
                }
            },
        ),
    )
    write_case(case, root / f"{case_id}.h5")


def test_declarations_come_from_the_card_and_the_spec() -> None:
    declared = declared_from_spec(_spec())
    assert declared.unit_system == "g-mm-ms"  # the label, without the card's note
    assert declared.discretisation == "SPH" and declared.erosion is False
    assert declared.yield_table == ((0.0, 1.0), (100.0e6, 200.0e6))  # MPa -> Pa
    assert declared.anchors == ()


def test_measuring_a_dataset_reads_each_case_and_its_stored_input(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "A-1")
    _write(tmp_path, "B-1")
    record = measure_dataset(_spec(), tmp_path, dataset_revision="v0")
    assert [c.case_id for c in record.cases] == ["A-1", "B-1"]
    assert (record.benchmark, record.dataset_revision) == ("DatacheckSmoke", "v0")
    case = record.cases[0]
    assert case.file_sha256 is not None and len(case.file_sha256) == 64
    assert case.run_traits == {"explicit", "dim2"}
    values = {m.quantity: m.value for m in case.measurements}
    assert values["reached_end_time"] == 1.0  # 0.2 ms requested, 2e-4 s stored
    assert values["density_slot_matches_input"] == 0.0
    assert values["yield_table_matches_input"] == 0.0
    assert values["fields_match_declaration"] == 0.0
    assert record.definition_versions["nonfinite_count"] == 1
    assert to_json(
        measure_dataset(_spec(), tmp_path, dataset_revision="v0")
    ) == to_json(record)


def test_a_case_that_cannot_be_read_is_recorded_not_fatal(tmp_path: Path) -> None:
    _write(tmp_path, "A-1")
    (tmp_path / "B-1.h5").write_bytes(b"not an hdf5 file")
    record = measure_dataset(_spec(), tmp_path, ["A-1", "B-1", "C-1"])
    broken, missing = record.cases[1], record.cases[2]
    assert broken.file_sha256 is not None and missing.file_sha256 is None
    for case in (broken, missing):
        reasons = {m.absence.reason for m in case.measurements if m.absence}
        assert reasons == {AbsenceReason.SOURCE_UNREADABLE}
        assert all(m.value is None for m in case.measurements)


def test_judge_needs_only_the_record(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    _write(tmp_path, "A-1")
    record_path, report_path = tmp_path / "out" / "m.json", tmp_path / "out" / "m.md"
    record_path.parent.mkdir()
    record_path.write_text(to_json(measure_dataset(_spec(), tmp_path, ["A-1"])))
    (tmp_path / "A-1.h5").unlink()  # judging must not look for the data

    assert main(["judge", "--measurements", str(record_path)]) == 0
    printed = capsys.readouterr().out
    assert "# Reference-data verification: DatacheckSmoke" in printed
    assert (
        main(
            ["judge", "--measurements", str(record_path), "--report", str(report_path)]
        )
        == 0
    )
    report = report_path.read_bytes()
    assert b"\r" not in report and report.decode("utf-8") in printed


def test_measure_writes_a_record_even_when_no_case_is_readable(tmp_path: Path) -> None:
    out = tmp_path / "record.json"
    code = main(
        [
            "measure",
            "--benchmark",
            "taylor_impact_2d",
            "--data-root",
            str(tmp_path),
            "--case",
            "T-0-0-0",
            "--out",
            str(out),
        ]
    )
    assert code == 0  # completed: what went wrong is in the record
    record = from_json(out.read_text(encoding="utf-8"))
    assert [c.case_id for c in record.cases] == ["T-0-0-0"]
    assert json.loads(out.read_text(encoding="utf-8"))["benchmark"]


def test_usage_and_io_errors_exit_with_two(tmp_path: Path) -> None:
    out = str(tmp_path / "x.json")
    here = str(tmp_path)
    assert main([]) == 2
    assert (
        main(["measure", "--benchmark", "nope", "--data-root", here, "--out", out]) == 2
    )
    absent = str(tmp_path / "absent")
    assert (
        main(
            [
                "measure",
                "--benchmark",
                "taylor_impact_2d",
                "--data-root",
                absent,
                "--out",
                out,
            ]
        )
        == 2
    )
    assert main(["judge", "--measurements", str(tmp_path / "absent.json")]) == 2
    (tmp_path / "bad.json").write_text('{"schema": "other/1"}')
    assert main(["judge", "--measurements", str(tmp_path / "bad.json")]) == 2
