"""Tests for tools/blessing_pooled_rmse.py (ADR-0043 SS8 blessing aggregator).

``tools/`` is not a package (mirrors ``tools/gen_benchmark_docs.py``), so the
module under test is loaded from its file path via ``importlib`` rather than
imported by dotted name.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from structbench.benchmarks.card import BenchmarkCard
from structbench.benchmarks.registry import BenchmarkSpec
from structbench.core import write_case
from structbench.core.io.meshgraphnets import build_deforming_plate_case
from structbench.datasets import load_case_trajectory

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "blessing_pooled_rmse.py"
_spec = importlib.util.spec_from_file_location("blessing_pooled_rmse", _TOOL_PATH)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


# --------------------------------------------------------------------------
# Unit tests: the pure pooled_rmse(pred, true) -> (sse, count) core.
# --------------------------------------------------------------------------


def test_pooled_rmse_known_error_vector():
    """A single node-step with a 3-4-0 position error: sse=25, count=3."""
    true = np.zeros((1, 1, 3), dtype=np.float32)
    pred = np.array([[[3.0, 4.0, 0.0]]], dtype=np.float32)
    sse, count = tool.pooled_rmse(pred, true)
    assert sse == pytest.approx(25.0)
    assert count == 3
    # sqrt(25/3): the per-node-step RMSE this single sample would carry.
    assert math.sqrt(sse / count) == pytest.approx(2.8867513459481287)


def test_pooled_rmse_zero_error():
    """Identical arrays pool to exactly zero error over every axis."""
    rng = np.random.default_rng(0)
    true = rng.random((3, 5, 3)).astype(np.float32)
    pred = true.copy()
    sse, count = tool.pooled_rmse(pred, true)
    assert sse == 0.0
    assert count == 3 * 5 * 3


def test_pooled_rmse_shape_mismatch_raises():
    true = np.zeros((2, 3, 3), dtype=np.float32)
    pred = np.zeros((2, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        tool.pooled_rmse(pred, true)


def test_pooled_vs_mean_of_case_diverges_on_unequal_node_counts():
    """The headline pooled stat is NOT the mean of per-case RMSEs.

    Case A is tiny (1 node-step) with a large error; case B is large
    (100 node-steps) with a tiny error. Pooling by total SSE/count weights
    by size (dominated by B, low), while a naive mean-of-per-case weights
    both cases equally (~average of a high and a low number) -- the two
    statistics diverge sharply, which is exactly why ADR-0043 pins the
    pooled (not mean-of-case) statistic as the headline gate value.
    """
    true_a = np.zeros((1, 1, 3), dtype=np.float64)
    pred_a = np.array([[[3.0, 4.0, 0.0]]], dtype=np.float64)
    sse_a, count_a = tool.pooled_rmse(pred_a, true_a)  # sse=25, count=3

    true_b = np.zeros((1, 100, 3), dtype=np.float64)
    pred_b = np.zeros((1, 100, 3), dtype=np.float64)
    pred_b[0, 0, 0] = 1.0  # sse=1, count=300
    sse_b, count_b = tool.pooled_rmse(pred_b, true_b)

    rmse_a = math.sqrt(sse_a / count_a)  # 2.8867513459481287
    rmse_b = math.sqrt(sse_b / count_b)  # 0.05773502691896258
    mean_of_case = (rmse_a + rmse_b) / 2

    pooled = math.sqrt((sse_a + sse_b) / (count_a + count_b))  # sqrt(26/303)

    assert rmse_a == pytest.approx(2.8867513459481287)
    assert rmse_b == pytest.approx(0.05773502691896258)
    assert mean_of_case == pytest.approx(1.4722431864335456)
    assert pooled == pytest.approx(math.sqrt(26 / 303))
    # The two conventions differ by roughly 5x here -- not a rounding
    # difference, a different statistic (ADR-0043 SS8's own point).
    assert abs(pooled - mean_of_case) > 1.0


# --------------------------------------------------------------------------
# File-level test: main() wired through synthetic evaluate()-shaped
# artifacts + canonical cases, asserting the JSON report and gate verdict.
# --------------------------------------------------------------------------


def _mini_spec(test_ids: list[str]) -> BenchmarkSpec:
    """A minimal valid BenchmarkSpec for deforming_plate-shaped data.

    Mirrors ``tests/cli/test_mgn_train_smoke.py``'s ``_mini_spec`` fixture
    style; only ``aux_field`` is actually read by the tool (to resolve
    :func:`structbench.datasets.load_case_trajectory`'s aux extraction), but
    a full valid card/spec is required to construct a ``BenchmarkSpec`` at
    all (``BenchmarkSpec.__post_init__`` validates split sizes against the
    card).
    """
    card = BenchmarkCard(
        name="BlessingSmoke",
        version="0.0",
        description="synthetic smoke benchmark for the blessing aggregator",
        provenance="synthetic (test fixture)",
        data_license="n/a (synthetic test data)",
        solver="COMSOL",
        discretisation="FEM",
        materials=("synthetic",),
        loading="synthetic actuator",
        erosion=False,
        source_units="kg-m-s",
        geometry="synthetic tets",
        n_cases=len(test_ids),
        splits={"train": 0, "val": 0, "test": len(test_ids)},
        task="smoke",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=(),
        fields=("node/displacement", "node/von_mises_stress"),
        particles_per_case="6-6",
        n_frames=8,
        output_dt_ms=1.0,
        input_frames=2,
        protocol_rationale="synthetic smoke fixture; not a benchmark",
    )
    return BenchmarkSpec(
        card=card,
        splits={"train": (), "val": (), "test": tuple(test_ids)},
        eval_splits=("test",),
        aux_field="von_mises_stress",
        dataset_id="blessing-smoke",
        kinematic_types=(1, 3),
    )


def _write_synthetic_case(root: Path, case_id: str, rng: np.random.Generator) -> None:
    """One tiny synthetic deforming_plate-shaped canonical case (P=6, T=8)."""
    P, T = 6, 8
    w0 = rng.random((P, 3)).astype(np.float32)
    drift = rng.random((T, P, 3)).astype(np.float32) * 0.01
    arrays = {
        "cells": np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32),
        "node_type": np.array([0, 0, 0, 0, 1, 3], dtype=np.int32),
        "mesh_pos": w0.copy(),
        "world_pos": (w0[None] + np.cumsum(drift, axis=0)).astype(np.float32),
        "stress": rng.random((T, P, 1)).astype(np.float32),
    }
    case = build_deforming_plate_case(arrays, source_units="kg-m-s", case_id=case_id)
    write_case(case, root / f"{case_id}.h5")


def test_main_end_to_end_pooled_and_gate(tmp_path, monkeypatch):
    """Wire main() through synthetic evaluate()-artifacts; hand-check the gate.

    A uniform (12, 16, 0) mm position error is injected at every node-step
    of every case's predicted trajectory. Because it is uniform, the
    pooled and per-case-mean statistics coincide exactly at
    sqrt((12^2 + 16^2 + 0^2) / 3) = sqrt(400/3) ~= 11.5470 mm, which the
    ADR-0043 SS8 gate band [11.1, 19.1] mm accepts (PASS) -- deliberately
    chosen inside the band so this test also exercises the pass branch of
    the gate logic end to end.
    """
    rng = np.random.default_rng(7)
    case_ids = ["test_0000", "test_0001"]
    data_root = tmp_path / "data"
    data_root.mkdir()
    for cid in case_ids:
        _write_synthetic_case(data_root, cid, rng)

    spec = _mini_spec(case_ids)
    monkeypatch.setattr(tool, "get_benchmark", lambda name: spec)

    run_dir = tmp_path / "run"
    rollout_dir = run_dir / "rollouts"
    rollout_dir.mkdir(parents=True)

    delta = np.array([12.0, 16.0, 0.0], dtype=np.float64)
    cases_record: dict[str, dict] = {}
    for cid in case_ids:
        trajectory = load_case_trajectory(
            data_root / f"{cid}.h5", aux_field="von_mises_stress"
        )
        true_positions = trajectory.positions  # (T, P, 3) mm, float32
        predicted_positions = (
            true_positions.astype(np.float64) + delta[None, None, :]
        ).astype(np.float32)
        np.savez(
            rollout_dir / f"test-{cid}.npz",
            predicted_positions=predicted_positions,
            predicted_aux=trajectory.aux,
        )
        cases_record[cid] = {"rollout_position_rmse": 0.0}  # tool reads keys only

    (run_dir / "metrics-test.json").write_text(
        json.dumps({"split": "test", "cases": cases_record}), encoding="utf-8"
    )

    out_path = run_dir / "blessing-pooled-test.json"
    rc = tool.main(
        [
            "--run-dir",
            str(run_dir),
            "--data-root",
            str(data_root),
            "--split",
            "test",
            "--benchmark",
            "deforming_plate",
            "--out",
            str(out_path),
        ]
    )

    expected_rmse_mm = math.sqrt((12.0**2 + 16.0**2 + 0.0**2) / 3.0)
    assert expected_rmse_mm == pytest.approx(11.5470053838, rel=1e-6)
    assert rc == 0  # inside the gate band -> PASS

    assert out_path.exists()
    report = json.loads(out_path.read_text(encoding="utf-8"))

    assert report["split"] == "test"
    assert report["benchmark"] == "deforming_plate"
    assert report["run_dir"] == str(run_dir)
    assert report["data_root"] == str(data_root)
    assert report["n_cases"] == 2
    assert report["pooled_rmse_mm"] == pytest.approx(expected_rmse_mm, rel=1e-3)
    assert report["pooled_rmse_native"] == pytest.approx(
        expected_rmse_mm / 1e3, rel=1e-3
    )
    # Uniform injected error across identically-shaped cases: pooled and
    # per-case-mean coincide, and per-case stderr is ~0 (both cases equal).
    assert report["per_case_mean_mm"] == pytest.approx(expected_rmse_mm, rel=1e-3)
    assert report["per_case_stderr_mm"] < 1e-3
    assert report["gate"]["band_mm"] == [11.1, 19.1]
    assert report["gate"]["pass"] is True
    assert set(report["per_case_rmse_mm"]) == set(case_ids)


def test_main_missing_metrics_file_hard_errors(tmp_path, monkeypatch):
    """No metrics-<split>.json: a hard, named error -- never a silent skip."""
    spec = _mini_spec(["test_0000"])
    monkeypatch.setattr(tool, "get_benchmark", lambda name: spec)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()

    with pytest.raises(FileNotFoundError, match="metrics-test.json"):
        tool.main(
            [
                "--run-dir",
                str(run_dir),
                "--data-root",
                str(data_root),
                "--split",
                "test",
                "--benchmark",
                "deforming_plate",
            ]
        )


def test_main_missing_case_npz_hard_errors(tmp_path, monkeypatch):
    """A case listed in metrics-<split>.json but missing its .npz: named error."""
    spec = _mini_spec(["test_0000"])
    monkeypatch.setattr(tool, "get_benchmark", lambda name: spec)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    (run_dir / "metrics-test.json").write_text(
        json.dumps({"split": "test", "cases": {"test_0000": {}}}), encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="test_0000"):
        tool.main(
            [
                "--run-dir",
                str(run_dir),
                "--data-root",
                str(data_root),
                "--split",
                "test",
                "--benchmark",
                "deforming_plate",
            ]
        )


def test_main_gate_fail_out_of_band_error(tmp_path, monkeypatch):
    """A uniform 25 mm error is out of the [11.1, 19.1] mm band: gate FAILs.

    Mirrors ``test_main_end_to_end_pooled_and_gate`` but injects a uniform
    (25, 25, 25) mm error, giving a hand value of
    sqrt((25^2 + 25^2 + 25^2) / 3) = 25.0 mm exactly -- above the gate's
    upper bound. Exercises the FAIL branch of both the JSON ``gate.pass``
    field and ``main()``'s return code (0 pass / 1 fail).
    """
    rng = np.random.default_rng(11)
    case_ids = ["test_0000", "test_0001"]
    data_root = tmp_path / "data"
    data_root.mkdir()
    for cid in case_ids:
        _write_synthetic_case(data_root, cid, rng)

    spec = _mini_spec(case_ids)
    monkeypatch.setattr(tool, "get_benchmark", lambda name: spec)

    run_dir = tmp_path / "run"
    rollout_dir = run_dir / "rollouts"
    rollout_dir.mkdir(parents=True)

    delta = np.array([25.0, 25.0, 25.0], dtype=np.float64)
    cases_record: dict[str, dict] = {}
    for cid in case_ids:
        trajectory = load_case_trajectory(
            data_root / f"{cid}.h5", aux_field="von_mises_stress"
        )
        predicted_positions = (
            trajectory.positions.astype(np.float64) + delta[None, None, :]
        ).astype(np.float32)
        np.savez(
            rollout_dir / f"test-{cid}.npz",
            predicted_positions=predicted_positions,
            predicted_aux=trajectory.aux,
        )
        cases_record[cid] = {"rollout_position_rmse": 0.0}

    (run_dir / "metrics-test.json").write_text(
        json.dumps({"split": "test", "cases": cases_record}), encoding="utf-8"
    )

    out_path = run_dir / "blessing-pooled-test.json"
    rc = tool.main(
        [
            "--run-dir",
            str(run_dir),
            "--data-root",
            str(data_root),
            "--split",
            "test",
            "--benchmark",
            "deforming_plate",
            "--out",
            str(out_path),
        ]
    )

    expected_rmse_mm = math.sqrt((25.0**2 + 25.0**2 + 25.0**2) / 3.0)
    assert expected_rmse_mm == pytest.approx(25.0)
    assert rc == 1  # outside the gate band -> FAIL

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["pooled_rmse_mm"] == pytest.approx(expected_rmse_mm, rel=1e-3)
    assert report["gate"]["band_mm"] == [11.1, 19.1]
    assert report["gate"]["pass"] is False


def test_main_missing_cases_key_hard_errors(tmp_path, monkeypatch):
    """metrics-<split>.json with no "cases" key: a named KeyError, not a crash."""
    spec = _mini_spec(["test_0000"])
    monkeypatch.setattr(tool, "get_benchmark", lambda name: spec)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    (run_dir / "metrics-test.json").write_text(
        json.dumps({"split": "test"}), encoding="utf-8"
    )

    with pytest.raises(KeyError, match="no 'cases' key"):
        tool.main(
            [
                "--run-dir",
                str(run_dir),
                "--data-root",
                str(data_root),
                "--split",
                "test",
                "--benchmark",
                "deforming_plate",
            ]
        )


def test_main_empty_case_list_hard_errors(tmp_path, monkeypatch):
    """metrics-<split>.json with an empty "cases" dict: a named ValueError."""
    spec = _mini_spec(["test_0000"])
    monkeypatch.setattr(tool, "get_benchmark", lambda name: spec)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    (run_dir / "metrics-test.json").write_text(
        json.dumps({"split": "test", "cases": {}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="lists no cases"):
        tool.main(
            [
                "--run-dir",
                str(run_dir),
                "--data-root",
                str(data_root),
                "--split",
                "test",
                "--benchmark",
                "deforming_plate",
            ]
        )


def test_main_missing_predicted_positions_key_hard_errors(tmp_path, monkeypatch):
    """A case .npz missing "predicted_positions": a named KeyError."""
    rng = np.random.default_rng(3)
    case_ids = ["test_0000"]
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_synthetic_case(data_root, case_ids[0], rng)

    spec = _mini_spec(case_ids)
    monkeypatch.setattr(tool, "get_benchmark", lambda name: spec)

    run_dir = tmp_path / "run"
    rollout_dir = run_dir / "rollouts"
    rollout_dir.mkdir(parents=True)
    # Deliberately omit "predicted_positions" -- only an unrelated key present.
    np.savez(rollout_dir / "test-test_0000.npz", predicted_aux=np.zeros((8, 6)))
    (run_dir / "metrics-test.json").write_text(
        json.dumps({"split": "test", "cases": {"test_0000": {}}}), encoding="utf-8"
    )

    with pytest.raises(KeyError, match="predicted_positions"):
        tool.main(
            [
                "--run-dir",
                str(run_dir),
                "--data-root",
                str(data_root),
                "--split",
                "test",
                "--benchmark",
                "deforming_plate",
            ]
        )
