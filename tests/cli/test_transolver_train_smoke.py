"""End-to-end Transolver training smoke: train() -> validation -> checkpoint.

The gate this module exercises: a REAL ``train(family="transolver")`` call on
a tiny synthetic mesh benchmark, through at least one validation pass, on
CPU, deterministic, under ~60 s -- plus the ``evaluate()`` smoke on the
resulting run directory (the transolver arm of Task 6's mesh-benchmark
evaluate() gate). Mirrors ``tests/cli/test_mgn_train_smoke.py`` (ADR-0043
§8/§9a MGN parity, ADR-0041/0044 for the Transolver family).
"""

import json
import math

import numpy as np
import torch

from structbench.benchmarks.card import BenchmarkCard
from structbench.benchmarks.registry import BenchmarkSpec
from structbench.cli.train import train
from structbench.config import TrainConfig, TransolverConfig
from structbench.core import write_case
from structbench.core.io.meshgraphnets import build_deforming_plate_case
from structbench.eval import peak_nodal_aux, terminal_peak_displacement


def _mini_spec(case_ids: dict[str, list[str]]) -> BenchmarkSpec:
    qois = {
        "peak_vm_stress": peak_nodal_aux(exclude_types=(1, 3)),
        "terminal_peak_deflection": terminal_peak_displacement(exclude_types=(1, 3)),
    }
    card = BenchmarkCard(
        name="TransolverSmoke",
        version="0.0",
        description="synthetic smoke benchmark",
        provenance="synthetic (test fixture)",
        data_license="n/a (synthetic test data)",
        solver="COMSOL",
        discretisation="FEM",
        materials=("synthetic",),
        loading="synthetic actuator",
        erosion=False,
        source_units="kg-m-s",
        geometry="synthetic tets",
        n_cases=sum(len(v) for v in case_ids.values()),
        splits={k: len(v) for k, v in case_ids.items()},
        task="smoke",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=tuple(qois),
        fields=("node/displacement", "node/von_mises_stress"),
        particles_per_case="6-6",
        n_frames=8,
        output_dt_ms=1.0,
        input_frames=2,
        protocol_rationale="synthetic smoke fixture; not a benchmark",
    )
    return BenchmarkSpec(
        card=card,
        splits={k: tuple(v) for k, v in case_ids.items()},
        eval_splits=("val",),
        aux_field="von_mises_stress",
        qois=qois,
        boundary_feature_fn=None,
        dataset_id="transolver-smoke",
        kinematic_types=(1, 3),
    )


def _write_cases(root, ids):
    rng = np.random.default_rng(11)
    for cid in ids:
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
        case = build_deforming_plate_case(arrays, source_units="kg-m-s", case_id=cid)
        write_case(case, root / f"{cid}.h5")


def _run_transolver_smoke(tmp_path):
    """Shared spec/data/train setup for both smoke tests below.

    Builds the tiny synthetic mesh benchmark, writes its cases, and runs a
    REAL ``train(..., family="transolver")`` call through validation and
    checkpointing. Both tests need this identical setup -- the evaluate
    smoke re-runs it (fresh ``tmp_path``, so a fresh run dir) rather than
    sharing a fixture across tests, keeping each test independently
    runnable/debuggable.

    Architecture sizes (hidden_dim=16, n_layers=2, n_heads=2, slice_num=8)
    match ``configs/deforming_plate/transolver_smoke.toml``; training_steps=50
    and val_every=25 stay below ``PERIODIC_CKPT_EVERY`` (10_000, unpatched),
    so no periodic ``ckpt-<step>.pt`` snapshot is expected here (unlike the
    MGN smoke, which patches the cadence down to reach it in 12 steps).
    """
    torch.manual_seed(0)
    ids = {
        "train": [f"train_{i:04d}" for i in range(4)],
        "val": ["val_0000"],
    }
    spec = _mini_spec(ids)
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_cases(data_root, [c for v in ids.values() for c in v])

    cfg = TransolverConfig(
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        slice_num=8,
        normalizer_warmup_steps=5,
    )
    tcfg = TrainConfig(
        benchmark="TransolverSmoke",
        batch_size=2,
        training_steps=50,
        val_every=25,
    )
    out = tmp_path / "run"
    train(spec, cfg, tcfg, data_root, out, "cpu", family="transolver")
    return spec, data_root, out, cfg, tcfg, ids


def test_transolver_train_smoke(tmp_path):
    _spec, _data_root, out, _cfg, _tcfg, _ids = _run_transolver_smoke(tmp_path)

    assert (out / "config.json").exists()
    record = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert record["model"]["family"] == "transolver"

    ckpts = list(out.glob("model-*.pt"))
    assert ckpts, "no checkpoint written"
    # a validation pass genuinely ran: best starts at inf, so the first val
    # always writes model-best-<step>.pt (model-final alone == dead val loop)
    assert any(p.name.startswith("model-best-") for p in ckpts), "no val pass ran"
    # training_steps=50 never crosses the default PERIODIC_CKPT_EVERY=10_000
    # cadence (left unpatched here), so no periodic snapshot is expected.
    assert not list(out.glob("ckpt-*.pt")), "unexpected periodic checkpoint"
    # normalizers actually warmed up: reload and check accumulation happened
    from structbench.models.transolver import TransolverSimulator

    sim = TransolverSimulator(dim=3, hidden_dim=16, n_layers=2, n_heads=2, slice_num=8)
    sim.load(sorted(ckpts)[-1])
    assert int(sim._node_normalizer._n_accumulations) > 0
    assert int(sim._target_normalizer._n_accumulations) > 0


def test_transolver_evaluate_smoke(tmp_path, monkeypatch):
    """evaluate() on a Transolver run dir: no stats file, bind/reset per case.

    Reuses the train smoke's setup (re-running the tiny train into this
    test's own tmp_path), then monkeypatches the registry lookup so
    evaluate() resolves the synthetic spec (precedent:
    tests/cli/test_train_eval.py's unregistered-benchmark monkeypatch,
    ``monkeypatch.setattr(train_mod, "get_benchmark", lambda name: spec)``).
    """
    import structbench.cli.train as cli_train

    spec, data_root, out, _cfg, _tcfg, ids = _run_transolver_smoke(tmp_path)
    monkeypatch.setattr(cli_train, "get_benchmark", lambda name: spec)

    assert not (out / "normalization_stats.npz").exists()
    metrics = cli_train.evaluate(ids["val"], data_root, out, "cpu", split_name="val")

    # transolver is self-contained (ADR-0041/0044, MGN parity): no stats
    # file was ever required.
    assert not (out / "normalization_stats.npz").exists()
    assert (out / "metrics-val.json").exists()
    assert metrics["split"] == "val"
    per_case = metrics["cases"][ids["val"][0]]
    assert np.isfinite(per_case["one_step_position_rmse"])
    assert np.isfinite(per_case["rollout_position_rmse"])
    assert np.isfinite(per_case["rollout_aux_rmse"])
    # per-case QoI triad: catches a NaN silently laundered to None in
    # cli.train's _json_safe before it ever reaches the mean aggregation.
    for key in ("qoi_pred", "qoi_true", "qoi_error"):
        assert set(per_case[key]) == {"peak_vm_stress", "terminal_peak_deflection"}
        assert all(math.isfinite(v) for v in per_case[key].values())

    # _mean_over_cases aggregation: the split mean these core metrics feed
    # into is exercised nowhere else end-to-end.
    mean = metrics["mean"]
    for key in ("one_step_position_rmse", "rollout_position_rmse", "rollout_aux_rmse"):
        assert isinstance(mean[key], float) and math.isfinite(mean[key])
    qoi_abs_error = mean["qoi_abs_error"]
    assert set(qoi_abs_error) == {"peak_vm_stress", "terminal_peak_deflection"}
    assert all(
        isinstance(v, float) and math.isfinite(v) for v in qoi_abs_error.values()
    )
