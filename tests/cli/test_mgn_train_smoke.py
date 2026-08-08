"""End-to-end MGN training smoke: train() -> validation -> checkpoint.

The gate this module exercises: a REAL ``train(family="mgn")`` call on a tiny
synthetic mesh benchmark, through at least one validation pass, on CPU,
deterministic, under ~60 s -- plus the ``evaluate()`` smoke on the resulting
run directory (Task 4a's no-stats-file / per-case bind-reset branch, the only
end-to-end coverage it has).
"""

import numpy as np
import torch

from structbench.benchmarks.card import BenchmarkCard
from structbench.benchmarks.registry import BenchmarkSpec
from structbench.cli.train import train
from structbench.config import MGNConfig, TrainConfig
from structbench.core import write_case
from structbench.core.io.meshgraphnets import build_deforming_plate_case
from structbench.eval import peak_nodal_aux, terminal_peak_displacement


def _mini_spec(case_ids: dict[str, list[str]]) -> BenchmarkSpec:
    qois = {
        "peak_vm_stress": peak_nodal_aux(exclude_types=(1, 3)),
        "terminal_peak_deflection": terminal_peak_displacement(exclude_types=(1, 3)),
    }
    card = BenchmarkCard(
        name="MgnSmoke",
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
        dataset_id="mgn-smoke",
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


def _run_mgn_smoke(tmp_path):
    """Shared spec/data/train setup for both smoke tests below.

    Builds the tiny synthetic mesh benchmark, writes its cases, and runs a
    REAL ``train(..., family="mgn")`` call through validation and
    checkpointing. Both tests need this identical setup -- the evaluate
    smoke re-runs it (fresh ``tmp_path``, so a fresh run dir) rather than
    sharing a fixture across tests, keeping each test independently
    runnable/debuggable.
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

    mgn = MGNConfig(
        hidden_dim=8,
        message_passing_steps=1,
        world_edge_radius=50.0,
        normalizer_warmup_steps=3,
    )
    tcfg = TrainConfig(
        benchmark="MgnSmoke",
        batch_size=2,
        training_steps=12,
        val_every=6,
    )
    out = tmp_path / "run"
    train(spec, mgn, tcfg, data_root, out, "cpu", family="mgn")
    return spec, data_root, out, mgn, tcfg, ids


def test_mgn_train_smoke(tmp_path):
    _spec, _data_root, out, _mgn, _tcfg, _ids = _run_mgn_smoke(tmp_path)

    assert (out / "config.json").exists()
    ckpts = list(out.glob("model-*.pt"))
    assert ckpts, "no checkpoint written"
    # a validation pass genuinely ran: best starts at inf, so the first val
    # always writes model-best-<step>.pt (model-final alone == dead val loop)
    assert any(p.name.startswith("model-best-") for p in ckpts), "no val pass ran"
    # normalizers actually warmed up: reload and check accumulation happened
    from structbench.models.mgn import MeshSimulator

    sim = MeshSimulator(latent=8, mp_steps=1, world_edge_radius=50.0)
    sim.load(sorted(ckpts)[-1])
    assert int(sim._target_normalizer._n_accumulations) > 0
    assert int(sim._node_normalizer._n_accumulations) > 0  # feature warmup too


def test_mgn_evaluate_smoke(tmp_path, monkeypatch):
    """evaluate() on an MGN run dir: no stats file, bind/reset per case.

    Reuses the train smoke's setup (re-running the tiny train into this
    test's own tmp_path), then monkeypatches the registry lookup so
    evaluate() resolves the synthetic spec (precedent:
    tests/cli/test_train_eval.py's unregistered-benchmark monkeypatch,
    ``monkeypatch.setattr(train_mod, "get_benchmark", lambda name: spec)``).
    """
    import structbench.cli.train as cli_train

    spec, data_root, out, _mgn, _tcfg, ids = _run_mgn_smoke(tmp_path)
    monkeypatch.setattr(cli_train, "get_benchmark", lambda name: spec)

    assert not (out / "normalization_stats.npz").exists()
    metrics = cli_train.evaluate(ids["val"], data_root, out, "cpu", split_name="val")

    # mgn is self-contained (ADR-0043 §8): no stats file was ever required.
    assert not (out / "normalization_stats.npz").exists()
    assert (out / "metrics-val.json").exists()
    assert metrics["split"] == "val"
    per_case = metrics["cases"][ids["val"][0]]
    assert np.isfinite(per_case["one_step_position_rmse"])
    assert np.isfinite(per_case["rollout_position_rmse"])
    assert np.isfinite(per_case["rollout_aux_rmse"])
