"""End-to-end Transolver training smoke: train() -> validation -> checkpoint.

The gate this module exercises: a REAL ``train(family="transolver")`` call on
a tiny synthetic mesh benchmark, through at least one validation pass, on
CPU, deterministic, under ~60 s -- plus the ``evaluate()`` smoke on the
resulting run directory (the transolver arm of Task 6's mesh-benchmark
evaluate() gate). Mirrors ``tests/cli/test_mgn_train_smoke.py`` (ADR-0043
§8/§9a MGN parity, ADR-0041/0044 for the Transolver family).
"""

import json
import logging
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
        # ADR-0051 B: a per-case scalar so impact_velocity_feature can be
        # smoke-tested end-to-end (varies by case to exercise the broadcast).
        loading_scalar=lambda cid: 100.0 + float(cid.rsplit("_", 1)[1]),
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


def _run_transolver_smoke(
    tmp_path,
    *,
    frames_per_call: int = 1,
    impact_velocity_feature: bool = False,
    time_conditioned: bool = False,
):
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
        frames_per_call=frames_per_call,
        impact_velocity_feature=impact_velocity_feature,
        time_conditioned=time_conditioned,
        # time-conditioning is history-free / non-autoregressive: noise is inert
        noise_std=0.0 if time_conditioned else TransolverConfig().noise_std,
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


def test_transolver_oneshot_kT_train_and_evaluate_smoke(tmp_path, monkeypatch):
    """One-shot (frames_per_call=0 sentinel) end-to-end: ADR-0051 phase 2.

    Verifies the k=T scheme trains (clean full-sequence L2, noise off), the
    sentinel resolves at train time to a concrete k = T_working - input_frames
    and that RESOLVED integer is stored in config.json (so evaluate rebuilds
    the identical fixed head without re-resolving), and the bundled one-shot
    rollout + teacher-forced one-step both produce finite metrics.
    """
    import structbench.cli.train as cli_train

    spec, data_root, out, _cfg, _tcfg, ids = _run_transolver_smoke(
        tmp_path, frames_per_call=0
    )

    record = json.loads((out / "config.json").read_text(encoding="utf-8"))
    k = record["model"]["frames_per_call"]
    # sentinel resolved and stored as a concrete horizon-covering k (> 1),
    # never the raw 0 (which evaluate() could not turn back into a head shape).
    assert isinstance(k, int) and k > 1

    ckpts = list(out.glob("model-*.pt"))
    assert any(p.name.startswith("model-best-") for p in ckpts), "no val pass ran"

    # evaluate() rebuilds the k-head from config.json (no spec re-resolution)
    # and runs the one-shot rollout + teacher-forced one-step.
    monkeypatch.setattr(cli_train, "get_benchmark", lambda name: spec)
    metrics = cli_train.evaluate(ids["val"], data_root, out, "cpu", split_name="val")
    per_case = metrics["cases"][ids["val"][0]]
    assert np.isfinite(per_case["one_step_position_rmse"])
    assert np.isfinite(per_case["rollout_position_rmse"])
    assert np.isfinite(per_case["rollout_aux_rmse"])


def test_transolver_pushforward_bundling_train_and_evaluate_smoke(
    tmp_path, monkeypatch, caplog
):
    """1<k<T temporal bundling (MP-PDE pushforward) end-to-end: ADR-0051 phase 3.

    k=2 on the T=8 / input_frames=2 fixture is genuine bundling (horizon 6 > 2,
    so not one-shot), exercising both the normalizer-warmup clean-bundle branch
    (first 5 steps) and the two-forward-pass seam pushforward after it, then the
    bundled rollout + teacher-forced one-step in evaluate().
    """
    import structbench.cli.train as cli_train

    caplog.set_level(logging.WARNING, logger="structbench.cli.train")
    spec, data_root, out, _cfg, _tcfg, ids = _run_transolver_smoke(
        tmp_path, frames_per_call=2
    )

    # noise_std (the fixture's default 0.003) is inert in the pushforward
    # regime, so the trainer warns rather than let it look active (ADR-0051).
    assert "noise_std" in caplog.text and "IGNORED" in caplog.text

    record = json.loads((out / "config.json").read_text(encoding="utf-8"))
    # An explicit 1<k<T is recorded verbatim (only the k=T sentinel is resolved).
    assert record["model"]["frames_per_call"] == 2

    ckpts = list(out.glob("model-*.pt"))
    assert any(p.name.startswith("model-best-") for p in ckpts), "no val pass ran"

    monkeypatch.setattr(cli_train, "get_benchmark", lambda name: spec)
    metrics = cli_train.evaluate(ids["val"], data_root, out, "cpu", split_name="val")
    per_case = metrics["cases"][ids["val"][0]]
    assert np.isfinite(per_case["one_step_position_rmse"])
    assert np.isfinite(per_case["rollout_position_rmse"])
    assert np.isfinite(per_case["rollout_aux_rmse"])


def test_transolver_time_conditioned_train_and_evaluate_smoke(tmp_path, monkeypatch):
    """Time-conditioned (ADR-0053) end-to-end: train -> checkpoint -> evaluate.

    Exercises the native structural scheme through a REAL train() (dedicated
    history-free TC loop, per-frame query targets, include_target_frame collate)
    and the TC evaluate() branch (independent per-frame query via
    time_conditioned_rollout, one_step_* reported as null). Uses
    impact_velocity_feature=true so the scalar-conditioning path is covered too.
    """
    import structbench.cli.train as cli_train

    spec, data_root, out, _cfg, _tcfg, ids = _run_transolver_smoke(
        tmp_path, time_conditioned=True, impact_velocity_feature=True
    )

    record = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert record["model"]["time_conditioned"] is True
    assert record["model"]["frames_per_call"] == 1

    ckpts = list(out.glob("model-*.pt"))
    assert any(p.name.startswith("model-best-") for p in ckpts), "no val pass ran"

    monkeypatch.setattr(cli_train, "get_benchmark", lambda name: spec)
    metrics = cli_train.evaluate(ids["val"], data_root, out, "cpu", split_name="val")
    per_case = metrics["cases"][ids["val"][0]]
    # rollout metrics are finite; one_step_* is undefined for TC -> null.
    assert np.isfinite(per_case["rollout_position_rmse"])
    assert np.isfinite(per_case["rollout_aux_rmse"])
    assert per_case["one_step_position_rmse"] is None
    assert per_case["one_step_aux_rmse"] is None
    assert metrics["mean"]["one_step_position_rmse"] is None
    assert np.isfinite(metrics["mean"]["rollout_position_rmse"])
    # QoIs still computed and finite.
    for key in ("qoi_pred", "qoi_true", "qoi_error"):
        assert set(per_case[key]) == {"peak_vm_stress", "terminal_peak_deflection"}
        assert all(math.isfinite(v) for v in per_case[key].values())
    # metrics JSON is strict-serializable with nulls (allow_nan=False).
    assert (out / "metrics-val.json").exists()


def test_transolver_pushforward_helper_shapes_and_grad_through_bundle2():
    """The pushforward helper: warmup and seam branches, k-frame shapes, and
    gradient that flows through bundle2 (the second, with-grad forward)."""
    from structbench.cli.train import _transolver_pushforward
    from structbench.models.transolver import TransolverSimulator

    torch.manual_seed(0)
    k, P, F, dim = 2, 5, 2, 3
    sim = TransolverSimulator(
        dim=dim,
        hidden_dim=8,
        n_layers=1,
        n_heads=2,
        slice_num=2,
        frames_per_call=k,
        kinematic_types=(1,),
        scripted_types=(1,),
    )
    position_seq = torch.randn(P, F, dim)
    next_position = torch.randn(P, 2 * k, dim)  # two consecutive GT bundles
    next_aux = torch.randn(P, 2 * k)
    ptype = torch.tensor([0, 0, 1, 0, 0])
    ref = torch.randn(P, dim)
    npp = torch.tensor([P])
    is_kin = ptype == 1
    args = (sim, position_seq, next_position, next_aux, ptype, ref, npp, is_kin, k)

    # warmup branch: plain clean bundle1, (P, k, dim+1) shapes.
    pw, tw = _transolver_pushforward(*args, velocity_history=False, warmup=True)
    assert pw.shape == (P, k, dim + 1) and tw.shape == (P, k, dim + 1)

    # pushforward branch: same shapes, and the bundle2 prediction carries
    # gradient (the two-pass seam training backprops through the net).
    pp, tp = _transolver_pushforward(*args, velocity_history=False, warmup=False)
    assert pp.shape == (P, k, dim + 1) and tp.shape == (P, k, dim + 1)
    assert pp.requires_grad
    pp.sum().backward()
    grads = [p.grad for p in sim.parameters() if p.grad is not None]
    assert grads, "no gradient reached the network through the pushforward"


def test_transolver_impact_velocity_feature_train_and_evaluate_smoke(
    tmp_path, monkeypatch
):
    """ADR-0051 B: the scalar loading-param channel trains + evaluates end-to-end.

    Exercises the full plumbing: per-trajectory loading_scalars -> collate
    loading_feature -> forward_train (node_in + 1) -> val/eval bind_case scalar
    -> predict_positions broadcast.
    """
    import structbench.cli.train as cli_train
    from structbench.models.transolver import TransolverSimulator

    spec, data_root, out, _cfg, _tcfg, ids = _run_transolver_smoke(
        tmp_path, impact_velocity_feature=True
    )

    record = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert record["model"]["impact_velocity_feature"] is True

    # the checkpoint's node input carries the extra global channel (node_in+1):
    # loads only into a feature-on simulator of the matching width.
    sim = TransolverSimulator(
        dim=3,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        slice_num=8,
        impact_velocity_feature=True,
    )
    sim.load(sorted(out.glob("model-*.pt"))[-1])
    assert sim._node_normalizer._sum.shape[0] == sim._node_type_size + 3 * 3 + 1

    monkeypatch.setattr(cli_train, "get_benchmark", lambda name: spec)
    metrics = cli_train.evaluate(ids["val"], data_root, out, "cpu", split_name="val")
    per_case = metrics["cases"][ids["val"][0]]
    assert np.isfinite(per_case["one_step_position_rmse"])
    assert np.isfinite(per_case["rollout_position_rmse"])
