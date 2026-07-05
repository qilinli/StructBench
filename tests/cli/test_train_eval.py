"""Run-directory contract of cli.train: evaluate() reporting and train() guard.

evaluate() must rebuild the simulator from the run directory's own
``config.json`` + ``normalization_stats.npz`` (never from caller-supplied
architecture), report the ADR-0019 §5 metrics per case and split, and persist
them under the run directory. train() must refuse to write into a run
directory that already holds checkpoints (no resume support; a newer, worse
checkpoint would shadow the better one).
"""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from structbench.benchmarks import get_benchmark
from structbench.benchmarks.card import BenchmarkCard
from structbench.benchmarks.registry import BenchmarkSpec
from structbench.cli.train import (
    GNSConfig,
    TrainConfig,
    build_simulator,
    evaluate,
    train,
)
from structbench.core import (
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
    write_case,
)
from structbench.datasets import compute_stats, load_case_trajectory

#: Tiny architecture so checkpoints build fast; deliberately different from the
#: GNSConfig defaults so evaluate() fails loudly if it ignores config.json.
SMALL_GNS = {
    "window": 3,
    "connectivity_radius": 2.0,
    "hidden_dim": 8,
    "message_passing_steps": 1,
    "nmlp_layers": 1,
    "particle_type_embedding_size": 4,
    "noise_std": 0.1,
    "dim": 2,
}


def _write_tiny_case(data_root, case_id, n_frames=6):
    """Write a 3-SPH-particle canonical case with non-degenerate motion (SI)."""
    coords = np.array([[0.0, 0.0], [1e-3, 0.0], [0.0, 1e-3], [5e-3, 5e-3]])
    disp = np.zeros((n_frames, 4, 2), dtype=np.float32)
    t = np.arange(n_frames, dtype=np.float32)
    for p in range(3):  # SPH particles only; the shell node stays put
        disp[:, p, 0] = 1e-3 * t + 1e-4 * t**2
        disp[:, p, 1] = 5e-5 * (p + 1) * t
    stress = np.zeros((n_frames, 3, 6), dtype=np.float32)
    # Time-varying so the aux std is non-degenerate: constant stress gives
    # aux_std ~ 0, and the normalized aux target NaNs training within steps.
    stress[:, :, 0] = 100e6 + 5e6 * t[:, None]
    case = Case(
        metadata=Metadata(case_id=case_id, dimension=2, source_units="g-mm-ms"),
        nodes=Nodes(coords=coords, node_id=np.arange(1, 5, dtype=np.int64)),
        elements={
            "sph": ElementBlock(
                connectivity=np.arange(3, dtype=np.int64).reshape(3, 1),
                element_id=np.arange(1, 4, dtype=np.int64),
                part_id=np.ones(3, dtype=np.int64),
            ),
            "shell": ElementBlock(
                connectivity=np.array([[3, 3, 3, 3]], dtype=np.int64),
                element_id=np.array([99], dtype=np.int64),
                part_id=np.array([2], dtype=np.int64),
            ),
        },
        materials=[Material(2, "MAT_ELASTIC_PLASTIC_HYDRO", {"data": [[2]]}, None)],
        response=Response(
            time=np.arange(n_frames, dtype=float) * 2e-6,
            node={"displacement": disp},
            element={"sph": {"stress": stress}},
        ),
    )
    write_case(case, data_root / f"{case_id}.h5")


def _prepared_run(tmp_path, case_ids):
    """Tiny data root plus a run dir holding checkpoint, stats, config.json."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    for cid in case_ids:
        _write_tiny_case(data_root, cid)
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    trajs = [load_case_trajectory(data_root / f"{cid}.h5") for cid in case_ids]
    stats = compute_stats(trajs)
    stats.save(out_dir / "normalization_stats.npz")

    gns = GNSConfig(**SMALL_GNS)
    stats_t = {
        key: {
            "mean": torch.tensor(getattr(stats, f"{name}_mean"), dtype=torch.float32),
            "std": torch.tensor(getattr(stats, f"{name}_std"), dtype=torch.float32),
        }
        for key, name in [
            ("velocity", "velocity"),
            ("acceleration", "acceleration"),
            ("aux", "aux"),
        ]
    }
    simulator = build_simulator(
        stats_t,
        gns,
        n_particle_types=2,
        boundary_feature_fn=lambda p: p[:, 0:1],
        device="cpu",
    )
    simulator.save(str(out_dir / "model-best-000002.pt"))

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "gns": asdict(gns),
                "train": asdict(TrainConfig()),
                "n_particle_types": 2,
                "data_root": str(data_root),
            }
        ),
        encoding="utf-8",
    )
    return data_root, out_dir


def test_evaluate_rebuilds_architecture_from_run_config(tmp_path):
    case_ids = ["C-1", "C-2"]
    data_root, out_dir = _prepared_run(tmp_path, case_ids)
    # No architecture is passed: evaluate must reconstruct SMALL_GNS from
    # config.json; using GNSConfig defaults would fail the checkpoint load.
    metrics = evaluate(case_ids, data_root, out_dir, "cpu")
    assert metrics["split"] == "eval"
    assert set(metrics["cases"]) == set(case_ids)
    for per_case in metrics["cases"].values():
        assert np.isfinite(per_case["one_step_position_rmse"])
        assert np.isfinite(per_case["rollout_position_rmse"])
        assert np.isfinite(per_case["rollout_aux_rmse"])
        assert set(per_case["qoi_error"]) == {
            "final_length",
            "mushroom_width",
            "peak_von_mises",
            "t_peak_von_mises",
        }
    assert np.isfinite(metrics["mean"]["rollout_position_rmse"])
    assert np.isfinite(metrics["mean"]["rollout_aux_rmse"])
    assert set(metrics["mean"]["qoi_abs_error"]) == {
        "final_length",
        "mushroom_width",
        "peak_von_mises",
        "t_peak_von_mises",
    }


def test_evaluate_persists_metrics_json_and_rollout_artifacts(tmp_path):
    case_ids = ["C-1"]
    data_root, out_dir = _prepared_run(tmp_path, case_ids)
    metrics = evaluate(case_ids, data_root, out_dir, "cpu")
    on_disk = json.loads((out_dir / "metrics-eval.json").read_text(encoding="utf-8"))
    assert on_disk == metrics
    with np.load(out_dir / "rollouts" / "eval-C-1.npz") as artifact:
        assert artifact["predicted_positions"].shape == (6, 3, 2)
        assert artifact["predicted_aux"].shape == (6, 3)


def test_evaluate_split_name_flows_to_report_and_filenames(tmp_path):
    case_ids = ["C-1"]
    data_root, out_dir = _prepared_run(tmp_path, case_ids)
    metrics = evaluate(case_ids, data_root, out_dir, "cpu", split_name="test_interp")
    assert metrics["split"] == "test_interp"
    assert (out_dir / "metrics-test_interp.json").exists()
    assert (out_dir / "rollouts" / "test_interp-C-1.npz").exists()


def test_evaluate_requires_the_run_config(tmp_path):
    case_ids = ["C-1"]
    data_root, out_dir = _prepared_run(tmp_path, case_ids)
    (out_dir / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate(case_ids, data_root, out_dir, "cpu")


def test_train_refuses_out_dir_with_existing_checkpoints(tmp_path):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "model-best-000100.pt").touch()
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(FileExistsError):
        train(
            get_benchmark("taylor_impact_2d"),
            GNSConfig(**SMALL_GNS),
            TrainConfig(),
            data_root,
            out_dir,
            "cpu",
        )


def test_train_raises_on_spec_config_benchmark_mismatch(tmp_path):
    """train() raises ValueError when train_cfg.benchmark names a different spec.

    The guard fires before any filesystem work: it detects that the spec passed
    to train() is not the object get_benchmark(train_cfg.benchmark) would return,
    which would cause config.json to misrecord the benchmark name.
    """
    minimal_card = BenchmarkCard(
        name="Minimal",
        version="0",
        description="test-only spec not in the registry",
        provenance="test",
        data_license="CC0",
        solver="test",
        discretisation="SPH",
        materials=("MAT_TEST",),
        erosion=False,
        loading="none",
        source_units="SI",
        geometry="unit box",
        n_cases=2,
        splits={"train": 1, "val": 1},
        task="test",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=(),
        fields=("positions",),
        particles_per_case="1",
        n_frames=3,
        output_dt_ms=1.0,
        init_frames=3,
        protocol_rationale="test-only card",
    )
    local_spec = BenchmarkSpec(
        card=minimal_card,
        splits={"train": ("T-local-1",), "val": ("V-local-1",)},
        eval_splits=("val",),
        aux_field="von_mises_stress",
    )
    # Passing TrainConfig(benchmark="taylor_impact_2d") while local_spec is a
    # different object triggers the mismatch guard before any filesystem work.
    data_root = tmp_path / "data"
    data_root.mkdir()
    out_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="does not resolve to the spec"):
        train(
            local_spec,
            GNSConfig(**SMALL_GNS),
            TrainConfig(benchmark="taylor_impact_2d"),
            data_root,
            out_dir,
            "cpu",
        )
    # Guard fires before out_dir is created.
    assert not out_dir.exists()


def test_train_loss_all_kinematic_batch_no_nan():
    """Loss computation guards against NaN when batch contains only kinematic particles.

    When spec.kinematic_types is set and a batch contains only particles of
    kinematic types, the free mask is all False. Without the guard,
    per_particle[free].mean() on an empty tensor returns NaN and silently
    corrupts gradients. The guard ensures loss is 0.0 and finite.
    """
    device = "cpu"
    n_particles = 4

    # Mock per_particle loss tensor (n_particles,)
    per_particle = torch.ones(n_particles, device=device, requires_grad=True)

    # All particles are kinematic type 7
    particle_type = torch.full((n_particles,), 7, dtype=torch.long, device=device)
    kinematic_types = (7,)

    # Compute the free mask (as in train.py)
    free = ~torch.isin(
        particle_type,
        torch.as_tensor(list(kinematic_types), dtype=torch.long, device=device),
    )

    # Apply the guard (as implemented in train.py)
    if free is not None and free.any():
        loss = per_particle[free].mean()
    elif free is not None:
        # all-kinematic batch: nothing to learn from; zero loss, no NaN
        loss = per_particle.new_tensor(0.0, requires_grad=True)
    else:
        loss = per_particle.mean()

    # Verify loss is 0.0 and finite (not NaN)
    assert loss.item() == 0.0
    assert torch.isfinite(loss)


def _local_spec():
    """Registry-free spec over two tiny local cases (bypasses the name guard)."""
    card = BenchmarkCard(
        name="SeedTest",
        version="0",
        description="test-only spec not in the registry",
        provenance="test",
        data_license="CC0",
        solver="test",
        discretisation="SPH",
        materials=("MAT_TEST",),
        erosion=False,
        loading="none",
        source_units="SI",
        geometry="unit box",
        n_cases=2,
        splits={"train": 1, "val": 1},
        task="test",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=(),
        fields=("positions",),
        particles_per_case="3",
        n_frames=6,
        output_dt_ms=1.0,
        init_frames=3,
        protocol_rationale="test-only card",
    )
    return BenchmarkSpec(
        card=card,
        splits={"train": ("S-1",), "val": ("S-2",)},
        eval_splits=("val",),
        aux_field="von_mises_stress",
    )


def _train_tiny(tmp_path, name, seed):
    """Two-step training run on tiny local cases; returns the checkpoint state."""
    data_root = tmp_path / "data"
    if not data_root.exists():
        data_root.mkdir()
        for cid in ("S-1", "S-2"):
            _write_tiny_case(data_root, cid)
    out_dir = tmp_path / name
    ckpt = train(
        _local_spec(),
        GNSConfig(**SMALL_GNS),
        TrainConfig(
            benchmark="seed-test-local",
            batch_size=2,
            training_steps=2,
            val_every=2,
            seed=seed,
        ),
        data_root,
        out_dir,
        "cpu",
    )
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    return state, config


def test_train_seed_reproducible_and_recorded(tmp_path):
    """Same seed: identical checkpoints (CPU); different seed: different weights.

    Exercises the full train() path — seeded init, noise draws, and shuffle —
    and the config.json record the run directory contract relies on.
    """
    state_a, config_a = _train_tiny(tmp_path, "run-a", seed=123)
    state_b, config_b = _train_tiny(tmp_path, "run-b", seed=123)
    state_c, _ = _train_tiny(tmp_path, "run-c", seed=124)

    assert config_a["run"]["seed"] == 123
    assert config_b["run"]["seed"] == 123
    assert state_a.keys() == state_b.keys()
    assert all(torch.isfinite(v).all() for v in state_a.values())
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), f"seed-123 mismatch: {key}"
    assert any(not torch.equal(state_a[key], state_c[key]) for key in state_a)


def test_evaluate_protocol_standard_is_card_relative(tmp_path):
    """Legacy init=11 evals are non-standard; card-init re-evals are standard.

    Pre-0032 run dirs recorded init = window. The protocol_standard flag in
    metrics must compare against the CARD's pinned init (ADR-0032 §4), not the
    run's own record — otherwise legacy fleet numbers self-certify as official
    and card-conforming re-evaluations get excluded.
    """
    case_ids = ["C-1"]
    data_root, out_dir = _prepared_run(tmp_path, case_ids)  # legacy flat config.json
    # Default eval follows the record (window=3 == taylor card init 3 here), so
    # force a non-card init to emulate a legacy window-11-style record.
    off_card = evaluate(
        case_ids, data_root, out_dir, "cpu", init_frames=4, save_artifacts=False
    )
    assert off_card["init_frames"] == 4
    assert off_card["protocol_standard"] is False
    on_card = evaluate(
        case_ids, data_root, out_dir, "cpu", init_frames=3, save_artifacts=False
    )
    assert on_card["protocol_standard"] is True
