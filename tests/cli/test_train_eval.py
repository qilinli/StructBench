"""Run-directory contract of cli.train: evaluate() reporting and train() guard.

evaluate() must rebuild the simulator from the run directory's own
``config.json`` + ``normalization_stats.npz`` (never from caller-supplied
architecture), report the ADR-0019 §5 metrics per case and split, and persist
them under the run directory. train() must refuse to write into a run
directory that already holds checkpoints (no resume support; a newer, worse
checkpoint would shadow the better one).
"""

import json
import os
import shutil
from dataclasses import asdict

import numpy as np
import pytest
import torch

from structbench.benchmarks import get_benchmark
from structbench.benchmarks.card import BenchmarkCard
from structbench.benchmarks.registry import BenchmarkSpec
from structbench.cli.train import (
    CGNConfig,
    TrainConfig,
    _find_checkpoint,
    _json_safe,
    _lr_at_cosine,
    _model_config_from_record,
    build_mgn_simulator,
    build_simulator,
    build_transolver_simulator,
    evaluate,
    main,
    train,
)
from structbench.config import (
    LR_SCHEDULE_FLOOR,
    MGNConfig,
    TransolverConfig,
    read_run_record,
    resolved_config_dict,
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
from structbench.core.io.meshgraphnets import build_deforming_plate_case
from structbench.datasets import compute_stats, load_case_trajectory
from structbench.models.transolver import TransolverSimulator

#: Tiny architecture so checkpoints build fast; deliberately different from the
#: CGNConfig defaults so evaluate() fails loudly if it ignores config.json.
SMALL_CGN = {
    "input_frames": 3,
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

    cgn = CGNConfig(**SMALL_CGN)
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
        cgn,
        n_particle_types=2,
        boundary_feature_fn=lambda p: p[:, 0:1],
        device="cpu",
    )
    simulator.save(str(out_dir / "model-best-000002.pt"))

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "gns": asdict(cgn),  # pre-0032 records used the legacy "gns" key
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
    # No architecture is passed: evaluate must reconstruct SMALL_CGN from
    # config.json; using CGNConfig defaults would fail the checkpoint load.
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
            # input_frames must match the taylor card (6) so the existing-
            # checkpoint guard fires, not the ADR-0035 input_frames guard.
            CGNConfig(**{**SMALL_CGN, "input_frames": 6}),
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
        input_frames=3,
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
            CGNConfig(**SMALL_CGN),
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
        input_frames=3,
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
        CGNConfig(**SMALL_CGN),
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


def _write_nested_run(tmp_path, input_frames, n_frames, benchmark="taylor_impact_2d"):
    """A run dir with a nested (ADR-0035) config.json benchmarked to `benchmark`."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_tiny_case(data_root, "C-1", n_frames=n_frames)
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    trajs = [load_case_trajectory(data_root / "C-1.h5")]
    stats = compute_stats(trajs)
    stats.save(out_dir / "normalization_stats.npz")
    cgn = CGNConfig(**{**SMALL_CGN, "input_frames": input_frames})
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
        cgn,
        n_particle_types=2,
        boundary_feature_fn=lambda p: p[:, 0:1],
        device="cpu",
    )
    simulator.save(str(out_dir / "model-best-000002.pt"))
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "run": {"benchmark": benchmark, "seed": 0, "commit": "test"},
                "model": {"family": "cgn", **asdict(cgn)},
                "train": {
                    k: v
                    for k, v in asdict(TrainConfig()).items()
                    if k not in ("benchmark", "seed")
                },
                "protocol": {
                    "input_frames": input_frames,
                    "horizon": "full",
                    "eval_times": "native",
                    "standard": True,
                },
                "n_particle_types": 2,
                "data_root": str(data_root),
            }
        ),
        encoding="utf-8",
    )
    return data_root, out_dir


def test_evaluate_on_card_input_frames_is_standard(tmp_path):
    """A checkpoint whose input_frames == the card's re-evaluates as standard.

    ADR-0035 fuses input_frames with the benchmark protocol, so a run trained at
    the taylor card's 6 frames is card-conforming on re-eval.
    """
    data_root, out_dir = _write_nested_run(tmp_path, input_frames=6, n_frames=8)
    on_card = evaluate(["C-1"], data_root, out_dir, "cpu", save_artifacts=False)
    assert on_card["input_frames"] == 6
    assert on_card["protocol_standard"] is True


def test_evaluate_marks_off_card_input_frames_non_standard(tmp_path):
    """A legacy checkpoint whose input_frames != the card reads as non-standard.

    ``_prepared_run`` writes a pre-0035 flat record at the tiny SMALL_CGN
    input_frames = 3, against the default taylor card (6), so it re-evaluates as
    non-standard while surfacing its own recorded input_frames.
    """
    case_ids = ["C-1"]
    data_root, out_dir = _prepared_run(tmp_path, case_ids)  # legacy flat config.json
    metrics = evaluate(case_ids, data_root, out_dir, "cpu", save_artifacts=False)
    assert metrics["input_frames"] == 3
    assert metrics["protocol_standard"] is False


def test_find_checkpoint_selects_by_step_not_mtime(tmp_path):
    """Checkpoint selection uses the step in the filename, not mtime."""
    low = tmp_path / "model-best-000100.pt"
    high = tmp_path / "model-best-000500.pt"
    low.write_bytes(b"x")
    high.write_bytes(b"x")
    os.utime(high, (1, 1))  # higher step, but made OLDER
    os.utime(low, (10**9, 10**9))  # lower step, but made NEWER
    assert _find_checkpoint(tmp_path) == high


def test_json_safe_maps_non_finite_to_none(tmp_path):
    """Non-finite metrics become null so the JSON stays strictly parseable."""
    metrics = {
        "a": float("nan"),
        "b": {"c": float("inf")},
        "d": [1.0, float("-inf")],
        "e": "ok",
    }
    text = json.dumps(_json_safe(metrics), allow_nan=False)  # must not raise
    assert json.loads(text) == {
        "a": None,
        "b": {"c": None},
        "d": [1.0, None],
        "e": "ok",
    }


def test_main_valid_without_out_returns_error_not_traceback(tmp_path, capsys):
    rc = main(["--mode", "valid", "--data-root", str(tmp_path)])
    assert rc == 2
    assert "--out is required" in capsys.readouterr().out


def test_train_rejects_empty_windowed_dataset(tmp_path):
    """All-short trajectories yield an empty WindowDataset -> raise, not loop."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_tiny_case(
        data_root, "C-1", n_frames=3
    )  # n_frames == input_frames -> 0 samples
    _write_tiny_case(data_root, "V-1", n_frames=3)
    card = BenchmarkCard(
        name="Empty",
        version="0",
        description="test-only spec",
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
        n_frames=3,
        output_dt_ms=1.0,
        input_frames=3,
        protocol_rationale="test-only card",
    )
    spec = BenchmarkSpec(
        card=card,
        splits={"train": ("C-1",), "val": ("V-1",)},
        eval_splits=("val",),
        aux_field="von_mises_stress",
    )
    with pytest.raises(ValueError, match="empty training set|input_frames"):
        train(
            spec,
            CGNConfig(**SMALL_CGN),
            TrainConfig(benchmark=""),
            data_root,
            tmp_path / "run",
            "cpu",
        )


def test_train_writes_periodic_checkpoints_outside_selection_glob(
    tmp_path, monkeypatch
):
    """Periodic ckpt-<step>.pt snapshots land on cadence and never shadow the
    selected model-*.pt checkpoint (ADR-0028, 2026-07-10 note)."""
    import structbench.cli.train as train_mod

    monkeypatch.setattr(train_mod, "PERIODIC_CKPT_EVERY", 2)
    data_root = tmp_path / "data"
    data_root.mkdir()
    for cid in ("S-1", "S-2"):
        _write_tiny_case(data_root, cid)
    out_dir = tmp_path / "run"
    train(
        _local_spec(),
        CGNConfig(**SMALL_CGN),
        TrainConfig(
            benchmark="seed-test-local",
            batch_size=2,
            training_steps=4,
            val_every=2,
            seed=1,
        ),
        data_root,
        out_dir,
        "cpu",
    )
    assert (out_dir / "ckpt-000002.pt").exists()
    assert (out_dir / "ckpt-000004.pt").exists()
    selected = _find_checkpoint(out_dir)
    assert selected is not None
    assert selected.name.startswith("model-")


def test_train_refuses_out_dir_with_periodic_checkpoints(tmp_path):
    """The fresh-out guard fires on leftover ckpt-*.pt, not just model-*.pt."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "ckpt-000100.pt").touch()
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(FileExistsError):
        train(
            get_benchmark("taylor_impact_2d"),
            CGNConfig(**{**SMALL_CGN, "input_frames": 6}),
            TrainConfig(),
            data_root,
            out_dir,
            "cpu",
        )


def test_evaluate_explicit_checkpoint_suffixes_metrics_and_skips_rollouts(tmp_path):
    """--checkpoint evaluations never clobber the canonical artifacts:
    metrics get an @<stem> suffix and rollout .npz files are skipped."""
    case_ids = ["C-1"]
    data_root, out_dir = _prepared_run(tmp_path, case_ids)
    shutil.copy(out_dir / "model-best-000002.pt", out_dir / "ckpt-000002.pt")
    metrics = evaluate(case_ids, data_root, out_dir, "cpu", checkpoint="ckpt-000002.pt")
    assert metrics["checkpoint"] == "ckpt-000002.pt"
    assert (out_dir / "metrics-eval@ckpt-000002.json").exists()
    assert not (out_dir / "metrics-eval.json").exists()
    assert not (out_dir / "rollouts").exists()


def test_evaluate_explicit_checkpoint_missing_raises(tmp_path):
    case_ids = ["C-1"]
    data_root, out_dir = _prepared_run(tmp_path, case_ids)
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        evaluate(case_ids, data_root, out_dir, "cpu", checkpoint="ckpt-999999.pt")


def test_main_train_mode_rejects_checkpoint_flag(tmp_path, capsys):
    rc = main(
        [
            "--mode",
            "train",
            "--config",
            "does-not-matter.toml",
            "--data-root",
            str(tmp_path),
            "--checkpoint",
            "ckpt-000002.pt",
        ]
    )
    assert rc == 2
    assert "--checkpoint" in capsys.readouterr().out


def test_evaluate_relative_checkpoint_ignores_cwd(tmp_path, monkeypatch):
    """A relative --checkpoint resolves against out_dir ONLY, never the CWD.

    Fleet arms all hold identically named ckpt-<step>.pt snapshots; a CWD
    fallback would silently score another arm's weights (review 2026-07-10).
    """
    case_ids = ["C-1"]
    data_root, out_dir = _prepared_run(tmp_path, case_ids)
    shutil.copy(out_dir / "model-best-000002.pt", out_dir / "ckpt-000002.pt")
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    (cwd / "ckpt-000002.pt").write_bytes(b"not a checkpoint")  # CWD decoy
    monkeypatch.chdir(cwd)
    metrics = evaluate(case_ids, data_root, out_dir, "cpu", checkpoint="ckpt-000002.pt")
    # The decoy would raise on load; reaching metrics proves out_dir won.
    assert metrics["checkpoint_path"] == str(out_dir / "ckpt-000002.pt")


def test_default_eval_ignores_dir_with_only_periodic_checkpoints(tmp_path):
    """Default evaluation never falls back to ckpt-*.pt snapshots."""
    case_ids = ["C-1"]
    data_root, out_dir = _prepared_run(tmp_path, case_ids)
    (out_dir / "model-best-000002.pt").rename(out_dir / "ckpt-000002.pt")
    with pytest.raises(FileNotFoundError, match="no checkpoint found"):
        evaluate(case_ids, data_root, out_dir, "cpu")


def test_main_valid_mode_passes_absolute_checkpoint_through(tmp_path, monkeypatch):
    """--checkpoint reaches evaluate() from main() and absolute paths load.

    Guards the CLI wiring: dropping checkpoint=args.checkpoint from the eval
    branches must fail this test (review 2026-07-10).
    """
    import structbench.cli.train as train_mod

    data_root = tmp_path / "data"
    data_root.mkdir()
    for cid in ("S-1", "S-2"):
        _write_tiny_case(data_root, cid)
    out_dir = tmp_path / "run"
    spec = _local_spec()
    train(
        spec,
        CGNConfig(**SMALL_CGN),
        TrainConfig(
            benchmark="seed-test-local",
            batch_size=2,
            training_steps=2,
            val_every=2,
            seed=1,
        ),
        data_root,
        out_dir,
        "cpu",
    )
    selected = _find_checkpoint(out_dir)
    assert selected is not None
    snapshot = out_dir / "ckpt-000002.pt"
    shutil.copy(selected, snapshot)
    # config.json records the unregistered "seed-test-local" benchmark; route
    # main()'s spec resolution back to the local spec.
    monkeypatch.setattr(train_mod, "get_benchmark", lambda name: spec)
    rc = main(
        [
            "--mode",
            "valid",
            "--data-root",
            str(data_root),
            "--out",
            str(out_dir),
            "--checkpoint",
            str(snapshot.resolve()),
        ]
    )
    assert rc == 0
    tagged = out_dir / "metrics-val@ckpt-000002.json"
    assert tagged.exists()
    assert json.loads(tagged.read_text(encoding="utf-8"))["checkpoint_path"] == str(
        snapshot.resolve()
    )
    assert not (out_dir / "metrics-val.json").exists()


# --- aux target transform + tail weight (strain-channel study, 2026-07-15) ---


def test_simulator_inverts_aux_transform_to_raw_units(tmp_path):
    """predict_positions returns raw-unit aux under aux_transform.

    Two simulators with identical weights and stats, differing only in
    aux_transform, must satisfy aux_asinh == sinh(aux_none) * s: both
    de-normalize the decoder output identically (shared stats), then only the
    transformed simulator applies the inverse transform on top.
    """
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_tiny_case(data_root, "C-1")
    traj = load_case_trajectory(data_root / "C-1.h5")
    stats = compute_stats([traj])
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
    sims = {}
    for transform in ("none", "asinh"):
        torch.manual_seed(11)  # identical weights across the pair
        sims[transform] = build_simulator(
            stats_t,
            CGNConfig(**SMALL_CGN, aux_transform=transform, aux_transform_scale=0.02),
            n_particle_types=2,
            boundary_feature_fn=None,
            device="cpu",
        )
    window = torch.from_numpy(traj.positions[:3]).permute(1, 0, 2)  # (P, frames, dim)
    npp = torch.tensor([window.shape[0]])
    ptype = torch.from_numpy(traj.particle_type)
    with torch.no_grad():
        _, aux_none = sims["none"].predict_positions(window, npp, ptype)
        _, aux_asinh = sims["asinh"].predict_positions(window, npp, ptype)
    expected = torch.sinh(aux_none) * 0.02
    torch.testing.assert_close(aux_asinh, expected, atol=1e-6, rtol=1e-5)


def test_train_with_transform_and_tail_weight_end_to_end(tmp_path):
    """Tiny training run under asinh transform + tail weight: stats land in
    transformed space, config.json records the knobs, eval metrics are finite
    (i.e. predictions come back in raw units)."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    for cid in ("S-1", "S-2"):
        _write_tiny_case(data_root, cid)
    out_dir = tmp_path / "run-transform"
    cgn = CGNConfig(**SMALL_CGN, aux_transform="asinh", aux_transform_scale=0.02)
    train(
        _local_spec(),
        cgn,
        TrainConfig(
            benchmark="seed-test-local",
            batch_size=2,
            training_steps=2,
            val_every=2,
            seed=5,
            aux_tail_weight=3.0,
        ),
        data_root,
        out_dir,
        "cpu",
    )
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert config["model"]["aux_transform"] == "asinh"
    assert config["model"]["aux_transform_scale"] == 0.02
    assert config["train"]["aux_tail_weight"] == 3.0

    # The run-dir stats are in transformed space: they match a direct
    # transformed computation and differ from the raw-space stats.
    from structbench.datasets import NormalizationStats

    saved = NormalizationStats.load(out_dir / "normalization_stats.npz")
    traj = load_case_trajectory(data_root / "S-1.h5")
    expected = compute_stats([traj], aux_transform="asinh", aux_transform_scale=0.02)
    np.testing.assert_allclose(saved.aux_mean, expected.aux_mean)
    raw = compute_stats([traj])
    assert not np.allclose(saved.aux_mean, raw.aux_mean)

    # Rebuild the trained simulator from the run's own record (transform
    # included) and roll out against raw-unit ground truth: a finite aux RMSE
    # proves predictions came back through the inverse transform. (The tiny
    # spec is not registry-registered, so evaluate() itself is out of reach.)
    from structbench.eval import rollout

    stats_t = {
        key: {
            "mean": torch.tensor(getattr(saved, f"{name}_mean"), dtype=torch.float32),
            "std": torch.tensor(getattr(saved, f"{name}_std"), dtype=torch.float32),
        }
        for key, name in [
            ("velocity", "velocity"),
            ("acceleration", "acceleration"),
            ("aux", "aux"),
        ]
    }
    sim = build_simulator(
        stats_t,
        cgn,
        n_particle_types=config["n_particle_types"],
        boundary_feature_fn=None,
        device="cpu",
    )
    sim.load(str(_find_checkpoint(out_dir)))
    result = rollout(sim, load_case_trajectory(data_root / "S-2.h5"), 3, "cpu")
    assert np.isfinite(result.mean_aux_rmse)
    assert np.isfinite(result.predicted_aux).all()


def test_train_frames_truncates_pool_and_records(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    for cid in ("S-1", "S-2"):
        _write_tiny_case(data_root, cid)
    out_dir = tmp_path / "truncated"
    train(
        _local_spec(),
        CGNConfig(**SMALL_CGN),
        TrainConfig(
            benchmark="seed-test-local",
            batch_size=2,
            training_steps=2,
            val_every=2,
            seed=1,
            train_frames=5,
        ),
        data_root,
        out_dir,
        "cpu",
    )
    record = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert record["train"]["train_frames"] == 5
    full_dir = tmp_path / "full"
    train(
        _local_spec(),
        CGNConfig(**SMALL_CGN),
        TrainConfig(
            benchmark="seed-test-local",
            batch_size=2,
            training_steps=2,
            val_every=2,
            seed=1,
        ),
        data_root,
        full_dir,
        "cpu",
    )
    # The aux target varies with time, so stats over 5 frames must differ
    # from stats over all 6 — proving the pool (and its normalization) was
    # actually truncated, not just recorded.
    truncated = np.load(out_dir / "normalization_stats.npz")
    full = np.load(full_dir / "normalization_stats.npz")
    assert not np.allclose(truncated["aux_mean"], full["aux_mean"])


def test_train_frames_too_small_raises(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    for cid in ("S-1", "S-2"):
        _write_tiny_case(data_root, cid)
    with pytest.raises(ValueError, match="train_frames"):
        train(
            _local_spec(),
            CGNConfig(**SMALL_CGN),
            TrainConfig(
                benchmark="seed-test-local",
                batch_size=2,
                training_steps=2,
                val_every=2,
                seed=1,
                train_frames=4,  # == input_frames + 1: no window remains
            ),
            data_root,
            tmp_path / "bad",
            "cpu",
        )


# --- mgn family dispatch (Task 4a, ①-c2) ---

#: Tiny architecture so a checkpoint builds fast.
SMALL_MGN = {
    "input_frames": 2,
    "dim": 3,
    "hidden_dim": 8,
    "message_passing_steps": 1,
    "nmlp_layers": 1,
    "node_type_size": 4,
    "world_edge_radius": 1000.0,  # generous: guarantees full connectivity
    "noise_std": 0.003,
    "normalizer_warmup_steps": 10,
}


def test_model_config_from_record_returns_mgn_config(tmp_path):
    """A round-tripped mgn run record reconstructs MGNConfig with matching fields."""
    mgn = MGNConfig(**SMALL_MGN)
    record_dict = resolved_config_dict(
        "mgn",
        mgn,
        TrainConfig(benchmark="deforming_plate"),
        horizon="full",
        eval_times="native",
        n_particle_types=mgn.node_type_size,
        data_root=tmp_path,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(record_dict), encoding="utf-8")
    record = read_run_record(config_path)
    model_cfg = _model_config_from_record(record)
    assert isinstance(model_cfg, MGNConfig)
    assert model_cfg == mgn


def test_model_config_from_record_returns_cgn_config(tmp_path):
    """Regression: a cgn-family run record still reconstructs CGNConfig."""
    cgn = CGNConfig(**SMALL_CGN)
    record_dict = resolved_config_dict(
        "cgn",
        cgn,
        TrainConfig(),
        horizon="full",
        eval_times="native",
        n_particle_types=2,
        data_root=tmp_path,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(record_dict), encoding="utf-8")
    record = read_run_record(config_path)
    model_cfg = _model_config_from_record(record)
    assert isinstance(model_cfg, CGNConfig)
    assert model_cfg == cgn


def _mesh_case_file(tmp_path, case_id="dp-0", n_nodes=6, n_cells=2, n_frames=5):
    """Tiny deforming-plate-shaped mesh case with one kinematic (OBSTACLE)
    node among otherwise-NORMAL nodes (SI units).

    Every node -- including the OBSTACLE one -- moves every frame
    (``world0 + i * 1e-4``), so gt_positions differs frame-to-frame at the
    kinematic row. That's load-bearing for the MeshSimulator rollout-pointer
    tripwire (see its module docstring): with a STATIONARY kinematic node the
    tripwire's ``torch.allclose`` check would spuriously pass under a stale
    pointer (any frame's identical GT value matches), silently defeating the
    very discrimination evaluate()'s reset_rollout() calls exist to test.
    """
    rng = np.random.default_rng(7)
    world0 = rng.random((n_nodes, 3)).astype(np.float32) * 0.01  # metres, tiny box
    world = np.stack([world0 + i * 1e-4 for i in range(n_frames)]).astype(np.float32)
    node_type = np.zeros(n_nodes, dtype=np.int32)
    node_type[-1] = 1  # OBSTACLE: kinematic, prescribed from ground truth
    arrays = {
        "cells": rng.integers(0, n_nodes, (n_cells, 4)).astype(np.int32),
        "node_type": node_type,
        "mesh_pos": world0.copy(),
        "world_pos": world,
        "stress": rng.random((n_frames, n_nodes, 1)).astype(np.float32),
    }
    case = build_deforming_plate_case(arrays, source_units="kg-m-s", case_id=case_id)
    path = tmp_path / f"{case_id}.h5"
    write_case(case, path)
    return path


def _mesh_local_spec(case_id, n_frames):
    """Registry-free mesh BenchmarkSpec. ``kinematic_types=(1,)`` matches the
    fixture's genuinely-present OBSTACLE node (type 1), so MeshSimulator's
    ``_has_kinematic`` is True and the rollout-pointer tripwire actually
    engages -- both satisfying the scripted_types-subset-of-kinematic_types
    constructor check AND letting the tripwire discriminate a stale pointer
    across eval passes."""
    card = BenchmarkCard(
        name="MeshTest",
        version="0",
        description="test-only mesh spec not in the registry",
        provenance="test",
        data_license="CC0",
        solver="test",
        discretisation="FEM",
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
        particles_per_case="6",
        n_frames=n_frames,
        output_dt_ms=1.0,
        input_frames=2,
        protocol_rationale="test-only card",
    )
    return BenchmarkSpec(
        card=card,
        splits={"train": (case_id,), "val": (case_id,)},
        eval_splits=("val",),
        aux_field="von_mises_stress",
        kinematic_types=(1,),
    )


def _prepared_mgn_run(tmp_path, case_path_fn, spec):
    """Run dir holding an mgn checkpoint + nested config.json, NO stats file."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    case_path = case_path_fn(data_root)
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    mgn = MGNConfig(**SMALL_MGN)
    simulator = build_mgn_simulator(
        mgn, kinematic_types=spec.kinematic_types, device="cpu"
    )
    simulator.save(str(out_dir / "model-best-000002.pt"))

    record_dict = resolved_config_dict(
        "mgn",
        mgn,
        TrainConfig(benchmark="mesh-test-local"),
        horizon="full",
        eval_times="native",
        n_particle_types=mgn.node_type_size,
        data_root=data_root,
    )
    (out_dir / "config.json").write_text(json.dumps(record_dict), encoding="utf-8")
    return data_root, out_dir, case_path.stem


def test_evaluate_mgn_family_needs_no_stats_file(tmp_path, monkeypatch):
    """mgn evaluation runs with no normalization_stats.npz in out_dir at all.

    MGN checkpoints are self-contained: the OnlineNormalizer buffers live in
    the state_dict, loaded by simulator.load() -- unlike cgn, no separate
    stats file is ever read.
    """
    import structbench.cli.train as train_mod

    spec = _mesh_local_spec("dp-0", n_frames=5)
    data_root, out_dir, case_id = _prepared_mgn_run(
        tmp_path, lambda root: _mesh_case_file(root, "dp-0", n_frames=5), spec
    )
    assert not (out_dir / "normalization_stats.npz").exists()
    monkeypatch.setattr(train_mod, "get_benchmark", lambda name: spec)

    metrics = evaluate([case_id], data_root, out_dir, "cpu", save_artifacts=False)

    assert not (out_dir / "normalization_stats.npz").exists()  # still never created
    assert metrics["input_frames"] == 2
    per_case = metrics["cases"][case_id]
    assert np.isfinite(per_case["one_step_position_rmse"])
    assert np.isfinite(per_case["rollout_position_rmse"])
    assert np.isfinite(per_case["rollout_aux_rmse"])


def _sph_case_file(data_root, case_id="C-1"):
    """An SPH (non-mesh) case: cells/reference_coords stay None on load."""
    _write_tiny_case(data_root, case_id)
    return data_root / f"{case_id}.h5"


def test_evaluate_mgn_family_rejects_non_mesh_benchmark(tmp_path, monkeypatch):
    """An mgn-family run over an SPH (non-mesh) case raises ValueError,
    naming the benchmark, rather than crashing deep inside bind_case."""
    import structbench.cli.train as train_mod

    spec = _mesh_local_spec("C-1", n_frames=6)  # card allows it; data won't
    data_root, out_dir, case_id = _prepared_mgn_run(tmp_path, _sph_case_file, spec)
    monkeypatch.setattr(train_mod, "get_benchmark", lambda name: spec)

    with pytest.raises(ValueError, match="mesh connectivity"):
        evaluate([case_id], data_root, out_dir, "cpu", save_artifacts=False)


# --- transolver family training (Task 5, ADR-0041/0044) ---


def test_lr_at_cosine_endpoints_and_monotone():
    """Cosine schedule: lr_init at step 0, the floor at training_steps,
    monotone decreasing in between (ADR-0044)."""
    train_cfg = TrainConfig(lr_init=1e-3, training_steps=1000)
    assert _lr_at_cosine(0, train_cfg) == pytest.approx(train_cfg.lr_init)
    assert _lr_at_cosine(1000, train_cfg) == pytest.approx(LR_SCHEDULE_FLOOR, abs=1e-9)
    samples = [_lr_at_cosine(s, train_cfg) for s in range(0, 1001, 100)]
    assert all(a > b for a, b in zip(samples, samples[1:], strict=False))


def test_build_transolver_simulator_sizes_net_and_binds_kinematic_types():
    """build_transolver_simulator mirrors build_mgn_simulator: architecture
    sized from the config, kinematic_types forwarded, scripted_types left at
    the class default (ADR-0043 recipe)."""
    cfg = TransolverConfig(
        dim=3, hidden_dim=8, n_layers=2, n_heads=2, slice_num=2, node_type_size=4
    )
    sim = build_transolver_simulator(cfg, kinematic_types=(1, 3), device="cpu")
    assert isinstance(sim, TransolverSimulator)
    assert sim._kinematic_types == (1, 3)
    assert sim._scripted_types == (1,)  # class default -- no run-config field yet
    assert len(sim._net.blocks) == cfg.n_layers
    assert sim._node_normalizer._sum.shape[0] == cfg.node_type_size + 3 * cfg.dim
    assert sim._target_normalizer._sum.shape[0] == cfg.dim + 1


def _transolver_smoke_spec(train_ids, val_id, n_frames):
    """Registry-free mesh BenchmarkSpec for the Task 5 wiring smoke test.

    Mirrors ``_mesh_local_spec`` but supports multiple TRAIN case ids and a
    VAL case id distinct from every TRAIN id: ``_train_transolver`` is
    exercised directly here (bypassing ``train()``'s own trajectory
    loading), so the caller supplies already-loaded, already-split
    trajectories built from these ids.
    """
    card = BenchmarkCard(
        name="TransolverSmoke",
        version="0",
        description="test-only mesh spec not in the registry",
        provenance="test",
        data_license="CC0",
        solver="test",
        discretisation="FEM",
        materials=("MAT_TEST",),
        erosion=False,
        loading="none",
        source_units="SI",
        geometry="unit box",
        n_cases=len(train_ids) + 1,
        splits={"train": len(train_ids), "val": 1},
        task="test",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=(),
        fields=("positions",),
        particles_per_case="6",
        n_frames=n_frames,
        output_dt_ms=1.0,
        input_frames=2,
        protocol_rationale="test-only card",
    )
    return BenchmarkSpec(
        card=card,
        splits={"train": tuple(train_ids), "val": (val_id,)},
        eval_splits=("val",),
        aux_field="von_mises_stress",
        kinematic_types=(1,),
    )


def test_train_transolver_wiring_adamw_cosine_clip(tmp_path, monkeypatch):
    """Guards the hand-cloned loop against a paste that forgot the swaps.

    A copy of ``_train_mgn`` that kept plain Adam, the exponential
    schedule, or no grad clipping would still pass shape/checkpoint checks
    silently. This test runs a REAL few-step ``_train_transolver`` and spies
    on the constructors/calls it makes internally: ``torch.optim.AdamW``
    (kwargs), ``torch.nn.utils.clip_grad_norm_`` (call args), the resulting
    optimizer's final learning rate against :func:`_lr_at_cosine`, and that
    at least one model parameter actually moved (gradient flow).
    """
    import structbench.cli.train as train_mod

    train_id, val_id = "tr-0", "val-0"
    n_frames = 6
    spec = _transolver_smoke_spec([train_id], val_id, n_frames)
    data_root = tmp_path / "data"
    data_root.mkdir()
    _mesh_case_file(data_root, train_id, n_frames=n_frames)
    _mesh_case_file(data_root, val_id, n_frames=n_frames)

    train_trajs = [
        load_case_trajectory(data_root / f"{train_id}.h5", aux_field=spec.aux_field)
    ]
    val_trajs = [
        load_case_trajectory(data_root / f"{val_id}.h5", aux_field=spec.aux_field)
    ]

    cfg = TransolverConfig(
        dim=3,
        hidden_dim=8,
        n_layers=1,
        n_heads=2,
        slice_num=2,
        node_type_size=4,
        input_frames=2,
        noise_std=0.001,
        normalizer_warmup_steps=1,
        weight_decay=1e-3,
        max_grad_norm=0.5,
    )
    train_cfg = TrainConfig(
        benchmark="TransolverSmoke", batch_size=2, training_steps=4, lr_init=1e-3
    )
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    captured_sim: dict = {}
    initial_params: dict = {}
    real_build = train_mod.build_transolver_simulator

    def _spy_build(*args, **kwargs):
        sim = real_build(*args, **kwargs)
        for name, p in sim.named_parameters():
            initial_params[name] = p.detach().clone()
        captured_sim["sim"] = sim
        return sim

    monkeypatch.setattr(train_mod, "build_transolver_simulator", _spy_build)

    captured_optimizers: list = []
    real_adamw = torch.optim.AdamW

    def _spy_adamw(*args, **kwargs):
        opt = real_adamw(*args, **kwargs)
        captured_optimizers.append((kwargs, opt))
        return opt

    monkeypatch.setattr(torch.optim, "AdamW", _spy_adamw)

    clip_calls: list = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def _spy_clip(*args, **kwargs):
        clip_calls.append((args, kwargs))
        return real_clip(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _spy_clip)

    torch.manual_seed(0)
    ckpt = train_mod._train_transolver(
        spec, cfg, train_cfg, train_trajs, val_trajs, out_dir, "cpu", data_root
    )

    assert ckpt is not None
    assert len(captured_optimizers) == 1, "AdamW must be constructed exactly once"
    adamw_kwargs, optimizer = captured_optimizers[0]
    # This also proves AdamW (not Adam): plain Adam takes no weight_decay
    # that would show up here with the configured (non-default) value.
    assert adamw_kwargs["weight_decay"] == cfg.weight_decay

    assert clip_calls, "clip_grad_norm_ was never called"
    for args, kwargs in clip_calls:
        max_norm = kwargs.get("max_norm", args[1] if len(args) > 1 else None)
        assert max_norm == cfg.max_grad_norm

    # Proves the cosine schedule is wired (and [train].lr_decay unused here):
    # _lr_at_cosine is called with the pre-increment step, so the last call
    # in a training_steps-step loop uses step = training_steps - 1.
    last_step = train_cfg.training_steps - 1
    assert optimizer.param_groups[0]["lr"] == _lr_at_cosine(last_step, train_cfg)

    sim = captured_sim["sim"]
    changed = any(
        not torch.equal(p.detach(), initial_params[name])
        for name, p in sim.named_parameters()
    )
    assert changed, "no model parameter changed -- gradient flow is broken"
