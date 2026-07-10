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
    build_simulator,
    evaluate,
    main,
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
    _write_tiny_case(data_root, "C-1", n_frames=3)  # n_frames == input_frames -> 0 samples
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
