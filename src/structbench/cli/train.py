"""Config-driven GNS training, validation, and rollout entry point.

This module ties together the StructBench ML layer: the canonical data
pipeline (:mod:`structbench.datasets`), the learned simulator
(:mod:`structbench.models.gns`), the rollout evaluation
(:mod:`structbench.eval`), and the v0.1 Taylor 2D benchmark
(:mod:`structbench.benchmarks.taylor_impact_2d`).

The training loop is ported from the sgnn reference
(``sgnn/single_scale/train.py``) and the random-walk position noise from
``sgnn/noise_utils.py``. The reference's npz/metadata data path is replaced by
the canonical pipeline: train trajectories come from
:func:`~structbench.datasets.load_case_trajectory` over the benchmark ``TRAIN``
split, batched through :class:`~structbench.datasets.WindowDataset` and
:func:`~structbench.datasets.collate_samples`, with normalization from
:func:`~structbench.datasets.compute_stats`. The simulator is built with the
Taylor :func:`~structbench.benchmarks.taylor_impact_2d.wall_distance_feature`
as its boundary feature and a single auxiliary (von Mises) channel.

Positions are in the millimetre working frame and the auxiliary field in MPa
(ADR-0019). Library functions log via :mod:`logging`; only :func:`main` prints.
"""

from __future__ import annotations

import argparse
import json
import logging
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..benchmarks.taylor_impact_2d import (
    QOIS,
    TEST_EXTRAP,
    TEST_INTERP,
    TRAIN,
    VAL,
    wall_distance_feature,
)
from ..datasets import (
    CaseTrajectory,
    NormalizationStats,
    WindowDataset,
    cached_compute_stats,
    collate_samples,
    load_case_trajectory,
)
from ..eval import one_step_position_rmse, rollout
from ..models.gns import LearnedSimulator
from ..models.gns.simulator import time_diff

logger = logging.getLogger(__name__)


@dataclass
class GNSConfig:
    """Architecture and noise hyperparameters of the learned simulator.

    Attributes
    ----------
    window : int
        Number of consecutive input frames per sample (history length).
    connectivity_radius : float
        Graph connectivity radius in the mm working frame.
    hidden_dim : int
        Latent/MLP hidden width.
    message_passing_steps : int
        Number of interaction-network message-passing steps.
    nmlp_layers : int
        Number of hidden layers in each MLP.
    particle_type_embedding_size : int
        Embedding width for the particle-type lookup (used only when more than
        one particle type is present).
    noise_std : float
        Standard deviation of the random-walk training noise at the last step.
    dim : int
        Spatial dimensionality (2 for the Taylor 2D benchmark).
    """

    window: int = 11
    connectivity_radius: float = 0.6
    hidden_dim: int = 64
    message_passing_steps: int = 5
    nmlp_layers: int = 1
    particle_type_embedding_size: int = 9
    noise_std: float = 0.02
    dim: int = 2

    @classmethod
    def from_toml(cls, path: str | Path) -> GNSConfig:
        """Build from a TOML file, overriding only the keys present.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a TOML file. Keys not matching a field are ignored; absent
            fields keep their default.

        Returns
        -------
        GNSConfig
        """
        return cls(**_toml_kwargs(cls, path))


@dataclass
class TrainConfig:
    """Optimization schedule and loss weights for training.

    Attributes
    ----------
    batch_size : int
        Number of trajectory windows per optimizer step.
    lr_init : float
        Initial Adam learning rate.
    lr_decay : float
        Multiplicative decay base for the exponential schedule.
    lr_decay_steps : int
        Step interval over which ``lr_decay`` is applied once.
    training_steps : int
        Total number of optimizer steps.
    val_every : int
        Validation/checkpoint interval in steps.
    w_pos : float
        Weight on the acceleration (position) loss term.
    w_aux : float
        Weight on the auxiliary (von Mises) loss term.
    """

    batch_size: int = 32
    lr_init: float = 1e-3
    lr_decay: float = 0.1
    lr_decay_steps: int = 30000
    training_steps: int = 100000
    val_every: int = 2000
    w_pos: float = 1.0
    w_aux: float = 1.0

    @classmethod
    def from_toml(cls, path: str | Path) -> TrainConfig:
        """Build from a TOML file, overriding only the keys present.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a TOML file. Keys not matching a field are ignored; absent
            fields keep their default.

        Returns
        -------
        TrainConfig
        """
        return cls(**_toml_kwargs(cls, path))


def _toml_kwargs(cls: type, path: str | Path) -> dict[str, Any]:
    """Read a TOML file and keep only top-level keys that name a field of ``cls``."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    valid = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in valid}


def random_walk_position_noise(
    position_sequence: Tensor, noise_std_last_step: float
) -> Tensor:
    """Random-walk noise added to an input position sequence (GNS training).

    Noise is sampled in the velocity domain so that the accumulated standard
    deviation at the last step equals ``noise_std_last_step``, then integrated
    to positions with a leading zero (the first position carries no noise, as it
    only sets the first velocity). Ported from the sgnn ``noise_utils`` helper.

    Parameters
    ----------
    position_sequence : torch.Tensor
        Position history, shape ``(nparticles, window, dim)``, in mm.
    noise_std_last_step : float
        Target velocity-noise standard deviation at the final step.

    Returns
    -------
    torch.Tensor
        Position-noise tensor with the same shape and device as
        ``position_sequence``.
    """
    velocity_sequence = time_diff(position_sequence)
    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = torch.randn_like(velocity_sequence) * (
        noise_std_last_step / num_velocities**0.5
    )
    # Random walk in velocity space.
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)
    # Integrate velocity noise to positions, leaving the first position clean.
    position_sequence_noise = torch.cat(
        [
            torch.zeros_like(velocity_sequence_noise[:, 0:1]),
            torch.cumsum(velocity_sequence_noise, dim=1),
        ],
        dim=1,
    )
    return position_sequence_noise


def build_simulator(
    stats: dict[str, dict[str, Tensor]],
    gns: GNSConfig,
    *,
    n_particle_types: int,
    boundary_feature_fn: Callable[[Tensor], Tensor] | None,
    device: str,
) -> LearnedSimulator:
    """Construct a :class:`LearnedSimulator` from stats and architecture config.

    The node-input width is computed as
    ``(window - 1) * dim + n_boundary + embedding`` where ``n_boundary`` is 1
    when ``boundary_feature_fn`` is given (else 0) and ``embedding`` is
    ``particle_type_embedding_size`` when ``n_particle_types > 1`` (else 0). The
    edge-input width is ``dim + 1``. Each normalization std is inflated by the
    training noise as ``sqrt(std**2 + noise_std**2)``, matching the source.

    Parameters
    ----------
    stats : dict
        Mapping ``{"velocity": ..., "acceleration": ..., "aux": ...}`` where
        each value is ``{"mean": Tensor, "std": Tensor}``. Velocity and
        acceleration stats are per-dimension (shape ``(dim,)``); the ``"aux"``
        stats are scalar (shape ``(1,)``). Velocity/acceleration std is inflated
        by the training noise; the auxiliary stats are passed through unchanged
        (the auxiliary target carries no input noise).
    gns : GNSConfig
        Architecture and noise configuration.
    n_particle_types : int
        Number of distinct particle types; controls the embedding.
    boundary_feature_fn : Callable or None
        Maps the most-recent positions ``(P, dim)`` to a boundary feature block
        ``(P, 1)``; ``None`` adds no boundary feature.
    device : str
        Torch device string for the stats tensors and batch-id construction.

    Returns
    -------
    LearnedSimulator
    """
    n_boundary = 1 if boundary_feature_fn is not None else 0
    embedding = gns.particle_type_embedding_size if n_particle_types > 1 else 0
    nnode_in = (gns.window - 1) * gns.dim + n_boundary + embedding
    nedge_in = gns.dim + 1

    noise_var = gns.noise_std**2
    normalization_stats: dict[str, dict[str, Tensor]] = {}
    for key in ("velocity", "acceleration"):
        mean = stats[key]["mean"].to(device)
        std = torch.sqrt(stats[key]["std"].to(device) ** 2 + noise_var)
        normalization_stats[key] = {"mean": mean, "std": std}

    # The auxiliary (von Mises) target carries no input noise, so its stats are
    # passed through without the sqrt(std^2 + noise^2) inflation applied above.
    normalization_stats["aux"] = {
        "mean": stats["aux"]["mean"].to(device),
        "std": stats["aux"]["std"].to(device),
    }

    return LearnedSimulator(
        particle_dimensions=gns.dim,
        nnode_in=nnode_in,
        nedge_in=nedge_in,
        latent_dim=gns.hidden_dim,
        nmessage_passing_steps=gns.message_passing_steps,
        nmlp_layers=gns.nmlp_layers,
        mlp_hidden_dim=gns.hidden_dim,
        connectivity_radius=gns.connectivity_radius,
        normalization_stats=normalization_stats,
        nparticle_types=n_particle_types,
        particle_type_embedding_size=gns.particle_type_embedding_size,
        n_aux=1,
        boundary_feature_fn=boundary_feature_fn,
        device=device,
    )


def _load_trajectories(case_ids: list[str], data_root: Path) -> list[CaseTrajectory]:
    """Load each ``<data_root>/<case_id>.h5`` into a :class:`CaseTrajectory`."""
    trajectories: list[CaseTrajectory] = []
    for case_id in case_ids:
        h5_path = data_root / f"{case_id}.h5"
        trajectories.append(load_case_trajectory(h5_path))
    return trajectories


def _stats_to_dict(stats: NormalizationStats) -> dict[str, dict[str, Tensor]]:
    """Convert :class:`NormalizationStats` to the nested-Tensor stats dict."""
    return {
        "velocity": {
            "mean": torch.tensor(stats.velocity_mean, dtype=torch.float32),
            "std": torch.tensor(stats.velocity_std, dtype=torch.float32),
        },
        "acceleration": {
            "mean": torch.tensor(stats.acceleration_mean, dtype=torch.float32),
            "std": torch.tensor(stats.acceleration_std, dtype=torch.float32),
        },
        "aux": {
            "mean": torch.tensor(stats.aux_mean, dtype=torch.float32),
            "std": torch.tensor(stats.aux_std, dtype=torch.float32),
        },
    }


def _n_particle_types(trajectories: list[CaseTrajectory]) -> int:
    """Particle-type count as ``max(part_id) + 1`` over all trajectories.

    Using ``max + 1`` (rather than the number of distinct values) keeps every
    raw LS-DYNA ``part_id`` a valid embedding index without remapping.  An
    embedding is created whenever any ``part_id`` is greater than zero — i.e.
    ``n_particle_types > 1`` — so the Taylor benchmark (whose raw LS-DYNA part
    ids are *not* zero-based) does use an embedding.  Non-contiguous or
    large raw part ids will oversize the embedding table; remapping ids to a
    compact range is a known deferred robustness item.
    """
    global_max = 0
    for tr in trajectories:
        if tr.particle_type.size:
            global_max = max(global_max, int(tr.particle_type.max()))
    return global_max + 1


def _validate(
    simulator: LearnedSimulator,
    trajectories: list[CaseTrajectory],
    window: int,
    device: str,
) -> float:
    """Mean (position + auxiliary) rollout RMSE over validation trajectories.

    This is the in-training checkpoint-selection score only (a single scalar,
    mm + MPa summed); the ADR-0019 reported metrics come from :func:`evaluate`.
    """
    simulator.eval()
    losses: list[float] = []
    for tr in trajectories:
        result = rollout(simulator, tr, window, device)
        losses.append(float(result.position_rmse.mean() + result.aux_rmse.mean()))
    return sum(losses) / len(losses) if losses else float("inf")


def _wall_feature_fn(gns: GNSConfig) -> Callable[[Tensor], Tensor]:
    """Bind the Taylor wall feature to the configured connectivity radius."""

    def feature(positions: Tensor) -> Tensor:
        return wall_distance_feature(positions, gns.connectivity_radius)

    return feature


def train(
    gns: GNSConfig,
    train_cfg: TrainConfig,
    data_root: Path,
    out_dir: Path,
    device: str,
) -> Path | None:
    """Run config-driven training with periodic validation and checkpoint-best.

    Builds the TRAIN trajectories, normalization stats, and the simulator (with
    the Taylor wall boundary feature), then optimizes with Adam under an
    exponential learning-rate decay and the dual MSE loss
    ``w_pos * ||Δacc||^2 + w_aux * (Δaux)^2``, where both the acceleration and
    the auxiliary (von Mises) targets are normalized so the two terms are O(1)
    and balanced. Every ``val_every`` steps it runs
    a validation rollout over ``VAL`` and saves the model when the mean RMSE
    improves. The resolved config and normalization stats are written under
    ``out_dir``.

    Parameters
    ----------
    gns : GNSConfig
        Architecture and noise configuration.
    train_cfg : TrainConfig
        Optimization schedule and loss weights.
    data_root : pathlib.Path
        Directory containing ``<case_id>.h5`` canonical cases.
    out_dir : pathlib.Path
        Output directory for checkpoints, stats, and the resolved config.
    device : str
        Torch device string.

    Returns
    -------
    pathlib.Path or None
        Path to the best (or fallback final) checkpoint, or ``None`` if no
        checkpoint was written.

    Raises
    ------
    FileExistsError
        If ``out_dir`` already holds ``model-*.pt`` checkpoints. Training has
        no resume, and :func:`evaluate` picks the newest checkpoint by mtime,
        so a fresh run into an old directory would shadow a better model.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("model-*.pt"))
    if existing:
        raise FileExistsError(
            f"{out_dir} already contains checkpoints (e.g. {existing[0].name}); "
            "training has no resume — use a fresh --out directory per attempt"
        )

    logger.info("loading %d TRAIN trajectories from %s", len(TRAIN), data_root)
    train_trajs = _load_trajectories(TRAIN, data_root)
    val_trajs = _load_trajectories(VAL, data_root)

    # Dataset-level cache (spec resolved-choice 2); the run-dir copy below is
    # the self-contained record evaluate() reads.
    stats = cached_compute_stats(train_trajs, dataset_root=data_root)
    stats.save(out_dir / "normalization_stats.npz")
    n_types = _n_particle_types(train_trajs)

    simulator = build_simulator(
        _stats_to_dict(stats),
        gns,
        n_particle_types=n_types,
        boundary_feature_fn=_wall_feature_fn(gns),
        device=device,
    )
    simulator.to(device)

    # Auxiliary (von Mises) normalization: the decoder predicts the auxiliary
    # channel in normalized space, so the target is normalized to match before
    # the loss, keeping it O(1) and balanced against the position loss.
    aux_mean = torch.tensor(stats.aux_mean, dtype=torch.float32, device=device)
    aux_std = torch.tensor(stats.aux_std, dtype=torch.float32, device=device)

    _write_resolved_config(out_dir, gns, train_cfg, n_types, data_root)

    dataset = WindowDataset(train_trajs, gns.window)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_samples,
    )
    optimizer = torch.optim.Adam(simulator.parameters(), lr=train_cfg.lr_init)

    logger.info(
        "starting training: %d steps, batch %d, %d particle types",
        train_cfg.training_steps,
        train_cfg.batch_size,
        n_types,
    )

    step = 0
    best_val = float("inf")
    best_ckpt: Path | None = None
    simulator.train()
    while step < train_cfg.training_steps:
        for batch in loader:
            position_seq = batch["position_seq"].to(device)
            particle_type = batch["particle_type"].to(device)
            npp = batch["n_particles_per_example"].to(device)
            next_position = batch["next_position"].to(device)
            next_aux = batch["next_aux"].to(device)

            noise = random_walk_position_noise(position_seq, gns.noise_std)

            optimizer.zero_grad()
            pred_acc, target_acc, pred_aux = simulator.predict_accelerations(
                next_positions=next_position,
                position_sequence_noise=noise,
                position_sequence=position_seq,
                nparticles_per_example=npp,
                particle_types=particle_type,
            )
            loss_pos = ((pred_acc - target_acc) ** 2).sum(dim=-1)
            next_aux_norm = (next_aux - aux_mean) / aux_std
            loss_aux = (pred_aux[:, 0] - next_aux_norm) ** 2
            loss = (train_cfg.w_pos * loss_pos + train_cfg.w_aux * loss_aux).mean()

            loss.backward()
            optimizer.step()

            lr_new = (
                train_cfg.lr_init
                * train_cfg.lr_decay ** (step / train_cfg.lr_decay_steps)
                + 1e-6
            )
            for group in optimizer.param_groups:
                group["lr"] = lr_new

            step += 1

            if step % train_cfg.val_every == 0:
                val_loss = _validate(simulator, val_trajs, gns.window, device)
                logger.info(
                    "step %d: train_loss %.6f val_loss %.6f (best %.6f)",
                    step,
                    loss.item(),
                    val_loss,
                    best_val,
                )
                if val_loss < best_val:
                    best_val = val_loss
                    best_ckpt = out_dir / f"model-best-{step:06d}.pt"
                    simulator.save(str(best_ckpt))
                    logger.info("saved improved checkpoint: %s", best_ckpt)
                simulator.train()

            if step >= train_cfg.training_steps:
                break

    if best_ckpt is None:
        best_ckpt = out_dir / f"model-final-{step:06d}.pt"
        simulator.save(str(best_ckpt))
        logger.info("no validation improvement; saved final checkpoint: %s", best_ckpt)
    return best_ckpt


def _write_resolved_config(
    out_dir: Path,
    gns: GNSConfig,
    train_cfg: TrainConfig,
    n_types: int,
    data_root: Path,
) -> None:
    """Dump the fully-resolved run configuration to ``out_dir/config.json``."""
    resolved = {
        "gns": asdict(gns),
        "train": asdict(train_cfg),
        "n_particle_types": n_types,
        "data_root": str(data_root),
    }
    (out_dir / "config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )


def _find_checkpoint(out_dir: Path) -> Path | None:
    """Return the most recently modified ``model-*.pt`` in ``out_dir``."""
    checkpoints = sorted(out_dir.glob("model-*.pt"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def evaluate(
    case_ids: list[str],
    data_root: Path,
    out_dir: Path,
    device: str,
    *,
    split_name: str = "eval",
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Roll out the run's checkpoint over ``case_ids`` and report ADR-0019 §5.

    The simulator is rebuilt entirely from the run directory's own record —
    architecture and ``n_particle_types`` from ``config.json``, stats from
    ``normalization_stats.npz`` — so evaluation always matches the trained
    checkpoint; no caller-supplied architecture is accepted. Both files are
    written by :func:`train`.

    Per case, reports the one-step (teacher-forced) position RMSE, the
    full-rollout position RMSE (mm), the rollout von Mises RMSE (MPa), and the
    benchmark QoIs with signed errors; the split mean aggregates each metric
    (QoIs as mean absolute error). When ``save_artifacts`` is true the report
    is written to ``out_dir/metrics-<split_name>.json`` and each predicted
    trajectory to ``out_dir/rollouts/<split_name>-<case_id>.npz``.

    Parameters
    ----------
    case_ids : list of str
        Cases to roll out over (validation or test split); must be non-empty.
    data_root : pathlib.Path
        Directory containing ``<case_id>.h5`` canonical cases.
    out_dir : pathlib.Path
        Run directory holding the checkpoint, stats, and resolved config.
    device : str
        Torch device string.
    split_name : str
        Label recorded in the report and used in artifact filenames.
    save_artifacts : bool
        Write the metrics JSON and per-case rollout ``.npz`` files.

    Returns
    -------
    dict
        ``{"split", "checkpoint", "cases": {case_id: ...}, "mean": ...}`` with
        plain JSON-serializable values.

    Raises
    ------
    FileNotFoundError
        If ``config.json``, ``normalization_stats.npz``, or a checkpoint is
        missing from ``out_dir``.
    """
    if not case_ids:
        raise ValueError("case_ids must be non-empty")
    config_path = out_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing run config: {config_path}")
    stats_path = out_dir / "normalization_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"missing normalization stats: {stats_path}")

    resolved = json.loads(config_path.read_text(encoding="utf-8"))
    gns = GNSConfig(**resolved["gns"])
    n_types = int(resolved["n_particle_types"])
    stats = NormalizationStats.load(stats_path)

    simulator = build_simulator(
        _stats_to_dict(stats),
        gns,
        n_particle_types=n_types,
        boundary_feature_fn=_wall_feature_fn(gns),
        device=device,
    )
    checkpoint = _find_checkpoint(out_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"no checkpoint found under {out_dir}")
    simulator.load(str(checkpoint))
    simulator.to(device)
    simulator.eval()

    rollout_dir = out_dir / "rollouts"
    if save_artifacts:
        rollout_dir.mkdir(parents=True, exist_ok=True)

    cases: dict[str, dict[str, Any]] = {}
    for case_id in case_ids:
        trajectory = load_case_trajectory(data_root / f"{case_id}.h5")
        result = rollout(simulator, trajectory, gns.window, device, qois=QOIS)
        one_step = one_step_position_rmse(simulator, trajectory, gns.window, device)
        cases[case_id] = {
            "one_step_position_rmse": float(one_step.mean()),
            "rollout_position_rmse": result.mean_position_rmse,
            "rollout_von_mises_rmse": result.mean_aux_rmse,
            "qoi_pred": result.qoi_pred,
            "qoi_true": result.qoi_true,
            "qoi_error": result.qoi_error,
        }
        logger.info(
            "[%s] %s: one-step %.4f mm | rollout %.4f mm | von Mises %.4f MPa",
            split_name,
            case_id,
            cases[case_id]["one_step_position_rmse"],
            result.mean_position_rmse,
            result.mean_aux_rmse,
        )
        if save_artifacts:
            np.savez(
                rollout_dir / f"{split_name}-{case_id}.npz",
                predicted_positions=result.predicted_positions,
                predicted_aux=result.predicted_aux,
                position_rmse=result.position_rmse,
                aux_rmse=result.aux_rmse,
                one_step_position_rmse=one_step,
            )

    def _mean_over_cases(key: str) -> float:
        return float(np.mean([case[key] for case in cases.values()]))

    metrics: dict[str, Any] = {
        "split": split_name,
        "checkpoint": checkpoint.name,
        "cases": cases,
        "mean": {
            "one_step_position_rmse": _mean_over_cases("one_step_position_rmse"),
            "rollout_position_rmse": _mean_over_cases("rollout_position_rmse"),
            "rollout_von_mises_rmse": _mean_over_cases("rollout_von_mises_rmse"),
            "qoi_abs_error": {
                name: float(
                    np.mean([abs(case["qoi_error"][name]) for case in cases.values()])
                )
                for name in QOIS
            },
        },
    }
    if save_artifacts:
        (out_dir / f"metrics-{split_name}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run training, validation, or rollout.

    Parameters
    ----------
    argv : list of str or None
        Argument vector (defaults to ``sys.argv[1:]`` when ``None``).

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="StructBench GNS training entry")
    parser.add_argument(
        "--mode",
        choices=["train", "valid", "rollout"],
        default="train",
        help="train, validate (VAL), or roll out (TEST).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="TOML config path (train mode only; valid/rollout rebuild the "
        "architecture from the run directory's config.json).",
    )
    parser.add_argument("--out", type=str, default=None, help="Run output directory.")
    parser.add_argument(
        "--data-root", type=str, default=None, help="Directory of <case_id>.h5 cases."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if args.config is not None:
        gns = GNSConfig.from_toml(args.config)
        train_cfg = TrainConfig.from_toml(args.config)
    else:
        gns = GNSConfig()
        train_cfg = TrainConfig()

    if args.data_root is None:
        print("error: --data-root is required")
        return 2
    data_root = Path(args.data_root)

    if args.out is not None:
        out_dir = Path(args.out)
    else:
        out_dir = Path("runs") / datetime.now().strftime("run-%Y%m%d-%H%M%S")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"mode={args.mode} device={device} out={out_dir}")

    if args.mode == "train":
        ckpt = train(gns, train_cfg, data_root, out_dir, device)
        print(f"training complete; best checkpoint: {ckpt}")
    elif args.mode == "valid":
        metrics = evaluate(VAL, data_root, out_dir, device, split_name="val")
        _print_split_report(metrics)
    else:  # rollout: interpolation is the headline; extrapolation separate
        for split_name, split_cases in (
            ("test_interp", TEST_INTERP),
            ("test_extrap", TEST_EXTRAP),
        ):
            metrics = evaluate(
                split_cases, data_root, out_dir, device, split_name=split_name
            )
            _print_split_report(metrics)
    return 0


def _print_split_report(metrics: dict[str, Any]) -> None:
    """Print one split's ADR-0019 metrics (mm / MPa) to stdout."""
    split, mean = metrics["split"], metrics["mean"]
    print(
        f"[{split}] one-step position RMSE {mean['one_step_position_rmse']:.4f} mm"
        f" | rollout position RMSE {mean['rollout_position_rmse']:.4f} mm"
        f" | rollout von Mises RMSE {mean['rollout_von_mises_rmse']:.4f} MPa"
    )
    qoi = ", ".join(
        f"{name} {value:.4f} mm" for name, value in mean["qoi_abs_error"].items()
    )
    print(f"[{split}] QoI mean |error|: {qoi}")


if __name__ == "__main__":
    raise SystemExit(main())
