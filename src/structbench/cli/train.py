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

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..benchmarks.taylor_impact_2d import (
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
    collate_samples,
    compute_stats,
    load_case_trajectory,
)
from ..eval import rollout
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
        Mapping ``{"velocity": {"mean": Tensor, "std": Tensor}, "acceleration":
        {"mean": Tensor, "std": Tensor}}`` with per-dimension stats, shape
        ``(dim,)`` each.
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
    """Mean (position + auxiliary) rollout RMSE over validation trajectories."""
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
    ``w_pos * ||Δacc||^2 + w_aux * (Δaux)^2``. Every ``val_every`` steps it runs
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
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading %d TRAIN trajectories from %s", len(TRAIN), data_root)
    train_trajs = _load_trajectories(TRAIN, data_root)
    val_trajs = _load_trajectories(VAL, data_root)

    stats = compute_stats(train_trajs)
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
            loss_aux = (pred_aux[:, 0] - next_aux) ** 2
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
    gns: GNSConfig,
    case_ids: list[str],
    data_root: Path,
    out_dir: Path,
    device: str,
) -> float:
    """Load the best checkpoint and roll out over ``case_ids``.

    Stats are read from ``out_dir/normalization_stats.npz`` when present, else
    recomputed from the ``TRAIN`` split, and ``n_particle_types`` from
    ``out_dir/config.json`` when present, so the architecture matches the
    trained checkpoint.

    Parameters
    ----------
    gns : GNSConfig
        Architecture and noise configuration.
    case_ids : list of str
        Cases to roll out over (validation or test split).
    data_root : pathlib.Path
        Directory containing ``<case_id>.h5`` canonical cases.
    out_dir : pathlib.Path
        Run directory holding the checkpoint, stats, and resolved config.
    device : str
        Torch device string.

    Returns
    -------
    float
        Mean (position + auxiliary) rollout RMSE over ``case_ids``.
    """
    stats_path = out_dir / "normalization_stats.npz"
    if stats_path.exists():
        stats = NormalizationStats.load(stats_path)
    else:
        stats = compute_stats(_load_trajectories(TRAIN, data_root))

    config_path = out_dir / "config.json"
    if config_path.exists():
        n_types = int(json.loads(config_path.read_text())["n_particle_types"])
    else:
        n_types = _n_particle_types(_load_trajectories(TRAIN, data_root))

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

    trajectories = _load_trajectories(case_ids, data_root)
    return _validate(simulator, trajectories, gns.window, device)


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
    parser.add_argument("--config", type=str, default=None, help="TOML config path.")
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
        score = evaluate(gns, VAL, data_root, out_dir, device)
        print(f"validation mean RMSE: {score:.6f}")
    else:  # rollout
        score = evaluate(gns, TEST_INTERP + TEST_EXTRAP, data_root, out_dir, device)
        print(f"rollout mean RMSE: {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
