"""Config-driven GNS training, validation, and rollout entry point.

This module ties together the StructBench ML layer: the canonical data
pipeline (:mod:`structbench.datasets`), the learned simulator
(:mod:`structbench.models.gns`), the rollout evaluation
(:mod:`structbench.eval`), and the benchmark registry
(:mod:`structbench.benchmarks`).

The training loop is ported from the sgnn reference
(``sgnn/single_scale/train.py``) and the random-walk position noise from
``sgnn/noise_utils.py``. The reference's npz/metadata data path is replaced by
the canonical pipeline: train trajectories come from
:func:`~structbench.datasets.load_case_trajectory` over the spec's train
split, batched through :class:`~structbench.datasets.WindowDataset` and
:func:`~structbench.datasets.collate_samples`, with normalization from
:func:`~structbench.datasets.compute_stats`. The active benchmark is resolved
via :data:`TrainConfig.benchmark` → :func:`~structbench.benchmarks.get_benchmark`
→ a :class:`~structbench.benchmarks.BenchmarkSpec` that supplies the splits,
auxiliary field, QoIs, and optional boundary feature.

Positions are in the millimetre working frame; the auxiliary field's unit is
specified by the benchmark card (MPa for the Taylor default). Library functions
log via :mod:`logging`; only :func:`main` prints.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..benchmarks import BenchmarkSpec, available_benchmarks, get_benchmark
from ..config import (
    GNSConfig,
    ProtocolOverride,
    TrainConfig,
    load_run_config,
    read_run_record,
    resolved_config_dict,
)
from ..datasets import (
    CaseTrajectory,
    NormalizationStats,
    WindowDataset,
    cached_compute_stats,
    collate_samples,
    load_case_trajectory,
)
from ..eval import one_step_aux_rmse, one_step_position_rmse, rollout
from ..models.gns import LearnedSimulator
from ..models.gns.simulator import time_diff

logger = logging.getLogger(__name__)

__all__ = [
    "GNSConfig",
    "TrainConfig",
    "build_simulator",
    "evaluate",
    "main",
    "train",
]


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

    # The auxiliary target carries no input noise, so its stats are
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
        max_neighbors=gns.max_neighbors,
        boundary_feature_fn=boundary_feature_fn,
        device=device,
    )


def _load_trajectories(
    case_ids: list[str], data_root: Path, aux_field: str
) -> list[CaseTrajectory]:
    """Load each ``<data_root>/<case_id>.h5`` into a :class:`CaseTrajectory`."""
    return [
        load_case_trajectory(data_root / f"{case_id}.h5", aux_field=aux_field)
        for case_id in case_ids
    ]


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
    kinematic_types: tuple[int, ...] = (),
    init_frames: int | None = None,
) -> tuple[float, float]:
    """Mean rollout position RMSE (mm) and von Mises RMSE (MPa) over VAL.

    The two channels are kept separate (ADR-0028): summing mm + MPa made the
    in-training score 98% stress and let checkpoint selection ignore position
    quality entirely. Selection uses the position channel; the ADR-0019
    reported metrics come from :func:`evaluate`.

    Parameters
    ----------
    kinematic_types:
        Forwarded to :func:`rollout`; kinematic particles are excluded from
        the reported RMSE (ADR-0026).
    init_frames:
        Protocol init forwarded to :func:`rollout` (ADR-0032); ``None``
        seeds with ``window`` frames as before.
    """
    simulator.eval()
    pos_losses: list[float] = []
    aux_losses: list[float] = []
    for tr in trajectories:
        result = rollout(
            simulator,
            tr,
            window,
            device,
            kinematic_types=kinematic_types,
            init_frames=init_frames,
        )
        pos_losses.append(float(result.position_rmse.mean()))
        aux_losses.append(float(result.aux_rmse.mean()))
    if not pos_losses:
        return float("inf"), float("inf")
    return (
        sum(pos_losses) / len(pos_losses),
        sum(aux_losses) / len(aux_losses),
    )


def _bind_boundary_feature(
    spec: BenchmarkSpec, gns: GNSConfig
) -> Callable[[Tensor], Tensor] | None:
    """Bind the spec's boundary feature to the configured radius, if any."""
    fn = spec.boundary_feature_fn
    if fn is None:
        return None

    def feature(positions: Tensor) -> Tensor:
        return fn(positions, gns.connectivity_radius)

    return feature


def train(
    spec: BenchmarkSpec,
    gns: GNSConfig,
    train_cfg: TrainConfig,
    data_root: Path,
    out_dir: Path,
    device: str,
    *,
    family: str = "gns",
    protocol_override: ProtocolOverride | None = None,
) -> Path | None:
    """Run config-driven training with periodic validation and checkpoint-best.

    Builds the benchmark spec's train trajectories, normalization stats, and the
    simulator (using the spec's boundary feature if any), then optimizes with
    Adam under an exponential learning-rate decay and the dual MSE loss
    ``w_pos * ||Δacc||^2 + w_aux * (Δaux)^2``, where both the acceleration and
    the auxiliary targets are normalized so the two terms are O(1) and balanced.
    Every ``val_every`` steps it runs a validation rollout over the spec's val
    split and saves the model when the mean RMSE improves. The resolved config
    and normalization stats are written under ``out_dir``.

    Parameters
    ----------
    spec : BenchmarkSpec
        Benchmark spec supplying splits, auxiliary field, QoIs, and boundary
        feature.
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
    family : str
        Model-family registry key recorded in ``config.json`` (ADR-0032).
    protocol_override : ProtocolOverride or None
        Research-only protocol override; when given, validation rollouts use
        its ``init_frames`` and the run records ``protocol.standard = false``
        (ADR-0032 §4). ``None`` uses the benchmark card's protocol.

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
    ValueError
        If ``train_cfg.benchmark`` names a registered benchmark that is not
        ``spec``; this would misrecord the benchmark in ``config.json``.
    """
    if (
        train_cfg.benchmark in available_benchmarks()
        and get_benchmark(train_cfg.benchmark) is not spec
    ):
        raise ValueError(
            f"train_cfg.benchmark {train_cfg.benchmark!r} does not resolve to the "
            "spec passed to train(); config.json would misrecord the benchmark"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("model-*.pt"))
    if existing:
        raise FileExistsError(
            f"{out_dir} already contains checkpoints (e.g. {existing[0].name}); "
            "training has no resume — use a fresh --out directory per attempt"
        )

    # Seeds weight init, noise draws, and shuffle order (torch.manual_seed
    # covers all CUDA devices and the DataLoader's base seed). CUDA scatter-add
    # stays nondeterministic, so GPU runs are statistically, not bitwise,
    # reproducible.
    torch.manual_seed(train_cfg.seed)

    train_ids = list(spec.splits["train"])
    logger.info("loading %d TRAIN trajectories from %s", len(train_ids), data_root)
    train_trajs = _load_trajectories(train_ids, data_root, spec.aux_field)
    val_trajs = _load_trajectories(list(spec.splits["val"]), data_root, spec.aux_field)

    # Dataset-level cache (spec resolved-choice 2); the run-dir copy below is
    # the self-contained record evaluate() reads.
    stats = cached_compute_stats(
        train_trajs, dataset_root=data_root, aux_field=spec.aux_field
    )
    stats.save(out_dir / "normalization_stats.npz")
    n_types = _n_particle_types(train_trajs)

    simulator = build_simulator(
        _stats_to_dict(stats),
        gns,
        n_particle_types=n_types,
        boundary_feature_fn=_bind_boundary_feature(spec, gns),
        device=device,
    )
    simulator.to(device)

    # Auxiliary-target normalization: the decoder predicts the auxiliary
    # channel in normalized space, so the target is normalized to match before
    # the loss, keeping it O(1) and balanced against the position loss.
    aux_mean = torch.tensor(stats.aux_mean, dtype=torch.float32, device=device)
    aux_std = torch.tensor(stats.aux_std, dtype=torch.float32, device=device)

    init_frames = (
        protocol_override.init_frames
        if protocol_override is not None
        else spec.card.init_frames
    )
    (out_dir / "config.json").write_text(
        json.dumps(
            resolved_config_dict(
                family,
                gns,
                train_cfg,
                init_frames=init_frames,
                horizon=spec.card.horizon,
                eval_times=spec.card.eval_times,
                standard=protocol_override is None,
                n_particle_types=n_types,
                data_root=data_root,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

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
    best_pos = float("inf")
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
            per_particle = train_cfg.w_pos * loss_pos + train_cfg.w_aux * loss_aux
            if spec.kinematic_types:
                free = ~torch.isin(
                    particle_type,
                    torch.as_tensor(
                        list(spec.kinematic_types), dtype=torch.long, device=device
                    ),
                )
                if free is not None and free.any():
                    loss = per_particle[free].mean()
                elif free is not None:
                    # all-kinematic batch: nothing to learn from; zero loss, no NaN
                    loss = per_particle.new_tensor(0.0, requires_grad=True)
                else:
                    loss = per_particle.mean()
            else:
                loss = per_particle.mean()

            loss.backward()
            # The unclipped run showed ~5x loss spikes (steps 28k, 42k);
            # standard global-norm clipping keeps those from kicking the
            # weights off the manifold (ADR-0028).
            torch.nn.utils.clip_grad_norm_(simulator.parameters(), max_norm=1.0)
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
                val_pos, val_aux = _validate(
                    simulator,
                    val_trajs,
                    gns.window,
                    device,
                    spec.kinematic_types,
                    init_frames=init_frames,
                )
                logger.info(
                    "step %d: train_loss %.6f val_pos %.4f mm val_aux %.4f MPa "
                    "(best_pos %.4f)",
                    step,
                    loss.item(),
                    val_pos,
                    val_aux,
                    best_pos,
                )
                if val_pos < best_pos:
                    best_pos = val_pos
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


def _find_checkpoint(out_dir: Path) -> Path | None:
    """Return the most recently modified ``model-*.pt`` in ``out_dir``."""
    checkpoints = sorted(out_dir.glob("model-*.pt"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def _resolve_run_spec(out_dir: Path) -> tuple[BenchmarkSpec, dict[str, Any]]:
    """Resolve the run directory's benchmark spec and resolved config.

    Parameters
    ----------
    out_dir : pathlib.Path
        Run directory holding ``config.json``.

    Returns
    -------
    tuple of (BenchmarkSpec, dict)
        The benchmark spec and the run record, normalized to the nested
        ADR-0032 shape (pre-0032 flat records are adapted by
        :func:`structbench.config.read_run_record`).

    Raises
    ------
    FileNotFoundError
        If ``config.json`` is missing from ``out_dir``.
    """
    record = read_run_record(out_dir / "config.json")
    return get_benchmark(record["run"]["benchmark"]), record


def evaluate(
    case_ids: list[str],
    data_root: Path,
    out_dir: Path,
    device: str,
    *,
    split_name: str = "eval",
    save_artifacts: bool = True,
    init_frames: int | None = None,
) -> dict[str, Any]:
    """Roll out the run's checkpoint over ``case_ids`` and report ADR-0019 §5.

    The simulator is rebuilt entirely from the run directory's own record —
    architecture and ``n_particle_types`` from ``config.json``, stats from
    ``normalization_stats.npz`` — so evaluation always matches the trained
    checkpoint; no caller-supplied architecture is accepted. Both files are
    written by :func:`train`.

    Per case, reports the one-step (teacher-forced) position RMSE, the
    full-rollout position RMSE (mm), the rollout auxiliary-field RMSE (in the
    card's aux unit), and the
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
    init_frames : int or None
        Protocol init for the rollouts. ``None`` (the default) uses the
        run's recorded protocol — pre-0032 runs recorded their window — so
        checkpoints are evaluated as trained; pass a value explicitly for
        protocol-sensitivity studies (ADR-0032 §4).

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
    spec, record = _resolve_run_spec(out_dir)
    stats_path = out_dir / "normalization_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"missing normalization stats: {stats_path}")

    gns = GNSConfig(**{k: v for k, v in record["model"].items() if k != "family"})
    n_types = int(record["n_particle_types"])
    init = init_frames if init_frames is not None else record["protocol"]["init_frames"]
    stats = NormalizationStats.load(stats_path)

    simulator = build_simulator(
        _stats_to_dict(stats),
        gns,
        n_particle_types=n_types,
        boundary_feature_fn=_bind_boundary_feature(spec, gns),
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
        trajectory = load_case_trajectory(
            data_root / f"{case_id}.h5", aux_field=spec.aux_field
        )
        result = rollout(
            simulator,
            trajectory,
            gns.window,
            device,
            qois=spec.qois,
            kinematic_types=spec.kinematic_types,
            init_frames=init,
        )
        one_step = one_step_position_rmse(
            simulator,
            trajectory,
            gns.window,
            device,
            kinematic_types=spec.kinematic_types,
        )
        one_step_aux = one_step_aux_rmse(
            simulator,
            trajectory,
            gns.window,
            device,
            kinematic_types=spec.kinematic_types,
        )
        cases[case_id] = {
            "one_step_position_rmse": float(one_step.mean()),
            "one_step_aux_rmse": float(one_step_aux.mean()),
            "rollout_position_rmse": result.mean_position_rmse,
            "rollout_aux_rmse": result.mean_aux_rmse,
            "qoi_pred": result.qoi_pred,
            "qoi_true": result.qoi_true,
            "qoi_error": result.qoi_error,
        }
        logger.info(
            "[%s] %s: one-step %.4f mm | rollout %.4f mm | %s %.4f %s",
            split_name,
            case_id,
            cases[case_id]["one_step_position_rmse"],
            result.mean_position_rmse,
            spec.aux_field,
            result.mean_aux_rmse,
            spec.card.aux_unit,
        )
        if save_artifacts:
            np.savez(
                rollout_dir / f"{split_name}-{case_id}.npz",
                predicted_positions=result.predicted_positions,
                predicted_aux=result.predicted_aux,
                position_rmse=result.position_rmse,
                aux_rmse=result.aux_rmse,
                one_step_position_rmse=one_step,
                one_step_aux_rmse=one_step_aux,
            )

    def _mean_over_cases(key: str) -> float:
        return float(np.mean([case[key] for case in cases.values()]))

    metrics: dict[str, Any] = {
        "split": split_name,
        "checkpoint": checkpoint.name,
        "init_frames": init,
        # Standard = card-conforming: pre-0032 runs recorded init = window and
        # would otherwise self-certify; a card-init re-eval of any standard
        # run IS official (ADR-0032 §4).
        "protocol_standard": bool(record["protocol"]["standard"])
        and init == spec.card.init_frames,
        "aux_field": spec.aux_field,
        "aux_unit": spec.card.aux_unit,
        "cases": cases,
        "mean": {
            "one_step_position_rmse": _mean_over_cases("one_step_position_rmse"),
            "one_step_aux_rmse": _mean_over_cases("one_step_aux_rmse"),
            "rollout_position_rmse": _mean_over_cases("rollout_position_rmse"),
            "rollout_aux_rmse": _mean_over_cases("rollout_aux_rmse"),
            "qoi_abs_error": {
                name: float(
                    np.mean([abs(case["qoi_error"][name]) for case in cases.values()])
                )
                for name in spec.qois
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
        help="Grouped TOML run config (ADR-0032; required in train mode; "
        "valid/rollout rebuild the architecture from the run directory's "
        "config.json).",
    )
    parser.add_argument("--out", type=str, default=None, help="Run output directory.")
    parser.add_argument(
        "--data-root", type=str, default=None, help="Directory of <case_id>.h5 cases."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if args.mode == "train" and args.config is None:
        print("error: --config is required in train mode (ADR-0032)")
        return 2
    run_config = load_run_config(args.config) if args.config is not None else None

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
        assert run_config is not None  # guarded above
        spec = get_benchmark(run_config.train.benchmark)
        ckpt = train(
            spec,
            run_config.model,
            run_config.train,
            data_root,
            out_dir,
            device,
            family=run_config.family,
            protocol_override=run_config.protocol_override,
        )
        print(f"training complete; best checkpoint: {ckpt}")
    else:
        spec, _resolved = _resolve_run_spec(out_dir)
        override_init = (
            run_config.protocol_override.init_frames
            if run_config is not None and run_config.protocol_override is not None
            else None
        )
        if args.mode == "valid":
            metrics = evaluate(
                list(spec.splits["val"]),
                data_root,
                out_dir,
                device,
                split_name="val",
                init_frames=override_init,
            )
            _print_split_report(metrics)
        else:  # rollout: every eval split except val, in spec order
            for split_name in spec.eval_splits:
                if split_name == "val":
                    continue
                metrics = evaluate(
                    list(spec.splits[split_name]),
                    data_root,
                    out_dir,
                    device,
                    split_name=split_name,
                    init_frames=override_init,
                )
                _print_split_report(metrics)
    return 0


def _print_split_report(metrics: dict[str, Any]) -> None:
    """Print one split's metrics to stdout.

    Position RMSE is in mm. Aux RMSE is in the run's aux unit
    (recorded in benchmark card). QoI errors are in each QoI's own unit
    (recorded in the benchmark card).
    """
    split, mean = metrics["split"], metrics["mean"]
    aux_field = metrics.get("aux_field", "aux")
    aux_unit = metrics.get("aux_unit", "")
    aux_rmse_str = f"rollout {aux_field} RMSE {mean['rollout_aux_rmse']:.4f}"
    if aux_unit:
        aux_rmse_str = f"{aux_rmse_str} {aux_unit}"
    print(
        f"[{split}] one-step position RMSE {mean['one_step_position_rmse']:.4f} mm"
        f" | one-step {aux_field} RMSE {mean['one_step_aux_rmse']:.4f}"
        f" | rollout position RMSE {mean['rollout_position_rmse']:.4f} mm"
        f" | {aux_rmse_str}"
    )
    qoi = ", ".join(
        f"{name} {value:.4f}" for name, value in mean["qoi_abs_error"].items()
    )
    print(f"[{split}] QoI mean |error|: {qoi}")


if __name__ == "__main__":
    raise SystemExit(main())
