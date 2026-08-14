"""Config-driven CGN training, validation, and rollout entry point.

This module ties together the StructBench ML layer: the canonical data
pipeline (:mod:`structbench.datasets`), the learned simulator
(:mod:`structbench.models.cgn`), the rollout evaluation
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
import functools
import json
import logging
import math
import re
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..benchmarks import BenchmarkSpec, available_benchmarks, get_benchmark
from ..config import (
    LR_SCHEDULE_FLOOR,
    MODEL_FAMILIES,
    CGNConfig,
    GeoFlareConfig,
    MGNConfig,
    TrainConfig,
    TransolverConfig,
    load_run_config,
    read_run_record,
    resolved_config_dict,
)
from ..datasets import (
    CaseTrajectory,
    NormalizationStats,
    WindowDataset,
    aux_forward_transform,
    cached_compute_stats,
    collate_samples,
    load_case_trajectory,
)
from ..eval import one_step_aux_rmse, one_step_position_rmse, rollout
from ..models.cgn import LearnedSimulator
from ..models.cgn.simulator import time_diff
from ..models.common import CaseBoundSimulator
from ..models.geoflare import GeoFlareSimulator
from ..models.mgn import (
    MeshSimulator,
    collate_mesh_samples,
    mesh_static_from_trajectory,
)
from ..models.transolver import TransolverSimulator

logger = logging.getLogger(__name__)

__all__ = [
    "CGNConfig",
    "GeoFlareConfig",
    "MGNConfig",
    "TrainConfig",
    "TransolverConfig",
    "build_geoflare_simulator",
    "build_mgn_simulator",
    "build_simulator",
    "build_transolver_simulator",
    "evaluate",
    "main",
    "train",
]

#: Cadence (steps) of the periodic ``ckpt-<step>.pt`` snapshots written
#: alongside the selection checkpoints. Fleet tooling, not recipe: they let
#: post-hoc smoothed selection re-score a run's trajectory of states
#: identically across ablation arms (ADR-0028, 2026-07-10 note). The name
#: sits outside the ``model-*.pt`` glob so default evaluation never picks
#: them up.
PERIODIC_CKPT_EVERY = 10_000

#: Families that consume mesh connectivity (``cells``/``reference_coords``)
#: and therefore apply a benchmark's ``spec.mesh_transform`` at load time
#: (ADR-0047). The cgn family never applies it — its data path stays
#: byte-identical.
_MESH_FAMILIES = frozenset({"mgn", "transolver", "geoflare"})


def random_walk_position_noise(
    position_sequence: Tensor, noise_std_last_step: float
) -> Tensor:
    """Random-walk noise added to an input position sequence (CGN training).

    Noise is sampled in the velocity domain so that the accumulated standard
    deviation at the last step equals ``noise_std_last_step``, then integrated
    to positions with a leading zero (the first position carries no noise, as it
    only sets the first velocity). Ported from the sgnn ``noise_utils`` helper.

    Parameters
    ----------
    position_sequence : torch.Tensor
        Position history, shape ``(nparticles, input_frames, dim)``, in mm.
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


def _mesh_family_noise(
    position_seq: Tensor,
    next_position: Tensor,
    is_kinematic: Tensor,
    noise_std: float,
    velocity_history: bool,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Noisy inputs, optional velocity-history feature, and the matched target.

    Shared by the three mesh-family training loops (ADR-0049). The two paths
    deliberately pair DIFFERENT noise schemes with DIFFERENT target
    conventions, each matching its reference recipe:

    * Reference path (``velocity_history=False``): single-frame Gaussian
      noise on the current frame; ``next_position`` is returned UNCHANGED,
      so the caller's velocity target ``next - x_noisy`` measures from the
      noisy position — the MeshGraphNets gamma = 1 convention (the model
      learns to step back onto the clean trajectory; the correction is a
      single sigma).
    * Velocity-history path: :func:`random_walk_position_noise` over the
      FULL window (the CGN recipe), so the velocity features are
      consistently noisy — and the returned target next position is
      ADJUSTED by the last frame's accumulated noise (GNS reference:
      ``next + noise[:, -1]``), so ``next_adjusted - x_noisy`` equals the
      CLEAN next velocity. The target corrects the velocity noise exactly
      and does NOT ask the model to undo the accumulated position offset
      (~3.3 sigma for 5 velocities) — a partially-unobservable component
      that would inflate the irreducible loss and the target-normalizer
      std, and bias rollouts toward over-contraction.

    Kinematic rows stay clean on both paths (ADR-0043 §4), so their
    adjusted target equals the ground truth and the scripted-velocity
    feature is unaffected.

    Parameters
    ----------
    position_seq : torch.Tensor
        ``(P, F, dim)`` collated position windows, most recent frame last.
    next_position : torch.Tensor
        ``(P, dim)`` ground-truth next-frame positions.
    is_kinematic : torch.Tensor
        ``(P,)`` bool mask of kinematic rows.
    noise_std : float
        The family config's ``noise_std``.
    velocity_history : bool
        The family config's ``velocity_history`` flag.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor or None, torch.Tensor)
        ``(x_noisy (P, dim), velocity_history (P, (F-1)*dim) or None,
        next_target (P, dim))`` — feed ``next_target``, not the raw
        ``next_position``, to ``forward_train``.
    """
    if not velocity_history:
        noise = torch.randn_like(position_seq[:, -1]) * noise_std
        noise = noise.masked_fill(is_kinematic.unsqueeze(-1), 0.0)
        return position_seq[:, -1] + noise, None, next_position
    noise_seq = random_walk_position_noise(position_seq, noise_std)
    noise_seq = noise_seq.masked_fill(is_kinematic.unsqueeze(-1).unsqueeze(-1), 0.0)
    noisy_seq = position_seq + noise_seq
    velocities = noisy_seq[:, 1:] - noisy_seq[:, :-1]
    return (
        noisy_seq[:, -1],
        velocities.flatten(1),
        next_position + noise_seq[:, -1],
    )


def build_simulator(
    stats: dict[str, dict[str, Tensor]],
    cgn: CGNConfig,
    *,
    n_particle_types: int,
    boundary_feature_fn: Callable[[Tensor], Tensor] | None,
    device: str,
) -> LearnedSimulator:
    """Construct a :class:`LearnedSimulator` from stats and architecture config.

    The node-input width is computed as
    ``(input_frames - 1) * dim + n_boundary + embedding`` where ``n_boundary``
    is 1 when ``boundary_feature_fn`` is given (else 0) and ``embedding`` is
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
    cgn : CGNConfig
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
    embedding = cgn.particle_type_embedding_size if n_particle_types > 1 else 0
    nnode_in = (cgn.input_frames - 1) * cgn.dim + n_boundary + embedding
    nedge_in = cgn.dim + 1

    noise_var = cgn.noise_std**2
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
        particle_dimensions=cgn.dim,
        nnode_in=nnode_in,
        nedge_in=nedge_in,
        latent_dim=cgn.hidden_dim,
        nmessage_passing_steps=cgn.message_passing_steps,
        nmlp_layers=cgn.nmlp_layers,
        mlp_hidden_dim=cgn.hidden_dim,
        connectivity_radius=cgn.connectivity_radius,
        normalization_stats=normalization_stats,
        nparticle_types=n_particle_types,
        particle_type_embedding_size=cgn.particle_type_embedding_size,
        n_aux=1,
        aux_transform=cgn.aux_transform,
        aux_transform_scale=cgn.aux_transform_scale,
        max_neighbors=cgn.max_neighbors,
        boundary_feature_fn=boundary_feature_fn,
        device=device,
    )


def build_mgn_simulator(
    mgn: MGNConfig,
    *,
    kinematic_types: tuple[int, ...],
    scripted_types: tuple[int, ...] | None = None,
    device: str,
) -> MeshSimulator:
    """Construct a :class:`MeshSimulator` from an :class:`MGNConfig`.

    Unlike :func:`build_simulator`, no normalization stats are needed at
    construction time: ``MeshSimulator``'s four
    :class:`~structbench.models.mgn.normalizers.OnlineNormalizer` buffers are
    part of its own ``state_dict`` (ADR-0043 §8), so a checkpoint carries its
    normalizer state and a fresh instance starts with the online normalizers'
    own (untrained) defaults until ``.load()`` restores them.

    Parameters
    ----------
    mgn : MGNConfig
        Architecture hyperparameters.
    kinematic_types : tuple of int
        Node-type codes whose motion is prescribed by ground truth (the
        benchmark spec's ``kinematic_types``); forwarded verbatim.
    device : str
        Torch device string.

    Returns
    -------
    MeshSimulator
        ``scripted_types=None`` leaves the class default ``(1,)`` (the
        ADR-0043 recipe scripts only the OBSTACLE node type); a benchmark
        overrides it via ``spec.scripted_types`` (ADR-0047).
    """
    return MeshSimulator(
        dim=mgn.dim,
        latent=mgn.hidden_dim,
        mp_steps=mgn.message_passing_steps,
        n_hidden=mgn.nmlp_layers,
        node_type_size=mgn.node_type_size,
        world_edge_radius=mgn.world_edge_radius,
        history_velocities=(mgn.input_frames - 1) if mgn.velocity_history else 0,
        mesh_edge_max_stretch=mgn.mesh_edge_max_stretch,
        kinematic_types=kinematic_types,
        **({} if scripted_types is None else {"scripted_types": scripted_types}),
        device=device,
    )


def build_transolver_simulator(
    cfg: TransolverConfig,
    *,
    kinematic_types: tuple[int, ...],
    scripted_types: tuple[int, ...] | None = None,
    device: str,
) -> TransolverSimulator:
    """Construct a :class:`TransolverSimulator` from a :class:`TransolverConfig`.

    Mirrors :func:`build_mgn_simulator`: no normalization stats are needed at
    construction time, since ``TransolverSimulator``'s two
    :class:`~structbench.models.mgn.normalizers.OnlineNormalizer` buffers
    (node and target) are part of its own ``state_dict``, matching MGN's
    self-contained-checkpoint pattern (ADR-0043 §8) — a checkpoint carries
    its normalizer state and a fresh instance starts with the online
    normalizers' own (untrained) defaults until ``.load()`` restores them.

    Parameters
    ----------
    cfg : TransolverConfig
        Architecture hyperparameters.
    kinematic_types : tuple of int
        Node-type codes whose motion is prescribed by ground truth (the
        benchmark spec's ``kinematic_types``); forwarded verbatim.
    device : str
        Torch device string.

    Returns
    -------
    TransolverSimulator
        ``scripted_types=None`` leaves the class default ``(1,)`` (the
        ADR-0043 recipe scripts only the OBSTACLE node type); a benchmark
        overrides it via ``spec.scripted_types`` (ADR-0047).
    """
    return TransolverSimulator(
        dim=cfg.dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        slice_num=cfg.slice_num,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        node_type_size=cfg.node_type_size,
        history_velocities=(cfg.input_frames - 1) if cfg.velocity_history else 0,
        # ADR-0050/0051: cfg.frames_per_call is the resolved k. The k=T sentinel
        # (0) is resolved to a concrete horizon upstream in _train_transolver
        # (train) / evaluate reads the resolved integer from config.json, so a
        # 0 never reaches here.
        frames_per_call=cfg.frames_per_call,
        kinematic_types=kinematic_types,
        **({} if scripted_types is None else {"scripted_types": scripted_types}),
        device=device,
    )


def build_geoflare_simulator(
    cfg: GeoFlareConfig,
    *,
    kinematic_types: tuple[int, ...],
    scripted_types: tuple[int, ...] | None = None,
    device: str,
) -> GeoFlareSimulator:
    """Construct a :class:`GeoFlareSimulator` from a :class:`GeoFlareConfig`.

    Mirrors :func:`build_transolver_simulator`: no normalization stats are
    needed at construction time, since ``GeoFlareSimulator``'s two
    :class:`~structbench.models.mgn.normalizers.OnlineNormalizer` buffers
    (node and target) are part of its own ``state_dict``, matching the
    MGN/Transolver self-contained-checkpoint pattern (ADR-0043 §8) — a
    checkpoint carries its normalizer state and a fresh instance starts
    with the online normalizers' own (untrained) defaults until ``.load()``
    restores them.

    The config's four ball-query scalars are assembled into the
    simulator's ``(near, far)`` tuples here — this scalar-to-tuple mapping
    lives ONLY in this function: ``radii=(cfg.radius_near, cfg.radius_far)``,
    ``neighbors=(cfg.neighbors_near, cfg.neighbors_far)``.

    Parameters
    ----------
    cfg : GeoFlareConfig
        Architecture hyperparameters.
    kinematic_types : tuple of int
        Node-type codes whose motion is prescribed by ground truth (the
        benchmark spec's ``kinematic_types``); forwarded verbatim.
    device : str
        Torch device string.

    Returns
    -------
    GeoFlareSimulator
        ``scripted_types=None`` leaves the class default ``(1,)`` (the
        ADR-0043 recipe scripts only the OBSTACLE node type); a benchmark
        overrides it via ``spec.scripted_types`` (ADR-0047).
    """
    return GeoFlareSimulator(
        dim=cfg.dim,
        n_hidden=cfg.n_hidden,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        slice_num=cfg.slice_num,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        n_hidden_local=cfg.n_hidden_local,
        radii=(cfg.radius_near, cfg.radius_far),
        neighbors=(cfg.neighbors_near, cfg.neighbors_far),
        node_type_size=cfg.node_type_size,
        history_velocities=(cfg.input_frames - 1) if cfg.velocity_history else 0,
        kinematic_types=kinematic_types,
        **({} if scripted_types is None else {"scripted_types": scripted_types}),
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
    input_frames: int,
    device: str,
    kinematic_types: tuple[int, ...] = (),
) -> tuple[float, float]:
    """Mean rollout position RMSE (mm) and von Mises RMSE (MPa) over VAL.

    The two channels are kept separate (ADR-0028): summing mm + MPa made the
    in-training score 98% stress and let checkpoint selection ignore position
    quality entirely. Selection uses the position channel; the ADR-0019
    reported metrics come from :func:`evaluate`.

    Parameters
    ----------
    input_frames:
        History length / rollout seed count, forwarded to :func:`rollout`
        (ADR-0035); the benchmark card's protocol value.
    kinematic_types:
        Forwarded to :func:`rollout`; kinematic particles are excluded from
        the reported RMSE (ADR-0026).
    """
    simulator.eval()
    pos_losses: list[float] = []
    aux_losses: list[float] = []
    for tr in trajectories:
        result = rollout(
            simulator,
            tr,
            input_frames,
            device,
            kinematic_types=kinematic_types,
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
    spec: BenchmarkSpec, cgn: CGNConfig
) -> Callable[[Tensor], Tensor] | None:
    """Bind the spec's boundary feature to the configured radius, if any."""
    fn = spec.boundary_feature_fn
    if fn is None:
        return None

    def feature(positions: Tensor) -> Tensor:
        return fn(positions, cgn.connectivity_radius)

    return feature


def _lr_at(step: int, train_cfg: TrainConfig) -> float:
    """Exponential learning-rate schedule value at ``step``.

    ``lr(step) = lr_init * lr_decay ** (step / lr_decay_steps) +
    LR_SCHEDULE_FLOOR`` (see :data:`~structbench.config.LR_SCHEDULE_FLOOR`).
    Move-only extraction of the formula shared by the CGN and MGN training
    loops in :func:`train`/:func:`_train_mgn`; behaviour is unchanged from
    the inline computation it replaces.

    Parameters
    ----------
    step : int
        Current optimizer step (pre-increment, matching the call site).
    train_cfg : TrainConfig
        Supplies ``lr_init``, ``lr_decay``, and ``lr_decay_steps``.

    Returns
    -------
    float
        The learning rate to apply at ``step``.
    """
    return (
        train_cfg.lr_init * train_cfg.lr_decay ** (step / train_cfg.lr_decay_steps)
        + LR_SCHEDULE_FLOOR
    )


def _lr_at_cosine(step: int, train_cfg: TrainConfig) -> float:
    """Cosine anneal lr_init → LR_SCHEDULE_FLOOR over training_steps (ADR-0044).

    Steps-port of the Transolver reference's per-epoch CosineAnnealingLR
    (grounding §4.1); the exponential `_lr_at` stays the CGN/MGN schedule.
    """
    frac = min(step / max(1, train_cfg.training_steps), 1.0)
    span = train_cfg.lr_init - LR_SCHEDULE_FLOOR
    return LR_SCHEDULE_FLOOR + span * 0.5 * (1.0 + math.cos(math.pi * frac))


def train(
    spec: BenchmarkSpec,
    model_cfg: CGNConfig | MGNConfig | TransolverConfig | GeoFlareConfig,
    train_cfg: TrainConfig,
    data_root: Path,
    out_dir: Path,
    device: str,
    *,
    family: str = "cgn",
) -> Path | None:
    """Run config-driven training with periodic validation and checkpoint-best.

    Every :data:`PERIODIC_CKPT_EVERY` steps, in both the ``"cgn"``/``"gns"``
    path and the ``"mgn"`` path, an additional ``ckpt-<step>.pt`` snapshot is
    written alongside the selection checkpoints for post-hoc analysis (never
    read by default evaluation) — this lets fleet tooling re-score a run's
    state trajectory retrospectively (ADR-0028 smoothed selection).

    For ``family="mgn"`` this immediately delegates to :func:`_train_mgn`,
    for ``family="transolver"`` to :func:`_train_transolver`, and for
    ``family="geoflare"`` to :func:`_train_geoflare`, after the shared
    trajectory loading and ADR-0039 truncation (each has its own training
    loop and inline validation; none needs a normalization-stats file,
    since their normalizer buffers are self-contained in the checkpoint,
    ADR-0043 §8/ADR-0041). The rest of this docstring describes the
    ``"cgn"``/``"gns"`` path.

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
    model_cfg : CGNConfig, MGNConfig, TransolverConfig, or GeoFlareConfig
        Architecture and noise configuration for the resolved ``family``.
    train_cfg : TrainConfig
        Optimization schedule and loss weights.
    data_root : pathlib.Path
        Directory containing ``<case_id>.h5`` canonical cases.
    out_dir : pathlib.Path
        Output directory for checkpoints, stats, and the resolved config.
    device : str
        Torch device string.
    family : str
        Model-family registry key recorded in ``config.json`` (ADR-0032);
        ``"mgn"`` dispatches to :func:`_train_mgn`, ``"transolver"`` to
        :func:`_train_transolver`, ``"geoflare"`` to :func:`_train_geoflare`.

    Returns
    -------
    pathlib.Path or None
        Path to the best (or fallback final) checkpoint, or ``None`` if no
        checkpoint was written.

    Raises
    ------
    FileExistsError
        If ``out_dir`` already holds ``model-*.pt`` or ``ckpt-*.pt``
        checkpoints. Training has no resume, and :func:`evaluate` picks the
        highest-step ``model-*.pt``, so a fresh run into an old directory
        would shadow a better model.
    ValueError
        If ``train_cfg.benchmark`` names a registered benchmark that is not
        ``spec`` (this would misrecord the benchmark in ``config.json``), or if
        ``model_cfg.input_frames`` disagrees with the benchmark card's protocol
        (ADR-0035: the model observes exactly the frames it inputs).
    """
    if (
        train_cfg.benchmark in available_benchmarks()
        and get_benchmark(train_cfg.benchmark) is not spec
    ):
        raise ValueError(
            f"train_cfg.benchmark {train_cfg.benchmark!r} does not resolve to the "
            "spec passed to train(); config.json would misrecord the benchmark"
        )
    if model_cfg.input_frames != spec.card.input_frames:
        raise ValueError(
            f"model input_frames ({model_cfg.input_frames}) must equal benchmark "
            f"{spec.card.name!r} protocol input_frames ({spec.card.input_frames}); "
            "a model observes exactly the frames it inputs (ADR-0035)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("model-*.pt")) + sorted(out_dir.glob("ckpt-*.pt"))
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
    if train_cfg.train_frames > 0:
        # ADR-0039 §4 recipe: train only on the scored window's frames. Must
        # precede cached_compute_stats so normalization follows the truncated
        # pool (the cache signature covers frame counts, so no stale hit).
        if train_cfg.train_frames <= model_cfg.input_frames + 1:
            raise ValueError(
                f"train_frames={train_cfg.train_frames} leaves no training "
                f"window (input_frames={model_cfg.input_frames})"
            )
        train_trajs = [
            replace(
                tr,
                positions=tr.positions[: train_cfg.train_frames],
                aux=tr.aux[: train_cfg.train_frames],
                time=tr.time[: train_cfg.train_frames],
            )
            for tr in train_trajs
        ]
        logger.info(
            "train_frames=%d: training pool truncated (ADR-0039 recipe)",
            train_cfg.train_frames,
        )
    val_trajs = _load_trajectories(list(spec.splits["val"]), data_root, spec.aux_field)
    if spec.scored_frames is not None:
        # In-training validation rolls out only the scored span (ADR-0039):
        # checkpoint selection then keys on the same window the benchmark
        # reports, and each val pass costs proportionally less.
        val_trajs = [
            replace(
                tr,
                positions=tr.positions[: spec.scored_frames],
                aux=tr.aux[: spec.scored_frames],
                time=tr.time[: spec.scored_frames],
            )
            for tr in val_trajs
        ]

    if family in _MESH_FAMILIES and spec.mesh_transform is not None:
        # ADR-0047: benchmark-declared synthesis (e.g. Taylor's lattice mesh
        # + wall nodes). Wall rows are static, so this commutes with the
        # frame truncations above.
        train_trajs = [spec.mesh_transform(tr) for tr in train_trajs]
        val_trajs = [spec.mesh_transform(tr) for tr in val_trajs]
        logger.info(
            "mesh_transform applied (ADR-0047): %d train + %d val trajectories",
            len(train_trajs),
            len(val_trajs),
        )

    if family == "mgn":
        assert isinstance(model_cfg, MGNConfig)
        return _train_mgn(
            spec,
            model_cfg,
            train_cfg,
            train_trajs,
            val_trajs,
            out_dir,
            device,
            data_root,
        )
    if family == "transolver":
        assert isinstance(model_cfg, TransolverConfig)
        return _train_transolver(
            spec,
            model_cfg,
            train_cfg,
            train_trajs,
            val_trajs,
            out_dir,
            device,
            data_root,
        )
    if family == "geoflare":
        assert isinstance(model_cfg, GeoFlareConfig)
        return _train_geoflare(
            spec,
            model_cfg,
            train_cfg,
            train_trajs,
            val_trajs,
            out_dir,
            device,
            data_root,
        )
    cgn = cast(CGNConfig, model_cfg)

    # Dataset-level cache (spec resolved-choice 2); the run-dir copy below is
    # the self-contained record evaluate() reads.
    stats = cached_compute_stats(
        train_trajs,
        dataset_root=data_root,
        aux_field=spec.aux_field,
        aux_transform=cgn.aux_transform,
        aux_transform_scale=cgn.aux_transform_scale,
    )
    stats.save(out_dir / "normalization_stats.npz")
    n_types = _n_particle_types(train_trajs)

    simulator = build_simulator(
        _stats_to_dict(stats),
        cgn,
        n_particle_types=n_types,
        boundary_feature_fn=_bind_boundary_feature(spec, cgn),
        device=device,
    )
    simulator.to(device)

    # Auxiliary-target normalization: the decoder predicts the auxiliary
    # channel in normalized (and, under aux_transform, transformed) space, so
    # the target is transformed and normalized to match before the loss,
    # keeping it O(1) and balanced against the position loss.
    aux_mean = torch.tensor(stats.aux_mean, dtype=torch.float32, device=device)
    aux_std = torch.tensor(stats.aux_std, dtype=torch.float32, device=device)

    (out_dir / "config.json").write_text(
        json.dumps(
            resolved_config_dict(
                family,
                cgn,
                train_cfg,
                horizon=spec.card.horizon,
                eval_times=spec.card.eval_times,
                n_particle_types=n_types,
                data_root=data_root,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = WindowDataset(train_trajs, cgn.input_frames)
    if len(dataset) == 0:
        raise ValueError(
            f"empty training set: no TRAIN trajectory has more than "
            f"input_frames={cgn.input_frames} frames, so there are no "
            f"autoregressive samples. Check the data root or reduce input_frames."
        )
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
    window_loss_sum: torch.Tensor | None = None
    window_loss_n = 0
    simulator.train()
    while step < train_cfg.training_steps:
        for batch in loader:
            position_seq = batch["position_seq"].to(device)
            particle_type = batch["particle_type"].to(device)
            npp = batch["n_particles_per_example"].to(device)
            next_position = batch["next_position"].to(device)
            next_aux = batch["next_aux"].to(device)

            noise = random_walk_position_noise(position_seq, cgn.noise_std)

            optimizer.zero_grad()
            pred_acc, target_acc, pred_aux = simulator.predict_accelerations(
                next_positions=next_position,
                position_sequence_noise=noise,
                position_sequence=position_seq,
                nparticles_per_example=npp,
                particle_types=particle_type,
            )
            loss_pos = ((pred_acc - target_acc) ** 2).sum(dim=-1)
            next_aux_t = aux_forward_transform(
                next_aux, cgn.aux_transform, cgn.aux_transform_scale
            )
            next_aux_norm = (next_aux_t - aux_mean) / aux_std
            loss_aux = (pred_aux[:, 0] - next_aux_norm) ** 2
            if train_cfg.aux_tail_weight > 0.0:
                # Upweight above-mean (tail) targets so the heavy-tailed crack
                # field's decision region is not starved by the bulk; weights
                # follow the target, never the prediction.
                loss_aux = loss_aux * (
                    1.0 + train_cfg.aux_tail_weight * torch.relu(next_aux_norm)
                )
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

            lr_new = _lr_at(step, train_cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr_new

            step += 1

            # Accumulate on-device so the mean costs one .item() sync per
            # window instead of one per step; the instantaneous train_loss
            # field stays for continuity with pre-2026-07 fleet logs.
            window_loss_sum = (
                loss.detach()
                if window_loss_sum is None
                else window_loss_sum + loss.detach()
            )
            window_loss_n += 1

            if step % train_cfg.val_every == 0:
                train_mean = (window_loss_sum / window_loss_n).item()
                window_loss_sum = None
                window_loss_n = 0
                val_pos, val_aux = _validate(
                    simulator,
                    val_trajs,
                    cgn.input_frames,
                    device,
                    spec.kinematic_types,
                )
                logger.info(
                    "step %d: train_loss %.6f train_mean %.6f val_pos %.4f mm "
                    "val_aux %.4f %s (best_pos %.4f)",
                    step,
                    loss.item(),
                    train_mean,
                    val_pos,
                    val_aux,
                    spec.card.aux_unit,
                    best_pos,
                )
                if val_pos < best_pos:
                    best_pos = val_pos
                    best_ckpt = out_dir / f"model-best-{step:06d}.pt"
                    simulator.save(str(best_ckpt))
                    logger.info("saved improved checkpoint: %s", best_ckpt)
                simulator.train()

            if step % PERIODIC_CKPT_EVERY == 0:
                periodic_ckpt = out_dir / f"ckpt-{step:06d}.pt"
                simulator.save(str(periodic_ckpt))
                logger.info("saved periodic checkpoint: %s", periodic_ckpt)

            if step >= train_cfg.training_steps:
                break

    if best_ckpt is None:
        best_ckpt = out_dir / f"model-final-{step:06d}.pt"
        simulator.save(str(best_ckpt))
        logger.info("no validation improvement; saved final checkpoint: %s", best_ckpt)
    return best_ckpt


def _train_mgn(
    spec: BenchmarkSpec,
    mgn: MGNConfig,
    train_cfg: TrainConfig,
    train_trajs: list[CaseTrajectory],
    val_trajs: list[CaseTrajectory],
    out_dir: Path,
    device: str,
    data_root: Path,
) -> Path | None:
    """Run MGN training with inline validation (ADR-0043 §8/§9a recipe).

    Called by :func:`train` for ``family="mgn"``, after the shared
    trajectory loading and ADR-0039 truncation. Unlike the CGN loop, MGN
    needs no separate normalization-stats file: its four
    :class:`~structbench.models.mgn.normalizers.OnlineNormalizer` buffers
    live inside the checkpoint's own ``state_dict`` (ADR-0043 §8), so
    ``config.json`` plus the checkpoint(s) are the run's only artifacts.

    Each step builds the noisy inputs and the matched target via
    :func:`_mesh_family_noise` (reference path: single-frame Gaussian noise
    on the current frame, MGN gamma = 1 target; ``velocity_history`` path:
    CGN random-walk over the window with the GNS adjusted-next target —
    ADR-0049) on non-kinematic (NORMAL) rows only — kinematic rows are
    never noised, since their motion is prescribed — then calls
    :meth:`~structbench.models.mgn.MeshSimulator.forward_train`.
    The loss is the mean, over non-kinematic
    rows, of ``w_pos * ||Δv||^2 + w_aux * (Δaux)^2`` on the normalized
    (velocity, auxiliary) output. The first ``mgn.normalizer_warmup_steps``
    steps run with ``accumulate=True`` so the online normalizers (node,
    mesh-edge, world-edge, and target) warm up on real batches before their
    outputs are used for anything but accumulation. Every ``val_every``
    steps the simulator is switched to eval mode and rolled out (via
    :func:`~structbench.eval.rollout`, after :meth:`bind_case`/
    :meth:`reset_rollout` per trajectory) over ``val_trajs``; the model is
    saved as ``model-best-<step>.pt`` when the mean position RMSE improves.
    Every :data:`PERIODIC_CKPT_EVERY` steps it additionally snapshots
    ``ckpt-<step>.pt`` for post-hoc analysis (never read by default
    evaluation), mirroring the CGN loop.

    Parameters
    ----------
    spec : BenchmarkSpec
        Benchmark spec supplying ``kinematic_types`` (the NORMAL-only noise
        and loss mask) and ``scored_frames`` (the validation rollout's
        scored span).
    mgn : MGNConfig
        Architecture, noise, and normalizer-warmup configuration.
    train_cfg : TrainConfig
        Optimization schedule and loss weights, shared with the CGN loop.
    train_trajs, val_trajs : list of CaseTrajectory
        Already-loaded trajectories from :func:`train` (already
        ``train_frames``/``scored_frames`` truncated per ADR-0039); both
        must be mesh (nodal-FE) trajectories.
    out_dir : pathlib.Path
        Output directory for checkpoints and the resolved config.
    device : str
        Torch device string.
    data_root : pathlib.Path
        Directory of canonical cases; recorded verbatim in ``config.json``
        (:func:`~structbench.config.resolved_config_dict` requires it).

    Returns
    -------
    pathlib.Path or None
        Path to the best (or fallback final) checkpoint, or ``None`` if no
        checkpoint was written.

    Raises
    ------
    ValueError
        If any trajectory in ``train_trajs`` or ``val_trajs`` lacks mesh
        connectivity (``cells``/``reference_coords`` are ``None``) — the
        benchmark is not a mesh benchmark.
    """
    for tr in (*train_trajs, *val_trajs):
        if tr.cells is None or tr.reference_coords is None:
            raise ValueError(
                f"benchmark {spec.card.name!r} has no mesh connectivity "
                "(cells/reference_coords); mgn training requires a mesh benchmark"
            )

    statics = [mesh_static_from_trajectory(tr) for tr in train_trajs]
    sim = build_mgn_simulator(
        mgn,
        kinematic_types=spec.kinematic_types,
        scripted_types=spec.scripted_types,
        device=device,
    )
    sim.to(device)

    kinematic = torch.as_tensor(
        list(spec.kinematic_types), dtype=torch.long, device=device
    )

    (out_dir / "config.json").write_text(
        json.dumps(
            resolved_config_dict(
                "mgn",
                mgn,
                train_cfg,
                horizon=spec.card.horizon,
                eval_times=spec.card.eval_times,
                n_particle_types=mgn.node_type_size,
                data_root=data_root,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = WindowDataset(train_trajs, mgn.input_frames)
    if len(dataset) == 0:
        raise ValueError(
            f"empty training set: no TRAIN trajectory has more than "
            f"input_frames={mgn.input_frames} frames, so there are no "
            f"autoregressive samples. Check the data root or reduce input_frames."
        )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=functools.partial(collate_mesh_samples, statics=statics),
    )
    optimizer = torch.optim.Adam(sim.parameters(), lr=train_cfg.lr_init)

    logger.info(
        "starting mgn training: %d steps, batch %d",
        train_cfg.training_steps,
        train_cfg.batch_size,
    )

    step = 0
    best_pos = float("inf")
    best_ckpt: Path | None = None
    sim.train()
    while step < train_cfg.training_steps:
        for batch in loader:
            position_seq = batch["position_seq"].to(device)
            particle_type = batch["particle_type"].to(device)
            next_position = batch["next_position"].to(device)
            next_aux = batch["next_aux"].to(device)
            mesh_edge_index = batch["mesh_edge_index"].to(device)
            reference_coords = batch["reference_coords"].to(device)
            n_particles_per_example = batch["n_particles_per_example"].to(device)

            is_kinematic = torch.isin(particle_type, kinematic)
            x_noisy, velocity_history, next_target = _mesh_family_noise(
                position_seq,
                next_position,
                is_kinematic,
                mgn.noise_std,
                mgn.velocity_history,
            )

            optimizer.zero_grad()
            pred, target = sim.forward_train(
                x_noisy,
                next_target,
                next_aux,
                particle_type,
                mesh_edge_index,
                reference_coords,
                n_particles_per_example,
                accumulate=(step < mgn.normalizer_warmup_steps),
                velocity_history=velocity_history,
            )
            delta_v = pred[:, :-1] - target[:, :-1]
            delta_aux = pred[:, -1] - target[:, -1]
            per_particle = (
                train_cfg.w_pos * (delta_v**2).sum(dim=-1)
                + train_cfg.w_aux * delta_aux**2
            )
            free = ~is_kinematic
            if free.any():
                loss = per_particle[free].mean()
            else:
                # all-kinematic batch: nothing to learn from; zero loss, no NaN
                loss = per_particle.new_tensor(0.0, requires_grad=True)

            loss.backward()
            optimizer.step()

            lr_new = _lr_at(step, train_cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr_new

            step += 1

            if step % train_cfg.val_every == 0:
                sim.eval()
                pos_losses: list[float] = []
                with torch.no_grad():
                    for tr in val_trajs:
                        sim.bind_case(
                            torch.from_numpy(tr.cells).to(device),
                            torch.from_numpy(tr.reference_coords).to(device),
                            torch.from_numpy(tr.particle_type).to(device),
                            torch.from_numpy(tr.positions).to(device),
                        )
                        sim.reset_rollout()
                        result = rollout(
                            sim,
                            tr,
                            mgn.input_frames,
                            device,
                            kinematic_types=spec.kinematic_types,
                            scored_frames=spec.scored_frames,
                        )
                        pos_losses.append(float(result.position_rmse.mean()))
                val_pos = (
                    sum(pos_losses) / len(pos_losses) if pos_losses else float("inf")
                )
                logger.info(
                    "step %d: train_loss %.6f val_pos %.4f mm (best_pos %.4f)",
                    step,
                    loss.item(),
                    val_pos,
                    best_pos,
                )
                if val_pos < best_pos:
                    best_pos = val_pos
                    best_ckpt = out_dir / f"model-best-{step:06d}.pt"
                    sim.save(str(best_ckpt))
                    logger.info("saved improved checkpoint: %s", best_ckpt)
                sim.train()

            if step % PERIODIC_CKPT_EVERY == 0:
                periodic_ckpt = out_dir / f"ckpt-{step:06d}.pt"
                sim.save(str(periodic_ckpt))
                logger.info("saved periodic checkpoint: %s", periodic_ckpt)

            if step >= train_cfg.training_steps:
                break

    if best_ckpt is None:
        best_ckpt = out_dir / f"model-final-{step:06d}.pt"
        sim.save(str(best_ckpt))
        logger.info("no validation improvement; saved final checkpoint: %s", best_ckpt)
    return best_ckpt


def _train_transolver(
    spec: BenchmarkSpec,
    cfg: TransolverConfig,
    train_cfg: TrainConfig,
    train_trajs: list[CaseTrajectory],
    val_trajs: list[CaseTrajectory],
    out_dir: Path,
    device: str,
    data_root: Path,
) -> Path | None:
    """Run Transolver training with inline validation (ADR-0041/0044 recipe).

    Called by :func:`train` for ``family="transolver"``, after the shared
    trajectory loading and ADR-0039 truncation. Structurally mirrors
    :func:`_train_mgn` (MGN parity, ADR-0043 §8/§9a): Transolver's two
    :class:`~structbench.models.mgn.normalizers.OnlineNormalizer` buffers
    (node and target) live inside the checkpoint's own ``state_dict``, so
    ``config.json`` plus the checkpoint(s) are the run's only artifacts and
    no separate normalization-stats file is written.

    The optimizer recipe departs from the CGN/MGN loops per ADR-0044 (the
    thuml Transolver reference): the optimizer is AdamW with
    ``cfg.weight_decay``, the learning rate follows the cosine schedule
    :func:`_lr_at_cosine` (a steps-port of the reference's per-epoch
    ``CosineAnnealingLR``, not the CGN/MGN exponential decay), and the
    gradient is clipped to global norm ``cfg.max_grad_norm`` (when positive)
    right after ``loss.backward()``.

    Each step builds the noisy inputs and the matched target via
    :func:`_mesh_family_noise` (reference path: single-frame Gaussian noise
    on the current frame, MGN gamma = 1 target; ``velocity_history`` path:
    CGN random-walk over the window with the GNS adjusted-next target —
    ADR-0049) on non-kinematic (NORMAL) rows only — kinematic rows are
    never noised, since their motion is prescribed — then calls
    :meth:`~structbench.models.transolver.TransolverSimulator.forward_train`. The
    loss is the mean, over non-kinematic rows, of
    ``w_pos * ||Δv||^2 + w_aux * (Δaux)^2`` on the normalized (velocity,
    auxiliary) output. The first ``cfg.normalizer_warmup_steps`` steps run
    with ``accumulate=True`` so the online normalizers (node and target)
    warm up on real batches before their outputs are used for anything but
    accumulation. Every ``val_every`` steps the simulator is switched to
    eval mode and rolled out (via :func:`~structbench.eval.rollout`, after
    :meth:`bind_case`/:meth:`reset_rollout` per trajectory) over
    ``val_trajs``; the model is saved as ``model-best-<step>.pt`` when the
    mean position RMSE improves. Every :data:`PERIODIC_CKPT_EVERY` steps it
    additionally snapshots ``ckpt-<step>.pt`` for post-hoc analysis (never
    read by default evaluation), mirroring the CGN/MGN loops.

    Parameters
    ----------
    spec : BenchmarkSpec
        Benchmark spec supplying ``kinematic_types`` (the NORMAL-only noise
        and loss mask) and ``scored_frames`` (the validation rollout's
        scored span).
    cfg : TransolverConfig
        Architecture, noise, normalizer-warmup, and optimizer-recipe
        configuration.
    train_cfg : TrainConfig
        Optimization schedule and loss weights, shared with the CGN/MGN
        loops (``lr_decay``/``lr_decay_steps`` go unused here: the schedule
        is :func:`_lr_at_cosine`, ADR-0044).
    train_trajs, val_trajs : list of CaseTrajectory
        Already-loaded trajectories from :func:`train` (already
        ``train_frames``/``scored_frames`` truncated per ADR-0039); both
        must be mesh (nodal-FE) trajectories.
    out_dir : pathlib.Path
        Output directory for checkpoints and the resolved config.
    device : str
        Torch device string.
    data_root : pathlib.Path
        Directory of canonical cases; recorded verbatim in ``config.json``
        (:func:`~structbench.config.resolved_config_dict` requires it).

    Returns
    -------
    pathlib.Path or None
        Path to the best (or fallback final) checkpoint, or ``None`` if no
        checkpoint was written.

    Raises
    ------
    ValueError
        If any trajectory in ``train_trajs`` or ``val_trajs`` lacks mesh
        connectivity (``cells``/``reference_coords`` are ``None``) — the
        benchmark is not a mesh benchmark.
    """
    for tr in (*train_trajs, *val_trajs):
        if tr.cells is None or tr.reference_coords is None:
            raise ValueError(
                f"benchmark {spec.card.name!r} has no mesh connectivity "
                "(cells/reference_coords); transolver training requires a "
                "mesh benchmark"
            )

    statics = [mesh_static_from_trajectory(tr) for tr in train_trajs]
    sim = build_transolver_simulator(
        cfg,
        kinematic_types=spec.kinematic_types,
        scripted_types=spec.scripted_types,
        device=device,
    )
    sim.to(device)

    kinematic = torch.as_tensor(
        list(spec.kinematic_types), dtype=torch.long, device=device
    )

    (out_dir / "config.json").write_text(
        json.dumps(
            resolved_config_dict(
                "transolver",
                cfg,
                train_cfg,
                horizon=spec.card.horizon,
                eval_times=spec.card.eval_times,
                n_particle_types=cfg.node_type_size,
                data_root=data_root,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = WindowDataset(train_trajs, cfg.input_frames)
    if len(dataset) == 0:
        raise ValueError(
            f"empty training set: no TRAIN trajectory has more than "
            f"input_frames={cfg.input_frames} frames, so there are no "
            f"autoregressive samples. Check the data root or reduce input_frames."
        )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=functools.partial(collate_mesh_samples, statics=statics),
    )
    optimizer = torch.optim.AdamW(
        sim.parameters(), lr=train_cfg.lr_init, weight_decay=cfg.weight_decay
    )

    logger.info(
        "starting transolver training: %d steps, batch %d",
        train_cfg.training_steps,
        train_cfg.batch_size,
    )

    step = 0
    best_pos = float("inf")
    best_ckpt: Path | None = None
    sim.train()
    while step < train_cfg.training_steps:
        for batch in loader:
            position_seq = batch["position_seq"].to(device)
            particle_type = batch["particle_type"].to(device)
            next_position = batch["next_position"].to(device)
            next_aux = batch["next_aux"].to(device)
            reference_coords = batch["reference_coords"].to(device)
            n_particles_per_example = batch["n_particles_per_example"].to(device)

            is_kinematic = torch.isin(particle_type, kinematic)
            x_noisy, velocity_history, next_target = _mesh_family_noise(
                position_seq,
                next_position,
                is_kinematic,
                cfg.noise_std,
                cfg.velocity_history,
            )

            optimizer.zero_grad()
            pred, target = sim.forward_train(
                x_noisy,
                next_target,
                next_aux,
                particle_type,
                reference_coords,
                n_particles_per_example,
                accumulate=(step < cfg.normalizer_warmup_steps),
                velocity_history=velocity_history,
            )
            delta_v = pred[:, :-1] - target[:, :-1]
            delta_aux = pred[:, -1] - target[:, -1]
            per_particle = (
                train_cfg.w_pos * (delta_v**2).sum(dim=-1)
                + train_cfg.w_aux * delta_aux**2
            )
            free = ~is_kinematic
            if free.any():
                loss = per_particle[free].mean()
            else:
                # all-kinematic batch: nothing to learn from; zero loss, no NaN
                loss = per_particle.new_tensor(0.0, requires_grad=True)

            loss.backward()
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(sim.parameters(), cfg.max_grad_norm)
            optimizer.step()

            lr_new = _lr_at_cosine(step, train_cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr_new

            step += 1

            if step % train_cfg.val_every == 0:
                sim.eval()
                pos_losses: list[float] = []
                with torch.no_grad():
                    for tr in val_trajs:
                        sim.bind_case(
                            torch.from_numpy(tr.cells).to(device),
                            torch.from_numpy(tr.reference_coords).to(device),
                            torch.from_numpy(tr.particle_type).to(device),
                            torch.from_numpy(tr.positions).to(device),
                        )
                        sim.reset_rollout()
                        result = rollout(
                            sim,
                            tr,
                            cfg.input_frames,
                            device,
                            kinematic_types=spec.kinematic_types,
                            scored_frames=spec.scored_frames,
                        )
                        pos_losses.append(float(result.position_rmse.mean()))
                val_pos = (
                    sum(pos_losses) / len(pos_losses) if pos_losses else float("inf")
                )
                logger.info(
                    "step %d: train_loss %.6f val_pos %.4f mm (best_pos %.4f)",
                    step,
                    loss.item(),
                    val_pos,
                    best_pos,
                )
                if val_pos < best_pos:
                    best_pos = val_pos
                    best_ckpt = out_dir / f"model-best-{step:06d}.pt"
                    sim.save(str(best_ckpt))
                    logger.info("saved improved checkpoint: %s", best_ckpt)
                sim.train()

            if step % PERIODIC_CKPT_EVERY == 0:
                periodic_ckpt = out_dir / f"ckpt-{step:06d}.pt"
                sim.save(str(periodic_ckpt))
                logger.info("saved periodic checkpoint: %s", periodic_ckpt)

            if step >= train_cfg.training_steps:
                break

    if best_ckpt is None:
        best_ckpt = out_dir / f"model-final-{step:06d}.pt"
        sim.save(str(best_ckpt))
        logger.info("no validation improvement; saved final checkpoint: %s", best_ckpt)
    return best_ckpt


def _train_geoflare(
    spec: BenchmarkSpec,
    cfg: GeoFlareConfig,
    train_cfg: TrainConfig,
    train_trajs: list[CaseTrajectory],
    val_trajs: list[CaseTrajectory],
    out_dir: Path,
    device: str,
    data_root: Path,
) -> Path | None:
    """Run GeoFLARE training with inline validation (ADR-0041/0044/0045 recipe).

    Called by :func:`train` for ``family="geoflare"``, after the shared
    trajectory loading and ADR-0039 truncation. A clone of
    :func:`_train_transolver` (itself an MGN-parity clone, ADR-0043 §8/§9a):
    ``GeoFlareSimulator``'s two
    :class:`~structbench.models.mgn.normalizers.OnlineNormalizer` buffers
    (node and target) live inside the checkpoint's own ``state_dict``, so
    ``config.json`` plus the checkpoint(s) are the run's only artifacts and
    no separate normalization-stats file is written.

    The optimizer recipe matches Transolver's (ADR-0044, extended to
    GeoFLARE by ADR-0045): the optimizer is AdamW with ``cfg.weight_decay``,
    the learning rate follows the cosine schedule :func:`_lr_at_cosine`, and
    the gradient is clipped to global norm ``cfg.max_grad_norm`` (when
    positive) right after ``loss.backward()`` — the GeoFLARE default
    ``max_grad_norm=0.0`` keeps the clip off, matching the upstream
    reference's own optimizer recipe (no clipping applied).

    Each step builds the noisy inputs and the matched target via
    :func:`_mesh_family_noise` (reference path: single-frame Gaussian noise
    on the current frame, MGN gamma = 1 target; ``velocity_history`` path:
    CGN random-walk over the window with the GNS adjusted-next target —
    ADR-0049) on non-kinematic (NORMAL) rows only — kinematic rows are
    never noised, since their motion is prescribed — then calls
    :meth:`~structbench.models.geoflare.simulator.GeoFlareSimulator.forward_train`
    (its call signature matches Transolver's exactly; coordinate threading
    to the network is internal to the simulator, see that module's
    docstring, so the caller-side noise/loss/warmup plumbing is unchanged
    from :func:`_train_transolver`). The loss is the mean, over
    non-kinematic rows, of ``w_pos * ||Δv||^2 + w_aux * (Δaux)^2`` on the
    normalized (velocity, auxiliary) output. The first
    ``cfg.normalizer_warmup_steps`` steps run with ``accumulate=True`` so
    the online normalizers (node and target) warm up on real batches
    before their outputs are used for anything but accumulation. Every
    ``val_every`` steps the simulator is switched to eval mode and rolled
    out (via :func:`~structbench.eval.rollout`, after
    :meth:`bind_case`/:meth:`reset_rollout` per trajectory) over
    ``val_trajs``; the model is saved as ``model-best-<step>.pt`` when the
    mean position RMSE improves. Every :data:`PERIODIC_CKPT_EVERY` steps it
    additionally snapshots ``ckpt-<step>.pt`` for post-hoc analysis (never
    read by default evaluation), mirroring the CGN/MGN/Transolver loops.

    Parameters
    ----------
    spec : BenchmarkSpec
        Benchmark spec supplying ``kinematic_types`` (the NORMAL-only noise
        and loss mask) and ``scored_frames`` (the validation rollout's
        scored span).
    cfg : GeoFlareConfig
        Architecture, noise, normalizer-warmup, and optimizer-recipe
        configuration.
    train_cfg : TrainConfig
        Optimization schedule and loss weights, shared with the CGN/MGN/
        Transolver loops (``lr_decay``/``lr_decay_steps`` go unused here:
        the schedule is :func:`_lr_at_cosine`, ADR-0044).
    train_trajs, val_trajs : list of CaseTrajectory
        Already-loaded trajectories from :func:`train` (already
        ``train_frames``/``scored_frames`` truncated per ADR-0039); both
        must be mesh (nodal-FE) trajectories.
    out_dir : pathlib.Path
        Output directory for checkpoints and the resolved config.
    device : str
        Torch device string.
    data_root : pathlib.Path
        Directory of canonical cases; recorded verbatim in ``config.json``
        (:func:`~structbench.config.resolved_config_dict` requires it).

    Returns
    -------
    pathlib.Path or None
        Path to the best (or fallback final) checkpoint, or ``None`` if no
        checkpoint was written.

    Raises
    ------
    ValueError
        If any trajectory in ``train_trajs`` or ``val_trajs`` lacks mesh
        connectivity (``cells``/``reference_coords`` are ``None``) — the
        benchmark is not a mesh benchmark.
    """
    for tr in (*train_trajs, *val_trajs):
        if tr.cells is None or tr.reference_coords is None:
            raise ValueError(
                f"benchmark {spec.card.name!r} has no mesh connectivity "
                "(cells/reference_coords); geoflare training requires a "
                "mesh benchmark"
            )

    statics = [mesh_static_from_trajectory(tr) for tr in train_trajs]
    sim = build_geoflare_simulator(
        cfg,
        kinematic_types=spec.kinematic_types,
        scripted_types=spec.scripted_types,
        device=device,
    )
    sim.to(device)

    kinematic = torch.as_tensor(
        list(spec.kinematic_types), dtype=torch.long, device=device
    )

    (out_dir / "config.json").write_text(
        json.dumps(
            resolved_config_dict(
                "geoflare",
                cfg,
                train_cfg,
                horizon=spec.card.horizon,
                eval_times=spec.card.eval_times,
                n_particle_types=cfg.node_type_size,
                data_root=data_root,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = WindowDataset(train_trajs, cfg.input_frames)
    if len(dataset) == 0:
        raise ValueError(
            f"empty training set: no TRAIN trajectory has more than "
            f"input_frames={cfg.input_frames} frames, so there are no "
            f"autoregressive samples. Check the data root or reduce input_frames."
        )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=functools.partial(collate_mesh_samples, statics=statics),
    )
    optimizer = torch.optim.AdamW(
        sim.parameters(), lr=train_cfg.lr_init, weight_decay=cfg.weight_decay
    )

    logger.info(
        "starting geoflare training: %d steps, batch %d",
        train_cfg.training_steps,
        train_cfg.batch_size,
    )

    step = 0
    best_pos = float("inf")
    best_ckpt: Path | None = None
    sim.train()
    while step < train_cfg.training_steps:
        for batch in loader:
            position_seq = batch["position_seq"].to(device)
            particle_type = batch["particle_type"].to(device)
            next_position = batch["next_position"].to(device)
            next_aux = batch["next_aux"].to(device)
            reference_coords = batch["reference_coords"].to(device)
            n_particles_per_example = batch["n_particles_per_example"].to(device)

            is_kinematic = torch.isin(particle_type, kinematic)
            x_noisy, velocity_history, next_target = _mesh_family_noise(
                position_seq,
                next_position,
                is_kinematic,
                cfg.noise_std,
                cfg.velocity_history,
            )

            optimizer.zero_grad()
            pred, target = sim.forward_train(
                x_noisy,
                next_target,
                next_aux,
                particle_type,
                reference_coords,
                n_particles_per_example,
                accumulate=(step < cfg.normalizer_warmup_steps),
                velocity_history=velocity_history,
            )
            delta_v = pred[:, :-1] - target[:, :-1]
            delta_aux = pred[:, -1] - target[:, -1]
            per_particle = (
                train_cfg.w_pos * (delta_v**2).sum(dim=-1)
                + train_cfg.w_aux * delta_aux**2
            )
            free = ~is_kinematic
            if free.any():
                loss = per_particle[free].mean()
            else:
                # all-kinematic batch: nothing to learn from; zero loss, no NaN
                loss = per_particle.new_tensor(0.0, requires_grad=True)

            loss.backward()
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(sim.parameters(), cfg.max_grad_norm)
            optimizer.step()

            lr_new = _lr_at_cosine(step, train_cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr_new

            step += 1

            if step % train_cfg.val_every == 0:
                sim.eval()
                pos_losses: list[float] = []
                with torch.no_grad():
                    for tr in val_trajs:
                        sim.bind_case(
                            torch.from_numpy(tr.cells).to(device),
                            torch.from_numpy(tr.reference_coords).to(device),
                            torch.from_numpy(tr.particle_type).to(device),
                            torch.from_numpy(tr.positions).to(device),
                        )
                        sim.reset_rollout()
                        result = rollout(
                            sim,
                            tr,
                            cfg.input_frames,
                            device,
                            kinematic_types=spec.kinematic_types,
                            scored_frames=spec.scored_frames,
                        )
                        pos_losses.append(float(result.position_rmse.mean()))
                val_pos = (
                    sum(pos_losses) / len(pos_losses) if pos_losses else float("inf")
                )
                logger.info(
                    "step %d: train_loss %.6f val_pos %.4f mm (best_pos %.4f)",
                    step,
                    loss.item(),
                    val_pos,
                    best_pos,
                )
                if val_pos < best_pos:
                    best_pos = val_pos
                    best_ckpt = out_dir / f"model-best-{step:06d}.pt"
                    sim.save(str(best_ckpt))
                    logger.info("saved improved checkpoint: %s", best_ckpt)
                sim.train()

            if step % PERIODIC_CKPT_EVERY == 0:
                periodic_ckpt = out_dir / f"ckpt-{step:06d}.pt"
                sim.save(str(periodic_ckpt))
                logger.info("saved periodic checkpoint: %s", periodic_ckpt)

            if step >= train_cfg.training_steps:
                break

    if best_ckpt is None:
        best_ckpt = out_dir / f"model-final-{step:06d}.pt"
        sim.save(str(best_ckpt))
        logger.info("no validation improvement; saved final checkpoint: %s", best_ckpt)
    return best_ckpt


def _find_checkpoint(out_dir: Path) -> Path | None:
    """Return the highest-step ``model-*.pt`` in ``out_dir``.

    Selection is by the step number embedded in the (zero-padded) filename,
    not filesystem mtime, so a run directory whose mtimes were scrambled by a
    copy or transfer still resolves to the latest (best) checkpoint. Periodic
    ``ckpt-*.pt`` snapshots are deliberately outside the glob: default
    evaluation always scores the run's selected (best/final) checkpoint.
    """

    def _step(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    checkpoints = sorted(out_dir.glob("model-*.pt"), key=_step)
    return checkpoints[-1] if checkpoints else None


def _json_safe(obj: Any) -> Any:
    """Recursively map non-finite floats to ``None`` for strict JSON output.

    A diverged rollout can yield NaN/Inf metrics; the default ``json.dumps``
    emits bare ``NaN``/``Infinity`` tokens that strict JSON parsers reject, so
    the run directory's evidence files would be unreadable exactly when a run
    misbehaves.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


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


def _model_config_from_record(
    record: dict[str, Any],
) -> CGNConfig | MGNConfig | TransolverConfig | GeoFlareConfig:
    """Reconstruct the family-appropriate model config from a run record.

    Parameters
    ----------
    record : dict
        A run record as returned by
        :func:`~structbench.config.read_run_record`, whose ``"model"``
        section carries ``"family"`` plus every field of that family's
        config dataclass (:data:`~structbench.config.MODEL_FAMILIES`).

    Returns
    -------
    CGNConfig, MGNConfig, TransolverConfig, or GeoFlareConfig
        Constructed from ``record["model"]`` with ``"family"`` dropped
        before the fields are splatted in. The legacy ``"gns"`` alias
        (pre-ADR-0034 records) resolves to :class:`CGNConfig` like ``"cgn"``.
    """
    model_table = {k: v for k, v in record["model"].items() if k != "family"}
    model_cls = MODEL_FAMILIES[record["model"]["family"]]
    return model_cls(**model_table)


def evaluate(
    case_ids: list[str],
    data_root: Path,
    out_dir: Path,
    device: str,
    *,
    split_name: str = "eval",
    save_artifacts: bool = True,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Roll out the run's checkpoint over ``case_ids`` and report ADR-0019 §5.

    The simulator is rebuilt entirely from the run directory's own record —
    architecture and ``n_particle_types`` from ``config.json``, model family
    from ``config.json["model"]["family"]`` via
    :func:`_model_config_from_record` — so evaluation always matches the
    trained checkpoint; no caller-supplied architecture is accepted. A
    ``cgn``-family run additionally rebuilds its normalizer from
    ``normalization_stats.npz`` (both files are written by :func:`train`); an
    ``mgn``-, ``transolver``-, or ``geoflare``-family run needs no stats file
    at all, since its normalizer buffers live in the checkpoint's own
    ``state_dict`` (ADR-0043 §8, verified in ①-c1; extended to
    ``transolver``/``geoflare`` in ADR-0041).

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
        Run directory holding the checkpoint, resolved config, and (for a
        ``cgn``-family run) normalization stats.
    device : str
        Torch device string.
    split_name : str
        Label recorded in the report and used in artifact filenames.
    save_artifacts : bool
        Write the metrics JSON and per-case rollout ``.npz`` files.
    checkpoint : str, pathlib.Path or None
        Explicit checkpoint file to evaluate (e.g. a periodic
        ``ckpt-<step>.pt``); a relative path is resolved against ``out_dir``.
        When given, the metrics file is suffixed
        (``metrics-<split_name>@<checkpoint stem>.json``) and rollout ``.npz``
        artifacts are skipped, so the canonical selected-checkpoint artifacts
        are never overwritten. ``None`` evaluates the run's selected
        checkpoint (highest-step ``model-*.pt``).

    Notes
    -----
    Rollouts seed with the checkpoint's ``input_frames`` (recorded in
    ``config.json``; pre-0035 runs recorded it as ``window``, normalized by
    :func:`~structbench.config.read_run_record`), so checkpoints are always
    evaluated as trained (ADR-0035). For an ``mgn``-, ``transolver``-, or
    ``geoflare``-family run, each case's mesh (cells, reference coordinates,
    and full ground-truth trajectory) is bound to the simulator before its
    three eval passes (rollout, one-step position, one-step aux), and the
    rollout pointer is reset before each pass, per
    :class:`~structbench.models.common.CaseBoundSimulator`'s statefulness
    contract.

    Returns
    -------
    dict
        ``{"split", "checkpoint", "checkpoint_path", "cases": {case_id: ...},
        "mean": ...}`` with plain JSON-serializable values.

    Raises
    ------
    FileNotFoundError
        If ``config.json`` or a checkpoint is missing from ``out_dir``; for a
        ``cgn``-family run, also if ``normalization_stats.npz`` is missing
        (an ``mgn``-, ``transolver``-, or ``geoflare``-family run needs no
        stats file).
    ValueError
        If ``case_ids`` is empty, or an ``mgn``-, ``transolver``-, or
        ``geoflare``-family run is evaluated against a benchmark whose cases
        carry no mesh connectivity (``cells``/``reference_coords`` are
        ``None``).
    """
    if not case_ids:
        raise ValueError("case_ids must be non-empty")
    spec, record = _resolve_run_spec(out_dir)
    family = record["model"]["family"]
    model_cfg = _model_config_from_record(record)
    n_types = int(record["n_particle_types"])

    simulator: (
        LearnedSimulator | MeshSimulator | TransolverSimulator | GeoFlareSimulator
    )
    if family == "mgn":
        assert isinstance(model_cfg, MGNConfig)
        simulator = build_mgn_simulator(
            model_cfg,
            kinematic_types=spec.kinematic_types,
            scripted_types=spec.scripted_types,
            device=device,
        )
    elif family == "transolver":
        assert isinstance(model_cfg, TransolverConfig)
        simulator = build_transolver_simulator(
            model_cfg,
            kinematic_types=spec.kinematic_types,
            scripted_types=spec.scripted_types,
            device=device,
        )
    elif family == "geoflare":
        assert isinstance(model_cfg, GeoFlareConfig)
        simulator = build_geoflare_simulator(
            model_cfg,
            kinematic_types=spec.kinematic_types,
            scripted_types=spec.scripted_types,
            device=device,
        )
    else:
        stats_path = out_dir / "normalization_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(f"missing normalization stats: {stats_path}")
        stats = NormalizationStats.load(stats_path)
        assert isinstance(model_cfg, CGNConfig)
        simulator = build_simulator(
            _stats_to_dict(stats),
            model_cfg,
            n_particle_types=n_types,
            boundary_feature_fn=_bind_boundary_feature(spec, model_cfg),
            device=device,
        )
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        # Relative paths resolve against out_dir ONLY (never the CWD): fleet
        # arms all hold identically named ckpt-<step>.pt snapshots, so a CWD
        # fallback would silently score another arm's weights.
        if not ckpt_path.is_absolute():
            ckpt_path = out_dir / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    else:
        found = _find_checkpoint(out_dir)
        if found is None:
            raise FileNotFoundError(f"no checkpoint found under {out_dir}")
        ckpt_path = found
    simulator.load(str(ckpt_path))
    simulator.to(device)
    simulator.eval()
    # Non-None for any CaseBoundSimulator arm (mgn, transolver and geoflare, ADR-0041):
    # gates the per-case bind_case/reset_rollout calls below without
    # re-checking `family` at each site.
    mesh_sim = simulator if isinstance(simulator, CaseBoundSimulator) else None

    # Explicit-checkpoint sweeps must not clobber the selected checkpoint's
    # canonical artifacts: suffix the metrics file and skip the rollout .npz.
    save_rollouts = save_artifacts and checkpoint is None
    metrics_tag = split_name if checkpoint is None else f"{split_name}@{ckpt_path.stem}"
    rollout_dir = out_dir / "rollouts"
    if save_rollouts:
        rollout_dir.mkdir(parents=True, exist_ok=True)

    cases: dict[str, dict[str, Any]] = {}
    for case_id in case_ids:
        trajectory = load_case_trajectory(
            data_root / f"{case_id}.h5", aux_field=spec.aux_field
        )
        if mesh_sim is not None and spec.mesh_transform is not None:
            # ADR-0047: same synthesis the training load applied.
            trajectory = spec.mesh_transform(trajectory)
        if mesh_sim is not None:
            if trajectory.cells is None or trajectory.reference_coords is None:
                raise ValueError(
                    f"benchmark {spec.card.name!r} has no mesh connectivity "
                    f"(cells/reference_coords); {family} evaluation requires "
                    "a mesh benchmark"
                )
            mesh_sim.bind_case(
                torch.from_numpy(trajectory.cells).to(device),
                torch.from_numpy(trajectory.reference_coords).to(device),
                torch.from_numpy(trajectory.particle_type).to(device),
                torch.from_numpy(trajectory.positions).to(device),
            )
            mesh_sim.reset_rollout()
        result = rollout(
            simulator,
            trajectory,
            model_cfg.input_frames,
            device,
            qois=spec.qois,
            kinematic_types=spec.kinematic_types,
            scored_frames=spec.scored_frames,
        )
        if mesh_sim is not None:
            mesh_sim.reset_rollout()
        one_step = one_step_position_rmse(
            simulator,
            trajectory,
            model_cfg.input_frames,
            device,
            kinematic_types=spec.kinematic_types,
        )
        if mesh_sim is not None:
            mesh_sim.reset_rollout()
        one_step_aux = one_step_aux_rmse(
            simulator,
            trajectory,
            model_cfg.input_frames,
            device,
            kinematic_types=spec.kinematic_types,
        )
        # One-step aggregates cover the same scored span as the rollout means
        # (ADR-0035 parity, ADR-0039 horizon); per-frame arrays stay full.
        n_scored = (
            None
            if spec.scored_frames is None
            else min(spec.scored_frames, len(trajectory.time)) - model_cfg.input_frames
        )
        cases[case_id] = {
            "one_step_position_rmse": float(one_step[:n_scored].mean()),
            "one_step_aux_rmse": float(one_step_aux[:n_scored].mean()),
            "rollout_position_rmse": result.mean_position_rmse,
            "rollout_aux_rmse": result.mean_aux_rmse,
            # Full-horizon diagnostic (ADR-0039 §3): mean over every predicted
            # frame to trajectory end. Non-leaderboard; equals the scored value
            # when the benchmark pins no horizon. Field name matches the
            # 2026-07-20 bless-fleet rescore (metrics-rescore-adr0039.json).
            "rollout_position_rmse_full": float(result.position_rmse.mean()),
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
        if save_rollouts:
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
        "checkpoint": ckpt_path.name,
        # Full resolved path so an explicitly scored checkpoint (possibly from
        # outside out_dir, via an absolute --checkpoint) stays traceable.
        "checkpoint_path": str(ckpt_path),
        "input_frames": model_cfg.input_frames,
        # Metric definition (ADR-0039): rollout/one-step means and QoIs cover
        # frames [input_frames, scored_frames); null means scored to the end.
        # Records with different scored_frames are not comparable.
        "scored_frames": spec.scored_frames,
        # Card-conforming by construction: a checkpoint's input_frames is
        # validated equal to the card's at config load and train (ADR-0035),
        # so a standard run stays standard on re-eval. Legacy off-card records
        # (e.g. a pre-0035 window=11 run re-evaluated here) read as non-standard.
        "protocol_standard": bool(record["protocol"].get("standard", True))
        and model_cfg.input_frames == spec.card.input_frames,
        "aux_field": spec.aux_field,
        "aux_unit": spec.card.aux_unit,
        "cases": cases,
        "mean": {
            "one_step_position_rmse": _mean_over_cases("one_step_position_rmse"),
            "one_step_aux_rmse": _mean_over_cases("one_step_aux_rmse"),
            "rollout_position_rmse": _mean_over_cases("rollout_position_rmse"),
            "rollout_aux_rmse": _mean_over_cases("rollout_aux_rmse"),
            "rollout_position_rmse_full": _mean_over_cases(
                "rollout_position_rmse_full"
            ),
            "qoi_abs_error": {
                name: float(
                    np.mean([abs(case["qoi_error"][name]) for case in cases.values()])
                )
                for name in spec.qois
            },
        },
    }
    if save_artifacts:
        (out_dir / f"metrics-{metrics_tag}.json").write_text(
            json.dumps(_json_safe(metrics), indent=2, allow_nan=False),
            encoding="utf-8",
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
    parser = argparse.ArgumentParser(description="StructBench CGN training entry")
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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Evaluate this specific checkpoint file (e.g. a periodic "
        "ckpt-<step>.pt; a relative path resolves against --out). "
        "valid/rollout only. Metrics land in metrics-<split>@<name>.json and "
        "rollout .npz artifacts are skipped, so the canonical "
        "selected-checkpoint artifacts are never overwritten.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if args.mode == "train" and args.config is None:
        print("error: --config is required in train mode (ADR-0032)")
        return 2
    if args.mode == "train" and args.checkpoint is not None:
        print("error: --checkpoint applies to valid/rollout modes only")
        return 2
    run_config = load_run_config(args.config) if args.config is not None else None

    if args.data_root is None:
        print("error: --data-root is required")
        return 2
    data_root = Path(args.data_root)

    if args.out is not None:
        out_dir = Path(args.out)
    elif args.mode == "train":
        out_dir = Path("runs") / datetime.now().strftime("run-%Y%m%d-%H%M%S")
    else:
        print(
            "error: --out is required in valid/rollout mode "
            "(the existing run directory to evaluate)"
        )
        return 2

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
        )
        print(f"training complete; best checkpoint: {ckpt}")
    else:
        spec, _resolved = _resolve_run_spec(out_dir)
        if args.mode == "valid":
            metrics = evaluate(
                list(spec.splits["val"]),
                data_root,
                out_dir,
                device,
                split_name="val",
                checkpoint=args.checkpoint,
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
                    checkpoint=args.checkpoint,
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
