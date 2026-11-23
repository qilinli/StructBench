"""The CGN training loop and the dispatch to the other families."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader

from ...benchmarks import (
    BenchmarkSpec,
    available_benchmarks,
    get_benchmark,
)
from ...config import (
    TRAINING_SPANS,
    CGNConfig,
    GeoFlareConfig,
    MGNConfig,
    TrainConfig,
    TransolverConfig,
    resolved_config_dict,
)
from ...datasets import (
    CaseTrajectory,
    WindowDataset,
    aux_forward_transform_channels,
    cached_compute_stats,
    collate_samples,
    expand_aux_knob,
    load_case_trajectory,
)
from ...eval import (
    rollout,
)
from ...models.cgn import LearnedSimulator
from .builders import build_simulator
from .common import (
    _MESH_FAMILIES,
    PERIODIC_CKPT_EVERY,
    _bind_boundary_feature,
    _definition_record,
    _lr_at,
    _record_training_pool,
    _state_fields_for,
    _stats_to_dict,
    _truncate_trajectory,
    random_walk_position_noise,
)
from .geoflare import _train_geoflare
from .mgn import _train_mgn
from .transolver import _train_transolver

logger = logging.getLogger(__name__)


def _load_trajectories(
    case_ids: list[str],
    data_root: Path,
    aux_field: str | tuple[str, ...],
    state_fields: tuple[str, ...] | None = None,
) -> list[CaseTrajectory]:
    """Load each ``<data_root>/<case_id>.h5`` into a :class:`CaseTrajectory`."""
    return [
        load_case_trajectory(
            data_root / f"{case_id}.h5", aux_field=aux_field, state_fields=state_fields
        )
        for case_id in case_ids
    ]


def _training_cut(
    spec: BenchmarkSpec, train_cfg: TrainConfig, case_id: str
) -> int | None:
    """The exclusive frame bound of one case's share of the training pool
    (ADR-0078), ``None`` for the whole stored record.

    Under ``training_span = "scored"`` it is the case's scored bound, read
    through the one accessor every stage uses; ``train_frames`` is an
    explicit fixed cut on top of either setting.
    """
    cuts: list[int] = []
    if train_cfg.training_span == "scored":
        bound = spec.scored_frames_for(case_id)
        if bound is not None:
            cuts.append(bound)
    if train_cfg.train_frames > 0:
        cuts.append(train_cfg.train_frames)
    return min(cuts) if cuts else None


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
) -> tuple[float, float, float]:
    """Mean rollout displacement relative L2, position RMSE (mm) and aux
    RMSE (aux unit) over VAL, in that order.

    Selection reads the first (ADR-0077): the pooled relative L2 of
    displacement, the number :func:`evaluate` writes as
    ``rollout_rel_l2_displacement``. The other two are logged, unselected,
    and kept apart (ADR-0028): a mm + MPa sum was 98% stress and let
    selection ignore position quality entirely.

    Parameters
    ----------
    input_frames:
        History length / rollout seed count, forwarded to :func:`rollout`
        (ADR-0035); the benchmark card's protocol value.
    kinematic_types:
        Forwarded to :func:`rollout`; kinematic particles are excluded from
        the reported numbers (ADR-0026).
    """
    simulator.eval()
    rel_losses: list[float] = []
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
        rel_losses.append(float(result.mean_rel_l2_displacement))
        pos_losses.append(float(result.position_rmse.mean()))
        aux_losses.append(float(result.aux_rmse.mean()))
    if not pos_losses:
        return float("inf"), float("inf"), float("inf")
    return (
        sum(rel_losses) / len(rel_losses),
        sum(pos_losses) / len(pos_losses),
        sum(aux_losses) / len(aux_losses),
    )


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
    split and saves the model when the mean displacement relative L2 improves
    (ADR-0077). The resolved config
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

    # ADR-0059: the run's aux channel selection — the benchmark's canonical
    # declaration unless the config sets train.aux_fields (variant fleets).
    # This is the single point where the aux block enters the training/val
    # pipeline, so the loss, in-training validation, and normalization all
    # follow it. Supersedes (and retires) the env-gated E-X swap screens.
    aux_field = train_cfg.aux_fields or spec.aux_field
    # Record the RESOLVED selection in config.json (ADR-0059) so evaluate()
    # and post-hoc tooling read the channels the run actually trained on,
    # never a null needing spec-side reconstruction. On a COPY: train() must
    # not mutate the caller's config object (programmatic sweeps reuse it).
    train_cfg = replace(
        train_cfg,
        aux_fields=(aux_field,) if isinstance(aux_field, str) else tuple(aux_field),
    )
    if tuple(train_cfg.aux_fields or ()) not in ((), (spec.aux_field,)):
        logger.info(
            "aux_fields override ACTIVE (ADR-0059): benchmark %r trains its "
            "aux block on %r instead of %r; normalization follows the "
            "selection automatically",
            train_cfg.benchmark,
            aux_field,
            spec.aux_field,
        )

    train_ids = list(spec.splits["train"])
    logger.info("loading %d TRAIN trajectories from %s", len(train_ids), data_root)
    state_fields = _state_fields_for(family, model_cfg, spec)
    train_trajs = _load_trajectories(train_ids, data_root, aux_field, state_fields)
    if train_cfg.training_span not in TRAINING_SPANS:
        raise ValueError(
            f"training_span must be one of {', '.join(TRAINING_SPANS)}; "
            f"got {train_cfg.training_span!r}"
        )
    if (
        train_cfg.train_frames > 0
        and train_cfg.train_frames <= model_cfg.input_frames + 1
    ):
        raise ValueError(
            f"train_frames={train_cfg.train_frames} leaves no training "
            f"window (input_frames={model_cfg.input_frames})"
        )
    # ADR-0078 (extending ADR-0039 §4): the pool is cut per case, here and
    # only here, before the statistics and before any loop builds a sample.
    # Every loop draws from these trajectories, so no sample holds a frame at
    # or beyond its case's cut, and the statistics (cached on a signature
    # that covers frame counts and content) follow the cut pool.
    stored = sum(int(tr.positions.shape[0]) for tr in train_trajs)
    train_trajs = [
        _truncate_trajectory(tr, _training_cut(spec, train_cfg, tr.case_id))
        for tr in train_trajs
    ]
    kept = sum(int(tr.positions.shape[0]) for tr in train_trajs)
    logger.info(
        "training span %r (train_frames=%d): the pool keeps %d of %d stored "
        "TRAIN frames (ADR-0078)",
        train_cfg.training_span,
        train_cfg.train_frames,
        kept,
        stored,
    )
    val_trajs = _load_trajectories(
        list(spec.splits["val"]), data_root, aux_field, state_fields
    )
    if spec.scored_frames is not None:
        # In-training validation rolls out only the scored span (ADR-0039):
        # checkpoint selection then keys on the same window the benchmark
        # reports, and each val pass costs proportionally less. The bound may
        # be the case's own (ADR-0039 note, 2026-11-06). It is an exclusive
        # frame index, so frames [0, bound) hold the whole scored span
        # [input_frames, bound). The loops must not read a time reference off
        # these truncated lengths.
        val_trajs = [
            _truncate_trajectory(tr, spec.scored_frames_for(tr.case_id))
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
        aux_field=aux_field,
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
    n_aux_channels = int(aux_mean.numel())
    aux_transforms = expand_aux_knob(cgn.aux_transform, n_aux_channels)
    aux_transform_scales = expand_aux_knob(cgn.aux_transform_scale, n_aux_channels)
    # None disables the tail-weight branch entirely (the reference path);
    # otherwise a (C,) tensor weights each channel's relu(z_target) term.
    _tails = expand_aux_knob(train_cfg.aux_tail_weight, n_aux_channels)
    aux_tail_weights = (
        torch.tensor(_tails, dtype=torch.float32, device=device)
        if any(w > 0.0 for w in _tails)
        else None
    )

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
                definition=_definition_record(spec),
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
    _record_training_pool(
        out_dir,
        spec,
        train_cfg,
        train_trajs,
        lambda trajs: WindowDataset(trajs, cgn.input_frames),
        len(dataset),
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
    best_rel = float("inf")
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
            next_aux_t = aux_forward_transform_channels(
                next_aux, aux_transforms, aux_transform_scales
            )
            next_aux_norm = (next_aux_t - aux_mean) / aux_std  # (P, C)
            # ADR-0059: per-channel squared error, averaged over channels —
            # identical to the historical scalar loss at C = 1.
            loss_aux = ((pred_aux - next_aux_norm) ** 2).mean(dim=-1)
            if aux_tail_weights is not None:
                # Upweight above-mean (tail) targets so the heavy-tailed crack
                # field's decision region is not starved by the bulk; weights
                # follow the target, never the prediction.
                loss_aux = (
                    (pred_aux - next_aux_norm) ** 2
                    * (1.0 + aux_tail_weights * torch.relu(next_aux_norm))
                ).mean(dim=-1)
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
                val_rel, val_pos, val_aux = _validate(
                    simulator,
                    val_trajs,
                    cgn.input_frames,
                    device,
                    spec.kinematic_types,
                )
                logger.info(
                    "step %d: train_loss %.6f train_mean %.6f val_rel_l2 %.6f "
                    "val_pos %.4f mm val_aux %.4f %s (best_rel_l2 %.6f)",
                    step,
                    loss.item(),
                    train_mean,
                    val_rel,
                    val_pos,
                    val_aux,
                    spec.card.aux_unit,
                    best_rel,
                )
                if val_rel < best_rel:
                    best_rel = val_rel
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
