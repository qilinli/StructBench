"""The GeoFLARE training loops, autoregressive and time-conditioned."""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ...benchmarks import (
    BenchmarkSpec,
)
from ...config import (
    GeoFlareConfig,
    TrainConfig,
    resolved_config_dict,
)
from ...datasets import (
    CaseTrajectory,
    WindowDataset,
)
from ...eval import (
    rollout,
    time_conditioned_rollout,
)
from ...models.mgn import (
    collate_mesh_samples,
    mesh_static_from_trajectory,
)
from .builders import build_geoflare_simulator
from .common import (
    _VAL_LOG,
    PERIODIC_CKPT_EVERY,
    _definition_record,
    _lr_at_cosine,
    _mesh_family_noise,
    _record_training_pool,
    _training_time_ref,
)

logger = logging.getLogger(__name__)


def _train_geoflare_tc(
    spec: BenchmarkSpec,
    cfg: GeoFlareConfig,
    train_cfg: TrainConfig,
    train_trajs: list[CaseTrajectory],
    val_trajs: list[CaseTrajectory],
    out_dir: Path,
    device: str,
    data_root: Path,
) -> Path | None:
    """Time-conditioned GeoFLARE training (ADR-0054).

    The faithful thuml structural (Plasticity) scheme: history-free and
    non-autoregressive. Each sample is a single scored frame ``t``; the model
    is trained to map ``(static geometry, node types, scalar impact velocity?,
    prescribed boundary state at t, normalized query time t) -> absolute state
    at t`` (regressed as rest-frame displacement + aux). There is no rollout
    window, no injected noise (inert without autoregression, ADR-0054 decision
    3), and no k-frames bundling. The optimizer recipe (AdamW + cosine LR +
    grad-norm clip) and the online-normalizer warmup mirror
    :func:`_train_geoflare`; validation rolls out the independent-query
    :func:`~structbench.eval.time_conditioned_rollout` per case and selects on
    its mean displacement relative L2 (ADR-0077).

    The normalized query time is ``t = frame / (time_ref_frames - 1)``, with
    ``time_ref_frames`` fixed via :func:`_tc_time_ref_frames` so train and eval
    agree on what each ``t`` means.
    """
    if cfg.noise_std:
        logger.warning(
            "noise_std=%.4g is IGNORED for time_conditioned=true: the "
            "time-conditioned scheme is non-autoregressive, so injected "
            "single-step noise is inert (ADR-0054). Set noise_std=0 to silence.",
            cfg.noise_std,
        )

    loading_scalars: list[float] | None = None
    if cfg.impact_velocity_feature:
        if spec.loading_scalar is None:
            raise ValueError(
                f"benchmark {spec.card.name!r} has no loading_scalar (scalar "
                "impact-velocity parameter), but the geoflare config sets "
                "impact_velocity_feature=True (ADR-0051 B / ADR-0054)."
            )
        loading_scalars = [spec.loading_scalar(tr.case_id) for tr in train_trajs]

    min_len = min(int(tr.positions.shape[0]) for tr in train_trajs)
    time_ref = _training_time_ref(spec, train_cfg, min_len)
    if time_ref < 2:
        raise ValueError(
            f"time-conditioned time_ref_frames={time_ref} < 2 (scored_frames="
            f"{spec.scored_frames_fixed}, train_frames={train_cfg.train_frames}, "
            f"min train length={min_len})"
        )

    statics = [mesh_static_from_trajectory(tr) for tr in train_trajs]
    sim = build_geoflare_simulator(
        cfg,
        n_aux=int(train_trajs[0].aux.shape[-1]),  # ADR-0059
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
                definition=_definition_record(spec),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = WindowDataset(train_trajs, cfg.input_frames, target_frames=1)
    if len(dataset) == 0:
        raise ValueError(
            "empty training set: no TRAIN trajectory has more than "
            f"input_frames = {cfg.input_frames} frames, so there are no "
            "time-conditioned query frames (check the data root or reduce "
            "input_frames)."
        )
    _record_training_pool(
        out_dir,
        spec,
        train_cfg,
        train_trajs,
        lambda trajs: WindowDataset(trajs, cfg.input_frames, target_frames=1),
        len(dataset),
        time_ref=time_ref,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=functools.partial(
            collate_mesh_samples,
            statics=statics,
            loading_scalars=loading_scalars,
            include_target_frame=True,
        ),
    )
    optimizer = torch.optim.AdamW(
        sim.parameters(), lr=train_cfg.lr_init, weight_decay=cfg.weight_decay
    )

    logger.info(
        "starting geoflare training: %d steps, batch %d, time-conditioned "
        "(ADR-0054), time_ref_frames=%d%s",
        train_cfg.training_steps,
        train_cfg.batch_size,
        time_ref,
        " (impact-velocity conditioned)" if cfg.impact_velocity_feature else "",
    )

    step = 0
    best_rel = float("inf")
    best_ckpt: Path | None = None
    sim.train()
    while step < train_cfg.training_steps:
        for batch in loader:
            particle_type = batch["particle_type"].to(device)
            next_position = batch["next_position"].to(device)  # (P, dim) GT at t
            next_aux = batch["next_aux"].to(device)  # (P, C) GT aux at t
            reference_coords = batch["reference_coords"].to(device)
            n_particles_per_example = batch["n_particles_per_example"].to(device)
            target_frame = batch["target_frame"].to(device)  # (B,)
            loading_feature = batch.get("loading_feature")
            if loading_feature is not None:
                loading_feature = loading_feature.to(device)

            t_norm = target_frame.to(torch.float32) / (time_ref - 1)  # (B,)
            is_kinematic = torch.isin(particle_type, kinematic)
            accumulate = step < cfg.normalizer_warmup_steps
            optimizer.zero_grad()
            pred, target = sim.forward_train_tc(
                next_position,
                next_aux,
                particle_type,
                reference_coords,
                n_particles_per_example,
                t_norm,
                accumulate=accumulate,
                loading_feature=loading_feature,
            )
            aux_c = next_aux.shape[-1]  # ADR-0059: trailing aux block
            delta_v = pred[..., :-aux_c] - target[..., :-aux_c]
            delta_aux = pred[..., -aux_c:] - target[..., -aux_c:]
            per_particle = train_cfg.w_pos * (delta_v**2).sum(
                dim=-1
            ) + train_cfg.w_aux * (delta_aux**2).mean(dim=-1)
            free = ~is_kinematic
            if free.any():
                loss = per_particle[free].mean()
            else:
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
                rel_losses: list[float] = []
                with torch.no_grad():
                    for tr in val_trajs:
                        sim.bind_case(
                            torch.from_numpy(tr.cells).to(device),
                            torch.from_numpy(tr.reference_coords).to(device),
                            torch.from_numpy(tr.particle_type).to(device),
                            torch.from_numpy(tr.positions).to(device),
                            loading_scalar=(
                                spec.loading_scalar(tr.case_id)
                                if cfg.impact_velocity_feature and spec.loading_scalar
                                else None
                            ),
                        )
                        # The clock the run trains on (``time_ref``), never
                        # one re-derived from a validation trajectory, which
                        # is truncated to its scored span.
                        result = time_conditioned_rollout(
                            sim,
                            tr,
                            cfg.input_frames,
                            time_ref,
                            device,
                            kinematic_types=spec.kinematic_types,
                            scored_frames=spec.scored_frames_for(tr.case_id),
                        )
                        pos_losses.append(float(result.position_rmse.mean()))
                        rel_losses.append(float(result.mean_rel_l2_displacement))
                val_pos = (
                    sum(pos_losses) / len(pos_losses) if pos_losses else float("inf")
                )
                # ADR-0077: the selector, the evaluator's headline number.
                val_rel = (
                    sum(rel_losses) / len(rel_losses) if rel_losses else float("inf")
                )
                logger.info(_VAL_LOG, step, loss.item(), val_rel, val_pos, best_rel)
                if val_rel < best_rel:
                    best_rel = val_rel
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
    on the current frame, MGN gamma = 1 target; ``history_frames > 0`` path:
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
    mean displacement relative L2 improves (ADR-0077). Every
    :data:`PERIODIC_CKPT_EVERY` steps it
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

    # ADR-0054: the time-conditioned scheme is a distinct, history-free,
    # non-autoregressive path — a dedicated loop, not the AR machinery below
    # (mirrors _train_transolver's dispatch).
    if cfg.time_conditioned:
        return _train_geoflare_tc(
            spec, cfg, train_cfg, train_trajs, val_trajs, out_dir, device, data_root
        )

    statics = [mesh_static_from_trajectory(tr) for tr in train_trajs]
    sim = build_geoflare_simulator(
        cfg,
        n_aux=int(train_trajs[0].aux.shape[-1]),  # ADR-0059
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
                definition=_definition_record(spec),
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
    _record_training_pool(
        out_dir,
        spec,
        train_cfg,
        train_trajs,
        lambda trajs: WindowDataset(trajs, cfg.input_frames),
        len(dataset),
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
    best_rel = float("inf")
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
                cfg.history_frames,
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
            aux_c = next_aux.shape[-1]  # ADR-0059: trailing aux block
            delta_v = pred[:, :-aux_c] - target[:, :-aux_c]
            delta_aux = pred[:, -aux_c:] - target[:, -aux_c:]
            per_particle = train_cfg.w_pos * (delta_v**2).sum(
                dim=-1
            ) + train_cfg.w_aux * (delta_aux**2).mean(dim=-1)
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
                rel_losses: list[float] = []
                with torch.no_grad():
                    for tr in val_trajs:
                        sim.bind_case(
                            torch.from_numpy(tr.cells).to(device),
                            torch.from_numpy(tr.reference_coords).to(device),
                            torch.from_numpy(tr.particle_type).to(device),
                            torch.from_numpy(tr.positions).to(device),
                            loading_scalar=(
                                spec.loading_scalar(tr.case_id)
                                if getattr(cfg, "impact_velocity_feature", False)
                                and spec.loading_scalar
                                else None
                            ),
                        )
                        sim.reset_rollout()
                        result = rollout(
                            sim,
                            tr,
                            cfg.input_frames,
                            device,
                            kinematic_types=spec.kinematic_types,
                            scored_frames=spec.scored_frames_for(tr.case_id),
                        )
                        pos_losses.append(float(result.position_rmse.mean()))
                        rel_losses.append(float(result.mean_rel_l2_displacement))
                val_pos = (
                    sum(pos_losses) / len(pos_losses) if pos_losses else float("inf")
                )
                # ADR-0077: the selector, the evaluator's headline number.
                val_rel = (
                    sum(rel_losses) / len(rel_losses) if rel_losses else float("inf")
                )
                logger.info(_VAL_LOG, step, loss.item(), val_rel, val_pos, best_rel)
                if val_rel < best_rel:
                    best_rel = val_rel
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
