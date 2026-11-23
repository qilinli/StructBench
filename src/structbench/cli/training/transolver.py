"""The autoregressive Transolver training loop."""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ...benchmarks import (
    BenchmarkSpec,
)
from ...config import (
    TrainConfig,
    TransolverConfig,
    resolved_config_dict,
)
from ...datasets import (
    CaseTrajectory,
    WindowDataset,
    expand_aux_knob,
)
from ...eval import (
    rollout,
)
from ...models.mgn import (
    collate_mesh_samples,
    mesh_static_from_trajectory,
)
from ...models.transolver import (
    TransolverSimulator,
)
from .builders import build_transolver_simulator
from .common import (
    _VAL_LOG,
    PERIODIC_CKPT_EVERY,
    _definition_record,
    _lr_at_cosine,
    _mesh_family_noise,
    _record_training_pool,
    _state_input_noise,
)
from .transolver_fm import _train_transolver_fm
from .transolver_tc import _train_transolver_tc

logger = logging.getLogger(__name__)


def _resolve_transolver_k(
    cfg: TransolverConfig, train_trajs: list[CaseTrajectory]
) -> tuple[TransolverConfig, bool, int]:
    """Resolve the ADR-0050/0051 ``frames_per_call`` (``k``) axis for a run.

    Resolves the one-shot sentinel (``frames_per_call = 0``) to a concrete
    ``k = T_working - input_frames`` from the loaded (already
    ``train_frames``-truncated) TRAIN trajectories, so ``config.json`` and the
    decoder head both record the resolved integer and evaluation rebuilds the
    identical head without a benchmark spec or a re-derivation of the
    data-dependent working length. One-shot requires a uniform trajectory
    length (the head is fixed-size).

    Returns ``(resolved_cfg, is_one_shot, horizon)`` where ``horizon`` is the
    number of frames predictable from the seed (``T_working - input_frames``,
    from the shortest trajectory) and ``is_one_shot`` is ``k >= horizon`` (one
    bundle covers the whole scored span, so there is no seam and training uses
    the clean full-sequence L2 regime).
    """
    lengths = {int(tr.positions.shape[0]) for tr in train_trajs}
    if cfg.frames_per_call == 0 and len(lengths) != 1:
        raise ValueError(
            "frames_per_call = 0 (one-shot / k=T) requires a uniform TRAIN "
            "trajectory length (the decoder head is fixed-size); got lengths "
            f"{sorted(lengths)}. A per-case scored span cuts each case to its "
            "own bound (ADR-0078), so one-shot cannot train on it."
        )
    horizon = min(lengths) - cfg.input_frames
    if cfg.frames_per_call == 0:
        if horizon < 1:
            raise ValueError(
                f"one-shot horizon = {horizon} < 1: TRAIN trajectories have "
                f"{min(lengths)} frames but input_frames = {cfg.input_frames}"
            )
        cfg = replace(cfg, frames_per_call=horizon)
    k = cfg.frames_per_call
    return cfg, (k >= horizon), horizon


def _transolver_pushforward(
    sim: TransolverSimulator,
    position_seq: Tensor,
    next_position: Tensor,
    next_aux: Tensor,
    particle_type: Tensor,
    reference_coords: Tensor,
    n_particles_per_example: Tensor,
    is_kinematic: Tensor,
    k: int,
    history_frames: int,
    loading_feature: Tensor | None = None,
    *,
    warmup: bool,
) -> tuple[Tensor, Tensor]:
    """MP-PDE bundle-seam pushforward for the 1<k<T temporal-bundling regime.

    The sample carries TWO consecutive GT bundles (``next_position`` is
    ``(P, 2k, dim)``): bundle1 ``[t, t+k)`` and bundle2 ``[t+k, t+2k)``. Robustness
    for bundling lives at the SEAM between bundles (no autoregressive feedback
    exists WITHIN a bundle), so — following Brandstetter et al. (MP-PDE, ICLR
    2022) — this runs bundle1 forward WITHOUT gradient, seeds bundle2's input
    window from bundle1's own drifted output, and backpropagates only through
    bundle2. The model thus learns to predict a bundle from a realistically
    drifted (predicted, not ground-truth) seam window.

    Kinematic rows of the drifted seam are clamped back to ground truth (their
    motion is prescribed and GT-known for the whole bundle), so bundle2's
    scripted-velocity input feature stays clean on a moving actuator (ADR-0043
    §4; the C4 review point).

    During normalizer warmup (``warmup=True``) the target normalizer's inverse
    is still untrained, so a decoded seam would be garbage; the step instead
    trains a plain clean-input bundle1 (and accumulates stats) until the
    normalizers are ready, then the pushforward engages.

    Returns ``(pred, target)`` — each ``(P, k, dim+1)`` — for the loss.
    """
    b1_next, b2_next = next_position[:, :k], next_position[:, k:]
    b1_aux, b2_aux = next_aux[:, :k], next_aux[:, k:]

    def _vh(window: Tensor) -> Tensor | None:
        if history_frames == 0:
            return None
        win = window[:, -(history_frames + 1) :]
        return (win[:, 1:] - win[:, :-1]).flatten(1)

    x_last = position_seq[:, -1]
    if warmup:
        # Warm the online normalizers on the CLEAN first bundle; the seam
        # pushforward needs a trained target normalizer to decode positions.
        return sim.forward_train(
            x_last,
            b1_next,
            b1_aux,
            particle_type,
            reference_coords,
            n_particles_per_example,
            accumulate=True,
            velocity_history=_vh(position_seq),
            loading_feature=loading_feature,
        )

    # bundle1: no-grad forward -> decode to positions -> clamp kinematic rows.
    with torch.no_grad():
        pred1_norm, _ = sim.forward_train(
            x_last,
            b1_next,
            b1_aux,
            particle_type,
            reference_coords,
            n_particles_per_example,
            accumulate=False,
            velocity_history=_vh(position_seq),
            loading_feature=loading_feature,
        )
        pred1_pos, _ = sim._decode_positions(pred1_norm, x_last)  # (P, k, dim)
        pred1_pos = pred1_pos.clone()
        pred1_pos[is_kinematic] = b1_next[is_kinematic]

    # seam window: last input_frames of [input window ++ drifted bundle1].
    f = position_seq.shape[1]
    seam_win = torch.cat([position_seq, pred1_pos], dim=1)[:, -f:]

    # bundle2: forward WITH gradient on the drifted seam; the loss target is the
    # clean GT bundle2, so gradient flows only through this pass (pushforward).
    return sim.forward_train(
        seam_win[:, -1],
        b2_next,
        b2_aux,
        particle_type,
        reference_coords,
        n_particles_per_example,
        accumulate=False,
        velocity_history=_vh(seam_win),
        loading_feature=loading_feature,
    )


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
    on the current frame, MGN gamma = 1 target; ``history_frames > 0`` path:
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
    mean displacement relative L2 improves (ADR-0077). Every
    :data:`PERIODIC_CKPT_EVERY` steps it
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

    # ADR-0054: the time-conditioned scheme is a distinct, history-free,
    # non-autoregressive prediction path — a dedicated loop, not the AR/k-frames
    # machinery below. ADR-0062: its anchored-flow-map mode (pair sampling +
    # Δt conditioning) gets its own loop in turn.
    if cfg.time_conditioned:
        if cfg.flow_map:
            return _train_transolver_fm(
                spec, cfg, train_cfg, train_trajs, val_trajs, out_dir, device, data_root
            )
        return _train_transolver_tc(
            spec, cfg, train_cfg, train_trajs, val_trajs, out_dir, device, data_root
        )

    # ADR-0050/0051 prediction-scheme axis. Resolve the k=T sentinel and record
    # the concrete k in cfg (so build_transolver_simulator and
    # resolved_config_dict below both see it). k=1 is the autoregressive scheme;
    # k=1 is autoregressive; k covering the whole horizon is one-shot (k=T,
    # clean full-sequence L2); 1<k<T is temporal bundling, trained with the
    # MP-PDE bundle-seam pushforward (two forward passes per step, below).
    cfg, is_one_shot, horizon = _resolve_transolver_k(cfg, train_trajs)
    k = cfg.frames_per_call
    # ADR-0061 knob 1: expanded once; None when the state input is off.
    aux_noise = (
        expand_aux_knob(cfg.aux_input_noise_std, int(train_trajs[0].aux.shape[-1]))
        if cfg.aux_input
        else None
    )
    is_pushforward = k > 1 and not is_one_shot
    # The pushforward needs TWO consecutive GT bundles per sample (bundle1 to
    # drift the seam, bundle2 for the loss), so its target span is 2k; the
    # single-forward regimes (k=1, one-shot) use a k-frame span.
    target_frames = 2 * k if is_pushforward else k
    # ADR-0061: the state-channel pushforward chain needs TWO consecutive
    # single-frame targets (step A teacher-forced, step B fed the detached
    # prediction); aux_input enforces k=1, so this never collides with the
    # ADR-0051 bundle pushforward above.
    if cfg.aux_input_pushforward:
        target_frames = 2
    # ADR-0050/0051: injected single-step noise is a k=1-only robustness
    # mechanism. At k>1 it is replaced by the pushforward (1<k<T) or dropped
    # (one-shot clean L2), so a nonzero noise_std is INERT — warn rather than
    # let it look active in a config cloned from a k=1 recipe.
    if k > 1 and cfg.noise_std:
        scheme = (
            "one-shot clean full-sequence L2"
            if is_one_shot
            else "MP-PDE bundle-seam pushforward"
        )
        logger.warning(
            "noise_std=%.4g is IGNORED at frames_per_call=%d: rollout "
            "robustness at k>1 comes from the %s, not injected single-step "
            "noise (ADR-0050/0051). Set noise_std=0 to silence this.",
            cfg.noise_std,
            k,
            scheme,
        )

    statics = [mesh_static_from_trajectory(tr) for tr in train_trajs]
    # ADR-0051 B: per-trajectory scalar loading parameter (impact velocity),
    # resolved from the benchmark spec; None (and no extra channel) when the
    # feature is off.
    loading_scalars: list[float] | None = None
    if cfg.impact_velocity_feature:
        if spec.loading_scalar is None:
            raise ValueError(
                f"benchmark {spec.card.name!r} has no loading_scalar (scalar "
                "impact-velocity parameter), but the transolver config sets "
                "impact_velocity_feature=True (ADR-0051 B)."
            )
        loading_scalars = [spec.loading_scalar(tr.case_id) for tr in train_trajs]
    sim = build_transolver_simulator(
        cfg,
        n_aux=int(train_trajs[0].aux.shape[-1]),  # ADR-0059
        kinematic_types=spec.kinematic_types,
        scripted_types=spec.scripted_types,
        hardening_curve=spec.hardening_curve,
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
                definition=_definition_record(spec),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = WindowDataset(train_trajs, cfg.input_frames, target_frames=target_frames)
    if len(dataset) == 0:
        raise ValueError(
            f"empty training set: no TRAIN trajectory has "
            f"input_frames + target_frames = {cfg.input_frames + target_frames} "
            f"or more frames, so there are no frames_per_call={k} samples "
            f"({'pushforward needs 2k target frames; ' if is_pushforward else ''}"
            f"check the data root, or reduce input_frames / frames_per_call)."
        )
    _record_training_pool(
        out_dir,
        spec,
        train_cfg,
        train_trajs,
        lambda trajs: WindowDataset(
            trajs, cfg.input_frames, target_frames=target_frames
        ),
        len(dataset),
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=functools.partial(
            collate_mesh_samples, statics=statics, loading_scalars=loading_scalars
        ),
    )
    optimizer = torch.optim.AdamW(
        sim.parameters(), lr=train_cfg.lr_init, weight_decay=cfg.weight_decay
    )

    regime = (
        " (autoregressive)"
        if k == 1
        else " (one-shot)"
        if is_one_shot
        else " (pushforward bundling)"
    )
    logger.info(
        "starting transolver training: %d steps, batch %d, frames_per_call %d%s",
        train_cfg.training_steps,
        train_cfg.batch_size,
        k,
        regime,
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
            # ADR-0051 B: (sum_P, 1) global loading-param channel, present only
            # when impact_velocity_feature is on (else None).
            loading_feature = batch.get("loading_feature")
            if loading_feature is not None:
                loading_feature = loading_feature.to(device)

            is_kinematic = torch.isin(particle_type, kinematic)
            accumulate = step < cfg.normalizer_warmup_steps
            optimizer.zero_grad()
            # ADR-0051 noise/target regime by frames_per_call:
            #   1<k<T -> MP-PDE bundle-seam pushforward (two forward passes);
            #   k=1   -> reference single-step / velocity-history noise;
            #   k=T   -> one-shot, noise off, clean full-sequence L2.
            if is_pushforward:
                pred, target = _transolver_pushforward(
                    sim,
                    position_seq,
                    next_position,
                    next_aux,
                    particle_type,
                    reference_coords,
                    n_particles_per_example,
                    is_kinematic,
                    k,
                    cfg.history_frames,
                    loading_feature,
                    warmup=accumulate,
                )
            elif cfg.aux_input_pushforward:
                # ADR-0061 knob 2: two-step chain on the STATE channel only.
                # Positions stay teacher-forced throughout (position exposure
                # bias is the recipe's position-noise job); only the state
                # input of step B carries the model's own (detached) one-step
                # error. Loss = mean over both steps (the (P, 2, dim+C)
                # stacking below feeds the rank-agnostic loss unchanged).
                input_aux_a = _state_input_noise(
                    batch["input_aux"].to(device), aux_noise, is_kinematic
                )
                x_a, vh_a, next_a = _mesh_family_noise(
                    position_seq,
                    next_position[:, 0],
                    is_kinematic,
                    cfg.noise_std,
                    cfg.history_frames,
                )
                pred_a, target_a = sim.forward_train(
                    x_a,
                    next_a,
                    next_aux[:, 0],
                    particle_type,
                    reference_coords,
                    n_particles_per_example,
                    accumulate=accumulate,
                    velocity_history=vh_a,
                    loading_feature=loading_feature,
                    input_aux=input_aux_a,
                )
                # Step B consumes step A's predicted state, DETACHED (the
                # pushforward trick). KINEMATIC rows are clamped to the GT
                # state at t — the training-side analog of the rollout
                # cache clamp (those rows carry no aux training signal).
                state_pred = sim.train_output_aux(pred_a).detach()
                if is_kinematic.any():
                    state_pred = state_pred.clone()
                    state_pred[is_kinematic] = next_aux[:, 0][is_kinematic]
                seq_b = torch.cat([position_seq[:, 1:], next_position[:, :1]], dim=1)
                x_b, vh_b, next_b = _mesh_family_noise(
                    seq_b,
                    next_position[:, 1],
                    is_kinematic,
                    cfg.noise_std,
                    cfg.history_frames,
                )
                pred_b, target_b = sim.forward_train(
                    x_b,
                    next_b,
                    next_aux[:, 1],
                    particle_type,
                    reference_coords,
                    n_particles_per_example,
                    # normalizers warm on the clean teacher-forced step only
                    accumulate=False,
                    velocity_history=vh_b,
                    loading_feature=loading_feature,
                    input_aux=state_pred,
                )
                pred = torch.stack([pred_a, pred_b], dim=1)  # (P, 2, dim+C)
                target = torch.stack([target_a, target_b], dim=1)
            else:
                x_noisy, velocity_history, next_target = _mesh_family_noise(
                    position_seq,
                    next_position,
                    is_kinematic,
                    cfg.noise_std,
                    cfg.history_frames,
                    noise_off=(k > 1),
                )
                pred, target = sim.forward_train(
                    x_noisy,
                    next_target,
                    next_aux,
                    particle_type,
                    reference_coords,
                    n_particles_per_example,
                    accumulate=accumulate,
                    velocity_history=velocity_history,
                    loading_feature=loading_feature,
                    # ADR-0060: teacher-forced state input (GT aux at the
                    # last input frame), ADR-0061-noised when the knob is on;
                    # None keeps the pre-0060 path.
                    input_aux=(
                        _state_input_noise(
                            batch["input_aux"].to(device), aux_noise, is_kinematic
                        )
                        if cfg.aux_input
                        else None
                    ),
                )
            # Rank-agnostic: (P, dim+1) at k=1 (byte-identical), (P, k, dim+1)
            # at k>1; the kinematic row mask selects on dim 0 either way and
            # .mean() averages uniformly over free rows (and k frames).
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
                            gt_aux=(
                                torch.from_numpy(tr.aux).to(device)
                                if getattr(cfg, "aux_input", False)
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
