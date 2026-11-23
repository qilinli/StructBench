"""The time-conditioned Transolver training loop."""

from __future__ import annotations

import functools
import json
import logging
import os
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
)
from ...eval import (
    time_conditioned_rollout,
)
from ...models.mgn import (
    collate_mesh_samples,
    mesh_static_from_trajectory,
)
from .builders import build_transolver_simulator
from .common import (
    _VAL_LOG,
    PERIODIC_CKPT_EVERY,
    _definition_record,
    _lr_at_cosine,
    _record_training_pool,
    _training_time_ref,
)

logger = logging.getLogger(__name__)


def _lsq_gradient_edges(
    coords: Tensor, k: int, eps: float = 1e-6
) -> tuple[Tensor, Tensor]:
    """Fixed least-squares displacement-gradient operator over rest-config kNN.

    Builds, from static reference coordinates, a linear operator mapping a
    displacement field ``u`` (P, dim) to its per-node gradient ``G`` (P, dim,
    dim) with ``G[i, a, b] = du_a/dx_b``, by unweighted least squares over each
    node's ``k`` nearest rest neighbours — the operator verified exact for a
    locally linear field in the strain-from-gradient study
    (``G^T = (dX^T dX)^-1 dX^T dU``). It depends only on ``coords`` (constant in
    ``u``), so it is built under ``no_grad`` and applied as a differentiable
    scatter in :func:`_apply_lsq_gradient`.

    Returns ``(edge_index, coeff)``: ``edge_index`` (2, E) with row 0 the centre
    node ``i`` and row 1 its neighbour ``j``; ``coeff`` (E, dim) with
    ``G[i, :, b] += coeff[e, b] * (u[j] - u[i])``.
    """
    n, dim = coords.shape
    kk = min(k, n - 1)
    with torch.no_grad():
        d = torch.cdist(coords, coords)
        d.fill_diagonal_(float("inf"))
        nbr = torch.topk(d, kk, largest=False).indices  # (n, kk)
        src = torch.arange(n, device=coords.device).repeat_interleave(kk)
        dst = nbr.reshape(-1)
        dx = (coords[dst] - coords[src]).view(n, kk, dim)  # (n, kk, dim)
        a = dx.transpose(1, 2) @ dx  # (n, dim, dim)
        a = a + eps * torch.eye(dim, device=coords.device, dtype=coords.dtype)
        # M = A^-1 dX^T : (n, dim, kk); coeff for edge (i, j_kk) is M[i, :, kk].
        m = torch.linalg.solve(a, dx.transpose(1, 2))  # (n, dim, kk)
        coeff = m.transpose(1, 2).reshape(n * kk, dim)  # (n*kk, dim)
        edge_index = torch.stack([src, dst], dim=0)
    return edge_index, coeff


def _batch_lsq_gradient_edges(
    reference_coords: Tensor, n_per_example: Tensor, k: int
) -> tuple[Tensor, Tensor]:
    """Per-example :func:`_lsq_gradient_edges`, node-offset and concatenated.

    Builds the operator independently within each batched case (no edge ever
    crosses a case's particle-row range), then offsets each case's edges by the
    cumulative particle count so the result applies to the collated
    ``(sum_P, dim)`` field.
    """
    edges: list[Tensor] = []
    coeffs: list[Tensor] = []
    offset = 0
    for n in n_per_example.tolist():
        ei, cf = _lsq_gradient_edges(reference_coords[offset : offset + n], k)
        edges.append(ei + offset)
        coeffs.append(cf)
        offset += n
    return torch.cat(edges, dim=1), torch.cat(coeffs, dim=0)


def _apply_lsq_gradient(edge_index: Tensor, coeff: Tensor, u: Tensor) -> Tensor:
    """Apply the :func:`_lsq_gradient_edges` operator to a field ``u`` (P, dim).

    Returns the per-node gradient ``G`` (P, dim, dim), differentiable in ``u``.
    """
    src, dst = edge_index[0], edge_index[1]
    du = u[dst] - u[src]  # (E, dim_a)
    contrib = coeff.unsqueeze(1) * du.unsqueeze(2)  # (E, dim_a, dim_b)
    grad = u.new_zeros(u.shape[0], u.shape[1], u.shape[1])
    grad.index_add_(0, src, contrib)
    return grad


def _train_transolver_tc(
    spec: BenchmarkSpec,
    cfg: TransolverConfig,
    train_cfg: TrainConfig,
    train_trajs: list[CaseTrajectory],
    val_trajs: list[CaseTrajectory],
    out_dir: Path,
    device: str,
    data_root: Path,
) -> Path | None:
    """Time-conditioned Transolver training (ADR-0054).

    The faithful thuml structural (Plasticity) scheme: history-free and
    non-autoregressive. Each sample is a single scored frame ``t``; the model
    is trained to map ``(static geometry, node types, scalar impact velocity?,
    prescribed boundary state at t, normalized query time t) -> absolute state
    at t`` (regressed as rest-frame displacement + aux). There is no rollout
    window, no injected noise (inert without autoregression, ADR-0054 decision
    3), and no k-frames bundling. The optimizer recipe (AdamW + cosine LR +
    grad-norm clip) and the online-normalizer warmup mirror
    :func:`_train_transolver`; validation rolls out the independent-query
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

    # H1 gradient-consistency (Sobolev) regulariser — SCREEN-ONLY, env-gated.
    # STRUCTBENCH_H1_RATIO (default 0 = OFF = byte-identical to the reference TC
    # loop) is the TARGET ratio of the H1 term to the position term. The raw
    # ``||G(pred_u) - G(gt_u)||^2`` scale is deeply mesh-dependent (gradient of
    # normalized displacement over the reference spacing), so a fixed weight is
    # unusable across benchmarks; the effective weight is instead AUTO-CALIBRATED
    # once, at the first post-warmup step, so ratio=0.1 means "H1 term = 10% of
    # the position term". STRUCTBENCH_H1_NEIGHBORS sets the rest-config kNN count.
    # Kept out of the config schema on purpose: promote to a TransolverConfig
    # knob only if the screen earns it.
    h1_ratio = float(os.environ.get("STRUCTBENCH_H1_RATIO", "0.0"))
    h1_neighbors = int(os.environ.get("STRUCTBENCH_H1_NEIGHBORS", "12"))
    h1_eff_weight: float | None = None
    if h1_ratio > 0.0:
        logger.info(
            "H1 gradient-consistency regulariser ACTIVE (screen): target "
            "ratio=%.3g of the position term, k=%d rest neighbours "
            "(effective weight auto-calibrated after normalizer warmup)",
            h1_ratio,
            h1_neighbors,
        )

    loading_scalars: list[float] | None = None
    if cfg.impact_velocity_feature:
        if spec.loading_scalar is None:
            raise ValueError(
                f"benchmark {spec.card.name!r} has no loading_scalar (scalar "
                "impact-velocity parameter), but the transolver config sets "
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
        "starting transolver training: %d steps, batch %d, time-conditioned "
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
            # Gradient-consistency (Sobolev/H^1) regulariser on the normalized
            # displacement: match grad(pred_u) to grad(gt_u) via the fixed
            # rest-config least-squares operator. Applied only after normalizer
            # warmup (so the auto-calibration sees settled loss scales); kinematic
            # rows are clamped to GT so a free node's gradient is not corrupted by
            # their untrained displacement predictions.
            if h1_ratio > 0.0 and not accumulate:
                with torch.no_grad():
                    h1_ei, h1_cf = _batch_lsq_gradient_edges(
                        reference_coords, n_particles_per_example, h1_neighbors
                    )
                pred_u = torch.where(
                    is_kinematic.unsqueeze(1),
                    target[..., :-aux_c],
                    pred[..., :-aux_c],
                )
                g_pred = _apply_lsq_gradient(h1_ei, h1_cf, pred_u)
                g_gt = _apply_lsq_gradient(h1_ei, h1_cf, target[..., :-aux_c])
                h1_per_particle = ((g_pred - g_gt) ** 2).sum(dim=(-1, -2))
                if h1_eff_weight is None and free.any():
                    with torch.no_grad():
                        wp = float(
                            (train_cfg.w_pos * (delta_v**2).sum(dim=-1))[free]
                            .mean()
                            .item()
                        )
                        wh = float(h1_per_particle[free].mean().item())
                    h1_eff_weight = h1_ratio * wp / max(wh, 1e-12)
                    logger.info(
                        "H1 calibrated @step%d: target_ratio=%.3g w_pos_term=%.4g "
                        "raw_h1_term=%.4g -> effective w_h1=%.4g",
                        step,
                        h1_ratio,
                        wp,
                        wh,
                        h1_eff_weight,
                    )
                if h1_eff_weight is not None:
                    per_particle = per_particle + h1_eff_weight * h1_per_particle
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
