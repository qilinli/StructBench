"""The MeshGraphNet training loop."""

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
    MGNConfig,
    TrainConfig,
    resolved_config_dict,
)
from ...datasets import (
    CaseTrajectory,
    WindowDataset,
)
from ...eval import (
    rollout,
)
from ...models.mgn import (
    collate_mesh_samples,
    mesh_static_from_trajectory,
)
from .builders import build_mgn_simulator
from .common import (
    _VAL_LOG,
    PERIODIC_CKPT_EVERY,
    _definition_record,
    _lr_at,
    _mesh_family_noise,
    _record_training_pool,
)

logger = logging.getLogger(__name__)


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
    on the current frame, MGN gamma = 1 target; ``history_frames > 0`` path:
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
    saved as ``model-best-<step>.pt`` when the mean displacement relative L2
    improves (ADR-0077).
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
                "mgn",
                mgn,
                train_cfg,
                horizon=spec.card.horizon,
                eval_times=spec.card.eval_times,
                n_particle_types=mgn.node_type_size,
                data_root=data_root,
                definition=_definition_record(spec),
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
    _record_training_pool(
        out_dir,
        spec,
        train_cfg,
        train_trajs,
        lambda trajs: WindowDataset(trajs, mgn.input_frames),
        len(dataset),
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
    best_rel = float("inf")
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
                mgn.history_frames,
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
            optimizer.step()

            lr_new = _lr_at(step, train_cfg)
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
                        )
                        sim.reset_rollout()
                        result = rollout(
                            sim,
                            tr,
                            mgn.input_frames,
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
