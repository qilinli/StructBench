"""The flow-map Transolver training loop."""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ...benchmarks import (
    BenchmarkSpec,
    CombinedHardeningLaw,
    ConstitutiveLaw,
)
from ...config import (
    TrainConfig,
    TransolverConfig,
    resolved_config_dict,
)
from ...datasets import (
    CaseTrajectory,
    FlowMapChainDataset,
    FlowMapPairDataset,
    expand_aux_knob,
)
from ...eval import (
    flow_map_rollout,
)
from ...models.common.two_site import (
    LAW_DECODERS,
    MONOTONE_DECODERS,
    LawRows,
    law_rows,
)
from ...models.mgn import (
    collate_mesh_samples,
    mesh_static_from_trajectory,
)
from ...models.mgn.collate import LawTuple
from ...models.transolver import (
    FlowMapAnchor,
    TransolverSimulator,
    anchor_from_predictions,
    hardening_sigma_y,
    plane_strain_vm,
)
from .builders import build_transolver_simulator
from .common import (
    _VAL_LOG,
    PERIODIC_CKPT_EVERY,
    TrainingDiverged,
    _case_law_rows,
    _definition_record,
    _domain_scale,
    _early_divergence,
    _fm_intervals_and_canonical,
    _lr_at_cosine,
    _record_training_pool,
    _state_input_noise,
    _training_time_ref,
)

logger = logging.getLogger(__name__)


def _law_tuple(law: ConstitutiveLaw | CombinedHardeningLaw) -> LawTuple:
    """A benchmark's :class:`ConstitutiveLaw` as the collate's plain tuple."""
    if not isinstance(law, ConstitutiveLaw):
        raise ValueError(
            "the law-reading two-site decoders read an isotropic hardening "
            f"table; the case's law is {type(law).__name__}, which none of them "
            "reads (ADR-0075)"
        )
    return (law.youngs_modulus, law.poisson_ratio, law.hardening[0], law.hardening[1])


def _anchor_kinematic_noise(
    anchor_pair: Tensor,
    noise_pos: float,
    noise_vel: float,
    is_kinematic: Tensor,
) -> Tensor:
    """ADR-0063 knob 2: structured Gaussian noise on the anchor position pair.

    The component structure is measurement-dictated (anchor-pair error
    correlation 0.991, `scratch/stage3/anchor_subchannel.json`): a
    COMMON-MODE draw (``noise_pos``, mm) added to BOTH pair frames corrupts
    the anchor displacement while cancelling exactly in the FD velocity,
    and a DIFFERENTIAL draw (``noise_vel``, mm/frame) added to the ``t0-1``
    frame only corrupts the FD velocity while leaving the anchor position
    untouched. Independent per-frame noise would mis-rehearse the
    interface ~30x (position-matched noise inflates the velocity error).
    Kinematic rows stay clean (their fed kinematics are prescribed/GT at
    rollout); targets stay clean GT. Zero scales return the input
    unchanged (byte-identical off path).

    ``anchor_pair`` is ``(P, 2, dim)`` — frames ``(t0-1, t0)``.
    """
    if noise_pos <= 0.0 and noise_vel <= 0.0:
        return anchor_pair
    noised = anchor_pair.clone()
    keep = is_kinematic.unsqueeze(-1)
    if noise_pos > 0.0:
        common = noise_pos * torch.randn_like(anchor_pair[:, 0])
        common = common.masked_fill(keep, 0.0)
        noised = noised + common.unsqueeze(1)  # same draw on both frames
    if noise_vel > 0.0:
        diff = noise_vel * torch.randn_like(anchor_pair[:, 0])
        diff = diff.masked_fill(keep, 0.0)
        noised[:, 0] = noised[:, 0] + diff  # t0-1 frame only -> FD velocity
    return noised


def _fm_admissibility_hinge(
    aux_raw: Tensor,
    anchor_peeq: Tensor,
    knots: tuple[Tensor, Tensor],
    peeq_scale: Tensor,
) -> Tensor:
    """ADR-0064 knob 2: per-particle admissibility hinge for ONE fm query.

    ``relu(vm_comp - sigma_y(peeq_pred)) / sigma_y0`` (yield admissibility)
    plus ``relu(anchor_peeq - peeq_pred) / peeq_scale`` (anchor-relative
    irreversibility), both dimensionless. ``aux_raw`` is the query's RAW
    working-unit aux block (gradients intact); ``anchor_peeq`` the FED
    anchor's peeq channel — the same reference the structured heads build
    on (ADR-0064); ``peeq_scale`` a detached batch-GT scale.
    """
    dev = aux_raw[..., :3]
    peeq = aux_raw[..., 3]
    vm = plane_strain_vm(dev)
    sy = hardening_sigma_y(peeq, knots[0], knots[1])
    return (
        torch.relu(vm - sy) / knots[1][0] + torch.relu(anchor_peeq - peeq) / peeq_scale
    )


def _fm_two_site_chain(
    sim: TransolverSimulator,
    next_position: Tensor,
    next_aux: Tensor,
    next_state: Tensor,
    particle_type: Tensor,
    reference_coords: Tensor,
    n_particles_per_example: Tensor,
    anchor_frame: Tensor,
    chain_frame: Tensor,
    target_frame: Tensor,
    time_ref: int,
    *,
    accumulate: bool,
    cells: Tensor,
    weights: Tensor,
    law: LawRows | None,
    anchor_disp: Tensor,
    anchor_vel: Tensor,
    anchor_aux: Tensor,
    anchor_state: Tensor,
    is_kinematic: Tensor,
    loading_feature: Tensor | None,
    anchor_time_feature: Tensor | None,
    anchor_time: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, FlowMapAnchor]:
    """One ADR-0063 two-hop chain with the element state carried (ADR-0075
    note 2026-11-11): three queries, two clean and one dirty.

    Hop A queries frames ``t1 - 1`` and ``t1`` from the stored anchor at
    ``t0`` (the two frames a re-anchor at ``t1`` consumes), in time order:
    the second is floored by the plastic strain the first emitted, as a
    rollout floors each frame by the one before it. The mid-chain anchor is
    then built from hop A's own emitted state by
    :func:`~structbench.models.transolver.anchor_from_predictions`, the
    function the rollout's hand-off calls, and is detached: no gradient
    reaches hop A through it. Hop B queries ``t2`` from that anchor.

    Every query carries the nodal and the element loss; the caller's
    equal-weight mean over the stacked queries is ADR-0063's 2:1
    clean-to-dirty proportion. Normalizers accumulate on the clean queries
    only.

    Parameters
    ----------
    next_position, next_aux, next_state:
        ``(P, 3, dim)``, ``(P, 3, C)`` and ``(E, 3, S)``: the stored state at
        the chain's three query frames ``(t1 - 1, t1, t2)``.
    anchor_frame, chain_frame, target_frame:
        ``(B,)`` the frames ``t0``, ``t1``, ``t2``.
    anchor_disp, anchor_vel, anchor_aux, anchor_state:
        The stored anchor at ``t0`` as the trainer prepared it (kinematics
        and nodal aux possibly noised; the carried element state clean).
    anchor_time:
        Whether the simulator takes the anchor-time feature.

    Returns
    -------
    tuple
        ``(pred (P, 3, dim + S), target (P, 3, dim + S), state_pred
        (E, 3, S), state_target (E, 3, S), anchor_b)``, the predictions and
        targets in their normalizers' spaces and the mid-chain anchor.
    """
    scale = float(time_ref - 1)
    preds, targets, state_preds, state_targets = [], [], [], []
    emitted = []
    floor: Tensor | None = None
    for j in (0, 1):
        dt_a = (chain_frame - 1 + j - anchor_frame).to(torch.float32) / scale
        p_j, t_j, sp_j, st_j, out_j = sim.forward_train_two_site_emitted(
            next_position[:, j],
            next_aux[:, j],
            next_state[:, j],
            particle_type,
            reference_coords,
            n_particles_per_example,
            dt_a,
            accumulate=accumulate,
            anchor_disp=anchor_disp,
            anchor_vel=anchor_vel,
            anchor_aux=anchor_aux,
            anchor_state=anchor_state,
            anchor_time_feature=anchor_time_feature,
            peeq_floor=floor,
            cells=cells,
            weights=weights,
            law=law,
            loading_feature=loading_feature,
        )
        preds.append(p_j)
        targets.append(t_j)
        state_preds.append(sp_j)
        state_targets.append(st_j)
        emitted.append(out_j)
        if sim.two_site in MONOTONE_DECODERS:
            floor = out_j[2][:, 4].detach()
    (pos_a1, _aux_a1, _state_a1), (pos_a2, aux_a2, state_a2) = emitted
    anchor_b = anchor_from_predictions(
        reference_coords,
        pos_a1,
        pos_a2,
        aux_a2,
        kin_mask=is_kinematic if bool(is_kinematic.any()) else None,
        gt_aux=next_aux[:, 1],
        state=state_a2,
    )
    assert anchor_b.state is not None
    anchor_time_b = None
    if anchor_time:
        anchor_time_b = torch.repeat_interleave(
            chain_frame.to(torch.float32) / scale, n_particles_per_example
        ).unsqueeze(1)
    dt_b = (target_frame - chain_frame).to(torch.float32) / scale
    p_b, t_b, sp_b, st_b = sim.forward_train_two_site(
        next_position[:, 2],
        next_aux[:, 2],
        next_state[:, 2],
        particle_type,
        reference_coords,
        n_particles_per_example,
        dt_b,
        accumulate=False,
        anchor_disp=anchor_b.disp,
        anchor_vel=anchor_b.vel,
        anchor_aux=anchor_b.aux,
        anchor_state=anchor_b.state,
        anchor_time_feature=anchor_time_b,
        cells=cells,
        weights=weights,
        law=law,
        loading_feature=loading_feature,
    )
    return (
        torch.stack([*preds, p_b], dim=1),
        torch.stack([*targets, t_b], dim=1),
        torch.stack([*state_preds, sp_b], dim=1),
        torch.stack([*state_targets, st_b], dim=1),
        anchor_b,
    )


def _train_transolver_fm(
    spec: BenchmarkSpec,
    cfg: TransolverConfig,
    train_cfg: TrainConfig,
    train_trajs: list[CaseTrajectory],
    val_trajs: list[CaseTrajectory],
    out_dir: Path,
    device: str,
    data_root: Path,
) -> Path | None:
    """Anchored flow-map Transolver training (ADR-0062; ADR-0063 knobs).

    The TC formulation with a movable anchor: each sample is one uniformly
    drawn ``(anchor t0, query t)`` pair (:class:`FlowMapPairDataset`), and the
    model maps ``(static geometry, node types, scalar impact velocity?,
    prescribed boundary state at t, anchor state + kinematics at t0, anchor
    time?, offset Δt) -> absolute state at t``. The anchor aux receives the
    ADR-0061 batch-std-relative noise when ``aux_input_noise_std`` is on
    (kinematic rows clean, clean targets); the anchor KINEMATICS receive
    the ADR-0063 structured noise when ``flow_map_anchor_noise_{pos,vel}``
    are on. With ``flow_map_pushforward`` (ADR-0063 knob 1) each sample is
    a two-hop chain (:class:`FlowMapChainDataset`): step A teacher-forces
    the GT anchor at ``t0`` and predicts the pair ``(t1-1, t1)``, the
    anchor at ``t1`` is rebuilt from those predictions DETACHED (house
    kinematic clamps), and step B predicts ``t2`` from it — loss = mean of
    the three query losses; normalizers accumulate on the clean step-A
    queries only. Targets, loss shape, optimizer recipe, and normalizer
    warmup mirror :func:`_train_transolver_tc`; validation runs the
    SELF-ANCHORED re-anchoring rollout at the canonical interval (the
    deployable mode, ADR-0062) and selects on its mean displacement relative
    L2 (ADR-0077).
    """
    if cfg.noise_std:
        logger.warning(
            "noise_std=%.4g is IGNORED for flow_map=true: the flow map has no "
            "autoregressive position feedback, so injected single-step noise "
            "is inert (ADR-0054/0062). Set noise_std=0 to silence.",
            cfg.noise_std,
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
            f"flow-map time_ref_frames={time_ref} < 2 (scored_frames="
            f"{spec.scored_frames_fixed}, train_frames={train_cfg.train_frames}, "
            f"min train length={min_len})"
        )

    statics = [mesh_static_from_trajectory(tr) for tr in train_trajs]
    # ADR-0075: the two-site decoder carries the element-site state block
    # (loaded by _state_fields_for) and, for a law-reading decoder, each
    # case's law.
    two_site = cfg.two_site_decoder != "off"
    laws: list[LawTuple] | None = None
    if two_site:
        missing = [tr.case_id for tr in train_trajs + val_trajs if tr.state is None]
        if missing:
            raise ValueError(
                f"two_site_decoder={cfg.two_site_decoder!r} needs the element-site "
                f"state block on every trajectory; {len(missing)} carry none "
                f"(first: {missing[0]!r}); does benchmark {spec.card.name!r} "
                "declare state_site='element'? (ADR-0075)"
            )
        if cfg.two_site_decoder in LAW_DECODERS:
            if spec.constitutive_law is None:
                raise ValueError(
                    f"two_site_decoder={cfg.two_site_decoder!r} needs the "
                    f"benchmark's constitutive_law; benchmark {spec.card.name!r} "
                    "has none (ADR-0075)"
                )
            laws = [_law_tuple(spec.constitutive_law(tr.case_id)) for tr in train_trajs]
            if len({len(law[2]) for law in laws}) != 1:
                raise ValueError(
                    "every case's constitutive_law must carry the same number of "
                    "hardening knots for batched training (ADR-0075)"
                )
    sim = build_transolver_simulator(
        cfg,
        n_aux=int(train_trajs[0].aux.shape[-1]),  # ADR-0059
        kinematic_types=spec.kinematic_types,
        scripted_types=spec.scripted_types,
        hardening_curve=spec.hardening_curve,
        element_formulation=spec.element_formulation,
        device=device,
    )
    sim.to(device)

    kinematic = torch.as_tensor(
        list(spec.kinematic_types), dtype=torch.long, device=device
    )
    # ADR-0061 knob 1 on the anchor aux (flow_map requires aux_input).
    aux_noise = expand_aux_knob(
        cfg.aux_input_noise_std, int(train_trajs[0].aux.shape[-1])
    )
    intervals, canonical = _fm_intervals_and_canonical(cfg)
    # ADR-0064 knob 2: the soft comparator needs the benchmark's hardening
    # table (the structured heads get theirs through the simulator ctor).
    hinge_w = cfg.flow_map_consistency_hinge
    hinge_knots: tuple[torch.Tensor, torch.Tensor] | None = None
    if hinge_w > 0:
        if spec.hardening_curve is None:
            raise ValueError(
                f"benchmark {spec.card.name!r} has no hardening_curve, but "
                "the transolver config sets flow_map_consistency_hinge="
                f"{hinge_w} (ADR-0064)"
            )
        hinge_knots = (
            torch.tensor(spec.hardening_curve[0], dtype=torch.float32, device=device),
            torch.tensor(spec.hardening_curve[1], dtype=torch.float32, device=device),
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

    chain = cfg.flow_map_pushforward

    def _fm_dataset(
        trajs: list[CaseTrajectory],
    ) -> FlowMapChainDataset | FlowMapPairDataset:
        if chain:
            # ADR-0063 knob 1: two-hop chains. Config load guarantees
            # flow_map_max_dt >= 2 with pushforward (uncapped chains would
            # enumerate O(T^3) triples).
            return FlowMapChainDataset(
                trajs,
                cfg.input_frames,
                max_dt=cfg.flow_map_max_dt,
                generations=cfg.flow_map_pushforward_generations,
            )
        return FlowMapPairDataset(trajs, cfg.input_frames, max_dt=cfg.flow_map_max_dt)

    dataset = _fm_dataset(train_trajs)
    if len(dataset) == 0:
        raise ValueError(
            "empty training set: no TRAIN trajectory yields a valid "
            f"{'anchor chain' if chain else '(anchor, query) pair'} "
            f"with input_frames = {cfg.input_frames}, flow_map_max_dt = "
            f"{cfg.flow_map_max_dt}, generations = "
            f"{cfg.flow_map_pushforward_generations} (a G-deep chain needs "
            f"trajectories longer than input_frames + 2G + 1; ADR-0062/0063)."
        )
    _record_training_pool(
        out_dir,
        spec,
        train_cfg,
        train_trajs,
        _fm_dataset,
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
            include_anchor_frame=True,
            include_chain_frame=chain,
            include_elements=two_site,
            laws=laws,
        ),
    )
    optimizer = torch.optim.AdamW(
        sim.parameters(), lr=train_cfg.lr_init, weight_decay=cfg.weight_decay
    )

    logger.info(
        "starting transolver training: %d steps, batch %d, anchored flow map "
        "(ADR-0062%s), time_ref_frames=%d, %d %s, max_dt=%d, "
        "eval intervals %s (canonical m=%d)%s",
        train_cfg.training_steps,
        train_cfg.batch_size,
        "; ADR-0063 pushforward chains" if chain else "",
        time_ref,
        len(dataset),
        "(t0, t1, t2) chains" if chain else "(t0, t) pairs",
        cfg.flow_map_max_dt,
        intervals,
        canonical,
        " (impact-velocity conditioned)" if cfg.impact_velocity_feature else "",
    )

    step = 0
    best_rel = float("inf")
    best_ckpt: Path | None = None
    sim.train()
    while step < train_cfg.training_steps:
        for batch in loader:
            particle_type = batch["particle_type"].to(device)
            # (P, dim)/(P, C) GT at the query t — or chain targets under
            # ADR-0063 pushforward: (P, 3, dim) at G=1 (t1-1, t1, t2);
            # (P, 2G+1, dim) under the generation curriculum
            # (t1-1, t1, ..., tG-1, tG, t_final).
            next_position = batch["next_position"].to(device)
            next_aux = batch["next_aux"].to(device)
            reference_coords = batch["reference_coords"].to(device)
            n_particles_per_example = batch["n_particles_per_example"].to(device)
            target_frame = batch["target_frame"].to(device)  # (B,)
            anchor_frame = batch["anchor_frame"].to(device)  # (B,)
            # ADR-0062 anchor pair: position_seq is (P, 2, dim) = frames
            # (t0-1, t0) — the FD-velocity partner and the anchor frame.
            anchor_pair = batch["position_seq"].to(device)
            loading_feature = batch.get("loading_feature")
            if loading_feature is not None:
                loading_feature = loading_feature.to(device)

            is_kinematic = torch.isin(particle_type, kinematic)
            # ADR-0063 knob 2: structured noise on the GT anchor kinematics
            # (common-mode pos + differential vel; kinematic rows clean).
            anchor_pair = _anchor_kinematic_noise(
                anchor_pair,
                cfg.flow_map_anchor_noise_pos,
                cfg.flow_map_anchor_noise_vel,
                is_kinematic,
            )
            anchor_pos = anchor_pair[:, 1]
            anchor_disp = anchor_pos - reference_coords
            anchor_vel = anchor_pos - anchor_pair[:, 0]
            # ADR-0061 noise on the anchor aux only; the targets stay clean.
            anchor_aux = _state_input_noise(
                batch["input_aux"].to(device), aux_noise, is_kinematic
            )
            anchor_time_feature = None
            if cfg.flow_map_anchor_time:
                # Per-example anchor time, broadcast to each example's rows
                # (the loading_feature convention).
                anchor_time_feature = torch.repeat_interleave(
                    anchor_frame.to(torch.float32) / (time_ref - 1),
                    n_particles_per_example,
                ).unsqueeze(1)

            accumulate = step < cfg.normalizer_warmup_steps
            optimizer.zero_grad()
            # ADR-0075: the element-state pair of a two-site step (else None).
            state_pred: Tensor | None = None
            state_target: Tensor | None = None
            # ADR-0064 knob 2: one hinge term per query, each against the
            # anchor actually FED to that query; empty when the knob is off.
            hinge_terms: list[Tensor] = []
            if hinge_knots is not None:
                # Review finding: a physical floor, not an epsilon — an
                # all-elastic batch has GT peeq std ~0, and a 1e-6 floor
                # would amplify the irreversibility term ~1e5x into
                # clip-saturating spikes that distort the comparator arms.
                # Taylor's per-batch peeq std is O(0.4); 0.05 never binds
                # on normal batches and caps the amplifier at ~20x.
                peeq_scale = next_aux[..., 3].std().detach().clamp(min=0.05)
            if chain and cfg.flow_map_pushforward_generations > 1:
                # ADR-0063 amendment: GENERATION CURRICULUM. The chain
                # deepens to g detached re-anchor events, g annealed in
                # equal phases over training; each generation's pair is
                # predicted from the previous generation's (detached,
                # clamped) self-anchor. Only generation-1 queries are clean
                # — they alone warm the normalizers.
                gens = cfg.flow_map_pushforward_generations
                g_level = min(gens, 1 + (step * gens) // train_cfg.training_steps)
                cf = batch["chain_frames"].to(device)  # (B, G)
                preds: list[Tensor] = []
                targets: list[Tensor] = []
                cur_disp, cur_vel, cur_aux = anchor_disp, anchor_vel, anchor_aux
                cur_time = anchor_time_feature
                prev_frames = anchor_frame  # (B,)
                for i in range(g_level):
                    t_i = cf[:, i]
                    pair_preds: list[Tensor] = []
                    for j in (0, 1):
                        dt_i = (t_i - 1 + j - prev_frames).to(torch.float32) / (
                            time_ref - 1
                        )
                        p_k, t_k = sim.forward_train_tc(
                            next_position[:, 2 * i + j],
                            next_aux[:, 2 * i + j],
                            particle_type,
                            reference_coords,
                            n_particles_per_example,
                            dt_i,
                            accumulate=accumulate and i == 0,
                            loading_feature=loading_feature,
                            anchor_disp=cur_disp,
                            anchor_vel=cur_vel,
                            anchor_aux=cur_aux,
                            anchor_time_feature=cur_time,
                        )
                        preds.append(p_k)
                        targets.append(t_k)
                        pair_preds.append(p_k)
                        if hinge_knots is not None:
                            hinge_terms.append(
                                _fm_admissibility_hinge(
                                    sim.train_output_state(p_k)[1],
                                    cur_aux[..., 3],
                                    hinge_knots,
                                    peeq_scale,
                                )
                            )
                    d1, _a1 = sim.train_output_state(pair_preds[0].detach())
                    d2, a2 = sim.train_output_state(pair_preds[1].detach())
                    cur_disp, cur_vel, cur_aux = d2, d2 - d1, a2
                    if is_kinematic.any():
                        cur_disp = cur_disp.clone()
                        cur_vel = cur_vel.clone()
                        cur_aux = cur_aux.clone()
                        kin = is_kinematic
                        gt_pair0 = next_position[:, 2 * i]
                        gt_pair1 = next_position[:, 2 * i + 1]
                        cur_disp[kin] = (gt_pair1 - reference_coords)[kin]
                        cur_vel[kin] = (gt_pair1 - gt_pair0)[kin]
                        cur_aux[kin] = next_aux[:, 2 * i + 1][kin]
                    cur_time = None
                    if cfg.flow_map_anchor_time:
                        cur_time = torch.repeat_interleave(
                            t_i.to(torch.float32) / (time_ref - 1),
                            n_particles_per_example,
                        ).unsqueeze(1)
                    prev_frames = t_i
                # Final (dirty) query: at full depth it targets t_final; at
                # a lower curriculum level it targets the next pair's first
                # frame (dt in [1, max_dt - 1] — always a valid offset).
                if g_level < gens:
                    fin_idx = 2 * g_level
                    fin_frames = cf[:, g_level] - 1
                else:
                    fin_idx = 2 * gens
                    fin_frames = target_frame
                dt_f = (fin_frames - prev_frames).to(torch.float32) / (time_ref - 1)
                p_k, t_k = sim.forward_train_tc(
                    next_position[:, fin_idx],
                    next_aux[:, fin_idx],
                    particle_type,
                    reference_coords,
                    n_particles_per_example,
                    dt_f,
                    accumulate=False,
                    loading_feature=loading_feature,
                    anchor_disp=cur_disp,
                    anchor_vel=cur_vel,
                    anchor_aux=cur_aux,
                    anchor_time_feature=cur_time,
                )
                preds.append(p_k)
                targets.append(t_k)
                if hinge_knots is not None:
                    hinge_terms.append(
                        _fm_admissibility_hinge(
                            sim.train_output_state(p_k)[1],
                            cur_aux[..., 3],
                            hinge_knots,
                            peeq_scale,
                        )
                    )
                pred = torch.stack(preds, dim=1)  # (P, 2*g_level+1, dim+C)
                target = torch.stack(targets, dim=1)
            elif chain and two_site:
                # ADR-0075 note 2026-11-11: the ADR-0063 two-hop chain with
                # the element state carried across the mid-chain anchor.
                pred, target, state_pred, state_target, _anchor_b = _fm_two_site_chain(
                    sim,
                    next_position,
                    next_aux,
                    batch["next_state"].to(device),
                    particle_type,
                    reference_coords,
                    n_particles_per_example,
                    anchor_frame,
                    batch["chain_frame"].to(device),
                    target_frame,
                    time_ref,
                    accumulate=accumulate,
                    cells=batch["cells"].to(device),
                    weights=batch["cell_weights"].to(device),
                    law=(
                        None
                        if laws is None
                        else law_rows(
                            batch["law_youngs"].to(device),
                            batch["law_poisson"].to(device),
                            batch["law_knots_peeq"].to(device),
                            batch["law_knots_sy"].to(device),
                        )
                    ),
                    anchor_disp=anchor_disp,
                    anchor_vel=anchor_vel,
                    anchor_aux=anchor_aux,
                    anchor_state=batch["input_state"].to(device),
                    is_kinematic=is_kinematic,
                    loading_feature=loading_feature,
                    anchor_time_feature=anchor_time_feature,
                    anchor_time=cfg.flow_map_anchor_time,
                )
            elif chain:
                # ADR-0063 knob 1. Step A (teacher-forced GT anchor at t0):
                # the TWO queries a re-anchor at t1 consumes — (t1-1, t1).
                chain_frame = batch["chain_frame"].to(device)  # (B,) = t1
                preds, targets = [], []
                for j in (0, 1):
                    dt_a = (chain_frame - 1 + j - anchor_frame).to(torch.float32) / (
                        time_ref - 1
                    )
                    p_j, t_j = sim.forward_train_tc(
                        next_position[:, j],
                        next_aux[:, j],
                        particle_type,
                        reference_coords,
                        n_particles_per_example,
                        dt_a,
                        accumulate=accumulate,
                        loading_feature=loading_feature,
                        anchor_disp=anchor_disp,
                        anchor_vel=anchor_vel,
                        anchor_aux=anchor_aux,
                        anchor_time_feature=anchor_time_feature,
                    )
                    preds.append(p_j)
                    targets.append(t_j)
                    if hinge_knots is not None:
                        hinge_terms.append(
                            _fm_admissibility_hinge(
                                sim.train_output_state(p_j)[1],
                                anchor_aux[..., 3],
                                hinge_knots,
                                peeq_scale,
                            )
                        )
                # Re-anchor at t1 on the DETACHED predictions: displacement /
                # FD velocity from the two raw displacement slices (the
                # reference cancels in the difference), aux from the t1
                # prediction; kinematic rows clamped to GT — the exact analog
                # of what the rollout hand-off feeds (ADR-0060 house clamp).
                disp_a1, _aux_a1 = sim.train_output_state(preds[0].detach())
                disp_a2, aux_a2 = sim.train_output_state(preds[1].detach())
                anchor_disp_b = disp_a2
                anchor_vel_b = disp_a2 - disp_a1
                anchor_aux_b = aux_a2
                if is_kinematic.any():
                    anchor_disp_b = anchor_disp_b.clone()
                    anchor_vel_b = anchor_vel_b.clone()
                    anchor_aux_b = anchor_aux_b.clone()
                    kin = is_kinematic
                    anchor_disp_b[kin] = (next_position[:, 1] - reference_coords)[kin]
                    anchor_vel_b[kin] = (next_position[:, 1] - next_position[:, 0])[kin]
                    anchor_aux_b[kin] = next_aux[:, 1][kin]
                anchor_time_b = None
                if cfg.flow_map_anchor_time:
                    anchor_time_b = torch.repeat_interleave(
                        chain_frame.to(torch.float32) / (time_ref - 1),
                        n_particles_per_example,
                    ).unsqueeze(1)
                dt_b = (target_frame - chain_frame).to(torch.float32) / (time_ref - 1)
                # Step B: the contraction term (normalizers never warm on
                # the dirty query; ADR-0063).
                p_b, t_b = sim.forward_train_tc(
                    next_position[:, 2],
                    next_aux[:, 2],
                    particle_type,
                    reference_coords,
                    n_particles_per_example,
                    dt_b,
                    accumulate=False,
                    loading_feature=loading_feature,
                    anchor_disp=anchor_disp_b,
                    anchor_vel=anchor_vel_b,
                    anchor_aux=anchor_aux_b,
                    anchor_time_feature=anchor_time_b,
                )
                if hinge_knots is not None:
                    hinge_terms.append(
                        _fm_admissibility_hinge(
                            sim.train_output_state(p_b)[1],
                            anchor_aux_b[..., 3],
                            hinge_knots,
                            peeq_scale,
                        )
                    )
                # (P, 3, dim+C): the rank-agnostic loss hunk below averages
                # over the three query losses (2:1 clean:dirty, ADR-0063).
                pred = torch.stack([*preds, p_b], dim=1)
                target = torch.stack([*targets, t_b], dim=1)
            elif two_site:
                # ADR-0075: the element state rides the anchor; the loss
                # below adds its per-channel z-scored error to the nodal one.
                dt_norm = (target_frame - anchor_frame).to(torch.float32) / (
                    time_ref - 1
                )  # (B,)
                law_batch: LawRows | None = None
                if laws is not None:
                    law_batch = law_rows(
                        batch["law_youngs"].to(device),
                        batch["law_poisson"].to(device),
                        batch["law_knots_peeq"].to(device),
                        batch["law_knots_sy"].to(device),
                    )
                pred, target, state_pred, state_target = sim.forward_train_two_site(
                    next_position,
                    next_aux,
                    batch["next_state"].to(device),
                    particle_type,
                    reference_coords,
                    n_particles_per_example,
                    dt_norm,
                    accumulate=accumulate,
                    cells=batch["cells"].to(device),
                    weights=batch["cell_weights"].to(device),
                    law=law_batch,
                    anchor_disp=anchor_disp,
                    anchor_vel=anchor_vel,
                    anchor_aux=anchor_aux,
                    anchor_state=batch["input_state"].to(device),
                    loading_feature=loading_feature,
                    anchor_time_feature=anchor_time_feature,
                )
            else:
                dt_norm = (target_frame - anchor_frame).to(torch.float32) / (
                    time_ref - 1
                )  # (B,)
                pred, target = sim.forward_train_tc(
                    next_position,
                    next_aux,
                    particle_type,
                    reference_coords,
                    n_particles_per_example,
                    dt_norm,
                    accumulate=accumulate,
                    loading_feature=loading_feature,
                    anchor_disp=anchor_disp,
                    anchor_vel=anchor_vel,
                    anchor_aux=anchor_aux,
                    anchor_time_feature=anchor_time_feature,
                )
                if hinge_knots is not None:
                    hinge_terms.append(
                        _fm_admissibility_hinge(
                            sim.train_output_state(pred)[1],
                            anchor_aux[..., 3],
                            hinge_knots,
                            peeq_scale,
                        )
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
            if state_pred is not None and state_target is not None:
                # ADR-0075: the element-site state loss, every element scored,
                # weighted like the nodal aux block.
                loss = (
                    loss + train_cfg.w_aux * ((state_pred - state_target) ** 2).mean()
                )
            if hinge_terms and free.any():
                # ADR-0064 knob 2: mean over queries then free rows — the
                # primary loss's pooling.
                hinge_pp = torch.stack(hinge_terms, dim=1).mean(dim=1)
                loss = loss + hinge_w * hinge_pp[free].mean()

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
                            # set_anchor's kinematic clamp reads the GT state.
                            gt_aux=torch.from_numpy(tr.aux).to(device),
                            # ADR-0075: the element state and the case's law.
                            gt_state=(
                                torch.from_numpy(tr.state).to(device)
                                if two_site and tr.state is not None
                                else None
                            ),
                            law=_case_law_rows(spec, tr, device) if two_site else None,
                        )
                        # Selection runs the DEPLOYABLE mode: self-anchored at
                        # the canonical interval (ADR-0062), on the clock the
                        # run trains on (``time_ref``; never one re-derived
                        # from a validation trajectory, which is truncated).
                        result = flow_map_rollout(
                            sim,
                            tr,
                            cfg.input_frames,
                            time_ref,
                            canonical,
                            "self",
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
                if best_rel == float("inf") and spec.source is not None and val_trajs:
                    # A by-path arm dies at its first validation, not hours
                    # later (TrainingDiverged).
                    reason = _early_divergence(
                        float(loss.item()), val_pos, _domain_scale(val_trajs)
                    )
                    if reason is not None:
                        raise TrainingDiverged(f"step {step}: {reason}")
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
