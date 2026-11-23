"""Helpers the training loops share: noise, schedules, divergence checks,
the run's training-pool record and the time references."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ...benchmarks import (
    BenchmarkSpec,
    ConstitutiveLaw,
)
from ...config import (
    LR_SCHEDULE_FLOOR,
    CGNConfig,
    TrainConfig,
)
from ...datasets import (
    CaseTrajectory,
    NormalizationStats,
)
from ...models.cgn.simulator import time_diff
from ...models.common.two_site import (
    LawRows,
    law_rows,
)

logger = logging.getLogger(__name__)

#: Cadence (steps) of the periodic ``ckpt-<step>.pt`` snapshots written
#: alongside the selection checkpoints. Fleet tooling, not recipe: they let
#: post-hoc smoothed selection re-score a run's trajectory of states
#: identically across ablation arms (ADR-0028, 2026-07-10 note). The name
#: sits outside the ``model-*.pt`` glob so default evaluation never picks
#: them up.
PERIODIC_CKPT_EVERY = 10_000


#: A z-scored training loss above this at the first validation is not a
#: training run: every loss term the trainers form is a squared error of
#: normalized quantities, O(1) by construction before any learning.
DIVERGENCE_LOSS_CEILING = 100.0


#: The validation line of the mesh-family loops. ``val_rel_l2`` selects
#: ``model-best`` (ADR-0077); the millimetre number stays beside it,
#: unselected, so logs from before the change compare.
_VAL_LOG = "step %d: train_loss %.6f val_rel_l2 %.6f val_pos %.4f mm (best_rel_l2 %.6f)"


class TrainingDiverged(RuntimeError):
    """A by-path run whose first validation shows it is not training.

    Raised by a trainer when, at its first validation, the rollout position
    RMSE exceeds the benchmark's own domain scale or the training loss is
    orders of magnitude above O(1) (:data:`DIVERGENCE_LOSS_CEILING`). The
    message is one line; :func:`main` prints it and exits with status 3.

    The trainer exits rather than warns: an arm submitted to a scheduler
    runs unattended for hours after its first validation, and a warning in a
    log nobody reads until the end saves none of that compute. The check is
    one-shot and scoped to benchmarks loaded by path (``spec.source`` set):
    runs of the shipped benchmarks behave exactly as before.
    """


def _domain_scale(trajectories: Sequence[CaseTrajectory]) -> float:
    """The benchmark's domain scale: the largest bounding-box diagonal of the
    reference coordinates over ``trajectories``, in the working length unit.
    A rollout position RMSE above it is a prediction that is not a deformed
    configuration of the body at all."""
    scale = 0.0
    for tr in trajectories:
        ref = np.asarray(
            tr.reference_coords if tr.reference_coords is not None else tr.positions[0],
            dtype=np.float64,
        )
        if ref.shape[0] == 0:
            continue
        scale = max(scale, float(np.linalg.norm(ref.max(axis=0) - ref.min(axis=0))))
    return scale


def _early_divergence(
    train_loss: float, val_pos: float, domain_scale: float
) -> str | None:
    """The one-line reason a first validation shows a run is not training,
    or ``None`` when it passes: the position RMSE exceeds ``domain_scale``
    (when that is positive), the loss exceeds
    :data:`DIVERGENCE_LOSS_CEILING`, or either is not finite."""
    reasons: list[str] = []
    if not math.isfinite(val_pos) or (domain_scale > 0 and val_pos > domain_scale):
        reasons.append(
            f"validation position RMSE {val_pos:.4g} exceeds the domain scale "
            f"{domain_scale:.4g} (bounding-box diagonal of the validation cases)"
        )
    if not math.isfinite(train_loss) or train_loss > DIVERGENCE_LOSS_CEILING:
        reasons.append(
            f"train_loss {train_loss:.4g} exceeds {DIVERGENCE_LOSS_CEILING:g} "
            "(a z-scored loss is O(1) by construction)"
        )
    if not reasons:
        return None
    return "first validation shows the run is not training: " + "; ".join(reasons)


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
    history_frames: int,
    noise_off: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Noisy inputs, optional velocity-history feature, and the matched target.

    Shared by the three mesh-family training loops (ADR-0049). The two paths
    deliberately pair DIFFERENT noise schemes with DIFFERENT target
    conventions, each matching its reference recipe:

    * Reference path (``history_frames == 0``): single-frame Gaussian
      noise on the current frame; ``next_position`` is returned UNCHANGED,
      so the caller's velocity target ``next - x_noisy`` measures from the
      noisy position — the MeshGraphNets gamma = 1 convention (the model
      learns to step back onto the clean trajectory; the correction is a
      single sigma).
    * History path (``history_frames > 0``): :func:`random_walk_position_noise`
      over the ``history_frames + 1`` frames the model uses (the CGN recipe,
      history-matched per ADR-0053; at ``history_frames == input_frames - 1``
      this is the full window), so the velocity features are consistently noisy
      — and the returned target next position is
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

    ``noise_off=True`` selects a THIRD regime, the ADR-0051 one-shot (k=T)
    scheme: with no autoregressive feedback, single-step drift noise is
    meaningless (it would bias the one-shot model toward over-contraction, the
    ADR-0049 pathology), so the inputs are CLEAN, the history feature
    (if enabled) is the CLEAN window's velocities, and the k-frame target is
    returned unchanged (clean full-sequence L2, the CarCrashNet regime). This
    decouples the history INPUT feature from the noise SCHEME, which the
    two-branch reference recipe conflates.

    Parameters
    ----------
    position_seq : torch.Tensor
        ``(P, F, dim)`` collated position windows, most recent frame last.
    next_position : torch.Tensor
        Ground-truth target: ``(P, dim)`` on the k=1 paths; ``(P, k, dim)`` on
        the one-shot (``noise_off``) path, returned unchanged.
    is_kinematic : torch.Tensor
        ``(P,)`` bool mask of kinematic rows.
    noise_std : float
        The family config's ``noise_std``.
    history_frames : int
        The family config's ``history_frames`` count (ADR-0053): ``0`` selects
        the Markovian single-frame Gaussian path; ``k > 0`` selects the
        random-walk path over the last ``k`` window velocities.
    noise_off : bool
        Select the one-shot clean regime (ADR-0051). Default ``False`` keeps
        the two k=1 reference paths byte-identical (MGN/GeoFLARE never pass it).

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor or None, torch.Tensor)
        ``(x_noisy (P, dim), velocity_history (P, history_frames*dim) or None,
        next_target)`` — feed ``next_target``, not the raw ``next_position``,
        to ``forward_train``.
    """
    if noise_off:
        x_clean = position_seq[:, -1]
        vh = None
        if history_frames:
            win = position_seq[:, -(history_frames + 1) :]
            vh = (win[:, 1:] - win[:, :-1]).flatten(1)
        return x_clean, vh, next_position
    if history_frames == 0:
        noise = torch.randn_like(position_seq[:, -1]) * noise_std
        noise = noise.masked_fill(is_kinematic.unsqueeze(-1), 0.0)
        return position_seq[:, -1] + noise, None, next_position
    # ADR-0053: the random walk (and its velocity features) spans exactly the
    # history_frames + 1 frames the model uses, so intermediate k gets a
    # history-matched noise scale; at k = input_frames - 1 the window is the
    # full seq and this is byte-identical to the pre-0053 velocity_history path.
    win = position_seq[:, -(history_frames + 1) :]
    noise_seq = random_walk_position_noise(win, noise_std)
    noise_seq = noise_seq.masked_fill(is_kinematic.unsqueeze(-1).unsqueeze(-1), 0.0)
    noisy_seq = win + noise_seq
    velocities = noisy_seq[:, 1:] - noisy_seq[:, :-1]
    return (
        noisy_seq[:, -1],
        velocities.flatten(1),
        next_position + noise_seq[:, -1],
    )


def _state_fields_for(
    family: str, model_cfg: Any, spec: BenchmarkSpec
) -> tuple[str, ...] | None:
    """The element-site fields a run loads, or ``None`` (ADR-0075).

    Only a Transolver run with a two-site decoder reads the element block;
    every other run keeps the nodal view alone, so its memory and its data
    path are unchanged.
    """
    if (
        family == "transolver"
        and getattr(model_cfg, "two_site_decoder", "off") != "off"
    ):
        return tuple(spec.state_channels)
    return None


def _case_law_rows(
    spec: BenchmarkSpec, trajectory: CaseTrajectory, device: str
) -> LawRows | None:
    """The per-element law rows of one case for a law-reading two-site decoder
    (ADR-0075), or ``None`` when the benchmark declares no law."""
    if spec.constitutive_law is None or trajectory.state is None:
        return None
    law = spec.constitutive_law(trajectory.case_id)
    if not isinstance(law, ConstitutiveLaw):
        return None  # no isotropic table to tile; no decoder reads this law
    n = int(trajectory.state.shape[1])
    knots_peeq, knots_sy = law.hardening
    return law_rows(
        torch.full((n,), law.youngs_modulus, dtype=torch.float32, device=device),
        torch.full((n,), law.poisson_ratio, dtype=torch.float32, device=device),
        torch.tensor(knots_peeq, dtype=torch.float32, device=device).expand(n, -1),
        torch.tensor(knots_sy, dtype=torch.float32, device=device).expand(n, -1),
    )


def _truncate_trajectory(tr: CaseTrajectory, n: int | None) -> CaseTrajectory:
    """The first ``n`` frames of every per-frame array (``None`` = all)."""
    if n is None:
        return tr
    return replace(
        tr,
        positions=tr.positions[:n],
        aux=tr.aux[:n],
        time=tr.time[:n],
        state=None if tr.state is None else tr.state[:n],
    )


def _record_training_pool(
    out_dir: Path,
    spec: BenchmarkSpec,
    train_cfg: TrainConfig,
    train_trajs: Sequence[CaseTrajectory],
    make_dataset: Callable[[list[CaseTrajectory]], Any],
    n_samples: int,
    time_ref: int | None = None,
) -> dict[str, Any]:
    """Log the training pool and write it into the run record (ADR-0078).

    ``run.training_pool`` in ``config.json`` holds the span setting in force,
    the number of samples in the pool, the share of them whose every frame
    lies inside the case's scored span, and the passes over the pool the
    schedule makes (steps x batch / samples). A time-conditioned or flow-map
    run also gets ``run.time_ref_frames``, the denominator of its normalized
    time, which the evaluator reads back.

    ``make_dataset`` builds the loop's own sample index from a trajectory
    list; the scored share is the size of that index on the pool cut to each
    case's scored bound, over its size on the pool as trained.
    """
    scored_trajs = [
        _truncate_trajectory(tr, spec.scored_frames_for(tr.case_id))
        for tr in train_trajs
    ]
    if all(
        a.positions.shape[0] == b.positions.shape[0]
        for a, b in zip(scored_trajs, train_trajs, strict=True)
    ):
        n_scored = n_samples
    else:
        n_scored = len(make_dataset(scored_trajs))
    pool: dict[str, Any] = {
        "training_span": train_cfg.training_span,
        "train_frames": train_cfg.train_frames,
        "scored_frames": _scored_frames_record(spec),
        "samples": n_samples,
        "scored_share": n_scored / n_samples,
        "passes": train_cfg.training_steps * train_cfg.batch_size / n_samples,
    }
    logger.info(
        "training pool: span %r (train_frames=%d), %d samples, %.1f%% inside "
        "the scored span, time reference %s, %.3g passes over the pool "
        "(%d steps x batch %d)",
        pool["training_span"],
        pool["train_frames"],
        n_samples,
        100.0 * pool["scored_share"],
        "n/a" if time_ref is None else f"{time_ref} frames",
        pool["passes"],
        train_cfg.training_steps,
        train_cfg.batch_size,
    )
    config_path = out_dir / "config.json"
    record = json.loads(config_path.read_text(encoding="utf-8"))
    record["run"]["training_pool"] = pool
    if time_ref is not None:
        record["run"]["time_ref_frames"] = time_ref
    config_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return pool


def _definition_record(spec: BenchmarkSpec) -> dict[str, Any] | None:
    """The ``run.definition`` block for a benchmark loaded from a definition
    directory (its absolute path and file hashes), else ``None``."""
    return None if spec.source is None else spec.source.to_json_dict()


def _scored_frames_record(spec: BenchmarkSpec) -> int | str | None:
    """What a metrics record says about the scored horizon: the benchmark's
    one bound, ``None`` when scored to the end, or ``"per_case"`` when each
    case has its own (the definition the run record hashes holds the rule)."""
    return "per_case" if callable(spec.scored_frames) else spec.scored_frames_fixed


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


#: ADR-0062: the per-interval flow-map field metrics, each recorded per case
#: under ``rollout_m{m}_*`` / ``rollout_oracle_m{m}_*`` AND aggregated into
#: the split means. One vocabulary, two consumers — keep them in lockstep by
#: construction (each suffix maps to ``RolloutResult.mean_<suffix>``).
_FM_INTERVAL_METRICS = (
    "position_rmse",
    "aux_rmse",
    "rel_l2_displacement",
    "rel_l2_aux",
)


def _fm_intervals_and_canonical(model_cfg: Any) -> tuple[tuple[int, ...], int]:
    """Resolve the ADR-0062 interval sweep and canonical interval (0 = first).

    The SINGLE resolution site for trainer and evaluator: the canonical
    interval selects both the in-training validation rollout and the eval
    pass that writes the standard ``rollout_*`` keys/artifacts/QoIs — a
    drift between the two would make checkpoint selection optimize a
    different rollout than the one the registry-facing artifacts record.
    """
    intervals = tuple(
        int(v) for v in (getattr(model_cfg, "flow_map_eval_intervals", ()) or ())
    )
    canonical = int(getattr(model_cfg, "flow_map_canonical_interval", 0) or 0) or (
        intervals[0] if intervals else 0
    )
    return intervals, canonical


def _tc_time_ref_frames(
    scored_frames: int | None, train_frames: int, traj_len: int
) -> int:
    """Denominator of the time-conditioned normalized query time (ADR-0054).

    The query time is ``t = frame / (time_ref_frames - 1)``, so ``time_ref_frames``
    is the scored horizon in frames. It must be identical at train and eval for
    ``t`` to mean the same physical instant. Precedence: the pinned scored
    horizon (``scored_frames``) if set; else the recipe truncation
    (``train_frames``) if set; else the trajectory's own frame count.

    This is the rule for a benchmark with a fixed bound or none, and the one
    a run record from before ADR-0078 is evaluated under. Callers go through
    :func:`_training_time_ref` (training, which also handles a per-case
    bound and whose value is written to the run record) and
    :func:`_run_time_ref` (evaluation, which reads it back). ``traj_len``
    must be an UNCUT trajectory's frame count.
    """
    if scored_frames is not None:
        return scored_frames
    if train_frames > 0:
        return train_frames
    return traj_len


def _training_time_ref(
    spec: BenchmarkSpec, train_cfg: TrainConfig, pool_min_frames: int
) -> int:
    """The time reference a time-conditioned or flow-map run trains on
    (ADR-0078): one number per benchmark, which does not move with the
    training span.

    A fixed scored bound is the reference (ADR-0054). A per-case bound takes
    the card's ``n_frames``: the pool is then cut per case, so its shortest
    trajectory is one case's span, not a clock. A benchmark that declares no
    span keeps the rule it had, the fixed cut ``train_frames`` or else the
    shortest training trajectory, which is uncut there. The value is written
    to the run record (``run.time_ref_frames``) and :func:`_run_time_ref`
    reads it back.
    """
    if callable(spec.scored_frames):
        return spec.card.n_frames
    return _tc_time_ref_frames(
        spec.scored_frames_fixed, train_cfg.train_frames, pool_min_frames
    )


def _state_input_noise(
    input_aux: Tensor,
    noise_std: tuple[float, ...] | None,
    is_kinematic: Tensor,
) -> Tensor:
    """ADR-0061 knob 1: per-channel batch-std-relative Gaussian state noise.

    ``input_aux + knob_c * std_batch_c * N(0, 1)`` — the scale is RELATIVE
    to each channel's batch standard deviation (mixed-unit channels make an
    absolute scale meaningless). The target stays the clean GT state, so the
    model learns to contract perturbed states. ``None`` or all-zero knobs
    return the input unchanged (byte-identical off path).

    Kinematic rows stay clean, matching :func:`_mesh_family_noise` (ADR-0043
    §4) and rollout, where their fed state is always GT (oracle mode reads
    ``gt_aux``; the self-fed cache clamps them) — noising them would train on
    an input distribution rollout never produces. The batch std is still
    pooled over all rows: kinematic aux is near-constant, which deflates the
    relative scale slightly, but the knob is a swept scale so this folds into
    the sweep.
    """
    if noise_std is None or not any(v > 0 for v in noise_std):
        return input_aux
    scale = torch.tensor(noise_std, dtype=input_aux.dtype, device=input_aux.device)
    std = input_aux.std(dim=0, keepdim=True)
    noise = scale * std * torch.randn_like(input_aux)
    noise = noise.masked_fill(is_kinematic.unsqueeze(-1), 0.0)
    return input_aux + noise
