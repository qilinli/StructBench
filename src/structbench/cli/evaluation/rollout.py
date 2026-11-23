"""The ``valid`` and ``rollout`` modes: metrics, properties, QoIs."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...benchmarks import (
    BenchmarkSpec,
    CombinedHardeningLaw,
    ConstitutiveLaw,
)
from ...datasets import (
    CaseTrajectory,
    aux_channel_labels,
    aux_channel_units,
    load_case_trajectory,
)
from ...eval import (
    RolloutResult,
    closure_ratio,
    error_growth,
    flow_map_rollout,
    one_step_aux_rmse,
    one_step_position_rmse,
    peeq_decrease_rate,
    relative_l2_pooled,
    rollout,
    state_sufficiency,
    time_conditioned_rollout,
    yield_violation_rate,
    yield_violation_rate_about_backstress,
)
from ...eval.properties import von_mises4
from ...models.common import CaseBoundSimulator
from ...models.geoflare import GeoFlareSimulator
from ...models.transolver import (
    TransolverSimulator,
)
from ..training.common import (
    _FM_INTERVAL_METRICS,
    _case_law_rows,
    _fm_intervals_and_canonical,
    _scored_frames_record,
    _state_fields_for,
    _truncate_trajectory,
)
from .run import _json_safe, _rebuild_run_simulator, _run_time_ref

logger = logging.getLogger(__name__)

#: The element-state labels the constraint properties read (plan 3c): the
#: four stress components of the return-map layout and the plastic strain.
_STATE_STRESS_LABELS: tuple[str, ...] = ("s_rr", "s_zz", "s_tt", "s_rz")


_STATE_PEEQ_LABEL = "effective_plastic_strain"


#: The backstress components a combined-hardening benchmark carries beside them.
_STATE_BACKSTRESS_LABELS: tuple[str, ...] = ("a_rr", "a_zz", "a_tt", "a_rz")


def _defined(value: float) -> float | None:
    """A property kernel's ``nan`` (undefined on this input) as ``None``, the
    record's word for undefined (as ``one_step_*`` under ADR-0054), in memory
    as on disk."""
    return value if math.isfinite(value) else None


def _state_constraint_properties(
    predicted_state: np.ndarray | None,
    state_labels: tuple[str, ...],
    law: ConstitutiveLaw | CombinedHardeningLaw | None,
    tolerances: Mapping[str, float],
    input_frames: int,
    bound: int | None,
) -> dict[str, float | None]:
    """The yield-bound and plastic-strain properties of one rollout's element
    state (ADR-0075 clause 6), ``None`` where the run carries no state block,
    the layout lacks the channel, or the case has no law. Under a combined-
    hardening law the yield bound is measured about the backstress, and is
    undefined on a layout that does not carry it.

    The plastic strain is read from the last seeded frame on, so the first
    predicted step is a step; the yield bound over the predicted frames only.
    Both run over the scored span ``[input_frames, bound)``.
    """
    out: dict[str, float | None] = {
        "prop_yield_violation_rate": None,
        "prop_peeq_decrease_rate": None,
    }
    if predicted_state is None:
        return out
    if _STATE_PEEQ_LABEL not in state_labels:
        logger.info(
            "element-site layout %s has no %r channel: the yield and plastic-"
            "strain properties are undefined on it, not unreported",
            state_labels,
            _STATE_PEEQ_LABEL,
        )
        return out
    labels = list(state_labels)
    n_frames = predicted_state.shape[0]
    stop = n_frames if bound is None else min(bound, n_frames)
    peeq_idx = labels.index(_STATE_PEEQ_LABEL)
    out["prop_peeq_decrease_rate"] = peeq_decrease_rate(
        predicted_state[input_frames - 1 : stop, :, peeq_idx], tolerances["peeq"]
    )
    if law is None or not all(n in labels for n in _STATE_STRESS_LABELS):
        return out
    if isinstance(law, CombinedHardeningLaw):
        if not all(n in labels for n in _STATE_BACKSTRESS_LABELS):
            logger.info(
                "element-site layout %s carries no backstress: the yield "
                "property of a combined-hardening law is undefined on it, not "
                "unreported",
                state_labels,
            )
            return out
        stress_idx = [labels.index(n) for n in _STATE_STRESS_LABELS]
        back_idx = [labels.index(n) for n in _STATE_BACKSTRESS_LABELS]
        scored = predicted_state[input_frames:stop]
        out["prop_yield_violation_rate"] = yield_violation_rate_about_backstress(
            scored[:, :, stress_idx],
            scored[:, :, back_idx],
            scored[:, :, peeq_idx],
            law.yield_stress,
            law.isotropic_saturation,
            law.isotropic_rate,
            tolerances["yield"],
        )
    else:
        stress_idx = [labels.index(n) for n in _STATE_STRESS_LABELS]
        scored = predicted_state[input_frames:stop]
        out["prop_yield_violation_rate"] = yield_violation_rate(
            von_mises4(scored[:, :, stress_idx]),
            scored[:, :, peeq_idx],
            law.hardening[0],
            law.hardening[1],
            tolerances["yield"],
        )
    return out


def _case_properties(
    spec: BenchmarkSpec,
    result: RolloutResult,
    one_step: np.ndarray | None,
    n_scored: int | None,
    state_labels: tuple[str, ...],
    case_id: str,
    input_frames: int,
    bound: int | None,
) -> dict[str, float | None]:
    """The properties block of one case (ADR-0075 clause 6; plan 3c): every
    key the benchmark defines, ``None`` where it is undefined on this run
    (closure without a one-step sweep, ADR-0054; sufficiency until the pair
    is paired after the loop). Keys ride beside the accuracy metrics."""
    scored = result.position_rmse[:n_scored]
    props: dict[str, float | None] = {
        "prop_closure_ratio": (
            None
            if one_step is None
            else _defined(closure_ratio(scored, one_step[:n_scored]))
        ),
        "prop_error_growth": _defined(error_growth(scored)),
    }
    if spec.state_site == "element":
        law = spec.constitutive_law(case_id) if spec.constitutive_law else None
        props.update(
            _state_constraint_properties(
                result.predicted_state,
                state_labels,
                law,
                spec.property_tolerances,
                input_frames,
                bound,
            )
        )
    if spec.matched_initial_state_pairs:
        props["prop_state_sufficiency"] = None
    return props


#: The numbers an ``unscored_tail`` block may hold (ADR-0078), in the order
#: the split mean lists them. Same names as the scored keys: the block they
#: sit in is what says they are unscored.
_UNSCORED_TAIL_KEYS = (
    "rollout_position_rmse",
    "rollout_aux_rmse",
    "rollout_rel_l2_displacement",
    "rollout_rel_l2_aux",
    "prop_closure_ratio",
    "prop_error_growth",
    "prop_yield_violation_rate",
    "prop_peeq_decrease_rate",
)


def _unscored_tail(
    spec: BenchmarkSpec,
    trajectory: CaseTrajectory,
    result: RolloutResult,
    one_step: np.ndarray | None,
    state_labels: tuple[str, ...],
    input_frames: int,
    bound: int,
) -> dict[str, Any]:
    """The errors and properties of one rollout over the frames at and beyond
    its case's scored bound, ``[bound, last stored frame]`` (ADR-0078; the
    long-horizon diagnostic of ADR-0039 §3).

    Nothing here is a registry metric. The forms are the scored ones (mean of
    per-frame RMSE, pooled relative L2 with the kinematic mask, the
    properties of ADR-0075) taken over the tail's frames; the plastic-strain
    property reads from the last scored frame on, so the first tail step is
    a step.
    """
    n_scored = bound - input_frames
    kin = np.isin(trajectory.particle_type, np.asarray(spec.kinematic_types))
    keep: np.ndarray | None = ~kin if kin.any() else None
    ref0 = trajectory.positions[0]
    pos_tail = result.position_rmse[n_scored:]
    true_aux = trajectory.aux[bound:]
    pred_aux = result.predicted_aux[bound:].reshape(true_aux.shape)
    out: dict[str, Any] = {
        "frames": [bound, len(trajectory.time)],
        "rollout_position_rmse": float(pos_tail.mean()),
        "rollout_aux_rmse": float(result.aux_rmse[n_scored:].mean()),
        "rollout_rel_l2_displacement": float(
            relative_l2_pooled(
                result.predicted_positions[bound:] - ref0,
                trajectory.positions[bound:] - ref0,
                mask=keep,
            )
        ),
        "rollout_rel_l2_aux": float(relative_l2_pooled(pred_aux, true_aux, mask=keep)),
        "prop_closure_ratio": (
            None
            if one_step is None
            else _defined(closure_ratio(pos_tail, one_step[n_scored:]))
        ),
        "prop_error_growth": _defined(error_growth(pos_tail)),
    }
    if spec.state_site == "element":
        law = (
            spec.constitutive_law(trajectory.case_id) if spec.constitutive_law else None
        )
        out.update(
            _state_constraint_properties(
                result.predicted_state,
                state_labels,
                law,
                spec.property_tolerances,
                bound,
                None,
            )
        )
    return out


def evaluate(
    case_ids: list[str],
    data_root: Path,
    out_dir: Path,
    device: str,
    *,
    split_name: str = "eval",
    save_artifacts: bool = True,
    checkpoint: str | Path | None = None,
    tail: bool = False,
) -> dict[str, Any]:
    """Roll out the run's checkpoint over ``case_ids`` and report ADR-0019 §5.

    A case with a scored bound is rolled out to that bound and no further
    (ADR-0078): the scored span is all the evaluator computes and saves.
    ``tail=True`` rolls on to the last stored frame and adds, per case and
    in the split mean, an ``unscored_tail`` block (the errors and properties
    of the frames at and beyond the bound) and the full-length rollout
    artifact. The scored numbers are the same either way.

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
    tail : bool
        Roll each bounded case on past its scored bound and report the tail
        under ``unscored_tail`` (ADR-0078; the ADR-0039 §3 diagnostic). No
        effect on a benchmark that declares no scored span.
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
    run = _rebuild_run_simulator(out_dir, device, checkpoint)
    spec, record, model_cfg, simulator = (
        run.spec,
        run.record,
        run.model_cfg,
        run.simulator,
    )
    family, ckpt_path = run.family, run.ckpt_path
    run_aux_fields, run_aux_labels = run.aux_fields, run.aux_labels
    n_aux_c = len(run_aux_labels)
    aux_in_run, qoi_aux_channel = run.aux_in_run, run.qoi_aux_channel
    # Non-None for any CaseBoundSimulator arm (mgn, transolver and geoflare, ADR-0041):
    # gates the per-case bind_case/reset_rollout calls below without
    # re-checking `family` at each site.
    mesh_sim = simulator if isinstance(simulator, CaseBoundSimulator) else None
    # ADR-0054: a time-conditioned run has no autoregressive rollout and no
    # teacher-forced one-step sweep — it queries every scored frame
    # independently and one_step_* is undefined (reported as null).
    tc = getattr(model_cfg, "time_conditioned", False)
    # ADR-0062: an anchored-flow-map run evaluates the re-anchoring rollout
    # per interval, in self- and oracle-anchored modes; the canonical
    # interval's self-anchored pass writes the standard keys/artifacts/QoIs.
    fm = family == "transolver" and bool(getattr(model_cfg, "flow_map", False))
    fm_intervals, fm_canonical = _fm_intervals_and_canonical(model_cfg)
    # ADR-0075: a two-site run scores the element state beside the nodal view.
    two_site_run = _state_fields_for(family, model_cfg, spec) is not None
    state_labels = aux_channel_labels(spec.state_channels) if two_site_run else ()
    fm_metrics = _FM_INTERVAL_METRICS + (
        ("state_rmse", "rel_l2_state") if two_site_run else ()
    )
    if fm and not fm_intervals:
        raise ValueError(
            "flow_map=true run record carries no flow_map_eval_intervals "
            "(ADR-0062); the record is corrupt or was built without "
            "load-time validation"
        )

    # Explicit-checkpoint sweeps must not clobber the selected checkpoint's
    # canonical artifacts: suffix the metrics file and skip the rollout .npz.
    save_rollouts = save_artifacts and checkpoint is None
    metrics_tag = split_name if checkpoint is None else f"{split_name}@{ckpt_path.stem}"
    rollout_dir = out_dir / "rollouts"
    if save_rollouts:
        rollout_dir.mkdir(parents=True, exist_ok=True)

    cases: dict[str, dict[str, Any]] = {}
    # ADR-0075 clause 6: the scored-span positions of every case that is a
    # member of a declared matched pair, paired after the loop.
    pair_members = {c for pair in spec.matched_initial_state_pairs for c in pair}
    kept_positions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case_id in case_ids:
        trajectory = load_case_trajectory(
            data_root / f"{case_id}.h5",
            aux_field=run_aux_fields,
            state_fields=_state_fields_for(family, model_cfg, spec),
        )
        if mesh_sim is not None and spec.mesh_transform is not None:
            # ADR-0047: same synthesis the training load applied.
            trajectory = spec.mesh_transform(trajectory)
        # ADR-0078: the case's scored bound, read once. Unless the tail is
        # asked for, the case is cut to it here, so the rollout, every pass
        # and the saved artifact stop at the bound. The time reference is
        # the run's own and never read off the cut length.
        case_bound = spec.scored_frames_for(case_id)
        stored_frames = len(trajectory.time)
        if case_bound is not None and not tail:
            trajectory = _truncate_trajectory(trajectory, case_bound)
        if mesh_sim is not None:
            if trajectory.cells is None or trajectory.reference_coords is None:
                raise ValueError(
                    f"benchmark {spec.card.name!r} has no mesh connectivity "
                    f"(cells/reference_coords); {family} evaluation requires "
                    "a mesh benchmark"
                )
            two_site_bind: dict[str, Any] = {}
            if two_site_run:
                two_site_bind = {
                    "gt_state": (
                        None
                        if trajectory.state is None
                        else torch.from_numpy(trajectory.state).to(device)
                    ),
                    "law": _case_law_rows(spec, trajectory, device),
                }
            mesh_sim.bind_case(
                torch.from_numpy(trajectory.cells).to(device),
                torch.from_numpy(trajectory.reference_coords).to(device),
                torch.from_numpy(trajectory.particle_type).to(device),
                torch.from_numpy(trajectory.positions).to(device),
                loading_scalar=(
                    spec.loading_scalar(case_id)
                    if getattr(model_cfg, "impact_velocity_feature", False)
                    and spec.loading_scalar
                    else None
                ),
                gt_aux=(
                    torch.from_numpy(trajectory.aux).to(device) if aux_in_run else None
                ),
                # ADR-0075: a two-site run binds the element state and the
                # case's law; every other run passes nothing new.
                **two_site_bind,
            )
            mesh_sim.reset_rollout()
        one_step: np.ndarray | None
        one_step_aux: np.ndarray | None
        result_oracle = None  # ADR-0060 oracle-state rollout (aux_input only)
        fm_extra: dict[str, Any] = {}  # ADR-0062 per-interval metrics
        if tc:
            # Time-conditioned: independent per-frame query, no accumulation and
            # no teacher-forced one-step sweep (ADR-0054). one_step_* is undefined.
            # tc is set only for the Transolver / GeoFLARE families (the two
            # exposing predict_state_at); assert so mypy narrows the union.
            assert isinstance(simulator, (TransolverSimulator, GeoFlareSimulator))
            time_ref = _run_time_ref(record, spec, stored_frames)
            if fm:
                # ADR-0062: every listed interval in both anchor modes; the
                # canonical self-anchored pass is the standard `result`
                # (artifacts + QoIs), the rest record field metrics only
                # (the ADR-0060 mode-2 convention). Per-channel rel-L2 is
                # recorded in BOTH modes — the AR oracle passes never had it
                # and the fleet prereg reads s_xy/peeq per interval.
                assert isinstance(simulator, TransolverSimulator)
                # Zero re-anchor events fire once the interval reaches the
                # case horizon (the re-anchor condition needs f < n_frames-1),
                # so self- and oracle-anchored rollouts coincide: run the
                # self pass once and record it under BOTH prefixes.
                horizon_case = len(trajectory.time) - model_cfg.input_frames
                result_fm: RolloutResult | None = None
                for m in fm_intervals:
                    single_anchor = m >= horizon_case
                    for mode in ("self",) if single_anchor else ("self", "oracle"):
                        canonical_pass = mode == "self" and m == fm_canonical
                        r = flow_map_rollout(
                            simulator,
                            trajectory,
                            model_cfg.input_frames,
                            time_ref,
                            m,
                            mode,
                            device,
                            qois=spec.qois if canonical_pass else None,
                            kinematic_types=spec.kinematic_types,
                            scored_frames=spec.scored_frames_for(trajectory.case_id),
                            qoi_aux_channel=qoi_aux_channel,
                        )
                        prefixes = [
                            f"rollout_m{m}"
                            if mode == "self"
                            else f"rollout_oracle_m{m}"
                        ]
                        if single_anchor:
                            prefixes.append(f"rollout_oracle_m{m}")
                        for prefix in prefixes:
                            for metric in fm_metrics:
                                fm_extra[f"{prefix}_{metric}"] = getattr(
                                    r, f"mean_{metric}"
                                )
                            if r.mean_rel_l2_aux_channels is not None:
                                fm_extra[f"{prefix}_rel_l2_aux_channels"] = dict(
                                    zip(
                                        run_aux_labels,
                                        r.mean_rel_l2_aux_channels,
                                        strict=True,
                                    )
                                )
                            if r.mean_rel_l2_state_channels is not None:
                                fm_extra[f"{prefix}_rel_l2_state_channels"] = dict(
                                    zip(
                                        state_labels,
                                        r.mean_rel_l2_state_channels,
                                        strict=True,
                                    )
                                )
                        if canonical_pass:
                            result_fm = r
                if result_fm is None:
                    raise RuntimeError(
                        f"flow-map canonical interval {fm_canonical} produced "
                        f"no self-anchored pass over intervals {fm_intervals} "
                        "— the record's flow_map_canonical_interval is not a "
                        "member (ADR-0062 load-time validation bypassed?)"
                    )
                result = result_fm
            else:
                result = time_conditioned_rollout(
                    simulator,
                    trajectory,
                    model_cfg.input_frames,
                    time_ref,
                    device,
                    qois=spec.qois,
                    kinematic_types=spec.kinematic_types,
                    scored_frames=spec.scored_frames_for(trajectory.case_id),
                    qoi_aux_channel=qoi_aux_channel,
                )
            one_step = None
            one_step_aux = None
        else:
            # ADR-0060: for an aux_input run the canonical rollout is SELF-FED
            # (the honest simulator mode; artifacts and standard metrics), and
            # a second ORACLE-STATE rollout (GT state input every step,
            # positions still self-fed) isolates the information effect from
            # accumulation through the state channel.
            if aux_in_run:
                simulator.aux_feedback = "self"
            result = rollout(
                simulator,
                trajectory,
                model_cfg.input_frames,
                device,
                qois=spec.qois,
                kinematic_types=spec.kinematic_types,
                scored_frames=spec.scored_frames_for(trajectory.case_id),
                qoi_aux_channel=qoi_aux_channel,
            )
            result_oracle = None
            if aux_in_run:
                assert mesh_sim is not None
                mesh_sim.reset_rollout()
                simulator.aux_feedback = "oracle"
                result_oracle = rollout(
                    simulator,
                    trajectory,
                    model_cfg.input_frames,
                    device,
                    # mode-2 records the four field metrics only (ADR-0060);
                    # QoIs come from the canonical self-fed rollout.
                    qois=None,
                    kinematic_types=spec.kinematic_types,
                    scored_frames=spec.scored_frames_for(trajectory.case_id),
                    qoi_aux_channel=qoi_aux_channel,
                )
            if mesh_sim is not None:
                mesh_sim.reset_rollout()
            # One-step sweeps are teacher-forced by definition: state input
            # (when any) reads ground truth (ADR-0060).
            if aux_in_run:
                simulator.aux_feedback = "oracle"
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
            if aux_in_run:
                simulator.aux_feedback = "self"
        # One-step aggregates cover the same scored span as the rollout means
        # (ADR-0035 parity, ADR-0039 horizon); per-frame arrays stay full.
        n_scored = (
            None
            if case_bound is None
            else min(case_bound, len(trajectory.time)) - model_cfg.input_frames
        )
        cases[case_id] = {
            # null one-step for a time-conditioned run (ADR-0054): the metric
            # is undefined, not zero — distinct from a finite AR value.
            "one_step_position_rmse": (
                None if one_step is None else float(one_step[:n_scored].mean())
            ),
            "one_step_aux_rmse": (
                None if one_step_aux is None else float(one_step_aux[:n_scored].mean())
            ),
            # NB: this is the StructBench registries' MEAN-OF-PER-STEP RMSE
            # (ADR-0019 SS5), NOT the paper's pooled space+time RMSE. For any
            # benchmark with a published pooled anchor (e.g. DeformingPlate's
            # 15.1+/-4.0, ADR-0043), gate/register against the pooled statistic
            # from tools/blessing_pooled_rmse.py instead - the two differ and
            # conflating them destroys comparability (the DP registry pins the
            # pooled convention for exactly this reason).
            "rollout_position_rmse": result.mean_position_rmse,
            "rollout_aux_rmse": result.mean_aux_rmse,
            # Rollout relative L2: the pooled space+time headline (ADR-0055
            # follow-up amendment). Rollout-only and pooled-only — the one-step
            # diagnostic stays RMSE (the form the GNS/MeshGraphNets lineage
            # reports), and the per-frame-mean variant was dropped (2026-08-16):
            # it blew up on zero-start aux fields and matched no reported
            # convention (the crash line uses per-timestep RMSE, served by the
            # per-frame position-RMSE diagnostic).
            "rollout_rel_l2_displacement": result.mean_rel_l2_displacement,
            "rollout_rel_l2_aux": result.mean_rel_l2_aux,
            # ADR-0059: per-channel forms, present only on C > 1 runs (keyed
            # by channel label); C = 1 records stay byte-identical.
            **(
                {}
                if result.mean_aux_rmse_channels is None
                else {
                    "rollout_aux_rmse_channels": dict(
                        zip(run_aux_labels, result.mean_aux_rmse_channels, strict=True)
                    ),
                    "rollout_rel_l2_aux_channels": dict(
                        zip(
                            run_aux_labels,
                            result.mean_rel_l2_aux_channels or (),
                            strict=True,
                        )
                    ),
                }
            ),
            # ADR-0060: oracle-state rollout metrics (GT state input each
            # step, positions self-fed) beside the canonical self-fed ones.
            **(
                {}
                if result_oracle is None
                else {
                    "rollout_oracle_position_rmse": result_oracle.mean_position_rmse,
                    "rollout_oracle_aux_rmse": result_oracle.mean_aux_rmse,
                    "rollout_oracle_rel_l2_displacement": (
                        result_oracle.mean_rel_l2_displacement
                    ),
                    "rollout_oracle_rel_l2_aux": result_oracle.mean_rel_l2_aux,
                }
            ),
            # ADR-0062: per-interval flow-map metrics (self- and oracle-
            # anchored per m), empty for every non-flow-map run.
            **fm_extra,
            # ADR-0075: the element-site state of a two-site run (the
            # canonical pass), per channel like the nodal aux block.
            **(
                {}
                if result.predicted_state is None
                else {
                    "rollout_state_rmse": result.mean_state_rmse,
                    "rollout_rel_l2_state": result.mean_rel_l2_state,
                    "rollout_state_rmse_channels": dict(
                        zip(
                            state_labels,
                            result.mean_state_rmse_channels or (),
                            strict=True,
                        )
                    ),
                    "rollout_rel_l2_state_channels": dict(
                        zip(
                            state_labels,
                            result.mean_rel_l2_state_channels or (),
                            strict=True,
                        )
                    ),
                }
            ),
            # ADR-0075 clause 6 (plan 3c): the properties block, every key the
            # benchmark defines, null where undefined on this run.
            **_case_properties(
                spec,
                result,
                one_step,
                n_scored,
                state_labels,
                case_id,
                model_cfg.input_frames,
                case_bound,
            ),
            # Full-horizon diagnostic (ADR-0039 §3): mean over every predicted
            # frame to the last stored one. Not a registry metric; equals the
            # scored value when the benchmark pins no horizon, and is null
            # when the rollout stopped at the bound (ADR-0078; ``tail`` rolls
            # on). Field name matches the 2026-07-20 bless-fleet rescore.
            "rollout_position_rmse_full": (
                float(result.position_rmse.mean())
                if len(trajectory.time) == stored_frames
                else None
            ),
            # ADR-0078: what lies at and beyond the bound, labelled unscored.
            **(
                {}
                if not tail or case_bound is None or case_bound >= stored_frames
                else {
                    "unscored_tail": _unscored_tail(
                        spec,
                        trajectory,
                        result,
                        one_step,
                        state_labels,
                        model_cfg.input_frames,
                        case_bound,
                    )
                }
            ),
            "qoi_pred": result.qoi_pred,
            "qoi_true": result.qoi_true,
            "qoi_error": result.qoi_error,
        }
        if case_id in pair_members:
            stop = len(trajectory.time) if case_bound is None else case_bound
            kept_positions[case_id] = (
                result.predicted_positions[model_cfg.input_frames : stop],
                trajectory.positions[model_cfg.input_frames : stop],
            )
        one_step_str = (
            "n/a"
            if one_step is None
            else f"{cases[case_id]['one_step_position_rmse']:.4f}"
        )
        logger.info(
            "[%s] %s: one-step %s mm | rollout %.4f mm | %s %.4f %s",
            split_name,
            case_id,
            one_step_str,
            result.mean_position_rmse,
            run_aux_labels[0] if n_aux_c > 1 else run_aux_fields[0],
            result.mean_aux_rmse,
            spec.card.aux_unit if run_aux_fields == (spec.aux_field,) else "",
        )
        if save_rollouts:
            npz_path = rollout_dir / f"{split_name}-{case_id}.npz"
            # ADR-0059 artifact policy: C = 1 keeps the historical (T, P)
            # shape so existing tooling and blessed artifacts stay readable;
            # C > 1 writes the full (T, P, C) block.
            saved_aux = result.predicted_aux
            if saved_aux.ndim == 3 and saved_aux.shape[-1] == 1:
                saved_aux = saved_aux[..., 0]
            # ADR-0075: the element-site state block of a two-site run.
            state_arrays: dict[str, Any] = {}
            if result.predicted_state is not None:
                state_arrays = {
                    "predicted_state": result.predicted_state,
                    "state_rmse": result.state_rmse,
                }
            # ADR-0078: each saved rollout carries its span, so a reader
            # (the figures) needs no benchmark to find the bound. The arrays
            # hold frames [0, saved); the scored span is
            # [input_frames, scored_frames). -1 = no bound, scored to the end.
            span_arrays: dict[str, Any] = {
                "input_frames": np.int64(model_cfg.input_frames),
                "scored_frames": np.int64(-1 if case_bound is None else case_bound),
                "stored_frames": np.int64(stored_frames),
            }
            if one_step is None:
                # Time-conditioned: no one-step arrays to persist (null; ADR-0054).
                np.savez(
                    npz_path,
                    predicted_positions=result.predicted_positions,
                    predicted_aux=saved_aux,
                    position_rmse=result.position_rmse,
                    aux_rmse=result.aux_rmse,
                    **state_arrays,
                    **span_arrays,
                )
            else:
                assert one_step_aux is not None  # set together with one_step
                np.savez(
                    npz_path,
                    predicted_positions=result.predicted_positions,
                    predicted_aux=saved_aux,
                    position_rmse=result.position_rmse,
                    aux_rmse=result.aux_rmse,
                    one_step_position_rmse=one_step,
                    one_step_aux_rmse=one_step_aux,
                    **state_arrays,
                    **span_arrays,
                )

    # ADR-0075 clause 6: state sufficiency of each declared pair whose two
    # cases were both evaluated here, written on both; the rest stay null.
    for a, b in spec.matched_initial_state_pairs:
        if a in kept_positions and b in kept_positions:
            value = _defined(
                state_sufficiency(
                    kept_positions[a][0],
                    kept_positions[b][0],
                    kept_positions[a][1],
                    kept_positions[b][1],
                )
            )
            cases[a]["prop_state_sufficiency"] = value
            cases[b]["prop_state_sufficiency"] = value

    def _mean_over_cases(key: str) -> float | None:
        # A time-conditioned run's one_step_* is null per case; its mean is
        # likewise null (ADR-0054), never a NaN or a spurious 0.
        vals = [case[key] for case in cases.values() if case[key] is not None]
        return float(np.mean(vals)) if vals else None

    # ADR-0078: the split mean of each unscored-tail number, over the cases
    # that have a tail (a case scored to its last stored frame has none).
    tails = [
        case["unscored_tail"] for case in cases.values() if "unscored_tail" in case
    ]
    tail_mean: dict[str, float | None] = {}
    for key in _UNSCORED_TAIL_KEYS:
        vals = [t[key] for t in tails if t.get(key) is not None]
        if any(key in t for t in tails):
            tail_mean[key] = float(np.mean(vals)) if vals else None

    metrics: dict[str, Any] = {
        "split": split_name,
        "checkpoint": ckpt_path.name,
        # Full resolved path so an explicitly scored checkpoint (possibly from
        # outside out_dir, via an absolute --checkpoint) stays traceable.
        "checkpoint_path": str(ckpt_path),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "input_frames": model_cfg.input_frames,
        # Metric definition (ADR-0039): rollout/one-step means and QoIs cover
        # frames [input_frames, scored_frames); null means scored to the end,
        # "per_case" that each case has its own bound (ADR-0039 note,
        # 2026-11-06). Records with different scored_frames are not comparable.
        "scored_frames": _scored_frames_record(spec),
        # ADR-0078: the span the run was trained on ("scored" or "full"; a
        # record from before the setting trained on the full record, cut by
        # its train_frames) and that fixed cut. Records trained on different
        # spans are not comparable either.
        "training_span": record["train"].get("training_span", "full"),
        "train_frames": record["train"].get("train_frames", 0),
        # Whether the rollouts ran past the scored bound (the unscored_tail
        # blocks). The scored numbers do not depend on it.
        "tail": bool(tail),
        # Card-conforming by construction: a checkpoint's input_frames is
        # validated equal to the card's at config load and train (ADR-0035),
        # so a standard run stays standard on re-eval. Legacy off-card records
        # (e.g. a pre-0035 window=11 run re-evaluated here) read as non-standard.
        "protocol_standard": bool(record["protocol"].get("standard", True))
        and model_cfg.input_frames == spec.card.input_frames,
        # ADR-0059: the run's actual aux selection. The benchmark's canonical
        # single field keeps the historical singular string keys; variant
        # selections record per-channel label/unit lists.
        **(
            {"aux_field": spec.aux_field, "aux_unit": spec.card.aux_unit}
            if run_aux_fields == (spec.aux_field,)
            else {
                "aux_field": list(run_aux_labels),
                "aux_unit": list(aux_channel_units(run_aux_fields)),
                "aux_fields": list(run_aux_fields),
            }
        ),
        "cases": cases,
        "mean": {
            "one_step_position_rmse": _mean_over_cases("one_step_position_rmse"),
            "one_step_aux_rmse": _mean_over_cases("one_step_aux_rmse"),
            "rollout_position_rmse": _mean_over_cases("rollout_position_rmse"),
            "rollout_aux_rmse": _mean_over_cases("rollout_aux_rmse"),
            "rollout_rel_l2_displacement": _mean_over_cases(
                "rollout_rel_l2_displacement"
            ),
            "rollout_rel_l2_aux": _mean_over_cases("rollout_rel_l2_aux"),
            # ADR-0075: split means of the element-state scalars.
            **(
                {
                    "rollout_state_rmse": _mean_over_cases("rollout_state_rmse"),
                    "rollout_rel_l2_state": _mean_over_cases("rollout_rel_l2_state"),
                }
                if two_site_run
                else {}
            ),
            # A TC/flow-map aux_input run computes no ADR-0060 oracle
            # rollout (its accumulation isolation is the per-interval
            # oracle-ANCHORED mode below), so the per-case dicts carry
            # no rollout_oracle_* keys to aggregate.
            **(
                {}
                if not aux_in_run or tc
                else {
                    k: _mean_over_cases(k)
                    for k in (
                        "rollout_oracle_position_rmse",
                        "rollout_oracle_aux_rmse",
                        "rollout_oracle_rel_l2_displacement",
                        "rollout_oracle_rel_l2_aux",
                    )
                }
            ),
            # ADR-0062: split means of the per-interval scalar metrics
            # (the per-channel dicts stay per-case, house convention).
            **(
                {}
                if not fm
                else {
                    f"{prefix}_{metric}": _mean_over_cases(f"{prefix}_{metric}")
                    for m in fm_intervals
                    for prefix in (f"rollout_m{m}", f"rollout_oracle_m{m}")
                    for metric in fm_metrics
                }
            ),
            "rollout_position_rmse_full": _mean_over_cases(
                "rollout_position_rmse_full"
            ),
            **({"unscored_tail": tail_mean} if tail_mean else {}),
            # ADR-0075 clause 6: the split means of the properties block.
            **{key: _mean_over_cases(key) for key in spec.defined_properties},
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
