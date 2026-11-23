"""The ``restart`` mode (ADR-0075 clause 5)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...config import (
    TransolverConfig,
)
from ...datasets import (
    CaseTrajectory,
    load_case_trajectory,
)
from ...eval import (
    RolloutResult,
    flow_map_rollout,
    rollout,
)
from ...eval.restart import (
    RESTART_FIELD_METRICS,
    RESTART_STATE_METRICS,
    compose_keys,
    restart_keys,
    restart_window,
    semigroup_error,
)
from ...models.common import CaseBoundSimulator
from ...models.transolver import (
    TransolverSimulator,
)
from ..training.common import (
    _case_law_rows,
    _fm_intervals_and_canonical,
    _state_fields_for,
)
from .rollout import _defined
from .run import _json_safe, _rebuild_run_simulator, _run_time_ref

logger = logging.getLogger(__name__)


def evaluate_restart(
    case_ids: list[str],
    data_root: Path,
    out_dir: Path,
    device: str,
    *,
    split_name: str = "eval",
    save_artifacts: bool = True,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """The restart task family (ADR-0075 clause 5; plan 3b) of a run.

    For every case, each declared anchor and horizon
    (:meth:`BenchmarkSpec.restart_anchors_for`, ``restart_horizons``) is a
    rollout on the window of the trajectory from the anchor's history to the
    end of the horizon (:func:`~structbench.eval.restart.restart_window`),
    re-bound to the simulator, scored over every predicted frame and keyed
    ``restart_a{a}_h{h}_<metric>``. A two-site run restarts from the stored
    element state the window carries (``restart_kind = "state"``); every
    other scheme restarts from the kinematic anchor alone (``"kinematic"``).
    A flow-map run also runs the composition sweep, each horizon ``T``
    reached self-anchored in ``k`` steps of ``T / k`` frames for every
    declared ``k`` that divides ``T`` (``compose_T{T}_k{k}_<metric>``), and
    reports the semigroup error between the two-step composition and the
    direct map as ``prop_semigroup_err``, a property of the block.

    The report records ``train_max_dt``, the longest interval the run trained
    on (the flow map's ``flow_map_max_dt``; ``None`` when uncapped, every
    interval to the horizon; ``1`` for an autoregressive scheme's unit steps)
    and the flow map's evaluation intervals, so a reader sees which horizons
    are held out.
    Written to ``out_dir/metrics-restart-<split_name>.json`` when
    ``save_artifacts`` (checkpoint-suffixed like :func:`evaluate`).

    Parameters
    ----------
    case_ids : list of str
        Cases to restart over; must be non-empty.
    data_root : pathlib.Path
        Directory containing ``<case_id>.h5`` canonical cases.
    out_dir : pathlib.Path
        Run directory holding the checkpoint and resolved config.
    device : str
        Torch device string.
    split_name : str
        Label recorded in the report and used in the artifact filename.
    save_artifacts : bool
        Write the report JSON.
    checkpoint : str, pathlib.Path or None
        Explicit checkpoint to evaluate, as in :func:`evaluate`.

    Returns
    -------
    dict
        ``{"split", "checkpoint", "checkpoint_path", "input_frames",
        "restart_kind", "horizons", "compositions", "anchors", "train_max_dt",
        "eval_intervals", "cases", "mean", "mean_n"}``, JSON-serialisable;
        ``mean_n`` counts the cases behind each mean (anchors may differ per
        case).

    Raises
    ------
    ValueError
        If ``case_ids`` is empty, the benchmark declares no restarts, the
        run is time-conditioned without the flow map (ADR-0054: it carries
        nothing between frames, so there is nothing to restart), or a case's
        stored trajectory is shorter than its card admits.
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
    family = run.family
    if not spec.defines_restarts:
        raise ValueError(
            f"benchmark {spec.card.name!r} declares no restarts (ADR-0075 clause 5)"
        )
    tc = getattr(model_cfg, "time_conditioned", False)
    fm = family == "transolver" and bool(getattr(model_cfg, "flow_map", False))
    if tc and not fm:
        raise ValueError(
            "a time-conditioned run carries nothing between frames (ADR-0054), so "
            "there is nothing to restart from; the restart family is for "
            "autoregressive and flow-map schemes"
        )
    state_fields = _state_fields_for(family, model_cfg, spec)
    two_site_run = state_fields is not None
    restart_kind = "state" if two_site_run else "kinematic"
    mesh_sim = simulator if isinstance(simulator, CaseBoundSimulator) else None
    fm_intervals, fm_canonical = _fm_intervals_and_canonical(model_cfg)
    metric_names = RESTART_FIELD_METRICS + (
        RESTART_STATE_METRICS if two_site_run else ()
    )
    input_frames = model_cfg.input_frames
    horizons = spec.restart_horizons

    def _bind(window: CaseTrajectory) -> None:
        """Bind a case-bound simulator to the window (its frame indices are
        the window's) and reset the rollout pointer."""
        if mesh_sim is None:
            return
        assert window.cells is not None and window.reference_coords is not None
        two_site_bind: dict[str, Any] = {}
        if two_site_run:
            two_site_bind = {
                "gt_state": (
                    None
                    if window.state is None
                    else torch.from_numpy(window.state).to(device)
                ),
                "law": _case_law_rows(spec, window, device),
            }
        mesh_sim.bind_case(
            torch.from_numpy(window.cells).to(device),
            torch.from_numpy(window.reference_coords).to(device),
            torch.from_numpy(window.particle_type).to(device),
            torch.from_numpy(window.positions).to(device),
            loading_scalar=(
                spec.loading_scalar(window.case_id)
                if getattr(model_cfg, "impact_velocity_feature", False)
                and spec.loading_scalar
                else None
            ),
            gt_aux=(
                torch.from_numpy(window.aux).to(device) if run.aux_in_run else None
            ),
            **two_site_bind,
        )
        mesh_sim.reset_rollout()

    def _roll(
        window: CaseTrajectory, offset: int, interval: int, n_case_frames: int
    ) -> RolloutResult:
        _bind(window)
        if fm:
            assert isinstance(simulator, TransolverSimulator)
            # The CASE's time reference, as training and evaluate() use it;
            # the window's own length would renormalise time per restart.
            time_ref = _run_time_ref(record, spec, n_case_frames)
            return flow_map_rollout(
                simulator,
                window,
                input_frames,
                time_ref,
                interval,
                "self",
                device,
                qois=None,
                kinematic_types=spec.kinematic_types,
                qoi_aux_channel=run.qoi_aux_channel,
                frame_offset=offset,
            )
        if run.aux_in_run:
            simulator.aux_feedback = "self"  # the deployable mode (ADR-0060)
        return rollout(
            simulator,
            window,
            input_frames,
            device,
            qois=None,
            kinematic_types=spec.kinematic_types,
            qoi_aux_channel=run.qoi_aux_channel,
        )

    cases: dict[str, dict[str, Any]] = {}
    anchors_record: dict[str, list[int]] = {}
    for case_id in case_ids:
        trajectory = load_case_trajectory(
            data_root / f"{case_id}.h5",
            aux_field=run.aux_fields,
            state_fields=state_fields,
        )
        if mesh_sim is not None and spec.mesh_transform is not None:
            trajectory = spec.mesh_transform(trajectory)
        anchors = spec.restart_anchors_for(case_id)
        anchors_record[case_id] = list(anchors)
        n_case_frames = len(trajectory.time)
        row: dict[str, Any] = {}
        semigroup: list[float] = []
        for anchor in anchors:
            for horizon in horizons:
                try:
                    window = restart_window(trajectory, anchor, horizon, input_frames)
                except ValueError as exc:
                    # The card admitted the anchor; the stored trajectory is
                    # shorter (an SPH case drops its terminal frame, ADR-0028).
                    raise ValueError(
                        f"case {case_id!r} ({len(trajectory.time)} stored frames, "
                        f"card n_frames {spec.card.n_frames}): {exc}"
                    ) from exc
                offset = anchor - input_frames + 1
                direct = _roll(window, offset, fm_canonical if fm else 1, n_case_frames)
                for name, key in zip(
                    metric_names,
                    restart_keys(anchor, horizon, metric_names),
                    strict=True,
                ):
                    row[key] = getattr(direct, f"mean_{name}")
                if not fm:
                    continue
                by_k: dict[int, RolloutResult] = {}
                for k in spec.restart_compositions:
                    if horizon % k:
                        continue
                    by_k[k] = _roll(window, offset, horizon // k, n_case_frames)
                    for name, key in zip(
                        metric_names,
                        compose_keys(horizon, k, metric_names),
                        strict=True,
                    ):
                        row[key] = getattr(by_k[k], f"mean_{name}")
                if 1 in by_k and 2 in by_k:
                    # The two maps share the anchor and the first T/2 queries,
                    # which are identical by construction: compare the frames
                    # after the re-anchor, as displacements from the case's
                    # frame 0 (design §6; ADR-0055).
                    composed_from = input_frames + horizon // 2
                    semigroup.append(
                        semigroup_error(
                            by_k[1].predicted_positions[composed_from:],
                            by_k[2].predicted_positions[composed_from:],
                            reference=trajectory.positions[0],
                        )
                    )
        if fm:
            row["prop_semigroup_err"] = (
                _defined(float(np.mean(semigroup))) if semigroup else None
            )
        cases[case_id] = row
        logger.info(
            "[restart:%s] %s: %d anchors x %d horizons (%s)",
            split_name,
            case_id,
            len(anchors),
            len(horizons),
            restart_kind,
        )

    keys: list[str] = []
    for row in cases.values():
        keys += [k for k in row if k not in keys]
    # Per-case anchors differ, so each mean says how many cases it pools.
    values_by_key = {
        k: [r[k] for r in cases.values() if r.get(k) is not None] for k in keys
    }
    mean = {k: float(np.mean(v)) if v else None for k, v in values_by_key.items()}
    mean_n = {k: len(v) for k, v in values_by_key.items()}
    report: dict[str, Any] = {
        "split": split_name,
        "checkpoint": run.ckpt_path.name,
        "checkpoint_path": str(run.ckpt_path),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "input_frames": input_frames,
        "restart_kind": restart_kind,
        "horizons": list(horizons),
        "compositions": list(spec.restart_compositions) if fm else [],
        "anchors": anchors_record,
        # The longest interval the run trained on: the flow map's cap, None
        # when uncapped (flow_map_max_dt 0 trains every interval to the
        # horizon); an autoregressive scheme trains on unit steps.
        "train_max_dt": (
            (int(model_cfg.flow_map_max_dt) or None)
            if fm and isinstance(model_cfg, TransolverConfig)
            else 1
        ),
        "eval_intervals": list(fm_intervals) if fm else [],
        "cases": cases,
        "mean": mean,
        "mean_n": mean_n,
    }
    if save_artifacts:
        tag = split_name if checkpoint is None else f"{split_name}@{run.ckpt_path.stem}"
        (out_dir / f"metrics-restart-{tag}.json").write_text(
            json.dumps(_json_safe(report), indent=2, allow_nan=False),
            encoding="utf-8",
        )
    return report
