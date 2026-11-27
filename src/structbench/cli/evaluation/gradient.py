"""The ``gradient`` mode (ADR-0076)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ...datasets import (
    load_case_trajectory,
)
from ...eval import (
    QoiInputs,
    gradient_floor,
    gradient_metrics,
    model_sensitivity,
    reference_sensitivities,
)
from ...models.common import CaseBoundSimulator
from ..training.common import _state_fields_for
from .run import _json_safe, _predict_case, _rebuild_run_simulator

logger = logging.getLogger(__name__)


def evaluate_gradient(
    data_root: Path,
    out_dir: Path,
    device: str,
    *,
    case_ids: list[str] | None = None,
    save_artifacts: bool = True,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """The gradient task family (ADR-0076): the run's sensitivities against
    the sweep's own differences.

    For every case on the benchmark's sweep (``case_ids`` narrows it, for a
    smoke or a line), the model is rolled out from seeds the family
    constructs at ``v ± δ`` for both declared deltas (a model that takes the
    parameter as its scalar loading input is given ``v ± δ`` there too, when
    the family declares the parameter as that scalar), its quantities of
    interest are differenced (:func:`~structbench.eval.sensitivity.model_sensitivity`),
    and the disagreement of the two deltas is its floor. The reference
    sensitivities are recomputed here from the stored cases with the same
    estimator the committed record uses (``docs/gradients/``), so a report
    never depends on a file outside the archive. Per quantity, the metrics
    pool the judged points of the scored sets (every split but ``train`` and
    ``val``) as the headline and every judged point as a diagnostic.

    Returns
    -------
    dict
        ``{"benchmark", "parameter", "deltas", "checkpoint",
        "loading_scalar_perturbed", "lines": {line:
        {"values", "cases", "sets", "reference": {qoi: [...]}, "model": {qoi:
        [...]}, "floor": {qoi: [...]}}}, "metrics": {...}, "metrics_all":
        {...}}``; written to ``out_dir/metrics-gradient.json`` when
        ``save_artifacts`` (suffixed by the checkpoint stem when one is
        named).

    Raises
    ------
    ValueError
        If the benchmark declares no gradient family, or the family has no
        seed construction.
    """
    run = _rebuild_run_simulator(out_dir, device, checkpoint)
    spec, model_cfg = run.spec, run.model_cfg
    family = spec.gradient_family
    if family is None:
        raise ValueError(f"benchmark {spec.card.name!r} declares no gradient family")
    if family.perturb_seed is None:
        raise ValueError(
            f"benchmark {spec.card.name!r} gradient family has no seed construction"
        )
    mesh_sim = isinstance(run.simulator, CaseBoundSimulator)
    all_cases = [c for s in spec.splits.values() for c in s]
    set_of = {c: s for s, cs in spec.splits.items() for c in cs}
    chosen = list(case_ids) if case_ids is not None else all_cases
    unknown = sorted(set(chosen) - set(all_cases))
    if unknown:
        raise ValueError(f"cases not on the benchmark's sweep: {unknown}")
    qoi_names = tuple(family.noise)
    scored_sets = {s for s in spec.splits if s not in ("train", "val")}

    # Per case: the reference quantities from the stored trajectory and the
    # model's quantities at the perturbed seeds.
    ref_q: dict[str, dict[str, float]] = {}
    model_s: dict[str, dict[str, dict[float, float]]] = {}
    for case_id in chosen:
        trajectory = load_case_trajectory(
            data_root / f"{case_id}.h5",
            aux_field=run.aux_fields,
            state_fields=_state_fields_for(run.family, model_cfg, spec),
        )
        value = family.parameter_of(case_id)
        gt = QoiInputs(
            time=trajectory.time,
            positions=trajectory.positions,
            aux=trajectory.aux[..., run.qoi_aux_channel or 0],
            particle_type=trajectory.particle_type,
            init=model_cfg.input_frames,
            case_id=case_id,
        )
        ref_q[case_id] = {name: float(spec.qois[name](gt)) for name in qoi_names}
        model_s[case_id] = {name: {} for name in qoi_names}
        for delta in family.deltas:
            q_at: dict[float, dict[str, float]] = {}
            for signed in (-delta, delta):
                seed = family.perturb_seed(
                    trajectory, value, value + signed, model_cfg.input_frames
                )
                if mesh_sim and spec.mesh_transform is not None:
                    seed = spec.mesh_transform(seed)
                result = _predict_case(
                    run,
                    seed,
                    case_id,
                    device,
                    loading_value=(
                        value + signed if family.parameter_is_loading_scalar else None
                    ),
                )
                q_at[signed] = {
                    name: float(result.qoi_pred[name]) for name in qoi_names
                }
            for name in qoi_names:
                model_s[case_id][name][delta] = model_sensitivity(
                    q_at[-delta][name], q_at[delta][name], delta
                )
        logger.info("gradient: %s done", case_id)

    # Lines: reference sensitivities need the whole line, so they come from
    # every stored case on it (the archive), model values where computed.
    lines_out: dict[str, Any] = {}
    pooled: dict[str, dict[str, list[Any]]] = {
        name: {
            "ref": [],
            "model": [],
            "floor": [],
            "ref_all": [],
            "model_all": [],
            "floor_all": [],
        }
        for name in qoi_names
    }
    by_line: dict[str, list[str]] = {}
    for c in all_cases:
        by_line.setdefault(family.line_of(c), []).append(c)
    d_small, d_large = family.deltas
    for line, cids in sorted(by_line.items()):
        cids = sorted(cids, key=family.parameter_of)
        if not any(c in model_s for c in cids):
            continue
        values = [family.parameter_of(c) for c in cids]
        line_ref: dict[str, dict[str, float]] = {}
        for c in cids:
            if c in ref_q:
                line_ref[c] = ref_q[c]
            else:
                traj = load_case_trajectory(
                    data_root / f"{c}.h5", aux_field=run.aux_fields
                )
                gt = QoiInputs(
                    time=traj.time,
                    positions=traj.positions,
                    aux=traj.aux[..., run.qoi_aux_channel or 0],
                    particle_type=traj.particle_type,
                    init=model_cfg.input_frames,
                    case_id=c,
                )
                line_ref[c] = {name: float(spec.qois[name](gt)) for name in qoi_names}
        entry: dict[str, Any] = {
            "cases": cids,
            "values": values,
            "sets": [set_of[c] for c in cids],
            "reference": {},
            "model": {},
            "floor": {},
        }
        for name in qoi_names:
            pts = reference_sensitivities(
                values, [line_ref[c][name] for c in cids], noise=family.noise[name]
            )
            entry["reference"][name] = [asdict(p) | {"judged": p.judged} for p in pts]
            model_vals: list[float | None] = []
            floor_vals: list[float | None] = []
            for c, p in zip(cids, pts, strict=True):
                if c in model_s:
                    s1 = model_s[c][name][d_small]
                    s2 = model_s[c][name][d_large]
                    model_vals.append(s1)
                    floor_vals.append(gradient_floor(s1, s2))
                    pooled[name]["ref_all"].append(p)
                    pooled[name]["model_all"].append(s1)
                    pooled[name]["floor_all"].append(gradient_floor(s1, s2))
                    if set_of[c] in scored_sets:
                        pooled[name]["ref"].append(p)
                        pooled[name]["model"].append(s1)
                        pooled[name]["floor"].append(gradient_floor(s1, s2))
                else:
                    model_vals.append(None)
                    floor_vals.append(None)
            entry["model"][name] = model_vals
            entry["floor"][name] = floor_vals
        lines_out[line] = entry

    def _metrics(suffix: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in qoi_names:
            ref = pooled[name]["ref" + suffix]
            if not ref:
                continue
            m = gradient_metrics(
                ref, pooled[name]["model" + suffix], pooled[name]["floor" + suffix]
            )
            out[f"gradient_rel_l2_{name}"] = m["rel_l2"]
            out[f"gradient_sign_{name}"] = m["sign"]
            out[f"gradient_floor_{name}"] = m["floor"]
            out[f"gradient_n_judged_{name}"] = m["n_judged"]
            out[f"gradient_n_reported_{name}"] = m["n_reported"]
        return out

    report: dict[str, Any] = {
        "benchmark": spec.card.name,
        "parameter": family.parameter,
        "deltas": list(family.deltas),
        "checkpoint": run.ckpt_path.name,
        "checkpoint_path": str(run.ckpt_path),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "scored_sets": sorted(scored_sets),
        "loading_scalar_perturbed": bool(
            family.parameter_is_loading_scalar
            and getattr(model_cfg, "impact_velocity_feature", False)
            and spec.loading_scalar is not None
        ),
        "lines": lines_out,
        "metrics": _metrics(""),
        "metrics_all": _metrics("_all"),
    }
    if save_artifacts:
        tag = "gradient" if checkpoint is None else f"gradient@{run.ckpt_path.stem}"
        (out_dir / f"metrics-{tag}.json").write_text(
            json.dumps(_json_safe(report), indent=2), encoding="utf-8"
        )
    return report
