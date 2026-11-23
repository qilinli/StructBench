"""Rebuilding a trained run for evaluation: the checkpoint, the simulator
and one case's prediction."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...benchmarks import (
    BenchmarkSpec,
    get_benchmark,
)
from ...config import (
    MODEL_FAMILIES,
    CGNConfig,
    GeoFlareConfig,
    MGNConfig,
    TransolverConfig,
    read_run_record,
)
from ...datasets import (
    CaseTrajectory,
    NormalizationStats,
    aux_channel_labels,
)
from ...eval import (
    RolloutResult,
    flow_map_rollout,
    rollout,
    time_conditioned_rollout,
)
from ...models.cgn import LearnedSimulator
from ...models.common import CaseBoundSimulator
from ...models.geoflare import GeoFlareSimulator
from ...models.mgn import (
    MeshSimulator,
)
from ...models.transolver import (
    TransolverSimulator,
)
from ..training.builders import (
    build_geoflare_simulator,
    build_mgn_simulator,
    build_simulator,
    build_transolver_simulator,
)
from ..training.common import (
    _bind_boundary_feature,
    _case_law_rows,
    _fm_intervals_and_canonical,
    _state_fields_for,
    _stats_to_dict,
    _tc_time_ref_frames,
)

logger = logging.getLogger(__name__)


def _run_time_ref(
    record: Mapping[str, Any], spec: BenchmarkSpec, stored_frames: int
) -> int:
    """The time reference a run was trained on, for evaluating it.

    Read from the run record where it is written (ADR-0078). A record from
    before that carries none and is evaluated by the rule it was trained
    under: the fixed scored bound, else its ``train_frames``, else the
    case's stored frame count, which must be passed uncut.
    """
    recorded = record.get("run", {}).get("time_ref_frames")
    if recorded is not None:
        return int(recorded)
    return _tc_time_ref_frames(
        spec.scored_frames_fixed,
        record["train"].get("train_frames", 0),
        stored_frames,
    )


def _find_checkpoint(out_dir: Path) -> Path | None:
    """Return the highest-step ``model-*.pt`` in ``out_dir``.

    Selection is by the step number embedded in the (zero-padded) filename,
    not filesystem mtime, so a run directory whose mtimes were scrambled by a
    copy or transfer still resolves to the latest (best) checkpoint. Periodic
    ``ckpt-*.pt`` snapshots are deliberately outside the glob: default
    evaluation always scores the run's selected (best/final) checkpoint.
    """

    def _step(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    checkpoints = sorted(out_dir.glob("model-*.pt"), key=_step)
    return checkpoints[-1] if checkpoints else None


def _json_safe(obj: Any) -> Any:
    """Recursively map non-finite floats to ``None`` for strict JSON output.

    A diverged rollout can yield NaN/Inf metrics; the default ``json.dumps``
    emits bare ``NaN``/``Infinity`` tokens that strict JSON parsers reject, so
    the run directory's evidence files would be unreadable exactly when a run
    misbehaves.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _resolve_run_spec(out_dir: Path) -> tuple[BenchmarkSpec, dict[str, Any]]:
    """Resolve the run directory's benchmark spec and resolved config.

    Parameters
    ----------
    out_dir : pathlib.Path
        Run directory holding ``config.json``.

    Returns
    -------
    tuple of (BenchmarkSpec, dict)
        The benchmark spec and the run record, normalized to the nested
        ADR-0032 shape (pre-0032 flat records are adapted by
        :func:`structbench.config.read_run_record`).

    Raises
    ------
    FileNotFoundError
        If ``config.json`` is missing from ``out_dir``.
    """
    record = read_run_record(out_dir / "config.json")
    return get_benchmark(record["run"]["benchmark"]), record


def _model_config_from_record(
    record: dict[str, Any],
) -> CGNConfig | MGNConfig | TransolverConfig | GeoFlareConfig:
    """Reconstruct the family-appropriate model config from a run record.

    Parameters
    ----------
    record : dict
        A run record as returned by
        :func:`~structbench.config.read_run_record`, whose ``"model"``
        section carries ``"family"`` plus every field of that family's
        config dataclass (:data:`~structbench.config.MODEL_FAMILIES`).

    Returns
    -------
    CGNConfig, MGNConfig, TransolverConfig, or GeoFlareConfig
        Constructed from ``record["model"]`` with ``"family"`` dropped
        before the fields are splatted in. The legacy ``"gns"`` alias
        (pre-ADR-0034 records) resolves to :class:`CGNConfig` like ``"cgn"``.
    """
    model_table = {k: v for k, v in record["model"].items() if k != "family"}
    # The JSON round-trip stores tuple-typed fields as lists; restore the
    # canonical tuple form (the ADR-0059 per-channel knobs and the ADR-0062
    # interval sweep) so a reconstructed config compares and reads uniformly.
    for key in (
        "aux_transform",
        "aux_transform_scale",
        "aux_input_noise_std",
        "flow_map_eval_intervals",
    ):
        if isinstance(model_table.get(key), list):
            model_table[key] = tuple(model_table[key])
    model_cls = MODEL_FAMILIES[record["model"]["family"]]
    return model_cls(**model_table)


@dataclass(frozen=True)
class _RunSimulator:
    """A run directory's simulator, rebuilt from its own record, with the
    run-level facts :func:`evaluate` and :func:`evaluate_gradient` share."""

    spec: BenchmarkSpec
    record: dict[str, Any]
    model_cfg: CGNConfig | MGNConfig | TransolverConfig | GeoFlareConfig
    family: str
    simulator: (
        LearnedSimulator | MeshSimulator | TransolverSimulator | GeoFlareSimulator
    )
    ckpt_path: Path
    aux_fields: tuple[str, ...]
    aux_labels: tuple[str, ...]
    aux_in_run: bool
    qoi_aux_channel: int | None


def _rebuild_run_simulator(
    out_dir: Path, device: str, checkpoint: str | Path | None
) -> _RunSimulator:
    """Rebuild a run's simulator entirely from its run directory.

    Architecture and ``n_particle_types`` come from ``config.json``, the
    family from ``config.json["model"]["family"]``; a ``cgn`` run also reads
    ``normalization_stats.npz``, the other families carry their normalizers
    in the checkpoint. The checkpoint is the run's selected one unless
    ``checkpoint`` names a file (a relative path resolves against ``out_dir``
    only, never the CWD: fleet arms hold identically named snapshots).
    Loaded, moved to ``device`` and in eval mode on return.
    """
    spec, record = _resolve_run_spec(out_dir)
    family = record["model"]["family"]
    model_cfg = _model_config_from_record(record)
    n_types = int(record["n_particle_types"])
    # ADR-0059: the run's aux channel selection, recorded at train time
    # (train.aux_fields); legacy records carry none and mean the benchmark's
    # canonical declaration.
    run_aux_fields = tuple(record.get("train", {}).get("aux_fields") or ()) or (
        spec.aux_field,
    )
    run_aux_labels = aux_channel_labels(run_aux_fields)
    n_aux_c = len(run_aux_labels)
    # ADR-0060: state-feedback runs get the oracle-state rollout alongside
    # the canonical self-fed one, and oracle-mode one-step sweeps.
    aux_in_run = family == "transolver" and bool(getattr(model_cfg, "aux_input", False))
    # ADR-0059: aux QoIs bind the benchmark's headline channel BY NAME within
    # the run's selection. A variant selection that lacks it gets NaN aux
    # QoIs (loud) rather than channel-0 values of a different quantity.
    _headline = aux_channel_labels((spec.aux_field,))[0]
    qoi_aux_channel: int | None
    try:
        qoi_aux_channel = run_aux_labels.index(_headline)
    except ValueError:
        qoi_aux_channel = None
        logger.warning(
            "run aux selection %r carries no %r channel; aux QoIs will be NaN",
            run_aux_labels,
            _headline,
        )

    simulator: (
        LearnedSimulator | MeshSimulator | TransolverSimulator | GeoFlareSimulator
    )
    if family == "mgn":
        assert isinstance(model_cfg, MGNConfig)
        simulator = build_mgn_simulator(
            model_cfg,
            n_aux=n_aux_c,
            kinematic_types=spec.kinematic_types,
            scripted_types=spec.scripted_types,
            device=device,
        )
    elif family == "transolver":
        assert isinstance(model_cfg, TransolverConfig)
        simulator = build_transolver_simulator(
            model_cfg,
            n_aux=n_aux_c,
            kinematic_types=spec.kinematic_types,
            scripted_types=spec.scripted_types,
            hardening_curve=spec.hardening_curve,
            element_formulation=spec.element_formulation,
            device=device,
        )
    elif family == "geoflare":
        assert isinstance(model_cfg, GeoFlareConfig)
        simulator = build_geoflare_simulator(
            model_cfg,
            n_aux=n_aux_c,
            kinematic_types=spec.kinematic_types,
            scripted_types=spec.scripted_types,
            device=device,
        )
    else:
        stats_path = out_dir / "normalization_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(f"missing normalization stats: {stats_path}")
        stats = NormalizationStats.load(stats_path)
        assert isinstance(model_cfg, CGNConfig)
        simulator = build_simulator(
            _stats_to_dict(stats),
            model_cfg,
            n_particle_types=n_types,
            boundary_feature_fn=_bind_boundary_feature(spec, model_cfg),
            device=device,
        )
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        # Relative paths resolve against out_dir ONLY (never the CWD): fleet
        # arms all hold identically named ckpt-<step>.pt snapshots, so a CWD
        # fallback would silently score another arm's weights.
        if not ckpt_path.is_absolute():
            ckpt_path = out_dir / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    else:
        found = _find_checkpoint(out_dir)
        if found is None:
            raise FileNotFoundError(f"no checkpoint found under {out_dir}")
        ckpt_path = found
    simulator.load(str(ckpt_path))
    simulator.to(device)
    simulator.eval()
    return _RunSimulator(
        spec=spec,
        record=record,
        model_cfg=model_cfg,
        family=family,
        simulator=simulator,
        ckpt_path=ckpt_path,
        aux_fields=run_aux_fields,
        aux_labels=run_aux_labels,
        aux_in_run=aux_in_run,
        qoi_aux_channel=qoi_aux_channel,
    )


def _predict_case(
    run: _RunSimulator,
    trajectory: CaseTrajectory,
    case_id: str,
    device: str,
    *,
    loading_value: float | None = None,
) -> RolloutResult:
    """The run's canonical prediction of one case under the benchmark's
    protocol: the self-fed rollout, or the time-conditioned query, or the
    anchored flow map's canonical self-anchored pass (ADR-0054, ADR-0062).

    Mirrors the canonical pass of :func:`evaluate` without its diagnostics
    (oracle rollouts, one-step sweeps, per-interval flow-map metrics), for
    callers that need the predicted quantities of interest alone — the
    gradient task family (ADR-0076). ``trajectory`` is the case as the model
    sees it (mesh-transformed where the family needs it); the stored frames
    past the seed are not scored here, so a perturbed seed is legitimate.
    ``loading_value`` replaces the case's scalar loading parameter for a
    model that takes one (``None``: the case's own value).
    """
    spec, model_cfg, simulator = run.spec, run.model_cfg, run.simulator
    mesh_sim = simulator if isinstance(simulator, CaseBoundSimulator) else None
    if mesh_sim is not None:
        if trajectory.cells is None or trajectory.reference_coords is None:
            raise ValueError(
                f"benchmark {spec.card.name!r} has no mesh connectivity "
                f"(cells/reference_coords); {run.family} evaluation requires "
                "a mesh benchmark"
            )
        two_site_bind: dict[str, Any] = {}
        if _state_fields_for(run.family, model_cfg, spec) is not None:
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
                (
                    spec.loading_scalar(case_id)
                    if loading_value is None
                    else float(loading_value)
                )
                if getattr(model_cfg, "impact_velocity_feature", False)
                and spec.loading_scalar
                else None
            ),
            gt_aux=(
                torch.from_numpy(trajectory.aux).to(device) if run.aux_in_run else None
            ),
            **two_site_bind,
        )
        mesh_sim.reset_rollout()
    scored = spec.scored_frames_for(trajectory.case_id)
    if getattr(model_cfg, "time_conditioned", False):
        assert isinstance(simulator, (TransolverSimulator, GeoFlareSimulator))
        time_ref = _run_time_ref(run.record, spec, len(trajectory.time))
        if run.family == "transolver" and bool(getattr(model_cfg, "flow_map", False)):
            assert isinstance(simulator, TransolverSimulator)
            _, canonical = _fm_intervals_and_canonical(model_cfg)
            return flow_map_rollout(
                simulator,
                trajectory,
                model_cfg.input_frames,
                time_ref,
                canonical,
                "self",
                device,
                qois=spec.qois,
                kinematic_types=spec.kinematic_types,
                scored_frames=scored,
                qoi_aux_channel=run.qoi_aux_channel,
            )
        return time_conditioned_rollout(
            simulator,
            trajectory,
            model_cfg.input_frames,
            time_ref,
            device,
            qois=spec.qois,
            kinematic_types=spec.kinematic_types,
            scored_frames=scored,
            qoi_aux_channel=run.qoi_aux_channel,
        )
    if run.aux_in_run:
        simulator.aux_feedback = "self"
    return rollout(
        simulator,
        trajectory,
        model_cfg.input_frames,
        device,
        qois=spec.qois,
        kinematic_types=spec.kinematic_types,
        scored_frames=scored,
        qoi_aux_channel=run.qoi_aux_channel,
    )
