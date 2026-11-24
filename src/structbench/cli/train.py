"""Config-driven CGN training, validation, and rollout entry point.

This module ties together the StructBench ML layer: the canonical data
pipeline (:mod:`structbench.datasets`), the learned simulator
(:mod:`structbench.models.cgn`), the rollout evaluation
(:mod:`structbench.eval`), and the benchmark registry
(:mod:`structbench.benchmarks`).

The training loop is ported from the sgnn reference
(``sgnn/single_scale/train.py``) and the random-walk position noise from
``sgnn/noise_utils.py``. The reference's npz/metadata data path is replaced by
the canonical pipeline: train trajectories come from
:func:`~structbench.datasets.load_case_trajectory` over the spec's train
split, batched through :class:`~structbench.datasets.WindowDataset` and
:func:`~structbench.datasets.collate_samples`, with normalization from
:func:`~structbench.datasets.compute_stats`. The active benchmark is resolved
via :data:`TrainConfig.benchmark` → :func:`~structbench.benchmarks.get_benchmark`
→ a :class:`~structbench.benchmarks.BenchmarkSpec` that supplies the splits,
auxiliary field, QoIs, and optional boundary feature.

Positions are in the millimetre working frame; the auxiliary field's unit is
specified by the benchmark card (MPa for the Taylor default). Library functions
log via :mod:`logging`; only :func:`main` prints.

This module is the command: the argument parser, :func:`main`, and the public
names re-exported. The training loops live in :mod:`structbench.cli.training`
and the evaluation of a trained run in :mod:`structbench.cli.evaluation`.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from ..benchmarks import (
    get_benchmark,
)
from ..config import (
    CGNConfig,
    GeoFlareConfig,
    MGNConfig,
    TrainConfig,
    TransolverConfig,
    load_run_config,
)
from .evaluation.gradient import evaluate_gradient
from .evaluation.restart import evaluate_restart
from .evaluation.rollout import evaluate
from .evaluation.run import _resolve_run_spec
from .training.builders import (
    build_geoflare_simulator,
    build_mgn_simulator,
    build_simulator,
    build_transolver_simulator,
)
from .training.cgn import train
from .training.common import TrainingDiverged

logger = logging.getLogger(__name__)

__all__ = [
    "CGNConfig",
    "GeoFlareConfig",
    "MGNConfig",
    "TrainConfig",
    "TransolverConfig",
    "build_geoflare_simulator",
    "build_mgn_simulator",
    "build_simulator",
    "build_transolver_simulator",
    "evaluate",
    "main",
    "train",
]


def _enable_deterministic_algorithms() -> None:
    """Ask torch for deterministic kernels for the rest of the process.

    On a GPU the scatter-adds a graph network aggregates with are otherwise
    not repeatable between two runs. cuBLAS needs its workspace configured
    before its first call, so the variable is set here when the caller has
    not set it. An operation with no deterministic kernel then raises a
    ``RuntimeError`` naming itself instead of running.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run training, validation, or rollout.

    Parameters
    ----------
    argv : list of str or None
        Argument vector (defaults to ``sys.argv[1:]`` when ``None``).

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="StructBench CGN training entry")
    parser.add_argument(
        "--mode",
        choices=["train", "valid", "rollout", "gradient", "restart"],
        default="train",
        help="train, validate (VAL), roll out (TEST), the gradient family "
        "(ADR-0076) or the restart family (ADR-0075 clause 5) on the TEST sets.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Grouped TOML run config (ADR-0032; required in train mode; "
        "valid/rollout rebuild the architecture from the run directory's "
        "config.json).",
    )
    parser.add_argument("--out", type=str, default=None, help="Run output directory.")
    parser.add_argument(
        "--data-root", type=str, default=None, help="Directory of <case_id>.h5 cases."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Evaluate this specific checkpoint file (e.g. a periodic "
        "ckpt-<step>.pt; a relative path resolves against --out). "
        "valid/rollout only. Metrics land in metrics-<split>@<name>.json and "
        "rollout .npz artifacts are skipped, so the canonical "
        "selected-checkpoint artifacts are never overwritten.",
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        help="valid/rollout only: roll each case on past its scored bound to "
        "the last stored frame and report the tail's errors and properties "
        "under 'unscored_tail' (ADR-0078). By default a rollout stops at the "
        "bound; the scored numbers are the same either way.",
    )
    parser.add_argument(
        "--save-restarts",
        action="store_true",
        help="restart only: save every predicted window under "
        "<out-dir>/restarts/, one .npz per anchor and horizon (and per "
        "composition for a flow map), so a row can be recomputed from the "
        "prediction.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="valid/rollout/gradient/restart only: evaluate with torch's "
        "deterministic kernels, so two evaluations of one checkpoint on one "
        "machine agree bit for bit where every operation has such a kernel "
        "(slower on a GPU; an operation without one stops the run and is "
        "named). The report records it under 'deterministic_algorithms'.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if args.mode == "train" and args.config is None:
        print("error: --config is required in train mode (ADR-0032)")
        return 2
    if args.mode == "train" and args.checkpoint is not None:
        print("error: --checkpoint applies to valid/rollout modes only")
        return 2
    if args.save_restarts and args.mode != "restart":
        print("error: --save-restarts applies to restart mode only")
        return 2
    if args.tail and args.mode not in ("valid", "rollout"):
        print("error: --tail applies to valid/rollout modes only")
        return 2
    if args.deterministic and args.mode == "train":
        print("error: --deterministic applies to the evaluation modes only")
        return 2
    if args.deterministic:
        _enable_deterministic_algorithms()
    run_config = load_run_config(args.config) if args.config is not None else None

    if args.data_root is None:
        print("error: --data-root is required")
        return 2
    data_root = Path(args.data_root)

    if args.out is not None:
        out_dir = Path(args.out)
    elif args.mode == "train":
        out_dir = Path("runs") / datetime.now().strftime("run-%Y%m%d-%H%M%S")
    else:
        print(
            "error: --out is required in valid/rollout/gradient mode "
            "(the existing run directory to evaluate)"
        )
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"mode={args.mode} device={device} out={out_dir}")

    if args.mode == "train":
        assert run_config is not None  # guarded above
        spec = get_benchmark(run_config.train.benchmark)
        try:
            ckpt = train(
                spec,
                run_config.model,
                run_config.train,
                data_root,
                out_dir,
                device,
                family=run_config.family,
            )
        except TrainingDiverged as exc:
            logger.error("%s", exc)
            print(f"error: {exc}")
            return 3
        print(f"training complete; best checkpoint: {ckpt}")
    else:
        spec, _resolved = _resolve_run_spec(out_dir)
        if args.mode == "valid":
            metrics = evaluate(
                list(spec.splits["val"]),
                data_root,
                out_dir,
                device,
                split_name="val",
                checkpoint=args.checkpoint,
                tail=args.tail,
            )
            _print_split_report(metrics)
        elif args.mode == "gradient":  # ADR-0076: the gradient task family
            report = evaluate_gradient(
                data_root, out_dir, device, checkpoint=args.checkpoint
            )
            for key, value in report["metrics"].items():
                print(f"[gradient] {key} {value:.4f}")
        elif args.mode == "restart":  # ADR-0075 clause 5: the restart family
            for split_name in spec.eval_splits:
                if split_name == "val":
                    continue
                try:
                    report = evaluate_restart(
                        list(spec.splits[split_name]),
                        data_root,
                        out_dir,
                        device,
                        split_name=split_name,
                        checkpoint=args.checkpoint,
                        save_windows=args.save_restarts,
                    )
                except ValueError as exc:
                    # A benchmark without restarts or a scheme that cannot
                    # restart: one line, not a traceback.
                    print(f"error: {exc}")
                    return 2
                for key, value in report["mean"].items():
                    if isinstance(value, dict):  # a per-channel row
                        shown = " ".join(f"{c}={v:.4f}" for c, v in value.items())
                    else:
                        shown = "null" if value is None else f"{value:.4f}"
                    print(f"[restart:{split_name}] {key} {shown}")
        else:  # rollout: every eval split except val, in spec order
            for split_name in spec.eval_splits:
                if split_name == "val":
                    continue
                metrics = evaluate(
                    list(spec.splits[split_name]),
                    data_root,
                    out_dir,
                    device,
                    split_name=split_name,
                    checkpoint=args.checkpoint,
                    tail=args.tail,
                )
                _print_split_report(metrics)
    return 0


def _print_split_report(metrics: dict[str, Any]) -> None:
    """Print one split's metrics to stdout.

    Position RMSE is in mm. Aux RMSE is in the run's aux unit
    (recorded in benchmark card). QoI errors are in each QoI's own unit
    (recorded in the benchmark card).
    """
    split, mean = metrics["split"], metrics["mean"]
    aux_field = metrics.get("aux_field", "aux")
    aux_unit = metrics.get("aux_unit", "")
    # ADR-0059 variant runs record per-channel lists; the summary line shows
    # the joined labels and pooled numbers (per-channel detail stays in JSON).
    if isinstance(aux_field, list):
        aux_field = "+".join(aux_field)
    if isinstance(aux_unit, list):
        aux_unit = "/".join(dict.fromkeys(aux_unit))

    def _fmt(value: float | None) -> str:
        # one_step_* is None under the ADR-0054 time-conditioned scheme
        # (independent per-t queries have no single-step notion).
        return "N/A" if value is None else f"{value:.4f}"

    aux_rmse_str = f"rollout {aux_field} RMSE {_fmt(mean['rollout_aux_rmse'])}"
    if aux_unit:
        aux_rmse_str = f"{aux_rmse_str} {aux_unit}"
    print(
        f"[{split}] one-step position RMSE {_fmt(mean['one_step_position_rmse'])} mm"
        f" | one-step {aux_field} RMSE {_fmt(mean['one_step_aux_rmse'])}"
        f" | rollout position RMSE {_fmt(mean['rollout_position_rmse'])} mm"
        f" | {aux_rmse_str}"
    )
    # Rollout relative-L2 (ADR-0055), dimensionless and rollout-only; .get()
    # tolerates a metrics dict predating this metric (older re-printed records).
    # The headline is pooled space+time (ADR-0055 follow-up).
    print(
        f"[{split}] rollout rel-L2 disp (pooled) "
        f"{_fmt(mean.get('rollout_rel_l2_displacement'))}"
        f" | rollout rel-L2 {aux_field} (pooled) "
        f"{_fmt(mean.get('rollout_rel_l2_aux'))}"
    )
    qoi = ", ".join(
        f"{name} {value:.4f}" for name, value in mean["qoi_abs_error"].items()
    )
    print(f"[{split}] QoI mean |error|: {qoi}")


if __name__ == "__main__":
    raise SystemExit(main())
