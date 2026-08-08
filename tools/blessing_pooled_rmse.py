"""ADR-0043 SS8 blessing aggregator: paper-convention pooled rollout RMSE.

Computes the pooled rollout-position RMSE from an ``evaluate()`` run's
on-disk artifacts, under the paper's pooling convention (ADR-0043 SS8, as
corrected by its 2026-08-08 dated note): sqrt(sum of squared position error
/ total count), pooled over all spatial coordinates x all mesh nodes
(kinematic rows included -- they are GT-prescribed so contribute zero
error) x all steps of a trajectory (including the GT-seeded input frames,
also zero error) x all cases of the split. This is a different statistic
from the StructBench leaderboard's mean-of-per-step-RMSE (ADR-0019 SS5);
this tool exists precisely to keep the two from being conflated.

Reads, per case, the predicted rollout trajectory written by
``structbench.cli.train.evaluate`` (``<run-dir>/rollouts/<split>-<case_id>.npz``,
key ``predicted_positions``) and the ground-truth trajectory via
``structbench.datasets.load_case_trajectory``. Not importable package code
(mirrors ``tools/gen_benchmark_docs.py``): a standalone script, not covered
by mypy.

Usage:
    python tools/blessing_pooled_rmse.py --run-dir RUN --data-root DATA
    python tools/blessing_pooled_rmse.py --run-dir RUN --data-root DATA \\
        --split test --benchmark deforming_plate --out RUN/blessing.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from structbench.benchmarks import get_benchmark
from structbench.datasets import load_case_trajectory

#: ADR-0043 SS8 gate band in the mm working frame: the paper's 15.1 +/- 4.0
#: (x1e-3, dataset-native/metre units) == 15.1 +/- 4.0 mm, since the
#: dataset is metre-native and our working frame is millimetres. Literal
#: bounds (not computed as 15.1 - 4.0 / 15.1 + 4.0) to avoid float
#: representation drift from the subtraction/addition.
GATE_BAND_MM: tuple[float, float] = (11.1, 19.1)


def pooled_rmse(
    pred: NDArray[np.floating], true: NDArray[np.floating]
) -> tuple[float, int]:
    """Summed squared error and element count, pooled over every axis.

    The pure core of the ADR-0043 SS8 pooled rollout RMSE (2026-08-08 dated
    note correction): pooling is over *every* element of the input arrays
    with no masking. Callers pass the full ``(T, P, dim)`` arrays exactly as
    written by :func:`structbench.cli.train.evaluate` -- kinematic node rows
    and the GT-seeded input frames already equal ground truth by
    construction (:func:`structbench.eval.rollout.rollout` overwrites
    kinematic predictions with ground truth at every step and seeds the
    first ``input_frames`` frames from ground truth verbatim), so they
    contribute exactly zero to ``sse`` without any explicit masking here --
    matching the paper's "all mesh nodes, all steps" pooling.

    Parameters
    ----------
    pred, true : numpy.ndarray
        Same-shape arrays (typically ``(T, P, dim)``, but pooled flat over
        every axis, so any matching shape works).

    Returns
    -------
    tuple of (float, int)
        ``(sse, count)`` -- summed squared error and element count. Combine
        across cases as ``sqrt(sum(sse) / sum(count))`` for the pooled
        (headline) statistic, or ``sqrt(sse / count)`` per case.

    Raises
    ------
    ValueError
        If ``pred`` and ``true`` have different shapes.
    """
    pred_arr = np.asarray(pred, dtype=np.float64)
    true_arr = np.asarray(true, dtype=np.float64)
    if pred_arr.shape != true_arr.shape:
        raise ValueError(
            f"pred/true shape mismatch: {pred_arr.shape} vs {true_arr.shape}"
        )
    sse = float(np.sum((pred_arr - true_arr) ** 2))
    count = int(pred_arr.size)
    return sse, count


def _rmse_from_sse_count(sse: float, count: int) -> float:
    """``sqrt(sse / count)``, guarding the empty-array degenerate case."""
    if count == 0:
        raise ValueError("pooled_rmse: zero-element array (empty trajectory)")
    return float(np.sqrt(sse / count))


def compute_pooled_rmse(
    run_dir: Path,
    data_root: Path,
    split: str,
    benchmark: str,
) -> dict[str, Any]:
    """Compute the ADR-0043 SS8 pooled rollout RMSE over one split.

    Binds to the exact artifact layout written by
    :func:`structbench.cli.train.evaluate`: the case list comes from
    ``<run_dir>/metrics-<split>.json``'s ``"cases"`` keys (not the
    benchmark's frozen split list -- ``evaluate()`` may have been run over a
    subset), and each case's predicted trajectory from
    ``<run_dir>/rollouts/<split>-<case_id>.npz``'s ``"predicted_positions"``
    array. Ground truth is loaded fresh via
    :func:`structbench.datasets.load_case_trajectory` so the comparison is
    never stale relative to the canonical archive.

    Parameters
    ----------
    run_dir : pathlib.Path
        An ``evaluate()`` run/output directory.
    data_root : pathlib.Path
        Canonical archive directory containing ``<case_id>.h5`` cases.
    split : str
        Split name (selects ``metrics-<split>.json`` and the
        ``rollouts/<split>-<case_id>.npz`` files).
    benchmark : str
        Registry name resolved via
        :func:`structbench.benchmarks.get_benchmark`; supplies ``aux_field``
        for :func:`~structbench.datasets.load_case_trajectory`.

    Returns
    -------
    dict
        JSON-serializable report: ``run_dir``, ``data_root``, ``benchmark``,
        ``split``, ``n_cases``, ``pooled_rmse_mm``, ``pooled_rmse_native``,
        ``per_case_mean_mm``, ``per_case_stderr_mm``, ``per_case_rmse_mm``
        (diagnostic, keyed by case id), and ``gate`` (``band_mm``, ``pass``).

    Raises
    ------
    FileNotFoundError
        If ``metrics-<split>.json`` is missing, or a case's rollout ``.npz``
        or canonical ``.h5`` is missing -- named explicitly so a blessing
        computation never silently skips a case.
    KeyError
        If the metrics file has no ``"cases"`` key, or a case's ``.npz`` has
        no ``"predicted_positions"`` array.
    ValueError
        If the metrics file lists no cases, or a case's predicted and
        ground-truth trajectories have mismatched shapes.
    """
    spec = get_benchmark(benchmark)

    metrics_path = run_dir / f"metrics-{split}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"missing {metrics_path}; run structbench.cli.train.evaluate() for "
            f"split {split!r} first (it writes metrics-{split}.json)"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "cases" not in metrics:
        raise KeyError(f"{metrics_path} has no 'cases' key")
    case_ids = sorted(metrics["cases"])
    if not case_ids:
        raise ValueError(f"{metrics_path} lists no cases for split {split!r}")

    rollout_dir = run_dir / "rollouts"
    per_case_sse: dict[str, float] = {}
    per_case_count: dict[str, int] = {}
    per_case_rmse_mm: dict[str, float] = {}

    for case_id in case_ids:
        npz_path = rollout_dir / f"{split}-{case_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"missing rollout artifact for case {case_id!r}: {npz_path} "
                "(a blessing computation must never silently skip cases)"
            )
        with np.load(npz_path) as npz:
            if "predicted_positions" not in npz:
                raise KeyError(
                    f"{npz_path} has no 'predicted_positions' array "
                    f"(keys: {sorted(npz.files)})"
                )
            pred_positions = np.asarray(npz["predicted_positions"])

        try:
            trajectory = load_case_trajectory(
                data_root / f"{case_id}.h5", aux_field=spec.aux_field
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"missing ground-truth case for {case_id!r}: {exc}"
            ) from exc

        try:
            sse, count = pooled_rmse(pred_positions, trajectory.positions)
        except ValueError as exc:
            raise ValueError(f"case {case_id!r}: {exc}") from exc

        per_case_sse[case_id] = sse
        per_case_count[case_id] = count
        per_case_rmse_mm[case_id] = _rmse_from_sse_count(sse, count)

    total_sse = sum(per_case_sse.values())
    total_count = sum(per_case_count.values())
    pooled_rmse_mm = _rmse_from_sse_count(total_sse, total_count)
    pooled_rmse_native = pooled_rmse_mm / 1e3  # mm -> m (dataset-native length unit)

    case_rmses = np.array(list(per_case_rmse_mm.values()), dtype=np.float64)
    n_cases = int(case_rmses.size)
    per_case_mean_mm = float(case_rmses.mean())
    per_case_stderr_mm = (
        float(case_rmses.std(ddof=1) / np.sqrt(n_cases)) if n_cases > 1 else 0.0
    )

    band_lo, band_hi = GATE_BAND_MM
    gate_pass = band_lo <= pooled_rmse_mm <= band_hi

    return {
        "run_dir": str(run_dir),
        "data_root": str(data_root),
        "benchmark": benchmark,
        "split": split,
        "n_cases": n_cases,
        "pooled_rmse_mm": pooled_rmse_mm,
        "pooled_rmse_native": pooled_rmse_native,
        "per_case_mean_mm": per_case_mean_mm,
        "per_case_stderr_mm": per_case_stderr_mm,
        "per_case_rmse_mm": per_case_rmse_mm,
        "gate": {
            "band_mm": [band_lo, band_hi],
            "pass": gate_pass,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, compute the pooled RMSE, write JSON, print a verdict.

    Parameters
    ----------
    argv : list of str or None
        Argument vector (defaults to ``sys.argv[1:]`` when ``None``).

    Returns
    -------
    int
        ``0`` if the gate passes, ``1`` if it fails. Artifact-layout errors
        (missing files, malformed records) propagate as exceptions rather
        than a soft error code -- a blessing computation must never
        silently report a wrong number.
    """
    parser = argparse.ArgumentParser(
        description="ADR-0043 SS8 blessing aggregator: paper-convention "
        "pooled rollout RMSE from an evaluate() run's artifacts."
    )
    parser.add_argument(
        "--run-dir", type=str, required=True, help="An evaluate()/rollout output dir."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Canonical archive directory of <case_id>.h5 cases.",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split name (default: test)."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="deforming_plate",
        help="Benchmark registry name (default: deforming_plate).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: <run-dir>/blessing-pooled-<split>.json).",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    data_root = Path(args.data_root)
    if args.out is not None:
        out_path = Path(args.out)
    else:
        out_path = run_dir / f"blessing-pooled-{args.split}.json"

    result = compute_pooled_rmse(run_dir, data_root, args.split, args.benchmark)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    verdict = "PASS" if result["gate"]["pass"] else "FAIL"
    lo, hi = result["gate"]["band_mm"]
    print(
        f"[{verdict}] pooled rollout RMSE = {result['pooled_rmse_mm']:.4f} mm "
        f"({result['pooled_rmse_native']:.6f} native) over {result['n_cases']} "
        f"{args.split!r} cases; gate band [{lo}, {hi}] mm "
        f"(per-case mean {result['per_case_mean_mm']:.4f} +/- "
        f"{result['per_case_stderr_mm']:.4f} mm)"
    )
    print(f"wrote {out_path}")
    return 0 if result["gate"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
