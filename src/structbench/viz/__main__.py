"""Regenerate the standard fringe figures for a training run.

For every predicted rollout in ``<run>/rollouts/*.npz`` this writes a
ground-truth vs prediction fringe grid (and optionally a GIF of the
predicted rollout) to ``<run>/plots/``::

    python -m structbench.viz --run runs/taylor-baseline \\
        --data-root /path/to/data [--gif] [--bands 12]

The benchmark (and therefore the auxiliary field compared) is resolved from
``<run>/config.json``; the default is ``taylor_impact_2d`` (von Mises stress).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ..benchmarks import BenchmarkSpec, get_benchmark
from ..config import read_run_record
from .fringe import animate_rollout, compare_rollout, load_case_field


def _resolve_run_spec(out_dir: Path) -> tuple[BenchmarkSpec, dict[str, Any]]:
    """Resolve the run directory's benchmark spec from its config.json.

    Parameters
    ----------
    out_dir : pathlib.Path
        Run directory holding ``config.json``.

    Returns
    -------
    tuple of (BenchmarkSpec, dict)
        The benchmark spec and the parsed config dict.

    Raises
    ------
    FileNotFoundError
        If ``config.json`` is missing from ``out_dir``.
    KeyError
        If the ``"benchmark"`` value is not a registered name; the registry
        message lists valid names.
    """
    record = read_run_record(out_dir / "config.json")
    return get_benchmark(record["run"]["benchmark"]), record


def split_and_case(stem: str) -> tuple[str, str]:
    """Parse ``<split>-<case_id>`` rollout-npz stems.

    Supports all benchmark families:

    - Taylor:     ``val-T-20-60-100``     → (``val``, ``T-20-60-100``)
    - Notch-beam: ``val-NB-B-320-Ab-16``  → (``val``, ``NB-B-320-Ab-16``)
    - Wave 1-D:   ``val-W1D-300-2``       → (``val``, ``W1D-300-2``)

    Split names (``val``, ``test_interp``, ``probe``, ``test_extrap``) never
    contain ``-``, so the first ``-`` unambiguously separates split from case
    id.
    """
    split, sep, case_id = stem.partition("-")
    if not sep:
        raise ValueError(f"rollout file {stem!r} does not match <split>-<case_id>")
    return split, case_id


def snapshot_frames(n_frames: int, window: int, columns: int) -> list[int]:
    """Evenly spaced frame indices from rollout start to the final frame.

    A single column shows the final frame; ``columns`` below 1 is an error.
    """
    if columns < 1:
        raise ValueError(f"columns must be >= 1, got {columns}")
    if columns == 1:
        return [n_frames - 1]
    span = n_frames - 1 - window
    return [window + (span * i) // (columns - 1) for i in range(columns)]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m structbench.viz", description=__doc__
    )
    parser.add_argument("--run", type=Path, required=True, help="run directory")
    parser.add_argument(
        "--data-root", type=Path, required=True, help="directory of canonical .h5 cases"
    )
    parser.add_argument(
        "--columns", type=int, default=4, help="snapshot columns per figure"
    )
    parser.add_argument(
        "--bands",
        type=int,
        default=None,
        help="discrete fringe bands (default: continuous)",
    )
    parser.add_argument(
        "--wall-x", type=float, default=-2.0, help="rigid-wall plane x in mm"
    )
    parser.add_argument(
        "--gif", action="store_true", help="also write a GIF of each predicted rollout"
    )
    args = parser.parse_args(argv)

    spec, _ = _resolve_run_spec(args.run)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = args.run / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts = sorted((args.run / "rollouts").glob("*.npz"))
    if not rollouts:
        raise SystemExit(f"no rollout .npz files under {args.run / 'rollouts'}")

    for npz_path in rollouts:
        split, case = split_and_case(npz_path.stem)
        pred = np.load(npz_path)
        gt = load_case_field(args.data_root / f"{case}.h5", spec.aux_field)
        n_frames = gt.positions.shape[0]
        window = n_frames - len(pred["position_rmse"])
        frames = snapshot_frames(n_frames, window, args.columns)

        pos_rmse = float(pred["position_rmse"].mean())
        aux_rmse = float(pred["aux_rmse"].mean())
        fig = compare_rollout(
            gt.positions,
            gt.values,
            pred["predicted_positions"],
            pred["predicted_aux"],
            frames=frames,
            times_us=gt.times_us,
            title=(
                f"{case} ({split})   |   rollout RMSE: "
                f"position {pos_rmse:.2f} mm, "
                f"{spec.aux_field} {aux_rmse:.1f} {spec.card.aux_unit}"
            ),
            bands=args.bands,
            wall_x=args.wall_x,
        )
        png = out_dir / f"rollout-{case}-{split}.png"
        fig.savefig(png, dpi=170)
        plt.close(fig)
        print(f"wrote {png}")

        if args.gif:
            gif = animate_rollout(
                pred["predicted_positions"],
                pred["predicted_aux"],
                out_dir / f"rollout-{case}-{split}.gif",
                times_us=gt.times_us,
                title=f"{case} ({split}) — GNS prediction",
                bands=args.bands,
                wall_x=args.wall_x,
            )
            print(f"wrote {gif}")


if __name__ == "__main__":
    main()
