"""Regenerate the standard fringe figures for a training run.

For every predicted rollout in ``<run>/rollouts/*.npz`` this writes a
ground-truth vs prediction fringe grid (and optionally a GIF of the
predicted rollout) to ``<run>/plots/``::

    python -m structbench.viz --run runs/taylor-baseline \\
        --data-root /path/to/data [--gif] [--bands 12]

The benchmark (and therefore the auxiliary field compared) is resolved from
``<run>/config.json``; the default is ``taylor_impact_2d`` (von Mises stress).

Given several runs, or a case, it instead draws the reference and every
run's stored rollout of the same case side by side, by element where the
case has a mesh, with the benchmark's rigid bodies, to ``--out``::

    python -m structbench.viz --run runs/a --run runs/b --label A --label B \
        --data-root /path/to/data --out figures/round-3 \
        --split test --case T-20-60-100 [--channel effective_plastic_strain]

Without ``--case`` the case of each ``--split`` is the one at ``--quantile``
(default the median) of the runs' mean displacement error; ``--pick-only``
prints that choice and draws nothing, so a study can fix its cases once.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ..benchmarks import BenchmarkSpec, get_benchmark
from ..config import read_run_record
from .compare import compare_runs, figure_span, open_runs, pick_cases
from .fringe import animate_rollout, compare_rollout, load_case_field

__all__ = ["figure_span", "main", "snapshot_frames", "split_and_case"]


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


def snapshot_frames(n_frames: int, input_frames: int, columns: int) -> list[int]:
    """Evenly spaced frame indices from rollout start to the final frame.

    ``input_frames`` is the seeded prefix length (the first predicted frame).
    A single column shows the final frame; ``columns`` below 1 is an error.
    """
    if columns < 1:
        raise ValueError(f"columns must be >= 1, got {columns}")
    if columns == 1:
        return [n_frames - 1]
    span = n_frames - 1 - input_frames
    return [input_frames + (span * i) // (columns - 1) for i in range(columns)]


def _compare(args: argparse.Namespace) -> None:
    """The several-run comparison of the command line."""
    splits = list(args.split or ())
    if args.case:
        if len(splits) == 1:
            splits = splits * len(args.case)
        if len(splits) != len(args.case):
            raise SystemExit(
                "--case needs its --split: one --split for all, or one per --case"
            )
        cases = list(zip(splits, args.case, strict=True))
    else:
        if not splits:
            raise SystemExit("give --case with --split, or --split to pick a case")
        cases = pick_cases(args.run, splits, args.quantile)
        for split, case in cases:
            print(f"picked {split}: {case}")
    if args.pick_only:
        return
    if args.out is None:
        raise SystemExit("a comparison needs --out")
    try:
        views = open_runs(args.run, args.label)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    import matplotlib

    matplotlib.use("Agg")
    for split, case in cases:
        gif, final = compare_runs(
            views,
            split,
            case,
            args.data_root,
            args.out,
            channel=args.channel,
            full=args.full,
            bands=args.bands,
            vmax=args.vmax,
            vmax_percentile=args.vmax_percentile,
            axis_labels=(f"{args.axis_names[0]} (mm)", f"{args.axis_names[1]} (mm)"),
        )
        print(f"wrote {gif}")
        print(f"wrote {final}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m structbench.viz", description=__doc__
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        action="append",
        help="run directory; give it several times to compare runs on one case",
    )
    compare = parser.add_argument_group("comparing several runs on one case")
    compare.add_argument(
        "--out", type=Path, help="directory the comparison is written to"
    )
    compare.add_argument(
        "--label", action="append", help="a panel title per --run, in order"
    )
    compare.add_argument(
        "--split", action="append", help="scored set of the case(s); repeatable"
    )
    compare.add_argument(
        "--case",
        action="append",
        help="case id in the --split given at the same position (or in the "
        "one --split); repeatable. Default: picked per split, see --quantile",
    )
    compare.add_argument(
        "--quantile",
        type=float,
        default=0.5,
        help="without --case: the rank of the picked case in the runs' mean "
        "displacement error (0 best, 1 worst; default the median)",
    )
    compare.add_argument(
        "--pick-only", action="store_true", help="print the picked cases and stop"
    )
    compare.add_argument(
        "--channel",
        help="channel label to draw (default the benchmark's headline field)",
    )
    compare.add_argument("--vmax", type=float, help="top of the fringe scale")
    compare.add_argument(
        "--vmax-percentile",
        type=float,
        help="top of the fringe scale as a percentile of the reference at "
        "the last frame shown",
    )
    compare.add_argument(
        "--axis-names",
        nargs=2,
        default=("x", "y"),
        metavar=("X", "Y"),
        help="names of the two drawn axes (e.g. r z for an axisymmetric case)",
    )
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
    parser.add_argument(
        "--full",
        action="store_true",
        help="show every saved frame, past the case's scored bound (ADR-0078); "
        "by default panels and animations cover the scored span only",
    )
    parser.add_argument(
        "--aux-channel",
        default="0",
        help="aux channel to fringe on a multi-channel (ADR-0059) rollout: a "
        "channel label (e.g. s_xx) or an index (default 0, the headline "
        "channel; ignored for single-channel runs)",
    )
    args = parser.parse_args(argv)

    comparing = (
        len(args.run) > 1 or args.case or args.split or args.out or args.pick_only
    )
    if comparing:
        _compare(args)
        return
    args.run = args.run[0]

    spec, record = _resolve_run_spec(args.run)
    # ADR-0059: a variant run's rollouts carry (T, P, C) aux; fringe rendering
    # is per-scalar, so one channel is selected — by label or index. Labels
    # follow the recorded selection; legacy artifacts without a recorded
    # selection keep today's labelling.
    from ..datasets import aux_channel_labels, aux_channel_units

    run_aux_fields = tuple(record.get("train", {}).get("aux_fields") or ()) or (
        spec.aux_field,
    )
    channel_labels = aux_channel_labels(run_aux_fields)
    channel_units = aux_channel_units(run_aux_fields)
    if args.aux_channel in channel_labels:
        aux_channel = channel_labels.index(args.aux_channel)
    else:
        try:
            aux_channel = int(args.aux_channel)
        except ValueError:
            raise SystemExit(
                f"--aux-channel {args.aux_channel!r} is neither a channel "
                f"label ({', '.join(channel_labels)}) nor an index"
            ) from None

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
        pred_aux = pred["predicted_aux"]
        # The run's recorded selection names the channels even for a C=1
        # variant (whose artifact is a squeezed (T, P)); the benchmark's
        # canonical selection reproduces today's labels exactly.
        aux_label = channel_labels[min(aux_channel, len(channel_labels) - 1)]
        aux_unit = (
            spec.card.aux_unit
            if run_aux_fields == (spec.aux_field,)
            else channel_units[min(aux_channel, len(channel_units) - 1)]
        )
        if pred_aux.ndim == 3:  # (T, P, C) variant artifact (ADR-0059)
            if not 0 <= aux_channel < pred_aux.shape[-1]:
                raise SystemExit(
                    f"--aux-channel {args.aux_channel} out of range for "
                    f"{pred_aux.shape[-1]} channels ({', '.join(channel_labels)})"
                )
            pred_aux = pred_aux[..., aux_channel]
        # The GT fringe row always shows the benchmark's canonical field —
        # per-channel GT extraction is not wired into the fringe loader. Say
        # so loudly when the prediction row shows something else.
        if aux_label != spec.aux_field:
            print(
                f"WARNING: prediction row shows {aux_label!r} but the GT row "
                f"shows the benchmark field {spec.aux_field!r} — the two "
                "fringe rows are different quantities"
            )
        gt = load_case_field(args.data_root / f"{case}.h5", spec.aux_field)
        # ADR-0078: the span shown is the scored one unless --full; the
        # errors in the title are means over the frames shown.
        input_frames, end = figure_span(pred, spec.scored_frames_for(case), args.full)
        frames = snapshot_frames(end, input_frames, args.columns)
        shown = end - input_frames
        past_bound = (
            args.full
            and "scored_frames" in pred
            and 0 <= int(pred["scored_frames"]) < end
        )
        span_note = "   |   includes the unscored tail" if past_bound else ""

        pos_rmse = float(pred["position_rmse"][:shown].mean())
        aux_rmse = float(pred["aux_rmse"][:shown].mean())
        fig = compare_rollout(
            gt.positions,
            gt.values,
            pred["predicted_positions"],
            pred_aux,
            frames=frames,
            times_us=gt.times_us,
            title=(
                f"{case} ({split})   |   rollout RMSE: "
                f"position {pos_rmse:.2f} mm, "
                f"{aux_label} {aux_rmse:.1f} {aux_unit}{span_note}"
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
                pred["predicted_positions"][:end],
                pred_aux[:end],
                out_dir / f"rollout-{case}-{split}.gif",
                times_us=gt.times_us[:end],
                title=f"{case} ({split}) — CGN prediction",
                bands=args.bands,
                wall_x=args.wall_x,
            )
            print(f"wrote {gif}")


if __name__ == "__main__":
    main()
