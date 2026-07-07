"""Ground-truth timeline analysis: the mandatory protocol rationale (ADR-0032 §5).

Before a benchmark's protocol values (``input_frames``, horizon, eval times)
are pinned, this analysis characterizes the ground-truth dynamics so the
values can be justified by the task rather than by any baseline's
performance::

    python -m structbench.benchmarks.timeline \\
        --benchmark taylor_impact_2d --data-root /path/to/cases [--out report.md]

Per case it reports kinetic-energy dissipation milestones, displacement
settlement, tail activity, the KE fraction dissipated within candidate init
prefixes, and the peak of the particle-mean aux field with its time. The
aggregate summary is the evidence base for the card's ``protocol_rationale``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..datasets import CaseTrajectory, load_case_trajectory
from .registry import get_benchmark

#: Candidate observed-prefix lengths reported by the analysis.
CANDIDATE_INITS = (3, 6, 11)


@dataclass(frozen=True)
class CaseTimeline:
    """Timeline characterization of one ground-truth case (ADR-0032 §5).

    Times are in the trajectory's time unit (seconds) unless suffixed.
    ``ke_frac_dissipated_at`` maps a candidate ``input_frames`` to the
    fraction of initial kinetic energy already dissipated within that
    observed prefix (what a model would be handed "for free").
    """

    case_id: str
    n_frames: int
    dt_median: float
    t_ke50: float
    t_ke90: float
    t_ke99: float
    t_settle99: float
    tail_activity: float
    ke_frac_dissipated_at: dict[int, float]
    peak_mean_aux: float
    t_peak_mean_aux: float


def analyze_trajectory(traj: CaseTrajectory) -> CaseTimeline:
    """Characterize one trajectory's transient timing.

    Parameters
    ----------
    traj : CaseTrajectory
        Ground-truth trajectory (positions in mm, time in s).

    Returns
    -------
    CaseTimeline

    Raises
    ------
    ValueError
        If the trajectory has fewer than 4 frames (no acceleration signal).
    """
    pos, t = traj.positions.astype(float), traj.time
    n_frames = pos.shape[0]
    if n_frames < 4:
        raise ValueError(f"{traj.case_id}: need >= 4 frames, got {n_frames}")
    dt = np.diff(t)
    velocity = np.diff(pos, axis=0) / dt[:, None, None]
    accel = np.diff(velocity, axis=0) / dt[1:, None, None]

    # Kinetic-energy proxy (uniform particle mass): sum of squared speeds per
    # inter-frame interval, referenced to the initial interval.
    ke = (np.linalg.norm(velocity, axis=-1) ** 2).sum(axis=1)
    ke_frac = ke / ke[0] if ke[0] > 0 else np.ones_like(ke)

    def time_at_ke_below(threshold: float) -> float:
        below = np.nonzero(ke_frac < threshold)[0]
        # NaN, not t[-1]: a milestone that never happens must not read as
        # "reached at the final frame" in the evidence table.
        return float(t[below[0]]) if below.size else float("nan")

    # Displacement settlement: sustained — the frame after the LAST excursion
    # outside the 1% band (first-crossing under-reports oscillatory cases).
    remaining = np.linalg.norm(pos - pos[-1], axis=-1).mean(axis=1)
    band = 0.01 * remaining[0]
    unsettled = np.nonzero(remaining >= band)[0]
    settle_idx = min(int(unsettled[-1]) + 1, n_frames - 1) if unsettled.size else 0
    t_settle = float(t[settle_idx])

    mean_accel = np.linalg.norm(accel, axis=-1).mean(axis=1)
    tail = mean_accel[int(0.8 * len(mean_accel)) :]
    peak_accel = float(mean_accel.max())
    # float32 positions put a rounding-noise floor of ~4*eps32*|pos|/dt^2 on
    # second differences; a "peak" below it is noise, not dynamics.
    noise_floor = (
        4.0
        * float(np.finfo(np.float32).eps)
        * float(np.abs(pos).max())
        / float(dt.min()) ** 2
    )
    tail_activity = float(tail.mean() / peak_accel) if peak_accel > noise_floor else 0.0

    # A k-frame observed prefix (frames 0..k-1) exposes velocity intervals
    # 0..k-2 only; interval k-1 ends at the first *predicted* frame. Negative
    # float noise is clamped; infeasible candidates report NaN.
    dissipated = {
        k: max(0.0, float(1.0 - ke_frac[k - 2])) if k < n_frames else float("nan")
        for k in CANDIDATE_INITS
    }

    mean_aux = np.asarray(traj.aux, float).mean(axis=1)
    peak_frame = int(mean_aux.argmax())

    return CaseTimeline(
        case_id=traj.case_id,
        n_frames=n_frames,
        dt_median=float(np.median(dt)),
        t_ke50=time_at_ke_below(0.5),
        t_ke90=time_at_ke_below(0.1),
        t_ke99=time_at_ke_below(0.01),
        t_settle99=t_settle,
        tail_activity=tail_activity,
        ke_frac_dissipated_at=dissipated,
        peak_mean_aux=float(mean_aux[peak_frame]),
        t_peak_mean_aux=float(t[peak_frame]),
    )


def render_report(benchmark_name: str, timelines: list[CaseTimeline]) -> str:
    """The markdown timeline report for one benchmark's cases.

    Parameters
    ----------
    benchmark_name : str
        Registry name, used in the heading.
    timelines : list of CaseTimeline
        One entry per analyzed case; must be non-empty.
    """
    if not timelines:
        raise ValueError("no timelines to render")

    def ms(seconds: float) -> str:
        return "not reached" if np.isnan(seconds) else f"{seconds * 1e3:.3g} ms"

    init_cols = "".join(f" KE diss @{k} |" for k in CANDIDATE_INITS)
    lines = [
        f"# GT timeline analysis — {benchmark_name}",
        "",
        "Protocol rationale evidence (ADR-0032 §5). KE milestones are the",
        "times at which the given fraction of initial kinetic energy has",
        "dissipated; settle99 is 99% displacement settlement; tail activity",
        "is the last-20%-of-horizon mean |acceleration| relative to its peak;",
        "`KE diss @k` is the KE fraction dissipated within a k-frame observed",
        "prefix — what a model at `input_frames = k` is handed for free.",
        "",
        f"| case | frames | dt | KE50 | KE90 | KE99 | settle99 | tail |{init_cols}"
        " peak mean aux | t(peak) |",
        f"|---|---|---|---|---|---|---|---|{'---|' * len(CANDIDATE_INITS)}---|---|",
    ]
    for tl in timelines:
        init_cells = "".join(
            " n/a |"
            if np.isnan(tl.ke_frac_dissipated_at[k])
            else f" {tl.ke_frac_dissipated_at[k]:.1%} |"
            for k in CANDIDATE_INITS
        )
        lines.append(
            f"| {tl.case_id} | {tl.n_frames} | {ms(tl.dt_median)} "
            f"| {ms(tl.t_ke50)} | {ms(tl.t_ke90)} | {ms(tl.t_ke99)} "
            f"| {ms(tl.t_settle99)} | {tl.tail_activity:.1%} |{init_cells}"
            f" {tl.peak_mean_aux:.4g} | {ms(tl.t_peak_mean_aux)} |"
        )
    worst = {
        k: np.nanmax([tl.ke_frac_dissipated_at[k] for tl in timelines])
        for k in CANDIDATE_INITS
    }
    latest_settle = np.nanmax([tl.t_settle99 for tl in timelines])
    lines += [
        "",
        "## Aggregate",
        "",
        "- Worst-case KE dissipated within candidate inits: "
        + ", ".join(f"@{k}: {worst[k]:.1%}" for k in CANDIDATE_INITS),
        f"- Latest 99% settlement: {ms(latest_settle)}",
        f"- Peak mean aux across cases: "
        f"{max(tl.peak_mean_aux for tl in timelines):.4g} "
        f"(latest at {ms(max(tl.t_peak_mean_aux for tl in timelines))})",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; prints (or writes) the markdown report."""
    parser = argparse.ArgumentParser(
        prog="python -m structbench.benchmarks.timeline", description=__doc__
    )
    parser.add_argument("--benchmark", required=True, help="registry name")
    parser.add_argument(
        "--data-root", required=True, type=Path, help="directory of <case_id>.h5"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="write the report here (else stdout)"
    )
    args = parser.parse_args(argv)

    try:
        spec = get_benchmark(args.benchmark)
    except KeyError as err:
        raise SystemExit(err.args[0]) from None
    if not args.data_root.is_dir():
        raise SystemExit(f"data root is not a directory: {args.data_root}")
    case_ids = sorted({cid for ids in spec.splits.values() for cid in ids})
    missing = [cid for cid in case_ids if not (args.data_root / f"{cid}.h5").exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} case files missing under {args.data_root}: "
            + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )
    timelines = [
        analyze_trajectory(
            load_case_trajectory(args.data_root / f"{cid}.h5", aux_field=spec.aux_field)
        )
        for cid in case_ids
    ]
    report = render_report(args.benchmark, timelines)
    if args.out is None:
        print(report, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
