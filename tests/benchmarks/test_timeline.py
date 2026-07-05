"""Timeline analysis (ADR-0032 §5): closed-form transient characterization."""

import numpy as np
import pytest

from structbench.benchmarks.timeline import (
    CANDIDATE_INITS,
    analyze_trajectory,
    render_report,
)
from structbench.datasets.canonical import CaseTrajectory


def _impact_like_traj(T: int = 40, P: int = 6) -> CaseTrajectory:
    """Free flight for 8 intervals, then exponential velocity decay to rest."""
    dt = 1e-6
    t = np.arange(T) * dt
    v = np.ones(T - 1)
    v[8:] = np.exp(-0.4 * np.arange(T - 1 - 8))
    x = np.concatenate([[0.0], np.cumsum(v * dt)])
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pos[:, :, 0] = x[:, None] * 1e3  # mm
    aux = np.zeros((T, P), dtype=np.float32)
    if T > 15:
        aux[15] = 7.0  # single-frame peak of the mean field
    return CaseTrajectory("IMP-1", pos, np.ones(P, np.int64), aux, t)


def test_analyze_trajectory_free_flight_prefix_dissipates_nothing():
    tl = analyze_trajectory(_impact_like_traj())
    # Free flight through interval 7: candidate inits 3 and 6 hand over 0% KE.
    # float32 positions round-trip: 'zero' is ~1e-7 relative KE
    assert tl.ke_frac_dissipated_at[3] == pytest.approx(0.0, abs=1e-5)
    assert tl.ke_frac_dissipated_at[6] == pytest.approx(0.0, abs=1e-5)
    # An 11-frame prefix (frames 0..10) exposes intervals 0..9; interval 9
    # has v = exp(-0.4), so KE ratio exp(-0.8).
    assert tl.ke_frac_dissipated_at[11] == pytest.approx(
        1.0 - np.exp(-0.8), rel=1e-4
    )
    assert tl.n_frames == 40
    assert tl.dt_median == pytest.approx(1e-6)
    # KE milestones ordered and inside the record.
    assert tl.t_ke50 <= tl.t_ke90 <= tl.t_ke99
    # Peak of the mean aux field: value 7.0 at frame 15.
    assert tl.peak_mean_aux == pytest.approx(7.0)
    assert tl.t_peak_mean_aux == pytest.approx(15e-6)


def test_analyze_trajectory_rejects_too_short():
    traj = _impact_like_traj(T=3)
    with pytest.raises(ValueError, match="frames"):
        analyze_trajectory(traj)


def test_render_report_contains_cases_and_aggregate():
    tl = analyze_trajectory(_impact_like_traj())
    report = render_report("taylor_impact_2d", [tl])
    assert "IMP-1" in report
    assert "## Aggregate" in report
    for k in CANDIDATE_INITS:
        assert f"@{k}" in report
    with pytest.raises(ValueError, match="no timelines"):
        render_report("x", [])
