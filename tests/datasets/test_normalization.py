import numpy as np

from structbench.datasets.canonical import CaseTrajectory
from structbench.datasets.normalization import NormalizationStats, compute_stats


def _const_accel_traj():
    # x(t) = 0.5 * a * t^2 with a=[2,0]; first diff = velocity, second diff = a.
    T, P = 5, 4
    t = np.arange(T)
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pos[:, :, 0] = (0.5 * 2.0 * t**2)[:, None]
    return CaseTrajectory(
        "c",
        pos,
        np.ones(P, np.int64),
        np.zeros((T, P), np.float32),
        t.astype(np.float64),
    )


def _known_vm_traj():
    # Same kinematics as above but with a known von Mises field [0, 1, ..., T*P-1].
    T, P = 5, 4
    t = np.arange(T)
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pos[:, :, 0] = (0.5 * 2.0 * t**2)[:, None]
    vm = np.arange(T * P, dtype=np.float32).reshape(T, P)
    return CaseTrajectory(
        "c",
        pos,
        np.ones(P, np.int64),
        vm,
        t.astype(np.float64),
    )


def test_compute_stats_constant_acceleration():
    stats = compute_stats([_const_accel_traj()])
    np.testing.assert_allclose(stats.acceleration_mean, [2.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(stats.acceleration_std, [0.0, 0.0], atol=1e-5)
    assert stats.velocity_mean.shape == (2,)


def test_compute_stats_aux_mean_std():
    vm = np.arange(5 * 4, dtype=np.float64)
    stats = compute_stats([_known_vm_traj()])
    assert stats.aux_mean.shape == (1,)
    assert stats.aux_std.shape == (1,)
    np.testing.assert_allclose(stats.aux_mean, [vm.mean()], atol=1e-5)
    np.testing.assert_allclose(stats.aux_std, [vm.std()], atol=1e-5)


def test_normalization_stats_roundtrip(tmp_path):
    stats = compute_stats([_known_vm_traj()])
    p = tmp_path / "norm.npz"
    stats.save(p)
    back = NormalizationStats.load(p)
    np.testing.assert_array_equal(back.acceleration_mean, stats.acceleration_mean)
    np.testing.assert_array_equal(back.aux_mean, stats.aux_mean)
    np.testing.assert_array_equal(back.aux_std, stats.aux_std)
