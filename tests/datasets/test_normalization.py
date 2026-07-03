import numpy as np

from structbench.datasets.canonical import CaseTrajectory
from structbench.datasets.normalization import (
    NormalizationStats,
    cached_compute_stats,
    compute_stats,
)


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


def _doctored(stats):
    """A visibly-wrong copy of ``stats`` to plant in the cache."""
    return NormalizationStats(
        velocity_mean=stats.velocity_mean + 99.0,
        velocity_std=stats.velocity_std,
        acceleration_mean=stats.acceleration_mean,
        acceleration_std=stats.acceleration_std,
        aux_mean=stats.aux_mean,
        aux_std=stats.aux_std,
    )


def test_cache_key_separates_aux_fields(tmp_path):
    trajs = [_const_accel_traj()]  # use this file's existing helper
    s1 = cached_compute_stats(trajs, dataset_root=tmp_path, aux_field="von_mises_stress")
    s2 = cached_compute_stats(trajs, dataset_root=tmp_path, aux_field="axial_stress")
    caches = sorted((tmp_path / "derived").glob("norm_*.npz"))
    assert len(caches) == 2  # one file per aux field, same case ids
    np.testing.assert_allclose(s1.aux_mean, s2.aux_mean)  # same trajs -> same stats


def test_cache_hit_same_aux_field(tmp_path):
    trajs = [_const_accel_traj()]
    cached_compute_stats(trajs, dataset_root=tmp_path, aux_field="von_mises_stress")
    cached_compute_stats(trajs, dataset_root=tmp_path, aux_field="von_mises_stress")
    caches = list((tmp_path / "derived").glob("norm_*.npz"))
    assert len(caches) == 1


def test_cached_stats_writes_then_reuses_the_cache(tmp_path):
    traj = _known_vm_traj()
    first = cached_compute_stats([traj], dataset_root=tmp_path, aux_field="von_mises_stress")
    np.testing.assert_array_equal(first.aux_mean, compute_stats([traj]).aux_mean)

    cache_files = list((tmp_path / "derived").glob("norm_*.npz"))
    assert len(cache_files) == 1

    # Plant doctored stats in the cache: a second call with the same case-id
    # list must return them, proving the cache is read instead of recomputed.
    _doctored(first).save(cache_files[0])
    second = cached_compute_stats([traj], dataset_root=tmp_path, aux_field="von_mises_stress")
    np.testing.assert_array_equal(second.velocity_mean, first.velocity_mean + 99.0)


def test_cached_stats_recomputes_when_case_ids_change(tmp_path):
    traj = _known_vm_traj()  # case_id "c"
    first = cached_compute_stats([traj], dataset_root=tmp_path, aux_field="von_mises_stress")
    cache_files = list((tmp_path / "derived").glob("norm_*.npz"))
    _doctored(first).save(cache_files[0])

    other = _known_vm_traj()
    other.case_id = "d"  # different split -> different cache key
    fresh = cached_compute_stats([other], dataset_root=tmp_path, aux_field="von_mises_stress")
    np.testing.assert_array_equal(fresh.velocity_mean, first.velocity_mean)
    assert len(list((tmp_path / "derived").glob("norm_*.npz"))) == 2


def test_cached_stats_recovers_from_corrupt_cache_file(tmp_path):
    traj = _known_vm_traj()
    good = cached_compute_stats([traj], dataset_root=tmp_path, aux_field="von_mises_stress")
    [cache_file] = (tmp_path / "derived").glob("norm_*.npz")
    cache_file.write_bytes(b"truncated garbage, not a zip")

    # A corrupt cache must degrade to recompute (and heal the file), not crash.
    recovered = cached_compute_stats([traj], dataset_root=tmp_path, aux_field="von_mises_stress")
    np.testing.assert_array_equal(recovered.aux_mean, good.aux_mean)
    healed = NormalizationStats.load(cache_file)
    np.testing.assert_array_equal(healed.aux_mean, good.aux_mean)


def test_cached_stats_survives_unwritable_cache(tmp_path, monkeypatch):
    def _raise(self, path):
        raise OSError("read-only dataset root")

    monkeypatch.setattr(NormalizationStats, "save", _raise)
    traj = _known_vm_traj()
    stats = cached_compute_stats([traj], dataset_root=tmp_path, aux_field="von_mises_stress")  # must not raise
    np.testing.assert_array_equal(stats.aux_mean, compute_stats([traj]).aux_mean)
