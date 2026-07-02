"""Velocity/acceleration normalization statistics over a set of trajectories."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .canonical import CaseTrajectory

logger = logging.getLogger(__name__)


@dataclass
class NormalizationStats:
    """Mean/std of velocity, acceleration, and the auxiliary field.

    Velocity and acceleration carry a per-dimension mean/std (mm/frame,
    mm/frame^2). The auxiliary (von Mises stress) field carries a scalar
    mean/std (shape ``(1,)``, MPa) so its training target can be normalized to
    O(1), balancing the dual position/auxiliary loss.
    """

    velocity_mean: NDArray[np.float64]
    velocity_std: NDArray[np.float64]
    acceleration_mean: NDArray[np.float64]
    acceleration_std: NDArray[np.float64]
    aux_mean: NDArray[np.float64]
    aux_std: NDArray[np.float64]

    def save(self, path: str | Path) -> None:
        """Write the six arrays to a ``.npz`` file."""
        np.savez(
            path,
            velocity_mean=self.velocity_mean,
            velocity_std=self.velocity_std,
            acceleration_mean=self.acceleration_mean,
            acceleration_std=self.acceleration_std,
            aux_mean=self.aux_mean,
            aux_std=self.aux_std,
        )

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        """Read stats back from a ``.npz`` file written by :meth:`save`."""
        d = np.load(path)
        return cls(
            d["velocity_mean"],
            d["velocity_std"],
            d["acceleration_mean"],
            d["acceleration_std"],
            d["aux_mean"],
            d["aux_std"],
        )


def compute_stats(trajectories: list[CaseTrajectory]) -> NormalizationStats:
    """Pool velocity/acceleration/aux stats over all particles, frames, and cases.

    Velocity is the first finite difference of positions along the frame axis;
    acceleration is the second. The auxiliary (von Mises stress) field is a
    direct quantity, so its stats are pooled over the raw values with no finite
    difference. Statistics are stacked over every particle in every frame of
    every trajectory.

    Parameters
    ----------
    trajectories:
        List of :class:`~structbench.datasets.canonical.CaseTrajectory` objects.
        Each must have at least 3 frames (``T >= 3``) so that both velocity and
        acceleration samples exist.

    Returns
    -------
    NormalizationStats
        Per-dimension mean and std for velocity ``(dim,)`` and acceleration
        ``(dim,)``, plus scalar mean and std for the auxiliary field ``(1,)``,
        pooled over all particles, frames, and cases.
    """
    vels, accs, auxs = [], [], []
    for tr in trajectories:
        p = tr.positions.astype(np.float64)  # (T, P, dim)
        v = p[1:] - p[:-1]  # (T-1, P, dim)
        a = v[1:] - v[:-1]  # (T-2, P, dim)
        vels.append(v.reshape(-1, p.shape[-1]))
        accs.append(a.reshape(-1, p.shape[-1]))
        auxs.append(tr.von_mises.astype(np.float64).reshape(-1))  # (T*P,)
    v_all = np.concatenate(vels, axis=0)
    a_all = np.concatenate(accs, axis=0)
    aux_all = np.concatenate(auxs, axis=0)
    return NormalizationStats(
        velocity_mean=v_all.mean(0),
        velocity_std=v_all.std(0),
        acceleration_mean=a_all.mean(0),
        acceleration_std=a_all.std(0),
        aux_mean=np.array([aux_all.mean()]),
        aux_std=np.array([aux_all.std()]),
    )


def cached_compute_stats(
    trajectories: list[CaseTrajectory],
    *,
    dataset_root: str | Path,
) -> NormalizationStats:
    """:func:`compute_stats` with a dataset-level cache keyed by the split.

    The cache lives at ``<dataset_root>/derived/norm_<key>.npz``, where the key
    hashes the trajectories' case-id list — so the stats are computed once per
    split and reused across runs, and a changed case-id list forces
    recomputation under a new filename. The cache never blocks training: a
    write failure (e.g. read-only dataset root) degrades to a warning, and an
    unreadable/corrupt cache file is recomputed and rewritten. Writes go
    through a temp file + atomic rename so a killed run cannot leave a
    truncated cache behind.

    Parameters
    ----------
    trajectories:
        Trajectories of the (train) split, as for :func:`compute_stats`.
    dataset_root:
        Directory the ``derived/`` cache folder lives under — normally the
        directory holding the split's ``<case_id>.h5`` files.

    Returns
    -------
    NormalizationStats
    """
    case_ids = [trajectory.case_id for trajectory in trajectories]
    key = hashlib.sha256("\n".join(case_ids).encode("utf-8")).hexdigest()[:12]
    cache_path = Path(dataset_root) / "derived" / f"norm_{key}.npz"
    if cache_path.exists():
        try:
            stats = NormalizationStats.load(cache_path)
            logger.info("normalization stats: cache hit at %s", cache_path)
            return stats
        except Exception as exc:  # noqa: BLE001 - corrupt/stale cache, any form
            logger.warning(
                "normalization stats: unreadable cache %s (%s); recomputing",
                cache_path,
                exc,
            )

    stats = compute_stats(trajectories)
    # Unique temp name (ends in .npz so np.savez appends nothing), then an
    # atomic replace: concurrent or killed runs can never leave a truncated
    # file at the final path.
    tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp{os.getpid()}.npz")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        stats.save(tmp_path)
        os.replace(tmp_path, cache_path)
        logger.info("normalization stats: cached at %s", cache_path)
    except OSError as exc:
        logger.warning("normalization stats: cache write failed (%s)", exc)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return stats
