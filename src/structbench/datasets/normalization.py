"""Velocity/acceleration normalization statistics over a set of trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .canonical import CaseTrajectory


@dataclass
class NormalizationStats:
    """Per-dimension mean/std of velocity and acceleration (mm/frame, mm/frame^2)."""

    velocity_mean: NDArray[np.float64]
    velocity_std: NDArray[np.float64]
    acceleration_mean: NDArray[np.float64]
    acceleration_std: NDArray[np.float64]

    def save(self, path: str | Path) -> None:
        """Write the four arrays to a ``.npz`` file."""
        np.savez(
            path,
            velocity_mean=self.velocity_mean,
            velocity_std=self.velocity_std,
            acceleration_mean=self.acceleration_mean,
            acceleration_std=self.acceleration_std,
        )

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        """Read stats back from a ``.npz`` file written by :meth:`save`."""
        d = np.load(path)
        return cls(
            d["velocity_mean"], d["velocity_std"],
            d["acceleration_mean"], d["acceleration_std"],
        )


def compute_stats(trajectories: list[CaseTrajectory]) -> NormalizationStats:
    """Pool velocity/acceleration stats over all particles, frames, and cases.

    Velocity is the first finite difference of positions along the frame axis;
    acceleration is the second. Statistics are stacked over every particle in
    every frame of every trajectory.

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
        ``(dim,)``, pooled over all particles, frames, and cases.
    """
    vels, accs = [], []
    for tr in trajectories:
        p = tr.positions.astype(np.float64)  # (T, P, dim)
        v = p[1:] - p[:-1]  # (T-1, P, dim)
        a = v[1:] - v[:-1]  # (T-2, P, dim)
        vels.append(v.reshape(-1, p.shape[-1]))
        accs.append(a.reshape(-1, p.shape[-1]))
    v_all = np.concatenate(vels, axis=0)
    a_all = np.concatenate(accs, axis=0)
    return NormalizationStats(
        velocity_mean=v_all.mean(0), velocity_std=v_all.std(0),
        acceleration_mean=a_all.mean(0), acceleration_std=a_all.std(0),
    )
