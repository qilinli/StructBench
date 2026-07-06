"""Windowed autoregressive training samples from particle trajectories.

A sample is a window of ``window`` consecutive positions plus the next
position and next auxiliary value, for every particle in one trajectory. The
collate function concatenates particles across a batch into one big graph, as
the GNS expects, tracking how many particles each example contributed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .canonical import CaseTrajectory


class WindowDataset(Dataset):
    """Autoregressive ``(position_seq, next_position, next_aux)`` samples.

    Each item corresponds to one prediction step for one trajectory: the
    ``window`` frames immediately preceding the target frame are packed as the
    input sequence, and the target frame's position and auxiliary field value
    are the labels.

    Parameters
    ----------
    trajectories:
        Collection of :class:`~structbench.datasets.canonical.CaseTrajectory`
        objects to draw samples from.
    window:
        Number of consecutive input frames per sample.

    Notes
    -----
    For a trajectory with ``T`` frames and a given ``window`` the number of
    samples is ``T - window``.  The first valid target index is ``window``
    (0-based), the last is ``T - 1``.
    """

    def __init__(self, trajectories: list[CaseTrajectory], window: int) -> None:
        self._window = window
        # index: list of (traj, t) where t is the index of the predicted frame.
        # Interleave across trajectories (t-major, traj-minor) so that a
        # shuffle=False DataLoader places one sample per trajectory in each
        # batch when all trajectories share the same length.
        self._index: list[tuple[CaseTrajectory, int]] = []
        if trajectories:
            max_frames = max(tr.positions.shape[0] for tr in trajectories)
            for t in range(window, max_frames):
                for tr in trajectories:
                    if t < tr.positions.shape[0]:
                        self._index.append((tr, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | int]:
        """Return one sample as a dict of tensors.

        Parameters
        ----------
        i:
            Sample index.

        Returns
        -------
        dict
            ``position_seq``: Tensor of shape ``(P, window, dim)``, mm.
            ``particle_type``: LongTensor of shape ``(P,)``.
            ``next_position``: Tensor of shape ``(P, dim)``, mm.
            ``next_aux``: Tensor of shape ``(P,)``; auxiliary target, units are
            benchmark-dependent (e.g. MPa for von Mises stress, dimensionless
            for max principal strain).
            ``n_particles``: int number of particles ``P``.
        """
        tr, t = self._index[i]
        w = self._window
        seq = tr.positions[t - w : t]  # (window, P, dim)
        seq = np.transpose(seq, (1, 0, 2))  # (P, window, dim)
        return {
            "position_seq": torch.from_numpy(np.ascontiguousarray(seq)),
            "particle_type": torch.from_numpy(tr.particle_type),
            "next_position": torch.from_numpy(tr.positions[t]),
            "next_aux": torch.from_numpy(tr.aux[t]),
            "n_particles": int(tr.positions.shape[1]),
        }


def collate_samples(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Concatenate per-example particle rows into one batched graph.

    All particle arrays are concatenated along dimension 0 so the resulting
    tensors have ``sum(n_particles)`` rows.  The per-example particle counts
    are preserved in ``n_particles_per_example`` so downstream code can split
    the batch back into individual graphs.

    Parameters
    ----------
    batch:
        List of sample dicts as returned by :meth:`WindowDataset.__getitem__`.

    Returns
    -------
    dict
        ``position_seq``: Tensor ``(sum_P, window, dim)``, mm.
        ``particle_type``: LongTensor ``(sum_P,)``.
        ``next_position``: Tensor ``(sum_P, dim)``, mm.
        ``next_aux``: Tensor ``(sum_P,)``; auxiliary target, benchmark-dependent units.
        ``n_particles_per_example``: LongTensor ``(B,)`` — particle count per
        example.
    """
    return {
        "position_seq": torch.cat([b["position_seq"] for b in batch], dim=0),
        "particle_type": torch.cat([b["particle_type"] for b in batch], dim=0),
        "next_position": torch.cat([b["next_position"] for b in batch], dim=0),
        "next_aux": torch.cat([b["next_aux"] for b in batch], dim=0),
        "n_particles_per_example": torch.tensor(
            [b["n_particles"] for b in batch], dtype=torch.long
        ),
    }
