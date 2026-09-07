"""Windowed autoregressive training samples from particle trajectories.

A sample is a window of ``input_frames`` consecutive positions plus the next
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
    ``input_frames`` frames immediately preceding the target frame are packed
    as the input sequence, and the target frame's position and auxiliary field
    value are the labels.

    Parameters
    ----------
    trajectories:
        Collection of :class:`~structbench.datasets.canonical.CaseTrajectory`
        objects to draw samples from.
    input_frames:
        Number of consecutive input frames per sample (the model's history
        length; ADR-0035).
    target_frames:
        Number of consecutive target frames per sample (ADR-0050/0051). ``1``
        (default) is the autoregressive single-frame target and is
        byte-identical to the pre-0051 dataset — every ``k=1`` family
        (CGN/MGN/GeoFLARE) constructs this positionally and is unaffected.
        ``> 1`` makes each sample's target the span ``positions[t:t+target_frames]``
        / ``aux[t:t+target_frames]``. It is the model's ``k`` for the one-shot
        and single-bundle target, and ``2*k`` for the 1<k<T pushforward (which
        needs two consecutive bundles per sample), so it is named for the span
        it produces rather than the model's ``k``.

    Notes
    -----
    For a trajectory with ``T`` frames the number of samples is
    ``T - input_frames - (target_frames - 1)``; the last valid target-span
    start is ``T - target_frames``. At ``target_frames=1`` this is
    ``T - input_frames``, the pre-0051 count.
    """

    def __init__(
        self,
        trajectories: list[CaseTrajectory],
        input_frames: int,
        target_frames: int = 1,
    ) -> None:
        self._input_frames = input_frames
        self._target_frames = target_frames
        # index: list of (traj, t, traj_idx) where t is the index of the FIRST
        # predicted frame and traj_idx is trajectories' position in the input
        # list (so a mesh collate can look up each sample's static mesh data
        # by trajectory). Interleave across trajectories (t-major, traj-minor)
        # so that a shuffle=False DataLoader places one sample per trajectory
        # in each batch when all trajectories share the same length. The target
        # span positions[t:t+target_frames] needs t+target_frames <= T, so the
        # last start is T-target_frames; at target_frames=1 this reduces to the
        # pre-0051 range(input_frames, max_frames) with guard t < T
        # (byte-identical).
        self._index: list[tuple[CaseTrajectory, int, int]] = []
        if trajectories:
            max_frames = max(tr.positions.shape[0] for tr in trajectories)
            for t in range(input_frames, max_frames - target_frames + 1):
                for traj_idx, tr in enumerate(trajectories):
                    if t + target_frames <= tr.positions.shape[0]:
                        self._index.append((tr, t, traj_idx))

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
            ``position_seq``: Tensor of shape ``(P, input_frames, dim)``, mm.
            ``particle_type``: LongTensor of shape ``(P,)``.
            ``next_position``: Tensor of shape ``(P, dim)`` at
            ``target_frames=1``, mm; ``(P, target_frames, dim)`` otherwise.
            ``next_aux``: Tensor of shape ``(P, C)`` at ``target_frames=1``
            (``(P, target_frames, C)`` otherwise); auxiliary target channels
            (ADR-0059), per-channel units are benchmark-dependent (e.g. MPa
            for von Mises stress, dimensionless for max principal strain).
            ``input_aux``: Tensor of shape ``(P, C)`` — the aux state at
            the last input frame ``t - 1`` (ADR-0060 state-feedback input;
            additive to the sample contract).
            ``n_particles``: int number of particles ``P``.
            ``traj_idx``: int index of the source trajectory in the
            ``trajectories`` list passed to the constructor. Additive to the
            sample contract: unused by :func:`collate_samples` (the CGN
            path), consumed by the mesh collate
            (:func:`structbench.models.mgn.collate.collate_mesh_samples`) to
            look up each sample's static mesh data.
            ``target_frame``: int index of the (first) predicted frame ``t``.
            Additive to the sample contract: unused by :func:`collate_samples`
            and by the mesh collate unless ``include_target_frame=True`` (the
            time-conditioned path, ADR-0054).
        """
        tr, t, traj_idx = self._index[i]
        w = self._input_frames
        m = self._target_frames
        seq = tr.positions[t - w : t]  # (input_frames, P, dim)
        seq = np.transpose(seq, (1, 0, 2))  # (P, input_frames, dim)
        if m == 1:
            # Single-frame target: (P, dim) / (P, C).
            next_position = torch.from_numpy(tr.positions[t])
            next_aux = torch.from_numpy(tr.aux[t])
        else:
            # target span: (P, m, dim) / (P, m, C).
            next_position = torch.from_numpy(
                np.ascontiguousarray(np.transpose(tr.positions[t : t + m], (1, 0, 2)))
            )
            next_aux = torch.from_numpy(
                np.ascontiguousarray(np.transpose(tr.aux[t : t + m], (1, 0, 2)))
            )
        return {
            "position_seq": torch.from_numpy(np.ascontiguousarray(seq)),
            "particle_type": torch.from_numpy(tr.particle_type),
            "next_position": next_position,
            "next_aux": next_aux,
            # ADR-0060: the aux state at the LAST input frame, for the
            # state-feedback input (teacher-forced training). Additive to the
            # sample contract; ignored by consumers that don't ask for it.
            "input_aux": torch.from_numpy(np.ascontiguousarray(tr.aux[t - 1])),
            "n_particles": int(tr.positions.shape[1]),
            "traj_idx": traj_idx,
            # Index of the (first) predicted frame, used by the time-conditioned
            # collate (ADR-0054) to derive the normalized query time. Additive
            # to the sample contract: ignored by collate_samples (the CGN path)
            # and by the mesh collate unless it is asked for it.
            "target_frame": t,
        }


class FlowMapPairDataset(Dataset):
    """ADR-0062 ``(anchor t0, query t)`` pair samples for the anchored flow map.

    One sample = one uniformly drawn valid pair: ``t0`` ranges over
    ``[max(input_frames - 1, 1), T - 2]`` (the last seed frame is the
    deployment anchor, ADR-0035; the ``max(..., 1)`` guard keeps the FD
    partner frame ``t0 - 1`` in range on tiny fixtures) and ``t`` over
    ``[t0 + 1, T - 1]``, capped at ``t <= t0 + max_dt`` when ``max_dt > 0``.
    Uniform-over-pairs skews the ``Δt`` marginal toward short offsets
    (linearly, as availability does) — the recorded ADR-0062 v1 choice.

    The sample dict reuses the :class:`WindowDataset` key contract so
    :func:`collate_samples` and the mesh collate work unchanged:
    ``position_seq`` is the ANCHOR PAIR ``(P, 2, dim)`` — positions at
    ``t0 - 1`` and ``t0`` (the FD-velocity partner and the anchor frame) —
    and ``input_aux`` is the anchor's aux state ``aux[t0]``. The additive
    ``anchor_frame`` key carries ``t0`` (per example, like ``target_frame``).
    """

    def __init__(
        self,
        trajectories: list[CaseTrajectory],
        input_frames: int,
        max_dt: int = 0,
    ) -> None:
        if max_dt < 0:
            raise ValueError(f"max_dt must be >= 0 (0 = no cap), got {max_dt}")
        self._index: list[tuple[CaseTrajectory, int, int, int]] = []
        for traj_idx, tr in enumerate(trajectories):
            n = int(tr.positions.shape[0])
            for t0 in range(max(input_frames - 1, 1), n - 1):
                t_hi = n - 1 if max_dt == 0 else min(t0 + max_dt, n - 1)
                for t in range(t0 + 1, t_hi + 1):
                    self._index.append((tr, t0, t, traj_idx))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | int]:
        """Return one ``(anchor, query)`` sample (see class docstring)."""
        tr, t0, t, traj_idx = self._index[i]
        pair = tr.positions[t0 - 1 : t0 + 1]  # (2, P, dim)
        return {
            "position_seq": torch.from_numpy(
                np.ascontiguousarray(np.transpose(pair, (1, 0, 2)))
            ),
            "particle_type": torch.from_numpy(tr.particle_type),
            "next_position": torch.from_numpy(tr.positions[t]),
            "next_aux": torch.from_numpy(tr.aux[t]),
            # The anchor's aux state (ADR-0062), riding the ADR-0060 key so
            # the shared collates concatenate it unchanged.
            "input_aux": torch.from_numpy(np.ascontiguousarray(tr.aux[t0])),
            "n_particles": int(tr.positions.shape[1]),
            "traj_idx": traj_idx,
            "target_frame": t,
            # Anchor frame index (per example, like target_frame): the
            # trainer derives Δt = target_frame - anchor_frame and the
            # anchor-time feature from it. Additive to the sample contract.
            "anchor_frame": t0,
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
        ``position_seq``: Tensor ``(sum_P, input_frames, dim)``, mm.
        ``particle_type``: LongTensor ``(sum_P,)``.
        ``next_position``: Tensor ``(sum_P, dim)``, mm.
        ``next_aux``: Tensor ``(sum_P, C)``; auxiliary target channels
        (ADR-0059), per-channel benchmark-dependent units.
        ``input_aux``: Tensor ``(sum_P, C)``; the aux state at each sample's
        last input frame (ADR-0060).
        ``n_particles_per_example``: LongTensor ``(B,)`` — particle count per
        example.
    """
    return {
        "position_seq": torch.cat([b["position_seq"] for b in batch], dim=0),
        "particle_type": torch.cat([b["particle_type"] for b in batch], dim=0),
        "next_position": torch.cat([b["next_position"] for b in batch], dim=0),
        "next_aux": torch.cat([b["next_aux"] for b in batch], dim=0),
        # ADR-0060 (tolerant: hand-built sample dicts may omit the key)
        **(
            {"input_aux": torch.cat([b["input_aux"] for b in batch], dim=0)}
            if "input_aux" in batch[0]
            else {}
        ),
        "n_particles_per_example": torch.tensor(
            [b["n_particles"] for b in batch], dtype=torch.long
        ),
    }
