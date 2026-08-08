"""Model-owned mesh collate: batches windowed samples with static mesh data.

``WindowDataset`` (``structbench.datasets.particle``) is benchmark-generic and
knows nothing about meshes; it only tags each sample with the ``traj_idx`` of
its source trajectory. The MGN-specific step of attaching each trajectory's
static mesh connectivity (edges, reference coordinates) and node-offsetting
those edges to build one collated graph lives here, next to the model that
consumes it (ADR-0020 precedent, matching ``mesh_ops.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ...datasets import CaseTrajectory, collate_samples
from .mesh_ops import cells_to_edges


@dataclass(frozen=True)
class MeshStatic:
    """One trajectory's static mesh data: connectivity and rest coordinates.

    Parameters
    ----------
    mesh_edge_index:
        ``(2, Em)`` int64 bidirectional mesh-edge index, node indices local
        to this trajectory (0-based).
    reference_coords:
        ``(P, dim)`` float32 mesh-space (rest/reference) coordinates.
    """

    mesh_edge_index: Tensor
    reference_coords: Tensor


def mesh_static_from_trajectory(traj: CaseTrajectory) -> MeshStatic:
    """Build a trajectory's static mesh data from its cell connectivity.

    Parameters
    ----------
    traj:
        Source trajectory. Must be a mesh (nodal-FE) trajectory — i.e. both
        ``traj.cells`` and ``traj.reference_coords`` are populated (ADR-0043);
        SPH trajectories have neither.

    Returns
    -------
    MeshStatic
        ``mesh_edge_index`` from :func:`~.mesh_ops.cells_to_edges` applied to
        ``traj.cells``; ``reference_coords`` from ``traj.reference_coords``.

    Raises
    ------
    ValueError
        If ``traj.cells`` or ``traj.reference_coords`` is ``None`` (the
        trajectory is not a mesh benchmark).
    """
    if traj.cells is None or traj.reference_coords is None:
        raise ValueError(
            f"trajectory {traj.case_id!r} has no cells/reference_coords; "
            "mesh_static_from_trajectory requires a mesh (nodal-FE) "
            "trajectory (ADR-0043)"
        )
    return MeshStatic(
        mesh_edge_index=cells_to_edges(torch.from_numpy(traj.cells)),
        reference_coords=torch.from_numpy(traj.reference_coords),
    )


def collate_mesh_samples(batch: list[dict], statics: Sequence[MeshStatic]) -> dict:
    """Collate a batch of windowed samples into one mesh-batched graph.

    Calls :func:`~structbench.datasets.particle.collate_samples` for the
    shared keys, then appends the batched mesh-edge index and reference
    coordinates.

    Parameters
    ----------
    batch:
        List of sample dicts as returned by
        :meth:`~structbench.datasets.particle.WindowDataset.__getitem__`;
        each must carry a ``"traj_idx"`` key.
    statics:
        Per-trajectory static mesh data, indexed by each sample's
        ``"traj_idx"`` (i.e. ``statics[sample["traj_idx"]]``) — NOT by the
        sample's position in ``batch``.

    Returns
    -------
    dict
        Every key :func:`~structbench.datasets.particle.collate_samples`
        returns, plus:
        ``mesh_edge_index``: ``(2, sum_Em)`` int64 — each sample's static
        mesh edges, offset by the cumulative particle count of the batch's
        preceding samples (batch order), then concatenated. An edge never
        crosses a sample's particle-row range.
        ``reference_coords``: ``(sum_P, dim)`` float32 — each sample's static
        reference coordinates, row-concatenated in batch order.
    """
    out: dict = dict(collate_samples(batch))

    edge_parts: list[Tensor] = []
    coord_parts: list[Tensor] = []
    offset = 0
    for sample in batch:
        static = statics[sample["traj_idx"]]
        edge_parts.append(static.mesh_edge_index + offset)
        coord_parts.append(static.reference_coords)
        offset += sample["n_particles"]

    out["mesh_edge_index"] = torch.cat(edge_parts, dim=1)
    out["reference_coords"] = torch.cat(coord_parts, dim=0)
    return out
