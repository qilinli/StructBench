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
from ..common.two_site import cell_weights as _cell_weights
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
    cells:
        ``(E, k)`` int64 cell connectivity, node indices local to this
        trajectory (ADR-0075: the element site's geometry).
    cell_weights:
        ``(E,)`` float32 positive cell weights of the element-to-node
        projection (the loader's area-weighted mean).
    """

    mesh_edge_index: Tensor
    reference_coords: Tensor
    cells: Tensor | None = None
    cell_weights: Tensor | None = None


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
    cells = torch.from_numpy(traj.cells)
    reference = torch.from_numpy(traj.reference_coords)
    return MeshStatic(
        mesh_edge_index=cells_to_edges(cells, dim=int(traj.reference_coords.shape[-1])),
        reference_coords=reference,
        cells=cells,
        cell_weights=_cell_weights(reference, cells).to(torch.float32),
    )


#: Per-trajectory constitutive law for the collate (ADR-0075): Young's
#: modulus, Poisson's ratio, ``(peeq knots, sigma_y knots)``; every law of
#: a batch must carry the same number of knots.
LawTuple = tuple[float, float, tuple[float, ...], tuple[float, ...]]


def collate_mesh_samples(
    batch: list[dict],
    statics: Sequence[MeshStatic],
    loading_scalars: Sequence[float] | None = None,
    include_target_frame: bool = False,
    include_anchor_frame: bool = False,
    include_chain_frame: bool = False,
    include_elements: bool = False,
    laws: Sequence[LawTuple] | None = None,
) -> dict:
    """Collate a batch of windowed samples into one mesh-batched graph.

    Calls :func:`~structbench.datasets.particle.collate_samples` for the
    shared keys, then appends the batched mesh-edge index and reference
    coordinates.

    Parameters
    ----------
    batch:
        List of sample dicts as returned by
        :meth:`~structbench.datasets.particle.WindowDataset.__getitem__` or
        :meth:`~structbench.datasets.particle.FlowMapPairDataset.__getitem__`
        (ADR-0062 — whose ``position_seq``/``input_aux`` carry anchor-pair
        semantics; see :func:`~structbench.datasets.particle.collate_samples`);
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
        With ``include_elements`` (ADR-0075): ``cells`` ``(sum_E, k)`` int64
        node-offset like the edges, ``cell_weights`` ``(sum_E,)`` and
        ``n_elements_per_example`` ``(B,)``; with ``laws`` also the per-
        element law rows ``law_youngs``, ``law_poisson`` ``(sum_E,)`` and
        ``law_knots_peeq``, ``law_knots_sy`` ``(sum_E, K)``, each sample's
        law (by ``traj_idx``) repeated over its elements.

    Raises
    ------
    ValueError
        If a sample's ``statics[traj_idx].reference_coords`` row count
        disagrees with the sample's own ``n_particles`` — a static/sample
        misalignment (e.g. ``statics`` built from a different trajectory
        list than the one passed to ``WindowDataset``) that would otherwise
        silently offset edges against the wrong particle rows.
    """
    out: dict = dict(collate_samples(batch))

    edge_parts: list[Tensor] = []
    coord_parts: list[Tensor] = []
    offset = 0
    for sample in batch:
        static = statics[sample["traj_idx"]]
        if static.reference_coords.shape[0] != sample["n_particles"]:
            raise ValueError(
                f"traj_idx={sample['traj_idx']}: static reference_coords has "
                f"{static.reference_coords.shape[0]} rows but the sample reports "
                f"n_particles={sample['n_particles']}; statics is misaligned with "
                "the trajectory list passed to WindowDataset"
            )
        edge_parts.append(static.mesh_edge_index + offset)
        coord_parts.append(static.reference_coords)
        offset += sample["n_particles"]

    out["mesh_edge_index"] = torch.cat(edge_parts, dim=1)
    out["reference_coords"] = torch.cat(coord_parts, dim=0)

    if include_elements:
        cell_parts: list[Tensor] = []
        weight_parts: list[Tensor] = []
        counts: list[int] = []
        offset = 0
        for sample in batch:
            static = statics[sample["traj_idx"]]
            if static.cells is None or static.cell_weights is None:
                raise ValueError(
                    f"traj_idx={sample['traj_idx']}: include_elements needs statics "
                    "built by mesh_static_from_trajectory (cells and cell_weights)"
                )
            cell_parts.append(static.cells + offset)
            weight_parts.append(static.cell_weights)
            counts.append(int(static.cells.shape[0]))
            offset += sample["n_particles"]
        out["cells"] = torch.cat(cell_parts, dim=0)
        out["cell_weights"] = torch.cat(weight_parts, dim=0)
        out["n_elements_per_example"] = torch.tensor(counts, dtype=torch.long)
        if laws is not None:
            n_knots = {len(laws[sample["traj_idx"]][2]) for sample in batch}
            if len(n_knots) != 1:
                raise ValueError(
                    "every law in a batch must carry the same number of hardening "
                    f"knots (ADR-0075); got {sorted(n_knots)}"
                )
            youngs, poisson, kp, ks = [], [], [], []
            for sample, n in zip(batch, counts, strict=True):
                e, nu, knots_peeq, knots_sy = laws[sample["traj_idx"]]
                youngs.append(torch.full((n,), float(e), dtype=torch.float32))
                poisson.append(torch.full((n,), float(nu), dtype=torch.float32))
                kp.append(torch.tensor(knots_peeq, dtype=torch.float32).expand(n, -1))
                ks.append(torch.tensor(knots_sy, dtype=torch.float32).expand(n, -1))
            out["law_youngs"] = torch.cat(youngs, dim=0)
            out["law_poisson"] = torch.cat(poisson, dim=0)
            out["law_knots_peeq"] = torch.cat(kp, dim=0)
            out["law_knots_sy"] = torch.cat(ks, dim=0)

    if loading_scalars is not None:
        # ADR-0051 B: broadcast each sample's scalar loading parameter (by
        # traj_idx) to its particle rows, giving a (sum_P, 1) global feature.
        out["loading_feature"] = torch.cat(
            [
                torch.full(
                    (sample["n_particles"], 1),
                    float(loading_scalars[sample["traj_idx"]]),
                    dtype=torch.float32,
                )
                for sample in batch
            ],
            dim=0,
        )

    if include_target_frame:
        # ADR-0054: one query-frame index per example (B,), for the
        # time-conditioned path's normalized query time. Per-EXAMPLE (not
        # per-particle): the network broadcasts each example's time embedding
        # to its own particle rows.
        out["target_frame"] = torch.tensor(
            [int(sample["target_frame"]) for sample in batch], dtype=torch.long
        )
    if include_anchor_frame:
        # ADR-0062: one anchor-frame index per example (B,), for the flow-map
        # path's Δt and anchor-time conditioning (per-EXAMPLE, as above).
        out["anchor_frame"] = torch.tensor(
            [int(sample["anchor_frame"]) for sample in batch], dtype=torch.long
        )
    if include_chain_frame:
        if "chain_frames" in batch[0]:
            # ADR-0063 amendment: G re-anchor frames per example, (B, G).
            out["chain_frames"] = torch.stack(
                [sample["chain_frames"] for sample in batch]
            )
        else:
            # ADR-0063: the mid-chain anchor frame t1 per example (B,) —
            # step B of the anchor-pushforward chain re-anchors here.
            out["chain_frame"] = torch.tensor(
                [int(sample["chain_frame"]) for sample in batch],
                dtype=torch.long,
            )
    return out
