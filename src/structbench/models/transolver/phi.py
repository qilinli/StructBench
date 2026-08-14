"""Input-only kinematic activity field ``phi`` for physics-conditioned slicing.

``phi_i`` is a per-point strain-rate proxy: the local velocity-gradient
magnitude over a point's spatial neighbours, computed from the velocity-history
window. It conditions the Transolver slice assignment (``persistent`` mode) or
is appended as a node feature (``feature`` mode). It is a function of positions
and velocities (legal model inputs) ONLY — never the stress/strain target — so
it is leakage-free. Computed once per rollout step, per example (neighbours
never cross concatenated examples, mirroring the network's segment loop, so a
batched forward matches the per-example forward).
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Query-axis chunk for the ``cdist`` neighbour search (memory bound only;
#: correctness is independent of the value), mirroring
#: :data:`structbench.models.geoflare.geo_ops._QUERY_CHUNK`.
_QUERY_CHUNK = 2048


def knn_neighbor_indices(coords: Tensor, k: int) -> Tensor:
    """Nearest-neighbour INDICES for one example (self excluded by the caller).

    Chunked ``torch.cdist`` + ``topk`` mirroring
    :func:`structbench.models.geoflare.geo_ops.ball_query`, but returning the
    neighbour *indices* (which ``ball_query`` discards) so a caller can gather
    neighbour velocities for a velocity gradient. Returns ``k + 1`` nearest
    including the self match; the caller drops self by index equality.

    Parameters
    ----------
    coords:
        ``(N, dim)`` point coordinates, a single example.
    k:
        Neighbour count with self excluded, so ``min(k + 1, N)`` indices are
        returned per point.

    Returns
    -------
    Tensor
        ``(N, eff_k)`` int64 neighbour indices, nearest first, with
        ``eff_k = min(k + 1, N)``.
    """
    n = coords.shape[0]
    eff_k = min(k + 1, n)
    idx = torch.empty((n, eff_k), dtype=torch.long, device=coords.device)
    for start in range(0, n, _QUERY_CHUNK):
        chunk = coords[start : start + _QUERY_CHUNK]
        dist = torch.cdist(chunk, coords)  # (chunk, N)
        _, top_idx = torch.topk(dist, eff_k, dim=-1, largest=False)
        idx[start : start + chunk.shape[0]] = top_idx
    return idx


def _segment_phi(x_e: Tensor, v_e: Tensor, k: int, clamp: float) -> Tensor:
    """Standardized strain-rate proxy for one example, ``(N, 1)``."""
    n = x_e.shape[0]
    if n < 2:
        return x_e.new_zeros((n, 1))
    idx = knn_neighbor_indices(x_e, k)  # (N, eff_k)
    rows = torch.arange(n, device=x_e.device).unsqueeze(1)  # (N, 1)
    valid = idx != rows  # exclude self by index equality (not by column 0)
    dx = torch.linalg.vector_norm(x_e[idx] - x_e.unsqueeze(1), dim=-1)  # (N, eff_k)
    dv = torch.linalg.vector_norm(v_e[idx] - v_e.unsqueeze(1), dim=-1)  # (N, eff_k)
    ratio = (dv / dx.clamp_min(1e-6)) * valid  # zero invalid (self) slots
    count = valid.sum(dim=1).clamp_min(1)  # masked mean over >= 1 neighbours
    phi = (ratio.sum(dim=1) / count).unsqueeze(1)  # (N, 1)
    phi = (phi - phi.mean()) / (phi.std() + 1e-6)  # per-example standardize
    return phi.clamp(-clamp, clamp)


def strain_rate_phi(
    x: Tensor,
    v: Tensor,
    n_particles_per_example: Tensor | None,
    k: int,
    clamp: float,
) -> Tensor:
    """Per-point strain-rate activity field ``phi``, computed per example.

    ``phi_i = mean_j ||v_i - v_j|| / max(||x_i - x_j||, 1e-6)`` over the ``k``
    nearest neighbours ``j`` (self excluded), then per-example standardized and
    clamped. Both ``x`` and ``v`` must come from the SAME (possibly noisy)
    window as the model's inputs, and neither may be the scored target.

    Parameters
    ----------
    x:
        ``(P, dim)`` current positions (examples concatenated along dim 0).
    v:
        ``(P, dim)`` current-frame velocities (the most recent window
        difference).
    n_particles_per_example:
        ``(B,)`` per-example particle counts, or ``None`` for a single example.
    k:
        Neighbour count for the local velocity gradient.
    clamp:
        Symmetric clamp on the standardized ``phi`` (rollout-stability guard).

    Returns
    -------
    Tensor
        ``(P, 1)`` standardized strain-rate proxy, on ``x``'s dtype and device.
    """
    if n_particles_per_example is None:
        return _segment_phi(x, v, k, clamp)
    out = x.new_empty((x.shape[0], 1))
    start = 0
    for count in n_particles_per_example.tolist():
        end = start + int(count)
        out[start:end] = _segment_phi(x[start:end], v[start:end], k, clamp)
        start = end
    return out
