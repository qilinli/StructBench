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


def _agg_standardize(
    bond: Tensor,
    valid: Tensor,
    idx: Tensor,
    clamp: float,
    smooth: bool,
    robust: bool,
) -> Tensor:
    """Aggregate a per-bond rate ``(N, eff_k)`` to a standardized ``(N, 1)`` channel.

    Shared by the single-channel :func:`_segment_phi` and the multi-channel
    :func:`_segment_phi_multi` so every channel is reduced identically:
    masked neighbour aggregation (mean, or **median** if ``robust``), optional
    spatial ``smooth`` low-pass over the same kNN set, then per-example
    mean/std (or **median/IQR** if ``robust``) standardization and a symmetric
    ``clamp``.
    """
    if robust:
        phi = torch.nanmedian(
            bond.masked_fill(~valid, float("nan")), dim=1
        ).values.unsqueeze(1)  # (N, 1) median over valid neighbours
    else:
        count = valid.sum(dim=1).clamp_min(1)
        phi = ((bond * valid).sum(dim=1) / count).unsqueeze(1)
    if smooth:  # spatial low-pass: masked mean of φ over the same neighbours
        phi_nb = phi[idx].squeeze(-1)  # (N, eff_k)
        phi = (phi_nb * valid).sum(dim=1, keepdim=True) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
    if robust:
        med = phi.median()
        iqr = phi.quantile(0.75) - phi.quantile(0.25)
        phi = (phi - med) / (iqr + 1e-6)
    else:
        phi = (phi - phi.mean()) / (phi.std() + 1e-6)  # per-example standardize
    return phi.clamp(-clamp, clamp)


def _segment_phi(
    x_e: Tensor,
    v_e: Tensor,
    k: int,
    clamp: float,
    smooth: bool = False,
    robust: bool = False,
) -> Tensor:
    """Standardized strain-rate proxy for one example, ``(N, 1)``.

    ``smooth`` adds a spatial neighbour-average pass over the same kNN set (a
    spatial low-pass that damps the ``dv/dx`` high-pass amplification of
    rollout drift, ADR-SRO Cause-1 fix). ``robust`` swaps the neighbour mean
    and the mean/std standardization for **median / IQR** order statistics, so
    a few blown-up points cannot shift every point's φ. Both default off ⇒
    the raw ADR-0047 behaviour.
    """
    n = x_e.shape[0]
    if n < 2:
        return x_e.new_zeros((n, 1))
    idx = knn_neighbor_indices(x_e, k)  # (N, eff_k)
    rows = torch.arange(n, device=x_e.device).unsqueeze(1)  # (N, 1)
    valid = idx != rows  # exclude self by index equality (not by column 0)
    dx = torch.linalg.vector_norm(x_e[idx] - x_e.unsqueeze(1), dim=-1)  # (N, eff_k)
    dv = torch.linalg.vector_norm(v_e[idx] - v_e.unsqueeze(1), dim=-1)  # (N, eff_k)
    ratio = dv / dx.clamp_min(1e-6)  # (N, eff_k) local velocity gradient
    return _agg_standardize(ratio, valid, idx, clamp, smooth, robust)


def _segment_phi_multi(
    x_e: Tensor,
    v_e: Tensor,
    k: int,
    clamp: float,
    channels: int,
    smooth: bool = False,
    robust: bool = False,
) -> Tensor:
    """Multi-channel strain-rate proxy for one example, ``(N, channels)``.

    Decomposes each neighbour bond's relative velocity into components along
    vs. perpendicular to the bond direction — a rotation-invariant split with
    **no matrix inverse** (numerically safer than a least-squares velocity
    gradient). Channels, in order:

    0. **magnitude** ``‖Δv‖/‖Δx‖`` — identical to :func:`_segment_phi`'s scalar.
    1. **volumetric** ``(Δv·x̂)/‖Δx‖`` — signed along-bond rate ≈ local
       divergence (negative under compaction).
    2. **shear** ``‖Δv_⊥‖/‖Δx‖`` — deviatoric / distortion rate.

    ``channels`` selects the leading ``channels`` of these three. Each channel
    is reduced by :func:`_agg_standardize`, so ``smooth``/``robust`` apply
    per-channel exactly as in the single-channel path.
    """
    n = x_e.shape[0]
    if n < 2:
        return x_e.new_zeros((n, channels))
    idx = knn_neighbor_indices(x_e, k)  # (N, eff_k)
    rows = torch.arange(n, device=x_e.device).unsqueeze(1)  # (N, 1)
    valid = idx != rows  # exclude self by index equality
    dx_vec = x_e[idx] - x_e.unsqueeze(1)  # (N, eff_k, dim)
    dv_vec = v_e[idx] - v_e.unsqueeze(1)  # (N, eff_k, dim)
    dx = torch.linalg.vector_norm(dx_vec, dim=-1)  # (N, eff_k)
    inv_dx = dx.clamp_min(1e-6).reciprocal()  # (N, eff_k)
    xhat = dx_vec * inv_dx.unsqueeze(-1)  # (N, eff_k, dim) unit bond
    v_along = (dv_vec * xhat).sum(dim=-1)  # (N, eff_k) signed along-bond rate
    v_perp = torch.linalg.vector_norm(
        dv_vec - v_along.unsqueeze(-1) * xhat, dim=-1
    )  # (N, eff_k) perpendicular (shear) component
    dv = torch.linalg.vector_norm(dv_vec, dim=-1)  # (N, eff_k)
    bonds = [dv * inv_dx, v_along * inv_dx, v_perp * inv_dx]  # mag, volumetric, shear
    cols = [
        _agg_standardize(bonds[c], valid, idx, clamp, smooth, robust)
        for c in range(channels)
    ]
    return torch.cat(cols, dim=1)  # (N, channels)


def strain_rate_phi(
    x: Tensor,
    v: Tensor,
    n_particles_per_example: Tensor | None,
    k: int,
    clamp: float,
    smooth: bool = False,
    robust: bool = False,
    channels: int = 1,
) -> Tensor:
    """Per-point strain-rate activity field ``phi``, computed per example.

    With ``channels == 1`` (default) this is the single scalar
    ``phi_i = mean_j ||v_i - v_j|| / max(||x_i - x_j||, 1e-6)`` over the ``k``
    nearest neighbours ``j`` (self excluded), then per-example standardized and
    clamped — byte-identical to the pre-multi-channel behaviour. With
    ``channels > 1`` the per-bond relative velocity is decomposed into
    magnitude / volumetric / shear channels (see :func:`_segment_phi_multi`).
    Both ``x`` and ``v`` must come from the SAME (possibly noisy) window as the
    model's inputs, and neither may be the scored target.

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
    channels:
        Number of φ channels, 1 (scalar magnitude) or up to 3
        (magnitude, volumetric, shear).

    Returns
    -------
    Tensor
        ``(P, channels)`` standardized strain-rate proxy, on ``x``'s dtype and
        device.
    """

    def seg(x_e: Tensor, v_e: Tensor) -> Tensor:
        if channels == 1:
            return _segment_phi(x_e, v_e, k, clamp, smooth, robust)
        return _segment_phi_multi(x_e, v_e, k, clamp, channels, smooth, robust)

    if n_particles_per_example is None:
        return seg(x, v)
    out = x.new_empty((x.shape[0], channels))
    start = 0
    for count in n_particles_per_example.tolist():
        end = start + int(count)
        out[start:end] = seg(x[start:end], v[start:end])
        start = end
    return out
