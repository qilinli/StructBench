"""Ball query + per-example coordinate standardization (ADR-0041 step 3).

The two primitives underlying GeoFLARE's GALE context pathway (ADR-0045,
draft, landing later in this plan): deterministic nearest-first ball query
and the scalar-isotropic coordinate standardization that ball query's
radii are defined against.

Graph/neighbourhood construction lives with the model that uses it
(ADR-0020 precedent, ``models/mgn/mesh_ops.py``): ``ball_query`` here
reimplements that module's chunked-``torch.cdist`` idiom (``_QUERY_CHUNK``)
locally rather than importing it, since MGN's single-radius world-edge
search and GeoFLARE's fixed-``k``-per-scale ball query are different
queries with different output shapes and padding rules.

``ball_query``'s semantics are pinned to NVIDIA PhysicsNeMo's torch
fallback for ``GALE``/``GeoTransolver`` ball query (Apache-2.0 License,
Copyright (c) NVIDIA CORPORATION & AFFILIATES): nearest-first top-k
neighbours within a radius, on ABSOLUTE coordinates, zero-padded past
however many neighbours qualify. PhysicsNeMo's accelerated Warp backend is
documented upstream as returning neighbours in an ARBITRARY order
(hash-grid traversal with an early break; upstream explicitly disclaims
relying on the exact order) -- StructBench pins the deterministic torch
fallback's nearest-first order as its own declared semantics, not the
Warp backend's. See ``scratch/2026-08-09-geoflare-grounding.md`` S10 for
the verified citation trail.
"""

from __future__ import annotations

import torch
from torch import Tensor

_QUERY_CHUNK = 2048


def standardize_coords(coords: Tensor) -> Tensor:
    """Per-example coordinate standardization for the geometry pathway (ADR-0045 gap 8).

    Centers each axis to exact zero mean, then divides by a single SCALAR
    isotropic divisor: the population root-mean-square of the centered
    coordinates, pooled over every element (every point, every axis)
    rather than computed per axis. A per-axis divisor would rescale each
    axis independently and distort the point cloud's shape (e.g. turn a
    sphere into an ellipsoid); the scalar divisor applies one uniform
    zoom instead, so relative distances and angles between points are
    preserved.

    Parameters
    ----------
    coords:
        ``(P, dim)`` raw per-example coordinates.

    Returns
    -------
    Tensor
        ``(P, dim)`` standardized coordinates: exact per-axis zero mean
        by construction, and (away from the degenerate all-identical-point
        clamp) exact unit population RMS pooled over every element.
    """
    centered = coords - coords.mean(dim=0, keepdim=True)
    rms = centered.pow(2).mean().sqrt().clamp_min(1e-8)
    return centered / rms


def ball_query(coords: Tensor, radius: float, k: int) -> Tensor:
    """Nearest-first top-``k`` neighbours within ``radius``, for one example.

    Every row of ``coords`` is queried against every row of the SAME
    ``coords`` (a self-query: a point is always its own distance-0
    neighbour and so, absent a tie, always its own nearest neighbour).
    Distances are computed via a chunked ``torch.cdist`` over the query
    axis (``_QUERY_CHUNK`` rows at a time, mirroring the chunking idiom in
    :func:`structbench.models.mgn.mesh_ops.world_edges` -- memory-bounded
    only; correctness does not depend on the chunk size). Per query row,
    the ``k`` smallest distances are taken (``torch.topk(..., largest=
    False)``, which returns its result sorted ascending -- nearest
    first). Any of those ``k`` slots whose distance exceeds ``radius`` is
    then zeroed out (coordinate row ``(0, 0, 0)``): the radius is a hard
    cutoff applied to the top-``k`` selection, not a pre-filter, so a
    point just outside ``radius`` is zero-padded even when the top-``k``
    would otherwise have room for it. The same zero-row padding covers
    the case where fewer than ``k`` points exist at all
    (``coords.shape[0] < k``).

    Parameters
    ----------
    coords:
        ``(P, dim)`` ABSOLUTE coordinates for one example (typically the
        output of :func:`standardize_coords`, since the configured radii
        are defined in per-example-standardized units per ADR-0045 -- but
        this function itself is standardization-agnostic and operates in
        whatever units ``coords`` is given).
    radius:
        Distance cutoff. A neighbour at exactly ``radius`` is kept (the
        comparison is ``<=``, not ``<``); a neighbour beyond it is
        zero-padded.
    k:
        Maximum neighbours retrieved per query point.

    Returns
    -------
    Tensor
        ``(P, k, dim)`` ABSOLUTE neighbour coordinates, nearest first,
        zero-padded wherever fewer than ``k`` neighbours qualify.
    """
    n, dim = coords.shape
    out = coords.new_zeros(n, k, dim)
    eff_k = min(k, n)
    if eff_k == 0:
        return out

    for start in range(0, n, _QUERY_CHUNK):
        chunk = coords[start : start + _QUERY_CHUNK]
        dist = torch.cdist(chunk, coords)  # (chunk, n)
        top_dist, top_idx = torch.topk(dist, eff_k, dim=-1, largest=False)
        keep = (top_dist <= radius).unsqueeze(-1).to(coords.dtype)  # (chunk, eff_k, 1)
        neighbor_coords = coords[top_idx]  # (chunk, eff_k, dim)
        out[start : start + chunk.shape[0], :eff_k] = neighbor_coords * keep

    return out
