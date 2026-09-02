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

#: Supported auxiliary target-space transforms (``CGNConfig.aux_transform``).
AUX_TRANSFORMS = frozenset({"none", "asinh"})


def aux_forward_transform(values, transform: str, scale: float):
    """Map raw auxiliary values into training target space.

    ``"none"`` returns the input unchanged; ``"asinh"`` returns
    ``asinh(values / scale)`` — linear below ``scale``, logarithmic above it,
    spreading a heavy-tailed field's decision region before z-scoring.
    Accepts numpy arrays or torch tensors and preserves the input kind.
    """
    if transform == "none":
        return values
    if transform == "asinh":
        import torch

        if isinstance(values, torch.Tensor):
            return torch.asinh(values / scale)
        return np.arcsinh(np.asarray(values) / scale)
    raise ValueError(
        f"unknown aux_transform {transform!r}; supported: {sorted(AUX_TRANSFORMS)}"
    )


def aux_inverse_transform(values, transform: str, scale: float):
    """Invert :func:`aux_forward_transform`, back to raw auxiliary units."""
    if transform == "none":
        return values
    if transform == "asinh":
        import torch

        if isinstance(values, torch.Tensor):
            return torch.sinh(values) * scale
        return np.sinh(np.asarray(values)) * scale
    raise ValueError(
        f"unknown aux_transform {transform!r}; supported: {sorted(AUX_TRANSFORMS)}"
    )


def expand_aux_knob(value, n_channels: int) -> tuple:
    """Expand a scalar-or-per-channel aux knob to a per-channel tuple (ADR-0059).

    A scalar (``str``/``float``/``int``) applies to every channel — the
    ``C = 1`` shorthand and the byte-identical legacy form. A sequence must
    match ``n_channels`` exactly.
    """
    if isinstance(value, str | float | int) and not isinstance(value, bool):
        return (value,) * n_channels
    values = tuple(value)
    if len(values) != n_channels:
        raise ValueError(
            f"per-channel aux knob has {len(values)} entries for "
            f"{n_channels} channel(s): {values!r}"
        )
    return values


def _apply_per_channel(fn, values, transforms, scales):
    """Apply a (values, transform, scale) fn per trailing-axis channel.

    Fast path: when every channel shares one (transform, scale) pair the fn
    is applied to the whole array in one call — bit-identical to the
    pre-0059 scalar path and free of per-channel stacking cost.
    """
    pairs = list(zip(transforms, scales, strict=True))
    if len(set(pairs)) == 1:
        return fn(values, pairs[0][0], pairs[0][1])
    import torch

    stack = torch.stack if isinstance(values, torch.Tensor) else np.stack
    return stack(
        [fn(values[..., i], tr, sc) for i, (tr, sc) in enumerate(pairs)],
        -1,
    )


def aux_forward_transform_channels(values, transforms, scales):
    """Per-channel :func:`aux_forward_transform` over the trailing axis.

    ``values`` is ``(..., C)``; ``transforms`` / ``scales`` are length-``C``
    sequences (see :func:`expand_aux_knob`).
    """
    return _apply_per_channel(aux_forward_transform, values, transforms, scales)


def aux_inverse_transform_channels(values, transforms, scales):
    """Per-channel :func:`aux_inverse_transform` over the trailing axis."""
    return _apply_per_channel(aux_inverse_transform, values, transforms, scales)


@dataclass
class NormalizationStats:
    """Mean/std of velocity, acceleration, and the auxiliary field.

    Velocity and acceleration carry a per-dimension mean/std (mm/frame,
    mm/frame^2). The auxiliary target block carries a per-channel mean/std
    (shape ``(C,)``; ADR-0059 — channels differ by orders of magnitude, so
    each is z-scored on its own statistics) so its training target can be
    normalized to O(1), balancing the dual position/auxiliary loss. ``C = 1``
    reproduces the historical scalar shape ``(1,)``.
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


def compute_stats(
    trajectories: list[CaseTrajectory],
    *,
    aux_transform="none",
    aux_transform_scale=0.01,
) -> NormalizationStats:
    """Pool velocity/acceleration/aux stats over all particles, frames, and cases.

    Velocity is the first finite difference of positions along the frame axis;
    acceleration is the second. The auxiliary target field (e.g. von Mises
    stress for Taylor) is a direct quantity, so its stats are pooled over the
    values with no finite difference — in **target space**: when
    ``aux_transform`` is set, the aux stats are computed over the transformed
    values (:func:`aux_forward_transform`), matching the space the training
    target and decoder live in. Statistics are stacked over every particle in
    every frame of every trajectory.

    Parameters
    ----------
    trajectories:
        List of :class:`~structbench.datasets.canonical.CaseTrajectory` objects.
        Each must have at least 3 frames (``T >= 3``) so that both velocity and
        acceleration samples exist.
    aux_transform, aux_transform_scale:
        Auxiliary target-space transform (``CGNConfig`` fields). Each is a
        scalar (applied to every channel) or a length-``C`` per-channel
        sequence (ADR-0059; see :func:`expand_aux_knob`).

    Returns
    -------
    NormalizationStats
        Per-dimension mean and std for velocity ``(dim,)`` and acceleration
        ``(dim,)``, plus per-channel mean and std for the auxiliary block
        ``(C,)``, pooled over all particles, frames, and cases.
    """
    n_channels = int(trajectories[0].aux.shape[-1]) if trajectories else 1
    transforms = expand_aux_knob(aux_transform, n_channels)
    scales = expand_aux_knob(aux_transform_scale, n_channels)
    vels, accs, auxs = [], [], []
    for tr in trajectories:
        p = tr.positions.astype(np.float64)  # (T, P, dim)
        v = p[1:] - p[:-1]  # (T-1, P, dim)
        a = v[1:] - v[:-1]  # (T-2, P, dim)
        vels.append(v.reshape(-1, p.shape[-1]))
        accs.append(a.reshape(-1, p.shape[-1]))
        aux = aux_forward_transform_channels(
            tr.aux.astype(np.float64), transforms, scales
        )
        auxs.append(np.asarray(aux).reshape(-1, n_channels))  # (T*P, C)
    v_all = np.concatenate(vels, axis=0)
    a_all = np.concatenate(accs, axis=0)
    aux_all = np.concatenate(auxs, axis=0)
    return NormalizationStats(
        velocity_mean=v_all.mean(0),
        velocity_std=v_all.std(0),
        acceleration_mean=a_all.mean(0),
        acceleration_std=a_all.std(0),
        aux_mean=aux_all.mean(0),
        aux_std=aux_all.std(0),
    )


#: Bump to invalidate every cache when the stats computation itself changes.
_STATS_VERSION = "2"


def _traj_signature(tr: CaseTrajectory) -> str:
    """Cache-key fragment: case-id, shape, and a content signature.

    The content signature (sum, sum-of-squares, min, max of positions and aux)
    changes whenever the underlying values change — even at fixed shape — so a
    shape-preserving in-place data edit (the ADR-0030 x1000 patch) cannot reuse
    stale stats. It also subsumes any loader-scale change, since those are
    already baked into the trajectory values.
    """
    pos = np.asarray(tr.positions, np.float64)
    aux = np.asarray(tr.aux, np.float64)
    return (
        f"{tr.case_id}:{pos.shape[0]}x{pos.shape[1]}"
        f":p{pos.sum():.6e},{(pos * pos).sum():.6e},{pos.min():.6e},{pos.max():.6e}"
        f":a{aux.sum():.6e},{(aux * aux).sum():.6e},{aux.min():.6e},{aux.max():.6e}"
    )


def cached_compute_stats(
    trajectories: list[CaseTrajectory],
    *,
    dataset_root: str | Path,
    aux_field,
    aux_transform="none",
    aux_transform_scale=0.01,
) -> NormalizationStats:
    """:func:`compute_stats` with a dataset-level cache.

    The cache lives at ``<dataset_root>/derived/norm_<key>.npz``, where the key
    hashes a stats-version salt, the auxiliary field name, and per-trajectory
    signatures. Each signature covers the case-id, frame/particle counts, and a
    content signature of the positions and aux values (sum, sum-of-squares,
    min, max). Stats are therefore computed once per (split, aux-field, data)
    combination and reused across runs, while a changed case list, a different
    auxiliary field, a shape change (e.g. the ADR-0028 terminal-frame trim),
    **or any change to the underlying values** (e.g. the ADR-0030 in-place
    x1000 patch) forces recomputation under a new filename. The cache never
    blocks training: a write failure (e.g. read-only dataset root) degrades to
    a warning, and an unreadable/corrupt cache file is recomputed and
    rewritten. Writes go through a temp file + atomic rename so a killed run
    cannot leave a truncated cache behind.

    Parameters
    ----------
    trajectories:
        Trajectories of the (train) split, as for :func:`compute_stats`.
    dataset_root:
        Directory the ``derived/`` cache folder lives under — normally the
        directory holding the split's ``<case_id>.h5`` files.
    aux_field:
        Auxiliary field name or sequence of names (ADR-0059). A single name
        keys exactly as before, so existing caches stay valid; a multi-field
        selection keys on the ``"+"``-joined names. Different selections
        produce separate cache files even for the same trajectories.
    aux_transform, aux_transform_scale:
        Auxiliary target-space transform, forwarded to :func:`compute_stats`
        (scalar or per-channel, ADR-0059). An all-``"none"`` transform keeps
        the pre-transform cache keys, so existing caches stay valid; scalar
        non-``"none"`` keys exactly as before; per-channel forms key on
        their tuple reprs.

    Returns
    -------
    NormalizationStats
    """
    from .canonical import as_aux_fields

    fingerprint = [_traj_signature(tr) for tr in trajectories]
    key_parts = [_STATS_VERSION, "+".join(as_aux_fields(aux_field))]
    t_seq = not isinstance(aux_transform, str)
    s_seq = not isinstance(aux_transform_scale, float | int)
    all_none = (
        all(t == "none" for t in aux_transform) if t_seq else aux_transform == "none"
    )
    if not all_none:
        if t_seq or s_seq:
            t_key = tuple(aux_transform) if t_seq else aux_transform
            s_key = tuple(aux_transform_scale) if s_seq else aux_transform_scale
            key_parts.append(f"aux_transform={t_key}:{s_key!r}")
        else:
            # scalar non-"none": the exact pre-0059 key, so caches stay valid
            key_parts.append(f"aux_transform={aux_transform}:{aux_transform_scale!r}")
    key_material = "\n".join([*key_parts, *fingerprint])
    key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:12]
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

    stats = compute_stats(
        trajectories,
        aux_transform=aux_transform,
        aux_transform_scale=aux_transform_scale,
    )
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
