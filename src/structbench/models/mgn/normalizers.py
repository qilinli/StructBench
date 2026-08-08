"""Online accumulating feature normalizer for the MGN baseline (ADR-0043 §8).

Mirrors the source MeshGraphNets framework's Normalizer: running sums are
accumulated across training batches and used to standardize node/edge
features to zero mean / unit variance, with statistics persisted through
``state_dict`` so a trained normalizer travels with its checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

import torch
from torch import Tensor, nn


class OnlineNormalizer(nn.Module):
    """Online accumulating mean/std normalizer for ``(N, size)`` features.

    Running sums are updated on calls with ``accumulate=True``, up to
    ``max_accumulations`` such *calls* (batches/steps, not sample rows —
    a single call may carry many rows, all of which are accumulated into
    ``_count``). Once the cap is reached, further ``accumulate=True`` calls
    are no-ops. Before any accumulation (``_count == 0``), ``forward`` and
    ``inverse`` are the identity, so an untrained simulator can still run.

    The ``_sum``/``_sum_sq``/``_count`` accumulator buffers are dtype-pinned
    to ``float64`` (``_n_accumulations`` to ``int64``): a whole-module cast
    such as ``.float()``, ``.half()``, ``.to(dtype=...)``, or an AMP wrapper
    cannot downcast them, since :meth:`_apply` re-pins their dtype after any
    such cast (device moves are preserved as normal).

    Parameters
    ----------
    size:
        Number of features (the last-dimension width of normalized tensors).
    max_accumulations:
        Maximum number of ``accumulate=True`` calls after which running
        statistics stop updating.
    std_epsilon:
        Floor applied to the computed standard deviation, guarding division
        by a near-zero std for constant or near-constant features.
    """

    def __init__(
        self,
        size: int,
        max_accumulations: int = 10**6,
        std_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self._max_accumulations = max_accumulations
        self._std_epsilon = std_epsilon
        # Accumulation buffers use float64: running sums/sum-of-squares over
        # many accumulate() calls otherwise lose precision in float32,
        # which the moment formula (sum_sq/count - mean**2) amplifies.
        self.register_buffer(
            "_count", torch.zeros((), dtype=torch.float64), persistent=True
        )
        self.register_buffer(
            "_sum", torch.zeros(size, dtype=torch.float64), persistent=True
        )
        self.register_buffer(
            "_sum_sq", torch.zeros(size, dtype=torch.float64), persistent=True
        )
        self.register_buffer(
            "_n_accumulations",
            torch.zeros((), dtype=torch.int64),
            persistent=True,
        )
        self._count: Tensor
        self._sum: Tensor
        self._sum_sq: Tensor
        self._n_accumulations: Tensor

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        """Apply ``fn`` (device/dtype cast) then re-pin accumulator dtypes.

        ``nn.Module``-wide casts (``.float()``, ``.half()``, ``.to(dtype=...)``,
        AMP wrappers) route every floating-point buffer through ``fn``
        indiscriminately, which would silently downcast the float64
        accumulators — including their already-accumulated values — that
        this normalizer relies on for precision (see ``__init__``). Re-pin
        the accumulator dtypes immediately after the module-wide cast so
        that only the device move (if any) survives.
        """
        super()._apply(fn, recurse)
        self._sum = self._sum.to(dtype=torch.float64)
        self._sum_sq = self._sum_sq.to(dtype=torch.float64)
        self._count = self._count.to(dtype=torch.float64)
        self._n_accumulations = self._n_accumulations.to(dtype=torch.int64)
        return self

    def _accumulate(self, x: Tensor) -> None:
        """Fold ``x`` into the running sums, unless the call cap is reached."""
        if int(self._n_accumulations) >= self._max_accumulations:
            return
        x64 = x.detach().to(torch.float64)
        self._sum += x64.sum(dim=0)
        self._sum_sq += (x64 * x64).sum(dim=0)
        self._count += x64.shape[0]
        self._n_accumulations += 1

    def _moments(self) -> tuple[Tensor, Tensor]:
        """Current (mean, std), or the identity moments before accumulation."""
        if float(self._count) == 0.0:
            return torch.zeros_like(self._sum), torch.ones_like(self._sum)
        mean = self._sum / self._count
        var = (self._sum_sq / self._count - mean * mean).clamp(min=0.0)
        std = var.sqrt().clamp(min=self._std_epsilon)
        return mean, std

    def forward(self, x: Tensor, accumulate: bool = False) -> Tensor:
        """Normalize ``x`` to zero mean / unit variance using running stats.

        Parameters
        ----------
        x:
            ``(N, size)`` features to normalize.
        accumulate:
            If ``True``, fold ``x`` into the running sums (subject to
            ``max_accumulations``) before computing the output.

        Returns
        -------
        Tensor
            ``(N, size)`` normalized features, matching ``x``'s dtype and
            device.
        """
        if accumulate:
            self._accumulate(x)
        mean, std = self._moments()
        return (x - mean.to(x.dtype)) / std.to(x.dtype)

    def inverse(self, x: Tensor) -> Tensor:
        """Map normalized features back to their raw scale.

        Parameters
        ----------
        x:
            ``(N, size)`` normalized features.

        Returns
        -------
        Tensor
            ``(N, size)`` de-normalized features, matching ``x``'s dtype and
            device.
        """
        mean, std = self._moments()
        return x * std.to(x.dtype) + mean.to(x.dtype)
