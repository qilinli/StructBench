"""Benchmark spec and name-based registry (ADR-0022, ADR-0025).

A benchmark module exposes one frozen :class:`BenchmarkSpec` named
``SPEC``; the training pipeline resolves it by name through
:func:`get_benchmark`, replacing per-benchmark imports in ``cli/``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from torch import Tensor

from ..datasets import available_aux_fields
from ..eval import QoiFn
from .card import BenchmarkCard

#: Registered benchmark modules; each must define a module-level ``SPEC``.
_MODULES: dict[str, str] = {
    "taylor_impact_2d": "structbench.benchmarks.taylor_impact_2d",
}


@dataclass(frozen=True)
class BenchmarkSpec:
    """The runtime contract of one benchmark.

    Parameters
    ----------
    card : BenchmarkCard
        Descriptive metadata (ADR-0025). Its ``splits`` sizes must match
        the actual split lists here — validated at construction.
    splits : dict of str to tuple of str
        Immutable case-id lists by split name; must contain ``"train"``
        and ``"val"``.
    eval_splits : tuple of str
        Split names evaluated after training, in reporting order; each
        must be a key of ``splits``.
    aux_field : str
        Auxiliary target name, resolved by
        :func:`structbench.datasets.load_case_trajectory`.
    qois : dict of str to QoiFn
        Quantities of interest evaluated on rolled-out trajectories.
    boundary_feature_fn : callable or None
        ``(positions (P, dim) mm, radius) -> (P, 1)`` boundary feature,
        or ``None`` when the benchmark has no analytic boundary.
    dataset_id : str
        The canonical dataset this benchmark reads.
    """

    card: BenchmarkCard
    splits: Mapping[str, tuple[str, ...]]
    eval_splits: tuple[str, ...]
    aux_field: str
    qois: Mapping[str, QoiFn] = field(default_factory=dict)
    boundary_feature_fn: Callable[[Tensor, float], Tensor] | None = None
    dataset_id: str = ""

    def __post_init__(self) -> None:
        for required in ("train", "val"):
            if required not in self.splits:
                raise ValueError(f"splits must include {required!r}")
        missing = [s for s in self.eval_splits if s not in self.splits]
        if missing:
            raise ValueError(f"eval_splits not present in splits: {missing}")
        actual = {name: len(ids) for name, ids in self.splits.items()}
        if self.card.splits != actual:
            raise ValueError(f"card split sizes {self.card.splits} != actual {actual}")
        if self.aux_field not in available_aux_fields():
            raise ValueError(
                f"aux_field {self.aux_field!r} not in {sorted(available_aux_fields())}"
            )
        # Wrap in read-only proxies to prevent accidental mutation
        object.__setattr__(self, "splits", MappingProxyType(dict(self.splits)))
        object.__setattr__(self, "qois", MappingProxyType(dict(self.qois)))


def available_benchmarks() -> tuple[str, ...]:
    """Registered benchmark names, sorted."""
    return tuple(sorted(_MODULES))


def get_benchmark(name: str) -> BenchmarkSpec:
    """Resolve a benchmark's :class:`BenchmarkSpec` by registry name.

    Raises
    ------
    KeyError
        If ``name`` is not registered; the message lists valid names.
    """
    if name not in _MODULES:
        raise KeyError(
            f"unknown benchmark {name!r}; available: "
            f"{', '.join(available_benchmarks())}"
        )
    module = importlib.import_module(_MODULES[name])
    spec: BenchmarkSpec = module.SPEC
    return spec
