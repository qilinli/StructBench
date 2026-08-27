"""Benchmark spec and name-based registry (ADR-0024, ADR-0027).

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
from ..datasets.canonical import CaseTrajectory
from ..eval import QoiFn
from .card import BenchmarkCard
from .results import BaselineResult

#: Registered benchmark modules; each must define a module-level ``SPEC``.
#: notch_beam_2d_bend was descoped from the public benchmark set (ADR-0056,
#: amends 0024/0026): parked with no plan and no blessed baseline, so it is
#: delisted from docs/README. The ``benchmarks/notch_beam_2d_bend`` module and
#: its configs remain in the tree, re-registerable by restoring this entry.
_MODULES: dict[str, str] = {
    "deforming_plate": "structbench.benchmarks.deforming_plate",
    "notch_beam_2d_impact": "structbench.benchmarks.notch_beam_2d_impact",
    "taylor_impact_2d": "structbench.benchmarks.taylor_impact_2d",
    "wave_propagation_1d": "structbench.benchmarks.wave_propagation_1d",
}


@dataclass(frozen=True)
class BenchmarkSpec:
    """The runtime contract of one benchmark.

    Parameters
    ----------
    card : BenchmarkCard
        Descriptive metadata (ADR-0027). Its ``splits`` sizes must match
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
    kinematic_types: tuple[int, ...] = ()
    """Particle part-ids whose motion is prescribed (kinematic loaders, fixed
    supports); excluded from training loss and rollout metrics, and driven by
    ground truth during rollout (ADR-0026)."""
    results: tuple[BaselineResult, ...] = ()
    """Official baseline results (ADR-0033), rendered by the generated views;
    empty until a run is blessed. Metric split names must exist in ``splits``
    and at most one result per ``(family, scheme)`` row is allowed (ADR-0046,
    amended 2026-08-27) — both validated at construction. May mix blessed and
    provisional entries (ADR-0046); use :attr:`blessed_results`, not this
    field directly, wherever "the blessed baseline" is meant."""
    scored_frames: int | None = None
    """Exclusive upper frame bound of the scored span (ADR-0039), mirroring
    the trajectory-end bound ``T``: rollout/one-step aggregates and QoIs are
    computed over ``[card.input_frames, scored_frames)``; per-frame arrays
    still cover the full trajectory as the long-horizon diagnostic. ``None``
    scores to the trajectory end. Must exceed ``card.input_frames`` and not
    exceed ``card.n_frames`` — validated at construction."""
    quickstart_family: str = "cgn"
    """Model family (ADR-0032) whose blessed run anchors the Quickstart
    recipe on the generated benchmark page; must be non-blank — validated
    at construction. Defaults to ``"cgn"``, the family every currently
    blessed benchmark uses."""
    mesh_transform: Callable[[CaseTrajectory], CaseTrajectory] | None = None
    """Applied to each loaded trajectory by the mesh-native families only
    (mgn/transolver/geoflare; ADR-0047): synthesizes mesh connectivity
    and/or boundary nodes for benchmarks whose canonical data is not
    nodal-FE (e.g. Taylor's lattice mesh + wall nodes). The cgn family
    never applies it — its data path stays byte-identical. ``None`` for
    benchmarks whose cases already carry a mesh (or that no mesh-native
    family runs on)."""
    scripted_types: tuple[int, ...] | None = None
    """Node-type codes the mesh-native simulators feed the ground-truth
    next-step velocity as an input feature (ADR-0043: OBSTACLE is scripted,
    HANDLE is not); must be a subset of ``kinematic_types`` — validated at
    construction. ``None`` leaves the family default (the ADR-0043
    ``(1,)``); Taylor pins its wall type here (ADR-0047), whose scripted
    velocity is identically zero."""
    loading_scalar: Callable[[str], float] | None = None
    """Maps a case id to its scalar loading parameter (impact velocity),
    consumed by the Transolver ``impact_velocity_feature`` (ADR-0051 B): the
    operator-learning / one-shot convention of feeding a known scalar loading
    parameter (Transolver Plasticity, GeoTransolver/CrashSolver) rather than a
    per-node velocity history. ``None`` when the benchmark has no such scalar
    (e.g. actuator-driven deforming-plate), in which case a run that requests
    the feature is rejected at train time."""

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
        for result in self.results:
            unknown = [s for s in result.metrics if s not in self.splits]
            if unknown:
                raise ValueError(
                    f"result {result.label!r} references unknown splits {unknown}"
                )
        # One row per (family, scheme): a benchmark may table the same model
        # family under several prediction schemes (the DeformingPlate scheme
        # matrix, 2026-08-21, extending ADR-0046) but never two rows for the
        # same family AND scheme.
        seen_rows: set[tuple[str, str]] = set()
        for result in self.results:
            row = (result.family, result.scheme or "")
            if row in seen_rows:
                raise ValueError(
                    f"duplicate (family, scheme) {row!r} in results for "
                    f"benchmark {self.card.name!r}"
                )
            seen_rows.add(row)
        if not self.quickstart_family.strip():
            raise ValueError("quickstart_family must be non-empty")
        if self.scripted_types is not None and not set(self.scripted_types) <= set(
            self.kinematic_types
        ):
            raise ValueError(
                f"scripted_types={self.scripted_types} must be a subset of "
                f"kinematic_types={self.kinematic_types}"
            )
        if self.scored_frames is not None and not (
            self.card.input_frames < self.scored_frames <= self.card.n_frames
        ):
            raise ValueError(
                f"scored_frames={self.scored_frames} must satisfy "
                f"input_frames ({self.card.input_frames}) < scored_frames "
                f"<= n_frames ({self.card.n_frames})"
            )
        # Wrap in read-only proxies to prevent accidental mutation
        object.__setattr__(self, "splits", MappingProxyType(dict(self.splits)))
        object.__setattr__(self, "qois", MappingProxyType(dict(self.qois)))
        # NB: dataclasses.asdict(spec) would raise on these proxies;
        # use spec.card.to_json_dict() for serialization.

    @property
    def blessed_results(self) -> tuple[BaselineResult, ...]:
        """The non-provisional entries of :attr:`results`, in declaration
        order (ADR-0046).

        Use this, not ``results``, wherever "the blessed baseline(s)" is
        meant — ``results`` may also carry provisional entries recorded
        for comparison only.
        """
        return tuple(r for r in self.results if not r.provisional)


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
