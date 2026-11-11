"""Benchmark spec and name-based registry (ADR-0024, ADR-0027).

A benchmark module exposes one frozen :class:`BenchmarkSpec` named
``SPEC``; the training pipeline resolves it by name through
:func:`get_benchmark`, replacing per-benchmark imports in ``cli/``.

A benchmark may also be a *definition* outside the package: a directory
holding ``benchmark.py`` that exports ``SPEC`` (ADR-0032 note, 2026-11-06),
loaded by path the way ``structbench.datagen`` loads a dataset definition.
Such a benchmark is unadmitted: it has no registry row, no landing page and
no results registry, and the spec records where it came from
(:attr:`BenchmarkSpec.source`) so a run can be tied to the definition that
produced it.
"""

from __future__ import annotations

import importlib
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import cast

from torch import Tensor

from ..core.definition_module import (
    DefinitionModuleError,
    definition_sha256s,
    load_definition_module,
)
from ..datasets import available_aux_fields
from ..datasets.canonical import CaseTrajectory
from ..eval import QoiFn
from ..eval.coverage import Coverage, coverage
from ..eval.sensitivity import GradientFamily
from .card import BenchmarkCard
from .constitutive import ELEMENT_FORMULATIONS, ConstitutiveLaw
from .results import BaselineResult

#: The file a benchmark definition directory must hold.
DEFINITION_FILE = "benchmark.py"

#: The key of a result's ``metrics`` that holds its gradient-family rows
#: (ADR-0076; the ``metrics`` block of ``metrics-gradient.json``). It sits
#: beside the split keys and is not a split.
GRADIENT_GROUP = "gradient"


@dataclass(frozen=True)
class BenchmarkSource:
    """Where a by-path benchmark definition was loaded from.

    Parameters
    ----------
    path:
        The definition directory, resolved to an absolute path.
    files:
        ``{relative path: sha256}`` of ``benchmark.py`` and the sibling
        modules it imported, LF-normalised; what a run record stores.
    """

    path: Path
    files: Mapping[str, str]

    def to_json_dict(self) -> dict[str, object]:
        return {"path": str(self.path), "files": dict(self.files)}


#: Registered benchmark modules; each must define a module-level ``SPEC``.
#: notch_beam_2d_bend was descoped (ADR-0056, amends 0024/0026) and then
#: excluded outright (ADR-0056 amendment 2026-09-23): notch-impact covers
#: what the pair was for. Its module and configs were removed 2026-09-23;
#: the 2DNotchBeam converter still reads the bend family, so the archive
#: stays reproducible from the raw runs.
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
    scored_frames: int | Callable[[str], int] | None = None
    """Exclusive upper frame bound of the scored span (ADR-0039), mirroring
    the trajectory-end bound ``T``: rollout/one-step aggregates and QoIs are
    computed over ``[card.input_frames, scored_frames)``; per-frame arrays
    still cover the full trajectory as the long-horizon diagnostic. ``None``
    scores to the trajectory end. An ``int`` is one bound for every case
    (validated at construction); a callable maps a case id to that case's
    bound (ADR-0039 note, 2026-11-06), checked when read through
    :meth:`scored_frames_for`. Either must exceed ``card.input_frames`` and
    not exceed ``card.n_frames``. Consumers that need one bound for the
    whole benchmark (the time-conditioned query's reference) read
    :attr:`scored_frames_fixed`, which is ``None`` for a per-case bound."""
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
    hardening_curve: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    """ADR-0064: the benchmark's isotropic hardening curve as
    ``(peeq_knots, sigma_y_knots)`` — sigma_y in the WORKING aux unit (MPa),
    knots verbatim from the solver deck (including any non-monotone knot),
    ``np.interp`` end-clamped semantics as the contract. The ONE
    authoritative table: consumed by the structured admissible heads and
    the consistency-hinge loss (both flow-map training paths) and by the
    admissibility analyses. ``None`` when the benchmark has no plasticity
    (structured heads are then rejected at train time)."""

    case_parameters: Callable[[str], Mapping[str, float | str]] | None = None
    """Case id to its design parameters, numeric and categorical, keyed as
    the card's ``parameter_space`` names them (ADR-0074 clause 2; the
    open-evaluation plan §4). ``None`` for a benchmark whose parameters are
    not known per case (DeformingPlate); :meth:`coverage` then has nothing to
    read."""
    novel_cases: tuple[str, ...] = ()
    """Scored cases declared *novel* (ADR-0074 clause 2): an unseen
    categorical level or a changed constant, which no distance expresses.
    Shown apart from the coverage bins. Every id must be a case of a scored
    split (validated at construction)."""
    gradient_family: GradientFamily | None = None
    """The benchmark's gradient task (ADR-0076): the swept parameter, the
    sweep lines, the declared noise per quantity of interest and the seed
    construction the surrogate side needs. ``None`` for a benchmark whose
    sweep supports no instance; every quantity the family names must be one
    of ``qois`` (validated at construction)."""
    loading_scalar: Callable[[str], float] | None = None
    """Maps a case id to its scalar loading parameter (impact velocity),
    consumed by the Transolver ``impact_velocity_feature`` (ADR-0051 B): the
    operator-learning / one-shot convention of feeding a known scalar loading
    parameter (Transolver Plasticity, GeoTransolver/CrashSolver) rather than a
    per-node velocity history. ``None`` when the benchmark has no such scalar
    (e.g. actuator-driven deforming-plate), in which case a run that requests
    the feature is rejected at train time."""

    source: BenchmarkSource | None = None
    """Set by :func:`get_benchmark` when the spec was loaded from a definition
    directory; ``None`` for a registered benchmark. A spec with a source is
    unadmitted, and its runs record the source beside the benchmark path."""

    state_site: str = "node"
    """ADR-0075: where the case's constitutive state lives. ``"node"`` (every
    benchmark before ADR-0075) scores the nodal ``aux`` block only;
    ``"element"`` declares the integration-point site as the state's home:
    ``state_channels`` are read per element into ``CaseTrajectory.state``,
    scored, property-checked and restarted from there, and the nodal ``aux``
    is their projection (the cheap baseline view)."""
    state_channels: tuple[str, ...] = ()
    """The registered element-block fields of the element-site state, in
    declaration order (ADR-0075); required and validated against
    :func:`available_aux_fields` when ``state_site == "element"``, empty
    otherwise. A two-site run's ``train.aux_fields`` must equal them."""
    element_formulation: str | None = None
    """The element kinematics of the state site, one of
    :data:`~structbench.benchmarks.constitutive.ELEMENT_FORMULATIONS`;
    required with ``state_site == "element"``, ``None`` otherwise."""
    constitutive_law: Callable[[str], ConstitutiveLaw] | None = None
    """Maps a case id to its own :class:`ConstitutiveLaw` (ADR-0075 clause 3):
    the law the admissible decoder applies, an input of the case rather than
    a table baked into a checkpoint. ``None`` when the benchmark declares no
    element site or its cases carry no return-map law."""
    property_tolerances: Mapping[str, float] = field(default_factory=dict)
    """ADR-0075 clause 6 (plan 3c): the tolerances of the constraint
    properties at the element site, keyed ``"yield"`` (relative excess over
    ``σ_y(peeq)`` a sample may show before it counts as a violation) and
    ``"peeq"`` (absolute backward step of the plastic strain that counts as a
    decrease). A definition takes both from the reference data's own measured
    excess in its verification record and never declares them tighter; both
    are required with ``state_site == "element"`` and rejected without it."""
    matched_initial_state_pairs: tuple[tuple[str, str], ...] = ()
    """ADR-0075 clause 6: pairs of case ids with identical kinematic seeds and
    different stored constitutive state, for the state-sufficiency property.
    Each pair is two distinct cases of the same split (so one evaluation sees
    both); empty when the benchmark has no such cases."""
    restart_anchors: tuple[int, ...] | Callable[[str], tuple[int, ...]] | None = None
    """ADR-0075 clause 5 (plan 3b): the frames a restart task starts from,
    fixed for every case or a callable from case id, like ``scored_frames``;
    ``None`` when the benchmark declares no restart family. Each anchor
    needs ``input_frames - 1`` frames of history before it and every declared
    horizon after it within the trajectory (:meth:`restart_anchors_for`)."""
    restart_horizons: tuple[int, ...] = ()
    """The horizons, in frames, scored after each anchor; non-empty exactly
    when ``restart_anchors`` is declared, strictly increasing."""
    restart_compositions: tuple[int, ...] = (1, 2, 4)
    """The composition counts ``k`` of the sweep: each horizon ``T`` is also
    reached by the flow map in ``k`` self-anchored steps of ``T / k`` frames,
    for every ``k`` that divides ``T``. Strictly increasing from 1."""

    @property
    def defines_restarts(self) -> bool:
        """Whether the benchmark declares a restart family (clause 5)."""
        return self.restart_anchors is not None

    def restart_anchors_for(self, case_id: str) -> tuple[int, ...]:
        """The anchors of one case, each validated to have ``input_frames - 1``
        frames before it and the longest horizon after it inside the scored
        span (the case's scored bound where the benchmark pins one, ADR-0039,
        else ``card.n_frames``): a restart never scores frames the protocol
        excludes (design §3).

        Raises
        ------
        ValueError
            If the benchmark declares no restarts, or an anchor lies outside
            ``[input_frames - 1, bound - max(horizons))``.
        """
        if self.restart_anchors is None:
            raise ValueError(f"benchmark {self.card.name!r} declares no restarts")
        raw = (
            self.restart_anchors
            if isinstance(self.restart_anchors, tuple)
            else tuple(self.restart_anchors(case_id))
        )
        if not raw:
            raise ValueError(
                f"restart_anchors({case_id!r}) must declare at least one anchor"
            )
        # Any integral type (a numpy int from a per-case callable) is an
        # anchor; a bool is not, whatever Python says.
        bad_type = [
            a for a in raw if isinstance(a, bool) or not isinstance(a, numbers.Integral)
        ]
        if bad_type:
            raise ValueError(
                f"restart_anchors({case_id!r}) = {raw!r}: anchors {bad_type} must "
                "be integers"
            )
        anchors = tuple(int(a) for a in raw)
        lo = self.card.input_frames - 1
        scored = (
            self.scored_frames_fixed
            if case_id == _FIXED_ANCHORS_CASE
            else self.scored_frames_for(case_id)
        )
        bound = (
            self.card.n_frames if scored is None else min(scored, self.card.n_frames)
        )
        hi = bound - max(self.restart_horizons)
        bad = [a for a in anchors if not lo <= a < hi]
        if bad:
            raise ValueError(
                f"restart_anchors({case_id!r}) = {anchors!r}: anchors {bad} must "
                f"lie in [{lo}, {hi}) (input_frames - 1 <= anchor < scored "
                "bound - max horizon)"
            )
        return anchors

    @property
    def defined_properties(self) -> tuple[str, ...]:
        """The properties-block keys this benchmark defines (ADR-0075 clause
        6), in render order: closure and error growth always; the yield and
        plastic-strain rates with an element site; state sufficiency with
        declared matched pairs. The renderer shows these columns whether or
        not a result has reported them; ``evaluate`` writes them."""
        keys = ["prop_closure_ratio", "prop_error_growth"]
        if self.state_site == "element":
            keys += ["prop_yield_violation_rate", "prop_peeq_decrease_rate"]
        if self.matched_initial_state_pairs:
            keys.append("prop_state_sufficiency")
        return tuple(keys)

    @property
    def scored_frames_fixed(self) -> int | None:
        """The one scored bound shared by every case, or ``None`` when the
        benchmark scores to the trajectory end or per case."""
        return self.scored_frames if isinstance(self.scored_frames, int) else None

    def scored_share(self) -> float | None:
        """The share of the stored frames that lie inside the declared scored
        span, pooled over every case of every split (ADR-0078), or ``None``
        when the benchmark scores to the trajectory end.

        Per case the span ``[input_frames, bound)`` holds
        ``bound - input_frames`` frames of the card's ``n_frames``. This is
        the declaration's own arithmetic and reads no data; the timeline
        analysis reports the same share from the stored files.
        """
        if self.scored_frames is None:
            return None
        ids = [cid for split in self.splits.values() for cid in split]
        inside = sum(
            cast(int, self.scored_frames_for(cid)) - self.card.input_frames
            for cid in ids
        )
        return inside / (len(ids) * self.card.n_frames)

    def scored_frames_for(self, case_id: str) -> int | None:
        """The scored bound of one case: the fixed ``int``, the callable's
        value for ``case_id``, or ``None`` (scored to the trajectory end).

        Raises
        ------
        ValueError
            If a per-case bound is not an ``int`` in
            ``(card.input_frames, card.n_frames]``.
        """
        if self.scored_frames is None or isinstance(self.scored_frames, int):
            return self.scored_frames
        bound = self.scored_frames(case_id)
        if not isinstance(bound, int) or not (
            self.card.input_frames < bound <= self.card.n_frames
        ):
            raise ValueError(
                f"scored_frames({case_id!r}) = {bound!r} must be an int with "
                f"input_frames ({self.card.input_frames}) < scored_frames "
                f"<= n_frames ({self.card.n_frames})"
            )
        return bound

    @property
    def aux_fields(self) -> tuple[str, ...]:
        """The canonical aux channel selection as a tuple (ADR-0059).

        Every registered benchmark declares one field today, so this is
        ``(aux_field,)``; variant runs widen the selection via
        ``train.aux_fields``, never by editing a benchmark in place.
        """
        return (self.aux_field,)

    def __post_init__(self) -> None:
        # ADR-0074: a novel case is a scored case.
        scored_ids = {
            c for s, cs in self.splits.items() if s not in ("train", "val") for c in cs
        }
        unknown_novel = sorted(set(self.novel_cases) - scored_ids)
        if unknown_novel:
            raise ValueError(f"novel_cases are not scored cases: {unknown_novel}")
        # ADR-0076: the gradient family scores quantities this benchmark
        # defines, and nothing else.
        if self.gradient_family is not None:
            unknown = sorted(set(self.gradient_family.noise) - set(self.qois))
            if unknown:
                raise ValueError(
                    f"gradient_family names quantities {unknown} that are not "
                    f"among this benchmark's qois {sorted(self.qois)}"
                )
        # ADR-0064: a malformed hardening curve must fail at spec
        # construction, not inside torch.bucketize (which returns wrong
        # intervals SILENTLY on unsorted knots — review finding).
        if self.hardening_curve is not None:
            peeq_knots, sy_knots = self.hardening_curve
            if len(peeq_knots) < 2 or len(peeq_knots) != len(sy_knots):
                raise ValueError(
                    "hardening_curve needs >= 2 (peeq, sigma_y) knot pairs "
                    f"of equal length; got {len(peeq_knots)}/{len(sy_knots)}"
                )
            if any(
                later <= earlier
                for earlier, later in zip(peeq_knots[:-1], peeq_knots[1:], strict=True)
            ):
                raise ValueError(
                    "hardening_curve peeq knots must be strictly increasing"
                )
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
        # ADR-0059: the card's descriptive aux declaration and the spec's
        # loaded-with declaration were previously kept equal only by
        # convention (both set from one per-benchmark constant); check it.
        if self.card.aux_field != self.aux_field:
            raise ValueError(
                f"card.aux_field {self.card.aux_field!r} != spec.aux_field "
                f"{self.aux_field!r} (one canonical declaration per benchmark)"
            )
        for result in self.results:
            # ADR-0076: the gradient family's rows sit under their own key
            # beside the splits (it is not a split: it scores a derivative
            # along the sweep), on a benchmark that declares the family.
            groups = [*self.splits]
            if self.gradient_family is not None:
                groups.append(GRADIENT_GROUP)
            unknown = [s for s in result.metrics if s not in groups]
            if unknown:
                raise ValueError(
                    f"result {result.label!r} references unknown splits {unknown}"
                )
            misplaced = sorted(
                key
                for group, values in result.metrics.items()
                for key in values
                if key.startswith("gradient_") != (group == GRADIENT_GROUP)
            )
            if misplaced:
                raise ValueError(
                    f"result {result.label!r}: gradient_* metrics and the "
                    f"{GRADIENT_GROUP!r} group come together (misplaced: "
                    f"{misplaced[:5]})"
                )
            # ADR-0074: per-case rows belong to scored cases of this benchmark.
            stray = sorted(set(result.case_metrics) - scored_ids)
            if stray:
                raise ValueError(
                    f"result {result.label!r} has case_metrics for cases that are "
                    f"not scored cases of this benchmark: {stray[:5]}"
                )
            # ADR-0075 clause 5: restart rows say what they restarted from, and
            # a kind without rows is a claim with nothing behind it.
            has_restart_keys = any(
                k.startswith(("restart_", "compose_"))
                for values in result.metrics.values()
                for k in values
            )
            if has_restart_keys != bool(result.restart_kind):
                raise ValueError(
                    f"result {result.label!r}: restart_* / compose_* metrics and "
                    f"restart_kind come together (keys: {has_restart_keys}, kind: "
                    f"{result.restart_kind!r})"
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
        if isinstance(self.scored_frames, int) and not (
            self.card.input_frames < self.scored_frames <= self.card.n_frames
        ):
            raise ValueError(
                f"scored_frames={self.scored_frames} must satisfy "
                f"input_frames ({self.card.input_frames}) < scored_frames "
                f"<= n_frames ({self.card.n_frames})"
            )
        if callable(self.scored_frames):
            # ADR-0078: a per-case bound is checked for every case of every
            # split when the definition loads, not when a stage first reads
            # the one case that is wrong.
            for ids in self.splits.values():
                for case_id in ids:
                    self.scored_frames_for(case_id)
        # ADR-0075: the state site and what it needs, all or nothing.
        if self.state_site not in ("node", "element"):
            raise ValueError(
                f"state_site must be 'node' or 'element', got {self.state_site!r}"
            )
        if self.state_site == "element":
            if not self.state_channels:
                raise ValueError("state_site='element' needs non-empty state_channels")
            bad = [n for n in self.state_channels if n not in available_aux_fields()]
            if bad:
                raise ValueError(
                    f"state_channels {bad} are not registered element-block fields "
                    f"({sorted(available_aux_fields())})"
                )
            if len(set(self.state_channels)) != len(self.state_channels):
                raise ValueError("state_channels must be unique")
            if self.element_formulation not in ELEMENT_FORMULATIONS:
                raise ValueError(
                    "state_site='element' needs element_formulation in "
                    f"{ELEMENT_FORMULATIONS}, got {self.element_formulation!r}"
                )
        else:
            if self.state_channels or self.element_formulation is not None:
                raise ValueError(
                    "state_channels and element_formulation require "
                    "state_site='element'"
                )
            if self.constitutive_law is not None:
                raise ValueError("constitutive_law requires state_site='element'")
        object.__setattr__(self, "state_channels", tuple(self.state_channels))
        # ADR-0075 clause 6 (plan 3c): the constraint properties' tolerances
        # come with the element site, both of them, from the reference's record.
        tol_keys = {"yield", "peeq"}
        if self.state_site == "element":
            if set(self.property_tolerances) != tol_keys:
                raise ValueError(
                    "state_site='element' needs property_tolerances with exactly "
                    f"the keys {sorted(tol_keys)} (the reference data's measured "
                    f"excess), got {sorted(self.property_tolerances)}"
                )
        elif self.property_tolerances:
            raise ValueError("property_tolerances require state_site='element'")
        negative = {k: v for k, v in self.property_tolerances.items() if v < 0.0}
        if negative:
            raise ValueError(f"property_tolerances must be non-negative: {negative}")
        object.__setattr__(
            self,
            "property_tolerances",
            MappingProxyType(dict(self.property_tolerances)),
        )
        split_of = {c: s for s, cs in self.splits.items() for c in cs}
        for p in self.matched_initial_state_pairs:
            if isinstance(p, str) or not isinstance(p, (tuple, list)):
                raise ValueError(
                    f"matched_initial_state_pairs: {p!r} must be a pair of two case "
                    "ids, not a string"
                )
        pairs = tuple(tuple(p) for p in self.matched_initial_state_pairs)
        for pair in pairs:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError(
                    f"matched_initial_state_pairs: {pair!r} is not two distinct cases"
                )
            unknown = [c for c in pair if c not in split_of]
            if unknown:
                raise ValueError(
                    f"matched_initial_state_pairs: unknown cases {unknown} in {pair!r}"
                )
            if split_of[pair[0]] != split_of[pair[1]]:
                raise ValueError(
                    f"matched_initial_state_pairs: {pair!r} must be two cases of the "
                    "same split"
                )
        object.__setattr__(self, "matched_initial_state_pairs", pairs)
        # ADR-0075 clause 5 (plan 3b): the restart declaration, all or nothing.
        horizons = tuple(self.restart_horizons)
        if (self.restart_anchors is None) != (not horizons):
            raise ValueError(
                "restart_anchors and restart_horizons are declared together: "
                f"got anchors={self.restart_anchors!r}, horizons={horizons!r}"
            )
        if horizons and (horizons[0] < 1 or not _strictly_increasing(horizons)):
            raise ValueError(
                f"restart_horizons must be positive and strictly increasing: {horizons}"
            )
        comps = tuple(self.restart_compositions)
        if not comps or comps[0] != 1 or not _strictly_increasing(comps):
            raise ValueError(
                f"restart_compositions must be strictly increasing from 1: {comps}"
            )
        object.__setattr__(self, "restart_horizons", horizons)
        object.__setattr__(self, "restart_compositions", comps)
        if isinstance(self.restart_anchors, tuple):
            object.__setattr__(self, "restart_anchors", tuple(self.restart_anchors))
            # Validates against the card and a fixed scored bound; a per-case
            # bound is checked when the case's anchors are read.
            self.restart_anchors_for(_FIXED_ANCHORS_CASE)
        # Wrap in read-only proxies to prevent accidental mutation
        object.__setattr__(self, "splits", MappingProxyType(dict(self.splits)))
        object.__setattr__(self, "qois", MappingProxyType(dict(self.qois)))
        # NB: dataclasses.asdict(spec) would raise on these proxies;
        # use spec.card.to_json_dict() for serialization.

    @property
    def scored_splits(self) -> tuple[str, ...]:
        """Every split but ``train`` and ``val``, in declaration order: the
        scored sets (ADR-0074 clause 2), whatever their legacy names."""
        return tuple(s for s in self.splits if s not in ("train", "val"))

    def coverage(self) -> dict[str, Coverage]:
        """Coverage of every scored case by the training design (ADR-0074).

        Reads each case's numeric parameters through :attr:`case_parameters`
        in the order of the card's ``parameter_space.box`` and runs the
        kernel (:func:`structbench.eval.coverage.coverage`) against the
        ``train`` split's points; :attr:`novel_cases` carry the declared
        novelty flag.

        Raises
        ------
        ValueError
            If the benchmark declares no ``case_parameters`` or no
            ``parameter_space`` box, or a case lacks a box parameter.
        """
        space = self.card.parameter_space
        if self.case_parameters is None or space is None or not space.box:
            raise ValueError(
                f"benchmark {self.card.name!r} declares no case parameters or no "
                "training box; coverage cannot be computed"
            )
        keys = list(space.box)

        def point(case_id: str) -> list[float]:
            params = self.case_parameters(case_id)  # type: ignore[misc]
            missing = [k for k in keys if k not in params]
            if missing:
                raise ValueError(f"case {case_id!r} lacks box parameters {missing}")
            return [float(params[k]) for k in keys]

        train = [point(c) for c in self.splits["train"]]
        scored = [c for s in self.scored_splits for c in self.splits[s]]
        novel = set(self.novel_cases)
        covs = coverage(
            train,
            [point(c) for c in scored],
            [space.box[k] for k in keys],
            novelty=[c in novel for c in scored],
        )
        return dict(zip(scored, covs, strict=True))

    @property
    def blessed_results(self) -> tuple[BaselineResult, ...]:
        """The non-provisional entries of :attr:`results`, in declaration
        order (ADR-0046).

        Use this, not ``results``, wherever "the blessed baseline(s)" is
        meant — ``results`` may also carry provisional entries recorded
        for comparison only.
        """
        return tuple(r for r in self.results if not r.provisional)


#: The pseudo case id under which fixed restart anchors are validated at
#: construction (against the card and a fixed scored bound).
_FIXED_ANCHORS_CASE = "<fixed>"


def _strictly_increasing(values: tuple[int, ...]) -> bool:
    return all(b > a for a, b in zip(values, values[1:], strict=False))


def available_benchmarks() -> tuple[str, ...]:
    """Registered benchmark names, sorted."""
    return tuple(sorted(_MODULES))


def is_benchmark_definition(name: str | Path) -> bool:
    """Whether ``name`` is a directory holding ``benchmark.py``."""
    try:
        return (Path(name) / DEFINITION_FILE).is_file()
    except (OSError, ValueError):  # a name no filesystem accepts
        return False


def load_benchmark_definition(directory: str | Path) -> BenchmarkSpec:
    """Load the ``SPEC`` of ``<directory>/benchmark.py`` and stamp its source.

    The directory is importable while the module loads, so ``benchmark.py``
    may import a sibling (a split table, a QoI module); the siblings are
    hashed into :attr:`BenchmarkSpec.source` with the file itself.

    Raises
    ------
    ValueError
        If the file is missing or raises on import, or exports no
        :class:`BenchmarkSpec` named ``SPEC``.
    """
    root = Path(directory).resolve()
    try:
        module = load_definition_module(root, DEFINITION_FILE, qualifier="benchmark")
    except DefinitionModuleError as exc:
        raise ValueError(str(exc)) from exc
    spec = getattr(module, "SPEC", None)
    if not isinstance(spec, BenchmarkSpec):
        raise ValueError(
            f"{DEFINITION_FILE} in {root} must export a BenchmarkSpec named SPEC"
            f" (found {type(spec).__name__})"
        )
    source = BenchmarkSource(
        path=root,
        files=MappingProxyType(definition_sha256s(root, DEFINITION_FILE, module)),
    )
    return replace(spec, source=source)


def resolve_benchmark(name: str) -> BenchmarkSpec | None:
    """The spec for a registry name or a definition directory, else ``None``.

    The ``None`` is for callers that must tolerate a name they cannot
    resolve (a config built in code around an unregistered spec);
    :func:`get_benchmark` is the strict form.
    """
    if name in _MODULES:
        return get_benchmark(name)
    if is_benchmark_definition(name):
        return load_benchmark_definition(name)
    return None


def get_benchmark(name: str) -> BenchmarkSpec:
    """Resolve a benchmark's :class:`BenchmarkSpec` by registry name or path.

    ``name`` is a registry key, or a directory holding ``benchmark.py``
    (:func:`load_benchmark_definition`).

    Raises
    ------
    KeyError
        If ``name`` is neither; the message lists the registry names.
    ValueError
        If the directory's definition cannot be loaded or exports no spec.
    """
    if name in _MODULES:
        module = importlib.import_module(_MODULES[name])
        spec: BenchmarkSpec = module.SPEC
        return spec
    if is_benchmark_definition(name):
        return load_benchmark_definition(name)
    raise KeyError(
        f"unknown benchmark {name!r}; available: "
        f"{', '.join(available_benchmarks())}, or a directory holding "
        f"{DEFINITION_FILE}"
    )
