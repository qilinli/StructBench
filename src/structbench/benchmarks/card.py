"""Typed benchmark card: descriptive metadata for one benchmark (ADR-0027).

Physics facts are declared by hand; ML statistics are computed from the
owning module's split constants (``len(TRAIN)`` etc.) so the card and the
benchmark cannot disagree. Stats that live only in the data (particle
counts, frames, size on disk) are validated by an environment-gated test
when a data root is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..core import UnitsAnchor
from ..core.evidence import CardDiscretisation as Discretisation

#: Structural tiers a card may declare (ADR-0065's roadmap vocabulary: the
#: metal *component-tier* benchmark, the crash benchmark as an *assembly-tier*
#: question). Empty on the card means undeclared.
STRUCTURAL_SCALES: tuple[str, ...] = ("material", "specimen", "component", "assembly")

#: The reference-data standard's items, in render order (plan of 2026-11-08).
#: The first four are required of a benchmark admitted after 2026-10-01
#: (ADR-0073 D14); the next two are recorded where they exist and not
#: required (D14); the last is ADR-0074's blind set. Keys are the card
#: vocabulary, values the rendered labels.
STANDARD_ITEMS: dict[str, str] = {
    "convergence_run": "Convergence run",
    "noise_floor": "Measured noise floor",
    "survivor_log": "Survivor log of discarded cases",
    "constitutive_sensitivity": "Constitutive sensitivity study",
    "physical_test_anchor": "Physical-test anchor",
    "second_solver_reproduction": "Second-solver reproduction",
    "blind_evaluation": "Blind evaluation set",
}

#: The items ADR-0073 D14 requires of a newly admitted benchmark.
REQUIRED_ITEMS: tuple[str, ...] = (
    "convergence_run",
    "noise_floor",
    "survivor_log",
    "constitutive_sensitivity",
)

#: What a compliance cell may say. ``pass``: the item exists and is recorded;
#: ``missing``: it does not exist yet; ``not_applicable``: the item has no
#: meaning for this data; ``never``: it cannot be produced.
COMPLIANCE_STATUSES: tuple[str, ...] = ("pass", "missing", "not_applicable", "never")


@dataclass(frozen=True)
class ComplianceRow:
    """One row of a benchmark's data-standards compliance table.

    Parameters
    ----------
    item : str
        A key of :data:`STANDARD_ITEMS`.
    status : str
        One of :data:`COMPLIANCE_STATUSES`.
    reason : str
        One short sentence naming the evidence, or what is absent and why.
        Never blank: a bare status hides more than it shows.

    Raises
    ------
    ValueError
        If ``item`` or ``status`` is outside its vocabulary, or ``reason`` is
        blank.
    """

    item: str
    status: str
    reason: str

    def __post_init__(self) -> None:
        if self.item not in STANDARD_ITEMS:
            raise ValueError(
                f"unknown standard item {self.item!r}; "
                f"expected one of {tuple(STANDARD_ITEMS)}"
            )
        if self.status not in COMPLIANCE_STATUSES:
            raise ValueError(
                f"unknown compliance status {self.status!r}; "
                f"expected one of {COMPLIANCE_STATUSES}"
            )
        if not self.reason.strip():
            raise ValueError(f"compliance row {self.item!r} needs a reason")


@dataclass(frozen=True)
class ParameterSpace:
    """The training design of a benchmark, as ADR-0074 clause 3 asks the card
    to declare it.

    Parameters
    ----------
    design : str
        How the training cases were placed: the grid levels, a Sobol count,
        or how the source authors sampled.
    box : dict of str to tuple of float
        The declared box of the *training* design, one ``(low, high)`` per
        numeric parameter, keyed by a name that carries the unit (e.g.
        ``"impact_velocity_m_s"``). A scored case outside it reads as
        *beyond* under ADR-0074 clause 2.
    margin : str
        The declared margin beyond the box within which blind cases may lie
        (ADR-0074 clause 1). Empty means none is declared.
    categorical : dict of str to tuple of str
        Levels of each categorical parameter; an unseen level is *novelty*
        (ADR-0074 clause 2).

    Raises
    ------
    ValueError
        If ``design`` is blank or a box range is not ``low <= high``.
    """

    design: str
    box: dict[str, tuple[float, float]] = field(default_factory=dict)
    margin: str = ""
    categorical: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.design.strip():
            raise ValueError("parameter space must name its design")
        for name, (low, high) in self.box.items():
            if not low <= high:
                raise ValueError(f"box {name!r}: low {low} exceeds high {high}")


@dataclass(frozen=True)
class GroundTruth:
    """The ground-truth regime of a benchmark and what is known of its own
    uncertainty (ADR-0065 follow-up 2).

    Parameters
    ----------
    regime : str
        The solver, formulation and what one case is (one run, an ensemble).
    uncertainty : str
        What is measured about the reference's own error — a noise floor, a
        refinement difference — or the statement that nothing is. Never
        blank: an unmeasured uncertainty is said, not implied.

    Raises
    ------
    ValueError
        If either field is blank.
    """

    regime: str
    uncertainty: str

    def __post_init__(self) -> None:
        if not self.regime.strip():
            raise ValueError("ground truth must name its regime")
        if not self.uncertainty.strip():
            raise ValueError(
                "ground truth must state its uncertainty, or that it is unmeasured"
            )


@dataclass(frozen=True)
class BenchmarkFigure:
    """One figure on a benchmark's generated landing page (ADR-0036).

    Parameters
    ----------
    path : str
        Repo-relative path to a committed asset (e.g.
        ``"assets/taylor_rollout.gif"``), or an absolute URL. Figures are
        deliberately promoted from gitignored ``runs/**/plots`` into
        ``assets/``; a test checks every path exists.
    caption : str
        One-line caption rendered beneath the figure.
    alt : str
        Accessibility alt text; falls back to ``caption`` when blank.

    Raises
    ------
    ValueError
        If ``path`` or ``caption`` is blank.
    """

    path: str
    caption: str
    alt: str = ""

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("figure path must be non-empty")
        if not self.caption.strip():
            raise ValueError("figure caption must be non-empty")


@dataclass(frozen=True)
class BenchmarkCard:
    """Descriptive metadata for one benchmark (ADR-0027).

    Parameters
    ----------
    name, version, description, provenance, data_license : str
        Identity block: leaderboard name, benchmark version, one-line
        description, data provenance (paper / who ran the simulations),
        and the data license.
    units_anchors : tuple of UnitsAnchor
        SI anchors that pin the declared unit system against something
        outside its label (ADR-0066 E10b): each states the SI value of a
        named input quantity, so a wrong ``source_units`` shows up as a
        factor. Empty until a benchmark declares them.
    solver, loading, source_units, geometry : str
        Physics block, for the structural engineer. ``source_units`` is
        the solver's unit convention (e.g. ``"g-mm-ms"``); canonical
        storage is SI regardless (ADR-0012).
    discretisation : {"SPH", "FEM", "coupled"}
        Spatial discretisation of the source simulations.
    materials : tuple of str
        Solver material models, verbatim keyword names.
    erosion : bool
        Whether the source simulations delete elements.
    n_cases : int
        Benchmark cases across all splits (held-aside cases excluded).
    splits : dict of str to int
        Split sizes; must sum to ``n_cases``.
    task, aux_field, aux_unit : str
        ML block: the learning task, the auxiliary target's canonical
        field name, and its reporting unit.
    qois : tuple of str
        QoI names.
    fields : tuple of str
        Complete response fields of the canonical data, as namespaced
        HDF5 paths (``node/*``, ``sph/*`` element fields, ``global/*``
        scalars). Derived conveniences (e.g. particle positions =
        coords + node/displacement) are loader behavior, not stored
        fields.
    particles_per_case : str
        Human-readable particle-count range (e.g. ``"4800-8000"``).
    n_frames : int
        Response frames per case.
    output_dt_ms : float
        Output interval of the source simulations, milliseconds.
    input_frames : int
        Benchmark protocol (ADR-0035): the number of ground-truth frames a
        model observes at rollout start, which by construction equals the
        model's input history length (there is no history backfill). The scored
        span is frames ``[input_frames, end]`` for every model, and a run whose
        model ``input_frames`` differs from this value is rejected.
    protocol_rationale : str
        Why the protocol values are what they are — the recorded conclusion
        of the mandatory ground-truth timeline analysis (ADR-0032 §5).
        Rendered alongside the protocol in the generated benchmark docs.
    size_gb : float or None
        Canonical dataset size on disk; ``None`` until measured.
    horizon : str
        Benchmark protocol: rollout extent. ``"full"`` scores to the
        (trimmed) end of trajectory.
    eval_times : str
        Benchmark protocol: where predictions are scored. ``"native"`` means
        the solver's output times; internal time-stepping is a model choice.
    overview : str
        Optional multi-paragraph problem/physics narrative (markdown) for the
        benchmark's landing page (ADR-0036); empty renders no lead section.
    figures : tuple of BenchmarkFigure
        Optional ordered figures for the landing page (ADR-0036); empty
        renders no figures section.
    data_access : str
        Optional one-paragraph markdown access story for this benchmark's
        dataset (public source pointer, DOI, request path), rendered as the
        landing page's "Dataset access" paragraph. Empty renders the
        ADR-0040 default (maintainer-held, shared on request). Name repo
        paths in backticks rather than relative links — the text also ships
        in ``card.json``, where page-relative links would dangle.
    data_access_label : str
        Optional short form of ``data_access`` for the landing page's
        "Data at a glance" bullet (e.g. ``"public source — download +
        convert"`` or a DOI link). Empty renders the ADR-0040 default
        label. Same no-relative-links rule as ``data_access``.
    structural_scale : str
        Optional tier of the structure, one of :data:`STRUCTURAL_SCALES`
        (ADR-0065 follow-up 2); empty is undeclared.
    load_type : str
        Optional kind of loading in a phrase (impact, stress wave,
        quasi-static contact); ``loading`` holds the parameters.
    autonomy : str
        Optional: what the surrogate is given during the scored rollout
        beyond the seed frames and the case parameters — nothing, or bodies
        driven from ground truth.
    parameter_space : ParameterSpace or None
        Optional training design, declared box, margin and categorical
        levels (ADR-0074 clause 3).
    ground_truth : GroundTruth or None
        Optional ground-truth regime with its measured uncertainty.
    compliance : tuple of ComplianceRow
        Optional data-standards compliance table, one row per item of
        :data:`STANDARD_ITEMS` (ADR-0073 D14; the blind row ADR-0074). Empty
        renders no table; a registered benchmark covers every item (a test
        enforces it).

    Raises
    ------
    ValueError
        If ``splits`` does not sum to ``n_cases``, ``input_frames`` is not at
        least 2, ``protocol_rationale`` is empty, ``structural_scale`` is
        outside its vocabulary, or ``compliance`` repeats an item.
    """

    # identity
    name: str
    version: str
    description: str
    provenance: str
    data_license: str
    # physics — for the structural engineer
    solver: str
    discretisation: Discretisation
    materials: tuple[str, ...]
    erosion: bool
    loading: str
    source_units: str
    geometry: str
    # ml — for the ML researcher
    n_cases: int
    splits: dict[str, int]
    task: str
    aux_field: str
    aux_unit: str
    qois: tuple[str, ...]
    fields: tuple[str, ...]
    particles_per_case: str
    n_frames: int
    output_dt_ms: float
    # protocol (ADR-0032, ADR-0035) — task definition, pinned per benchmark ADR
    input_frames: int
    protocol_rationale: str
    size_gb: float | None = None
    horizon: str = "full"
    eval_times: str = "native"
    # landing page (ADR-0036) — non-derivable narrative + figures
    overview: str = ""
    figures: tuple[BenchmarkFigure, ...] = ()
    data_access: str = ""
    data_access_label: str = ""
    units_anchors: tuple[UnitsAnchor, ...] = ()
    # V&V identity (ADR-0065 follow-up 2, plan of 2026-11-08) — all optional
    structural_scale: str = ""
    load_type: str = ""
    autonomy: str = ""
    parameter_space: ParameterSpace | None = None
    ground_truth: GroundTruth | None = None
    compliance: tuple[ComplianceRow, ...] = ()

    def __post_init__(self) -> None:
        total = sum(self.splits.values())
        if self.n_cases != total:
            raise ValueError(f"n_cases ({self.n_cases}) != sum of splits ({total})")
        if self.input_frames < 2:
            raise ValueError(f"input_frames must be >= 2, got {self.input_frames}")
        if not self.protocol_rationale.strip():
            raise ValueError(
                "protocol_rationale must record the timeline analysis "
                "behind the protocol values (ADR-0032 §5)"
            )
        if self.structural_scale and self.structural_scale not in STRUCTURAL_SCALES:
            raise ValueError(
                f"unknown structural_scale {self.structural_scale!r}; "
                f"expected one of {STRUCTURAL_SCALES} or empty"
            )
        items = [row.item for row in self.compliance]
        if len(set(items)) != len(items):
            dupes = sorted({i for i in items if items.count(i) > 1})
            raise ValueError(f"compliance table repeats {dupes}")

    def compliance_covers_standard(self) -> bool:
        """Whether the compliance table carries a row for every standard item."""
        return {row.item for row in self.compliance} == set(STANDARD_ITEMS)

    def to_json_dict(self) -> dict[str, object]:
        """Return a plain, JSON-serializable dict of the card."""
        return asdict(self)
