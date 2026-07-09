"""Typed benchmark card: descriptive metadata for one benchmark (ADR-0027).

Physics facts are declared by hand; ML statistics are computed from the
owning module's split constants (``len(TRAIN)`` etc.) so the card and the
benchmark cannot disagree. Stats that live only in the data (particle
counts, frames, size on disk) are validated by an environment-gated test
when a data root is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Discretisation = Literal["SPH", "FEM", "coupled"]


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

    Raises
    ------
    ValueError
        If ``splits`` does not sum to ``n_cases``, ``input_frames`` is not at
        least 2, or ``protocol_rationale`` is empty.
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

    def to_json_dict(self) -> dict[str, object]:
        """Return a plain, JSON-serializable dict of the card."""
        return asdict(self)
