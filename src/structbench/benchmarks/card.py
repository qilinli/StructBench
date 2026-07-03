"""Typed benchmark card: descriptive metadata for one benchmark (ADR-0025).

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
class BenchmarkCard:
    """Descriptive metadata for one benchmark (ADR-0025).

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
    qois, fields : tuple of str
        QoI names and the response fields available in the canonical data.
    particles_per_case : str
        Human-readable particle-count range (e.g. ``"4804-8004"``).
    n_frames : int
        Response frames per case.
    output_dt_ms : float
        Output interval of the source simulations, milliseconds.
    size_gb : float or None
        Canonical dataset size on disk; ``None`` until measured.

    Raises
    ------
    ValueError
        If ``splits`` does not sum to ``n_cases``.
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
    size_gb: float | None = None

    def __post_init__(self) -> None:
        total = sum(self.splits.values())
        if self.n_cases != total:
            raise ValueError(
                f"n_cases ({self.n_cases}) != sum of splits ({total})"
            )

    def to_json_dict(self) -> dict[str, object]:
        """Return a plain, JSON-serializable dict of the card."""
        return asdict(self)
