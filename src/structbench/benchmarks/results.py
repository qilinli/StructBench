"""Official baseline results for benchmarks (ADR-0033).

Each benchmark module records its blessed baselines as
``RESULTS: tuple[BaselineResult, ...]``, wired into the module ``SPEC``.
Results are rendered only through the generated views (``docs/benchmarks.md``
and the per-archive README) — never hand-edited into generated output. An
entry is transcribed from a blessed run's ``metrics-*.json`` and is traceable
via the run directory's ``config.json`` and the recorded git commit. Adding
or revising a result never bumps the benchmark version; protocol changes do
(ADR-0032).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class BaselineResult:
    """One official baseline entry for a benchmark.

    Parameters
    ----------
    family : str
        Model-family key from the config registry (ADR-0032), e.g. ``"cgn"``.
    label : str
        Display name, e.g. ``"CGN baseline"``.
    run_commit : str
        Git commit recorded by the blessed training run.
    run_date : str
        Run date, ``YYYY-MM-DD``.
    metrics : mapping of str to mapping of str to float
        Split name -> metric name -> value, in physical units (mm, MPa, ...).
        Split names are validated against the owning card's splits where the
        spec wires results (``BenchmarkSpec.__post_init__``).
    checkpoint : str or None
        Pointer to the blessed checkpoint: an archive-relative path into the
        private ``models/`` mirror (ADR-0037), rewritten to a public URL if
        and when checkpoints publish.
    checkpoint_sha256 : str or None
        SHA-256 digest of the checkpoint file (64 lowercase hex chars),
        making the pointer verifiable. Requires ``checkpoint``.
    notes : str
        Free-text caveats (hardware, walltime, deviations).

    Raises
    ------
    ValueError
        If a required text field is blank, ``metrics`` is empty, any split's
        metric mapping is empty, or ``checkpoint_sha256`` is malformed or set
        without ``checkpoint``.
    """

    family: str
    label: str
    run_commit: str
    run_date: str
    metrics: Mapping[str, Mapping[str, float]]
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("family", "label", "run_commit", "run_date"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.checkpoint_sha256 is not None:
            if self.checkpoint is None:
                raise ValueError("checkpoint_sha256 requires checkpoint")
            if not re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256):
                raise ValueError("checkpoint_sha256 must be 64 lowercase hex chars")
        if not self.metrics:
            raise ValueError("metrics must record at least one split")
        for split, values in self.metrics.items():
            if not values:
                raise ValueError(f"metrics[{split!r}] must not be empty")
        # Read-only proxies, matching the BenchmarkSpec mapping convention.
        frozen = {k: MappingProxyType(dict(v)) for k, v in self.metrics.items()}
        object.__setattr__(self, "metrics", MappingProxyType(frozen))
