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
from dataclasses import dataclass, field
from types import MappingProxyType

#: Who produced a result's numbers (ADR-0074 clause 6; open-evaluation plan §8).
TIERS: tuple[str, ...] = ("platform", "verified", "self_reported")

#: What a result was trained on (ADR-0074 clause 4).
DIVISIONS: tuple[str, ...] = ("closed", "open")


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
    scheme : str
        Prediction scheme this result was produced under, e.g.
        ``"autoregressive"``, ``"time-conditioned"``, ``"one-shot"``. Surfaced
        as the results registry's Scheme column; ``""`` (default, unknown) renders
        ``"—"``. Purely descriptive — no validation.
    provisional : bool
        ``False`` (default) = blessed — validated against its published
        anchor or protocol gate. ``True`` = best-effort implementation
        recorded for comparison, fidelity unvalidated (ADR-0044/0045/0046).
        Never read a provisional entry as a blessed baseline; use
        :attr:`BenchmarkSpec.blessed_results`, not ``BenchmarkSpec.results``,
        wherever "the blessed baseline" is meant.

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
    provisional: bool = False
    scheme: str = ""
    reference: str = ""
    pending: bool = False
    # ADR-0074 (open-evaluation plan §8): additive, defaults are the
    # platform-run values so every existing row is unchanged.
    tier: str = "platform"
    """Who produced the numbers: ``"platform"`` (the maintainer ran it;
    blessed or provisional as before), ``"verified"`` (the platform executed
    the submitter's container) or ``"self_reported"`` (ADR-0074 clause 6)."""
    division: str = "closed"
    """``"closed"`` (trained on the published train set only) or ``"open"``
    (any data, declared in ``data``; ADR-0074 clause 4)."""
    blind_tag: str = ""
    """The blind set the result was scored under (``"blind-v1"``), or empty
    when it was scored on public sets alone."""
    submitter: str = ""
    """Who submitted an external result; empty for a platform run."""
    writeup: str = ""
    """URL or DOI of the submitter's write-up."""
    data: str = ""
    """The open division's declared training data."""
    case_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    """Case id -> metric -> value for the scored cases, from which the
    registry renders the headline by coverage bin (ADR-0074 clause 7). Case
    ids are validated against the owning spec's scored splits."""

    def __post_init__(self) -> None:
        for name in ("family", "label", "run_commit", "run_date"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {self.tier!r}")
        if self.division not in DIVISIONS:
            raise ValueError(
                f"division must be one of {DIVISIONS}, got {self.division!r}"
            )
        if self.division == "open" and not self.data.strip():
            raise ValueError("an open-division result must declare its data")
        if self.tier != "platform" and not self.submitter.strip():
            raise ValueError("an external result must name its submitter")
        if self.checkpoint_sha256 is not None:
            if self.checkpoint is None:
                raise ValueError("checkpoint_sha256 requires checkpoint")
            if not re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256):
                raise ValueError("checkpoint_sha256 must be 64 lowercase hex chars")
        # A ``pending`` entry is a placeholder for a still-training baseline
        # (its run hasn't produced numbers yet), so empty metrics are allowed;
        # the results registry renders its row as "—". Every other entry must carry
        # metrics.
        if not self.metrics and not self.pending:
            raise ValueError("metrics must record at least one split")
        for split, values in self.metrics.items():
            if not values:
                raise ValueError(f"metrics[{split!r}] must not be empty")
        # Read-only proxies, matching the BenchmarkSpec mapping convention.
        frozen = {k: MappingProxyType(dict(v)) for k, v in self.metrics.items()}
        object.__setattr__(self, "metrics", MappingProxyType(frozen))
        for case_id, values in self.case_metrics.items():
            if not values:
                raise ValueError(f"case_metrics[{case_id!r}] must not be empty")
        frozen_cases = {
            k: MappingProxyType(dict(v)) for k, v in self.case_metrics.items()
        }
        object.__setattr__(self, "case_metrics", MappingProxyType(frozen_cases))
