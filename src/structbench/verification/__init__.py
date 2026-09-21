"""Reference-data verification: is a simulation run trustworthy? (ADR-0066).

Deterministic checks that judge the reference data itself, not a model.
``measure`` turns evidence into threshold-free measurements or typed
absences; ``judge`` turns measurements plus the platform's criteria into one
of five verdicts and never touches data. This package imports only
``structbench.core`` and ``structbench.datasets``; benchmark declarations are
passed in by the caller.
"""

from __future__ import annotations

from .results import (
    CaseMeasurements,
    Category,
    DatasetMeasurements,
    Location,
    Measurement,
    Status,
    Verdict,
)

__all__ = [
    "CaseMeasurements",
    "Category",
    "DatasetMeasurements",
    "Location",
    "Measurement",
    "Status",
    "Verdict",
]
