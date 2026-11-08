"""Benchmark problem definitions (ARCHITECTURE.md; registry per ADR-0024)."""

from .card import (
    COMPLIANCE_STATUSES,
    REQUIRED_ITEMS,
    STANDARD_ITEMS,
    STRUCTURAL_SCALES,
    BenchmarkCard,
    ComplianceRow,
    GroundTruth,
    ParameterSpace,
)
from .constitutive import (
    ELEMENT_FORMULATIONS,
    LAW_DECODERS,
    RETURN_MAP_STATE_CHANNELS,
    TWO_SITE_DECODERS,
    ConstitutiveLaw,
)
from .registry import (
    BenchmarkSource,
    BenchmarkSpec,
    available_benchmarks,
    get_benchmark,
    is_benchmark_definition,
    load_benchmark_definition,
    resolve_benchmark,
)

__all__ = [
    "COMPLIANCE_STATUSES",
    "ELEMENT_FORMULATIONS",
    "LAW_DECODERS",
    "REQUIRED_ITEMS",
    "RETURN_MAP_STATE_CHANNELS",
    "STANDARD_ITEMS",
    "STRUCTURAL_SCALES",
    "TWO_SITE_DECODERS",
    "BenchmarkCard",
    "BenchmarkSource",
    "BenchmarkSpec",
    "ComplianceRow",
    "ConstitutiveLaw",
    "GroundTruth",
    "ParameterSpace",
    "available_benchmarks",
    "get_benchmark",
    "is_benchmark_definition",
    "load_benchmark_definition",
    "resolve_benchmark",
]
