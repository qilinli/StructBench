"""Benchmark problem definitions (ARCHITECTURE.md; registry per ADR-0024)."""

from .card import BenchmarkCard
from .constitutive import (
    ELEMENT_FORMULATIONS,
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
    "ELEMENT_FORMULATIONS",
    "RETURN_MAP_STATE_CHANNELS",
    "TWO_SITE_DECODERS",
    "BenchmarkCard",
    "BenchmarkSource",
    "BenchmarkSpec",
    "ConstitutiveLaw",
    "available_benchmarks",
    "get_benchmark",
    "is_benchmark_definition",
    "load_benchmark_definition",
    "resolve_benchmark",
]
