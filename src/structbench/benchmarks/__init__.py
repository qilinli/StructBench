"""Benchmark problem definitions (ARCHITECTURE.md; registry per ADR-0024)."""

from .card import BenchmarkCard
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
    "BenchmarkCard",
    "BenchmarkSource",
    "BenchmarkSpec",
    "available_benchmarks",
    "get_benchmark",
    "is_benchmark_definition",
    "load_benchmark_definition",
    "resolve_benchmark",
]
