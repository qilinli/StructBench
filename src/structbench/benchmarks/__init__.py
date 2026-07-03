"""Benchmark problem definitions (ARCHITECTURE.md; registry per ADR-0024)."""

from .card import BenchmarkCard
from .registry import BenchmarkSpec, available_benchmarks, get_benchmark

__all__ = [
    "BenchmarkCard",
    "BenchmarkSpec",
    "available_benchmarks",
    "get_benchmark",
]
