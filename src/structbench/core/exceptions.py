"""Custom exceptions for the StructBench core."""

from __future__ import annotations


class StructBenchError(Exception):
    """Base class for all StructBench errors."""


class SchemaError(StructBenchError):
    """A case violates the schema's structure or validity rules (ADR-0012)."""
