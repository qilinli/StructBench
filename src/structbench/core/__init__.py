"""Core data structures and I/O primitives for StructBench.

This module's public surface is the case schema, its HDF5 reader/writer, the
validator, the custom exceptions, and the solver-neutral evidence records
(ADR-0066). Other packages import only from here.
"""

from __future__ import annotations

from .evidence import (
    PLATFORM_REASONS,
    Absence,
    AbsenceReason,
    DeclaredFacts,
    EvidenceItem,
    InputFacts,
    MaterialInput,
    PartTraits,
    RigidPlane,
    UnitsAnchor,
)
from .exceptions import SchemaError, StructBenchError
from .io import (
    build_deforming_plate_case,
    lsdyna_to_case,
    parse_meta,
    read_case,
    read_input_facts,
    write_case,
)
from .schema import (
    SCHEMA_VERSION,
    UNITS_CONVENTION,
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Provenance,
    Response,
)
from .validation import validate

__all__ = [
    "PLATFORM_REASONS",
    "Absence",
    "AbsenceReason",
    "DeclaredFacts",
    "EvidenceItem",
    "InputFacts",
    "MaterialInput",
    "PartTraits",
    "RigidPlane",
    "UnitsAnchor",
    "SCHEMA_VERSION",
    "UNITS_CONVENTION",
    "Case",
    "ElementBlock",
    "Material",
    "Metadata",
    "Nodes",
    "Provenance",
    "Response",
    "SchemaError",
    "StructBenchError",
    "read_case",
    "read_input_facts",
    "write_case",
    "lsdyna_to_case",
    "validate",
    "build_deforming_plate_case",
    "parse_meta",
]
