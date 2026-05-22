"""In-memory representation of a StructBench *case*.

A case is one record — one specimen under one scenario, with the resulting
response (ADR-0011). This module defines the typed dataclass tree that
:mod:`structbench.core.io` reads into and writes from. The field set and
validity tiers follow ADR-0012; the on-disk HDF5 layout follows ADR-0013.

Only the field groups needed for the first I/O slice are modelled here:
metadata, geometry/topology (nodes, elements), materials, and response.
Groups deferred to a later checkpoint (parts, sections, boundary_conditions,
loading, initial_conditions, time_curves, sets, sensors) are not yet present;
they are valid schema content and will be added without breaking this tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Schema version pinned by ADR-0013. Additive field changes bump the minor
#: version; structural changes bump the major version (with a superseding ADR).
SCHEMA_VERSION = "0.1.0"

#: Canonical units convention for every case (ADR-0012).
UNITS_CONVENTION = "SI"


@dataclass
class Provenance:
    """Origin of a solver-generated case (required when solver-generated)."""

    solver_name: str
    solver_version: str
    generation_date: str


@dataclass
class Metadata:
    """Case-level metadata (ADR-0012 ``metadata`` group)."""

    case_id: str
    dimension: int
    schema_version: str = SCHEMA_VERSION
    units_convention: str = UNITS_CONVENTION
    provenance: Provenance | None = None
    source_units: str | None = None
    source_deck: str | None = None
    asset_id: str | None = None
    dataset_id: str | None = None


@dataclass
class Nodes:
    """Nodal coordinates and original solver node IDs."""

    coords: NDArray[np.float64]  # (n_nodes, dim)
    node_id: NDArray[np.int64]  # (n_nodes,)


@dataclass
class ElementBlock:
    """Connectivity and identity for one element type (e.g. ``solid``, ``sph``).

    ``connectivity`` is 0-indexed into :attr:`Nodes.coords`. For SPH it is the
    degenerate ``(n_elem, 1)`` case (one particle references one node).
    """

    connectivity: NDArray[np.int64]  # (n_elem, n_nodes_per_elem)
    element_id: NDArray[np.int64]  # (n_elem,)
    part_id: NDArray[np.int64]  # (n_elem,)


@dataclass
class Material:
    """A material entry, hybrid canonical / solver-native (ADR-0012).

    ``canonical_model`` is a name from the canonical enum, or ``None`` when no
    clean mapping exists. ``source_model`` / ``source_params`` carry the
    solver-native description verbatim; solver sub-models linked to the
    material (EOS, hourglass) are nested inside ``source_params``.
    """

    material_id: int
    source_model: str
    source_params: dict[str, Any]
    canonical_model: str | None = None


@dataclass
class Response:
    """Temporal evolution of the case under its scenario (ADR-0012).

    All fields share the single global time axis :attr:`time`. Per-node and
    per-element fields are indexed ``(n_frames, n_entities, ...)``.
    """

    time: NDArray[np.float64]  # (n_frames,)
    node: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    element: dict[str, dict[str, NDArray[np.float32]]] = field(default_factory=dict)
    globals_: dict[str, NDArray[np.float32]] = field(default_factory=dict)


@dataclass
class Case:
    """One StructBench case: specimen + scenario + (optional) response.

    A case with ``response is None`` is a valid "stub" — specimen and scenario
    are specified but the simulation has not been run (ADR-0012).
    """

    metadata: Metadata
    nodes: Nodes
    elements: dict[str, ElementBlock]
    materials: list[Material]
    response: Response | None = None
