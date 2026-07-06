"""Validation of a :class:`~structbench.core.schema.Case` against ADR-0012.

This checks the structural and validity-tier rules a case must satisfy. It
operates on the in-memory :class:`Case` (not the HDF5 file); the writer calls
it before persisting, so an invalid case never reaches disk.

Coverage matches the field groups currently modelled in
:mod:`structbench.core.schema`. Tiers for not-yet-modelled groups (parts,
sections, loading, ...) will be added alongside those groups.
"""

from __future__ import annotations

from .exceptions import SchemaError
from .schema import UNITS_CONVENTION, Case


def validate(case: Case) -> None:
    """Validate a case in place, raising on the first violation.

    Parameters
    ----------
    case:
        The case to check.

    Raises
    ------
    SchemaError
        If the case violates a required-tier rule or a structural invariant
        (shape/length agreement, dimension, units convention).
    """
    _validate_metadata(case)
    dim = case.metadata.dimension
    n_nodes = _validate_nodes(case, dim)
    _validate_elements(case, n_nodes)
    _validate_materials(case)
    if case.response is not None:
        _validate_response(case, n_nodes, dim)


def _validate_metadata(case: Case) -> None:
    md = case.metadata
    if not md.case_id:
        raise SchemaError("metadata.case_id must be a non-empty string")
    if md.dimension not in (2, 3):
        raise SchemaError(f"metadata.dimension must be 2 or 3, got {md.dimension}")
    if md.units_convention != UNITS_CONVENTION:
        raise SchemaError(
            f"metadata.units_convention must be {UNITS_CONVENTION!r}, "
            f"got {md.units_convention!r}"
        )
    if not md.schema_version:
        raise SchemaError("metadata.schema_version must be set")


def _validate_nodes(case: Case, dim: int) -> int:
    coords = case.nodes.coords
    if coords.ndim != 2 or coords.shape[1] != dim:
        raise SchemaError(
            f"nodes.coords must have shape (n_nodes, {dim}), got {coords.shape}"
        )
    n_nodes = int(coords.shape[0])
    if case.nodes.node_id.shape != (n_nodes,):
        raise SchemaError(
            f"nodes.node_id must have shape ({n_nodes},), "
            f"got {case.nodes.node_id.shape}"
        )
    return n_nodes


def _validate_elements(case: Case, n_nodes: int) -> None:
    if not case.elements:
        raise SchemaError("a case must contain at least one element block")
    for etype, block in case.elements.items():
        conn = block.connectivity
        if conn.ndim != 2:
            raise SchemaError(
                f"elements/{etype}.connectivity must be 2D, got {conn.ndim}D"
            )
        n_elem = int(conn.shape[0])
        if block.element_id.shape != (n_elem,):
            raise SchemaError(
                f"elements/{etype}.element_id must have shape ({n_elem},), "
                f"got {block.element_id.shape}"
            )
        if block.part_id.shape != (n_elem,):
            raise SchemaError(
                f"elements/{etype}.part_id must have shape ({n_elem},), "
                f"got {block.part_id.shape}"
            )
        if n_elem and conn.shape[1] < 1:
            raise SchemaError(
                f"elements/{etype}.connectivity must have >=1 node per element, "
                f"got shape {conn.shape}"
            )
        if n_elem and (conn.min() < 0 or conn.max() >= n_nodes):
            raise SchemaError(
                f"elements/{etype}.connectivity references a node outside "
                f"[0, {n_nodes})"
            )


def _validate_materials(case: Case) -> None:
    if not case.materials:
        raise SchemaError("a case must contain at least one material")
    for mat in case.materials:
        if not mat.source_model:
            raise SchemaError(
                f"material {mat.material_id} must have a non-empty source_model"
            )


def _validate_response(case: Case, n_nodes: int, dim: int) -> None:
    assert case.response is not None  # narrowed by caller
    resp = case.response
    if resp.time.ndim != 1:
        raise SchemaError("response.time must be 1D")
    n_frames = int(resp.time.shape[0])
    if n_frames < 1:
        raise SchemaError(
            "a simulated case must have at least one response frame (the t=0 "
            "state lives at frame 0, ADR-0012)"
        )
    if "displacement" not in resp.node:
        raise SchemaError("a simulated case must contain response.node['displacement']")
    for fieldname, arr in resp.node.items():
        if arr.shape != (n_frames, n_nodes, dim):
            raise SchemaError(
                f"response.node[{fieldname!r}] must have shape "
                f"({n_frames}, {n_nodes}, {dim}), got {arr.shape}"
            )
    for etype, fields in resp.element.items():
        if etype not in case.elements:
            raise SchemaError(
                f"response.element has type {etype!r} with no matching "
                f"elements/{etype} block"
            )
        n_elem = int(case.elements[etype].connectivity.shape[0])
        for fieldname, arr in fields.items():
            if arr.shape[:2] != (n_frames, n_elem):
                raise SchemaError(
                    f"response.element[{etype!r}][{fieldname!r}] must lead with "
                    f"({n_frames}, {n_elem}), got {arr.shape}"
                )
    for name, arr in resp.globals_.items():
        if arr.shape != (n_frames,):
            raise SchemaError(
                f"response.globals_[{name!r}] must have shape ({n_frames},), "
                f"got {arr.shape}"
            )
