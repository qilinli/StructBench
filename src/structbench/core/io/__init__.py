"""HDF5 readers and writers for the canonical case format (ADR-0013).

One case is one HDF5 file. Paths are lowercase ``snake_case`` mirroring the
field names; small scalars are attributes, arrays are datasets. Geometry and
the time axis are float64, bulk response is float32, ids and connectivity are
int64, strings are variable-length UTF-8. Response arrays are gzip-compressed
(level 4) and chunked along the frame axis. Heterogeneous solver-native data
(``materials`` ``source_params``) is stored as JSON strings.

Note on ``source_deck``: ADR-0013 specifies a gzip-compressed deck. HDF5's
gzip filter does not compress the heap backing a variable-length string, so
the deck is stored here as an uncompressed vlen UTF-8 scalar dataset. Faithful
compression needs a byte-array representation; that refinement is tracked for
a later checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..schema import (
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Provenance,
    Response,
)
from ..validation import validate
from .lsdyna import (
    build_case,
    extract_geometry,
    extract_response,
    lsdyna_to_case,
    parse_deck_materials,
    read_d3plot,
    unit_factors,
)

__all__ = [
    "read_case",
    "write_case",
    "lsdyna_to_case",
    "read_d3plot",
    "build_case",
    "extract_geometry",
    "extract_response",
    "parse_deck_materials",
    "unit_factors",
]

_STR_DT = h5py.string_dtype(encoding="utf-8")
_GZIP_LEVEL = 4
_CHUNK_TARGET_BYTES = 1 << 20  # ~1 MiB per chunk


def write_case(case: Case, path: str | Path) -> None:
    """Validate ``case`` and write it to ``path`` as a canonical HDF5 file.

    Parameters
    ----------
    case:
        The case to persist. It is validated first; an invalid case raises
        before any file is written.
    path:
        Destination ``.h5`` path. Overwritten if it exists.
    """
    validate(case)
    with h5py.File(path, "w") as f:
        _write_metadata(f, case.metadata)
        _write_nodes(f, case.nodes)
        _write_elements(f, case.elements)
        _write_materials(f, case.materials)
        if case.response is not None:
            _write_response(f, case.response)


def read_case(path: str | Path) -> Case:
    """Read a canonical HDF5 file into a :class:`Case`.

    Parameters
    ----------
    path:
        Source ``.h5`` path.

    Returns
    -------
    Case
        The reconstructed case.
    """
    with h5py.File(path, "r") as f:
        return Case(
            metadata=_read_metadata(f),
            nodes=_read_nodes(f),
            elements=_read_elements(f),
            materials=_read_materials(f),
            response=_read_response(f) if "response" in f else None,
        )


# --- metadata ---------------------------------------------------------------


def _write_metadata(f: h5py.File, md: Metadata) -> None:
    g = f.create_group("metadata")
    g.attrs["case_id"] = md.case_id
    g.attrs["schema_version"] = md.schema_version
    g.attrs["units_convention"] = md.units_convention
    g.attrs["dimension"] = md.dimension
    for name in ("source_units", "asset_id", "dataset_id"):
        value = getattr(md, name)
        if value is not None:
            g.attrs[name] = value
    if md.source_deck is not None:
        g.create_dataset("source_deck", data=md.source_deck, dtype=_STR_DT)
    if md.provenance is not None:
        pg = g.create_group("provenance")
        pg.attrs["solver_name"] = md.provenance.solver_name
        pg.attrs["solver_version"] = md.provenance.solver_version
        pg.attrs["generation_date"] = md.provenance.generation_date


def _read_metadata(f: h5py.File) -> Metadata:
    g = f["metadata"]
    provenance = None
    if "provenance" in g:
        pa = g["provenance"].attrs
        provenance = Provenance(
            solver_name=str(pa["solver_name"]),
            solver_version=str(pa["solver_version"]),
            generation_date=str(pa["generation_date"]),
        )
    source_deck = _decode(g["source_deck"][()]) if "source_deck" in g else None
    return Metadata(
        case_id=str(g.attrs["case_id"]),
        dimension=int(g.attrs["dimension"]),
        schema_version=str(g.attrs["schema_version"]),
        units_convention=str(g.attrs["units_convention"]),
        provenance=provenance,
        source_units=_opt_attr(g, "source_units"),
        source_deck=source_deck,
        asset_id=_opt_attr(g, "asset_id"),
        dataset_id=_opt_attr(g, "dataset_id"),
    )


def _opt_attr(g: h5py.Group, name: str) -> str | None:
    return str(g.attrs[name]) if name in g.attrs else None


# --- geometry / topology ----------------------------------------------------


def _write_nodes(f: h5py.File, nodes: Nodes) -> None:
    g = f.create_group("nodes")
    g.create_dataset("coords", data=np.asarray(nodes.coords, dtype=np.float64))
    g.create_dataset("node_id", data=np.asarray(nodes.node_id, dtype=np.int64))


def _read_nodes(f: h5py.File) -> Nodes:
    g = f["nodes"]
    return Nodes(
        coords=np.asarray(g["coords"][()], dtype=np.float64),
        node_id=np.asarray(g["node_id"][()], dtype=np.int64),
    )


def _write_elements(f: h5py.File, elements: dict[str, ElementBlock]) -> None:
    g = f.create_group("elements")
    for etype, block in elements.items():
        eg = g.create_group(etype)
        eg.create_dataset(
            "connectivity", data=np.asarray(block.connectivity, dtype=np.int64)
        )
        eg.create_dataset(
            "element_id", data=np.asarray(block.element_id, dtype=np.int64)
        )
        eg.create_dataset("part_id", data=np.asarray(block.part_id, dtype=np.int64))


def _read_elements(f: h5py.File) -> dict[str, ElementBlock]:
    g = f["elements"]
    blocks: dict[str, ElementBlock] = {}
    for etype in g:
        eg = g[etype]
        blocks[etype] = ElementBlock(
            connectivity=np.asarray(eg["connectivity"][()], dtype=np.int64),
            element_id=np.asarray(eg["element_id"][()], dtype=np.int64),
            part_id=np.asarray(eg["part_id"][()], dtype=np.int64),
        )
    return blocks


# --- materials --------------------------------------------------------------


def _write_materials(f: h5py.File, materials: list[Material]) -> None:
    g = f.create_group("materials")
    g.create_dataset(
        "material_id",
        data=np.array([m.material_id for m in materials], dtype=np.int64),
    )
    g.create_dataset(
        "source_model",
        data=np.array([m.source_model for m in materials], dtype=object),
        dtype=_STR_DT,
    )
    g.create_dataset(
        "canonical_model",
        data=np.array([m.canonical_model or "" for m in materials], dtype=object),
        dtype=_STR_DT,
    )
    g.create_dataset(
        "source_params",
        data=np.array([json.dumps(m.source_params) for m in materials], dtype=object),
        dtype=_STR_DT,
    )


def _read_materials(f: h5py.File) -> list[Material]:
    g = f["materials"]
    material_id = g["material_id"][()]
    source_model = g["source_model"][()]
    canonical_model = g["canonical_model"][()]
    source_params = g["source_params"][()]
    materials: list[Material] = []
    for i in range(len(material_id)):
        canonical = _decode(canonical_model[i])
        materials.append(
            Material(
                material_id=int(material_id[i]),
                source_model=_decode(source_model[i]),
                source_params=json.loads(_decode(source_params[i])),
                canonical_model=canonical or None,
            )
        )
    return materials


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


# --- response ---------------------------------------------------------------


def _write_response(f: h5py.File, resp: Response) -> None:
    g = f.create_group("response")
    g.create_dataset("time/t", data=np.asarray(resp.time, dtype=np.float64))
    ng = g.create_group("node")
    for fieldname, arr in resp.node.items():
        _write_compressed(ng, fieldname, np.asarray(arr, dtype=np.float32))
    if resp.element:
        eg = g.create_group("element")
        for etype, fields in resp.element.items():
            etg = eg.create_group(etype)
            for fieldname, arr in fields.items():
                _write_compressed(etg, fieldname, np.asarray(arr, dtype=np.float32))
    if resp.globals_:
        gg = g.create_group("global")
        for name, arr in resp.globals_.items():
            _write_compressed(gg, name, np.asarray(arr, dtype=np.float32))


def _read_response(f: h5py.File) -> Response:
    g = f["response"]
    node = {name: np.asarray(g["node"][name][()], np.float32) for name in g["node"]}
    element: dict[str, dict[str, np.ndarray]] = {}
    if "element" in g:
        for etype in g["element"]:
            etg = g["element"][etype]
            element[etype] = {
                name: np.asarray(etg[name][()], np.float32) for name in etg
            }
    globals_: dict[str, np.ndarray] = {}
    if "global" in g:
        gg = g["global"]
        globals_ = {name: np.asarray(gg[name][()], np.float32) for name in gg}
    return Response(
        time=np.asarray(g["time/t"][()], dtype=np.float64),
        node=node,
        element=element,
        globals_=globals_,
    )


def _write_compressed(group: h5py.Group, name: str, arr: np.ndarray) -> None:
    group.create_dataset(
        name,
        data=arr,
        chunks=_frame_chunks(arr),
        compression="gzip",
        compression_opts=_GZIP_LEVEL,
    )


def _frame_chunks(arr: np.ndarray) -> tuple[int, ...]:
    """Chunk along the frame (leading) axis, ~1 MiB per chunk."""
    n_frames = arr.shape[0]
    per_frame = int(arr.itemsize) * int(np.prod(arr.shape[1:], dtype=np.int64))
    per_frame = max(per_frame, 1)
    c = max(1, min(n_frames, _CHUNK_TARGET_BYTES // per_frame))
    return (c, *arr.shape[1:])
