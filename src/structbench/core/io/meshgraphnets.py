"""Adapter: MeshGraphNets ``deforming_plate`` tfrecord -> canonical Case (ADR-0042).

TensorFlow is imported lazily inside :func:`read_deforming_plate` only, so
``import structbench`` never requires it (mirrors ``lsdyna.read_d3plot``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import Case, ElementBlock, Material, Metadata, Nodes, Provenance, Response
from ..validation import validate
from .lsdyna import unit_factors

_VALID_FTYPES = {"static", "dynamic", "dynamic_varlen"}


@dataclass(frozen=True)
class FieldSpec:
    """One field's shape and typing metadata, as declared in ``meta.json``.

    Parameters
    ----------
    name:
        Field name, e.g. ``"world_pos"``.
    ftype:
        MeshGraphNets storage type: one of ``"static"`` (one value for the
        whole trajectory), ``"dynamic"`` (one value per frame), or
        ``"dynamic_varlen"`` (one variable-length value per frame).
    dtype:
        Source dtype token as given in ``meta.json`` (e.g. ``"float32"``).
    shape:
        Declared shape, leading dimension is the frame count and may be
        ``-1`` for a variable/ragged extent.
    """

    name: str
    ftype: str
    dtype: str
    shape: tuple[int, ...]


def parse_meta(meta: dict) -> dict[str, FieldSpec]:
    """Parse a MeshGraphNets ``meta.json`` dict into per-field specs.

    Parameters
    ----------
    meta:
        Parsed JSON content of a MeshGraphNets dataset's ``meta.json``,
        containing at least ``"field_names"`` (list of field names) and
        ``"features"`` (dict mapping each field name to a dict with
        ``"type"``, ``"dtype"``, and ``"shape"`` keys).

    Returns
    -------
    dict[str, FieldSpec]
        One :class:`FieldSpec` per entry in ``meta["field_names"]``, keyed by
        field name.

    Raises
    ------
    ValueError
        If a field's declared ``"type"`` is not one of ``"static"``,
        ``"dynamic"``, or ``"dynamic_varlen"``.
    """
    specs: dict[str, FieldSpec] = {}
    for name in meta["field_names"]:
        f = meta["features"][name]
        if f["type"] not in _VALID_FTYPES:
            raise ValueError(f"invalid data format: field {name!r} type {f['type']!r}")
        specs[name] = FieldSpec(name, f["type"], f["dtype"], tuple(f["shape"]))
    return specs


_PLATE_MATERIAL = Material(
    material_id=1,
    source_model="COMSOL hyperelastic (deforming_plate)",
    source_params={"note": "material constants not published with the dataset"},
    canonical_model=None,
)


def build_deforming_plate_case(
    arrays: dict[str, np.ndarray],
    *,
    source_units: str,
    case_id: str,
    dataset_id: str | None = None,
) -> Case:
    """Assemble one ``deforming_plate`` trajectory into a validated 3D Case.

    Parameters
    ----------
    arrays:
        One trajectory's worth of MeshGraphNets ``deforming_plate`` arrays:
        ``cells`` ``(n_cells, 4)`` int tetra connectivity, ``node_type``
        ``(n_nodes,)`` int per-node type code, ``mesh_pos`` ``(n_nodes, 3)``
        float undeformed reference coordinates, ``world_pos``
        ``(T, n_nodes, 3)`` float deformed coordinates per frame, and
        ``stress`` ``(T, n_nodes, 1)`` float per-node von Mises stress per
        frame.
    source_units:
        ``"mass-length-time"`` token (e.g. ``"kg-m-s"``) describing the units
        ``arrays`` is expressed in; converted to SI via
        :func:`structbench.core.io.lsdyna.unit_factors`.
    case_id:
        Unique identifier for the assembled case.
    dataset_id:
        Optional identifier for the source dataset/split this trajectory was
        drawn from.

    Returns
    -------
    Case
        A validated case: ``nodes.coords`` is ``world_pos[0]`` (converted to
        SI), ``response.node["displacement"]`` is the delta from that initial
        state per frame, and ``response.time`` is the frame index (the
        dataset is quasi-static with ``dt=0``, so there is no physical time
        axis to carry).
    """
    f = unit_factors(source_units)
    world = arrays["world_pos"].astype(np.float64)  # (T, N, 3)
    coords = world[0] * f["length"]  # (N, 3)
    disp = (world - world[0][None]) * f["length"]  # (T, N, 3), delta-from-initial
    n_nodes = coords.shape[0]
    n_frames = world.shape[0]

    nodes = Nodes(
        coords=coords,
        node_id=np.arange(n_nodes, dtype=np.int64),
        node_type=arrays["node_type"].reshape(-1).astype(np.int64),
        reference_coords=arrays["mesh_pos"].astype(np.float64) * f["length"],
    )
    elements = {
        "tetra": ElementBlock(
            connectivity=arrays["cells"].astype(np.int64),
            element_id=np.arange(arrays["cells"].shape[0], dtype=np.int64),
            part_id=np.ones(arrays["cells"].shape[0], dtype=np.int64),
        )
    }
    response = Response(
        time=np.arange(n_frames, dtype=np.float64),  # pseudo-time: quasi-static (dt=0)
        node={
            "displacement": disp.astype(np.float32),
            "von_mises_stress": (
                arrays["stress"].astype(np.float64) * f["stress"]
            ).astype(np.float32),
        },
    )
    metadata = Metadata(
        case_id=case_id,
        dimension=3,
        source_units=source_units,
        dataset_id=dataset_id,
        provenance=Provenance("COMSOL", "unknown", "unknown"),
    )
    case = Case(
        metadata=metadata,
        nodes=nodes,
        elements=elements,
        materials=[_PLATE_MATERIAL],
        response=response,
    )
    validate(case)
    return case
