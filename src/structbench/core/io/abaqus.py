"""Adapter: an ``abaqus-npz/1`` export -> a canonical :class:`Case` (ADR-0069).

``data_generation/abaqus/odb_export.py`` runs under ``abaqus python`` and
writes the ``.odb`` as plain arrays in the deck's own units, keyed by Abaqus
labels (layout: ``STANDARD_INPUT_BLOCK.md``). This module turns that into the
strict-SI canonical case. It decides three things the export leaves open:

1. **The duplicate end frame.** Abaqus/Explicit writes an extra field frame
   at the step's end time and samples the history there too. The last frame
   is dropped only when its time equals the previous frame's *and* every
   field and history series repeats frame N to float32 storage
   (``_END_FRAME_RTOL``), except the acceleration, which that frame writes
   afresh (``_END_FRAME_RECOMPUTED``); otherwise the export is refused.
   Frame N is kept, and a time axis with a repeated instant never reaches a
   case.
2. **A rigid body's reference node.** It carries no field output, so it is
   left out of ``Nodes``. Each ``RF<k>`` it writes to the history becomes the
   global ``reaction_force_<k>_node_<label>`` [N].
3. **A deck without ``*Part`` blocks** is one part, ``PART-1``, as the input
   reader mints it; the export's instance is that part's (``PART-1-1``).

Anything else unanticipated -- several instances or element types, more than
one integration point, a stress layout other than ``S11 S22 S33 S12`` -- is
refused by name rather than guessed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..schema import Case, ElementBlock, Material, Metadata, Nodes, Provenance, Response
from ..validation import validate
from .abaqus_run import read_abaqus_input_facts
from .lsdyna import unit_factors

__all__ = [
    "ABAQUS_NPZ_FORMAT",
    "AbaqusExport",
    "abaqus_export_to_case",
    "read_abaqus_export",
]

#: The export layout this adapter reads; ``odb_export.py`` writes the same.
ABAQUS_NPZ_FORMAT = "abaqus-npz/1"

#: `*Energy Output` terms -> canonical global names (all ten are requested).
_ENERGY = {
    "ALLKE": "kinetic_energy",
    "ALLIE": "internal_energy",
    "ETOTAL": "total_energy",
    "ALLAE": "hourglass_energy",
    "ALLPD": "plastic_dissipation",
    "ALLVD": "viscous_dissipation",
    "ALLWK": "external_work",
    "ALLFD": "frictional_dissipation",
    "ALLSE": "strain_energy",
    "ALLCD": "creep_dissipation",
}
#: Nodal fields -> (canonical name, unit-factor key).
_NODE_FIELDS = {
    "U": ("displacement", "length"),
    "V": ("velocity", "velocity"),
    "A": ("acceleration", "acceleration"),
}
#: The in-plane stress components of a 2D continuum element, in this order.
_STRESS_2D = ["S11", "S22", "S33", "S12"]
#: Fields the end-of-step frame writes afresh rather than repeating. In every
#: run of a 2026-09-24 sweep its A differed from frame N's, by up to 64 % of
#: the field's peak; why is not established. Frame N, on the output grid and
#: made the way every other frame is, is the one kept.
_END_FRAME_RECOMPUTED = frozenset({"A"})
#: Every other series must repeat frame N to float32 storage: in that sweep a
#: few percent of runs re-stored a value or two a few ulp apart, at most
#: 1.8e-7 of the field's peak. Any larger difference is a different state.
_END_FRAME_RTOL = 1e-6
_NODE_REGION = re.compile(r"^Node (?P<instance>.+)\.(?P<label>\d+)$")
_REACTION = re.compile(r"^RF(?P<k>[1-3])$")
_RELEASE_YEAR = re.compile(r"\b(\d{4})\b")


@dataclass(frozen=True)
class AbaqusExport:
    """One step of an ``abaqus-npz/1`` export, the duplicate end frame dropped.

    Parameters
    ----------
    manifest : dict
        The export's JSON manifest.
    step : str
        The step's name.
    time : ndarray, shape (T,)
        Frame step times in the deck's time unit.
    arrays : mapping of str to ndarray
        Every array of the file; ``field/…/data`` and ``history/…`` arrays hold
        ``T`` samples.
    """

    manifest: dict[str, Any]
    step: str
    time: NDArray[np.float64]
    arrays: Mapping[str, NDArray[Any]]


def read_abaqus_export(npz_path: str | Path) -> AbaqusExport:
    """Load an export, check its format, and apply the duplicate-frame rule.

    Raises
    ------
    ValueError
        On another format, or a repeated end time whose frames differ.
    NotImplementedError
        On an export with more than one step.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    manifest = json.loads(str(arrays["manifest"][()]))
    if manifest.get("format") != ABAQUS_NPZ_FORMAT:
        raise ValueError(
            f"{npz_path}: format {manifest.get('format')!r}, "
            f"expected {ABAQUS_NPZ_FORMAT}"
        )
    steps = sorted({key.split("/")[1] for key in arrays if key.startswith("step/")})
    if len(steps) != 1:
        raise NotImplementedError(f"{npz_path}: {len(steps)} steps; one is supported")
    (step,) = steps
    time = arrays[f"step/{step}/frame_times"]
    series = [
        key
        for key in arrays
        if (key.startswith(f"field/{step}/") and key.endswith("/data"))
        or key.startswith(f"history/{step}/")
    ]
    if len(time) >= 2 and time[-1] == time[-2]:
        for key in series:
            data = arrays[key]
            if len(data) != len(time):
                raise ValueError(
                    f"{npz_path}: {key} has {len(data)} samples, not {len(time)}"
                )
            recomputed = (
                key.startswith("field/") and key.split("/")[2] in _END_FRAME_RECOMPUTED
            )
            kept = data[-2].astype(np.float64)
            change = np.abs(data[-1] - kept).max(initial=0.0)
            if not recomputed and change > _END_FRAME_RTOL * np.abs(kept).max(
                initial=0.0
            ):
                raise ValueError(
                    f"{npz_path}: the repeated end time's duplicate frame differs "
                    f"in {key}; refusing to choose one"
                )
            arrays[key] = data[:-1]
        time = time[:-1]
    return AbaqusExport(manifest, step, time, arrays)


def _positions(
    labels: NDArray[Any], among: NDArray[Any], what: str
) -> NDArray[np.int64]:
    """0-based positions of ``labels`` in ``among``; raises on a missing label."""
    lookup = {int(label): i for i, label in enumerate(among)}
    try:
        flat = [lookup[int(label)] for label in labels.ravel()]
    except KeyError as exc:
        raise ValueError(f"{what} label {exc.args[0]} is not in the kept set") from None
    return np.asarray(flat, dtype=np.int64).reshape(labels.shape)


def _clocked(export: AbaqusExport, key: str) -> NDArray[np.float64]:
    """A history output's values, after checking it shares the frame clock."""
    series = export.arrays[key]
    if not np.array_equal(series[:, 0], export.time):
        raise ValueError(
            f"{key}: history clock differs from the field frame clock; "
            "a ledger sampled elsewhere cannot share the case's time axis"
        )
    return series[:, 1]


def abaqus_export_to_case(
    npz_path: str | Path,
    deck_text: str,
    *,
    source_units: str,
    dimension: int,
    case_id: str,
    dataset_id: str | None = None,
    generation_date: str = "unknown",
) -> Case:
    """Convert an ``abaqus-npz/1`` export and its deck into a canonical case.

    Parameters
    ----------
    npz_path : str or Path
        The export ``odb_export.py`` wrote beside the ``.odb``.
    deck_text : str
        The job's ``.inp``; parsed for parts and materials and stored verbatim
        as ``metadata.source_deck``.
    source_units : str
        The deck's ``"<mass>-<length>-<time>"`` convention, e.g. ``"t-mm-s"``.
    dimension : int
        2 or 3.
    case_id : str
    dataset_id : str, optional
    generation_date : str
        ISO date the run finished.

    Returns
    -------
    Case
        Validated, strict SI.
    """
    f = unit_factors(source_units)
    export = read_abaqus_export(npz_path)
    a, step = export.arrays, export.step

    instances = sorted(
        {k.split("/")[1] for k in a if re.match(r"mesh/[^/]+/elements/", k)}
    )
    if len(instances) != 1:
        raise NotImplementedError(f"multi-instance decks: {instances}")
    (inst,) = instances
    types = sorted(
        {k.split("/")[3] for k in a if k.startswith(f"mesh/{inst}/elements/")}
    )
    if len(types) != 1:
        raise NotImplementedError(f"several element types in one instance: {types}")
    (etype,) = types

    # Nodes: those with field output, which leaves out a rigid body's
    # reference node (decision 2).
    field_nodes = a[f"field/{step}/U/{inst}/node_labels"]
    mesh_nodes, xyz = a[f"mesh/{inst}/node_labels"], a[f"mesh/{inst}/node_coords"]
    _positions(field_nodes, mesh_nodes, "field node")  # every field node is meshed
    keep = np.isin(mesh_nodes, field_nodes)
    node_id, xyz = mesh_nodes[keep].astype(np.int64), xyz[keep]
    if np.any(xyz[:, dimension:] != 0):
        raise ValueError(f"a {dimension}D export has non-zero out-of-plane coordinates")
    nodes = Nodes(
        coords=xyz[:, :dimension].astype(np.float64) * f["length"], node_id=node_id
    )

    facts = read_abaqus_input_facts(deck_text, source_units=source_units)
    if len(facts.parts) != 1:
        raise NotImplementedError(
            f"{len(facts.parts)} parts in the deck; one is supported"
        )
    element_id = a[f"mesh/{inst}/elements/{etype}/labels"].astype(np.int64)
    connectivity = _positions(
        a[f"mesh/{inst}/elements/{etype}/connectivity"], node_id, "node"
    )
    part_id = np.full(element_id.shape, facts.parts[0].part_id, dtype=np.int64)
    elements = {"solid": ElementBlock(connectivity, element_id, part_id)}

    node: dict[str, NDArray[np.float32]] = {}
    for name, (canonical, unit) in _NODE_FIELDS.items():
        key = f"field/{step}/{name}/{inst}"
        if f"{key}/data" in a:
            order = _positions(node_id, a[f"{key}/node_labels"], f"{name} node")
            node[canonical] = (a[f"{key}/data"][:, order, :] * f[unit]).astype(
                np.float32
            )

    element: dict[str, NDArray[np.float32]] = {}
    fields = export.manifest.get("fields", {})
    for name in ("S", "PEEQ"):
        key = f"field/{step}/{name}/{inst}"
        if f"{key}/data" not in a:
            continue
        if np.any(a[f"{key}/integration_points"] != 1):
            raise NotImplementedError(f"{name}: more than one integration point")
        order = _positions(element_id, a[f"{key}/element_labels"], f"{name} element")
        data = a[f"{key}/data"][:, order, :]
        if name == "PEEQ":
            element["effective_plastic_strain"] = data[..., 0].astype(np.float32)
            continue
        labels = fields.get(key, {}).get("component_labels") or _STRESS_2D
        if labels != _STRESS_2D or data.shape[-1] != 4:
            raise NotImplementedError(f"stress components {labels}")
        zero = np.zeros(data.shape[:-1], dtype=data.dtype)
        voigt = np.stack(
            [data[..., 0], data[..., 1], data[..., 2], data[..., 3], zero, zero], -1
        )
        element["stress"] = (voigt * f["stress"]).astype(np.float32)

    globals_: dict[str, NDArray[np.float32]] = {}
    for key in a:
        parts = key.split("/")
        if parts[0] != "history" or len(parts) != 4:
            continue
        region, output = parts[2], parts[3]
        if region.startswith("Assembly") and output in _ENERGY:
            values = _clocked(export, key) * f["energy"]
            globals_[_ENERGY[output]] = values.astype(np.float32)
        elif (at := _NODE_REGION.match(region)) and (rf := _REACTION.match(output)):
            name = f"reaction_force_{rf['k']}_node_{at['label']}"
            globals_[name] = (_clocked(export, key) * f["force"]).astype(np.float32)

    materials = [
        Material(
            m.material_id,
            "ABAQUS",
            {
                k: v
                for k, v in (
                    ("density", m.density),
                    ("youngs_modulus", m.youngs_modulus),
                    ("poisson_ratio", m.poisson_ratio),
                    (
                        "yield_table",
                        [list(t) for t in m.yield_table] if m.yield_table else None,
                    ),
                )
                if v is not None
            },
            m.canonical_model,
        )
        for m in facts.materials
    ]
    year = _RELEASE_YEAR.search(str(export.manifest.get("abaqus_release", "")))
    case = Case(
        metadata=Metadata(
            case_id=case_id,
            dimension=dimension,
            provenance=Provenance(
                "Abaqus", year.group(1) if year else "unknown", generation_date
            ),
            source_units=source_units,
            source_deck=deck_text,
            dataset_id=dataset_id,
        ),
        nodes=nodes,
        elements=elements,
        materials=materials,
        response=Response(
            time=export.time.astype(np.float64) * f["time"],
            node=node,
            element={"solid": element} if element else {},
            globals_=globals_,
        ),
    )
    validate(case)
    return case
