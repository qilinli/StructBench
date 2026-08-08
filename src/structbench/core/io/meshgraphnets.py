"""Adapter: MeshGraphNets ``deforming_plate`` tfrecord -> canonical Case (ADR-0042).

TensorFlow is imported lazily inside :func:`read_deforming_plate` only, so
``import structbench`` never requires it (mirrors ``lsdyna.read_d3plot``).
"""

from __future__ import annotations

from dataclasses import dataclass

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
