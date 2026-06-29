"""LS-DYNA d3plot -> canonical :class:`~structbench.core.schema.Case` adapter.

This is the canonical ingestion path committed in ADR-0016: an LS-DYNA d3plot
binary (read via ``lasso.dyna.D3plot``) plus its paired ``.k`` deck become one
canonical case. The adapter extracts *everything* the d3plot contains -- all
response fields, all frames, all element types -- and performs no feature
engineering at ingestion (ADR-0016 section 4). Unit conversion to strict SI
(ADR-0012) happens at the write boundary, driven by :func:`unit_factors`.

This first slice targets the field groups the schema currently models
(metadata, nodes, elements, materials, response). Deck cards beyond the
material definitions are ignored on first read (ADR-0016 section 3); the
adapter grows card coverage as new datasets surface needs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from lasso.dyna import D3plot

_LOG = logging.getLogger(__name__)

#: Base-unit -> SI factors, keyed by the lowercased unit token.
_MASS = {"kg": 1.0, "g": 1e-3, "mg": 1e-6, "t": 1e3, "tonne": 1e3}
_LENGTH = {"m": 1.0, "dm": 1e-1, "cm": 1e-2, "mm": 1e-3, "um": 1e-6}
_TIME = {"s": 1.0, "ms": 1e-3, "us": 1e-6}


def unit_factors(source_units: str) -> dict[str, float]:
    """Return per-quantity multipliers converting ``source_units`` to SI.

    The source convention is given as a ``"mass-length-time"`` token string,
    e.g. ``"g-mm-ms"`` for the LS-DYNA gram/millimetre/millisecond system.
    Every physical quantity the adapter writes is derived from the three base
    factors, so the same helper serves any consistent unit system.

    Parameters
    ----------
    source_units:
        ``"<mass>-<length>-<time>"``; tokens drawn from kg/g/mg/t, m/dm/cm/mm/um,
        and s/ms/us (case-insensitive).

    Returns
    -------
    dict[str, float]
        Multipliers keyed by quantity (``length``, ``velocity``,
        ``acceleration``, ``mass``, ``density``, ``force``, ``stress``,
        ``pressure``, ``energy``, ``time``) plus ``strain`` (always ``1.0``,
        dimensionless). Multiply a source-unit value by the factor to get SI.

    Raises
    ------
    ValueError
        If the string is not three ``-``-separated tokens, or a token is not a
        recognised unit.
    """
    parts = source_units.lower().split("-")
    if len(parts) != 3:
        raise ValueError(
            f"source_units must be 'mass-length-time', got {source_units!r}"
        )
    m_tok, l_tok, t_tok = parts
    try:
        m, length, t = _MASS[m_tok], _LENGTH[l_tok], _TIME[t_tok]
    except KeyError as exc:
        raise ValueError(
            f"unrecognised unit token {exc.args[0]!r} in {source_units!r}"
        ) from exc
    return {
        "length": length,
        "velocity": length / t,
        "acceleration": length / t**2,
        "mass": m,
        "density": m / length**3,
        "force": m * length / t**2,
        "stress": m / (length * t**2),
        "pressure": m / (length * t**2),
        "energy": m * length**2 / t**2,
        "time": t,
        "strain": 1.0,
        "strain_rate": 1.0 / t,
        "count": 1.0,
    }


# --- deck parsing -----------------------------------------------------------

#: Source-model names with a clean canonical-enum mapping (ADR-0012). Cards
#: not listed here get ``canonical_model=None`` and keep their verbatim
#: ``source_params`` only.
_CANONICAL_MAT = {
    "MAT_ELASTIC_PLASTIC_HYDRO": "elastic_plastic_hydro",
    "MAT_ELASTIC": "linear_elastic",
    "MAT_RIGID": "rigid",
    "MAT_NULL": "null",
}

_FIELD_WIDTH = 10


def _numeric_fields(line: str, width: int = _FIELD_WIDTH) -> list[float] | None:
    """Parse a fixed-width LS-DYNA line into floats.

    Returns ``None`` if any non-empty column is not a number -- this is how
    title lines (e.g. a ``*PART`` name) are told apart from data lines.
    """
    out: list[float] = []
    for i in range(0, len(line), width):
        chunk = line[i : i + width].strip()
        if not chunk:
            continue
        try:
            out.append(float(chunk))
        except ValueError:
            return None
    return out


def _card_blocks(deck_text: str) -> Iterator[tuple[str, list[list[float]]]]:
    """Yield ``(keyword, data_rows)`` per ``*``-card, in deck order.

    ``keyword`` has the leading ``*`` stripped; ``data_rows`` are the numeric
    data lines (``$`` comments and non-numeric title lines excluded).
    """
    keyword: str | None = None
    rows: list[list[float]] = []
    for raw in deck_text.splitlines():
        if raw.startswith("*"):
            if keyword is not None:
                yield keyword, rows
            keyword = raw.strip()[1:]
            rows = []
        elif raw.startswith("$") or keyword is None:
            continue
        else:
            fields = _numeric_fields(raw)
            if fields:
                rows.append(fields)
    if keyword is not None:
        yield keyword, rows


def parse_deck_materials(deck_text: str) -> list[Material]:
    """Parse the material definitions from an LS-DYNA ``.k`` deck.

    Reads every ``*MAT_*`` card and links each material's solver sub-models
    (``*EOS_*``, ``*HOURGLASS``) by following the ``*PART`` card that
    references it, nesting them inside ``source_params`` (ADR-0013). Field
    values are captured verbatim as a list of numeric rows under
    ``source_params["data"]``; no per-card field naming is attempted on this
    first slice. Cards other than materials and their sub-models are ignored
    (ADR-0016 section 3).

    Parameters
    ----------
    deck_text:
        Full text of the ``.k`` deck.

    Returns
    -------
    list[Material]
        One :class:`~structbench.core.schema.Material` per ``*MAT_*`` card, in
        deck order. Empty if the deck defines no materials.
    """
    mats: dict[int, dict[str, Any]] = {}
    eoses: dict[int, dict[str, Any]] = {}
    hourglasses: dict[int, dict[str, Any]] = {}
    parts: list[dict[str, int]] = []

    for keyword, rows in _card_blocks(deck_text):
        if not rows:
            continue
        head = rows[0]
        if keyword.startswith("MAT_"):
            mats[int(head[0])] = {"source_model": keyword, "data": rows}
        elif keyword.startswith("EOS_"):
            eoses[int(head[0])] = {"source_model": keyword, "data": rows}
        elif keyword.startswith("HOURGLASS"):
            hourglasses[int(head[0])] = {"source_model": keyword, "data": rows}
        elif keyword.split("_")[0] == "PART":
            parts.append(
                {
                    "pid": int(head[0]),
                    "mid": int(head[2]) if len(head) > 2 else 0,
                    "eosid": int(head[3]) if len(head) > 3 else 0,
                    "hgid": int(head[4]) if len(head) > 4 else 0,
                }
            )

    materials: list[Material] = []
    for mid, mat in mats.items():
        source_params: dict[str, Any] = {"data": mat["data"]}
        part = next((p for p in parts if p["mid"] == mid), None)
        if part is not None:
            if part["eosid"] in eoses:
                source_params["eos"] = eoses[part["eosid"]]
            if part["hgid"] in hourglasses:
                source_params["hourglass"] = hourglasses[part["hgid"]]
        materials.append(
            Material(
                material_id=mid,
                source_model=mat["source_model"],
                source_params=source_params,
                canonical_model=_CANONICAL_MAT.get(mat["source_model"]),
            )
        )
    return materials


# --- d3plot extraction ------------------------------------------------------

#: Standard finite-element block types in a d3plot, keyed by the prefix lasso
#: uses (``element_<type>_node_indexes`` etc.). SPH is handled separately.
_FE_TYPES = ("shell", "solid", "beam", "tshell")

#: Per-node response arrays: lasso name -> (schema field, unit-factor key).
#: These are vector fields reduced to the case dimension.
_NODE_FIELDS = {
    "node_displacement": ("displacement", "length"),
    "node_velocity": ("velocity", "velocity"),
    "node_acceleration": ("acceleration", "acceleration"),
}

#: Per-SPH-particle response arrays. Tensor fields (stress/strain/strainrate)
#: keep their full 6-component Voigt layout verbatim (ADR-0016 section 4).
_SPH_FIELDS = {
    "sph_stress": ("stress", "stress"),
    "sph_strain": ("strain", "strain"),
    "sph_strainrate": ("strain_rate", "strain_rate"),
    "sph_effective_plastic_strain": ("effective_plastic_strain", "strain"),
    "sph_pressure": ("pressure", "pressure"),
    "sph_density": ("density", "density"),
    "sph_internal_energy": ("internal_energy", "energy"),
    "sph_mass": ("mass", "mass"),
    "sph_radius": ("radius", "length"),
    "sph_n_neighbors": ("n_neighbors", "count"),
    "sph_deletion": ("deletion", "count"),
}

#: Per-element response arrays for finite-element blocks, by name suffix.
#: Restricted to quantities with an unambiguous SI factor; section-force
#: resultants (normal/shear/bending) are left out until their per-length shell
#: units are pinned down, and logged as skipped rather than guessed.
_FE_FIELDS = {
    "stress": ("stress", "stress"),
    "effective_plastic_strain": ("effective_plastic_strain", "strain"),
    "internal_energy": ("internal_energy", "energy"),
    "thickness": ("thickness", "length"),
    "is_alive": ("is_alive", "count"),
}

#: Per-frame global scalars. Non-scalar globals (global_velocity, part_*,
#: rigid_wall_force) have no home in the current scalar-only response/global
#: group and are deferred.
_GLOBAL_FIELDS = {
    "global_kinetic_energy": ("kinetic_energy", "energy"),
    "global_internal_energy": ("internal_energy", "energy"),
    "global_total_energy": ("total_energy", "energy"),
}


def _part_id_lookup(arrays: dict[str, Any], part_indexes: Any) -> np.ndarray:
    """Map 0-based part indexes to LS-DYNA part ids via ``part_titles_ids``."""
    idx = np.asarray(part_indexes, dtype=np.int64)
    part_ids = np.asarray(arrays.get("part_titles_ids", []), dtype=np.int64)
    return part_ids[idx] if part_ids.size else idx


def extract_geometry(
    arrays: dict[str, Any], dim: int, factors: dict[str, float]
) -> tuple[Nodes, dict[str, ElementBlock]]:
    """Build the geometry/topology of a case from a d3plot ``arrays`` mapping.

    Parameters
    ----------
    arrays:
        The ``D3plot.arrays`` mapping (or a compatible dict).
    dim:
        Case dimension, 2 or 3. For ``dim < 3`` the trailing coordinate
        columns are dropped; a :class:`ValueError` is raised if they are not
        ~0, guarding against a 3D model being mislabelled 2D.
    factors:
        Unit multipliers from :func:`unit_factors`; coordinates are scaled by
        ``factors["length"]`` to SI.

    Returns
    -------
    tuple[Nodes, dict[str, ElementBlock]]
        All nodes (SI coordinates) and one :class:`ElementBlock` per non-empty
        element type present (``sph`` plus any of ``shell``/``solid``/``beam``/
        ``tshell``). Connectivity is 0-indexed into the node array.
    """
    coords3 = np.asarray(arrays["node_coordinates"], dtype=np.float64)
    dropped = coords3[:, dim:]
    if dropped.size:
        # The model must be planar in the kept axes: the dropped axis may be
        # offset (constant) but must not vary across the bulk of nodes. A few
        # off-plane nodes are tolerated as visualisation scaffolding (e.g. the
        # null shell LS-PrePost adds to SPH models); a genuinely 3D model
        # mislabelled 2D has most nodes off-plane and is rejected.
        tol = 1e-6 * (float(np.abs(coords3[:, :dim]).max()) + 1.0)
        off_plane = np.abs(dropped - np.median(dropped, axis=0)).max(axis=1) > tol
        frac = float(off_plane.mean())
        if frac > 0.01:
            raise ValueError(
                f"dimension={dim} requested but {frac:.1%} of nodes lie off the "
                f"{dim}D plane (coordinate column(s) >= {dim} vary); the model "
                f"may be {coords3.shape[1]}D"
            )
        if off_plane.any():
            _LOG.warning(
                "lsdyna: %d node(s) lie off the %dD plane (likely visualisation "
                "scaffolding); their out-of-plane coordinate is dropped",
                int(off_plane.sum()),
                dim,
            )
    nodes = Nodes(
        coords=coords3[:, :dim] * factors["length"],
        node_id=np.asarray(arrays["node_ids"], dtype=np.int64),
    )

    elements: dict[str, ElementBlock] = {}
    sph_idx = np.asarray(arrays.get("sph_node_indexes", []), dtype=np.int64)
    if sph_idx.size:
        mat_idx = arrays.get(
            "sph_node_material_index", np.zeros(sph_idx.size, dtype=np.int64)
        )
        elements["sph"] = ElementBlock(
            connectivity=sph_idx.reshape(-1, 1),
            element_id=nodes.node_id[sph_idx],
            part_id=_part_id_lookup(arrays, mat_idx),
        )
    for etype in _FE_TYPES:
        conn = np.asarray(
            arrays.get(f"element_{etype}_node_indexes", np.empty((0, 0))),
            dtype=np.int64,
        )
        if conn.shape[0] == 0:
            continue
        elements[etype] = ElementBlock(
            connectivity=conn,
            element_id=np.asarray(arrays[f"element_{etype}_ids"], dtype=np.int64),
            part_id=_part_id_lookup(arrays, arrays[f"element_{etype}_part_indexes"]),
        )
    return nodes, elements


def _to_si_f32(arr: Any, factor: float) -> np.ndarray:
    """Convert in float64, cast to the schema's float32 response dtype."""
    return (np.asarray(arr, dtype=np.float64) * factor).astype(np.float32)


def extract_response(
    arrays: dict[str, Any], dim: int, factors: dict[str, float]
) -> Response:
    """Build the response of a case from a d3plot ``arrays`` mapping.

    Extracts every mapped response field (per-node, per-SPH-particle,
    per-finite-element, and per-frame global scalars), converts to SI, and
    stores it as float32 on the single global time axis. Per-node vector
    fields are reduced to ``dim`` components; SPH tensor fields keep their full
    6-component Voigt layout. Response arrays present but not mapped (e.g.
    shell section-force resultants) are logged, not silently dropped.

    Parameters
    ----------
    arrays:
        The ``D3plot.arrays`` mapping (or a compatible dict).
    dim:
        Case dimension, used to slice per-node vector fields.
    factors:
        Unit multipliers from :func:`unit_factors`.

    Returns
    -------
    Response
        The populated response with its SI time axis.
    """
    time = np.asarray(arrays["timesteps"], dtype=np.float64) * factors["time"]
    n_frames = time.shape[0]
    mapped: set[str] = {"timesteps", *_NODE_FIELDS, *_SPH_FIELDS, *_GLOBAL_FIELDS}

    coords = np.asarray(arrays["node_coordinates"], dtype=np.float64)
    node: dict[str, np.ndarray] = {}
    for key, (name, fkey) in _NODE_FIELDS.items():
        if key not in arrays:
            continue
        field = np.asarray(arrays[key], dtype=np.float64)
        if name == "displacement":
            # lasso's node_displacement holds the deformed position per state
            # (frame 0 == initial coords); convert to displacement-from-initial
            # as the schema's response/node/displacement requires (ADR-0012).
            field = field - coords[None, :, :]
        node[name] = _to_si_f32(field[:, :, :dim], factors[fkey])

    element: dict[str, dict[str, np.ndarray]] = {}
    sph = {
        name: _to_si_f32(arrays[key], factors[fkey])
        for key, (name, fkey) in _SPH_FIELDS.items()
        if key in arrays
    }
    if sph:
        element["sph"] = sph
    for etype in _FE_TYPES:
        block: dict[str, np.ndarray] = {}
        for suffix, (name, fkey) in _FE_FIELDS.items():
            key = f"element_{etype}_{suffix}"
            mapped.add(key)
            if key in arrays and np.asarray(arrays[key]).shape[0] == n_frames:
                block[name] = _to_si_f32(arrays[key], factors[fkey])
        if block:
            element[etype] = block

    globals_ = {
        name: _to_si_f32(arrays[key], factors[fkey])
        for key, (name, fkey) in _GLOBAL_FIELDS.items()
        if key in arrays
    }

    _log_skipped_response_arrays(arrays, mapped)
    return Response(time=time, node=node, element=element, globals_=globals_)


def _log_skipped_response_arrays(arrays: dict[str, Any], mapped: set[str]) -> None:
    """Log time-varying response arrays present but not extracted."""
    skipped = sorted(
        k
        for k in arrays
        if k not in mapped
        and (k.startswith(("sph_", "global_")) or "_" in k)
        and getattr(np.asarray(arrays[k]), "ndim", 0) >= 1
        and k
        not in {
            "node_coordinates",
            "node_ids",
            "sph_node_indexes",
            "sph_node_material_index",
            "part_titles_ids",
            "part_titles",
        }
        and not k.endswith(("_node_indexes", "_part_indexes", "_ids"))
    )
    if skipped:
        _LOG.info(
            "lsdyna: response arrays present but not extracted: %s", ", ".join(skipped)
        )


# --- assembly / entry points ------------------------------------------------


def build_case(
    arrays: dict[str, Any],
    deck_text: str,
    *,
    source_units: str,
    dimension: int,
    case_id: str,
    dataset_id: str | None = None,
    provenance: Provenance | None = None,
) -> Case:
    """Assemble a validated :class:`Case` from a d3plot ``arrays`` mapping.

    This is the pure core of the adapter: it takes the d3plot arrays (already
    loaded) and the deck text, and composes geometry, response, and materials
    into a canonical, strict-SI case. Separating it from file reading keeps the
    assembly logic testable without a d3plot binary.

    Parameters
    ----------
    arrays:
        The ``D3plot.arrays`` mapping (or a compatible dict).
    deck_text:
        Full ``.k`` deck text; parsed for materials and stored verbatim as
        ``metadata.source_deck``.
    source_units:
        Source unit convention, e.g. ``"g-mm-ms"`` (see :func:`unit_factors`).
    dimension:
        Case dimension, 2 or 3.
    case_id:
        Identifier for the case.
    dataset_id:
        Optional dataset the case belongs to.
    provenance:
        Optional provenance; defaults to LS-DYNA with unknown version/date.

    Returns
    -------
    Case
        A validated case (raises :class:`~structbench.core.exceptions.SchemaError`
        if the assembled case is invalid).
    """
    factors = unit_factors(source_units)
    nodes, elements = extract_geometry(arrays, dimension, factors)
    response = extract_response(arrays, dimension, factors)
    materials = parse_deck_materials(deck_text)
    metadata = Metadata(
        case_id=case_id,
        dimension=dimension,
        source_units=source_units,
        source_deck=deck_text or None,
        dataset_id=dataset_id,
        provenance=provenance or Provenance("LS-DYNA", "unknown", "unknown"),
    )
    case = Case(
        metadata=metadata,
        nodes=nodes,
        elements=elements,
        materials=materials,
        response=response,
    )
    validate(case)
    return case


def read_d3plot(path: str | Path) -> D3plot:
    """Open an LS-DYNA d3plot (and its state continuation files) with lasso.

    ``lasso.dyna`` is imported lazily so that importing this module does not
    require the dependency unless the adapter is actually used.

    Parameters
    ----------
    path:
        Path to the base ``d3plot`` file; lasso discovers ``d3plot01`` ... too.

    Returns
    -------
    lasso.dyna.D3plot
        The loaded d3plot, exposing the ``arrays`` mapping.
    """
    from lasso.dyna import D3plot

    return D3plot(str(path))


def lsdyna_to_case(
    d3plot_path: str | Path,
    deck_path: str | Path,
    *,
    source_units: str,
    dimension: int,
    case_id: str | None = None,
    dataset_id: str | None = None,
) -> Case:
    """Convert an LS-DYNA d3plot + deck into a canonical :class:`Case`.

    The top-level ingestion entry point (ADR-0016). Reads the d3plot binary
    and the paired deck, then delegates assembly to :func:`build_case`.

    Parameters
    ----------
    d3plot_path:
        Path to the base ``d3plot`` file.
    deck_path:
        Path to the paired ``.k`` deck.
    source_units:
        Source unit convention, e.g. ``"g-mm-ms"``. Supplied by the per-dataset
        glue when the deck lacks a ``*CONTROL_UNITS`` card (ADR-0016 section 5).
    dimension:
        Case dimension, 2 or 3.
    case_id:
        Identifier; defaults to the d3plot's parent directory name.
    dataset_id:
        Optional dataset the case belongs to.

    Returns
    -------
    Case
        The validated, strict-SI case.
    """
    d3plot_path = Path(d3plot_path)
    deck_text = Path(deck_path).read_text(encoding="utf-8", errors="replace")
    d3 = read_d3plot(d3plot_path)
    generation_date = date.fromtimestamp(d3plot_path.stat().st_mtime).isoformat()
    return build_case(
        d3.arrays,
        deck_text,
        source_units=source_units,
        dimension=dimension,
        case_id=case_id or d3plot_path.parent.name,
        dataset_id=dataset_id,
        provenance=Provenance("LS-DYNA", "unknown", generation_date),
    )
