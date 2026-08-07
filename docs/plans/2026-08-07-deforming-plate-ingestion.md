# Deforming-Plate Ingestion Implementation Plan (schema 0.2.0 + tfrecord converter)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the additive schema 0.2.0 per-node fields (ADR-0042) and an offline MeshGraphNets `deforming_plate` → canonical HDF5 converter, so a `canonical/deforming_plate/*.h5` archive of valid cases can be produced. This is checkpoint ①-a of v0.3 (ADR-0041); it does **not** include the benchmark module, `datasets/` mesh-edge support, or `models/mgn` (later plans, gated on the benchmark-protocol ADR).

**Architecture:** Two parts. **Part A** extends the pure `numpy`/`h5py` substrate (`core/`) with schema 0.2.0: two optional per-node fields on `Nodes` (`node_type`, `reference_coords`) and a relaxed `response.node` trailing-dim rule so per-node scalars validate — with a version bump and full backward-compatibility for 0.1.0 files. **Part B** adds a `core/io/meshgraphnets.py` adapter shaped exactly like `lsdyna.py` (a pure `build_deforming_plate_case` core + a lazy-`tensorflow` reader), driven by a non-importable `data_generation/meshgraphnets/deforming_plate/convert.py` that downloads from source and converts locally.

**Tech Stack:** Python 3.12+, existing runtime deps only (`numpy`/`h5py`). TensorFlow appears **only** inside the converter's lazy read path and its throwaway environment — never a runtime dependency of `structbench` (ADR-0042 §2a).

**Plan 1 of the v0.3 build order** (ADR-0041 ①→②→③). Follow-on plans: DeformingPlate benchmark module + `datasets/` mesh edges; `models/mgn`; then Transolver/GeoFLARE + the per-method comparison registry.

## Global Constraints

- Python floor **3.12**; ruff line length **88** + `ruff format`; mypy `disallow_untyped_defs = true`; NumPy-style docstrings on every public API; `_`-prefix symbols are private across module boundaries.
- **No new runtime dependencies.** `tensorflow` is imported lazily *inside* the reader function only (mirror `lsdyna.read_d3plot`'s inline `from lasso.dyna import D3plot`), so `import structbench` never requires it.
- **Schema change is additive and backward-compatible** (ADR-0042): 0.1.0 files must still read unchanged; new `Nodes` fields default `None`; `SCHEMA_VERSION` → `"0.2.0"`.
- **`units_convention` is always `"SI"`** on canonical output; `world_pos`/`stress` source units are **undocumented** and must be *measured* at conversion (Task 8), recorded in `metadata.source_units`. Reuse `lsdyna.unit_factors` (a `"mass-length-time"` token → SI multipliers); `"kg-m-s"` is the identity if the data proves to be SI.
- **Data caution (CORRECTIONS.md):** never run recursive scans/globs over `..\data`; the converter reads only the explicit `deforming_plate/{meta.json,train,valid,test}.tfrecord` files it downloaded.
- **Verified dataset facts** (`meta.json`, ADR-0042): fields `cells [1,-1,4]` int32 static (linear tetrahedra), `node_type [1,-1,1]` int32 static (NORMAL=0, OBSTACLE=1 actuator, HANDLE=3 fixed), `mesh_pos [1,-1,3]` float32 static, `world_pos [400,-1,3]` float32 dynamic, `stress [400,-1,1]` float32 dynamic (per-node von Mises); `trajectory_length=400`; quasi-static (`dt=0`); splits 1000/100/100; ~1271 nodes avg; COMSOL. No dataset redistribution licence → download-and-convert, no rehost.
- Tests: pytest, **synthetic-only** (build `arrays`-shaped dicts, never a real tfrecord — mirror `tests/core/test_lsdyna.py::_synthetic_arrays`); deterministic (`np.random.default_rng(seed)`); TF-touching code paths are `pytest.importorskip("tensorflow")`-gated. Run via the `structbench` conda env interpreter.
- Commits: Conventional Commits + the repo `Co-Authored-By:` trailer. Branch: **`feat/deforming-plate-ingestion`** off `main`. Never commit to `main`; merge/push are confirm-gated human calls (ADR-0023).

## File Structure

```
src/structbench/core/
  schema.py            # MODIFY: Nodes +node_type +reference_coords; SCHEMA_VERSION -> "0.2.0"
  validation.py        # MODIFY: _validate_nodes (new fields); _validate_response (relax trailing dim)
  io/__init__.py       # MODIFY: _write_nodes/_read_nodes persist the new fields; response.node any width
  io/meshgraphnets.py  # CREATE: build_deforming_plate_case (pure) + read_deforming_plate (lazy tf) + parse_meta
  __init__.py          # MODIFY: re-export deforming_plate_to_cases / build_deforming_plate_case
data_generation/meshgraphnets/deforming_plate/
  convert.py           # CREATE: download-and-convert batch driver (mirrors lsdyna Taylor convert.py)
  README.md            # CREATE: how to run (throwaway TF env), source URL, no-rehost note
tests/core/
  test_io_roundtrip.py # MODIFY: per-node field round-trip + backward-compat + scalar response.node
  test_meshgraphnets.py# CREATE: build_deforming_plate_case + parse_meta (synthetic arrays)
```

---

### Task 1: `Nodes` gains `node_type` and `reference_coords` (schema + validator)

**Files:**
- Modify: `src/structbench/core/schema.py:55-60` (the `Nodes` dataclass)
- Modify: `src/structbench/core/validation.py:56-68` (`_validate_nodes`)
- Test: `tests/core/test_io_roundtrip.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Nodes(coords, node_id, node_type=None, reference_coords=None)` — `node_type: NDArray[np.int64] | None` shape `(n_nodes,)`; `reference_coords: NDArray[np.float64] | None` shape `(n_nodes, dim)`. `validate` raises `SchemaError` on shape mismatch when either is present.

- [ ] **Step 1: Write the failing tests** (add to `tests/core/test_io_roundtrip.py`)

```python
def test_validate_accepts_optional_node_fields():
    from structbench.core import Case, Metadata, Nodes, ElementBlock, Material, validate
    case = _shell_case()  # existing 2D 4-node helper
    case.nodes.node_type = np.array([0, 0, 3, 3], dtype=np.int64)
    case.nodes.reference_coords = case.nodes.coords.copy()
    validate(case)  # must not raise


def test_validate_rejects_bad_node_type_shape():
    from structbench.core import validate
    from structbench.core.exceptions import SchemaError
    case = _shell_case()
    case.nodes.node_type = np.array([0, 0, 3], dtype=np.int64)  # 3 != 4 nodes
    with pytest.raises(SchemaError, match="node_type"):
        validate(case)


def test_validate_rejects_bad_reference_coords_shape():
    from structbench.core import validate
    from structbench.core.exceptions import SchemaError
    case = _shell_case()
    case.nodes.reference_coords = np.zeros((4, 3), dtype=np.float64)  # dim=2, not 3
    with pytest.raises(SchemaError, match="reference_coords"):
        validate(case)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/test_io_roundtrip.py -k "node_type or reference_coords or optional_node_fields" -v`
Expected: FAIL — `Nodes.__init__` has no `node_type`/`reference_coords` (TypeError), then validator gaps.

- [ ] **Step 3: Add the dataclass fields**

In `src/structbench/core/schema.py`, extend `Nodes`:

```python
@dataclass
class Nodes:
    coords: NDArray[np.float64]                          # (n_nodes, dim)
    node_id: NDArray[np.int64]                           # (n_nodes,)
    node_type: NDArray[np.int64] | None = None           # (n_nodes,) — schema 0.2.0
    reference_coords: NDArray[np.float64] | None = None  # (n_nodes, dim) — schema 0.2.0
```

- [ ] **Step 4: Extend the validator**

In `src/structbench/core/validation.py`, inside `_validate_nodes` (after the existing `node_id` check, using the `n_nodes` and `dim` already in scope):

```python
    if nodes.node_type is not None and nodes.node_type.shape != (n_nodes,):
        raise SchemaError(
            f"nodes.node_type shape {nodes.node_type.shape} != ({n_nodes},)"
        )
    if nodes.reference_coords is not None and nodes.reference_coords.shape != nodes.coords.shape:
        raise SchemaError(
            "nodes.reference_coords shape "
            f"{nodes.reference_coords.shape} != coords {nodes.coords.shape}"
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/core/test_io_roundtrip.py -k "node_type or reference_coords or optional_node_fields" -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/structbench/core/schema.py src/structbench/core/validation.py tests/core/test_io_roundtrip.py
git commit -m "feat(schema): add optional per-node node_type and reference_coords (0.2.0)"
```

---

### Task 2: Relax `response.node` to allow per-node scalar/tensor fields

**Files:**
- Modify: `src/structbench/core/validation.py:113-150` (`_validate_response`)
- Test: `tests/core/test_io_roundtrip.py`

**Interfaces:**
- Consumes: Task 1's schema.
- Produces: `_validate_response` accepts any `response.node[field]` of shape `(n_frames, n_nodes, k)` with `k >= 1`; `displacement` remains required and must be `dim`-wide.

- [ ] **Step 1: Write the failing tests**

```python
def test_validate_accepts_per_node_scalar_field():
    from structbench.core import validate
    case = _shell_case()  # 2D, 4 nodes, 3 frames
    case.response.node["von_mises_stress"] = np.zeros((3, 4, 1), dtype=np.float32)
    validate(case)  # (T, N, 1) must be allowed now


def test_validate_still_requires_displacement_dim_wide():
    from structbench.core import validate
    from structbench.core.exceptions import SchemaError
    case = _shell_case()
    case.response.node["displacement"] = np.zeros((3, 4, 1), dtype=np.float32)  # not dim=2
    with pytest.raises(SchemaError, match="displacement"):
        validate(case)


def test_validate_rejects_per_node_field_wrong_node_count():
    from structbench.core import validate
    from structbench.core.exceptions import SchemaError
    case = _shell_case()
    case.response.node["von_mises_stress"] = np.zeros((3, 5, 1), dtype=np.float32)  # 5 != 4
    with pytest.raises(SchemaError, match="von_mises_stress"):
        validate(case)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/test_io_roundtrip.py -k "per_node_scalar or displacement_dim_wide or wrong_node_count" -v`
Expected: FAIL — current rule forces every node field to `(n_frames, n_nodes, dim)`.

- [ ] **Step 3: Rewrite the node-field loop in `_validate_response`**

Replace the existing per-field shape check so displacement is special-cased and other fields allow any `k >= 1`:

```python
    if "displacement" not in response.node:
        raise SchemaError("response.node must contain 'displacement'")
    for name, arr in response.node.items():
        if arr.ndim != 3 or arr.shape[0] != n_frames or arr.shape[1] != n_nodes:
            raise SchemaError(
                f"response.node[{name!r}] shape {arr.shape} != "
                f"({n_frames}, {n_nodes}, k>=1)"
            )
        if arr.shape[2] < 1:
            raise SchemaError(f"response.node[{name!r}] trailing dim must be >= 1")
        if name == "displacement" and arr.shape[2] != dim:
            raise SchemaError(
                f"response.node['displacement'] must be dim-wide: "
                f"{arr.shape[2]} != {dim}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/core/test_io_roundtrip.py -k "per_node_scalar or displacement_dim_wide or wrong_node_count" -v`
Expected: PASS.

- [ ] **Step 5: Run the full validation-rejection suite (no regressions)**

Run: `python -m pytest tests/core/test_io_roundtrip.py -v`
Expected: PASS — the pre-existing `test_validate_rejects_simulated_case_without_displacement` and `test_validate_rejects_zero_frame_response` still pass.

- [ ] **Step 6: Commit**

```bash
git add src/structbench/core/validation.py tests/core/test_io_roundtrip.py
git commit -m "feat(schema): allow per-node scalar/tensor response.node fields (0.2.0)"
```

---

### Task 3: Persist and read the new fields; bump `SCHEMA_VERSION`

**Files:**
- Modify: `src/structbench/core/io/__init__.py` (`_write_nodes`/`_read_nodes` near lines 160-163; response.node writer/reader already width-agnostic — confirm)
- Modify: `src/structbench/core/schema.py:25` (`SCHEMA_VERSION = "0.2.0"`)
- Test: `tests/core/test_io_roundtrip.py`

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: `write_case`/`read_case` round-trip `node_type`, `reference_coords`, and per-node scalar response fields; `SCHEMA_VERSION == "0.2.0"`; a case with `node_type=None` writes no such dataset and reads back `None` (backward-compat).

- [ ] **Step 1: Write the failing tests**

```python
def test_roundtrip_per_node_fields(tmp_path):
    from structbench.core import read_case, write_case
    case = _shell_case()
    case.nodes.node_type = np.array([0, 1, 3, 3], dtype=np.int64)
    case.nodes.reference_coords = case.nodes.coords.copy()
    case.response.node["von_mises_stress"] = np.arange(12, dtype=np.float32).reshape(3, 4, 1)
    path = tmp_path / "c.h5"
    write_case(case, path)
    back = read_case(path)
    np.testing.assert_array_equal(back.nodes.node_type, case.nodes.node_type)
    assert back.nodes.node_type.dtype == np.int64
    np.testing.assert_array_equal(back.nodes.reference_coords, case.nodes.reference_coords)
    np.testing.assert_array_equal(
        back.response.node["von_mises_stress"], case.response.node["von_mises_stress"]
    )
    assert back.metadata.schema_version == "0.2.0"


def test_roundtrip_without_optional_node_fields_is_backward_compatible(tmp_path):
    from structbench.core import read_case, write_case
    case = _shell_case()  # no node_type / reference_coords set
    path = tmp_path / "c.h5"
    write_case(case, path)
    back = read_case(path)
    assert back.nodes.node_type is None
    assert back.nodes.reference_coords is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/core/test_io_roundtrip.py -k "per_node_fields or backward_compatible" -v`
Expected: FAIL — writer/reader ignore the new fields; `schema_version` still `"0.1.0"`.

- [ ] **Step 3: Bump the version constant**

`src/structbench/core/schema.py:25`: `SCHEMA_VERSION = "0.2.0"`.

- [ ] **Step 4: Persist/read the new node datasets**

In `_write_nodes` (`io/__init__.py`), after writing `coords`/`node_id`, guard the optional fields:

```python
    if nodes.node_type is not None:
        grp.create_dataset("node_type", data=nodes.node_type.astype(np.int64))
    if nodes.reference_coords is not None:
        grp.create_dataset("reference_coords", data=nodes.reference_coords.astype(np.float64))
```

In `_read_nodes`, read them only if present (else `None`):

```python
    node_type = grp["node_type"][()].astype(np.int64) if "node_type" in grp else None
    reference_coords = (
        grp["reference_coords"][()].astype(np.float64) if "reference_coords" in grp else None
    )
    return Nodes(coords=..., node_id=..., node_type=node_type, reference_coords=reference_coords)
```

Confirm the `response/node` writer already stores arrays of arbitrary trailing width (it writes `arr.astype(np.float32)` chunked along the frame axis, so `(T, N, 1)` is fine). If it hardcodes `dim`, generalize it to `arr.shape`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/core/test_io_roundtrip.py -k "per_node_fields or backward_compatible" -v`
Expected: PASS.

- [ ] **Step 6: Run the whole core test suite (no regressions)**

Run: `python -m pytest tests/core -v`
Expected: PASS (the existing `test_roundtrip[shell]`/`[sph]` still pass; they set no new fields, so `metadata` equality now expects `schema_version="0.2.0"` on both sides — it is set by default, so equality holds).

- [ ] **Step 7: Commit**

```bash
git add src/structbench/core/io/__init__.py src/structbench/core/schema.py tests/core/test_io_roundtrip.py
git commit -m "feat(schema): persist per-node fields, bump SCHEMA_VERSION to 0.2.0"
```

---

### Task 4: `parse_meta` — pure meta.json parsing

**Files:**
- Create: `src/structbench/core/io/meshgraphnets.py`
- Test: `tests/core/test_meshgraphnets.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `parse_meta(meta: dict) -> dict[str, FieldSpec]` where `FieldSpec` is a small dataclass `(name: str, ftype: str, dtype: str, shape: tuple[int, ...])`; raises `ValueError` on an unknown `type`. This isolates the (testable, TF-free) structure of `meta.json` from the (TF-dependent) tfrecord decode in Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_meshgraphnets.py
import pytest
from structbench.core.io.meshgraphnets import parse_meta

_DEFORMING_PLATE_META = {
    "simulator": "comsol", "dt": 0, "trajectory_length": 400,
    "field_names": ["cells", "node_type", "mesh_pos", "world_pos", "stress"],
    "features": {
        "cells": {"type": "static", "dtype": "int32", "shape": [1, -1, 4]},
        "node_type": {"type": "static", "dtype": "int32", "shape": [1, -1, 1]},
        "mesh_pos": {"type": "static", "dtype": "float32", "shape": [1, -1, 3]},
        "world_pos": {"type": "dynamic", "dtype": "float32", "shape": [400, -1, 3]},
        "stress": {"type": "dynamic", "dtype": "float32", "shape": [400, -1, 1]},
    },
}


def test_parse_meta_reads_all_fields():
    specs = parse_meta(_DEFORMING_PLATE_META)
    assert set(specs) == set(_DEFORMING_PLATE_META["field_names"])
    assert specs["cells"].ftype == "static" and specs["cells"].shape == (1, -1, 4)
    assert specs["world_pos"].ftype == "dynamic" and specs["world_pos"].dtype == "float32"


def test_parse_meta_rejects_unknown_type():
    bad = {"field_names": ["x"], "trajectory_length": 1,
           "features": {"x": {"type": "bogus", "dtype": "float32", "shape": [1]}}}
    with pytest.raises(ValueError, match="bogus"):
        parse_meta(bad)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/core/test_meshgraphnets.py -k parse_meta -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `parse_meta` and `FieldSpec`**

```python
# src/structbench/core/io/meshgraphnets.py
"""Adapter: MeshGraphNets `deforming_plate` tfrecord -> canonical Case (ADR-0042).

TensorFlow is imported lazily inside :func:`read_deforming_plate` only, so
``import structbench`` never requires it (mirrors ``lsdyna.read_d3plot``).
"""
from __future__ import annotations
from dataclasses import dataclass

_VALID_FTYPES = {"static", "dynamic", "dynamic_varlen"}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    ftype: str
    dtype: str
    shape: tuple[int, ...]


def parse_meta(meta: dict) -> dict[str, FieldSpec]:
    """Parse a MeshGraphNets ``meta.json`` dict into per-field specs."""
    specs: dict[str, FieldSpec] = {}
    for name in meta["field_names"]:
        f = meta["features"][name]
        if f["type"] not in _VALID_FTYPES:
            raise ValueError(f"invalid data format: field {name!r} type {f['type']!r}")
        specs[name] = FieldSpec(name, f["type"], f["dtype"], tuple(f["shape"]))
    return specs
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/core/test_meshgraphnets.py -k parse_meta -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/structbench/core/io/meshgraphnets.py tests/core/test_meshgraphnets.py
git commit -m "feat(io): parse_meta for MeshGraphNets meta.json"
```

---

### Task 5: `build_deforming_plate_case` — the pure assembly core

**Files:**
- Modify: `src/structbench/core/io/meshgraphnets.py`
- Test: `tests/core/test_meshgraphnets.py`

**Interfaces:**
- Consumes: `FieldSpec`, Task 1-3 schema, `lsdyna.unit_factors`.
- Produces:
  `build_deforming_plate_case(arrays: dict[str, np.ndarray], *, source_units: str, case_id: str, dataset_id: str | None = None) -> Case`
  where `arrays` holds one trajectory: `cells (n_cells, 4) int`, `node_type (n_nodes,) int`, `mesh_pos (n_nodes, 3) float`, `world_pos (T, n_nodes, 3) float`, `stress (T, n_nodes, 1) float`. Returns a validated 3D `Case`.

- [ ] **Step 1: Write the failing test** (synthetic arrays — no TF, no real data)

```python
import numpy as np
from structbench.core import validate
from structbench.core.io.meshgraphnets import build_deforming_plate_case


def _synthetic_traj(n_nodes=5, n_cells=2, T=4):
    rng = np.random.default_rng(0)
    world0 = rng.random((n_nodes, 3)).astype(np.float32)
    world = np.stack([world0 + i * 0.1 for i in range(T)]).astype(np.float32)  # (T,N,3)
    return {
        "cells": rng.integers(0, n_nodes, (n_cells, 4)).astype(np.int32),
        "node_type": np.array([0, 0, 1, 3, 0][:n_nodes], dtype=np.int32),
        "mesh_pos": world0.copy(),
        "world_pos": world,
        "stress": rng.random((T, n_nodes, 1)).astype(np.float32),
    }


def test_build_case_maps_fields_and_validates():
    a = _synthetic_traj()
    case = build_deforming_plate_case(a, source_units="kg-m-s", case_id="dp-000")
    validate(case)  # must not raise
    assert case.metadata.dimension == 3
    assert case.metadata.units_convention == "SI"
    # coords == world_pos[0]; displacement is delta-from-initial
    np.testing.assert_allclose(case.nodes.coords, a["world_pos"][0], rtol=1e-6)
    np.testing.assert_allclose(
        case.response.node["displacement"][0], np.zeros((5, 3)), atol=1e-6
    )
    np.testing.assert_allclose(
        case.response.node["displacement"][2], a["world_pos"][2] - a["world_pos"][0], rtol=1e-5
    )
    # per-node static fields
    np.testing.assert_array_equal(case.nodes.node_type, a["node_type"].astype(np.int64))
    np.testing.assert_allclose(case.nodes.reference_coords, a["mesh_pos"], rtol=1e-6)
    # mesh edges live as tetra connectivity
    np.testing.assert_array_equal(case.elements["tetra"].connectivity, a["cells"].astype(np.int64))
    # per-node von Mises stress
    assert case.response.node["von_mises_stress"].shape == (4, 5, 1)
    # exactly one material with non-empty source_model
    assert len(case.materials) == 1 and case.materials[0].source_model


def test_build_case_identity_units_are_si():
    a = _synthetic_traj()
    case = build_deforming_plate_case(a, source_units="kg-m-s", case_id="dp-000")
    # kg-m-s is SI identity: coords equal raw world_pos[0]
    np.testing.assert_allclose(case.nodes.coords, a["world_pos"][0], rtol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/core/test_meshgraphnets.py -k build_case -v`
Expected: FAIL — `build_deforming_plate_case` not defined.

- [ ] **Step 3: Implement `build_deforming_plate_case`**

Append to `meshgraphnets.py` (reuse `unit_factors`; `pseudo-time` = step index because `dt=0`):

```python
import numpy as np
from ..schema import Case, ElementBlock, Material, Metadata, Nodes, Provenance, Response
from ..validation import validate
from .lsdyna import unit_factors

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
    """Assemble one deforming_plate trajectory into a validated 3D Case."""
    f = unit_factors(source_units)
    world = arrays["world_pos"].astype(np.float64)          # (T, N, 3)
    coords = world[0] * f["length"]                          # (N, 3)
    disp = (world - world[0][None]) * f["length"]            # (T, N, 3), delta-from-initial
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
        time=np.arange(n_frames, dtype=np.float64),         # pseudo-time: quasi-static (dt=0)
        node={
            "displacement": disp.astype(np.float32),
            "von_mises_stress": (arrays["stress"].astype(np.float64) * f["stress"]).astype(np.float32),
        },
    )
    metadata = Metadata(
        case_id=case_id,
        dimension=3,
        source_units=source_units,
        dataset_id=dataset_id,
        provenance=Provenance("COMSOL", "unknown", "unknown"),
    )
    case = Case(metadata=metadata, nodes=nodes, elements=elements,
                materials=[_PLATE_MATERIAL], response=response)
    validate(case)
    return case
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/core/test_meshgraphnets.py -k build_case -v`
Expected: PASS.

- [ ] **Step 5: Re-export from the package**

Add `build_deforming_plate_case` (and `parse_meta`) to `src/structbench/core/io/__init__.py`'s `__all__` and to `src/structbench/core/__init__.py` re-exports. Run: `python -c "from structbench.core import build_deforming_plate_case"` — no error.

- [ ] **Step 6: Commit**

```bash
git add src/structbench/core/io/meshgraphnets.py src/structbench/core/io/__init__.py src/structbench/core/__init__.py tests/core/test_meshgraphnets.py
git commit -m "feat(io): build_deforming_plate_case pure assembly core (ADR-0042)"
```

---

### Task 6: `read_deforming_plate` — lazy-TF tfrecord reader (env-gated)

**Files:**
- Modify: `src/structbench/core/io/meshgraphnets.py`
- Test: `tests/core/test_meshgraphnets.py` (importorskip-gated)

**Interfaces:**
- Consumes: `parse_meta`, `build_deforming_plate_case`.
- Produces: `read_deforming_plate(data_dir: str | Path, split: str) -> Iterator[dict[str, np.ndarray]]` — yields one `arrays` dict per trajectory in `<data_dir>/<split>.tfrecord`, decoded per `<data_dir>/meta.json`. `import structbench` must not import TF; TF is imported inside this function.

- [ ] **Step 1: Write the guard test** (verifies laziness without TF installed)

```python
def test_import_structbench_does_not_import_tensorflow():
    import sys, importlib
    for m in list(sys.modules):
        if m == "tensorflow" or m.startswith("tensorflow."):
            del sys.modules[m]
    importlib.import_module("structbench.core.io.meshgraphnets")
    assert "tensorflow" not in sys.modules  # not imported at module load
```

- [ ] **Step 2: Run to verify it fails or passes trivially**

Run: `python -m pytest tests/core/test_meshgraphnets.py -k does_not_import_tensorflow -v`
Expected: PASS if you keep the `import tensorflow` inside the function from the start (write the function per Step 3, then this guards it stays lazy).

- [ ] **Step 3: Implement the reader with the decode inside**

```python
from collections.abc import Iterator
from pathlib import Path
import json


def read_deforming_plate(data_dir: str | Path, split: str) -> Iterator[dict[str, np.ndarray]]:
    """Yield per-trajectory ``arrays`` dicts from ``<data_dir>/<split>.tfrecord``."""
    import tensorflow as tf  # lazy: keep TF out of the runtime (ADR-0042 §2a)

    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    specs = parse_meta(meta)
    ds = tf.data.TFRecordDataset(str(data_dir / f"{split}.tfrecord"))
    for raw in ds:
        parsed = tf.io.parse_single_example(
            raw, {name: tf.io.VarLenFeature(tf.string) for name in specs}
        )
        out: dict[str, np.ndarray] = {}
        for name, spec in specs.items():
            data = tf.io.decode_raw(parsed[name].values, getattr(tf, spec.dtype))
            data = tf.reshape(data, spec.shape)
            arr = data.numpy()
            if spec.ftype == "static":
                arr = arr[0]            # drop leading length-1 axis -> (N|cells, ...)
            out[name] = arr
        out["node_type"] = out["node_type"].reshape(-1)
        yield out
```

- [ ] **Step 4: Add an env-gated end-to-end decode test** (skipped unless TF present *and* a fixture path is set)

```python
import os


@pytest.mark.skipif(
    not os.environ.get("STRUCTBENCH_DEFORMING_PLATE_DIR"),
    reason="set STRUCTBENCH_DEFORMING_PLATE_DIR to a dir with meta.json + valid.tfrecord",
)
def test_read_first_trajectory_builds_valid_case():
    pytest.importorskip("tensorflow")
    from structbench.core.io.meshgraphnets import read_deforming_plate, build_deforming_plate_case
    from structbench.core import validate
    d = os.environ["STRUCTBENCH_DEFORMING_PLATE_DIR"]
    a = next(read_deforming_plate(d, "valid"))
    assert a["world_pos"].ndim == 3 and a["world_pos"].shape[2] == 3
    case = build_deforming_plate_case(a, source_units="kg-m-s", case_id="dp-valid-000")
    validate(case)
```

- [ ] **Step 5: Run the gated suite**

Run: `python -m pytest tests/core/test_meshgraphnets.py -v` (the decode test SKIPs without the env var; the laziness guard and synthetic tests PASS).

- [ ] **Step 6: Commit**

```bash
git add src/structbench/core/io/meshgraphnets.py tests/core/test_meshgraphnets.py
git commit -m "feat(io): lazy-TF read_deforming_plate tfrecord reader"
```

---

### Task 7: The offline `convert.py` batch driver

**Files:**
- Create: `data_generation/meshgraphnets/deforming_plate/convert.py`
- Create: `data_generation/meshgraphnets/deforming_plate/README.md`

**Interfaces:**
- Consumes: `read_deforming_plate`, `build_deforming_plate_case`, `write_case`.
- Produces: a CLI that writes `canonical/deforming_plate/<split>_<i>.h5`. Not importable (ADR-0010); no unit test (data-dependent) — verified by Task 8.

- [ ] **Step 1: Write `convert.py`** (mirror `data_generation/lsdyna/2D-Copper-Bar-Taylor-Impact/convert.py`)

```python
#!/usr/bin/env python
"""Convert MeshGraphNets deforming_plate tfrecords to canonical HDF5 (ADR-0042).

Run in a throwaway environment that has tensorflow + structbench installed.
Data is downloaded from source and converted locally; StructBench does not
redistribute it (the dataset carries no redistribution licence).
"""
from __future__ import annotations
import argparse
from pathlib import Path

from structbench.core import write_case
from structbench.core.io.meshgraphnets import build_deforming_plate_case, read_deforming_plate

DATASET_ID = "deforming_plate"
SOURCE_UNITS = "kg-m-s"  # PLACEHOLDER until Task 8 measures the true convention
SPLITS = {"train": "train", "valid": "val", "test": "test"}  # source split -> canonical split


def convert_split(data_dir: Path, source_split: str, canon_split: str,
                  out_dir: Path, *, limit: int | None, overwrite: bool) -> int:
    n = 0
    for i, arrays in enumerate(read_deforming_plate(data_dir, source_split)):
        if limit is not None and i >= limit:
            break
        case_id = f"{canon_split}_{i:04d}"
        out_path = out_dir / f"{case_id}.h5"
        if out_path.exists() and not overwrite:
            print(f"SKIP {case_id}")
            continue
        case = build_deforming_plate_case(arrays, source_units=SOURCE_UNITS,
                                          case_id=case_id, dataset_id=DATASET_ID)
        write_case(case, out_path)
        print(f"OK   {case_id}  ({case.nodes.coords.shape[0]} nodes)")
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert deforming_plate to canonical HDF5")
    p.add_argument("--data-root", required=True, help="dir with meta.json + *.tfrecord")
    p.add_argument("--out", required=True, help="canonical/deforming_plate output dir")
    p.add_argument("--split", choices=list(SPLITS), help="one split only (default all)")
    p.add_argument("--limit", type=int, default=None, help="first N trajectories per split")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    data_dir, out_dir = Path(args.data_root), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {args.split: SPLITS[args.split]} if args.split else SPLITS
    total = 0
    for canon_split, source_split in splits.items():
        total += convert_split(data_dir, source_split, canon_split, out_dir,
                               limit=args.limit, overwrite=args.overwrite)
    print(f"wrote {total} cases to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write the README** documenting: the throwaway TF env, the source download (`download_dataset.sh deforming_plate <dir>` or the GCS bucket URL), the no-rehost policy, and the `--limit 2` smoke invocation.

- [ ] **Step 3: Smoke the driver's argument parsing without data**

Run: `python data_generation/meshgraphnets/deforming_plate/convert.py --help`
Expected: prints usage, exit 0 (no TF/data touched by `--help`).

- [ ] **Step 4: Commit**

```bash
git add data_generation/meshgraphnets/deforming_plate/
git commit -m "feat(data-gen): deforming_plate download-and-convert driver"
```

---

### Task 8: Determine units on real data; convert a 2-case smoke; verify SI

**Files:** none (produces an ADR-0042 §2b units finding + a units patch to `convert.py:SOURCE_UNITS`).

This task needs the downloaded dataset (throwaway TF env). It is the units-measurement gate ADR-0042 requires before the archive is trusted.

- [ ] **Step 1: Download one split** to a scratch dir: `bash download_dataset.sh deforming_plate <scratch>/deforming_plate` (from the DeepMind meshgraphnets repo), or fetch the four GCS files directly.

- [ ] **Step 2: Measure the source units.** In a python shell in the TF env:

```python
from structbench.core.io.meshgraphnets import read_deforming_plate
import numpy as np
a = next(read_deforming_plate("<scratch>/deforming_plate", "valid"))
print("world_pos range:", np.ptp(a["world_pos"]), a["world_pos"].min(), a["world_pos"].max())
print("stress range:", a["stress"].min(), a["stress"].max())
```

Decide `SOURCE_UNITS`: a plate on the order of ~1 (m) with `collision_radius=0.03` implies metres → length identity; stress ~1e6-1e8 implies Pa (SI) vs ~1-100 implies MPa. Record the reasoning (Taylor/Concrete-Beam precedent, ADR-0030). If not SI-identity, set `SOURCE_UNITS` to the matching `unit_factors` token (e.g. `"kg-mm-s"`, `"kg-m-s"` with a stress note) and re-run `test_build_case_identity_units_are_si` semantics against the chosen token.

- [ ] **Step 3: Patch `convert.py:SOURCE_UNITS`** to the measured convention and commit: `git commit -am "fix(data-gen): deforming_plate source units = <measured> (ADR-0042 units gate)"`.

- [ ] **Step 4: Convert a 2-case smoke and validate on read.**

Run: `python data_generation/meshgraphnets/deforming_plate/convert.py --data-root <scratch>/deforming_plate --out <scratch>/canonical/deforming_plate --split valid --limit 2`
Then:
```python
from structbench.core import read_case, validate
for p in ["valid_0000.h5", "valid_0001.h5"]:
    c = read_case(f"<scratch>/canonical/deforming_plate/{p}")
    validate(c)
    assert c.metadata.dimension == 3 and c.metadata.units_convention == "SI"
    assert "tetra" in c.elements and c.nodes.node_type is not None
    assert c.response.node["von_mises_stress"].shape[2] == 1
print("smoke OK")
```
Expected: both cases validate; von Mises magnitudes are physically plausible in SI (Pa). This closes the ingestion loop; the full 1000/100/100 conversion is a later data-gen run (not in this plan).

- [ ] **Step 5: Record the smoke result** in the branch (a short note in `scratch/` or the PR description) — node counts, unit decision, VM range — as the evidence trail for blessing later.

---

## Self-Review

**Spec coverage (ADR-0042):** per-node `node_type` (Tasks 1,3,5) ✓; `reference_coords`/`mesh_pos` (Tasks 1,3,5) ✓; relaxed `response.node` for per-node scalar stress (Tasks 2,3,5) ✓; `SCHEMA_VERSION` 0.2.0 + backward-compat (Task 3) ✓; `cells`→`elements["tetra"]` (Task 5) ✓; download-and-convert, no rehost (Task 7 + README) ✓; TF stays offline/lazy (Tasks 6 guard) ✓; units measured, not assumed (Task 8) ✓; quasi-static pseudo-time (Task 5 `response.time`) ✓; synthesized material (Task 5) ✓.

**Out of scope (correctly deferred):** the benchmark module/card/registry, `datasets/` mesh-edge + `node_type`→`particle_type` plumbing, `models/mgn`, and the scored-horizon/metrics protocol — all need the benchmark-protocol ADR and are later plans.

**Placeholder scan:** the one intentional unknown is `SOURCE_UNITS` in Task 7, flagged as PLACEHOLDER and resolved by measurement in Task 8 — not a silent gap.

**Type consistency:** `build_deforming_plate_case(arrays, *, source_units, case_id, dataset_id=None)` and `read_deforming_plate(data_dir, split) -> Iterator[dict]` are used identically in Tasks 5-8; `Nodes(..., node_type, reference_coords)` and the `(T,N,k)` `response.node` rule are consistent across Tasks 1-5; `write_case`/`read_case`/`validate`/`unit_factors` are used per the verified signatures.
