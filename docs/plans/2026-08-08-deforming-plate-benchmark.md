# DeformingPlate Benchmark Module + Mesh-Aware Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land checkpoint ①-b of ADR-0041: the mesh-aware trajectory loader in `datasets/`, the two ADR-0043 QoI factories in `eval/`, and the `benchmarks/deforming_plate` module (splits, card, spec, registry entry, generated docs) — so that `get_benchmark("deforming_plate")` resolves a fully validated spec and a canonical mesh case loads into a model-ready `CaseTrajectory`. This plan does **not** include `models/mgn`, family dispatch in `cli/train.py`, or `WindowDataset`/`collate_samples` edge-passing (those live in the MGN plan) — nor real-data conversion (Task 8 of the ingestion plan, human-gated).

**Architecture:** Two substrate extensions come first: (1) `datasets/canonical.py` gains a mesh path — `CaseTrajectory` grows optional `cells`/`reference_coords` fields and `load_case_trajectory` dispatches on element type (`"sph"` present → existing SPH path unchanged; otherwise the mesh path: nodes-as-particles, `particle_type` from `nodes.node_type`, aux read directly from `response.node`); (2) `eval/metrics.py` gains the two declared QoI factories (`peak_nodal_aux`, `terminal_peak_displacement`), both masking excluded node types. On top of those, the benchmark module freezes the ADR-0043 splits and protocol and registers in the benchmark registry; docs regenerate.

**Tech Stack:** Python 3.12+, existing runtime deps only (numpy/h5py; the eval/datasets layers already sit above torch per ADR-0018 but nothing here adds a dep). **No TensorFlow anywhere** — test fixtures reuse the synthetic-array builder pattern from `tests/core/test_meshgraphnets.py`.

**Plan 2 of the v0.3 build order** (ADR-0041 ①-b; plan 1 was `2026-08-07-deforming-plate-ingestion.md`). Plan 3 (`models/mgn` + family dispatch + windowing edges) follows.

## Global Constraints

- Python floor **3.12**; ruff line length **88** + `ruff format`; mypy `disallow_untyped_defs = true`; NumPy-style docstrings on every public API; `_`-prefix symbols private across module boundaries.
- **No new dependencies. No TensorFlow imports anywhere in this plan.**
- **ADR-0043 frozen contract** (do not deviate): splits exactly `train_0000…train_0999` / `val_0000…val_0099` / `test_0000…test_0099` (1000/100/100, keys `train`/`val`/`test`, `eval_splits = ("val", "test")`); `input_frames = 2`; `kinematic_types = (1, 3)` (node-type codes: OBSTACLE=1, HANDLE=3; NORMAL=0 scored); `aux_field = "von_mises_stress"` read directly from `response/node/von_mises_stress`; full scored horizon (`scored_frames = None`, card `horizon = "full"`); `output_dt_ms = 1.0` nominal pseudo-time; QoIs `peak_vm_stress` and `terminal_peak_deflection` masked to exclude types (1, 3).
- **Pre-approved public-API change** (ADR-0043 Consequences, maintainer approved 2026-08-08): `CaseTrajectory` gains optional `cells` and `reference_coords` fields (default `None`) and `load_case_trajectory` gains the mesh dispatch path — signature unchanged, SPH behaviour byte-identical. `eval/` gains the two QoI factory exports. **No OTHER public-API changes without flagging.**
- Card numeric placeholders pending Task 8 (stated in ADR-0043): `source_units` carries the ingestion placeholder wording, `aux_unit = "MPa"` assumes Pa source (noted in `protocol_rationale`), `size_gb = None`. Do not invent measured values.
- Tests: pytest, **synthetic-only** (build mesh cases via `structbench.core.io.meshgraphnets.build_deforming_plate_case` on synthetic arrays — no real data, no downloads); deterministic (`np.random.default_rng(seed)`). RUN with the structbench conda env interpreter (PowerShell): `& "C:\Users\272766h\AppData\Local\miniconda3\envs\structbench\python.exe" -m pytest <target> -v`. Focused tests while iterating; the FULL suite (`python -m pytest -q`, baseline 252 passed/6 skipped + this plan's additions) once before each commit.
- After any card or registry change: regenerate docs (`python tools/gen_benchmark_docs.py`) — the drift test fails otherwise.
- Branch: **`feat/deforming-plate-benchmark`** off `docs/adr-0043-deforming-plate-protocol` (the stacked v0.3 chain). Never commit to `main`; merge/push are human calls (ADR-0023).
- Commits: Conventional Commits; append to every commit message, after a blank line:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01HHpG1wUFUfb2Q31Hp948YX`

## File Structure

```
src/structbench/datasets/
  canonical.py                 # MODIFY: CaseTrajectory +cells+reference_coords;
                               #   load_case_trajectory dispatch; _load_mesh_trajectory
src/structbench/eval/
  metrics.py                   # MODIFY: peak_nodal_aux + terminal_peak_displacement factories
  __init__.py                  # MODIFY: export the two new factories
src/structbench/benchmarks/
  registry.py                  # MODIFY: +"deforming_plate" in _MODULES
  deforming_plate/
    __init__.py                # CREATE: SPEC (kinematic_types=(1,3), results=())
    benchmark.py               # CREATE: TRAIN/VAL/TEST, AUX_FIELD, KINEMATIC_TYPES, QOIS
    card.py                    # CREATE: CARD
docs/benchmarks.md             # REGENERATE (Task 3 — before the benchmarks suite runs)
docs/benchmarks/deforming_plate.md  # GENERATED (Task 3)
tests/
  datasets/test_canonical.py   # MODIFY: mesh-path tests (+ SPH regression untouched)
  eval/test_metrics.py         # MODIFY: QoI factory tests
  benchmarks/test_deforming_plate_split.py  # CREATE
```

---

### Task 1: Mesh path in `load_case_trajectory` (+ `CaseTrajectory` fields)

**Files:**
- Modify: `src/structbench/datasets/canonical.py` (the `CaseTrajectory` dataclass and `load_case_trajectory`; add `_load_mesh_trajectory`)
- Test: `tests/datasets/test_canonical.py`

**Interfaces:**
- Consumes: `structbench.core.read_case`; `structbench.core.io.meshgraphnets.build_deforming_plate_case` (test fixture only); the existing `_apply_n_valid_frames`-style trim helper and SPH loader body (read the file first — reuse its internals, do not duplicate logic).
- Produces:
  - `CaseTrajectory` gains `cells: NDArray[np.int64] | None = None` (`(n_cells, nodes_per_cell)`, 0-indexed into the particle axis) and `reference_coords: NDArray[np.float32] | None = None` (`(P, dim)`, working frame — mm). SPH loads leave both `None`.
  - `load_case_trajectory(h5_path, *, aux_field="von_mises_stress", length_scale=1e3, stress_scale=1e-6) -> CaseTrajectory` — signature unchanged. Dispatch: `"sph" in case.elements` → existing SPH path, byte-identical behaviour; otherwise → mesh path.
  - Mesh path semantics: particles = all nodes; `positions = (coords[None] + displacement) * length_scale` as float32 (mm); `particle_type = nodes.node_type` (int64) — `ValueError` if `node_type` is `None`; `aux = response.node[aux_field][:, :, 0] * stress_scale` (float32) — `ValueError` naming available `response.node` keys if absent; `time` = the stored pseudo-time axis unchanged; `cells` = the single element block's connectivity (int64) — `ValueError` if the case has more than one element type (YAGNI: no mesh benchmark needs mixed blocks yet); `reference_coords = nodes.reference_coords * length_scale` (float32) when present, else `None`. The `n_valid_frames` trim applies exactly as in the SPH path (shown in Step 4; a uniform index axis is never trimmed).
  - **Ordering & gate placement (load-bearing):** `read_case` → the existing response-is-None `ValueError` (shared) → dispatch on `"sph" in case.elements`. The `_AUX_EXTRACTORS[aux_field]` lookup — today the function's FIRST statement, raising `KeyError` before any file read — **moves into the SPH branch**; the mesh branch performs no extractor lookup (`aux_field` is a `response.node` key there). Update the module docstring (drop "SPH particles only"), `load_case_trajectory`'s docstring (Raises: `KeyError` on the SPH path, `ValueError` on the mesh path), and `available_aux_fields()`'s docstring (its registry governs the SPH path; `BenchmarkSpec` still validates `aux_field` against it, which `"von_mises_stress"` satisfies — accepted debt until a mesh-only aux name arrives).

- [ ] **Step 1: Read `src/structbench/datasets/canonical.py` in full** — the exact `CaseTrajectory` definition, the SPH loader body, the trim helper's name and call pattern, and the module's error-message style. The code below names the pieces; bind it to the file's real spellings.

- [ ] **Step 2: Write the failing tests** (add to `tests/datasets/test_canonical.py`; reuse its imports/tmp-path style)

```python
# Merge new imports into the file's EXISTING top-of-file import block — it
# already imports CaseTrajectory/load_case_trajectory (from
# structbench.datasets.canonical); re-importing mid-file is a ruff F811.
# New names needed: write_case (structbench.core) if absent, and
# build_deforming_plate_case (structbench.core.io.meshgraphnets).


def _mesh_case_file(tmp_path, n_nodes=5, n_cells=2, T=4):
    """Synthetic deforming-plate-shaped canonical file (SI units)."""
    rng = np.random.default_rng(3)
    world0 = rng.random((n_nodes, 3)).astype(np.float32)
    world = np.stack([world0 + i * 0.01 for i in range(T)]).astype(np.float32)
    arrays = {
        "cells": rng.integers(0, n_nodes, (n_cells, 4)).astype(np.int32),
        "node_type": np.array([0, 0, 1, 3, 0], dtype=np.int32)[:n_nodes],
        "mesh_pos": world0.copy(),
        "world_pos": world,
        "stress": rng.random((T, n_nodes, 1)).astype(np.float32),
    }
    case = build_deforming_plate_case(arrays, source_units="kg-m-s", case_id="dp-t")
    path = tmp_path / "dp-t.h5"
    write_case(case, path)
    return path, arrays


def test_mesh_trajectory_loads_nodes_as_particles(tmp_path):
    path, a = _mesh_case_file(tmp_path)
    traj = load_case_trajectory(path, aux_field="von_mises_stress")
    T, P = a["world_pos"].shape[0], a["world_pos"].shape[1]
    assert traj.positions.shape == (T, P, 3)
    # positions are world_pos in mm (SI m * 1e3), frame 0 == initial coords
    np.testing.assert_allclose(
        traj.positions[0], a["world_pos"][0] * 1e3, rtol=1e-5
    )
    np.testing.assert_allclose(
        traj.positions[2], a["world_pos"][2] * 1e3, rtol=1e-5
    )
    np.testing.assert_array_equal(traj.particle_type, a["node_type"].astype(np.int64))
    # aux: stored Pa scalar -> MPa working frame
    np.testing.assert_allclose(
        traj.aux, a["stress"][:, :, 0] * 1e-6, rtol=1e-5
    )
    np.testing.assert_array_equal(traj.cells, a["cells"].astype(np.int64))
    np.testing.assert_allclose(
        traj.reference_coords, a["mesh_pos"] * 1e3, rtol=1e-5
    )


def test_mesh_trajectory_missing_aux_field_raises(tmp_path):
    # Deliberately an UNREGISTERED aux name: proves the mesh branch raises its
    # own ValueError and the _AUX_EXTRACTORS KeyError gate no longer runs first.
    path, _ = _mesh_case_file(tmp_path)
    with pytest.raises(ValueError, match="response.node"):
        load_case_trajectory(path, aux_field="volumetric_strain")


def test_sph_trajectory_leaves_mesh_fields_none(tmp_path):
    traj = load_case_trajectory(_sph_case(tmp_path))  # the file's existing helper
    assert traj.cells is None
    assert traj.reference_coords is None
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `python -m pytest tests/datasets/test_canonical.py -k "mesh or leaves_mesh" -v`
Expected: FAIL — the mesh case raises `KeyError: 'sph'` inside the current loader; the missing-aux test fails because today's extractor gate raises `KeyError` for the unregistered name (not the expected `ValueError`); the SPH test fails on the missing `cells` attribute.

- [ ] **Step 4: Implement** — extend the dataclass, split the current body into the SPH branch, add `_load_mesh_trajectory`:

```python
def _load_mesh_trajectory(
    case: Case,
    *,
    aux_field: str,
    length_scale: float,
    stress_scale: float,
) -> CaseTrajectory:
    """Load a nodal-FE (mesh) case: nodes are the particles (ADR-0043)."""
    nodes = case.nodes
    if nodes.node_type is None:
        raise ValueError(
            f"mesh case {case.metadata.case_id!r} has no nodes.node_type; "
            "mesh benchmarks require it (schema 0.2.0, ADR-0042)"
        )
    if len(case.elements) != 1:
        raise ValueError(
            f"mesh case {case.metadata.case_id!r} has element types "
            f"{sorted(case.elements)}; exactly one is supported"
        )
    response = case.response
    assert response is not None  # guaranteed: shared response-None check runs first
    if aux_field not in response.node:
        raise ValueError(
            f"aux field {aux_field!r} not in response.node "
            f"(available: {sorted(response.node)})"
        )
    n = n_valid_frames(np.asarray(response.time))  # same trim as the SPH path
    disp = response.node["displacement"][:n].astype(np.float64)
    positions = ((nodes.coords[None, :, :] + disp) * length_scale).astype(np.float32)
    aux = (
        response.node[aux_field][:n, :, 0].astype(np.float64) * stress_scale
    ).astype(np.float32)
    (block,) = case.elements.values()
    reference = (
        (nodes.reference_coords * length_scale).astype(np.float32)
        if nodes.reference_coords is not None
        else None
    )
    return CaseTrajectory(
        case_id=case.metadata.case_id,
        positions=positions,
        particle_type=nodes.node_type.astype(np.int64),
        aux=aux,
        time=response.time[:n].copy(),
        cells=block.connectivity.astype(np.int64),
        reference_coords=reference,
    )
```

Wire the dispatch as: `read_case` → the existing response-is-None `ValueError` (shared) → branch on `"sph" in case.elements`. **Move the `_AUX_EXTRACTORS[aux_field]` lookup inside the SPH branch** — the mesh branch performs no extractor lookup. Keep the SPH branch's code otherwise physically unchanged (move-only). Update the three docstrings named in the Interfaces bullet (module, `load_case_trajectory` Raises section, `available_aux_fields` scope).

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `python -m pytest tests/datasets/test_canonical.py -v`
Expected: new tests PASS and every pre-existing test in the file PASSES unchanged (SPH regression).

- [ ] **Step 6: Run the datasets + core suites, ruff, mypy**

Run: `python -m pytest tests/datasets tests/core -q` then `python -m ruff check src/structbench/datasets/canonical.py` / `ruff format --check` / `mypy` on the changed file.
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/structbench/datasets/canonical.py tests/datasets/test_canonical.py
git commit -m "feat(datasets): mesh-aware trajectory loading (nodes-as-particles, ADR-0043)"
```

---

### Task 2: QoI factories `peak_nodal_aux` and `terminal_peak_displacement`

**Files:**
- Modify: `src/structbench/eval/metrics.py` (two new factories, after the existing factory section)
- Modify: `src/structbench/eval/__init__.py` (exports)
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: `QoiInputs` (fields `time`, `positions (T,P,dim)`, `aux (T,P)`, `particle_type (P,) | None`, `init: int`) and `QoiFn = Callable[[QoiInputs], float]` — both already in `metrics.py`.
- Produces:
  - `peak_nodal_aux(*, exclude_types: tuple[int, ...] = ()) -> QoiFn` — max of `aux[init:]` over kept nodes (ADR-0043 `peak_vm_stress`).
  - `terminal_peak_displacement(*, exclude_types: tuple[int, ...] = ()) -> QoiFn` — max over kept nodes of `‖positions[-1] − positions[0]‖₂` (ADR-0043 `terminal_peak_deflection`; frame 0 is the ground-truth initial frame in both pred and true rollouts).
  - When `particle_type is None` or `exclude_types` is empty, no masking (all nodes kept).

- [ ] **Step 1: Write the failing tests** (add to `tests/eval/test_metrics.py`, following its QoiInputs-construction style)

```python
def test_peak_nodal_aux_masks_excluded_types():
    aux = np.zeros((4, 3), dtype=np.float32)
    aux[2, 0] = 5.0      # NORMAL node peak
    aux[3, 2] = 99.0     # kinematic node — must be ignored
    inp = QoiInputs(
        time=np.arange(4, dtype=np.float64),
        positions=np.zeros((4, 3, 3), dtype=np.float32),
        aux=aux,
        particle_type=np.array([0, 0, 1], dtype=np.int64),
        init=2,
    )
    qoi = peak_nodal_aux(exclude_types=(1, 3))
    assert qoi(inp) == pytest.approx(5.0)


def test_peak_nodal_aux_respects_init():
    aux = np.zeros((4, 2), dtype=np.float32)
    aux[0, 0] = 50.0     # before init — must be ignored
    aux[3, 1] = 2.0
    inp = QoiInputs(
        time=np.arange(4, dtype=np.float64),
        positions=np.zeros((4, 2, 3), dtype=np.float32),
        aux=aux,
        particle_type=None,
        init=2,
    )
    assert peak_nodal_aux()(inp) == pytest.approx(2.0)


def test_terminal_peak_displacement_masks_and_measures():
    pos = np.zeros((3, 3, 3), dtype=np.float32)
    pos[-1, 0] = [3.0, 4.0, 0.0]   # NORMAL: |disp| = 5
    pos[-1, 2] = [100.0, 0.0, 0.0]  # kinematic: ignored
    inp = QoiInputs(
        time=np.arange(3, dtype=np.float64),
        positions=pos,
        aux=np.zeros((3, 3), dtype=np.float32),
        particle_type=np.array([0, 0, 3], dtype=np.int64),
        init=2,
    )
    qoi = terminal_peak_displacement(exclude_types=(1, 3))
    assert qoi(inp) == pytest.approx(5.0)
```

(Add `peak_nodal_aux` and `terminal_peak_displacement` to the file's **existing** top-of-file `structbench.eval.metrics` import block together with the tests — no mid-file imports.)

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/eval/test_metrics.py -k "peak_nodal or terminal_peak" -v`
Expected: FAIL at collection — `ImportError: cannot import name 'peak_nodal_aux'` (the right RED reason: the import block now names the not-yet-implemented factories).

- [ ] **Step 3: Implement the factories** (in `metrics.py`, NumPy docstrings; mirror the existing factory style, e.g. `arrival_time`)

```python
def peak_nodal_aux(*, exclude_types: tuple[int, ...] = ()) -> QoiFn:
    """Peak nodal aux value over the scored span (ADR-0043 ``peak_vm_stress``)."""

    def qoi(inputs: QoiInputs) -> float:
        aux = inputs.aux[inputs.init :]
        if exclude_types and inputs.particle_type is not None:
            aux = aux[:, ~np.isin(inputs.particle_type, exclude_types)]
        return float(aux.max())

    return qoi


def terminal_peak_displacement(*, exclude_types: tuple[int, ...] = ()) -> QoiFn:
    """Peak displacement magnitude at the final frame (ADR-0043
    ``terminal_peak_deflection``); frame 0 is the ground-truth initial state."""

    def qoi(inputs: QoiInputs) -> float:
        disp = np.linalg.norm(
            inputs.positions[-1] - inputs.positions[0], axis=-1
        )
        if exclude_types and inputs.particle_type is not None:
            disp = disp[~np.isin(inputs.particle_type, exclude_types)]
        return float(disp.max())

    return qoi
```

Add both to `eval/__init__.py` exports alongside the existing QoI names.

- [ ] **Step 4: Run to verify they pass**, then the eval suite: `python -m pytest tests/eval -q`. Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/structbench/eval/metrics.py src/structbench/eval/__init__.py tests/eval/test_metrics.py
git commit -m "feat(eval): peak_nodal_aux + terminal_peak_displacement QoI factories (ADR-0043)"
```

---

### Task 3: The `benchmarks/deforming_plate` module + registry entry

**Files:**
- Create: `src/structbench/benchmarks/deforming_plate/benchmark.py`
- Create: `src/structbench/benchmarks/deforming_plate/card.py`
- Create: `src/structbench/benchmarks/deforming_plate/__init__.py`
- Modify: `src/structbench/benchmarks/registry.py` (one `_MODULES` line)
- Test: `tests/benchmarks/test_deforming_plate_split.py`

**Interfaces:**
- Consumes: `BenchmarkCard` (`benchmarks/card.py`), `BenchmarkSpec`/registry (`benchmarks/registry.py`), Task 2's factories.
- Produces: `get_benchmark("deforming_plate") -> BenchmarkSpec` with `kinematic_types = (1, 3)`, `scored_frames = None`, `results = ()`, `eval_splits = ("val", "test")`, `dataset_id = "deforming_plate"`.

- [ ] **Step 1: Read one exemplar module end to end** (`src/structbench/benchmarks/wave_propagation_1d/{benchmark,card,__init__}.py`) and `benchmarks/card.py` + `registry.py` validation, so every required card field and spec kwarg is bound to the real current definitions.

- [ ] **Step 2: Write the failing tests**

```python
# tests/benchmarks/test_deforming_plate_split.py
from structbench.benchmarks import get_benchmark


def test_split_sizes_and_ids():
    spec = get_benchmark("deforming_plate")
    assert len(spec.splits["train"]) == 1000
    assert len(spec.splits["val"]) == 100
    assert len(spec.splits["test"]) == 100
    assert spec.splits["train"][0] == "train_0000"
    assert spec.splits["train"][-1] == "train_0999"
    assert spec.splits["val"][0] == "val_0000"
    assert spec.splits["test"][-1] == "test_0099"
    all_ids = [i for s in ("train", "val", "test") for i in spec.splits[s]]
    assert len(set(all_ids)) == 1200  # disjoint


def test_protocol_pins():
    spec = get_benchmark("deforming_plate")
    assert spec.card.input_frames == 2
    assert spec.kinematic_types == (1, 3)
    assert spec.scored_frames is None
    assert spec.card.horizon == "full"
    assert spec.aux_field == "von_mises_stress"
    assert set(spec.qois) == {"peak_vm_stress", "terminal_peak_deflection"}
    assert spec.eval_splits == ("val", "test")
    assert spec.results == ()
```

- [ ] **Step 3: Run to verify they fail** (`KeyError`/unknown benchmark).

- [ ] **Step 4: Implement `benchmark.py`**

```python
"""DeformingPlate task facts: frozen split, aux field, kinematics, QoIs (ADR-0043)."""

from ...eval import peak_nodal_aux, terminal_peak_displacement
from ...eval.metrics import QoiFn

#: Fixed, immutable split (ADR-0043 §2) — the published MeshGraphNets split
#: verbatim; ids follow the converter naming. Changing it is a new benchmark
#: version.
TRAIN = [f"train_{i:04d}" for i in range(1000)]
VAL = [f"val_{i:04d}" for i in range(100)]
TEST = [f"test_{i:04d}" for i in range(100)]

#: Nodal auxiliary target (ADR-0043 §5): stored per-node von Mises stress.
AUX_FIELD = "von_mises_stress"

#: Node-type codes prescribed from ground truth and excluded from scoring
#: (ADR-0043 §4): OBSTACLE = 1 (scripted actuator), HANDLE = 3 (fixed).
KINEMATIC_TYPES = (1, 3)

QOIS: dict[str, QoiFn] = {
    "peak_vm_stress": peak_nodal_aux(exclude_types=KINEMATIC_TYPES),
    "terminal_peak_deflection": terminal_peak_displacement(
        exclude_types=KINEMATIC_TYPES
    ),
}
```

(Adjust the relative-import depth/names to match how the wave module imports from `eval` — bind to the real pattern found in Step 1.)

- [ ] **Step 5: Implement `card.py`** — `CARD = BenchmarkCard(...)` with exactly these protocol values (prose fields may be edited for accuracy, values may not):

```python
CARD = BenchmarkCard(
    name="DeformingPlate",
    version="0.1",
    description=(
        "Quasi-static deformation of a hyperelastic 3D plate pressed by a "
        "scripted rigid actuator; the MeshGraphNets deforming_plate dataset "
        "(Pfaff et al. 2021) under the ADR-0043 rollout protocol."
    ),
    provenance=(
        "MeshGraphNets dataset (Pfaff et al., ICLR 2021; COMSOL ground "
        "truth), downloaded from the DeepMind source bucket and converted "
        "locally to canonical HDF5 (ADR-0042; not redistributed)."
    ),
    data_license="None stated by the source; downloaded from source, not redistributed (ADR-0042)",
    solver="COMSOL",
    discretisation="FEM",
    materials=("Hyperelastic (constants not published with the dataset)",),
    erosion=False,
    loading=(
        "Scripted rigid actuator (OBSTACLE nodes, kinematic); "
        "HANDLE nodes fixed"
    ),
    source_units="kg-m-s (ingestion placeholder — measured at conversion, ADR-0042 §2b)",
    geometry="3D tetrahedral mesh: deformable plate + actuator, ~1,271 nodes avg (ragged)",
    n_cases=len(TRAIN) + len(VAL) + len(TEST),
    splits={"train": len(TRAIN), "val": len(VAL), "test": len(TEST)},
    task="quasi-static load-stepping autoregressive rollout (ADR-0043)",
    aux_field=AUX_FIELD,
    aux_unit="MPa",
    qois=tuple(QOIS),
    fields=("node/displacement", "node/von_mises_stress"),
    particles_per_case="~1,271 nodes avg (lo-hi range measured at Task 8)",
    n_frames=400,
    output_dt_ms=1.0,
    input_frames=2,
    horizon="full",
    protocol_rationale=(
        "input_frames=2 is the floor (a velocity needs two frames) and the "
        "faithful value: the source model uses h=0 history — node inputs are "
        "the one-hot node type only — so no window tuning question exists "
        "and no ground-truth timeline analysis can move it (ADR-0043 §3). "
        "Pseudo-time: dt=0 in the source (quasi-static); time is the frame "
        "index and output_dt_ms=1.0 is nominal, not milliseconds. aux_unit "
        "MPa assumes the source stress is Pa (SI); confirmed or corrected "
        "when the units measurement lands (ADR-0042 §2b). Scored span is "
        "[2, 400), exclusive end (ADR-0043 §6)."
    ),
    size_gb=None,
)
```

(Include every field the real `BenchmarkCard` requires — bind to the actual dataclass found in Step 1; if a required field is missing above, take the wave card's treatment.)

(Card conventions, deliberate: `loading` is a plain `str` — implicit concatenation, **no trailing comma**. `fields` lists **response** fields only, in the established `node/*` namespace — no `response/` prefix, no geometry/topology entries; the env-gated card-vs-data test asserts set equality against the response groups. `particles_per_case` deliberately deviates from the parseable `"lo-hi"` convention until the range is measured: **Task 8 must replace it with the measured `"lo-hi"` string before wiring `deforming_plate` into `test_card_data.py`'s data-root table**, whose parser requires that format.)

- [ ] **Step 6: Implement `__init__.py`** (mirror the wave module's shape)

```python
RESULTS: tuple[BaselineResult, ...] = ()

SPEC = BenchmarkSpec(
    card=CARD,
    results=RESULTS,
    splits={"train": tuple(TRAIN), "val": tuple(VAL), "test": tuple(TEST)},
    eval_splits=("val", "test"),
    aux_field=AUX_FIELD,
    qois=dict(QOIS),
    boundary_feature_fn=None,
    dataset_id="deforming_plate",
    kinematic_types=KINEMATIC_TYPES,
)
```

Register in `registry.py`: add `"deforming_plate": "structbench.benchmarks.deforming_plate",` to `_MODULES`.

- [ ] **Step 7: Regenerate the docs BEFORE running the benchmarks suite.** Registering in `_MODULES` makes the render drift tests look for `docs/benchmarks/deforming_plate.md` (FileNotFoundError until generated):

Run: `python tools/gen_benchmark_docs.py` — writes `docs/benchmarks.md` + `docs/benchmarks/deforming_plate.md` (generated; never hand-edit).

- [ ] **Step 8: Run the tests to verify they pass**, plus the whole benchmarks suite: `python -m pytest tests/benchmarks -q`. Expected: green — `BenchmarkSpec.__post_init__` validates split presence and card-vs-actual split sizes, `eval_splits` membership, `aux_field` registration, results split names, and `scored_frames` bounds (QoI *wiring* is asserted by `test_protocol_pins`, not by construction).

- [ ] **Step 9: Commit** (the generated docs travel with the module — the drift test binds them)

```bash
git add src/structbench/benchmarks/deforming_plate/ src/structbench/benchmarks/registry.py tests/benchmarks/test_deforming_plate_split.py docs/benchmarks.md docs/benchmarks/deforming_plate.md
git commit -m "feat(benchmarks): deforming_plate module — ADR-0043 protocol, frozen split, registry entry"
```

---

### Task 4: Drift guard + full-suite gate (verification only)

**Files:** none modified — the generated docs were produced and committed in Task 3.

- [ ] **Step 1: Drift guard** — `python tools/gen_benchmark_docs.py --check`. Expected: green (any drift here is a Task 3 bug — fix there, not here). The new landing page renders without narrative (no `overview` yet — the ADR-0036 pattern; authored later).

- [ ] **Step 2: Full gate** — `python -m pytest -q` (expect the 252-passed/6-skipped baseline plus this plan's additions, no failures), `ruff check src tests`, `ruff format --check src tests`, `mypy src`.

- [ ] **Step 3: Nothing to commit.** If any check fails, the fix belongs to the owning task's files — report it as such.

---

## Self-Review

**Spec coverage (ADR-0043):** split §2 (Task 3: generated id lists + sizes) ✓; input_frames=2 §3 (card) ✓; kinematic_types=(1,3) as node-type codes §4 (spec + loader sources particle_type from node_type) ✓; nodal aux read §5 (Task 1 mesh path) ✓; full horizon §6 (scored_frames=None, horizon="full") ✓; QoIs §7 (Task 2 factories, Task 3 wiring, masked to (1,3)) ✓; pseudo-time/output_dt_ms §1 (card + rationale) ✓; Task-8-gated card values marked as placeholders per Consequences ✓. **Deliberately out** (MGN plan): §8 recipe/gate, family dispatch, windowing/collate edges, §4's training-side noise/loss NORMAL-masking (a training-loop concern), and §9's declared choices — §9a's stress-supervision loss lives in the MGN trainer; §9b's GT actuator prescription is *activated here* by `kinematic_types=(1,3)` through the existing `eval.rollout` mechanism. (Ingestion Task 8): real-data conversion, units, `size_gb`, the measured `particles_per_case` range.

**Placeholder scan:** the card's `source_units`/`aux_unit`/`size_gb`/`particles_per_case` placeholders are ADR-mandated pending Task 8 and labeled as such in the card text and the card-conventions note — intentional, not plan gaps. Two "bind to the real spelling" steps (Task 1 Step 1, Task 3 Step 1) direct the implementer to read the actual current definitions before transcribing — deliberate guards against drift between plan-writing and execution, with the semantics fully specified here.

**Type consistency:** `CaseTrajectory(cells, reference_coords)` (Task 1) matches nothing downstream in this plan (consumed by the MGN plan); `peak_nodal_aux`/`terminal_peak_displacement` signatures in Task 2 match Task 3's imports; `KINEMATIC_TYPES=(1,3)` flows benchmark.py → SPEC → tests; split names/ids consistent across converter (`val_*.h5`), benchmark.py, and tests.
