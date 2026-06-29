# Taylor 2D GNS Surrogate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate an autoregressive GNS surrogate on the canonical Taylor 2D dataset, end to end.

**Architecture:** Light `core/` substrate (unchanged) feeds a torch-based ML layer. `datasets/` turns canonical HDF5 cases into model-ready particle trajectories and windowed training samples; `models/gns/` is the ported single-scale GNS (encode-process-decode learned simulator); `eval/` does autoregressive rollout + metrics; `benchmarks/taylor_impact_2d/` fixes the split + benchmark specifics; `cli/train.py` runs train/valid/rollout.

**Tech Stack:** Python 3.11+, NumPy, h5py, PyTorch, PyTorch Geometric, pytest, ruff, mypy.

## Global Constraints

- **Python 3.11+** floor; may use `tomllib`. (PRINCIPLES.md)
- **Ruff** formatter+linter, **line length 88**; imports stdlib/third-party/first-party. Match ruff output.
- **mypy** must pass on public APIs (re-exported from `__init__.py`); type hints required on every public API.
- **NumPy-style docstrings** on every public API; state units and array shapes.
- **`core/` imports neither `torch` nor `torch_geometric`** (ADR-0018). The ML stack lives only in `datasets/`, `models/`, `eval/`, `cli/`.
- **Tests**: pytest under `tests/` mirroring package layout; deterministic; **CPU-only, no GPU, no large data files, no solver, no network** (PRINCIPLES.md). Use tiny synthetic fixtures.
- **Logging** via stdlib `logging`, `logging.getLogger(__name__)`; no `print` in library code (CLI may print).
- **Units**: canonical storage is strict SI; the ML layer works in **mm** (positions ×1e3) and **MPa** (stress ×1e-6) so the ported hyperparameters transfer. (ADR-0019, spec)
- **Dev env**: `.venv` (uv, py3.11). Run tools as `.venv/Scripts/python.exe -m pytest|ruff|mypy`. Torch+PyG are added in Task 1.
- **Reference source** (read-only, not in repo tree): the original model is at `../code/sgnn/sgnn/single_scale/{graph_network.py,learned_simulator.py,train.py,evaluate.py,config.yaml}`. Tasks 6–10 port from these.
- **Commits**: Conventional Commits; agent commits end with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer. Feature branch only (currently `init/foundation`); never `main`.

---

## File Structure

```
src/structbench/
  benchmarks/__init__.py                         # NEW namespace
  benchmarks/taylor_impact_2d/__init__.py        # re-exports split + helpers
  benchmarks/taylor_impact_2d/benchmark.py       # split lists, wall feature, QoIs
  datasets/__init__.py                           # NEW namespace; re-exports
  datasets/canonical.py                          # CaseTrajectory, load_case_trajectory, von_mises_from_voigt
  datasets/normalization.py                      # NormalizationStats, compute_stats, save/load
  datasets/particle.py                           # WindowDataset, collate_samples
  models/__init__.py                             # NEW namespace
  models/gns/__init__.py                         # re-exports
  models/gns/graph_network.py                    # EncodeProcessDecode (ported)
  models/gns/simulator.py                        # LearnedSimulator (ported, generalised)
  eval/__init__.py                               # NEW namespace; re-exports
  eval/metrics.py                                # position_rmse, von_mises_rmse, QoIs
  eval/rollout.py                                # rollout, RolloutResult
  cli/__init__.py                                # NEW namespace
  cli/train.py                                   # GNSConfig, TrainConfig, train/validate/rollout entry
tests/
  benchmarks/test_taylor_split.py
  datasets/test_canonical.py
  datasets/test_normalization.py
  datasets/test_particle.py
  models/gns/test_graph_network.py
  models/gns/test_simulator.py
  eval/test_metrics.py
  eval/test_rollout.py
```

Each `tests/<module>/` needs an `__init__.py` only if the existing suite uses them — it does not (flat test modules), so do not add package inits to tests.

---

### Task 1: Add the ML stack as runtime dependencies

**Files:**
- Modify: `pyproject.toml` (dependencies; add `[[tool.mypy.overrides]]` for torch/PyG)
- Modify: `PRINCIPLES.md` (approved runtime dependency table)
- Test: `tests/test_torch_free_core.py` (Create)

**Interfaces:**
- Produces: an environment where `import torch`, `import torch_geometric`, and `import structbench.core` (without torch) all work.

- [ ] **Step 1: Write the failing test** — `core/` must not import torch.

```python
# tests/test_torch_free_core.py
"""ADR-0018: the core substrate must not depend on the ML stack."""
import ast
import pathlib

CORE = pathlib.Path("src/structbench/core")


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def test_core_does_not_import_torch():
    offenders = {
        str(p): {"torch", "torch_geometric"} & _imported_modules(p)
        for p in CORE.rglob("*.py")
        if {"torch", "torch_geometric"} & _imported_modules(p)
    }
    assert not offenders, f"core/ must stay torch-free (ADR-0018): {offenders}"
```

- [ ] **Step 2: Run test to verify it passes already** (core is currently torch-free)

Run: `.venv/Scripts/python.exe -m pytest tests/test_torch_free_core.py -v`
Expected: PASS (this is a guard test; it stays green for the whole project).

- [ ] **Step 3: Add deps to `pyproject.toml`**

In `[project].dependencies` add (keep loose lower bounds):
```toml
    "torch>=2.0",
    "torch-geometric>=2.3",
```
In the mypy overrides list extend the module array:
```toml
[[tool.mypy.overrides]]
module = ["h5py.*", "lasso.*", "torch_geometric.*"]
ignore_missing_imports = true
```
(`torch` ships its own type stubs, so it does not need an override.)

- [ ] **Step 4: Add to the PRINCIPLES.md approved runtime table**

Add two rows under the Runtime table:
```
| torch | Autograd + tensors for the reference ML models | ADR-0018 |
| torch-geometric | Message-passing + radius_graph for the GNS | ADR-0018 |
```

- [ ] **Step 5: Install into the dev env**

Run: `uv pip install --python .venv "torch>=2.0" "torch-geometric>=2.3"`
Expected: both import; `.venv/Scripts/python.exe -c "import torch, torch_geometric; print(torch.__version__)"` prints a version.

- [ ] **Step 6: Full check + commit**

Run: `.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m mypy`
Expected: all green.
```bash
git add pyproject.toml PRINCIPLES.md tests/test_torch_free_core.py
git commit -m "build: add torch + torch_geometric runtime deps (ADR-0018)"
```

---

### Task 2: Taylor benchmark split + specifics

**Files:**
- Create: `src/structbench/benchmarks/__init__.py`
- Create: `src/structbench/benchmarks/taylor_impact_2d/__init__.py`
- Create: `src/structbench/benchmarks/taylor_impact_2d/benchmark.py`
- Test: `tests/benchmarks/test_taylor_split.py`

**Interfaces:**
- Produces:
  - `TRAIN, VAL, TEST_INTERP, TEST_EXTRAP, HELD_ASIDE: list[str]` (case-id lists, ADR-0019).
  - `ALL_BENCHMARK_CASES: list[str]` (train+val+test, excludes held-aside).
  - `WALL_X_MM: float = -2.0`
  - `wall_distance_feature(positions_mm: Tensor, radius: float) -> Tensor` → `(P, 1)` clamped distance to the wall plane.
  - `AUX_FIELD: str = "von_mises_stress"`

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_taylor_split.py
import torch

from structbench.benchmarks.taylor_impact_2d import (
    HELD_ASIDE, TEST_EXTRAP, TEST_INTERP, TRAIN, VAL, wall_distance_feature,
)


def test_split_partitions_the_33_parametric_cases():
    splits = [TRAIN, VAL, TEST_INTERP, TEST_EXTRAP]
    sizes = [len(s) for s in splits]
    assert sizes == [21, 3, 6, 3]
    all_cases = [c for s in splits for c in s]
    assert len(all_cases) == len(set(all_cases)) == 33   # no overlap
    assert HELD_ASIDE == ["T-20-80-Convergence"]


def test_split_velocities_match_adr_0019():
    def vels(split):
        return sorted({int(c.split("-")[3]) for c in split})
    assert vels(VAL) == [150]
    assert vels(TEST_INTERP) == [130, 170]
    assert vels(TEST_EXTRAP) == [200]


def test_wall_distance_feature_clamps_to_radius():
    pos = torch.tensor([[-2.0, 0.0], [-1.5, 0.0], [10.0, 0.0]])  # mm
    feat = wall_distance_feature(pos, radius=0.6)
    assert feat.shape == (3, 1)
    torch.testing.assert_close(feat[:, 0], torch.tensor([0.0, 0.5, 0.6]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/benchmarks/test_taylor_split.py -v`
Expected: FAIL (ModuleNotFoundError: structbench.benchmarks).

- [ ] **Step 3: Write the benchmark module**

```python
# src/structbench/benchmarks/__init__.py
"""Benchmark problem definitions (ARCHITECTURE.md)."""
```

```python
# src/structbench/benchmarks/taylor_impact_2d/benchmark.py
"""The v0.1 Taylor 2D impact benchmark: split, wall feature, QoIs (ADR-0019)."""

from __future__ import annotations

import torch

_GEOMS = (60, 80, 100)


def _cases(velocities: tuple[int, ...]) -> list[str]:
    return [f"T-20-{g}-{v}" for v in velocities for g in _GEOMS]


#: Fixed, immutable split (ADR-0019). Changing it is a new benchmark version.
TRAIN: list[str] = _cases((100, 110, 120, 140, 160, 180, 190))
VAL: list[str] = _cases((150,))
TEST_INTERP: list[str] = _cases((130, 170))
TEST_EXTRAP: list[str] = _cases((200,))
HELD_ASIDE: list[str] = ["T-20-80-Convergence"]
ALL_BENCHMARK_CASES: list[str] = TRAIN + VAL + TEST_INTERP + TEST_EXTRAP

#: Auxiliary per-particle target field (named correctly, not "strain").
AUX_FIELD = "von_mises_stress"

#: Rigidwall plane position in the model's mm working frame.
WALL_X_MM = -2.0


def wall_distance_feature(positions_mm: torch.Tensor, radius: float) -> torch.Tensor:
    """Per-particle distance to the rigidwall plane, clamped to ``[0, radius]``.

    Parameters
    ----------
    positions_mm:
        Current particle positions, shape ``(P, dim)``, in mm.
    radius:
        Connectivity radius (mm); distances are clamped to it.

    Returns
    -------
    torch.Tensor
        Shape ``(P, 1)``: ``clamp(x - WALL_X_MM, 0, radius)``.
    """
    return torch.clamp(positions_mm[:, 0:1] - WALL_X_MM, min=0.0, max=radius)
```

```python
# src/structbench/benchmarks/taylor_impact_2d/__init__.py
"""v0.1 Taylor 2D impact benchmark (ADR-0019)."""

from .benchmark import (
    ALL_BENCHMARK_CASES,
    AUX_FIELD,
    HELD_ASIDE,
    TEST_EXTRAP,
    TEST_INTERP,
    TRAIN,
    VAL,
    WALL_X_MM,
    wall_distance_feature,
)

__all__ = [
    "TRAIN", "VAL", "TEST_INTERP", "TEST_EXTRAP", "HELD_ASIDE",
    "ALL_BENCHMARK_CASES", "AUX_FIELD", "WALL_X_MM", "wall_distance_feature",
]
```

- [ ] **Step 4: Run tests + lint + types**

Run: `.venv/Scripts/python.exe -m pytest tests/benchmarks/ -v && .venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m mypy`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/structbench/benchmarks tests/benchmarks
git commit -m "feat(benchmarks): Taylor 2D split + wall feature (ADR-0019)"
```

---

### Task 3: `datasets/canonical` — case → trajectory + von Mises

**Files:**
- Create: `src/structbench/datasets/__init__.py`
- Create: `src/structbench/datasets/canonical.py`
- Test: `tests/datasets/test_canonical.py`

**Interfaces:**
- Consumes: `structbench.core.write_case` (to build HDF5 fixtures in tests), the canonical HDF5 layout.
- Produces:
  - `von_mises_from_voigt(stress: NDArray) -> NDArray` — input `(..., 6)` Voigt `[xx,yy,zz,xy,yz,zx]`, output `(...)`.
  - `@dataclass CaseTrajectory` with fields `case_id: str`, `positions: NDArray[float32]` `(T,P,dim)` mm, `particle_type: NDArray[int64]` `(P,)`, `von_mises: NDArray[float32]` `(T,P)` MPa, `time: NDArray[float64]` `(T,)` s.
  - `load_case_trajectory(h5_path: str | Path, *, length_scale: float = 1e3, stress_scale: float = 1e-6) -> CaseTrajectory`.

- [ ] **Step 1: Write the failing test**

```python
# tests/datasets/test_canonical.py
import numpy as np

from structbench.core import (
    Case, ElementBlock, Material, Metadata, Nodes, Response, write_case,
)
from structbench.datasets.canonical import (
    CaseTrajectory, load_case_trajectory, von_mises_from_voigt,
)


def test_von_mises_uniaxial_equals_axial_stress():
    s = np.zeros((1, 6))
    s[0, 0] = 250.0  # pure sigma_xx
    np.testing.assert_allclose(von_mises_from_voigt(s), [250.0], rtol=1e-6)


def _sph_case(tmp_path):
    # 3 SPH particles + 1 shell node, 2 frames, SI units.
    coords = np.array([[0.0, 0.0], [1e-3, 0.0], [0.0, 1e-3], [5e-3, 5e-3]])
    disp = np.zeros((2, 4, 2), dtype=np.float32)
    disp[1, :3, 0] = 2e-3  # +2 mm in x at frame 1, SPH particles only
    stress = np.zeros((2, 3, 6), dtype=np.float32)
    stress[1, :, 0] = 300e6  # 300 MPa sigma_xx at frame 1
    case = Case(
        metadata=Metadata(case_id="T-test", dimension=2, source_units="g-mm-ms"),
        nodes=Nodes(coords=coords, node_id=np.arange(1, 5, dtype=np.int64)),
        elements={
            "sph": ElementBlock(
                connectivity=np.arange(3, dtype=np.int64).reshape(3, 1),
                element_id=np.arange(1, 4, dtype=np.int64),
                part_id=np.ones(3, dtype=np.int64),
            ),
            "shell": ElementBlock(
                connectivity=np.array([[3, 3, 3, 3]], dtype=np.int64),
                element_id=np.array([99], dtype=np.int64),
                part_id=np.array([2], dtype=np.int64),
            ),
        },
        materials=[Material(2, "MAT_ELASTIC_PLASTIC_HYDRO", {"data": [[2]]}, None)],
        response=Response(
            time=np.array([0.0, 2e-6]),
            node={"displacement": disp},
            element={"sph": {"stress": stress}},
        ),
    )
    path = tmp_path / "case.h5"
    write_case(case, path)
    return path


def test_load_case_trajectory_sph_only_in_mm_and_mpa(tmp_path):
    traj = load_case_trajectory(_sph_case(tmp_path))
    assert isinstance(traj, CaseTrajectory)
    assert traj.positions.shape == (2, 3, 2)          # SPH particles only
    np.testing.assert_allclose(traj.positions[0, 1], [1.0, 0.0])  # 1 mm
    np.testing.assert_allclose(traj.positions[1, 0], [2.0, 0.0])  # +2 mm disp
    assert traj.von_mises.shape == (2, 3)
    np.testing.assert_allclose(traj.von_mises[1], [300.0, 300.0, 300.0])  # MPa
    np.testing.assert_array_equal(traj.particle_type, [1, 1, 1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/datasets/test_canonical.py -v`
Expected: FAIL (ModuleNotFoundError: structbench.datasets).

- [ ] **Step 3: Write the module**

```python
# src/structbench/datasets/__init__.py
"""Data loading: canonical cases -> model-ready trajectories and samples."""
```

```python
# src/structbench/datasets/canonical.py
"""Read a canonical case into a model-ready particle trajectory.

The ML layer works in millimetres and megapascals (ADR-0019); canonical
storage is strict SI, so positions are scaled by ``length_scale`` (m->mm) and
stress by ``stress_scale`` (Pa->MPa) here. Only SPH particles are returned;
visualization shell nodes are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..core import read_case


def von_mises_from_voigt(stress: NDArray[np.floating]) -> NDArray[np.float64]:
    """von Mises stress from a Voigt tensor ``[xx, yy, zz, xy, yz, zx]``.

    Parameters
    ----------
    stress:
        Array with last axis of length 6.

    Returns
    -------
    numpy.ndarray
        Same leading shape as ``stress`` with the last axis removed.
    """
    s = np.asarray(stress, dtype=np.float64)
    sx, sy, sz, sxy, syz, szx = (s[..., i] for i in range(6))
    return np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + szx**2)
    )


@dataclass
class CaseTrajectory:
    """One case as a particle trajectory in the ML working frame (mm, MPa)."""

    case_id: str
    positions: NDArray[np.float32]  # (T, P, dim), mm
    particle_type: NDArray[np.int64]  # (P,)
    von_mises: NDArray[np.float32]  # (T, P), MPa
    time: NDArray[np.float64]  # (T,), s


def load_case_trajectory(
    h5_path: str | Path,
    *,
    length_scale: float = 1e3,
    stress_scale: float = 1e-6,
) -> CaseTrajectory:
    """Load a canonical case into a :class:`CaseTrajectory` (SPH particles only).

    Parameters
    ----------
    h5_path:
        Path to a canonical ``.h5`` case.
    length_scale:
        Multiplier applied to SI positions (default 1e3: m -> mm).
    stress_scale:
        Multiplier applied to SI stress (default 1e-6: Pa -> MPa).

    Returns
    -------
    CaseTrajectory
    """
    case = read_case(h5_path)
    if case.response is None:
        raise ValueError(f"case {case.metadata.case_id} has no response")
    sph = case.elements["sph"]
    idx = sph.connectivity[:, 0]  # node indices of the SPH particles
    dim = case.metadata.dimension

    coords0 = case.nodes.coords[idx][:, :dim]  # (P, dim) SI
    disp = case.response.node["displacement"][:, idx, :]  # (T, P, dim) SI
    positions = ((coords0[None] + disp) * length_scale).astype(np.float32)

    stress = case.response.element["sph"]["stress"]  # (T, P, 6) Pa
    von_mises = (von_mises_from_voigt(stress) * stress_scale).astype(np.float32)

    return CaseTrajectory(
        case_id=case.metadata.case_id,
        positions=positions,
        particle_type=np.asarray(sph.part_id, dtype=np.int64),
        von_mises=von_mises,
        time=np.asarray(case.response.time, dtype=np.float64),
    )
```

- [ ] **Step 4: Run tests + lint + types**

Run: `.venv/Scripts/python.exe -m pytest tests/datasets/test_canonical.py -v && .venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m mypy`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/structbench/datasets tests/datasets/test_canonical.py
git commit -m "feat(datasets): canonical case -> particle trajectory + von Mises"
```

---

### Task 4: `datasets/normalization` — velocity/acceleration stats

**Files:**
- Create: `src/structbench/datasets/normalization.py`
- Modify: `src/structbench/datasets/__init__.py` (re-export)
- Test: `tests/datasets/test_normalization.py`

**Interfaces:**
- Consumes: `CaseTrajectory` (Task 3).
- Produces:
  - `@dataclass NormalizationStats` with `velocity_mean, velocity_std, acceleration_mean, acceleration_std: NDArray[float64]` each `(dim,)`.
  - `compute_stats(trajectories: list[CaseTrajectory]) -> NormalizationStats` — velocity = first difference of positions over frames; acceleration = second difference; mean/std pooled over all particles, frames, and cases.
  - `NormalizationStats.save(path)` / `NormalizationStats.load(path)` via `.npz`.

- [ ] **Step 1: Write the failing test**

```python
# tests/datasets/test_normalization.py
import numpy as np

from structbench.datasets.canonical import CaseTrajectory
from structbench.datasets.normalization import NormalizationStats, compute_stats


def _const_accel_traj():
    # x(t) = 0.5 * a * t^2 with a=[2,0]; first diff = velocity, second diff = a.
    T, P = 5, 4
    t = np.arange(T)
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pos[:, :, 0] = (0.5 * 2.0 * t**2)[:, None]
    return CaseTrajectory("c", pos, np.ones(P, np.int64),
                          np.zeros((T, P), np.float32), t.astype(np.float64))


def test_compute_stats_constant_acceleration():
    stats = compute_stats([_const_accel_traj()])
    np.testing.assert_allclose(stats.acceleration_mean, [2.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(stats.acceleration_std, [0.0, 0.0], atol=1e-5)
    assert stats.velocity_mean.shape == (2,)


def test_normalization_stats_roundtrip(tmp_path):
    stats = compute_stats([_const_accel_traj()])
    p = tmp_path / "norm.npz"
    stats.save(p)
    back = NormalizationStats.load(p)
    np.testing.assert_array_equal(back.acceleration_mean, stats.acceleration_mean)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/datasets/test_normalization.py -v`
Expected: FAIL (ImportError: normalization).

- [ ] **Step 3: Write the module**

```python
# src/structbench/datasets/normalization.py
"""Velocity/acceleration normalization statistics over a set of trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .canonical import CaseTrajectory


@dataclass
class NormalizationStats:
    """Per-dimension mean/std of velocity and acceleration (mm/frame, mm/frame^2)."""

    velocity_mean: NDArray[np.float64]
    velocity_std: NDArray[np.float64]
    acceleration_mean: NDArray[np.float64]
    acceleration_std: NDArray[np.float64]

    def save(self, path: str | Path) -> None:
        """Write the four arrays to a ``.npz`` file."""
        np.savez(
            path,
            velocity_mean=self.velocity_mean,
            velocity_std=self.velocity_std,
            acceleration_mean=self.acceleration_mean,
            acceleration_std=self.acceleration_std,
        )

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        """Read stats back from a ``.npz`` file written by :meth:`save`."""
        d = np.load(path)
        return cls(
            d["velocity_mean"], d["velocity_std"],
            d["acceleration_mean"], d["acceleration_std"],
        )


def compute_stats(trajectories: list[CaseTrajectory]) -> NormalizationStats:
    """Pool velocity/acceleration stats over all particles, frames, and cases.

    Velocity is the first finite difference of positions along the frame axis;
    acceleration is the second. Statistics are stacked over every particle in
    every frame of every trajectory.
    """
    vels, accs = [], []
    for tr in trajectories:
        p = tr.positions.astype(np.float64)  # (T, P, dim)
        v = p[1:] - p[:-1]  # (T-1, P, dim)
        a = v[1:] - v[:-1]  # (T-2, P, dim)
        vels.append(v.reshape(-1, p.shape[-1]))
        accs.append(a.reshape(-1, p.shape[-1]))
    v_all = np.concatenate(vels, axis=0)
    a_all = np.concatenate(accs, axis=0)
    return NormalizationStats(
        velocity_mean=v_all.mean(0), velocity_std=v_all.std(0),
        acceleration_mean=a_all.mean(0), acceleration_std=a_all.std(0),
    )
```

Add to `src/structbench/datasets/__init__.py`:
```python
from .canonical import CaseTrajectory, load_case_trajectory, von_mises_from_voigt
from .normalization import NormalizationStats, compute_stats

__all__ = [
    "CaseTrajectory", "load_case_trajectory", "von_mises_from_voigt",
    "NormalizationStats", "compute_stats",
]
```

- [ ] **Step 4: Run tests + lint + types**

Run: `.venv/Scripts/python.exe -m pytest tests/datasets/ -v && .venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m mypy`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/structbench/datasets tests/datasets/test_normalization.py
git commit -m "feat(datasets): velocity/acceleration normalization stats"
```

---

### Task 5: `datasets/particle` — windowed training samples + collate

**Files:**
- Create: `src/structbench/datasets/particle.py`
- Modify: `src/structbench/datasets/__init__.py` (re-export)
- Test: `tests/datasets/test_particle.py`

**Interfaces:**
- Consumes: `CaseTrajectory` (Task 3).
- Produces:
  - `WindowDataset(trajectories: list[CaseTrajectory], window: int)` — a `torch.utils.data.Dataset`; `__getitem__` returns a dict with `position_seq: Tensor (P, window, dim)`, `particle_type: Tensor (P,)`, `next_position: Tensor (P, dim)`, `next_aux: Tensor (P,)`, `n_particles: int`.
  - `collate_samples(batch: list[dict]) -> dict` — concatenates particles across examples; adds `n_particles_per_example: LongTensor (B,)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/datasets/test_particle.py
import numpy as np
import torch
from torch.utils.data import DataLoader

from structbench.datasets.canonical import CaseTrajectory
from structbench.datasets.particle import WindowDataset, collate_samples


def _traj(case_id, P, T=6):
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pos[:, :, 0] = np.arange(T)[:, None]  # moves +1 mm/frame in x
    vm = np.zeros((T, P), dtype=np.float32)
    return CaseTrajectory(case_id, pos, np.ones(P, np.int64), vm,
                          np.arange(T, dtype=np.float64))


def test_window_dataset_sample_shapes_and_target():
    ds = WindowDataset([_traj("a", P=5)], window=3)
    # T=6, window=3 -> next index from 3..5 -> 3 samples
    assert len(ds) == 3
    s = ds[0]
    assert s["position_seq"].shape == (5, 3, 2)
    assert s["next_position"].shape == (5, 2)
    # frame 3 position is x=3 for all particles
    torch.testing.assert_close(s["next_position"][:, 0], torch.full((5,), 3.0))


def test_collate_concatenates_particles():
    ds = WindowDataset([_traj("a", 5), _traj("b", 4)], window=3)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_samples, shuffle=False)
    batch = next(iter(loader))
    # two examples with 5 and 4 particles -> 9 rows
    assert batch["position_seq"].shape == (9, 3, 2)
    torch.testing.assert_close(
        batch["n_particles_per_example"], torch.tensor([5, 4])
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/datasets/test_particle.py -v`
Expected: FAIL (ImportError: particle).

- [ ] **Step 3: Write the module**

```python
# src/structbench/datasets/particle.py
"""Windowed autoregressive training samples from particle trajectories.

A sample is a window of ``window`` consecutive positions plus the next
position and next auxiliary value, for every particle in one trajectory. The
collate function concatenates particles across a batch into one big graph, as
the GNS expects, tracking how many particles each example contributed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .canonical import CaseTrajectory


class WindowDataset(Dataset):
    """Autoregressive ``(position_seq, next_position, next_aux)`` samples."""

    def __init__(self, trajectories: list[CaseTrajectory], window: int) -> None:
        self._window = window
        # index: list of (traj, t) where t is the index of the predicted frame.
        self._index: list[tuple[CaseTrajectory, int]] = []
        for tr in trajectories:
            n_frames = tr.positions.shape[0]
            for t in range(window, n_frames):
                self._index.append((tr, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | int]:
        tr, t = self._index[i]
        w = self._window
        seq = tr.positions[t - w : t]  # (window, P, dim)
        seq = np.transpose(seq, (1, 0, 2))  # (P, window, dim)
        return {
            "position_seq": torch.from_numpy(np.ascontiguousarray(seq)),
            "particle_type": torch.from_numpy(tr.particle_type),
            "next_position": torch.from_numpy(tr.positions[t]),
            "next_aux": torch.from_numpy(tr.von_mises[t]),
            "n_particles": int(tr.positions.shape[1]),
        }


def collate_samples(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Concatenate per-example particle rows into one batched graph."""
    return {
        "position_seq": torch.cat([b["position_seq"] for b in batch], dim=0),
        "particle_type": torch.cat([b["particle_type"] for b in batch], dim=0),
        "next_position": torch.cat([b["next_position"] for b in batch], dim=0),
        "next_aux": torch.cat([b["next_aux"] for b in batch], dim=0),
        "n_particles_per_example": torch.tensor(
            [b["n_particles"] for b in batch], dtype=torch.long
        ),
    }
```

Extend `datasets/__init__.py` `__all__` and imports with `WindowDataset, collate_samples`.

- [ ] **Step 4: Run tests + lint + types**

Run: `.venv/Scripts/python.exe -m pytest tests/datasets/ -v && .venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m mypy`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/structbench/datasets tests/datasets/test_particle.py
git commit -m "feat(datasets): windowed training samples + graph-batch collate"
```

---

### Task 6: `models/gns/graph_network` — EncodeProcessDecode (port)

**Files:**
- Create: `src/structbench/models/__init__.py`
- Create: `src/structbench/models/gns/__init__.py`
- Create: `src/structbench/models/gns/graph_network.py`
- Test: `tests/models/gns/test_graph_network.py`

**Port instruction:** copy `../code/sgnn/sgnn/single_scale/graph_network.py` into the target file **verbatim** (it is already general — `build_mlp`, `Encoder`, `InteractionNetwork`, `Processor`, `Decoder`, `EncodeProcessDecode`). Then: add a module docstring; convert two-space indentation to four-space to satisfy ruff; keep all class/method names. Do **not** change behaviour. The `Processor` uses `aggr='max'` at the stack level and `InteractionNetwork` uses `aggr='add'` — preserve both exactly.

**Interfaces:**
- Produces: `EncodeProcessDecode(nnode_in_features, nnode_out_features, nedge_in_features, latent_dim, nmessage_passing_steps, nmlp_layers, mlp_hidden_dim)` with `forward(x, edge_index, edge_features) -> Tensor (N, nnode_out_features)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/gns/test_graph_network.py
import torch

from structbench.models.gns.graph_network import EncodeProcessDecode


def test_encode_process_decode_output_shape_and_finite():
    n, e = 6, 10
    net = EncodeProcessDecode(
        nnode_in_features=7, nnode_out_features=3, nedge_in_features=3,
        latent_dim=16, nmessage_passing_steps=2, nmlp_layers=1, mlp_hidden_dim=16,
    )
    x = torch.randn(n, 7)
    edge_index = torch.randint(0, n, (2, e))
    edge_features = torch.randn(e, 3)
    out = net(x, edge_index, edge_features)
    assert out.shape == (n, 3)
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/models/gns/test_graph_network.py -v`
Expected: FAIL (ModuleNotFoundError: structbench.models).

- [ ] **Step 3: Create `models/__init__.py`, `models/gns/__init__.py`, and the ported `graph_network.py`**

```python
# src/structbench/models/__init__.py
"""Reference ML models (ARCHITECTURE.md)."""
```
```python
# src/structbench/models/gns/__init__.py
"""Single-scale Graph Network Simulator (ported from the sgnn reference)."""

from .graph_network import EncodeProcessDecode

__all__ = ["EncodeProcessDecode"]
```
Port `graph_network.py` per the **Port instruction** above (verbatim logic, 4-space indent, module docstring, type hints already present).

- [ ] **Step 4: Run tests + lint + types**

Run: `.venv/Scripts/python.exe -m pytest tests/models/gns/test_graph_network.py -v && .venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m mypy`
Expected: PASS / clean. (If mypy flags untyped torch internals, add precise hints; do not silence.)

- [ ] **Step 5: Commit**

```bash
git add src/structbench/models tests/models
git commit -m "feat(models): port GNS encode-process-decode network"
```

---

### Task 7: `models/gns/simulator` — LearnedSimulator (port + generalise)

**Files:**
- Create: `src/structbench/models/gns/simulator.py`
- Modify: `src/structbench/models/gns/__init__.py` (re-export)
- Test: `tests/models/gns/test_simulator.py`

**Port instruction:** adapt `../code/sgnn/sgnn/single_scale/learned_simulator.py`. Keep `_compute_graph_connectivity`, `_encoder_preprocessor`, `_decoder_postprocessor`, `predict_positions`, `predict_accelerations`, `_inverse_decoder_postprocessor`, and `time_diff` behaviour. **Three changes:**
1. Drop the hardcoded wall-distance block in `_encoder_preprocessor`. Instead accept `boundary_feature_fn: Callable[[Tensor], Tensor] | None = None` in `__init__`; in `_encoder_preprocessor`, if set, append `boundary_feature_fn(most_recent_position)` to `node_features`.
2. Make the auxiliary width explicit: `n_aux: int` in `__init__`; decoder out features `= particle_dimensions + n_aux`; in `predict_positions`/`predict_accelerations`, `predicted_aux = pred[:, particle_dimensions:]` (shape `(N, n_aux)`).
3. Delete the debug/`print` methods (`_test_graph_connectivity`, `test_graph_connectivity_once`) — not allowed in library code.

**Interfaces:**
- Consumes: `EncodeProcessDecode` (Task 6).
- Produces: `LearnedSimulator(particle_dimensions, nnode_in, nedge_in, latent_dim, nmessage_passing_steps, nmlp_layers, mlp_hidden_dim, connectivity_radius, normalization_stats, nparticle_types, particle_type_embedding_size, *, n_aux=1, boundary_feature_fn=None, device="cpu")` with:
  - `predict_positions(position_sequence, nparticles_per_example, particle_types) -> tuple[Tensor (N,dim), Tensor (N,n_aux)]`
  - `predict_accelerations(next_positions, position_sequence_noise, position_sequence, nparticles_per_example, particle_types) -> tuple[pred_acc (N,dim), target_acc (N,dim), pred_aux (N,n_aux)]`
  - `normalization_stats`: dict `{"velocity": {"mean": Tensor, "std": Tensor}, "acceleration": {...}}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/gns/test_simulator.py
import torch

from structbench.models.gns.simulator import LearnedSimulator


def _stats(dim=2):
    z, o = torch.zeros(dim), torch.ones(dim)
    return {"velocity": {"mean": z, "std": o},
            "acceleration": {"mean": z, "std": o}}


def _sim(n_aux=1, boundary_feature_fn=None, window=3):
    nnode_in = (window - 1) * 2  # velocities only; +embedding handled internally
    return LearnedSimulator(
        particle_dimensions=2, nnode_in=nnode_in, nedge_in=3, latent_dim=16,
        nmessage_passing_steps=2, nmlp_layers=1, mlp_hidden_dim=16,
        connectivity_radius=5.0, normalization_stats=_stats(),
        nparticle_types=1, particle_type_embedding_size=4,
        n_aux=n_aux, boundary_feature_fn=boundary_feature_fn, device="cpu",
    )


def test_predict_positions_shapes():
    sim = _sim(n_aux=1)
    P, window = 4, 3
    pos_seq = torch.randn(P, window, 2)
    npp = torch.tensor([P])
    ptype = torch.zeros(P, dtype=torch.long)
    next_pos, aux = sim.predict_positions(pos_seq, npp, ptype)
    assert next_pos.shape == (P, 2)
    assert aux.shape == (P, 1)


def test_boundary_feature_fn_changes_node_input_width():
    # With a boundary fn adding 1 feature, the encoder must accept nnode_in+1.
    def wall(pos):  # (P, dim) -> (P, 1)
        return pos[:, 0:1].clamp(min=0.0, max=5.0)
    sim = LearnedSimulator(
        particle_dimensions=2, nnode_in=(3 - 1) * 2 + 1, nedge_in=3, latent_dim=16,
        nmessage_passing_steps=1, nmlp_layers=1, mlp_hidden_dim=16,
        connectivity_radius=5.0, normalization_stats=_stats(),
        nparticle_types=1, particle_type_embedding_size=4,
        n_aux=1, boundary_feature_fn=wall, device="cpu",
    )
    out, _ = sim.predict_positions(torch.randn(4, 3, 2), torch.tensor([4]),
                                   torch.zeros(4, dtype=torch.long))
    assert out.shape == (4, 2)
```

Note: `nnode_in` passed to `__init__` must equal `(window-1)*dim + (n_boundary_features) + (particle_type_embedding_size if nparticle_types>1 else 0)`. The caller (Task 10) computes it; here the tests pass matching values.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/models/gns/test_simulator.py -v`
Expected: FAIL (ImportError: simulator).

- [ ] **Step 3: Write `simulator.py`** per the **Port instruction**. Re-export in `models/gns/__init__.py`:
```python
from .graph_network import EncodeProcessDecode
from .simulator import LearnedSimulator

__all__ = ["EncodeProcessDecode", "LearnedSimulator"]
```

- [ ] **Step 4: Run tests + lint + types** (whole suite)

Run: `.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m mypy`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/structbench/models tests/models/gns/test_simulator.py
git commit -m "feat(models): GNS learned simulator (generalised wall feature + n_aux)"
```

---

### Task 8: `eval/metrics` — RMSE + QoIs

**Files:**
- Create: `src/structbench/eval/__init__.py`
- Create: `src/structbench/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Produces (NumPy in, float/array out):
  - `position_rmse(pred: NDArray, true: NDArray) -> NDArray` — inputs `(T, P, dim)`, returns per-frame RMSE `(T,)` (sqrt of mean over particles and dims).
  - `field_rmse(pred: NDArray, true: NDArray) -> NDArray` — inputs `(T, P)`, returns `(T,)`.
  - `final_length(positions: NDArray) -> float` — `x`-extent of the last frame `(P, dim)`-slice; takes `(T,P,dim)`, uses `[-1]`.
  - `mushroom_width(positions: NDArray) -> float` — `y`-extent of the last frame.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_metrics.py
import numpy as np

from structbench.eval.metrics import (
    field_rmse, final_length, mushroom_width, position_rmse,
)


def test_position_rmse_per_frame():
    true = np.zeros((2, 3, 2))
    pred = np.zeros((2, 3, 2))
    pred[1] = 3.0  # every component off by 3 at frame 1 -> rmse sqrt(mean(9))=3
    out = position_rmse(pred, true)
    np.testing.assert_allclose(out, [0.0, 3.0])


def test_field_rmse_per_frame():
    true = np.zeros((2, 4))
    pred = np.array([[0, 0, 0, 0], [2, 2, 2, 2]], dtype=float)
    np.testing.assert_allclose(field_rmse(pred, true), [0.0, 2.0])


def test_qois_use_last_frame_extents():
    pos = np.zeros((1, 4, 2))
    pos[0, :, 0] = [0, 10, 5, 5]   # x extent 10
    pos[0, :, 1] = [-3, 3, 0, 0]   # y extent 6
    assert final_length(pos) == 10.0
    assert mushroom_width(pos) == 6.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/eval/test_metrics.py -v`
Expected: FAIL (ModuleNotFoundError: structbench.eval).

- [ ] **Step 3: Write the module**

```python
# src/structbench/eval/__init__.py
"""Evaluation: autoregressive rollout and benchmark metrics."""
```

```python
# src/structbench/eval/metrics.py
"""Rollout metrics for the Taylor benchmark (ADR-0019)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def position_rmse(pred: NDArray, true: NDArray) -> NDArray[np.float64]:
    """Per-frame position RMSE over particles and dimensions.

    Parameters
    ----------
    pred, true:
        Arrays of shape ``(T, P, dim)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(T,)``.
    """
    d = (np.asarray(pred, float) - np.asarray(true, float)) ** 2
    return np.sqrt(d.mean(axis=(1, 2)))


def field_rmse(pred: NDArray, true: NDArray) -> NDArray[np.float64]:
    """Per-frame RMSE of a scalar per-particle field, shapes ``(T, P)``."""
    d = (np.asarray(pred, float) - np.asarray(true, float)) ** 2
    return np.sqrt(d.mean(axis=1))


def final_length(positions: NDArray) -> float:
    """x-extent of the final frame. ``positions`` is ``(T, P, dim)``."""
    last = np.asarray(positions, float)[-1]
    return float(last[:, 0].ptp())


def mushroom_width(positions: NDArray) -> float:
    """y-extent of the final frame. ``positions`` is ``(T, P, dim)``."""
    last = np.asarray(positions, float)[-1]
    return float(last[:, 1].ptp())
```

- [ ] **Step 4: Run tests + lint + types**

Run: `.venv/Scripts/python.exe -m pytest tests/eval/test_metrics.py -v && .venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m mypy`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/structbench/eval tests/eval/test_metrics.py
git commit -m "feat(eval): rollout metrics (position/field RMSE, QoIs)"
```

---

### Task 9: `eval/rollout` — autoregressive rollout

**Files:**
- Create: `src/structbench/eval/rollout.py`
- Modify: `src/structbench/eval/__init__.py` (re-export)
- Test: `tests/eval/test_rollout.py`

**Interfaces:**
- Consumes: a simulator exposing `predict_positions(position_sequence, nparticles_per_example, particle_types) -> (next_pos, aux)` (Task 7); `CaseTrajectory` (Task 3); metrics (Task 8).
- Produces:
  - `@dataclass RolloutResult` with `predicted_positions: NDArray (T,P,dim)`, `predicted_aux: NDArray (T,P)`, `position_rmse: NDArray (nsteps,)`, `aux_rmse: NDArray (nsteps,)`.
  - `rollout(simulator, trajectory: CaseTrajectory, window: int, device: str = "cpu") -> RolloutResult` — seed with the first `window` ground-truth frames, autoregress to the end.

- [ ] **Step 1: Write the failing test** (a stub simulator with a known constant-velocity rule)

```python
# tests/eval/test_rollout.py
import numpy as np
import torch

from structbench.datasets.canonical import CaseTrajectory
from structbench.eval.rollout import RolloutResult, rollout


class _ConstVelSim:
    """Predicts next = last + (last - prev): perfect constant-velocity motion."""

    def predict_positions(self, position_sequence, nparticles_per_example, particle_types):
        last = position_sequence[:, -1]
        prev = position_sequence[:, -2]
        nxt = last + (last - prev)
        aux = torch.zeros(position_sequence.shape[0], 1)
        return nxt, aux


def test_rollout_is_exact_for_constant_velocity():
    T, P = 6, 4
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pos[:, :, 0] = np.arange(T)[:, None]  # const velocity +1 in x
    traj = CaseTrajectory("a", pos, np.ones(P, np.int64),
                          np.zeros((T, P), np.float32), np.arange(T, dtype=float))
    res = rollout(_ConstVelSim(), traj, window=3)
    assert isinstance(res, RolloutResult)
    assert res.predicted_positions.shape == (T, P, 2)
    np.testing.assert_allclose(res.predicted_positions, pos, atol=1e-5)
    np.testing.assert_allclose(res.position_rmse, 0.0, atol=1e-5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/eval/test_rollout.py -v`
Expected: FAIL (ImportError: rollout).

- [ ] **Step 3: Write the module**

```python
# src/structbench/eval/rollout.py
"""Autoregressive rollout of a learned simulator over a trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from ..datasets.canonical import CaseTrajectory
from .metrics import field_rmse, position_rmse


@dataclass
class RolloutResult:
    """Predicted trajectory and per-step error against ground truth."""

    predicted_positions: NDArray[np.float32]  # (T, P, dim)
    predicted_aux: NDArray[np.float32]  # (T, P)
    position_rmse: NDArray[np.float64]  # (nsteps,)
    aux_rmse: NDArray[np.float64]  # (nsteps,)


def rollout(
    simulator: object,
    trajectory: CaseTrajectory,
    window: int,
    device: str = "cpu",
) -> RolloutResult:
    """Seed with ``window`` ground-truth frames, then autoregress to the end.

    Parameters
    ----------
    simulator:
        Object with
        ``predict_positions(position_sequence, nparticles_per_example,
        particle_types) -> (next_positions (P,dim), aux (P,n_aux))``.
    trajectory:
        Ground-truth :class:`CaseTrajectory`.
    window:
        History length (frames used to seed and to predict each step).
    device:
        Torch device string.

    Returns
    -------
    RolloutResult
    """
    pos = torch.from_numpy(trajectory.positions).to(device)  # (T, P, dim)
    n_frames, n_particles, _ = pos.shape
    ptype = torch.from_numpy(trajectory.particle_type).to(device)
    npp = torch.tensor([n_particles], device=device)

    seq = pos[:window].clone()  # (window, P, dim)
    predicted = [pos[i] for i in range(window)]
    aux_pred = [torch.zeros(n_particles, device=device) for _ in range(window)]

    with torch.no_grad():
        for _ in range(window, n_frames):
            seq_pw = seq.permute(1, 0, 2).contiguous()  # (P, window, dim)
            next_pos, aux = simulator.predict_positions(seq_pw, npp, ptype)
            predicted.append(next_pos)
            aux_pred.append(aux[:, 0])
            seq = torch.cat([seq[1:], next_pos[None]], dim=0)

    pred_pos = torch.stack(predicted, dim=0).cpu().numpy().astype(np.float32)
    pred_aux = torch.stack(aux_pred, dim=0).cpu().numpy().astype(np.float32)
    return RolloutResult(
        predicted_positions=pred_pos,
        predicted_aux=pred_aux,
        position_rmse=position_rmse(pred_pos[window:], trajectory.positions[window:]),
        aux_rmse=field_rmse(pred_aux[window:], trajectory.von_mises[window:]),
    )
```

Re-export in `eval/__init__.py`:
```python
from .metrics import field_rmse, final_length, mushroom_width, position_rmse
from .rollout import RolloutResult, rollout

__all__ = [
    "position_rmse", "field_rmse", "final_length", "mushroom_width",
    "RolloutResult", "rollout",
]
```

- [ ] **Step 4: Run tests + lint + types**

Run: `.venv/Scripts/python.exe -m pytest tests/eval/ -v && .venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m mypy`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/structbench/eval tests/eval/test_rollout.py
git commit -m "feat(eval): autoregressive rollout"
```

---

### Task 10: `cli/train` — config + training/validation/rollout entry

**Files:**
- Create: `src/structbench/cli/__init__.py`
- Create: `src/structbench/cli/train.py`
- Test: `tests/cli/test_train_config.py`

**Port instruction:** adapt the training loop from `../code/sgnn/sgnn/single_scale/train.py` and the noise helper from `../code/sgnn/sgnn/noise_utils.py` (random-walk position noise). Replace its npz/metadata data path with the canonical pipeline: build train trajectories via `load_case_trajectory` over `benchmarks.taylor_impact_2d.TRAIN`, `WindowDataset` + `collate_samples`, and `compute_stats` for normalization. Build the simulator with the Taylor `wall_distance_feature` as `boundary_feature_fn` and `n_aux=1`. Keep: GNS noise, Adam, exp LR decay, dual MSE loss (`w_pos*||Δacc||² + w_aux*(Δaux)²`), periodic validation rollout over `VAL`, checkpoint-best.

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `@dataclass GNSConfig` (defaults: `window=11`, `connectivity_radius=0.6`, `hidden_dim=64`, `message_passing_steps=5`, `nmlp_layers=1`, `particle_type_embedding_size=9`, `noise_std=0.02`, `dim=2`).
  - `@dataclass TrainConfig` (defaults: `batch_size=32`, `lr_init=1e-3`, `lr_decay=0.1`, `lr_decay_steps=30000`, `training_steps=100000`, `val_every=2000`, `w_pos=1.0`, `w_aux=1.0`) + `TrainConfig.from_toml(path) -> TrainConfig`.
  - `build_simulator(stats, gns: GNSConfig, *, n_particle_types, boundary_feature_fn, device) -> LearnedSimulator` (computes `nnode_in`).
  - `main(argv=None) -> int` (argparse: `--mode {train,valid,rollout}`, `--config`, `--out`, `--data-root`).

- [ ] **Step 1: Write the failing test** (config + nnode_in arithmetic only — no training)

```python
# tests/cli/test_train_config.py
import torch

from structbench.cli.train import GNSConfig, TrainConfig, build_simulator


def test_train_config_from_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("batch_size = 8\nlr_init = 0.0005\n", encoding="utf-8")
    cfg = TrainConfig.from_toml(p)
    assert cfg.batch_size == 8 and cfg.lr_init == 0.0005
    assert cfg.training_steps == 100000  # default preserved


def test_build_simulator_node_input_width():
    gns = GNSConfig()
    stats = {"velocity": {"mean": torch.zeros(2), "std": torch.ones(2)},
             "acceleration": {"mean": torch.zeros(2), "std": torch.ones(2)}}
    sim = build_simulator(
        stats, gns, n_particle_types=2,
        boundary_feature_fn=lambda p: p[:, 0:1], device="cpu",
    )
    # (window-1)*dim + 1 boundary + embedding(9) = 10*2 + 1 + 9 = 30
    out, aux = sim.predict_positions(
        torch.randn(5, gns.window, 2), torch.tensor([5]),
        torch.zeros(5, dtype=torch.long),
    )
    assert out.shape == (5, 2) and aux.shape == (5, 1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/cli/test_train_config.py -v`
Expected: FAIL (ModuleNotFoundError: structbench.cli).

- [ ] **Step 3: Write `cli/train.py`** per the **Port instruction**, including the `GNSConfig`/`TrainConfig` dataclasses, `build_simulator` (computing `nnode_in = (window-1)*dim + n_boundary + (embedding if n_particle_types>1 else 0)` where `n_boundary=1` when a boundary fn is given), the random-walk noise helper, the training loop, validation rollout, and `main`. Use `logging`, not `print`, in library functions; `main` may print progress.

- [ ] **Step 4: Run tests + lint + types** (full suite)

Run: `.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m mypy`
Expected: PASS / clean.

- [ ] **Step 5: Commit**

```bash
git add src/structbench/cli tests/cli
git commit -m "feat(cli): config-driven GNS training/validation/rollout entry"
```

---

### Task 11: Integration smoke check + docs

**Files:**
- Create: `tests/test_pipeline_smoke.py` (CPU, tiny, no real data)
- Modify: `ARCHITECTURE.md` (data-flow note for the ML layer)
- Modify: `pyproject.toml` (`[project.scripts]` entry `structbench-train = "structbench.cli.train:main"`)

- [ ] **Step 1: Write a CPU smoke test** that wires datasets → simulator → rollout on a tiny synthetic trajectory (3–5 particles, 6 frames, window 3), asserting one training step reduces or computes a finite loss and a rollout returns the right shapes. (Use the public APIs; no GPU, no files.)

```python
# tests/test_pipeline_smoke.py
import numpy as np
import torch

from structbench.datasets.canonical import CaseTrajectory
from structbench.datasets.normalization import compute_stats
from structbench.eval.rollout import rollout


def _tiny_traj(P=5, T=6):
    pos = np.zeros((T, P, 2), dtype=np.float32)
    pos[:, :, 0] = np.arange(T)[:, None]
    pos[:, :, 1] = np.linspace(0, 1, P)[None, :]
    return CaseTrajectory("a", pos, np.ones(P, np.int64),
                          np.zeros((T, P), np.float32), np.arange(T, float))


def test_stats_and_rollout_shapes_compose():
    traj = _tiny_traj()
    stats = compute_stats([traj])
    assert stats.acceleration_mean.shape == (2,)
    # constant-velocity stub stands in for a trained model here
    class _Stub:
        def predict_positions(self, seq, npp, pt):
            nxt = seq[:, -1] + (seq[:, -1] - seq[:, -2])
            return nxt, torch.zeros(seq.shape[0], 1)
    res = rollout(_Stub(), traj, window=3)
    assert res.predicted_positions.shape == (6, 5, 2)
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python.exe -m pytest tests/test_pipeline_smoke.py -v`
Expected: PASS.

- [ ] **Step 3: Add the ARCHITECTURE.md data-flow note** (one short paragraph under `datasets/`): canonical case → `CaseTrajectory` (mm/MPa) → windowed samples / normalization → GNS simulator → rollout + metrics; benchmark supplies split + boundary feature.

- [ ] **Step 4: Add the console-script entry to `pyproject.toml`**
```toml
[project.scripts]
structbench-train = "structbench.cli.train:main"
```

- [ ] **Step 5: Full check + commit**

Run: `.venv/Scripts/python.exe -m pytest -q && .venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m mypy`
Expected: all green.
```bash
git add tests/test_pipeline_smoke.py ARCHITECTURE.md pyproject.toml
git commit -m "test: end-to-end pipeline smoke + ML data-flow docs"
```

---

## Manual validation (after the plan, not a test)

On GPU in the `.venv` (or a torch+CUDA env), run a short real training to confirm the loss decreases and a rollout runs against a held-out case:

```
.venv/Scripts/python.exe -m structbench.cli.train --mode train --out ./runs/smoke \
    --data-root "../data/2D-Copper-Bar-Taylor-Impact/h5_canonical" \
    # with a tiny override config: training_steps=200, batch_size=4
```
This is a developer check, not part of the CI suite (PRINCIPLES.md: no GPU/large data in tests).

---

## Self-review notes

- **Spec coverage:** benchmarks/datasets/models/eval/cli all covered (Tasks 2–10); deps (Task 1); torch-free-core guard (Task 1); units mm/MPa (Task 3); von Mises aux (Tasks 3, 9); split (Task 2); normalization with noise combination (Tasks 4, 10); rollout + QoIs (Tasks 8, 9); testing strategy honored throughout. Resolved-choices items: config dataclass+TOML (Task 10), norm cache (`NormalizationStats.save/load`, Task 4), run outputs `--out` (Task 10).
- **Deferred (non-goals, explicitly out of scope):** `effective_plastic_strain` second aux head; checkpoint registry/publishing; multi-scale GNN; other datasets.
- **Type consistency:** `predict_positions`/`predict_accelerations` signatures match between Tasks 7, 9, 10; `CaseTrajectory` fields used identically in Tasks 3–11; `n_particles_per_example` naming consistent (Tasks 5, 7, 9).
