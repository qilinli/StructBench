# Benchmark Registry and Cards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the training pipeline benchmark-generic (registry + spec, replacing the hard-wired Taylor imports in `cli/train.py`) and introduce the ADR-0025 benchmark-card convention with generated views, retrofitting Taylor.

**Architecture:** Each benchmark module exposes a frozen `BenchmarkSpec` (splits, aux field, QoIs, boundary feature, card) resolved by name through `structbench.benchmarks.get_benchmark()`. The `datasets/` loader's auxiliary channel becomes selectable by field name (the `CaseTrajectory.von_mises` attribute is renamed `aux`). Cards render to `docs/benchmarks.md` and per-archive README/`card.json` via `benchmarks/render.py` + a thin `tools/` script; a drift test keeps the committed index current.

**Tech Stack:** Python 3.11+, stdlib only for new code (dataclasses, importlib, json); existing deps numpy/h5py/torch untouched. No new dependencies.

**Plan 1 of 3 for v0.2** (ADRs 0022–0025). Plan 2 (wave-1d benchmark end-to-end) and Plan 3 (notch-beam pair) build on this one and are written after it lands.

## Global Constraints

- Python floor **3.11** (`requires-python = ">=3.11"`); ruff line length **88**; mypy `disallow_untyped_defs = true`.
- **No new dependencies** (runtime or dev) — stdlib solutions only (PRINCIPLES.md dependency policy).
- **NumPy-style docstrings** on every public API; module `__init__.py` docstrings state responsibility.
- `_`-prefixed symbols are **private across module boundaries** (ARCHITECTURE.md interface discipline).
- Dependency graph: `benchmarks/` may import `core/` and `datasets/`; `cli/` may import everything. `benchmarks/` importing `eval/` for `QoiFn` is a **pre-existing deviation** (`taylor_impact_2d/benchmark.py:7`) — follow it, do not expand it beyond type/QoI imports, and do not "fix" it in this plan.
- Tests: pytest, synthetic data only, deterministic, no network/solver/large files. The one data-gated test (Task 7) must skip cleanly when `STRUCTBENCH_DATA_ROOT` is unset.
- Library code logs via `logging.getLogger(__name__)`; `print` only in `cli/` and `tools/`.
- Commits: Conventional Commits, imperative mood, `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.
- Branch: `feat/benchmark-registry-cards`, branched from `main` if `adr/v02-benchmark-scope` has been merged, else from `adr/v02-benchmark-scope`. Never commit to `main`.
- Verification before completion: `python -m pytest -q`, `ruff check src tests tools`, `ruff format --check src tests tools`, `python -m mypy src` — all clean. Interpreter: `C:\Users\272766h\AppData\Local\miniconda3\envs\structbench\python.exe`.
- **Split lists, QoI sets, and aux fields are ADR-frozen contract** (ADR-0019/0023/0024): this plan must not change any Taylor split/QoI value, only relocate access to them.

## File Structure

```
src/structbench/benchmarks/
  __init__.py            # MODIFY: docstring + re-export BenchmarkCard, BenchmarkSpec,
                         #         get_benchmark, available_benchmarks
  card.py                # CREATE: BenchmarkCard dataclass (ADR-0025)
  registry.py            # CREATE: BenchmarkSpec dataclass + name registry
  render.py              # CREATE: card → markdown/json renderers
  taylor_impact_2d/
    __init__.py          # MODIFY: build and export SPEC (+ keep existing exports)
    benchmark.py         # UNCHANGED (splits/QoIs are ADR-frozen)
    card.py              # CREATE: Taylor CARD instance
src/structbench/datasets/
  canonical.py           # MODIFY: aux-extractor registry; CaseTrajectory.von_mises → .aux;
                         #         load_case_trajectory(aux_field=...)
  particle.py            # MODIFY: read .aux instead of .von_mises
  normalization.py       # MODIFY: read .aux instead of .von_mises
  __init__.py            # MODIFY: export available_aux_fields
src/structbench/eval/
  rollout.py             # MODIFY: ground-truth aux from trajectory.aux
src/structbench/cli/
  train.py               # MODIFY: TrainConfig.benchmark; spec threading; generic metrics keys
configs/
  taylor_2d.toml         # MODIFY: add `benchmark = "taylor_impact_2d"`
  taylor_2d_smoke.toml   # MODIFY: same
tools/
  gen_benchmark_docs.py  # CREATE: thin CLI over benchmarks/render.py
docs/
  benchmarks.md          # CREATE: generated index (committed)
  ARCHITECTURE.md        # MODIFY: benchmarks/ + datasets/ paragraphs (Task 8)
tests/
  benchmarks/test_card.py       # CREATE
  benchmarks/test_taylor_card.py# CREATE
  benchmarks/test_registry.py   # CREATE
  benchmarks/test_render.py     # CREATE
  benchmarks/test_card_data.py  # CREATE (env-gated)
  datasets/test_canonical.py    # MODIFY (rename + aux_field tests)
  datasets/test_normalization.py# MODIFY (mechanical rename)
  datasets/test_particle.py     # MODIFY (mechanical rename)
  eval/test_rollout.py          # MODIFY (mechanical rename)
  cli/test_train_config.py      # MODIFY (benchmark field test)
```

---

### Task 1: `BenchmarkCard` dataclass

**Files:**
- Create: `src/structbench/benchmarks/card.py`
- Test: `tests/benchmarks/test_card.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces: `BenchmarkCard` frozen dataclass with fields exactly as in the code below, `Discretisation = Literal["SPH", "FEM", "coupled"]`, and method `to_json_dict(self) -> dict[str, object]`. Constructor raises `ValueError` when `n_cases != sum(splits.values())`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/benchmarks/test_card.py
"""BenchmarkCard invariants (ADR-0025)."""

import json

import pytest

from structbench.benchmarks.card import BenchmarkCard


def _kwargs(**overrides):
    base = dict(
        name="Demo-Bench",
        version="0.1",
        description="A demo benchmark.",
        provenance="Synthetic, for tests.",
        data_license="CC BY 4.0",
        solver="LS-DYNA",
        discretisation="SPH",
        materials=("*MAT_ELASTIC",),
        erosion=False,
        loading="rigid-wall impact",
        source_units="g-mm-ms",
        geometry="2D bar",
        n_cases=3,
        splits={"train": 2, "val": 1},
        task="autoregressive transition",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=("final_length",),
        fields=("positions", "stress"),
        particles_per_case="100",
        n_frames=10,
        output_dt_ms=0.1,
    )
    base.update(overrides)
    return base


def test_card_accepts_consistent_splits():
    card = BenchmarkCard(**_kwargs())
    assert card.n_cases == 3
    assert card.size_gb is None


def test_card_rejects_split_sum_mismatch():
    with pytest.raises(ValueError, match="n_cases"):
        BenchmarkCard(**_kwargs(n_cases=99))


def test_card_json_dict_serializes():
    card = BenchmarkCard(**_kwargs())
    payload = json.dumps(card.to_json_dict())
    assert "Demo-Bench" in payload
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/benchmarks/test_card.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'structbench.benchmarks.card'`

- [ ] **Step 3: Write the implementation**

```python
# src/structbench/benchmarks/card.py
"""Typed benchmark card: descriptive metadata for one benchmark (ADR-0025).

Physics facts are declared by hand; ML statistics are computed from the
owning module's split constants (``len(TRAIN)`` etc.) so the card and the
benchmark cannot disagree. Stats that live only in the data (particle
counts, frames, size on disk) are validated by an environment-gated test
when a data root is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Discretisation = Literal["SPH", "FEM", "coupled"]


@dataclass(frozen=True)
class BenchmarkCard:
    """Descriptive metadata for one benchmark (ADR-0025).

    Parameters
    ----------
    name, version, description, provenance, data_license : str
        Identity block: leaderboard name, benchmark version, one-line
        description, data provenance (paper / who ran the simulations),
        and the data license.
    solver, loading, source_units, geometry : str
        Physics block, for the structural engineer. ``source_units`` is
        the solver's unit convention (e.g. ``"g-mm-ms"``); canonical
        storage is SI regardless (ADR-0012).
    discretisation : {"SPH", "FEM", "coupled"}
        Spatial discretisation of the source simulations.
    materials : tuple of str
        Solver material models, verbatim keyword names.
    erosion : bool
        Whether the source simulations delete elements.
    n_cases : int
        Benchmark cases across all splits (held-aside cases excluded).
    splits : dict of str to int
        Split sizes; must sum to ``n_cases``.
    task, aux_field, aux_unit : str
        ML block: the learning task, the auxiliary target's canonical
        field name, and its reporting unit.
    qois, fields : tuple of str
        QoI names and the response fields available in the canonical data.
    particles_per_case : str
        Human-readable particle-count range (e.g. ``"4804-8004"``).
    n_frames : int
        Response frames per case.
    output_dt_ms : float
        Output interval of the source simulations, milliseconds.
    size_gb : float or None
        Canonical dataset size on disk; ``None`` until measured.

    Raises
    ------
    ValueError
        If ``splits`` does not sum to ``n_cases``.
    """

    # identity
    name: str
    version: str
    description: str
    provenance: str
    data_license: str
    # physics — for the structural engineer
    solver: str
    discretisation: Discretisation
    materials: tuple[str, ...]
    erosion: bool
    loading: str
    source_units: str
    geometry: str
    # ml — for the ML researcher
    n_cases: int
    splits: dict[str, int]
    task: str
    aux_field: str
    aux_unit: str
    qois: tuple[str, ...]
    fields: tuple[str, ...]
    particles_per_case: str
    n_frames: int
    output_dt_ms: float
    size_gb: float | None = None

    def __post_init__(self) -> None:
        total = sum(self.splits.values())
        if self.n_cases != total:
            raise ValueError(
                f"n_cases ({self.n_cases}) != sum of splits ({total})"
            )

    def to_json_dict(self) -> dict[str, object]:
        """Return a plain, JSON-serializable dict of the card."""
        return asdict(self)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/benchmarks/test_card.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/structbench/benchmarks/card.py tests/benchmarks/test_card.py
git commit -m "feat: BenchmarkCard dataclass (ADR-0025)"
```

---

### Task 2: Taylor card

**Files:**
- Create: `src/structbench/benchmarks/taylor_impact_2d/card.py`
- Test: `tests/benchmarks/test_taylor_card.py`

**Interfaces:**
- Consumes: `BenchmarkCard` (Task 1); `TRAIN`, `VAL`, `TEST_INTERP`, `TEST_EXTRAP`, `AUX_FIELD`, `QOIS` from `taylor_impact_2d.benchmark`.
- Produces: module constant `CARD: BenchmarkCard` in `structbench.benchmarks.taylor_impact_2d.card`.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_taylor_card.py
"""The Taylor card's ML stats are computed from the split constants."""

from structbench.benchmarks.taylor_impact_2d.benchmark import (
    AUX_FIELD,
    QOIS,
    TEST_EXTRAP,
    TEST_INTERP,
    TRAIN,
    VAL,
)
from structbench.benchmarks.taylor_impact_2d.card import CARD


def test_ml_stats_are_computed_from_split_constants():
    assert CARD.n_cases == len(TRAIN) + len(VAL) + len(TEST_INTERP) + len(TEST_EXTRAP)
    assert CARD.splits == {
        "train": len(TRAIN),
        "val": len(VAL),
        "test_interp": len(TEST_INTERP),
        "test_extrap": len(TEST_EXTRAP),
    }
    assert CARD.aux_field == AUX_FIELD
    assert set(CARD.qois) == set(QOIS)


def test_physics_facts_match_adr_0019():
    assert CARD.discretisation == "SPH"
    assert CARD.erosion is False
    assert CARD.source_units == "g-mm-ms"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/benchmarks/test_taylor_card.py -v`
Expected: FAIL — no module `taylor_impact_2d.card`

- [ ] **Step 3: Write the implementation**

```python
# src/structbench/benchmarks/taylor_impact_2d/card.py
"""Benchmark card for the Taylor 2D impact benchmark (ADR-0025)."""

from ..card import BenchmarkCard
from .benchmark import AUX_FIELD, QOIS, TEST_EXTRAP, TEST_INTERP, TRAIN, VAL

CARD = BenchmarkCard(
    name="Taylor2D-Impact",
    version="0.1",
    description=(
        "Autoregressive next-step surrogate of a 2D SPH copper bar under "
        "Taylor impact against a rigid wall (ADR-0019)."
    ),
    provenance=(
        "LS-DYNA parametric sweep (3 bar lengths x 11 impact velocities) "
        "produced by Curtin collaborators; benchmark protocol per ADR-0019. "
        "One extra Convergence run is held aside for a mesh-resolution check."
    ),
    data_license="CC BY 4.0",
    solver="LS-DYNA",
    discretisation="SPH",
    materials=("*MAT_ELASTIC_PLASTIC_HYDRO", "*EOS_GRUNEISEN"),
    erosion=False,
    loading="rigid-wall impact; initial velocity 100-200 m/s",
    source_units="g-mm-ms",
    geometry="2D bar, 20 mm x {60, 80, 100} mm",
    n_cases=len(TRAIN) + len(VAL) + len(TEST_INTERP) + len(TEST_EXTRAP),
    splits={
        "train": len(TRAIN),
        "val": len(VAL),
        "test_interp": len(TEST_INTERP),
        "test_extrap": len(TEST_EXTRAP),
    },
    task="autoregressive transition (ADR-0019)",
    aux_field=AUX_FIELD,
    aux_unit="MPa",
    qois=tuple(QOIS),
    fields=(
        "positions",
        "velocity",
        "acceleration",
        "stress",
        "effective_plastic_strain",
    ),
    particles_per_case="4804-8004",
    n_frames=152,
    output_dt_ms=0.002,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/benchmarks/test_taylor_card.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/structbench/benchmarks/taylor_impact_2d/card.py tests/benchmarks/test_taylor_card.py
git commit -m "feat: Taylor 2D benchmark card (ADR-0025 retrofit)"
```

---

### Task 3: Generic auxiliary channel in `datasets/` (rename `von_mises` → `aux`)

This is the invasive rename. `CaseTrajectory.von_mises` becomes `CaseTrajectory.aux`, and `load_case_trajectory` selects the extraction by name. Everything else already uses generic `"aux"` naming (normalization keys, `WindowDataset`'s `next_aux`, `RolloutResult.predicted_aux`).

**Files:**
- Modify: `src/structbench/datasets/canonical.py` (CaseTrajectory at lines 41–49, `load_case_trajectory` at lines 52–93)
- Modify: `src/structbench/datasets/particle.py` (WindowDataset reads `.von_mises` — swap to `.aux`)
- Modify: `src/structbench/datasets/normalization.py` (`compute_stats` reads `.von_mises` — swap to `.aux`)
- Modify: `src/structbench/datasets/__init__.py` (export `available_aux_fields`)
- Modify: `src/structbench/eval/rollout.py` (ground-truth aux reads `trajectory.von_mises` — swap to `.aux`)
- Modify: `src/structbench/cli/train.py` (any `.von_mises` attribute access — mechanical)
- Test: `tests/datasets/test_canonical.py` (new tests below + mechanical rename), mechanical renames in `tests/datasets/test_normalization.py`, `tests/datasets/test_particle.py`, `tests/eval/test_rollout.py`, `tests/test_pipeline_smoke.py` (wherever `von_mises=` / `.von_mises` appears in trajectory construction)

**Interfaces:**
- Consumes: existing `von_mises_from_voigt(stress: NDArray) -> NDArray` (unchanged, stays public).
- Produces:
  - `CaseTrajectory` with field `aux: NDArray[np.float32]` (shape `(T, P)`) replacing `von_mises`.
  - `load_case_trajectory(h5_path, *, aux_field: str = "von_mises_stress", length_scale: float = 1e3, stress_scale: float = 1e-6) -> CaseTrajectory`.
  - `available_aux_fields() -> frozenset[str]` — initially `frozenset({"von_mises_stress"})`. Plans 2/3 add `"axial_stress"` and `"damage"` as one-entry additions.
  - Unknown `aux_field` raises `KeyError` naming the available fields.

- [ ] **Step 1: Write the failing tests** (add to `tests/datasets/test_canonical.py`)

```python
def test_load_case_trajectory_default_aux_is_von_mises(tmp_path):
    # reuse the existing _sph_case(tmp_path) helper in this file
    h5_path = _sph_case(tmp_path)
    tr = load_case_trajectory(h5_path)
    assert tr.aux.shape == tr.positions.shape[:2]


def test_load_case_trajectory_rejects_unknown_aux_field(tmp_path):
    h5_path = _sph_case(tmp_path)
    with pytest.raises(KeyError, match="von_mises_stress"):
        load_case_trajectory(h5_path, aux_field="no_such_field")


def test_available_aux_fields_lists_von_mises():
    from structbench.datasets import available_aux_fields

    assert "von_mises_stress" in available_aux_fields()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/datasets/test_canonical.py -v`
Expected: FAIL — `AttributeError: ... no attribute 'aux'` / `ImportError: available_aux_fields`

- [ ] **Step 3: Implement in `datasets/canonical.py`**

Rename the dataclass field (`von_mises: ...` → `aux: ...`, update its docstring to "auxiliary target field, selected by ``aux_field``"). Then add the extractor registry and rework the aux computation inside `load_case_trajectory`:

```python
# near the top of canonical.py, after von_mises_from_voigt
AuxExtractor = Callable[[h5py.Group, float], NDArray[np.float32]]
"""Maps (response/element/sph group, stress_scale) to a (T, P) aux array."""


def _aux_von_mises(sph: h5py.Group, stress_scale: float) -> NDArray[np.float32]:
    """von Mises stress derived from the 6-component Voigt stress, scaled."""
    vm = von_mises_from_voigt(sph["stress"][...])
    return (vm * stress_scale).astype(np.float32)


_AUX_EXTRACTORS: dict[str, AuxExtractor] = {
    "von_mises_stress": _aux_von_mises,
}


def available_aux_fields() -> frozenset[str]:
    """Names accepted by :func:`load_case_trajectory`'s ``aux_field``."""
    return frozenset(_AUX_EXTRACTORS)
```

In `load_case_trajectory`, add the keyword-only parameter `aux_field: str = "von_mises_stress"`, and replace the inline von Mises computation with:

```python
    try:
        extractor = _AUX_EXTRACTORS[aux_field]
    except KeyError:
        raise KeyError(
            f"unknown aux_field {aux_field!r}; available: "
            f"{', '.join(sorted(_AUX_EXTRACTORS))}"
        ) from None
    aux = extractor(sph_group, stress_scale)
```

(where `sph_group` is the already-open `response/element/sph` h5py group the current code reads `stress` from). Construct `CaseTrajectory(..., aux=aux, ...)`. Update the function docstring: parameters + the note that stress-like extractors receive `stress_scale`.

- [ ] **Step 4: Mechanical rename across the package**

In each of `datasets/particle.py`, `datasets/normalization.py`, `eval/rollout.py`, `cli/train.py`: replace attribute access `.von_mises` with `.aux` (constructor keyword `von_mises=` with `aux=` where CaseTrajectory is built). Export `available_aux_fields` from `datasets/__init__.py`. Do the same mechanical rename in the four test files listed above (their trajectory-builder helpers construct `CaseTrajectory(...)` directly).

Search to confirm nothing is left: `grep -rn "von_mises\b" src tests` — remaining hits must only be `von_mises_from_voigt`, the string `"von_mises_stress"`, and `rollout_von_mises_rmse` in `cli/train.py` (renamed in Task 5).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: all pass (66+ tests, plus the 3 new ones)

- [ ] **Step 6: Commit**

```bash
git add src tests
git commit -m "refactor: generic aux channel in datasets (von_mises -> aux, named extractors)"
```

---

### Task 4: `BenchmarkSpec` + registry, Taylor `SPEC`

**Files:**
- Create: `src/structbench/benchmarks/registry.py`
- Modify: `src/structbench/benchmarks/__init__.py`
- Modify: `src/structbench/benchmarks/taylor_impact_2d/__init__.py`
- Test: `tests/benchmarks/test_registry.py`

**Interfaces:**
- Consumes: `BenchmarkCard` (Task 1), `available_aux_fields()` (Task 3), `QoiFn` from `eval` (pre-existing import direction), Taylor module constants.
- Produces:
  - `BenchmarkSpec` frozen dataclass: `card: BenchmarkCard`, `splits: dict[str, tuple[str, ...]]`, `eval_splits: tuple[str, ...]`, `aux_field: str`, `qois: dict[str, QoiFn]`, `boundary_feature_fn: Callable[[Tensor, float], Tensor] | None`, `dataset_id: str`. `__post_init__` validates: `"train"` and `"val"` present; every `eval_splits` name exists in `splits`; `card.splits == {name: len(ids)}`; `aux_field in available_aux_fields()`.
  - `get_benchmark(name: str) -> BenchmarkSpec` (lazy import; `KeyError` listing available names on miss).
  - `available_benchmarks() -> tuple[str, ...]`.
  - `structbench.benchmarks.taylor_impact_2d.SPEC: BenchmarkSpec`.
  - `benchmarks/__init__.py` re-exports: `BenchmarkCard`, `BenchmarkSpec`, `get_benchmark`, `available_benchmarks`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/benchmarks/test_registry.py
"""Benchmark registry resolution and spec invariants."""

import pytest

from structbench.benchmarks import (
    BenchmarkSpec,
    available_benchmarks,
    get_benchmark,
)


def test_taylor_is_registered():
    assert "taylor_impact_2d" in available_benchmarks()


def test_get_benchmark_resolves_taylor_spec():
    spec = get_benchmark("taylor_impact_2d")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.card.name == "Taylor2D-Impact"
    assert spec.eval_splits == ("val", "test_interp", "test_extrap")
    assert len(spec.splits["train"]) == 21
    assert spec.aux_field == "von_mises_stress"
    assert spec.boundary_feature_fn is not None
    assert spec.dataset_id == "2D-Copper-Bar-Taylor-Impact"


def test_unknown_benchmark_raises_with_available_names():
    with pytest.raises(KeyError, match="taylor_impact_2d"):
        get_benchmark("no_such_benchmark")


def test_spec_validates_card_split_sizes():
    spec = get_benchmark("taylor_impact_2d")
    bad_card_splits = dict(spec.card.splits)
    bad_card_splits["train"] += 1
    from dataclasses import replace

    with pytest.raises(ValueError, match="split"):
        replace(spec, card=replace(spec.card, n_cases=spec.card.n_cases + 1,
                                   splits=bad_card_splits))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/benchmarks/test_registry.py -v`
Expected: FAIL — `ImportError` on `BenchmarkSpec`

- [ ] **Step 3: Write `registry.py`**

```python
# src/structbench/benchmarks/registry.py
"""Benchmark spec and name-based registry (ADR-0022, ADR-0025).

A benchmark module exposes one frozen :class:`BenchmarkSpec` named
``SPEC``; the training pipeline resolves it by name through
:func:`get_benchmark`, replacing per-benchmark imports in ``cli/``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

from torch import Tensor

from ..datasets import available_aux_fields
from ..eval import QoiFn
from .card import BenchmarkCard

#: Registered benchmark modules; each must define a module-level ``SPEC``.
_MODULES: dict[str, str] = {
    "taylor_impact_2d": "structbench.benchmarks.taylor_impact_2d",
}


@dataclass(frozen=True)
class BenchmarkSpec:
    """The runtime contract of one benchmark.

    Parameters
    ----------
    card : BenchmarkCard
        Descriptive metadata (ADR-0025). Its ``splits`` sizes must match
        the actual split lists here — validated at construction.
    splits : dict of str to tuple of str
        Immutable case-id lists by split name; must contain ``"train"``
        and ``"val"``.
    eval_splits : tuple of str
        Split names evaluated after training, in reporting order; each
        must be a key of ``splits``.
    aux_field : str
        Auxiliary target name, resolved by
        :func:`structbench.datasets.load_case_trajectory`.
    qois : dict of str to QoiFn
        Quantities of interest evaluated on rolled-out trajectories.
    boundary_feature_fn : callable or None
        ``(positions (P, dim) mm, radius) -> (P, 1)`` boundary feature,
        or ``None`` when the benchmark has no analytic boundary.
    dataset_id : str
        The canonical dataset this benchmark reads.
    """

    card: BenchmarkCard
    splits: dict[str, tuple[str, ...]]
    eval_splits: tuple[str, ...]
    aux_field: str
    qois: dict[str, QoiFn] = field(default_factory=dict)
    boundary_feature_fn: Callable[[Tensor, float], Tensor] | None = None
    dataset_id: str = ""

    def __post_init__(self) -> None:
        for required in ("train", "val"):
            if required not in self.splits:
                raise ValueError(f"splits must include {required!r}")
        missing = [s for s in self.eval_splits if s not in self.splits]
        if missing:
            raise ValueError(f"eval_splits not present in splits: {missing}")
        actual = {name: len(ids) for name, ids in self.splits.items()}
        if self.card.splits != actual:
            raise ValueError(
                f"card split sizes {self.card.splits} != actual {actual}"
            )
        if self.aux_field not in available_aux_fields():
            raise ValueError(
                f"aux_field {self.aux_field!r} not in "
                f"{sorted(available_aux_fields())}"
            )


def available_benchmarks() -> tuple[str, ...]:
    """Registered benchmark names, sorted."""
    return tuple(sorted(_MODULES))


def get_benchmark(name: str) -> BenchmarkSpec:
    """Resolve a benchmark's :class:`BenchmarkSpec` by registry name.

    Raises
    ------
    KeyError
        If ``name`` is not registered; the message lists valid names.
    """
    if name not in _MODULES:
        raise KeyError(
            f"unknown benchmark {name!r}; available: "
            f"{', '.join(available_benchmarks())}"
        )
    module = importlib.import_module(_MODULES[name])
    spec: BenchmarkSpec = module.SPEC
    return spec
```

- [ ] **Step 4: Build Taylor's `SPEC` in `taylor_impact_2d/__init__.py`**

Keep every existing re-export (train.py still imports them until Task 5), and add:

```python
from ..registry import BenchmarkSpec
from .benchmark import (
    AUX_FIELD,
    QOIS,
    TEST_EXTRAP,
    TEST_INTERP,
    TRAIN,
    VAL,
    wall_distance_feature,
)
from .card import CARD

SPEC = BenchmarkSpec(
    card=CARD,
    splits={
        "train": tuple(TRAIN),
        "val": tuple(VAL),
        "test_interp": tuple(TEST_INTERP),
        "test_extrap": tuple(TEST_EXTRAP),
    },
    eval_splits=("val", "test_interp", "test_extrap"),
    aux_field=AUX_FIELD,
    qois=dict(QOIS),
    boundary_feature_fn=wall_distance_feature,
    dataset_id="2D-Copper-Bar-Taylor-Impact",
)
```

Update `benchmarks/__init__.py`:

```python
"""Benchmark problem definitions (ARCHITECTURE.md; registry per ADR-0022)."""

from .card import BenchmarkCard
from .registry import BenchmarkSpec, available_benchmarks, get_benchmark

__all__ = [
    "BenchmarkCard",
    "BenchmarkSpec",
    "available_benchmarks",
    "get_benchmark",
]
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/benchmarks -v` then `python -m pytest -q`
Expected: registry tests PASS; full suite green.

- [ ] **Step 6: Commit**

```bash
git add src/structbench/benchmarks tests/benchmarks/test_registry.py
git commit -m "feat: BenchmarkSpec + name registry; Taylor SPEC (ADR-0022)"
```

---

### Task 5: De-Taylorize `cli/train.py`

**Files:**
- Modify: `src/structbench/cli/train.py`
- Modify: `configs/taylor_2d.toml`, `configs/taylor_2d_smoke.toml` (add `benchmark = "taylor_impact_2d"` as the first key)
- Test: `tests/cli/test_train_config.py` (add), existing `tests/cli/test_train_eval.py` + `tests/test_pipeline_smoke.py` must stay green

**Interfaces:**
- Consumes: `get_benchmark` (Task 4), `load_case_trajectory(..., aux_field=...)` (Task 3).
- Produces:
  - `TrainConfig` gains field `benchmark: str = "taylor_impact_2d"`.
  - `train(spec: BenchmarkSpec, gns: GNSConfig, train_cfg: TrainConfig, data_root: Path, out_dir: Path, device: str) -> Path | None` — spec is the new first parameter.
  - `evaluate(...)` signature unchanged; it resolves the spec internally from `config.json`'s `"benchmark"` key (default `"taylor_impact_2d"` when absent, so pre-existing run dirs still evaluate).
  - `config.json` gains `"benchmark": <name>`; metrics key `rollout_von_mises_rmse` renamed `rollout_aux_rmse` and the metrics dict gains `"aux_field": spec.aux_field`.

- [ ] **Step 1: Write the failing test** (add to `tests/cli/test_train_config.py`)

```python
def test_train_config_benchmark_defaults_to_taylor(tmp_path):
    cfg = TrainConfig()
    assert cfg.benchmark == "taylor_impact_2d"


def test_train_config_benchmark_from_toml(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('benchmark = "taylor_impact_2d"\nbatch_size = 4\n')
    cfg = TrainConfig.from_toml(toml)
    assert cfg.benchmark == "taylor_impact_2d"
    assert cfg.batch_size == 4
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/cli/test_train_config.py -v`
Expected: FAIL — `TrainConfig` has no attribute `benchmark`

- [ ] **Step 3: Implement the refactor**

In `src/structbench/cli/train.py`, apply exactly these changes:

1. **Imports** (lines 41–48): replace the Taylor import block with

```python
from ..benchmarks import BenchmarkSpec, get_benchmark
```

2. **`TrainConfig`**: add as the first field, with docstring line "benchmark : str — registry name resolved via `structbench.benchmarks.get_benchmark`":

```python
    benchmark: str = "taylor_impact_2d"
```

3. **`_wall_feature_fn`** (lines 355–361): replace with

```python
def _bind_boundary_feature(
    spec: BenchmarkSpec, gns: GNSConfig
) -> Callable[[Tensor], Tensor] | None:
    """Bind the spec's boundary feature to the configured radius, if any."""
    fn = spec.boundary_feature_fn
    if fn is None:
        return None

    def feature(positions: Tensor) -> Tensor:
        return fn(positions, gns.connectivity_radius)

    return feature
```

4. **`_load_trajectories`** (lines 291–297): add parameter `aux_field: str` and pass it through:

```python
def _load_trajectories(
    case_ids: list[str], data_root: Path, aux_field: str
) -> list[CaseTrajectory]:
    """Load each ``<data_root>/<case_id>.h5`` into a :class:`CaseTrajectory`."""
    return [
        load_case_trajectory(data_root / f"{case_id}.h5", aux_field=aux_field)
        for case_id in case_ids
    ]
```

5. **`train`**: new first parameter `spec: BenchmarkSpec`. Replace `TRAIN`/`VAL` with `list(spec.splits["train"])` / `list(spec.splits["val"])`, pass `spec.aux_field` to both `_load_trajectories` calls, replace `boundary_feature_fn=_wall_feature_fn(gns)` with `boundary_feature_fn=_bind_boundary_feature(spec, gns)`, and pass `train_cfg` through to `_write_resolved_config` unchanged. Update the docstring's Taylor-specific sentences to refer to "the benchmark spec".

6. **`_write_resolved_config`** (lines 526–542): add parameter `benchmark: str` and the key `"benchmark": benchmark` to the resolved dict; the `train` call site passes `train_cfg.benchmark`.

7. **`evaluate`** (lines 551–688): after `resolved = json.loads(...)` add

```python
    spec = get_benchmark(resolved.get("benchmark", "taylor_impact_2d"))
```

then use `spec` for the three consumers: `boundary_feature_fn=_bind_boundary_feature(spec, gns)`, `load_case_trajectory(..., aux_field=spec.aux_field)`, `qois=spec.qois` (replacing `QOIS`, including the `for name in QOIS` aggregation loop → `for name in spec.qois`). Rename the per-case and mean metric key `rollout_von_mises_rmse` → `rollout_aux_rmse`, add `"aux_field": spec.aux_field` to the top-level metrics dict, and change the log line `"von Mises %.4f MPa"` to use `spec.aux_field` and `spec.card.aux_unit`.

8. **`main`** (lines 691–763): after building `train_cfg`, resolve `spec = get_benchmark(train_cfg.benchmark)` for train mode. For valid/rollout modes read the run dir instead:

```python
    if args.mode == "train":
        spec = get_benchmark(train_cfg.benchmark)
        ckpt = train(spec, gns, train_cfg, data_root, out_dir, device)
        print(f"training complete; best checkpoint: {ckpt}")
    else:
        resolved = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
        spec = get_benchmark(resolved.get("benchmark", "taylor_impact_2d"))
        if args.mode == "valid":
            metrics = evaluate(
                list(spec.splits["val"]), data_root, out_dir, device,
                split_name="val",
            )
            _print_split_report(metrics)
        else:  # rollout: every eval split except val, in spec order
            for split_name in spec.eval_splits:
                if split_name == "val":
                    continue
                metrics = evaluate(
                    list(spec.splits[split_name]), data_root, out_dir, device,
                    split_name=split_name,
                )
                _print_split_report(metrics)
```

9. **`_print_split_report`** (lines 766–777): read the aux label from the metrics dict — replace the hardcoded `rollout von Mises RMSE ... MPa` segment with `f"rollout {metrics['aux_field']} RMSE {mean['rollout_aux_rmse']:.4f}"`. Drop the hardcoded `mm` on the QoI line only if the value isn't a length — it is for Taylor; keep `mm` for now and note QoI units come from the card in a later plan.

10. **Module docstring** (lines 1–22): rewrite the two Taylor-specific sentences to describe the registry (`TrainConfig.benchmark` → `get_benchmark` → spec supplies splits, aux field, QoIs, boundary feature).

11. **Configs**: add `benchmark = "taylor_impact_2d"` as the first line of `configs/taylor_2d.toml` and `configs/taylor_2d_smoke.toml`.

- [ ] **Step 4: Fix the ripple in existing tests**

`tests/cli/test_train_eval.py` and `tests/test_pipeline_smoke.py` call `train(...)`: add the spec argument `get_benchmark("taylor_impact_2d")` (or a locally-built minimal `BenchmarkSpec` if the test builds synthetic case ids that aren't Taylor ids — in that case construct a spec whose splits are the synthetic ids and whose card is a minimal consistent `BenchmarkCard`). Where those tests assert on `rollout_von_mises_rmse`, rename to `rollout_aux_rmse`.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: all green.

- [ ] **Step 6: Verify the CLI contract manually (no data needed)**

Run: `python -m structbench.cli.train --mode train` (no data-root)
Expected: exits 2 with `error: --data-root is required` — argument plumbing intact.

- [ ] **Step 7: Commit**

```bash
git add src/structbench/cli/train.py configs tests
git commit -m "refactor: benchmark-generic training pipeline via registry (ADR-0022)"
```

---

### Task 6: Card renderers + generated `docs/benchmarks.md` + archive files

**Files:**
- Create: `src/structbench/benchmarks/render.py`
- Create: `tools/gen_benchmark_docs.py`
- Create: `docs/benchmarks.md` (generated, committed)
- Test: `tests/benchmarks/test_render.py`

**Interfaces:**
- Consumes: `available_benchmarks()`, `get_benchmark()` (Task 4).
- Produces:
  - `render_index(specs: list[BenchmarkSpec]) -> str` — full `docs/benchmarks.md` content: header comment `<!-- generated by tools/gen_benchmark_docs.py; do not edit by hand -->`, a comparison table (columns: Benchmark, Solver, Discretisation, Erosion, Loading, Cases, Particles, Frames, Aux target), then one `##` section per benchmark with the description, materials, provenance, license, splits, and QoIs.
  - `render_archive_readme(spec: BenchmarkSpec) -> str` — standalone dataset README for the hosted archive.
  - `card_json(card: BenchmarkCard) -> str` — `json.dumps(card.to_json_dict(), indent=2)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/benchmarks/test_render.py
"""Card renderers and the committed-index drift check."""

from pathlib import Path

from structbench.benchmarks import available_benchmarks, get_benchmark
from structbench.benchmarks.render import (
    card_json,
    render_archive_readme,
    render_index,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _all_specs():
    return [get_benchmark(name) for name in available_benchmarks()]


def test_index_contains_taylor_row_and_generation_marker():
    text = render_index(_all_specs())
    assert "do not edit by hand" in text
    assert "Taylor2D-Impact" in text
    assert "SPH" in text


def test_archive_readme_is_self_describing():
    spec = get_benchmark("taylor_impact_2d")
    text = render_archive_readme(spec)
    assert "Taylor2D-Impact" in text
    assert "CC BY 4.0" in text
    assert "g-mm-ms" in text


def test_card_json_round_trips():
    import json

    spec = get_benchmark("taylor_impact_2d")
    data = json.loads(card_json(spec.card))
    assert data["name"] == "Taylor2D-Impact"


def test_committed_index_is_up_to_date():
    committed = (REPO_ROOT / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    assert committed == render_index(_all_specs())
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/benchmarks/test_render.py -v`
Expected: FAIL — no module `render`

- [ ] **Step 3: Implement `render.py`**

```python
# src/structbench/benchmarks/render.py
"""Render benchmark cards to human-facing views (ADR-0025).

The card instances in each benchmark module are the source of truth;
``docs/benchmarks.md`` and the per-archive README/``card.json`` are
generated from them by ``tools/gen_benchmark_docs.py``. A drift test
asserts the committed index matches :func:`render_index`.
"""

from __future__ import annotations

import json

from .card import BenchmarkCard
from .registry import BenchmarkSpec

_MARKER = "<!-- generated by tools/gen_benchmark_docs.py; do not edit by hand -->"


def _yesno(flag: bool) -> str:
    return "yes" if flag else "no"


def render_index(specs: list[BenchmarkSpec]) -> str:
    """The full ``docs/benchmarks.md`` content for the given specs."""
    lines = [
        _MARKER,
        "",
        "# Benchmarks",
        "",
        "One row per benchmark; see each section for the full card.",
        "",
        "| Benchmark | Solver | Discretisation | Erosion | Loading | Cases "
        "| Particles | Frames | Aux target |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for spec in specs:
        c = spec.card
        lines.append(
            f"| {c.name} | {c.solver} | {c.discretisation} | {_yesno(c.erosion)} "
            f"| {c.loading} | {c.n_cases} | {c.particles_per_case} "
            f"| {c.n_frames} | {c.aux_field} ({c.aux_unit}) |"
        )
    for spec in specs:
        lines.extend(["", *_section(spec)])
    return "\n".join(lines) + "\n"


def _section(spec: BenchmarkSpec) -> list[str]:
    c = spec.card
    splits = ", ".join(f"{name} {n}" for name, n in c.splits.items())
    return [
        f"## {c.name} (v{c.version})",
        "",
        c.description,
        "",
        f"- **Task**: {c.task}",
        f"- **Materials**: {', '.join(c.materials)}",
        f"- **Geometry**: {c.geometry}; source units {c.source_units}",
        f"- **Splits**: {splits}",
        f"- **QoIs**: {', '.join(c.qois)}",
        f"- **Fields**: {', '.join(c.fields)}",
        f"- **Provenance**: {c.provenance}",
        f"- **License**: {c.data_license}",
    ]


def render_archive_readme(spec: BenchmarkSpec) -> str:
    """A standalone README for the hosted dataset archive."""
    c = spec.card
    return "\n".join(
        [
            f"# {c.name} — StructBench canonical dataset",
            "",
            c.description,
            "",
            f"- Solver: {c.solver} ({c.discretisation}; erosion: {_yesno(c.erosion)})",
            f"- Loading: {c.loading}",
            f"- Source units: {c.source_units} (files are strict SI, ADR-0012)",
            f"- Cases: {c.n_cases} ({', '.join(f'{k} {v}' for k, v in c.splits.items())})",
            f"- Particles per case: {c.particles_per_case}; "
            f"{c.n_frames} frames at {c.output_dt_ms} ms",
            f"- Provenance: {c.provenance}",
            f"- License: {c.data_license}",
            "",
            "Machine-readable metadata: `card.json` alongside this file.",
            "Consume with `structbench.datasets.load_case_trajectory` or any "
            "HDF5 reader (layout per ADR-0013).",
        ]
    ) + "\n"


def card_json(card: BenchmarkCard) -> str:
    """The card as pretty-printed JSON for the dataset archive."""
    return json.dumps(card.to_json_dict(), indent=2)
```

- [ ] **Step 4: Implement `tools/gen_benchmark_docs.py`**

```python
# tools/gen_benchmark_docs.py
"""Regenerate docs/benchmarks.md (and, optionally, archive card files).

Usage:
    python tools/gen_benchmark_docs.py                 # rewrite docs/benchmarks.md
    python tools/gen_benchmark_docs.py --check         # exit 1 if stale
    python tools/gen_benchmark_docs.py --archive taylor_impact_2d --out DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from structbench.benchmarks import available_benchmarks, get_benchmark
from structbench.benchmarks.render import (
    card_json,
    render_archive_readme,
    render_index,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX = REPO_ROOT / "docs" / "benchmarks.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--archive", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.archive:
        if not args.out:
            print("error: --archive requires --out")
            return 2
        spec = get_benchmark(args.archive)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.md").write_text(
            render_archive_readme(spec), encoding="utf-8", newline="\n"
        )
        (out / "card.json").write_text(
            card_json(spec.card), encoding="utf-8", newline="\n"
        )
        print(f"wrote {out / 'README.md'} and {out / 'card.json'}")
        return 0

    text = render_index([get_benchmark(n) for n in available_benchmarks()])
    if args.check:
        current = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        if current != text:
            print("docs/benchmarks.md is stale; run tools/gen_benchmark_docs.py")
            return 1
        print("docs/benchmarks.md is up to date")
        return 0
    INDEX.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {INDEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Generate the committed index**

Run: `python tools/gen_benchmark_docs.py`
Expected: `wrote ...docs\benchmarks.md`; the file contains the Taylor row.

- [ ] **Step 6: Run the tests (drift check now passes)**

Run: `python -m pytest tests/benchmarks/test_render.py -v` then `python -m pytest -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/structbench/benchmarks/render.py tools/gen_benchmark_docs.py docs/benchmarks.md tests/benchmarks/test_render.py
git commit -m "feat: card renderers + generated benchmark index (ADR-0025)"
```

---

### Task 7: Environment-gated card-vs-data validation test

**Files:**
- Create: `tests/benchmarks/test_card_data.py`

**Interfaces:**
- Consumes: `get_benchmark`, `read_case` from `structbench.core.io`.
- Produces: the repo's first data-gated test pattern: module-level `pytest.mark.skipif` on `STRUCTBENCH_DATA_ROOT`.

- [ ] **Step 1: Write the test** (it "fails" only by skipping when data is absent; with data present it validates for real)

```python
# tests/benchmarks/test_card_data.py
"""Card-vs-data validation (ADR-0025) — runs only when data is present.

Set STRUCTBENCH_DATA_ROOT to the canonical HDF5 directory of the Taylor
dataset (the folder holding <case_id>.h5) to enable.
"""

import os
from pathlib import Path

import pytest

from structbench.benchmarks import get_benchmark
from structbench.core.io import read_case

DATA_ROOT = os.environ.get("STRUCTBENCH_DATA_ROOT")

pytestmark = pytest.mark.skipif(
    DATA_ROOT is None, reason="STRUCTBENCH_DATA_ROOT not set"
)


def test_taylor_card_matches_one_canonical_case():
    spec = get_benchmark("taylor_impact_2d")
    case_id = spec.splits["train"][0]
    case = read_case(Path(DATA_ROOT) / f"{case_id}.h5")

    lo, hi = (int(x) for x in spec.card.particles_per_case.split("-"))
    n_particles = case.elements["sph"].element_id.shape[0]
    assert lo <= n_particles <= hi

    assert case.response is not None
    assert case.response.time.shape[0] == spec.card.n_frames
```

- [ ] **Step 2: Run without the env var**

Run: `python -m pytest tests/benchmarks/test_card_data.py -v`
Expected: 1 skipped (`STRUCTBENCH_DATA_ROOT not set`)

- [ ] **Step 3: Run with the env var if the Taylor canonical data is present locally**

Run (PowerShell): `$env:STRUCTBENCH_DATA_ROOT = "..\data\2D-Copper-Bar-Taylor-Impact\h5_canonical"; python -m pytest tests/benchmarks/test_card_data.py -v; Remove-Item Env:STRUCTBENCH_DATA_ROOT`
Expected: PASS (or a real card error to fix — if `n_frames`/particle range disagrees with the data, correct **the card**, not the test).

- [ ] **Step 4: Commit**

```bash
git add tests/benchmarks/test_card_data.py
git commit -m "test: env-gated card-vs-data validation (ADR-0025)"
```

---

### Task 8: Docs touch-up + full verification

**Files:**
- Modify: `docs/ARCHITECTURE.md` (two paragraphs)
- Verify: whole tree

- [ ] **Step 1: Update `docs/ARCHITECTURE.md`**

In the `### benchmarks/` section, append one sentence to the first paragraph: benchmarks are resolved by name through a registry (`get_benchmark`), and each module ships a typed `BenchmarkCard` (ADR-0025) from which `docs/benchmarks.md` and per-archive metadata are generated.

In the `### datasets/` **ML data flow** paragraph, replace the phrase describing the trajectory ("positions in mm and stress in MPa (ADR-0019)") with a benchmark-generic version: positions in mm plus one auxiliary target field selected by name (`aux_field`, e.g. von Mises stress for Taylor), per the owning benchmark's spec.

- [ ] **Step 2: Full verification**

Run, in order:
- `python -m pytest -q` — expected: all pass, 1 skip (card-data test)
- `ruff check src tests tools` — expected: clean
- `ruff format --check src tests tools` — expected: clean
- `python -m mypy src` — expected: clean

- [ ] **Step 3: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: ARCHITECTURE reflects benchmark registry + cards"
```

- [ ] **Step 4: Hand off**

Do not merge; leave `feat/benchmark-registry-cards` for human review (CLAUDE.md forbidden tier). Report test counts and any deviations from this plan.

---

## Post-plan notes

- **Known pre-existing deviation, not addressed here**: `benchmarks/` imports `eval/` (`QoiFn`, metric fns), which the ARCHITECTURE dependency graph text disallows. This plan follows the existing practice; reconciling the graph text is flagged for a docs distillation pass.
- **README summary row (ADR-0025) deferred to Plan 3**: the repository README is Taylor-only and mid-v0.1-release (human actions pending); the benchmarks summary row + link to `docs/benchmarks.md` lands when the index has multiple rows, avoiding churn against the release edits.
- **Plan 2 (wave-1d)** adds: `"axial_stress"` extractor (one entry in `_AUX_EXTRACTORS`), the conversion script `data_generation/lsdyna/1DWavePropagation/convert.py` (modelled on the Taylor `convert.py`), the `wave_propagation_1d` module (splits per ADR-0023's table), two new QoI functions in `eval/metrics.py` (wave-front arrival time at gauge stations, peak stress error), a card, a registry entry, and `configs/wave_1d.toml`.
- **Plan 3 (notch-beam pair)** adds: `"damage"` extractor, the two-family conversion script with spec-sheet enumeration + extras flagging, the split-freezing step (ADR-0024 rule → committed case-id lists), two benchmark modules + cards + configs, and the mid-span-deflection QoI.
