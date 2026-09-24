# Abaqus Data Pipeline — Plan 1 of 2: generator, runner, conformance run, ODB export

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the first half of the Abaqus pipeline end to end on real runs:
- generate a sweep's decks;
- run them;
- export their ODBs to `abaqus-npz/1`;
- record what the first Abaqus/Explicit runs establish.

**Architecture:**
- Shared, dataset-blind scripts live in `data_generation/abaqus/` (`sampling.py`, `deck.py`, `generate.py`, `run_jobs.py`, `odb_export.py`).
- A dataset is a `sweep.toml` plus a `model.py`, kept in a private repository and passed by path.
- Every stage reads and writes only its case folder, so any stage can be re-run.

**Why two plans.** Plan 2 covers the adapter, the reader extensions, the verification rows, `datacheck --declaration`, `validate.py` and `archive.py`. It is written **after** Task 9, because its fixtures and reader wording must come from real Explicit output rather than from recall (ADR-0068 clause 8).

**Tech Stack:**
- Python 3.12+ project environment, plus scipy through a new optional extra `datagen`.
- Abaqus 2025, whose interpreter is Python 3.10.5 with numpy 1.22.4 and no h5py, used for `odb_export.py` only.
- pytest, ruff, mypy.

**Spec:** `docs/plans/2026-09-24-abaqus-data-pipeline-design.md` (approved 2026-09-24).

## Global Constraints

- **Branch:** `feat/abaqus-data-pipeline`. Never commit to `main`. Merging and pushing happen only on the maintainer's word (ADR-0023).
- **Interpreter:** `PY=/c/Users/kylin/.venvs/structbench/Scripts/python.exe`, the uv env outside the repo. Never create a venv inside the repo.
- **Gates** (all must pass before every commit; there is no CI):
  ```bash
  set -o pipefail; PY=/c/Users/kylin/.venvs/structbench/Scripts/python.exe
  $PY -m ruff format --check . && $PY -m ruff check . && $PY -m mypy src && $PY -m pytest -q && $PY tools/gen_benchmark_docs.py --check
  ```
  Run the **full** suite; test directories have no `__init__.py`, so test file basenames must be unique across `tests/`.
- **Public-repo rule** (CORRECTIONS 2026-08-12): nothing from the maintainer's study goes in the repository. That means no sweep values, no split rationale, and no paper names. The study's files live in the private repository only; the StructBench tests use made-up toy datasets.
- **`data_generation/` is not importable** (ADR-0010). Scripts in `data_generation/abaqus/` import their siblings by module name, since the script folder is on `sys.path` when run. Tests put that folder on `sys.path`.
- **`odb_export.py` must parse as Python 3.10** and import only the standard library, numpy and `odbAccess`, with `odbAccess` imported inside the function that needs it.
- **Test rules** (PRINCIPLES.md): no test needs Abaqus, network access or large files. Tests that need a real run are env-gated and skip when the variable is unset.
- **Numbers in decks** are written with `repr(float(x))`, so identical parameters give byte-identical decks.
- **Units stay as the deck's own** (`t-mm-s`) everywhere upstream of the adapter.
- **Runs** happen in a local work root with no spaces in its path (default `C:\structbench-runs\`), never inside OneDrive.
- **Compute and deletion:** any Abaqus job needs the maintainer's yes, with the estimated count and wall time, first. Nothing is ever deleted.
- **Commits:** Conventional Commits, each ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Lint:** line length is 88, and `E501` is enforced. Run `$PY -m ruff format <paths>` and then `$PY -m ruff check --fix <paths>` for import order. Any line `ruff format` cannot wrap (a single long string) gets split by hand.

## Review Focus

1. **A case id Abaqus won't accept as a job name.** The generator rejects any id outside `^[A-Za-z][A-Za-z0-9_-]{0,37}$` before writing anything (Task 4 test). The conformance run confirms that hyphens are accepted (Task 7).
2. **A sweep folder whose path contains whitespace**, such as a OneDrive folder. The runner refuses before launching anything (Task 6 test).
3. **Editing `sweep.toml` after runs exist.** A changed deck for a case that already has `run.json` is refused even with `--force`. Raising `n` only appends cases, leaving the existing ones byte-identical (Task 4 tests).
4. **A crashed runner leaving `.runner.lock`.** The next run refuses with a message naming the lock file (Task 6 test).
5. **A misconfigured split.** An unknown region, a region naming an unknown variable, an explicit point missing a variable, or a filter leaving too few points each raise an error naming the split (Task 2 tests).

---

## File Structure

```
pyproject.toml                          MODIFY: optional extra datagen = ["scipy>=1.15"]
uv.lock                                 MODIFY: relocked
docs/PRINCIPLES.md                      MODIFY: approved-dependency row for scipy
decisions/0069-abaqus-data-pipeline.md  CREATE: ADR (Proposed)
decisions/README.md                     MODIFY: index row 0069
data_generation/abaqus/
  sampling.py                           CREATE: Sobol splits, regions, explicit points, variants
  deck.py                               CREATE: keyword writers + structured quad mesh
  generate.py                           CREATE: CLI, sweep.toml + model.py -> case folders
  run_jobs.py                           CREATE: CLI, runs decks, run.json + run_log.csv
  odb_export.py                         CREATE: CLI under `abaqus python`, .odb -> .npz
  STANDARD_INPUT_BLOCK.md               MODIFY: what the conformance run established (Task 9)
data_generation/README.md               MODIFY: the shared-code pattern for Abaqus (Task 9)
tests/tools/
  abaqus_paths.py                       CREATE: puts data_generation/abaqus on sys.path
  test_abaqus_sampling.py               CREATE
  test_abaqus_deck.py                   CREATE
  test_abaqus_generate.py               CREATE
  test_abaqus_run_jobs.py               CREATE
  test_abaqus_odb_export.py             CREATE
<private-repo>/abaqus/<dataset>/{sweep.toml, model.py}   Task 5
```

---

### Task 1: ADR-0069 and the `datagen` extra

**Files:**
- Create: `decisions/0069-abaqus-data-pipeline.md`
- Modify: `decisions/README.md` (index row after 0068), `pyproject.toml` (`[project.optional-dependencies]`), `uv.lock`, `docs/PRINCIPLES.md` (runtime table)

**Interfaces:**
- Produces: `scipy.stats.qmc` importable in `$PY`, and ADR-0069 for later tasks and Plan 2 to cite.

- [ ] **Step 1: Baseline gates are green before any change**

Run the gate command from Global Constraints. Expected: all pass. If not, stop and report; don't build on a red tree.

- [ ] **Step 2: Add the extra.** In `pyproject.toml`, under `[project.optional-dependencies]` after `data = [...]`:

```toml
datagen = [
    "scipy>=1.15",
]
```

- [ ] **Step 3: Relock and install without removing anything already in the env**

```bash
cd /c/Users/kylin/Desktop/StructBench && uv lock && UV_PROJECT_ENVIRONMENT=/c/Users/kylin/.venvs/structbench uv sync --inexact --extra dev --extra datagen
/c/Users/kylin/.venvs/structbench/Scripts/python.exe -c "import scipy, scipy.stats.qmc as q; print(scipy.__version__); q.Sobol(d=2, scramble=True, rng=__import__('numpy').random.default_rng(1)).random_base2(m=3)"
```
Expected: a version ≥ 1.15 and no error. Sobol's `rng=` keyword needs ≥ 1.15. If it raises `TypeError`, the lock resolved an older scipy, so fix the lock rather than the code.

- [ ] **Step 4: PRINCIPLES row.** In `docs/PRINCIPLES.md`, runtime table, after the `huggingface_hub` row:

```markdown
| scipy | Scrambled-Sobol sampling in the Abaqus data-generation scripts; optional `datagen` extra, never imported by the package | ADR-0069 |
```

- [ ] **Step 5: Write the ADR** `decisions/0069-abaqus-data-pipeline.md`:

```markdown
# 0069 — The Abaqus data-generation pipeline

**Status**: Proposed
**Type**: Durable
**Date**: 2026-09-24

## Context

The README Roadmap names agent-driven data generation with Abaqus as the first
solver. ADR-0068 admitted Abaqus and its two text readers but left the
intermediate format, the adapter and every generation step open. The design is
`docs/plans/2026-09-24-abaqus-data-pipeline-design.md`.

## Decision

1. **Four stages, each an idempotent command**: generator, runner, extractor,
   validator. State lives in the case folder; a stage does only missing work.
   Claude Code operates them; compute and deletions wait for the maintainer.
2. **Shared code in `data_generation/abaqus/`; a dataset is a `sweep.toml` and
   a `model.py`**, passed by path. A dataset may live in a private repository
   until admission (CORRECTIONS 2026-08-12); on admission it moves to
   `data_generation/abaqus/<name>/`. Provenance records the commits of both
   repositories.
3. **The intermediate is `abaqus-npz/1`** (answers ADR-0068's open format
   question). Measured 2026-09-24: the Abaqus 2025 interpreter is Python
   3.10.5 with numpy 1.22.4 and scipy 1.11.1 and no h5py, so an HDF5
   intermediate would need a package the interpreter lacks.
4. **scipy enters as the optional extra `datagen`** for scrambled Sobol
   sampling. It is never imported by `structbench`, following ADR-0058's `data`
   extra.
5. **The package side is `core/io/abaqus.py`** (npz + deck to `Case`, mirroring
   `lsdyna_to_case`), and **`datacheck measure --declaration <file>`** lets a
   sweep that is not a registered benchmark be measured. Both are built in
   plan 2.
6. **Runs execute in a local work root** and are archived afterwards to the
   ADR-0031 tree (`raw/<name>/abaqus/<case_id>/`, `canonical/<name>/`); a
   solver never writes inside the OneDrive tree.
7. **Validator verdicts**: definitional rows pass/fail; energy and other
   solution-verification indicators are measured only (the 2026-09-21
   maintainer decision, unchanged).

## Alternatives considered

- **One driver running all stages per case**: rejected; a mid-case failure
  forces a full redo or re-grows the skip logic the stage split gives for free.
- **A workflow engine (Snakemake, doit)**: rejected; a new dependency and a DSL
  for four stages.
- **HDF5 intermediate**: rejected by measurement (clause 3).
- **scipy inline as PEP 723 script metadata**: rejected; it splits the
  environment, so sampler tests could not run in the project env.
- **Dataset definitions in the public repository from the start**: rejected;
  they are study designs until publication.

## Consequences

- New Abaqus datasets cost a `sweep.toml` and a `model.py`.
- `odb_export.py` must stay Python-3.10-parseable; a test pins it.
- ADR-0068's "not decided" item on the intermediate's format is settled.
- Not decided here: the material class for isotropic tabulated plasticity (its
  own ADR, plan 2); any schema field for axisymmetry (only if a consumer
  needs one).
```

- [ ] **Step 6: Index row.** In `decisions/README.md`, after the 0068 row:

```markdown
| 0069 | The Abaqus data-generation pipeline (four stages, shared scripts, `abaqus-npz/1`, `datagen` extra) | Durable | Proposed |
```

- [ ] **Step 7: Gates, then commit**

```bash
git add pyproject.toml uv.lock docs/PRINCIPLES.md decisions/0069-abaqus-data-pipeline.md decisions/README.md
git commit -m "docs: add ADR-0069 on the Abaqus data pipeline; add the datagen extra" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `sampling.py` — splits, regions, explicit points

**Files:**
- Create: `data_generation/abaqus/sampling.py`, `tests/tools/abaqus_paths.py`, `tests/tools/test_abaqus_sampling.py`

**Interfaces:**
- Produces:
  - `Bounds = tuple[float, float]`
  - `parse_bounds(raw: Mapping[str, Any], where: str) -> dict[str, Bounds]`
  - `Split` (frozen dataclass: `name, n, seed, exclude, within, extra, categorical, variants, points`)
  - `parse_splits(sweep: Mapping[str, Any]) -> list[Split]`
  - `Point` (frozen dataclass: `split: str, index: int, params: dict[str, float | str]`)
  - `sample_split(variables, regions, split) -> list[Point]`

- [ ] **Step 1: Path helper** `tests/tools/abaqus_paths.py`:

```python
"""Put data_generation/abaqus on sys.path so tests import its scripts by name."""

import sys
from pathlib import Path

ABAQUS_DIR = Path(__file__).resolve().parents[2] / "data_generation" / "abaqus"
if str(ABAQUS_DIR) not in sys.path:
    sys.path.insert(0, str(ABAQUS_DIR))
```

- [ ] **Step 2: Write the failing tests** `tests/tools/test_abaqus_sampling.py`:

```python
"""Tests for the shared Sobol sampler (ADR-0069). Toy variables only."""

import abaqus_paths  # noqa: F401
import pytest

pytest.importorskip("scipy")

import sampling  # noqa: E402

VARIABLES = {"a": (0.0, 1.0), "b": (10.0, 20.0)}
REGIONS = {"corner": {"a": (0.8, 1.0), "b": (18.0, 20.0)}}


def _split(**raw):
    return sampling.parse_splits({"splits": {"s": raw}})[0]


def test_same_seed_same_points_and_nested_prefixes():
    small = sampling.sample_split(VARIABLES, REGIONS, _split(n=10, seed=3))
    large = sampling.sample_split(VARIABLES, REGIONS, _split(n=20, seed=3))
    again = sampling.sample_split(VARIABLES, REGIONS, _split(n=10, seed=3))
    assert [p.params for p in small] == [p.params for p in again]
    assert [p.params for p in small] == [p.params for p in large[:10]]
    assert [p.index for p in large] == list(range(20))


def test_exclude_filters_the_region_out():
    pts = sampling.sample_split(
        VARIABLES, REGIONS, _split(n=50, seed=1, exclude=["corner"])
    )
    assert not any(p.params["a"] >= 0.8 and p.params["b"] >= 18.0 for p in pts)


def test_within_draws_inside_the_box():
    pts = sampling.sample_split(
        VARIABLES, REGIONS, _split(n=60, seed=2, within="corner")
    )
    assert len(pts) == 60
    assert all(0.8 <= p.params["a"] <= 1.0 and 18.0 <= p.params["b"] <= 20.0 for p in pts)


def test_extra_categorical_and_explicit_points():
    pts = sampling.sample_split(
        VARIABLES,
        REGIONS,
        _split(
            points=[{"a": 0.5, "b": 15.0, "c": 2.0}, {"a": 0.1, "b": 11.0, "c": 3.0}],
            extra={"c": [1.0, 5.0]},
            categorical={"kind": ["x", "y"]},
        ),
    )
    assert [p.params["kind"] for p in pts] == ["x", "y"]
    assert pts[1].params == {"a": 0.1, "b": 11.0, "c": 3.0, "kind": "y"}


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"n": 5, "seed": 1, "exclude": ["nowhere"]}, "unknown region"),
        ({"points": [{"a": 0.5}]}, "missing"),
        ({"n": 5}, "needs a seed"),
        ({"n": 5, "seed": 1, "exclude": ["corner"], "within": "corner"}, "both"),
        ({"points": [{"a": 0.9, "b": 19.0}], "exclude": ["corner"]}, "only 0 of 1"),
    ],
)
def test_misconfigured_splits_raise_naming_the_split(raw, match):
    with pytest.raises((KeyError, ValueError), match=match):
        sampling.sample_split(VARIABLES, REGIONS, _split(**raw))


def test_region_naming_unknown_variable_raises():
    regions = {"bad": {"zz": (0.0, 1.0)}}
    with pytest.raises(KeyError, match="unknown variables"):
        sampling.sample_split(VARIABLES, regions, _split(n=2, seed=1, exclude=["bad"]))


def test_bounds_must_be_increasing():
    with pytest.raises(ValueError, match="low < high"):
        sampling.parse_bounds({"a": [1.0, 1.0]}, "variables")
```

- [ ] **Step 3: Run to verify it fails**

Run: `$PY -m pytest tests/tools/test_abaqus_sampling.py -q`
Expected: `ModuleNotFoundError: No module named 'sampling'`.

- [ ] **Step 4: Implement** `data_generation/abaqus/sampling.py`:

```python
"""Scrambled-Sobol sampling of a sweep's splits, shared by every Abaqus dataset.

A split draws 1024 points from its own seeded, scrambled Sobol engine and keeps
the first ``n`` that survive its ``exclude`` regions, so the first ``k`` cases
of a split are nested subsets. A ``within`` split draws inside its region's box
rather than filtering, because a small box filtered from a global draw leaves
too few points. Explicit ``points`` bypass the engine (pilots, probes).
``categorical`` values are assigned by cycling through the list by index;
``variants`` are expanded by the generator, not here (ADR-0069).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import qmc

N_DRAW_LOG2 = 10  # 1024 points per split

Bounds = tuple[float, float]


@dataclass(frozen=True)
class Split:
    name: str
    n: int
    seed: int | None = None
    exclude: tuple[str, ...] = ()
    within: str | None = None
    extra: dict[str, Bounds] = field(default_factory=dict)
    categorical: dict[str, tuple[str, ...]] = field(default_factory=dict)
    variants: tuple[str, ...] = ()
    points: tuple[dict[str, float], ...] = ()


@dataclass(frozen=True)
class Point:
    split: str
    index: int
    params: dict[str, float | str]


def parse_bounds(raw: Mapping[str, Any], where: str) -> dict[str, Bounds]:
    """``{"a": [0, 1]}`` -> ``{"a": (0.0, 1.0)}``; each range must increase."""
    out: dict[str, Bounds] = {}
    for name, pair in raw.items():
        low, high = (float(v) for v in pair)
        if not low < high:
            raise ValueError(f"{where}.{name}: need low < high, got {pair!r}")
        out[name] = (low, high)
    return out


def parse_splits(sweep: Mapping[str, Any]) -> list[Split]:
    """The ``[splits.*]`` tables of a parsed sweep.toml, in file order."""
    splits = []
    for name, raw in sweep["splits"].items():
        points = tuple({k: float(v) for k, v in p.items()} for p in raw.get("points", ()))
        if points and "seed" in raw:
            raise ValueError(f"splits.{name}: explicit points take no seed")
        if not points and "seed" not in raw:
            raise ValueError(f"splits.{name}: a sampled split needs a seed")
        if raw.get("exclude") and raw.get("within"):
            raise ValueError(f"splits.{name}: set exclude or within, not both")
        n = len(points) if points else int(raw["n"])
        if not 0 < n <= 2**N_DRAW_LOG2:
            raise ValueError(f"splits.{name}: n must be in 1..{2**N_DRAW_LOG2}")
        splits.append(
            Split(
                name=name,
                n=n,
                seed=raw.get("seed"),
                exclude=tuple(raw.get("exclude", ())),
                within=raw.get("within"),
                extra=parse_bounds(raw.get("extra", {}), f"splits.{name}.extra"),
                categorical={k: tuple(v) for k, v in raw.get("categorical", {}).items()},
                variants=tuple(raw.get("variants", ())),
                points=points,
            )
        )
    return splits


def _inside(params: Mapping[str, float | str], region: Mapping[str, Bounds]) -> bool:
    return all(low <= float(params[k]) <= high for k, (low, high) in region.items())


def sample_split(
    variables: Mapping[str, Bounds],
    regions: Mapping[str, Mapping[str, Bounds]],
    split: Split,
) -> list[Point]:
    """The split's points, in Sobol (or listed) order, indexed from 0."""
    box = {**variables, **split.extra}
    named = (*split.exclude, *((split.within,) if split.within else ()))
    for region in named:
        if region not in regions:
            raise KeyError(f"splits.{split.name}: unknown region {region!r}")
        unknown = set(regions[region]) - set(box)
        if unknown:
            raise KeyError(f"region {region!r} names unknown variables {sorted(unknown)}")
    if split.points:
        rows = [dict(p) for p in split.points]
        for row in rows:
            missing = set(box) - set(row)
            if missing:
                raise ValueError(f"splits.{split.name}: point missing {sorted(missing)}")
    else:
        if split.within:
            box.update(regions[split.within])
        names = list(box)
        engine = qmc.Sobol(
            d=len(names), scramble=True, rng=np.random.default_rng(split.seed)
        )
        scaled = qmc.scale(
            engine.random_base2(m=N_DRAW_LOG2),
            [box[k][0] for k in names],
            [box[k][1] for k in names],
        )
        rows = [{k: float(v) for k, v in zip(names, r, strict=True)} for r in scaled]
    points: list[Point] = []
    for row in rows:
        if any(_inside(row, regions[r]) for r in split.exclude):
            continue
        index = len(points)
        params: dict[str, float | str] = dict(row)
        for key, values in split.categorical.items():
            params[key] = values[index % len(values)]
        points.append(Point(split.name, index, params))
        if len(points) == split.n:
            break
    if len(points) < split.n:
        raise ValueError(
            f"splits.{split.name}: only {len(points)} of {split.n} points "
            "survive its regions"
        )
    return points
```

- [ ] **Step 5: Run to verify it passes**

Run: `$PY -m pytest tests/tools/test_abaqus_sampling.py -q`. Expected: all pass.

- [ ] **Step 6: Gates, then commit**

```bash
git add data_generation/abaqus/sampling.py tests/tools/abaqus_paths.py tests/tools/test_abaqus_sampling.py
git commit -m "feat(datagen): shared Sobol sampler for Abaqus sweeps" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `deck.py` — keyword writers and the structured quad mesh

**Files:**
- Create: `data_generation/abaqus/deck.py`, `tests/tools/test_abaqus_deck.py`

**Interfaces:**
- Produces:
  - `num(x: float) -> str`, and `ENERGY_TERMS: tuple[str, ...]`
  - `QuadMesh` (fields `nx, ny, node_labels, coords, element_labels, connectivity`; methods `node(i, j)`, `element(i, j)`, `centroids()`)
  - `structured_quad_mesh(nx, ny, x0, x1, y0, y1) -> QuadMesh`
  - writers, each returning text ending `\n`:
    - `heading(text)`, `node_block(labels, coords)`, `element_block(element_type, elset, labels, connectivity)`
    - `node_set(name, labels)`, `element_set(name, labels)`, `element_surface(name, faces)`
    - `analytical_surface_2d(name, points)`, `rigid_body(ref_node, surface)`
    - `material(name, *, density, youngs, poisson, plastic)`, `solid_section(elset, material, *, data_line=None)`
    - `surface_interaction(name)`, `boundary(target, first_dof, last_dof=None)`, `encastre(target)`
    - `initial_velocity(nset, dof, value)`, `initial_hardening(labels, values)`, `contact_pair(slave, master, interaction)`
    - `standard_output(interval, *, node_set, node_vars, element_set, element_vars, history_node_set=None, history_node_vars=())`
    - `explicit_step(name, period, body)`

Every keyword here is a **hypothesis** until Task 7's conformance run. The tests pin text only, not solver acceptance.

- [ ] **Step 1: Write the failing tests** `tests/tools/test_abaqus_deck.py`:

```python
"""Tests for the shared Abaqus keyword writers (ADR-0069). Text only."""

import abaqus_paths  # noqa: F401
import deck
import numpy as np


def test_structured_mesh_numbering_and_orientation():
    mesh = deck.structured_quad_mesh(2, 1, 0.0, 2.0, 0.0, 1.0)
    assert mesh.node_labels.tolist() == [1, 2, 3, 4, 5, 6]
    assert mesh.coords.tolist() == [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
    assert mesh.connectivity.tolist() == [[1, 2, 5, 4], [2, 3, 6, 5]]  # CCW
    assert mesh.node(2, 1) == 6 and mesh.element(1, 0) == 2
    assert mesh.centroids().tolist() == [[0.5, 0.5], [1.5, 0.5]]


def test_numbers_round_trip_and_sets_wrap_at_sixteen():
    assert deck.num(2.5e-5) == "2.5e-05" and deck.num(3000) == "3000.0"
    text = deck.node_set("ALL", range(1, 18))
    lines = text.splitlines()
    assert lines[0] == "*NSET, NSET=ALL"
    assert lines[1].count(",") == 15 and lines[2] == "17"


def test_small_deck_snapshot():
    mesh = deck.structured_quad_mesh(1, 1, 0.0, 1.0, 0.0, 2.0)
    text = (
        deck.heading("toy")
        + deck.node_block(mesh.node_labels, mesh.coords)
        + deck.element_block("CAX4R", "E", mesh.element_labels, mesh.connectivity)
        + deck.material(
            "M", density=7.0e-9, youngs=200000.0, poisson=0.3,
            plastic=[(250.0, 0.0), (1250.0, 5.0)],
        )
        + deck.solid_section("E", "M")
        + deck.initial_velocity("N", 2, -3.0e4)
        + deck.initial_hardening([1], np.array([0.15]))
        + deck.explicit_step(
            "S", 1.0e-3,
            deck.standard_output(
                1.0e-5, node_set="N", node_vars=("U",), element_set="E",
                element_vars=("S", "PEEQ"), history_node_set="RP",
                history_node_vars=("RF2",),
            ),
        )
    )
    assert text == EXPECTED


EXPECTED = """\
*HEADING
toy
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 0.0, 2.0
4, 1.0, 2.0
*ELEMENT, TYPE=CAX4R, ELSET=E
1, 1, 2, 4, 3
*MATERIAL, NAME=M
*DENSITY
7e-09
*ELASTIC
200000.0, 0.3
*PLASTIC
250.0, 0.0
1250.0, 5.0
*SECTION CONTROLS, NAME=HG_ENHANCED, HOURGLASS=ENHANCED
*SOLID SECTION, ELSET=E, MATERIAL=M, CONTROLS=HG_ENHANCED
*INITIAL CONDITIONS, TYPE=VELOCITY
N, 2, -30000.0
*INITIAL CONDITIONS, TYPE=HARDENING
1, 0.15
*STEP, NAME=S, NLGEOM=YES
*DYNAMIC, EXPLICIT
, 0.001
*OUTPUT, FIELD, TIME INTERVAL=1e-05, TIME MARKS=YES
*NODE OUTPUT, NSET=N
U
*ELEMENT OUTPUT, ELSET=E
S, PEEQ
*OUTPUT, HISTORY, TIME INTERVAL=1e-05
*ENERGY OUTPUT
ALLAE, ALLCD, ALLFD, ALLIE, ALLKE, ALLPD, ALLSE, ALLVD, ALLWK, ETOTAL
*NODE OUTPUT, NSET=RP
RF2
*END STEP
"""


def test_contact_and_rigid_surface_writers():
    assert deck.element_surface("F", [("ROW", "S1"), ("COL", "S2")]) == (
        "*SURFACE, TYPE=ELEMENT, NAME=F\nROW, S1\nCOL, S2\n"
    )
    assert deck.analytical_surface_2d("W", [(-0.5, 0.0), (10.0, 0.0)]) == (
        "*SURFACE, TYPE=SEGMENTS, NAME=W\nSTART, -0.5, 0.0\nLINE, 10.0, 0.0\n"
    )
    assert deck.rigid_body(7, "W") == "*RIGID BODY, ANALYTICAL SURFACE=W, REF NODE=7\n"
    assert deck.surface_interaction("SMOOTH") == "*SURFACE INTERACTION, NAME=SMOOTH\n"
    assert deck.contact_pair("F", "W", "SMOOTH") == (
        "*CONTACT PAIR, INTERACTION=SMOOTH\nF, W\n"
    )
    assert deck.boundary("AXIS", 1) == "*BOUNDARY\nAXIS, 1, 1\n"
    assert deck.encastre("RP") == "*BOUNDARY\nRP, ENCASTRE\n"
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/tools/test_abaqus_deck.py -q`. Expected: `ModuleNotFoundError: No module named 'deck'`.

- [ ] **Step 3: Implement** `data_generation/abaqus/deck.py`:

```python
"""Keyword writers for Abaqus input decks, shared by every Abaqus dataset.

Every writer returns text ending in a newline. Numbers are written with
``repr(float)``, the shortest string that round-trips, so a deck is
byte-identical for identical parameters. Each keyword is a hypothesis until the
conformance run confirms it (``STANDARD_INPUT_BLOCK.md``, ADR-0068 clause 8).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Whole-model energy terms requested on the history clock (the E5 ledger).
ENERGY_TERMS = (
    "ALLAE", "ALLCD", "ALLFD", "ALLIE", "ALLKE",
    "ALLPD", "ALLSE", "ALLVD", "ALLWK", "ETOTAL",
)
_PER_LINE = 16  # labels per data line in a set


def num(x: float) -> str:
    """Shortest round-tripping text for ``x``."""
    return repr(float(x))


@dataclass(frozen=True)
class QuadMesh:
    """A structured ``nx`` x ``ny`` grid of 4-node quads, labels from 1.

    Node ``(i, j)`` is ``1 + i + j (nx + 1)``; element ``(i, j)`` is
    ``1 + i + j nx``; connectivity is counterclockwise from the lower-left
    node, so face S1 is the element's lower edge and S2 its right edge.
    """

    nx: int
    ny: int
    node_labels: NDArray[np.int64]
    coords: NDArray[np.float64]
    element_labels: NDArray[np.int64]
    connectivity: NDArray[np.int64]

    def node(self, i: int, j: int) -> int:
        return 1 + i + j * (self.nx + 1)

    def element(self, i: int, j: int) -> int:
        return 1 + i + j * self.nx

    def centroids(self) -> NDArray[np.float64]:
        """``(n_elements, 2)`` element centroids."""
        return self.coords[self.connectivity - 1].mean(axis=1)


def structured_quad_mesh(
    nx: int, ny: int, x0: float, x1: float, y0: float, y1: float
) -> QuadMesh:
    xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    i, j = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny)))
    row = nx + 1
    connectivity = np.column_stack(
        [1 + i + j * row, 2 + i + j * row, 2 + i + (j + 1) * row, 1 + i + (j + 1) * row]
    ).astype(np.int64)
    return QuadMesh(
        nx,
        ny,
        np.arange(1, row * (ny + 1) + 1, dtype=np.int64),
        coords.astype(np.float64),
        np.arange(1, nx * ny + 1, dtype=np.int64),
        connectivity,
    )


def _labels(labels: Iterable[int]) -> str:
    items = [str(int(v)) for v in labels]
    rows = [items[k : k + _PER_LINE] for k in range(0, len(items), _PER_LINE)]
    return "".join(", ".join(r) + "\n" for r in rows)


def heading(text: str) -> str:
    return f"*HEADING\n{text}\n"


def node_block(labels: Iterable[int], coords: ArrayLike) -> str:
    rows = np.asarray(coords, dtype=np.float64)
    body = "".join(
        f"{int(n)}, " + ", ".join(num(c) for c in xy) + "\n"
        for n, xy in zip(labels, rows, strict=True)
    )
    return "*NODE\n" + body


def element_block(
    element_type: str, elset: str, labels: Iterable[int], connectivity: ArrayLike
) -> str:
    conn = np.asarray(connectivity, dtype=np.int64)
    body = "".join(
        f"{int(e)}, " + ", ".join(str(int(n)) for n in nodes) + "\n"
        for e, nodes in zip(labels, conn, strict=True)
    )
    return f"*ELEMENT, TYPE={element_type}, ELSET={elset}\n" + body


def node_set(name: str, labels: Iterable[int]) -> str:
    return f"*NSET, NSET={name}\n" + _labels(labels)


def element_set(name: str, labels: Iterable[int]) -> str:
    return f"*ELSET, ELSET={name}\n" + _labels(labels)


def element_surface(name: str, faces: Sequence[tuple[str, str]]) -> str:
    return f"*SURFACE, TYPE=ELEMENT, NAME={name}\n" + "".join(
        f"{elset}, {face}\n" for elset, face in faces
    )


def analytical_surface_2d(name: str, points: Sequence[tuple[float, float]]) -> str:
    (x0, y0), rest = points[0], points[1:]
    lines = [f"START, {num(x0)}, {num(y0)}"] + [f"LINE, {num(x)}, {num(y)}" for x, y in rest]
    return f"*SURFACE, TYPE=SEGMENTS, NAME={name}\n" + "\n".join(lines) + "\n"


def rigid_body(ref_node: int, surface: str) -> str:
    return f"*RIGID BODY, ANALYTICAL SURFACE={surface}, REF NODE={ref_node}\n"


def material(
    name: str,
    *,
    density: float,
    youngs: float,
    poisson: float,
    plastic: Sequence[tuple[float, float]],
) -> str:
    table = "".join(f"{num(s)}, {num(e)}\n" for s, e in plastic)
    return (
        f"*MATERIAL, NAME={name}\n*DENSITY\n{num(density)}\n"
        f"*ELASTIC\n{num(youngs)}, {num(poisson)}\n*PLASTIC\n{table}"
    )


def solid_section(elset: str, material: str, *, data_line: str | None = None) -> str:
    text = (
        "*SECTION CONTROLS, NAME=HG_ENHANCED, HOURGLASS=ENHANCED\n"
        f"*SOLID SECTION, ELSET={elset}, MATERIAL={material}, "
        "CONTROLS=HG_ENHANCED\n"
    )
    return text + (f"{data_line}\n" if data_line is not None else "")


def surface_interaction(name: str) -> str:
    """A contact property with no friction card: frictionless."""
    return f"*SURFACE INTERACTION, NAME={name}\n"


def boundary(target: str, first_dof: int, last_dof: int | None = None) -> str:
    return f"*BOUNDARY\n{target}, {first_dof}, {last_dof or first_dof}\n"


def encastre(target: str) -> str:
    return f"*BOUNDARY\n{target}, ENCASTRE\n"


def initial_velocity(nset: str, dof: int, value: float) -> str:
    return f"*INITIAL CONDITIONS, TYPE=VELOCITY\n{nset}, {dof}, {num(value)}\n"


def initial_hardening(labels: Iterable[int], values: ArrayLike) -> str:
    vals = np.asarray(values, dtype=np.float64)
    body = "".join(
        f"{int(e)}, {num(v)}\n" for e, v in zip(labels, vals, strict=True)
    )
    return "*INITIAL CONDITIONS, TYPE=HARDENING\n" + body


def contact_pair(slave: str, master: str, interaction: str) -> str:
    """Step data under Abaqus/Explicit; the rigid surface is the master."""
    return f"*CONTACT PAIR, INTERACTION={interaction}\n{slave}, {master}\n"


def standard_output(
    interval: float,
    *,
    node_set: str,
    node_vars: Sequence[str],
    element_set: str,
    element_vars: Sequence[str],
    history_node_set: str | None = None,
    history_node_vars: Sequence[str] = (),
) -> str:
    """Field output on a fixed clock with time marks; every energy term on it too."""
    text = (
        f"*OUTPUT, FIELD, TIME INTERVAL={num(interval)}, TIME MARKS=YES\n"
        f"*NODE OUTPUT, NSET={node_set}\n{', '.join(node_vars)}\n"
        f"*ELEMENT OUTPUT, ELSET={element_set}\n{', '.join(element_vars)}\n"
        f"*OUTPUT, HISTORY, TIME INTERVAL={num(interval)}\n"
        f"*ENERGY OUTPUT\n{', '.join(ENERGY_TERMS)}\n"
    )
    if history_node_set is not None:
        variables = ", ".join(history_node_vars)
        text += f"*NODE OUTPUT, NSET={history_node_set}\n{variables}\n"
    return text


def explicit_step(name: str, period: float, body: str) -> str:
    return (
        f"*STEP, NAME={name}, NLGEOM=YES\n*DYNAMIC, EXPLICIT\n, {num(period)}\n"
        f"{body}*END STEP\n"
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/tools/test_abaqus_deck.py -q`. Expected: all pass. Then run `$PY -m ruff format data_generation/abaqus tests/tools`; ruff may reflow the `ENERGY_TERMS` tuple.

- [ ] **Step 5: Gates, then commit**

```bash
git add data_generation/abaqus/deck.py tests/tools/test_abaqus_deck.py
git commit -m "feat(datagen): shared Abaqus keyword writers and structured quad mesh" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `generate.py` — case folders, provenance, manifest

**Files:**
- Create: `data_generation/abaqus/generate.py`, `tests/tools/test_abaqus_generate.py`

**Interfaces:**
- Consumes: `sampling.parse_bounds`, `sampling.parse_splits`, `sampling.sample_split` (Task 2); a dataset `model.py` exposing `build(params: dict, variant: str | None) -> str`.
- Produces:
  - `CaseSpec(case_id, split, index, variant, seed, params)`
  - `load_sweep(dataset_dir) -> dict`, `load_model(dataset_dir) -> ModuleType`
  - `plan_cases(sweep) -> list[CaseSpec]`
  - `write_case(case_dir, deck_text, provenance, *, force) -> Literal["written", "unchanged", "conflict", "locked"]`
  - `git_state(path) -> {"commit": str, "dirty": bool}`
  - `main(argv) -> int`
  - Folder layout `<work-root>/<dataset.name>/<case_id>/{<case_id>.inp, provenance.json}` plus `<work-root>/<dataset.name>/manifest.csv`.
  - `provenance.json` keys: `format`, `case_id`, `dataset`, `units`, `split`, `index`, `variant`, `seed`, `params`, `sweep_sha256`, `inp_sha256`, `repository`, `dataset_repository`, `numpy`, `scipy`, `created_utc`. The runner reads `units` and `split`.

- [ ] **Step 1: Write the failing tests** `tests/tools/test_abaqus_generate.py`:

```python
"""Tests for the sweep generator (ADR-0069). A toy dataset, never a real one."""

import csv
import json
import subprocess

import abaqus_paths  # noqa: F401
import pytest

pytest.importorskip("scipy")

import generate  # noqa: E402

TOY_MODEL = '''
import deck

def build(params, variant):
    mesh = deck.structured_quad_mesh(1, 1, 0.0, params["a"], 0.0, 1.0)
    nodes = deck.node_block(mesh.node_labels, mesh.coords)
    return deck.heading(f"toy {variant}") + nodes
'''
TOY_SWEEP = """
[dataset]
name = "toy"
case_prefix = "TOY"
units = "t-mm-s"

[fixed]
k = 2.0

[variables]
a = [1.0, 2.0]

[splits.main]
n = {n}
seed = 5

[splits.listed]
points = [{{ a = 1.5 }}]
variants = ["x", "y"]
"""


def _dataset(tmp_path, n=3, model=TOY_MODEL):
    ds = tmp_path / "ds"
    ds.mkdir(exist_ok=True)
    (ds / "sweep.toml").write_text(TOY_SWEEP.format(n=n), encoding="utf-8")
    (ds / "model.py").write_text(model, encoding="utf-8")
    git = ["git", "-C", str(ds), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "toy", "--allow-empty"], check=True)
    return ds


def _run(ds, work, *extra):
    return generate.main(["--dataset", str(ds), "--work-root", str(work), *extra])


def test_writes_cases_provenance_and_manifest(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    assert _run(ds, work) == 0
    ids = sorted(p.name for p in (work / "toy").iterdir() if p.is_dir())
    assert ids == [
        "TOY-listed-0000-x", "TOY-listed-0000-y",
        "TOY-main-0000", "TOY-main-0001", "TOY-main-0002",
    ]
    prov = json.loads((work / "toy/TOY-listed-0000-y/provenance.json").read_text())
    assert prov["variant"] == "y" and prov["split"] == "listed" and prov["units"] == "t-mm-s"
    assert prov["params"] == {"a": 1.5, "k": 2.0}
    assert set(prov["dataset_repository"]) == {"commit", "dirty"}
    rows = list(csv.DictReader((work / "toy/manifest.csv").open(encoding="utf-8")))
    assert [r["case_id"] for r in rows][:3] == ids[2:] and set(rows[0]) >= {"a", "k"}


def test_rerun_is_byte_identical_and_skips(tmp_path, capsys):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    _run(ds, work)
    before = (work / "toy/TOY-main-0001/TOY-main-0001.inp").read_bytes()
    assert _run(ds, work) == 0
    assert (work / "toy/TOY-main-0001/TOY-main-0001.inp").read_bytes() == before
    assert "unchanged=5" in capsys.readouterr().out


def test_raising_n_appends_only(tmp_path):
    work = tmp_path / "work"
    _run(_dataset(tmp_path, n=3), work)
    first = (work / "toy/TOY-main-0002/TOY-main-0002.inp").read_bytes()
    assert _run(_dataset(tmp_path, n=5), work) == 0
    assert (work / "toy/TOY-main-0002/TOY-main-0002.inp").read_bytes() == first
    assert (work / "toy/TOY-main-0004").is_dir()


def test_changed_deck_conflicts_then_force_rewrites_but_never_a_run(tmp_path):
    work = tmp_path / "work"
    _run(_dataset(tmp_path), work)
    (work / "toy/TOY-main-0000/run.json").write_text("{}")
    changed = _dataset(tmp_path, model=TOY_MODEL.replace("toy {variant}", "toy2"))
    assert _run(changed, work) == 1  # conflicts
    assert _run(changed, work, "--force") == 1  # the case with run.json stays locked
    assert b"toy2" in (work / "toy/TOY-main-0001/TOY-main-0001.inp").read_bytes()
    assert b"toy2" not in (work / "toy/TOY-main-0000/TOY-main-0000.inp").read_bytes()


def test_unsafe_case_ids_are_refused(tmp_path):
    sweep = {
        "dataset": {"name": "x", "case_prefix": "X" * 40, "units": "t-mm-s"},
        "variables": {"a": [0.0, 1.0]},
        "splits": {"s": {"n": 1, "seed": 1}},
    }
    with pytest.raises(ValueError, match="job name"):
        generate.plan_cases(sweep)


def test_fixed_and_sampled_names_must_not_collide():
    sweep = {
        "dataset": {"name": "x", "case_prefix": "X", "units": "t-mm-s"},
        "fixed": {"a": 1.0},
        "variables": {"a": [0.0, 1.0]},
        "splits": {"s": {"n": 1, "seed": 1}},
    }
    with pytest.raises(ValueError, match="both fixed and sampled"):
        generate.plan_cases(sweep)
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/tools/test_abaqus_generate.py -q`. Expected: `ModuleNotFoundError: No module named 'generate'`.

- [ ] **Step 3: Implement** `data_generation/abaqus/generate.py`:

```python
"""Generate a sweep's Abaqus case folders from a dataset's sweep.toml and model.py.

    python data_generation/abaqus/generate.py --dataset <dir> --work-root C:\\structbench-runs
        [--split NAME ...] [--force] [--dry-run]

Each case gets ``<work-root>/<name>/<case_id>/<case_id>.inp`` and a
``provenance.json``; the sweep gets a ``manifest.csv`` listing every case. An
existing identical deck is skipped. A different one is a conflict unless
``--force``, and a deck whose case already ran (``run.json``) is never replaced
(ADR-0069).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np
import sampling
import scipy

_REPO = Path(__file__).resolve().parents[2]
PROVENANCE_FORMAT = "abaqus-provenance/1"
#: A conservative Abaqus job-name rule; the conformance run confirms or relaxes it.
_CASE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,37}$")

Outcome = Literal["written", "unchanged", "conflict", "locked"]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    split: str
    index: int
    variant: str | None
    seed: int | None
    params: dict[str, float | str]


def load_sweep(dataset_dir: Path) -> dict[str, Any]:
    with (dataset_dir / "sweep.toml").open("rb") as handle:
        return tomllib.load(handle)


def load_model(dataset_dir: Path) -> ModuleType:
    """Import the dataset's model.py by path; it must define ``build``."""
    path = dataset_dir / "model.py"
    spec = importlib.util.spec_from_file_location(f"dataset_model_{dataset_dir.name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.name} from {dataset_dir.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build", None)):
        raise AttributeError(f"{dataset_dir.name}/model.py defines no build()")
    return module


def plan_cases(sweep: dict[str, Any]) -> list[CaseSpec]:
    """Every case of every split, in file then Sobol order."""
    prefix = sweep["dataset"]["case_prefix"]
    fixed = dict(sweep.get("fixed", {}))
    variables = sampling.parse_bounds(sweep["variables"], "variables")
    regions = {
        name: sampling.parse_bounds(box, f"regions.{name}")
        for name, box in sweep.get("regions", {}).items()
    }
    specs = []
    for split in sampling.parse_splits(sweep):
        clash = set(fixed) & (set(variables) | set(split.extra) | set(split.categorical))
        if clash:
            raise ValueError(f"{sorted(clash)} are both fixed and sampled")
        for point in sampling.sample_split(variables, regions, split):
            for variant in split.variants or (None,):
                case_id = f"{prefix}-{split.name}-{point.index:04d}"
                case_id += f"-{variant}" if variant else ""
                if not _CASE_ID.fullmatch(case_id):
                    raise ValueError(f"case id {case_id!r} is not a safe Abaqus job name")
                params = {**fixed, **point.params}
                specs.append(
                    CaseSpec(case_id, split.name, point.index, variant, split.seed, params)
                )
    return specs


def git_state(path: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(f"{path.name} is not a git repository with a commit") from exc


def write_case(
    case_dir: Path, deck_text: str, provenance: dict[str, Any], *, force: bool
) -> Outcome:
    inp = case_dir / f"{case_dir.name}.inp"
    data = deck_text.encode("utf-8")
    if inp.exists():
        if inp.read_bytes() == data:
            return "unchanged"
        if (case_dir / "run.json").exists():
            return "locked"
        if not force:
            return "conflict"
    case_dir.mkdir(parents=True, exist_ok=True)
    inp.write_bytes(data)
    (case_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return "written"


def write_manifest(sweep_dir: Path, specs: list[CaseSpec]) -> None:
    names = sorted({k for s in specs for k in s.params})
    with (sweep_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case_id", "split", "index", "variant", *names])
        for s in specs:
            row = [s.params.get(k, "") for k in names]
            writer.writerow([s.case_id, s.split, s.index, s.variant or "", *row])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--split", action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    dataset_dir = args.dataset.resolve()
    sweep = load_sweep(dataset_dir)
    specs = plan_cases(sweep)
    selected = [s for s in specs if not args.split or s.split in args.split]
    if args.dry_run:
        for split, count in Counter(s.split for s in selected).items():
            print(f"{split}: {count} cases")
        return 0

    model = load_model(dataset_dir)
    sweep_dir = args.work_root / sweep["dataset"]["name"]
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_sha = hashlib.sha256((dataset_dir / "sweep.toml").read_bytes()).hexdigest()
    repo, dataset_repo = git_state(_REPO), git_state(dataset_dir)
    if dataset_repo["dirty"]:
        print("warning: the dataset repository has uncommitted changes", file=sys.stderr)
    counts: Counter[str] = Counter()
    problems = []
    for spec in selected:
        text = model.build(dict(spec.params), spec.variant)
        provenance = {
            "format": PROVENANCE_FORMAT,
            "case_id": spec.case_id,
            "dataset": sweep["dataset"]["name"],
            "units": sweep["dataset"]["units"],
            "split": spec.split,
            "index": spec.index,
            "variant": spec.variant,
            "seed": spec.seed,
            "params": spec.params,
            "sweep_sha256": sweep_sha,
            "inp_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "repository": repo,
            "dataset_repository": dataset_repo,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        outcome = write_case(sweep_dir / spec.case_id, text, provenance, force=args.force)
        counts[outcome] += 1
        if outcome in ("conflict", "locked"):
            problems.append(f"{spec.case_id}: {outcome}")
    write_manifest(sweep_dir, specs)
    print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/tools/test_abaqus_generate.py -q`. Expected: all pass.

- [ ] **Step 5: Gates, then commit**

```bash
git add data_generation/abaqus/generate.py tests/tools/test_abaqus_generate.py
git commit -m "feat(datagen): sweep generator with provenance and a manifest" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The private dataset repository (outside StructBench)

**Files (all outside this repository):**
- Create: `<private-repo>/README.md`, `<private-repo>/abaqus/<dataset>/sweep.toml`, `<private-repo>/abaqus/<dataset>/model.py`

The content is in the gitignored companion note in `scratch/`. It is **not** reproduced here (Global Constraints, public-repo rule).

**Interfaces:**
- Consumes: the `deck.py` writers (Task 3) and the `generate.py` CLI (Task 4).
- Produces: a committed private dataset folder that `generate.py --dataset` accepts. Its splits include `pilot` (explicit corner points) and `probe` (conformance cases). The companion note lists the rest and names one probe case, called `<PROBE_CASE>` below.

- [ ] **Step 1: Create the repo and copy the three files verbatim from the companion note**

```bash
mkdir -p <private-repo>/abaqus/<dataset> && cd <private-repo> && git init -q
# write README.md, abaqus/<dataset>/sweep.toml, abaqus/<dataset>/model.py from the companion note
git add . && git commit -q -m "feat: dataset definition (private)"
```
Also move the companion note from StructBench `scratch/` into `<private-repo>/abaqus/<dataset>/NOTES.md`, so it lives with the files it describes.

- [ ] **Step 2: Dry run (counts only; no model code runs)**

```bash
cd /c/Users/kylin/Desktop/StructBench && $PY data_generation/abaqus/generate.py --dataset <private-repo>/abaqus/<dataset> --work-root /c/structbench-runs --dry-run
```
Expected: one line per split, and the counts equal those in the companion note's "Expected checks".

- [ ] **Step 3: Generate only the pilot and probe decks, then read one**

```bash
$PY data_generation/abaqus/generate.py --dataset <private-repo>/abaqus/<dataset> --work-root /c/structbench-runs --split pilot --split probe
head -c 1500 /c/structbench-runs/<dataset>/<PROBE_CASE>/<PROBE_CASE>.inp
```
Expected: `written=7`. Check the deck by eye against the node, element and line counts in the companion note's "Expected checks".

Nothing is committed to StructBench in this task.

---

### Task 6: `run_jobs.py` — the runner

**Files:**
- Create: `data_generation/abaqus/run_jobs.py`, `tests/tools/test_abaqus_run_jobs.py`

**Interfaces:**
- Consumes: case folders from Task 4, reading `provenance.json` keys `units` and `split`; and `structbench.core.io.abaqus_run.read_abaqus_run_evidence(*, status_text, messages_text, printed_text, source_units) -> RunEvidence` (existing).
- Produces:
  - `JobResult(case_id, status, wall_s, return_code)`, where status is one of `completed`, `error`, `unrecognised`, `timeout`, `launch_error`
  - `case_state(case_dir) -> "pending" | "done" | "failed" | "interrupted"`
  - `run_sweep(sweep, abaqus: list[str], *, cases=None, splits=None, limit=None, workers=6, timeout=None, retry_failed=False, dry_run=False, echo=print) -> list[JobResult]`
  - `main(argv) -> int`
  - Per case: `run.json` with keys `case_id, command, start_utc, end_utc, wall_s, return_code, status, abaqus_version, n_errors, n_warnings, termination, unparsable`. Per sweep: `run_log.csv`.
  - Plan 2's exporter and validator key on `run.json["status"] == "completed"`.

- [ ] **Step 1: Write the failing tests** `tests/tools/test_abaqus_run_jobs.py`. The fake solver takes its behaviour from the case id (`C-ok-0`, `C-reject-0`, `C-hang-0`):

```python
"""Tests for the Abaqus runner (ADR-0069) against a fake solver. No Abaqus."""

import csv
import json
import sys

import abaqus_paths  # noqa: F401
import pytest
import run_jobs

FAKE = r'''
import pathlib, sys, time
args = sys.argv[1:]
if args[0] == "terminate":
    sys.exit(0)
job = next(a.split("=", 1)[1] for a in args if a.startswith("job="))
here = pathlib.Path.cwd()
(here / f"{job}.lck").write_text("")
mode = job.split("-")[1]
done = " THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n"
counts = "      0 ERROR MESSAGES\n      0 WARNING MESSAGES\n"
fatal = " THE PROGRAM HAS DISCOVERED     1 FATAL ERRORS\n"
if mode == "hang":
    time.sleep(60)
elif mode == "ok":
    (here / f"{job}.sta").write_text(" Abaqus/Explicit 2025\n" + done)
    (here / f"{job}.msg").write_text(" Abaqus 2025\n" + counts)
    (here / f"{job}.dat").write_text(" Abaqus 2025\n")
elif mode == "reject":
    (here / f"{job}.dat").write_text(" Abaqus 2025\n" + fatal)
(here / f"{job}.lck").unlink()
sys.exit(0 if mode == "ok" else 1)
'''


@pytest.fixture
def abaqus(tmp_path):
    fake = tmp_path / "fake_abaqus.py"
    fake.write_text(FAKE, encoding="utf-8")
    return [sys.executable, str(fake)]


def _case(sweep, case_id, split="s"):
    d = sweep / case_id
    d.mkdir(parents=True)
    (d / f"{case_id}.inp").write_text("*HEADING\n", encoding="utf-8")
    (d / "provenance.json").write_text(json.dumps({"units": "t-mm-s", "split": split}))
    return d


def test_completed_runs_record_and_skip_on_rerun(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    for k in range(3):
        _case(sweep, f"C-ok-{k}")
    results = run_jobs.run_sweep(sweep, abaqus, workers=2, echo=lambda s: None)
    assert sorted(r.status for r in results) == ["completed"] * 3
    record = json.loads((sweep / "C-ok-0/run.json").read_text())
    assert record["abaqus_version"] == "2025" and record["n_errors"] == 0
    assert record["command"][0].startswith("python")  # the name only, never a path
    rows = list(csv.DictReader((sweep / "run_log.csv").open(encoding="utf-8")))
    assert len(rows) == 3
    assert run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None) == []


def test_rejected_job_is_error_and_retried_only_on_request(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    _case(sweep, "C-reject-0")
    assert run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None)[0].status == "error"
    assert run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None) == []
    again = run_jobs.run_sweep(sweep, abaqus, retry_failed=True, echo=lambda s: None)
    assert again[0].status == "error"
    assert (sweep / "C-reject-0/attempts/1/run.json").is_file()


def test_interrupted_case_is_moved_aside_and_rerun(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    d = _case(sweep, "C-ok-9")
    (d / "C-ok-9.lck").write_text("")
    (d / "C-ok-9.sta").write_text("partial")
    assert run_jobs.case_state(d) == "interrupted"
    assert run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None)[0].status == "completed"
    assert (d / "attempts/1/C-ok-9.sta").read_text() == "partial"


def test_timeout_stops_the_job(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    _case(sweep, "C-hang-0")
    result = run_jobs.run_sweep(sweep, abaqus, timeout=2, echo=lambda s: None)[0]
    assert result.status == "timeout" and result.wall_s < 30


def test_split_filter_and_limit(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    _case(sweep, "C-ok-0", split="a")
    _case(sweep, "C-ok-1", split="b")
    _case(sweep, "C-ok-2", split="b")
    out = run_jobs.run_sweep(sweep, abaqus, splits=["b"], limit=1, echo=lambda s: None)
    assert [r.case_id for r in out] == ["C-ok-1"]


def test_a_held_lock_refuses_a_second_runner(tmp_path, abaqus):
    sweep = tmp_path / "sweep"
    _case(sweep, "C-ok-0")
    (sweep / ".runner.lock").write_text("123")
    with pytest.raises(RuntimeError, match=r"\.runner\.lock"):
        run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None)


def test_whitespace_in_the_sweep_path_is_refused(tmp_path, abaqus):
    sweep = tmp_path / "One Drive" / "sweep"
    _case(sweep, "C-ok-0")
    with pytest.raises(ValueError, match="whitespace"):
        run_jobs.run_sweep(sweep, abaqus, echo=lambda s: None)
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/tools/test_abaqus_run_jobs.py -q`. Expected: `ModuleNotFoundError: No module named 'run_jobs'`.

- [ ] **Step 3: Implement** `data_generation/abaqus/run_jobs.py`:

```python
"""Run a sweep's Abaqus decks, N at a time, and record each run.

    python data_generation/abaqus/run_jobs.py --sweep C:\\structbench-runs\\<name>
        [--split NAME ...] [--cases ID ...] [--limit N] [--workers 6]
        [--timeout S] [--retry-failed] [--abaqus EXE] [--dry-run]

Solver-generic: it knows only the case-folder layout the generator writes. A
case is done once its ``run.json`` exists; nothing is deleted -- an interrupted
or retried attempt moves to ``attempts/<n>/``. Status comes from
``read_abaqus_run_evidence``, the reader the validator uses, so "did it finish"
has one answer everywhere (ADR-0069).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from structbench.core.io.abaqus_run import read_abaqus_run_evidence

LOCK_NAME = ".runner.lock"
_KEEP = {"provenance.json", "attempts"}


@dataclass(frozen=True)
class JobResult:
    case_id: str
    status: str
    wall_s: float
    return_code: int | None


def _cases(sweep: Path) -> list[Path]:
    return sorted(
        p for p in sweep.iterdir() if p.is_dir() and (p / f"{p.name}.inp").is_file()
    )


def case_state(case_dir: Path) -> str:
    run = case_dir / "run.json"
    if run.is_file():
        status = json.loads(run.read_text(encoding="utf-8"))["status"]
        return "done" if status == "completed" else "failed"
    keep = _KEEP | {f"{case_dir.name}.inp"}
    return "interrupted" if any(p.name not in keep for p in case_dir.iterdir()) else "pending"


def _move_attempt(case_dir: Path) -> None:
    root = case_dir / "attempts"
    n = 1 + (sum(1 for p in root.iterdir() if p.is_dir()) if root.is_dir() else 0)
    target = root / str(n)
    target.mkdir(parents=True)
    keep = _KEEP | {f"{case_dir.name}.inp"}
    for item in list(case_dir.iterdir()):
        if item.name not in keep:
            shutil.move(str(item), str(target / item.name))


def _classify(case_dir: Path, units: str) -> tuple[str, dict[str, Any]]:
    def text(ext: str) -> str | None:
        path = case_dir / f"{case_dir.name}{ext}"
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None

    evidence = read_abaqus_run_evidence(
        status_text=text(".sta"),
        messages_text=text(".msg"),
        printed_text=text(".dat"),
        source_units=units,
    )
    records = evidence.termination or ()
    if any(r.status == "error" for r in records):
        status = "error"
    elif records and all(r.status == "normal" for r in records):
        status = "completed"
    else:
        status = "unrecognised"
    return status, {
        "abaqus_version": evidence.identity.version if evidence.identity else None,
        "n_errors": evidence.n_errors,
        "n_warnings": evidence.n_warnings,
        "termination": [r.status for r in records],
        "unparsable": sorted(evidence.unparsable),
    }


def _stop(proc: subprocess.Popen[bytes], abaqus: list[str], case_id: str, cwd: Path) -> None:
    try:
        subprocess.run(
            [*abaqus, "terminate", f"job={case_id}"], cwd=cwd, capture_output=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    if proc.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.kill()
    proc.wait()


def _run_one(case_dir: Path, abaqus: list[str], timeout: float | None) -> JobResult:
    case_id = case_dir.name
    units = json.loads((case_dir / "provenance.json").read_text(encoding="utf-8"))["units"]
    command = [*abaqus, f"job={case_id}", f"input={case_id}.inp", "double=both", "cpus=1", "interactive"]
    start, t0 = datetime.now(UTC), time.monotonic()
    return_code: int | None = None
    facts: dict[str, Any] = {}
    try:
        with (case_dir / "runner.log").open("wb") as log:
            proc = subprocess.Popen(command, cwd=case_dir, stdout=log, stderr=subprocess.STDOUT)
            try:
                return_code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _stop(proc, abaqus, case_id, case_dir)
                status = "timeout"
            else:
                status, facts = _classify(case_dir, units)
    except OSError as exc:
        status, facts = "launch_error", {"error": type(exc).__name__}
    wall = time.monotonic() - t0
    record = {
        "case_id": case_id,
        "command": [Path(command[0]).name, *command[1:]],
        "start_utc": start.isoformat(timespec="seconds"),
        "end_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "wall_s": round(wall, 1),
        "return_code": return_code,
        "status": status,
        **facts,
    }
    (case_dir / "run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return JobResult(case_id, status, wall, return_code)


def _append_log(sweep: Path, result: JobResult) -> None:
    path = sweep / "run_log.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["case_id", "status", "wall_s", "return_code", "logged_utc"])
        writer.writerow(
            [result.case_id, result.status, round(result.wall_s, 1), result.return_code,
             datetime.now(UTC).isoformat(timespec="seconds")]
        )


def run_sweep(
    sweep: Path,
    abaqus: list[str],
    *,
    cases: list[str] | None = None,
    splits: list[str] | None = None,
    limit: int | None = None,
    workers: int = 6,
    timeout: float | None = None,
    retry_failed: bool = False,
    dry_run: bool = False,
    echo: Callable[[str], None] = print,
) -> list[JobResult]:
    if any(ch.isspace() for ch in str(sweep.resolve())):
        raise ValueError(
            "the sweep path contains whitespace, which Abaqus job folders "
            f"must not: {sweep.name}"
        )
    chosen: list[tuple[Path, str]] = []
    for case_dir in _cases(sweep):
        if cases and case_dir.name not in cases:
            continue
        if splits:
            prov = json.loads((case_dir / "provenance.json").read_text(encoding="utf-8"))
            if prov["split"] not in splits:
                continue
        state = case_state(case_dir)
        if state == "done" or (state == "failed" and not retry_failed):
            continue
        chosen.append((case_dir, state))
    chosen = chosen[:limit] if limit else chosen
    if dry_run:
        for case_dir, state in chosen:
            echo(f"{case_dir.name} {state}")
        return []
    lock = sweep / LOCK_NAME
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"{LOCK_NAME} exists: another runner holds this sweep, or one crashed. "
            "Check, then delete it."
        ) from None
    with os.fdopen(fd, "w") as fh:
        fh.write(str(os.getpid()))
    results: list[JobResult] = []
    log_lock = threading.Lock()
    try:
        for case_dir, state in chosen:
            if state in ("interrupted", "failed"):
                _move_attempt(case_dir)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_one, d, abaqus, timeout) for d, _ in chosen]
            for k, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                with log_lock:
                    _append_log(sweep, result)
                progress = f"[{k}/{len(chosen)}] {result.case_id}"
                echo(f"{progress} {result.status} {result.wall_s:.0f} s")
    finally:
        lock.unlink(missing_ok=True)
    return results


def _summarise(results: list[JobResult]) -> None:
    counts = Counter(r.status for r in results)
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for r in sorted(results, key=lambda r: -r.wall_s)[:3]:
        print(f"slowest: {r.case_id} {r.wall_s:.0f} s")
    for r in results:
        if r.status != "completed":
            print(f"FAILED: {r.case_id} {r.status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--split", action="append")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--abaqus", default="abaqus")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    exe = shutil.which(args.abaqus)
    if exe is None:
        print(f"abaqus executable {args.abaqus!r} not found", file=sys.stderr)
        return 2
    try:
        results = run_sweep(
            args.sweep, [exe], cases=args.cases, splits=args.split, limit=args.limit,
            workers=args.workers, timeout=args.timeout, retry_failed=args.retry_failed,
            dry_run=args.dry_run,
        )
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    _summarise(results)
    return 1 if any(r.status != "completed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/tools/test_abaqus_run_jobs.py -q`. Expected: all pass, with the timeout test taking about 2–5 s. Then run `$PY -m ruff format data_generation/abaqus tests/tools` to reflow the long lines.

- [ ] **Step 5: Gates, then commit**

```bash
git add data_generation/abaqus/run_jobs.py tests/tools/test_abaqus_run_jobs.py
git commit -m "feat(datagen): Abaqus runner with resumable case state and a run log" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: The conformance run (seven real jobs)

**Files:** none in the repository. Notes go to `scratch/2026-09-24-abaqus-conformance-notes.md` (gitignored).

**Interfaces:**
- Consumes: the Task 5 decks in `C:\structbench-runs\<dataset>\` and the Task 6 runner.
- Produces: seven finished case folders, with ODBs, for Task 8; plus notes of the observed facts, each naming the file it came from, for Task 9 and Plan 2.

- [ ] **Step 1: Ask the maintainer.** "Seven Abaqus/Explicit jobs, single core each, 6 at once. The plan estimated 1–5 min per job, so about 10 minutes of wall time. OK to run?" Wait for a yes (CLAUDE.md, flag-first on compute).

- [ ] **Step 2: Check the two riskiest keyword choices against the documentation, if it's installed.** Look for the Keywords Reference under `C:\SIMULIA` without a recursive scan: check `C:\SIMULIA\EstProducts\2025\win_b64\docs` first. Two things matter:
  - the normal direction of a 2D `*SURFACE, TYPE=SEGMENTS`;
  - the data-line format of `*INITIAL CONDITIONS, TYPE=HARDENING`.

  If the documentation isn't there, note that, and let Step 4's checks decide.

- [ ] **Step 3: Run in the background**

```bash
cd /c/Users/kylin/Desktop/StructBench && $PY data_generation/abaqus/run_jobs.py --sweep /c/structbench-runs/<dataset> --split pilot --split probe --workers 6
```
(Use a background run and wait for the notification.) Expected: seven result lines and a summary.

- [ ] **Step 4: Read the outputs and write the notes.** Record each item below with the file it came from:
  1. **Did every job start?** If one failed with a job-name error, the hyphen question is answered: change `_CASE_ID` and the id separator in Task 4, then regenerate.
  2. **The `.sta` wording.** Its last lines (the completion message), its header line (the version token), and the column headings of the increment table (for the time-step series).
  3. **The `.msg` diagnostics summary** (ERROR and WARNING MESSAGES lines) and any warnings about contact, hourglassing or distortion.
  4. **What the runner reported.** If a case reads `unrecognised`, the reader's `_COMPLETED` pattern does not match Explicit's wording. That is expected and is fixed in Plan 2 Task 1, not here.
  5. **Runtime per case**, from `run_log.csv`, as the measured replacement for the plan's estimate.
  6. **Whether the pilot corners completed**, or where each aborted (distortion error text).

  Scrub the licence line whenever text is copied into the notes (STANDARD_INPUT_BLOCK.md, "Nothing of the licence line may be published").

- [ ] **Step 5: Report to the maintainer.** Give the per-case status and runtime, and whether any corner aborted. What to do about an aborted case is a data-plan decision and belongs to the maintainer. No commit.

---

### Task 8: `odb_export.py` — ODB to `abaqus-npz/1`

**Files:**
- Create: `data_generation/abaqus/odb_export.py`, `tests/tools/test_abaqus_odb_export.py`

**Interfaces:**
- Consumes: case folders whose `run.json["status"] == "completed"`, from Task 7.
- Produces: `<case_id>.npz` next to the ODB, and pure helpers testable without Abaqus:
  - `selected_cases(sweep, cases=None) -> list[Path]`
  - `stack_blocks(blocks) -> dict`, with keys `data (m, c)`, `labels (m,)`, `label_kind ("node" | "element")`, `integration_points ((m,) or None)`
  - `time_series(per_frame: list[dict]) -> dict`, the same keys with `data` shaped `(T, m, c)`
  - `export(odb_path, out_path) -> dict`, which returns the manifest
  - Key layout (documented in the module docstring): `manifest`; `mesh/<instance>/node_labels|node_coords`; `mesh/<instance>/elements/<type>/labels|connectivity`; `step/<step>/frame_times`; `field/<step>/<name>/<instance>/data|node_labels|element_labels|integration_points`; `history/<step>/<region>/<output>`.
  - Manifest keys: `format` (`"abaqus-npz/1"`), `odb_sha256`, `abaqus_release`, `precision`, `materials`, `sections`, `fields` (per key: `component_labels`, `position`, `label_kind`), `skipped`.

- [ ] **Step 1: Write the failing tests** `tests/tools/test_abaqus_odb_export.py`:

```python
"""Tests for the ODB exporter's pure helpers (ADR-0069). No Abaqus needed,
except the env-gated end-to-end test."""

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import abaqus_paths
import numpy as np
import odb_export
import pytest


def test_parses_as_python_3_10():
    source = (abaqus_paths.ABAQUS_DIR / "odb_export.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 10))
    assert "\nimport odbAccess" not in source and "\nfrom odbAccess" not in source


def _block(data, nodes=(), elements=(), ips=()):
    return SimpleNamespace(
        data=np.asarray(data), nodeLabels=np.asarray(nodes),
        elementLabels=np.asarray(elements), integrationPoints=np.asarray(ips),
    )


def test_stack_blocks_nodal_and_scalar_element():
    nodal = odb_export.stack_blocks(
        [_block([[1.0, 2.0]], nodes=[1]), _block([[3.0, 4.0]], nodes=[2])]
    )
    assert nodal["label_kind"] == "node" and nodal["labels"].tolist() == [1, 2]
    assert nodal["data"].shape == (2, 2) and nodal["integration_points"] is None
    scalar = odb_export.stack_blocks([_block([0.1, 0.2], elements=[7, 8], ips=[1, 1])])
    assert scalar["label_kind"] == "element" and scalar["data"].shape == (2, 1)


def test_time_series_refuses_changing_labels():
    a = odb_export.stack_blocks([_block([[1.0]], nodes=[1])])
    b = odb_export.stack_blocks([_block([[1.0]], nodes=[2])])
    assert odb_export.time_series([a, a])["data"].shape == (2, 1, 1)
    with pytest.raises(ValueError, match="labels change"):
        odb_export.time_series([a, b])


def test_selects_only_completed_unexported_cases(tmp_path):
    for name, status, exported in [
        ("A", "completed", False), ("B", "error", False), ("C", "completed", True),
    ]:
        d = tmp_path / name
        d.mkdir()
        (d / "run.json").write_text(json.dumps({"status": status}))
        if exported:
            (d / f"{name}.npz").write_bytes(b"")
    (tmp_path / "D").mkdir()  # never ran
    assert [p.name for p in odb_export.selected_cases(tmp_path)] == ["A"]


@pytest.mark.skipif(
    "STRUCTBENCH_ABAQUS_RUN_DIR" not in os.environ,
    reason="needs a finished Abaqus case folder",
)
def test_exports_a_real_odb(tmp_path):
    source = Path(os.environ["STRUCTBENCH_ABAQUS_RUN_DIR"])
    case = tmp_path / source.name
    case.mkdir()
    shutil.copy2(source / f"{source.name}.odb", case)
    (case / "run.json").write_text(json.dumps({"status": "completed"}))
    exe = shutil.which("abaqus")
    assert exe is not None
    subprocess.run(
        [exe, "python", str(abaqus_paths.ABAQUS_DIR / "odb_export.py"), "--sweep", str(tmp_path)],
        check=True,
    )
    with np.load(case / f"{source.name}.npz", allow_pickle=False) as npz:
        manifest = json.loads(str(npz["manifest"]))
        assert manifest["format"] == "abaqus-npz/1"
        times = [k for k in npz.files if k.endswith("/frame_times")]
        assert times and npz[times[0]][0] == 0.0
        assert any(k.startswith("history/") for k in npz.files)
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/tools/test_abaqus_odb_export.py -q`. Expected: `ModuleNotFoundError: No module named 'odb_export'`.

- [ ] **Step 3: Implement** `data_generation/abaqus/odb_export.py`. The ODB API calls are the ones Step 5 checks against a real ODB:

```python
"""Export Abaqus output databases to the neutral ``abaqus-npz/1`` intermediate.

Runs under Abaqus's own interpreter (Python 3.10, numpy, odbAccess)::

    abaqus python data_generation/abaqus/odb_export.py --sweep C:\\structbench-runs\\<name> [--cases ID ...]

Dataset-blind: it writes whatever the ODB holds, for cases whose ``run.json``
says ``completed`` and that have no ``<case_id>.npz`` yet. It must stay parseable
as Python 3.10 and import ``odbAccess`` only inside :func:`export` (ADR-0069).

Layout of ``<case_id>.npz`` (``np.savez_compressed``, no pickles)::

    manifest                                        0-d str: JSON (format, odb_sha256,
                                                    abaqus_release, precision, materials,
                                                    sections, fields, skipped)
    mesh/<instance>/node_labels                     (n,) int64
    mesh/<instance>/node_coords                     (n, d) float64
    mesh/<instance>/elements/<type>/labels          (e,) int64
    mesh/<instance>/elements/<type>/connectivity    (e, k) int64 node labels
    step/<step>/frame_times                         (T,) float64, step time
    field/<step>/<name>/<instance>/data             (T, m, c) as stored
    field/<step>/<name>/<instance>/node_labels      (m,) int64, or element_labels
    field/<step>/<name>/<instance>/integration_points  (m,) int64, element fields
    history/<step>/<region>/<output>                (s, 2) float64 (time, value)

Units are the deck's own; ids are Abaqus labels, never minted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

FORMAT = "abaqus-npz/1"
ASSEMBLY = "ASSEMBLY"  # instance key for assembly-level (instance-less) data


def selected_cases(sweep: Path, cases: list[str] | None = None) -> list[Path]:
    """Case folders whose run completed and that have no export yet."""
    out = []
    for case_dir in sorted(p for p in sweep.iterdir() if p.is_dir()):
        if cases and case_dir.name not in cases:
            continue
        run = case_dir / "run.json"
        if not run.is_file() or (case_dir / f"{case_dir.name}.npz").exists():
            continue
        if json.loads(run.read_text(encoding="utf-8")).get("status") != "completed":
            continue
        out.append(case_dir)
    return out


def stack_blocks(blocks: list) -> dict:
    """Concatenate one frame's bulk-data blocks for one instance."""
    data = [np.asarray(b.data) for b in blocks]
    data = [d.reshape(-1, 1) if d.ndim == 1 else d for d in data]
    elements = [np.asarray(b.elementLabels) for b in blocks]
    if all(e.size for e in elements):
        return {
            "data": np.concatenate(data),
            "labels": np.concatenate(elements).astype(np.int64),
            "label_kind": "element",
            "integration_points": np.concatenate(
                [np.asarray(b.integrationPoints) for b in blocks]
            ).astype(np.int64),
        }
    return {
        "data": np.concatenate(data),
        "labels": np.concatenate([np.asarray(b.nodeLabels) for b in blocks]).astype(np.int64),
        "label_kind": "node",
        "integration_points": None,
    }


def time_series(per_frame: list) -> dict:
    """Stack frames to ``(T, m, c)``; the labels must not change between frames."""
    first = per_frame[0]
    for frame in per_frame[1:]:
        if not np.array_equal(frame["labels"], first["labels"]):
            raise ValueError("labels change between frames")
    return dict(first, data=np.stack([f["data"] for f in per_frame]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(odb_path: Path, out_path: Path) -> dict:
    """Write ``out_path`` from ``odb_path``; return the manifest."""
    from odbAccess import openOdb  # Abaqus interpreter only

    arrays = {}
    manifest = {"format": FORMAT, "odb_sha256": _sha256(odb_path), "fields": {}, "skipped": []}
    odb = openOdb(str(odb_path), readOnly=True)
    try:
        manifest["abaqus_release"] = str(odb.jobData.version)
        manifest["precision"] = str(odb.jobData.precision)
        manifest["materials"] = sorted(odb.materials.keys())
        manifest["sections"] = sorted(odb.sections.keys())
        for iname, inst in odb.rootAssembly.instances.items():
            arrays[f"mesh/{iname}/node_labels"] = np.array([n.label for n in inst.nodes], dtype=np.int64)
            arrays[f"mesh/{iname}/node_coords"] = np.array([n.coordinates for n in inst.nodes], dtype=np.float64)
            by_type = {}
            for element in inst.elements:
                by_type.setdefault(element.type, []).append(element)
            for etype, elements in by_type.items():
                base = f"mesh/{iname}/elements/{etype}"
                arrays[f"{base}/labels"] = np.array([e.label for e in elements], dtype=np.int64)
                arrays[f"{base}/connectivity"] = np.array([e.connectivity for e in elements], dtype=np.int64)
        for sname, step in odb.steps.items():
            frames = step.frames
            arrays[f"step/{sname}/frame_times"] = np.array([f.frameValue for f in frames], dtype=np.float64)
            for fname in sorted(frames[0].fieldOutputs.keys()):
                if any(fname not in f.fieldOutputs.keys() for f in frames):
                    manifest["skipped"].append({"step": sname, "field": fname, "reason": "absent_in_some_frames"})
                    continue
                per_instance = {}
                positions = {}
                for frame in frames:
                    groups = {}
                    for block in frame.fieldOutputs[fname].bulkDataBlocks:
                        key = block.instance.name if block.instance is not None else ASSEMBLY
                        groups.setdefault(key, []).append(block)
                        positions[key] = str(block.position)
                    for key, blocks in groups.items():
                        per_instance.setdefault(key, []).append(stack_blocks(blocks))
                components = [str(c) for c in frames[0].fieldOutputs[fname].componentLabels]
                for key, series in per_instance.items():
                    base = f"field/{sname}/{fname}/{key}"
                    if len(series) != len(frames):
                        manifest["skipped"].append({"step": sname, "field": fname, "instance": key, "reason": "absent_in_some_frames"})
                        continue
                    stacked = time_series(series)
                    arrays[f"{base}/data"] = stacked["data"]
                    arrays[f"{base}/{stacked['label_kind']}_labels"] = stacked["labels"]
                    if stacked["integration_points"] is not None:
                        arrays[f"{base}/integration_points"] = stacked["integration_points"]
                    manifest["fields"][base] = {
                        "component_labels": components,
                        "position": positions[key],
                        "label_kind": stacked["label_kind"],
                    }
            for rname, region in step.historyRegions.items():
                for oname, output in region.historyOutputs.items():
                    arrays[f"history/{sname}/{rname}/{oname}"] = np.array(output.data, dtype=np.float64)
    finally:
        odb.close()
    arrays["manifest"] = np.array(json.dumps(manifest, sort_keys=True))
    partial = out_path.with_name(out_path.stem + ".partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(out_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args(argv)
    todo = selected_cases(args.sweep, args.cases)
    failures = 0
    for k, case_dir in enumerate(todo, 1):
        name = case_dir.name
        tag = f"[{k}/{len(todo)}] {name}"
        try:
            manifest = export(case_dir / f"{name}.odb", case_dir / f"{name}.npz")
            n_fields, n_skipped = len(manifest["fields"]), len(manifest["skipped"])
            print(f"{tag} exported: {n_fields} fields, {n_skipped} skipped")
        except Exception as exc:  # report, keep going; no partial .npz survives
            failures += 1
            print(f"{tag} FAILED: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the unit tests.** Run `$PY -m pytest tests/tools/test_abaqus_odb_export.py -q`. Expected: pure-helper tests pass, and the real-ODB test is skipped. Then run `$PY -m ruff format data_generation/abaqus`, and re-run the 3.10 parse test afterwards, since ruff targets 3.12 and must not introduce newer syntax.

- [ ] **Step 5: Export the conformance ODBs and check the file**

```bash
cd /c/structbench-runs/<dataset> && abaqus python /c/Users/kylin/Desktop/StructBench/data_generation/abaqus/odb_export.py --sweep /c/structbench-runs/<dataset>
cd /c/Users/kylin/Desktop/StructBench && STRUCTBENCH_ABAQUS_RUN_DIR=/c/structbench-runs/<dataset>/<PROBE_CASE> $PY -m pytest tests/tools/test_abaqus_odb_export.py -q
```
If an ODB API call fails (attribute names on blocks, `position`, a missing `instance`, the shape of `history` data), fix the exporter from the real ODB's behaviour, keep the pure-helper tests green, and write the fix into the Task 7 notes. Then load one `.npz` in `$PY` and add these facts to the notes:
  1. **Clock:** do the `frame_times` sit on exact multiples of 1 µs, and how many are there (401 expected)?
  2. **History clock:** do the history outputs share the frame clock?
  3. **Stored components:** the S component labels (S11, S22, S33, S12 expected) and the stored dtype (field precision under `double=both`).
  4. **Initial state at frame 0:** V2 equals −v₀ on every node. PEEQ at frame 0 equals the per-element values in the deck's `TYPE=HARDENING` block for the case that has one, and is zero for the case that doesn't.
  5. **Energy terms:** which terms the history holds, and whether ETOTAL stays within a small fraction of the initial kinetic energy. This is a measurement for the notes, not a verdict.
  6. **Wall penetration:** the minimum z of the impact face over all frames. Clearly negative means the wall's normal is wrong: fix the point order in the private `model.py`, regenerate with `--force` after moving the case aside, and rerun that case with the maintainer's yes.
  7. **Export time and `.npz` size** per case.

- [ ] **Step 6: Gates, then commit**

```bash
git add data_generation/abaqus/odb_export.py tests/tools/test_abaqus_odb_export.py
git commit -m "feat(datagen): ODB exporter to the abaqus-npz/1 intermediate" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Record what the conformance run established

**Files:**
- Modify: `data_generation/abaqus/STANDARD_INPUT_BLOCK.md`, `data_generation/README.md`
- Create (gitignored): `scratch/2026-09-24-abaqus-plan-2-handoff.md`

**Interfaces:**
- Consumes: the notes from Tasks 7–8.
- Produces: an Abaqus requirements doc that states Explicit facts with the files they came from, and a handoff note that Plan 2 is written from.

- [ ] **Step 1: Amend `STANDARD_INPUT_BLOCK.md`.** Under "What is established", add a subsection `### Abaqus/Explicit (conformance run, <YYYY-MM-DD of the run>)`, with the run's actual date filled in. Each statement names its file, following the existing rows' style (for example, "`.sta` of the probe case: …"). Cover:
  - the completion wording;
  - the version line;
  - the `.msg` summary;
  - the stable-increment column;
  - the frame clock under `TIME INTERVAL` and `TIME MARKS=YES`;
  - whether history shares it;
  - the energy terms written;
  - the stress components of `CAX4R`;
  - how `TYPE=HARDENING` appears at frame 0;
  - the field-data precision.

  Move each open point that the run answered (1, 3, 5, 7, and whichever others) out of "Open points", with a pointer to the new rows. Leave unanswered ones open. Do **not** name the study, its values or its splits; the run is described as "an axisymmetric CAX4R impact case with an analytical rigid wall".

- [ ] **Step 2: Add the shared pattern to `data_generation/README.md`.** Append to the `## Abaqus` section:

```markdown
- **Shared scripts, dataset-blind** (ADR-0069): `sampling.py` (Sobol splits),
  `deck.py` (keyword writers), `generate.py` (sweep -> case folders +
  provenance), `run_jobs.py` (runs decks, `run.json` + `run_log.csv`),
  `odb_export.py` (runs under `abaqus python`; `.odb` -> `abaqus-npz/1`). A
  dataset is a `sweep.toml` plus a `model.py`, passed with `--dataset <dir>`;
  it may live in a private repository until admission, then moves to
  `abaqus/<name>/`. State lives in each case folder, so every stage can be
  re-run. Runs execute in a local work root outside OneDrive.
```

- [ ] **Step 3: Write the Plan 2 handoff** `scratch/2026-09-24-abaqus-plan-2-handoff.md` (gitignored). It holds the observed Explicit wording (licence line scrubbed) that the reader must match, the exporter's key names as actually written, the measured runtimes and `.npz` sizes, and any exporter fixes from Task 8.

- [ ] **Step 4: Gates, then commit**

```bash
git add data_generation/abaqus/STANDARD_INPUT_BLOCK.md data_generation/README.md
git commit -m "docs(abaqus): record what the first Explicit conformance run established" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Stop at the checkpoint.** Summarise for the maintainer what the run established, and propose writing Plan 2. Do not merge; `main` moves only on the maintainer's word.

---

## Plan 2 (written after Task 9, from the handoff note)

This is an outline, not tasks. The spec's remaining sections, in order:

1. The readers learn Explicit wording, `CAX*` and `CPE*` element codes, `*DYNAMIC, EXPLICIT`, `*PLASTIC`, `*INITIAL CONDITIONS` and output requests, plus the stable-increment series (fixtures from the conformance run, licence line scrubbed).
2. `core/io/abaqus.py` (`abaqus_export_to_case`, with fail-closed version and clock checks) and `convert.py`.
3. ADR on the `elastic_plastic_isotropic` material class and its build, with the yield table read per case from the input.
4. The rows `sampling_clock_consistent`, `initial_state_matches_input` and `plastic_dissipation_late_growth` (ADR-0066 dated note), and an energy ledger read from the `.npz`.
5. `datacheck measure --declaration`, and `validate.py`.
6. `archive.py` (move to `raw/<name>/abaqus/`, `canonical/<name>/`; prune ODBs by the `[retention]` declaration, dry run first, explicit confirmation).
7. Pilot review, then the production sweep, which needs the maintainer's go-ahead with measured cost.
