# Datagen platform, part one — implementation plan

> **Executed** — merged 2026-09-27 (`602ab0d`). A historical record, not instructions: do not re-run it. See `docs/plans/README.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the Abaqus data-generation pipeline into the package as `structbench.datagen` with one console entry point, give datasets a stated, checked contract (`dataset.toml` + `problem.py`) with a scaffold and a public example, and prove it by rebuilding dataset A's 520 decks byte for byte.

**Architecture:** Nine tasks. The first is a mechanical move of the existing scripts and their tests into `src/structbench/datagen/` (solver-agnostic stages) and `src/structbench/datagen/abaqus/` (deck writers and the exporter, which stays a standalone Python 3.10 file). Then the contract: `definition.py` parses and validates `dataset.toml` (today's `sweep.toml` plus `[levels]`, `[pilot]`, `[qoi]`, `[limits]`) and loads `problem.py` (`input_deck`, `feasible`, `mesh`, `qoi`); the stages switch to it; `template.py` scaffolds a definition from the public example and checks any definition without a solver; `cli.py` exposes every stage as `structbench-datagen <stage>`. Docs record the layering change. The last task migrates the private dataset A and regresses its decks against the sha256 in their provenance.

**Tech Stack:** Python ≥ 3.12, numpy, scipy (the `datagen` extra, Sobol), tomllib, h5py through `structbench.core`; setuptools console scripts and package data; pytest; ruff (88 columns); mypy with `disallow_untyped_defs`.

**Spec:** `docs/plans/2026-09-27-abaqus-datagen-platform-design.md` (ADR-0071, `decisions/0071-datagen-platform.md`). Part one covers the design's "Package layout", "The dataset definition: the template" (`new`, `check`, the tables, the four hooks), the writer arguments named under "Migration of dataset A", and the migration itself. Preflight, converge, run hardening, follow, card, the verification additions and the full user guide are parts two and three.

## Global Constraints

- Python 3.12 syntax everywhere **except** `src/structbench/datagen/abaqus/odb_export.py`, which must parse with `ast.parse(source, feature_version=(3, 10))` and import nothing from `structbench` (it runs under Abaqus's interpreter).
- `ruff check .` and `ruff format --check .` clean at 88 columns; `mypy src` clean with `disallow_untyped_defs = true` (moved modules are now in scope).
- scipy is used only through the `datagen` extra; tests that need it call `pytest.importorskip("scipy")`.
- No private dataset's box, split names, constants, case ids or numbers enter the public repository. The example carries one rod, one material, three explicit points.
- Files written by tools (`new`, provenance, decks) are written as bytes with `\n` endings; on Windows `Path.write_text` writes CRLF and would change the hashes the pipeline records.
- Deck text is byte-stable: identical parameters produce identical text; every keyword is written by `structbench.datagen.abaqus.deck`.
- Dataset A is never re-run; its migration is proven by rebuilding decks against `inp_sha256` in each case's `provenance.json`.
- Work on branch `feat/datagen-platform-1` of `<repo>`; `main` moves only on the maintainer's instruction; never push.
- Test command for the whole suite (partial collection breaks on duplicate test basenames): `<venv>/python -m pytest -q -p no:cacheprovider`. Gates before every commit: `ruff check .`, `ruff format --check .`, `mypy src`, the full suite, chained with `&&`.
- The private dataset repository (`<private repo>`) holds dataset A's definition and its runs (gitignored there). Only Task 9 touches it, and nothing of it is quoted here.

## Review Focus

1. A `dataset.toml` with a missing required table or field: `check` must name `table.field` in plain words, not raise a `KeyError` (Task 3, `test_missing_table_is_named`).
2. `[pilot].split` naming a sampled split, or a split without `probe = true`: refused at load, since the gate of part two would otherwise guard the pilots themselves (Task 3, `test_pilot_split_must_be_an_explicit_probe`).
3. A `problem.py` whose `input_deck` is not a pure function (it reads the clock or a set's iteration order): `check` must report it as unstable rather than let provenance hashes drift (Task 6, `test_check_reports_an_unstable_deck`).
4. A `mesh()` whose levels do not nest (a rounding rule like A's first one): `check` names the level pair (Task 6, `test_check_names_the_level_that_does_not_nest`).
5. `structbench-datagen export` on a machine without `abaqus` on PATH: exit 2 with one line, not a traceback (Task 7, `test_export_without_abaqus_exits_two`).
6. `new` on Windows: the scaffolded files must have `\n` endings so that the hashes `generate` records are the same on every platform (Task 6, `test_scaffold_writes_lf_endings`).

---

### Task 1: Move the pipeline into the package

**Files:**
- Create: `src/structbench/datagen/__init__.py`, `src/structbench/datagen/abaqus/__init__.py`, `tests/datagen/__init__.py` (empty; keeps test basenames unique under one package)
- Move (git mv): `data_generation/abaqus/sampling.py` → `src/structbench/datagen/sampling.py`; `generate.py` → `src/structbench/datagen/generate.py`; `run_jobs.py` → `src/structbench/datagen/run.py`; `convert.py` → `src/structbench/datagen/convert.py`; `validate.py` → `src/structbench/datagen/validate.py`; `collect_run_evidence.py` → `src/structbench/datagen/collect.py`; `archive.py` → `src/structbench/datagen/archive.py`; `deck.py` → `src/structbench/datagen/abaqus/deck.py`; `odb_export.py` → `src/structbench/datagen/abaqus/odb_export.py`
- Move (git mv): `tests/tools/test_abaqus_sampling.py` → `tests/datagen/test_sampling.py`; `test_abaqus_generate.py` → `test_generate.py`; `test_abaqus_run_jobs.py` → `test_run.py`; `test_abaqus_convert.py` → `test_convert.py`; `test_abaqus_validate.py` → `test_validate.py`; `test_abaqus_collect.py` → `test_collect.py`; `test_abaqus_archive.py` → `test_archive.py`; `test_abaqus_deck.py` → `test_deck.py`; `test_abaqus_odb_export.py` → `test_odb_export.py`
- Delete: `tests/tools/abaqus_paths.py`
- Modify: `pyproject.toml` (mypy override for `odbAccess`), the moved modules' imports

**Interfaces:**
- Consumes: nothing.
- Produces: the module paths every later task imports: `structbench.datagen.{sampling, generate, run, convert, validate, collect, archive}`, `structbench.datagen.abaqus.{deck, odb_export}`. Public functions keep their names (`generate.plan_cases`, `generate.write_case`, `run.run_sweep`, `run.case_state`, `convert.convert_sweep`, `validate.validate_sweep`, `collect.collect_sweep`, `archive.retained`, `archive.redact`, `deck.*`).

- [ ] **Step 1: Create the packages and move the modules**

```bash
cd <repo>
git checkout -b feat/datagen-platform-1 main
mkdir -p src/structbench/datagen/abaqus tests/datagen
git mv data_generation/abaqus/sampling.py src/structbench/datagen/sampling.py
git mv data_generation/abaqus/generate.py src/structbench/datagen/generate.py
git mv data_generation/abaqus/run_jobs.py src/structbench/datagen/run.py
git mv data_generation/abaqus/convert.py src/structbench/datagen/convert.py
git mv data_generation/abaqus/validate.py src/structbench/datagen/validate.py
git mv data_generation/abaqus/collect_run_evidence.py src/structbench/datagen/collect.py
git mv data_generation/abaqus/archive.py src/structbench/datagen/archive.py
git mv data_generation/abaqus/deck.py src/structbench/datagen/abaqus/deck.py
git mv data_generation/abaqus/odb_export.py src/structbench/datagen/abaqus/odb_export.py
```

Write `src/structbench/datagen/__init__.py`:

```python
"""Data generation: from a dataset definition to verified canonical cases.

Solver-agnostic stages live here (sampling, generate, run, convert, validate,
archive, the definition contract and its scaffold); solver-specific writers and
exporters live in a subpackage per solver (``abaqus``). Nothing here is
imported by the rest of the package: ``datagen`` sits beside ``eval`` and
``benchmarks`` in the layering (ADR-0071).
"""
```

Write `src/structbench/datagen/abaqus/__init__.py`:

```python
"""Abaqus: the deck writers (``deck``) and the ODB exporter (``odb_export``).

``odb_export`` runs under Abaqus's own Python 3.10 and must not be imported by
anything here; the CLI hands its file path to ``abaqus python``.
"""
```

Write `tests/datagen/__init__.py` as an empty file.

- [ ] **Step 2: Rewrite the in-package imports**

In `src/structbench/datagen/generate.py` replace `import sampling` with `from structbench.datagen import sampling`. In `src/structbench/datagen/validate.py` replace `from collect_run_evidence import collect_sweep` with `from structbench.datagen.collect import collect_sweep`. Replace the module docstrings' first lines so that `argparse` descriptions stay meaningful (the CLI in Task 7 rewrites the usage lines; here only the moved imports change).

- [ ] **Step 3: Move and rewrite the tests**

```bash
git mv tests/tools/test_abaqus_sampling.py tests/datagen/test_sampling.py
git mv tests/tools/test_abaqus_generate.py tests/datagen/test_generate.py
git mv tests/tools/test_abaqus_run_jobs.py tests/datagen/test_run.py
git mv tests/tools/test_abaqus_convert.py tests/datagen/test_convert.py
git mv tests/tools/test_abaqus_validate.py tests/datagen/test_validate.py
git mv tests/tools/test_abaqus_collect.py tests/datagen/test_collect.py
git mv tests/tools/test_abaqus_archive.py tests/datagen/test_archive.py
git mv tests/tools/test_abaqus_deck.py tests/datagen/test_deck.py
git mv tests/tools/test_abaqus_odb_export.py tests/datagen/test_odb_export.py
git rm tests/tools/abaqus_paths.py
```

In every moved test: delete `import abaqus_paths  # noqa: F401`; replace `import sampling` with `from structbench.datagen import sampling`, `import generate` with `from structbench.datagen import generate`, `import run_jobs` with `from structbench.datagen import run as run_jobs` (keeps the test bodies unchanged), `import convert` / `import validate` / `import archive` likewise from `structbench.datagen`, `import collect_run_evidence` with `from structbench.datagen import collect as collect_run_evidence`, `import deck` with `from structbench.datagen.abaqus import deck`, `import odb_export` with `from structbench.datagen.abaqus import odb_export`. In `tests/datagen/test_odb_export.py` replace `abaqus_paths.ABAQUS_DIR / "odb_export.py"` with `Path(odb_export.__file__)`.

- [ ] **Step 4: Run the full suite and read the failures**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen 2>&1 | tail -20`
Expected: every moved test passes (the same counts as before the move: 9, 10, 11, 6, 2, 3, 7, 4, 6). A failure here is an import the rewrite missed; fix the import, not the test.

- [ ] **Step 5: Bring the moved modules under mypy**

Add to `pyproject.toml`, in the existing override list:

```toml
[[tool.mypy.overrides]]
module = ["h5py.*", "lasso.*", "torch_geometric.*", "matplotlib.*", "tensorflow.*", "odbAccess.*", "scipy.*"]
ignore_missing_imports = true
```

Run: `<venv>/mypy src`
Expected: 0 errors. If the moved modules report errors, fix them with annotations only (no behaviour change); a `# type: ignore[<code>]` needs a comment saying why. Do not loosen the mypy configuration.

- [ ] **Step 6: Lint, format, full suite**

Run: `<venv>/ruff check . && <venv>/ruff format --check . && <venv>/mypy src && <venv>/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all clean; the suite's pass count equals the count on `main` (1145 passed, 8 skipped at the time of writing).

- [ ] **Step 7: Commit**

```bash
git add -A src/structbench/datagen tests/datagen tests/tools data_generation pyproject.toml
git commit -m "refactor(datagen): move the Abaqus pipeline into structbench.datagen (ADR-0071)"
```

---

### Task 2: Writer arguments for the two options datasets set by hand

**Files:**
- Modify: `src/structbench/datagen/abaqus/deck.py` (`explicit_step`, new `contact_damping`)
- Test: `tests/datagen/test_deck.py`

**Interfaces:**
- Produces: `deck.explicit_step(name: str, period: float, body: str, *, scale_factor: float | None = None) -> str`; `deck.contact_damping(fraction: float) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/datagen/test_deck.py`:

```python
def test_explicit_step_takes_a_stable_increment_scale_factor():
    plain = deck.explicit_step("S", 4.0e-4, "")
    scaled = deck.explicit_step("S", 4.0e-4, "", scale_factor=0.5)
    assert plain.splitlines()[1] == "*DYNAMIC, EXPLICIT"
    assert scaled.splitlines()[1] == "*DYNAMIC, EXPLICIT, SCALE FACTOR=0.5"
    assert plain.splitlines()[2] == scaled.splitlines()[2] == ", 0.0004"


def test_contact_damping_is_a_critical_damping_fraction():
    assert deck.contact_damping(0.1) == (
        "*CONTACT DAMPING, DEFINITION=CRITICAL DAMPING FRACTION\n0.1\n"
    )
```

- [ ] **Step 2: Run them to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_deck.py -k "scale_factor or contact_damping"`
Expected: 2 failed — `TypeError: explicit_step() got an unexpected keyword argument 'scale_factor'` and `AttributeError: module ... has no attribute 'contact_damping'`.

- [ ] **Step 3: Implement**

In `deck.py` replace `explicit_step` and add `contact_damping` after `surface_interaction`:

```python
def contact_damping(fraction: float) -> str:
    """Contact damping as a fraction of critical, for the interaction that follows."""
    return (
        "*CONTACT DAMPING, DEFINITION=CRITICAL DAMPING FRACTION\n"
        f"{num(fraction)}\n"
    )
```

```python
def explicit_step(
    name: str, period: float, body: str, *, scale_factor: float | None = None
) -> str:
    """An explicit dynamic step of ``period``; ``scale_factor`` scales the stable
    increment (0.5 halves it: with kinematic contact against a rigid surface the
    full increment created energy in some runs, see the conformance document)."""
    option = f", SCALE FACTOR={num(scale_factor)}" if scale_factor is not None else ""
    return (
        f"*STEP, NAME={name}, NLGEOM=YES\n*DYNAMIC, EXPLICIT{option}\n, {num(period)}\n"
        f"{body}*END STEP\n"
    )
```

- [ ] **Step 4: Run the deck tests**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_deck.py`
Expected: 6 passed (the snapshot test is unchanged because the default writes the same text).

- [ ] **Step 5: Commit**

```bash
git add src/structbench/datagen/abaqus/deck.py tests/datagen/test_deck.py
git commit -m "feat(datagen): explicit_step scale_factor and a contact_damping writer"
```

---

### Task 3: The definition contract — `dataset.toml` and `problem.py`

**Files:**
- Create: `src/structbench/datagen/definition.py`
- Modify: `src/structbench/datagen/sampling.py` (`Split.probe`)
- Test: `tests/datagen/conftest.py` (new), `tests/datagen/test_definition.py` (new), `tests/datagen/test_sampling.py`

**Interfaces:**
- Consumes: `sampling.parse_bounds`, `sampling.parse_splits`, `sampling.Split`, `structbench.core.io.unit_factors`.
- Produces:
  - `class DefinitionError(ValueError)` — message names `table.field`.
  - `@dataclass(frozen=True) class Levels: refine_key: str; production: str; pilot: tuple[str, ...]`
  - `@dataclass(frozen=True) class Pilot: split: str; fine_cases: tuple[str, ...]; min_free_gb: float; accepted_gaps: tuple[str, ...]`
  - `@dataclass(frozen=True) class QoiSpec: names: tuple[str, ...]; units: tuple[str, ...]`
  - `@dataclass(frozen=True) class Definition: path: Path; raw: dict[str, Any]; name: str; case_prefix: str; units: str; solver: str; fixed: dict[str, Any]; variables: dict[str, tuple[float, float]]; regions: dict[str, dict[str, tuple[float, float]]]; splits: tuple[sampling.Split, ...]; limits: dict[str, Any]; levels: Levels; pilot: Pilot; qoi: QoiSpec; retention: dict[str, Any]` with `def split(self, name: str) -> sampling.Split` and `def sha256(self) -> str` (of the file bytes).
  - `def load_definition(dataset_dir: Path) -> Definition`
  - `def load_problem(dataset_dir: Path) -> ModuleType` — requires callables `input_deck`, `mesh`, `qoi`; `feasible` optional; raises `DefinitionError` naming the missing hook.
  - `def problem_sha256(dataset_dir: Path) -> str`
  - `sampling.Split.probe: bool = False`, read from the split table's `probe` key.
  - `tests/datagen/conftest.py`: `write_definition(dataset_dir: Path, *, toml: str | None = None, problem: str | None = None) -> Path` writing a minimal valid pair (`MINIMAL_TOML`, `MINIMAL_PROBLEM` module constants).

- [ ] **Step 1: Write the shared fixture**

`tests/datagen/conftest.py`:

```python
"""Shared fixtures: a minimal valid dataset definition (ADR-0071)."""

from pathlib import Path

import pytest

pytest.importorskip("scipy")

MINIMAL_TOML = """\
[dataset]
name = "toy"
case_prefix = "TOY"
units = "t-mm-s"
solver = "abaqus"

[declaration]
unit_system = "t-mm-s"
discretisation = "FEM"
erosion = false
fields = ["node/displacement", "solid/stress", "global/kinetic_energy"]

[fixed]
E = 1000.0

[variables]
L = [1.0, 2.0]
v0 = [10.0, 20.0]

[splits.train]
n = 4
seed = 1

[splits.pilot]
points = [{ L = 1.0, v0 = 10.0 }, { L = 2.0, v0 = 20.0 }]
probe = true

[levels]
refine_key = "refine"
production = "1"
pilot = ["1", "2"]

[pilot]
split = "pilot"
fine_cases = ["TOY-pilot-0000"]
min_free_gb = 5.0
accepted_gaps = ["solver_identity_complete"]

[qoi]
names = ["length"]
units = ["m"]
"""

MINIMAL_PROBLEM = '''\
"""A toy problem: one quad per level, byte-stable."""

from structbench.datagen.abaqus import deck


def input_deck(params, variant):
    k = int(params.get("refine", 1))
    mesh = deck.structured_quad_mesh(k, k, 0.0, float(params["L"]), 0.0, 1.0)
    return deck.heading("toy") + deck.node_block(mesh.node_labels, mesh.coords)


def mesh(params):
    k = int(params.get("refine", 1))
    return deck.structured_quad_mesh(k, k, 0.0, float(params["L"]), 0.0, 1.0)


def qoi(case):
    x = case.nodes.coords
    return {"length": float(x[:, 0].max() - x[:, 0].min())}
'''


def write_definition(
    dataset_dir: Path, *, toml: str | None = None, problem: str | None = None
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset.toml").write_bytes((toml or MINIMAL_TOML).encode())
    (dataset_dir / "problem.py").write_bytes((problem or MINIMAL_PROBLEM).encode())
    return dataset_dir


@pytest.fixture
def definition_dir(tmp_path: Path) -> Path:
    return write_definition(tmp_path / "toy")
```

- [ ] **Step 2: Write the failing tests**

`tests/datagen/test_definition.py`:

```python
"""The dataset definition contract (ADR-0071): dataset.toml and problem.py."""

import pytest
from conftest import MINIMAL_PROBLEM, MINIMAL_TOML, write_definition

from structbench.datagen import definition


def test_a_minimal_definition_loads_every_table(definition_dir):
    d = definition.load_definition(definition_dir)
    assert (d.name, d.case_prefix, d.units, d.solver) == ("toy", "TOY", "t-mm-s", "abaqus")
    assert d.variables == {"L": (1.0, 2.0), "v0": (10.0, 20.0)}
    assert [s.name for s in d.splits] == ["train", "pilot"]
    assert d.split("pilot").probe is True and d.split("train").probe is False
    assert d.levels.production == "1" and d.levels.pilot == ("1", "2")
    assert d.pilot.split == "pilot" and d.pilot.accepted_gaps == ("solver_identity_complete",)
    assert d.qoi.names == ("length",) and d.qoi.units == ("m",)
    assert d.limits == {} and d.retention == {}
    assert len(d.sha256()) == 64


@pytest.mark.parametrize(
    "drop, expected",
    [
        ("[levels]", "levels"),
        ("[pilot]", "pilot"),
        ("[qoi]", "qoi"),
        ("refine_key = \"refine\"", "levels.refine_key"),
        ("min_free_gb = 5.0", "pilot.min_free_gb"),
    ],
)
def test_missing_table_is_named(tmp_path, drop, expected):
    toml = MINIMAL_TOML.replace(drop, "")
    with pytest.raises(definition.DefinitionError, match=expected):
        definition.load_definition(write_definition(tmp_path / "d", toml=toml))


def test_pilot_split_must_be_an_explicit_probe(tmp_path):
    sampled = MINIMAL_TOML.replace('split = "pilot"', 'split = "train"')
    with pytest.raises(definition.DefinitionError, match="pilot.split"):
        definition.load_definition(write_definition(tmp_path / "a", toml=sampled))
    unmarked = MINIMAL_TOML.replace("probe = true\n", "")
    with pytest.raises(definition.DefinitionError, match="probe"):
        definition.load_definition(write_definition(tmp_path / "b", toml=unmarked))


def test_production_level_must_be_a_pilot_level(tmp_path):
    toml = MINIMAL_TOML.replace('production = "1"', 'production = "4"')
    with pytest.raises(definition.DefinitionError, match="levels.production"):
        definition.load_definition(write_definition(tmp_path / "d", toml=toml))


def test_refine_key_may_not_be_a_variable_or_a_constant(tmp_path):
    toml = MINIMAL_TOML.replace("E = 1000.0", "E = 1000.0\nrefine = 2")
    with pytest.raises(definition.DefinitionError, match="refine"):
        definition.load_definition(write_definition(tmp_path / "d", toml=toml))


def test_qoi_names_and_units_must_pair(tmp_path):
    toml = MINIMAL_TOML.replace('units = ["m"]', 'units = ["m", "m"]')
    with pytest.raises(definition.DefinitionError, match="qoi.units"):
        definition.load_definition(write_definition(tmp_path / "d", toml=toml))


def test_unknown_unit_label_and_solver_are_refused(tmp_path):
    with pytest.raises(definition.DefinitionError, match="dataset.units"):
        definition.load_definition(
            write_definition(tmp_path / "u", toml=MINIMAL_TOML.replace('units = "t-mm-s"', 'units = "furlongs"'))
        )
    with pytest.raises(definition.DefinitionError, match="dataset.solver"):
        definition.load_definition(
            write_definition(tmp_path / "s", toml=MINIMAL_TOML.replace('solver = "abaqus"', 'solver = "ansys"'))
        )


def test_problem_must_define_the_three_required_hooks(definition_dir):
    problem = definition.load_problem(definition_dir)
    assert callable(problem.input_deck) and callable(problem.mesh) and callable(problem.qoi)
    (definition_dir / "problem.py").write_bytes(MINIMAL_PROBLEM.replace("def qoi", "def qoi_").encode())
    with pytest.raises(definition.DefinitionError, match="qoi"):
        definition.load_problem(definition_dir)


def test_problem_hash_is_of_the_file_bytes(definition_dir):
    import hashlib

    assert definition.problem_sha256(definition_dir) == hashlib.sha256(
        (definition_dir / "problem.py").read_bytes()
    ).hexdigest()
```

Append to `tests/datagen/test_sampling.py`:

```python
def test_a_split_may_be_marked_as_a_probe():
    from structbench.datagen import sampling

    splits = sampling.parse_splits(
        {"splits": {"p": {"points": [{"a": 1.0}], "probe": True}, "s": {"n": 2, "seed": 3}}}
    )
    assert [s.probe for s in splits] == [True, False]
```

- [ ] **Step 3: Run them to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_definition.py tests/datagen/test_sampling.py 2>&1 | tail -5`
Expected: `test_definition.py` errors on import (`cannot import name 'definition'`); the sampling test fails with `AttributeError: 'Split' object has no attribute 'probe'`.

- [ ] **Step 4: Implement `Split.probe`**

In `sampling.py` add the field `probe: bool = False` to `Split` (after `points`) and, in `parse_splits`, pass `probe=bool(raw.get("probe", False))`.

- [ ] **Step 5: Implement `definition.py`**

```python
"""The dataset definition: ``dataset.toml`` and ``problem.py`` (ADR-0071).

``load_definition`` parses and validates the tables a dataset must supply;
``load_problem`` imports the dataset's code and checks it defines the hooks the
stages call. Both raise ``DefinitionError`` naming the table and field, so a
user who has just filled the template learns what is wrong before any solver
runs. The contract is the design's table ("The dataset definition").
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from structbench.core.io import unit_factors
from structbench.datagen import sampling

DEFINITION_FILE = "dataset.toml"
PROBLEM_FILE = "problem.py"
SOLVERS = ("abaqus",)
REQUIRED_HOOKS = ("input_deck", "mesh", "qoi")
OPTIONAL_HOOKS = ("feasible",)


class DefinitionError(ValueError):
    """A dataset definition that does not meet the contract; says which field."""


@dataclass(frozen=True)
class Levels:
    refine_key: str
    production: str
    pilot: tuple[str, ...]


@dataclass(frozen=True)
class Pilot:
    split: str
    fine_cases: tuple[str, ...]
    min_free_gb: float
    accepted_gaps: tuple[str, ...]


@dataclass(frozen=True)
class QoiSpec:
    names: tuple[str, ...]
    units: tuple[str, ...]


@dataclass(frozen=True)
class Definition:
    path: Path
    raw: dict[str, Any]
    name: str
    case_prefix: str
    units: str
    solver: str
    fixed: dict[str, Any]
    variables: dict[str, tuple[float, float]]
    regions: dict[str, dict[str, tuple[float, float]]]
    splits: tuple[sampling.Split, ...]
    limits: dict[str, Any]
    levels: Levels
    pilot: Pilot
    qoi: QoiSpec
    retention: dict[str, Any]

    def split(self, name: str) -> sampling.Split:
        for s in self.splits:
            if s.name == name:
                return s
        raise KeyError(name)

    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    table = raw.get(name)
    if not isinstance(table, dict):
        raise DefinitionError(f"{name}: table missing from {DEFINITION_FILE}")
    return table


def _field(table: dict[str, Any], where: str, key: str, kind: type) -> Any:
    if key not in table:
        raise DefinitionError(f"{where}.{key}: missing")
    value = table[key]
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind):
        raise DefinitionError(f"{where}.{key}: expected {kind.__name__}")
    return value


def _strings(table: dict[str, Any], where: str, key: str) -> tuple[str, ...]:
    value = _field(table, where, key, list)
    if not all(isinstance(v, str) for v in value):
        raise DefinitionError(f"{where}.{key}: expected a list of strings")
    return tuple(value)


def load_definition(dataset_dir: Path) -> Definition:
    """Parse and validate ``<dataset_dir>/dataset.toml``."""
    path = dataset_dir / DEFINITION_FILE
    if not path.is_file():
        raise DefinitionError(f"{DEFINITION_FILE}: not found in {dataset_dir}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    dataset = _table(raw, "dataset")
    name = _field(dataset, "dataset", "name", str)
    prefix = _field(dataset, "dataset", "case_prefix", str)
    units = _field(dataset, "dataset", "units", str)
    try:
        unit_factors(units)
    except ValueError as exc:
        raise DefinitionError(f"dataset.units: {exc}") from exc
    solver = _field(dataset, "dataset", "solver", str)
    if solver not in SOLVERS:
        raise DefinitionError(f"dataset.solver: {solver!r} is not one of {SOLVERS}")
    _table(raw, "declaration")  # its contents are validated by declared_from_toml

    fixed = dict(_table(raw, "fixed")) if "fixed" in raw else {}
    limits = dict(raw.get("limits", {}))
    variables = sampling.parse_bounds(_table(raw, "variables"), "variables")
    regions = {
        r: sampling.parse_bounds(box, f"regions.{r}")
        for r, box in raw.get("regions", {}).items()
    }
    if "splits" not in raw:
        raise DefinitionError("splits: at least one [splits.<name>] table is required")
    try:
        splits = tuple(sampling.parse_splits(raw))
    except (ValueError, KeyError) as exc:
        raise DefinitionError(f"splits: {exc}") from exc

    lv = _table(raw, "levels")
    levels = Levels(
        _field(lv, "levels", "refine_key", str),
        _field(lv, "levels", "production", str),
        _strings(lv, "levels", "pilot"),
    )
    if levels.production not in levels.pilot:
        raise DefinitionError(
            f"levels.production: {levels.production!r} is not among levels.pilot"
        )
    for where, table in (("fixed", fixed), ("variables", variables), ("limits", limits)):
        if levels.refine_key in table:
            raise DefinitionError(
                f"{where}.{levels.refine_key}: the refine key is reserved for levels"
            )

    pt = _table(raw, "pilot")
    pilot = Pilot(
        _field(pt, "pilot", "split", str),
        _strings(pt, "pilot", "fine_cases"),
        _field(pt, "pilot", "min_free_gb", float),
        _strings(pt, "pilot", "accepted_gaps"),
    )
    names = {s.name: s for s in splits}
    if pilot.split not in names:
        raise DefinitionError(f"pilot.split: no [splits.{pilot.split}] table")
    if not names[pilot.split].points:
        raise DefinitionError(f"pilot.split: [splits.{pilot.split}] must list points")
    if not names[pilot.split].probe:
        raise DefinitionError(f"splits.{pilot.split}: the pilot split needs probe = true")

    q = _table(raw, "qoi")
    qoi = QoiSpec(_strings(q, "qoi", "names"), _strings(q, "qoi", "units"))
    if not qoi.names:
        raise DefinitionError("qoi.names: at least one quantity of interest")
    if len(qoi.units) != len(qoi.names):
        raise DefinitionError("qoi.units: one unit per name")

    retention = dict(raw.get("retention", {}))
    unknown = set(retention) - {"odb_fraction", "odb_seed", "odb_cases"}
    if unknown:
        raise DefinitionError(f"retention: unknown keys {sorted(unknown)}")

    return Definition(
        path, raw, name, prefix, units, solver, fixed, variables, regions, splits,
        limits, levels, pilot, qoi, retention,
    )


def load_problem(dataset_dir: Path) -> ModuleType:
    """Import ``<dataset_dir>/problem.py`` by path; no bytecode is written.

    A ``__pycache__`` inside the dataset's repository would make its provenance
    read as uncommitted work.
    """
    path = dataset_dir / PROBLEM_FILE
    if not path.is_file():
        raise DefinitionError(f"{PROBLEM_FILE}: not found in {dataset_dir}")
    spec = importlib.util.spec_from_file_location(f"dataset_problem_{dataset_dir.name}", path)
    if spec is None or spec.loader is None:
        raise DefinitionError(f"{PROBLEM_FILE}: cannot be loaded from {dataset_dir}")
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    for hook in REQUIRED_HOOKS:
        if not callable(getattr(module, hook, None)):
            raise DefinitionError(f"{PROBLEM_FILE}: defines no {hook}()")
    return module


def problem_sha256(dataset_dir: Path) -> str:
    return hashlib.sha256((dataset_dir / PROBLEM_FILE).read_bytes()).hexdigest()
```

- [ ] **Step 6: Run the tests**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_definition.py tests/datagen/test_sampling.py`
Expected: all pass (10 in test_definition, 10 in test_sampling).

- [ ] **Step 7: Gates and commit**

Run the four gates. Expected: clean; suite count = previous + 11.

```bash
git add src/structbench/datagen/definition.py src/structbench/datagen/sampling.py tests/datagen/conftest.py tests/datagen/test_definition.py tests/datagen/test_sampling.py
git commit -m "feat(datagen): the dataset definition contract -- dataset.toml tables and problem.py hooks"
```

---

### Task 4: The stages read the contract

**Files:**
- Modify: `src/structbench/datagen/generate.py` (`load_sweep`, `load_model`, `plan_cases`, `main`), `src/structbench/datagen/archive.py` (retention source), `src/structbench/datagen/validate.py` (declaration source), `src/structbench/cli/datacheck.py` (`declared_from_toml` docstring and `--declaration` help)
- Test: `tests/datagen/test_generate.py`, `tests/datagen/test_archive.py`, `tests/datagen/test_validate.py`

**Interfaces:**
- Consumes: `definition.load_definition`, `definition.load_problem`, `definition.problem_sha256`, `Definition.limits`.
- Produces: `generate.plan_cases(defn: Definition, feasible: Callable | None) -> list[CaseSpec]` (same `CaseSpec`); `generate.PROVENANCE_FORMAT = "abaqus-provenance/2"` with keys `definition_sha256` and `problem_sha256` replacing `sweep_sha256`; `generate.load_sweep` and `generate.load_model` removed.

- [ ] **Step 1: Update the generate tests to the contract**

In `tests/datagen/test_generate.py`, the fixture that writes `sweep.toml` and `model.py` becomes the shared one: replace the local `TOY_SWEEP`/model writing with `write_definition(ds, toml=..., problem=...)` from `conftest`, renaming `build` to `input_deck` in the toy problem text and adding the `[levels]`, `[pilot]`, `[qoi]` tables to the toy TOML (copy them from `MINIMAL_TOML`; the pilot split's points must cover the toy's variables). Change every `generate.load_sweep(ds)` to `definition.load_definition(ds)` and `generate.load_model(ds)` to `definition.load_problem(ds)`, and `model.build(` to `problem.input_deck(`. Add:

```python
def test_provenance_records_both_definition_hashes(tmp_path):
    ds = write_definition(tmp_path / "toy")
    rc = generate.main(["--dataset", str(ds), "--work-root", str(tmp_path / "runs")])
    assert rc == 0
    prov = json.loads(next((tmp_path / "runs" / "toy").glob("TOY-train-0000/provenance.json")).read_text())
    assert prov["format"] == "abaqus-provenance/2"
    assert prov["definition_sha256"] == definition.load_definition(ds).sha256()
    assert prov["problem_sha256"] == definition.problem_sha256(ds)
    assert "sweep_sha256" not in prov


def test_limits_reach_feasible_but_not_the_deck(tmp_path):
    toml = MINIMAL_TOML + "\n[limits]\nvmax = 15.0\n"
    problem = MINIMAL_PROBLEM + "\n\ndef feasible(params):\n    return params['v0'] <= params['vmax']\n"
    ds = write_definition(tmp_path / "toy", toml=toml, problem=problem)
    d = definition.load_definition(ds)
    specs = generate.plan_cases(d, definition.load_problem(ds).feasible)
    train = [s for s in specs if s.split == "train"]
    assert train and all(s.params["v0"] <= 15.0 for s in train)
    assert all("vmax" not in s.params for s in train)
```

(`MINIMAL_TOML`'s train split draws 4 of 1024 Sobol points, so a `v0 ≤ 15` filter leaves enough.)

- [ ] **Step 2: Run to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_generate.py 2>&1 | tail -8`
Expected: failures on `load_sweep`/`load_model` no longer matching, `sweep_sha256` present, and `plan_cases` receiving a `Definition`.

- [ ] **Step 3: Implement in `generate.py`**

Delete `load_sweep` and `load_model`. Change `plan_cases`:

```python
def plan_cases(
    defn: Definition, feasible: Callable[[dict[str, Any]], bool] | None = None
) -> list[CaseSpec]:
    """Every case of every split, in file then Sobol order.

    ``feasible`` is the problem's optional hook, its declared feasibility or
    severity limit; it sees the fixed values and the ``[limits]`` too, and it
    filters sampled points only (explicit points are deliberate).
    """
    unit_factors(defn.units)
    seeds = Counter(s.seed for s in defn.splits if s.seed is not None)
    shared = sorted(seed for seed, count in seeds.items() if count > 1)
    if shared:
        raise ValueError(f"sampled splits share seed {shared[0]}; give each its own")
    check = None
    if feasible is not None:
        model_feasible = feasible

        def check(params: dict[str, float | str]) -> bool:
            return bool(model_feasible({**defn.fixed, **defn.limits, **params}))

    specs = []
    for split in defn.splits:
        sampled = set(defn.variables) | set(split.extra) | set(split.categorical)
        clash = set(defn.fixed) & sampled
        if clash:
            raise ValueError(f"{sorted(clash)} are both fixed and sampled")
        for point in sampling.sample_split(defn.variables, defn.regions, split, check):
            for variant in split.variants or (None,):
                case_id = f"{defn.case_prefix}-{split.name}-{point.index:04d}"
                case_id += f"-{variant}" if variant else ""
                if not _CASE_ID.fullmatch(case_id):
                    raise ValueError(f"case id {case_id!r} is not a safe Abaqus job name")
                params = {**defn.fixed, **point.params}
                specs.append(CaseSpec(case_id, split.name, point.index, variant, split.seed, params))
    return specs
```

In `main`: `defn = load_definition(dataset_dir)`, `problem = load_problem(dataset_dir)`, `specs = plan_cases(defn, getattr(problem, "feasible", None))`, `sweep_dir = args.work_root / defn.name`, the provenance dict gets `"format": PROVENANCE_FORMAT` (now `"abaqus-provenance/2"`), `"dataset": defn.name`, `"units": defn.units`, `"definition_sha256": defn.sha256()`, `"problem_sha256": problem_sha256(dataset_dir)`, and `text = problem.input_deck(dict(spec.params), spec.variant)`. Imports: `from structbench.datagen.definition import Definition, load_definition, load_problem, problem_sha256`. Update the module docstring's first line to "Generate a sweep's Abaqus case folders from a dataset's dataset.toml and problem.py."

- [ ] **Step 4: `archive.py` and `validate.py` read `dataset.toml`**

In `archive.py` `main`, replace the `tomllib.loads((args.dataset / "sweep.toml")...)` block by `rules = load_definition(args.dataset).retention` and update the docstring's `<dataset-dir>/sweep.toml` to `dataset.toml`. In `validate.py` replace `declared_from_toml(dataset / "sweep.toml")` with `declared_from_toml(dataset / "dataset.toml")` and the docstring likewise. In `cli/datacheck.py` change the docstring of `declared_from_toml` and the `--declaration` help from `sweep.toml` to `dataset.toml` (the function is path-based; nothing else changes). Update `tests/datagen/test_archive.py` and `test_validate.py`: they write a `sweep.toml` fixture; write `dataset.toml` instead, with the `[levels]`, `[pilot]`, `[qoi]` tables added (they load through `load_definition` now).

- [ ] **Step 5: Run the three test files, then the gates**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_generate.py tests/datagen/test_archive.py tests/datagen/test_validate.py`
Expected: all pass. Then the four gates: clean.

- [ ] **Step 6: Commit**

```bash
git add src/structbench/datagen src/structbench/cli/datacheck.py tests/datagen
git commit -m "feat(datagen): generate, archive and validate read the dataset contract; provenance/2 hashes both files"
```

---

### Task 5: The public example definition

**Files:**
- Create: `src/structbench/datagen/examples/abaqus_conformance/dataset.toml`, `.../problem.py`, `.../README.md`, `.../DATA_CARD.md`
- Modify: `pyproject.toml` (package data)
- Test: `tests/datagen/test_example.py` (new)

**Interfaces:**
- Consumes: `definition.load_definition`, `definition.load_problem`, `deck` writers including `explicit_step(scale_factor=)`.
- Produces: `structbench.datagen.template.EXAMPLE_DIR` will point here (Task 6); the example is importable as data only.

- [ ] **Step 1: Write the failing test**

`tests/datagen/test_example.py`:

```python
"""The shipped example definition: the single-rod conformance case."""

import hashlib
from importlib import resources

import numpy as np
import pytest

pytest.importorskip("scipy")

from structbench.core import Case, ElementBlock, Metadata, Nodes, Response  # noqa: E402
from structbench.datagen import definition, sampling  # noqa: E402

EXAMPLE = resources.files("structbench.datagen") / "examples" / "abaqus_conformance"


def _dir():
    with resources.as_file(EXAMPLE) as path:
        return path


def test_the_example_loads_and_names_its_pilots():
    d = definition.load_definition(_dir())
    assert d.name == "abaqus_conformance" and d.solver == "abaqus"
    assert d.split(d.pilot.split).probe and len(d.split(d.pilot.split).points) == 3
    assert d.levels.production in d.levels.pilot


def test_the_example_deck_is_byte_stable_and_requests_the_ledger():
    d, problem = definition.load_definition(_dir()), definition.load_problem(_dir())
    point = sampling.sample_split(d.variables, d.regions, d.split(d.pilot.split))[0]
    params = {**d.fixed, **point.params, d.levels.refine_key: d.levels.production}
    a, b = problem.input_deck(params, None), problem.input_deck(params, None)
    assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()
    assert "ALLPW" in a and "*DYNAMIC, EXPLICIT, SCALE FACTOR=0.5" in a


def test_the_example_qoi_returns_the_declared_names():
    d, problem = definition.load_definition(_dir()), definition.load_problem(_dir())
    point = sampling.sample_split(d.variables, d.regions, d.split(d.pilot.split))[0]
    params = {**d.fixed, **point.params, d.levels.refine_key: d.levels.production}
    mesh = problem.mesh(params)
    n, e = len(mesh.node_labels), len(mesh.element_labels)
    case = Case(
        metadata=Metadata(case_id="x", dimension=2, source_units=d.units),
        nodes=Nodes(coords=mesh.coords * 1e-3, node_id=mesh.node_labels),
        elements={"solid": ElementBlock(connectivity=mesh.connectivity - 1,
                                        element_id=mesh.element_labels,
                                        part_id=np.ones(e, np.int64))},
        materials=[],
        response=Response(
            time=np.array([0.0, 1e-6]),
            node={"displacement": np.zeros((2, n, 2), np.float32)},
            element={"solid": {"effective_plastic_strain": np.zeros((2, e), np.float32)}},
            globals_={},
        ),
    )
    out = problem.qoi(case)
    assert tuple(out) == d.qoi.names and all(isinstance(v, float) for v in out.values())
```

- [ ] **Step 2: Run to see it fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_example.py`
Expected: 3 failed with `DefinitionError: dataset.toml: not found`.

- [ ] **Step 3: Write the example**

`src/structbench/datagen/examples/abaqus_conformance/dataset.toml`:

```toml
# The example dataset definition: one copper-like rod striking a frictionless
# rigid wall (the conformance case of docs/datagen/abaqus-conformance.md).
# Every table below is part of the contract; "required" tables must be present.
# Units: t-mm-s (tonne, mm, s -> N, MPa, mJ). Canonical files are written in SI.

[dataset]                       # required: identity
name = "abaqus_conformance"     # the sweep folder name and the canonical dataset id
case_prefix = "ACX"             # case ids are <prefix>-<split>-<index>[-<variant>]
units = "t-mm-s"                # a unit label structbench.core.io.unit_factors knows
solver = "abaqus"

[declaration]                   # required: what a benchmark card would declare
unit_system = "t-mm-s"
discretisation = "FEM"
erosion = false
fields = [
    "node/displacement", "node/velocity", "node/acceleration",
    "solid/stress", "solid/effective_plastic_strain",
    "global/kinetic_energy", "global/internal_energy", "global/total_energy",
    "global/hourglass_energy", "global/plastic_dissipation",
    "global/viscous_dissipation", "global/external_work",
    "global/frictional_dissipation", "global/strain_energy",
    "global/creep_dissipation", "global/reaction_force_2_reference_node",
]

[fixed]                         # required (may be empty): constants every case gets
E = 124000.0                    # MPa, copper
nu = 0.34
rho = 8.96e-9                   # t/mm^3
L = 40.0                        # mm
D = 10.0                        # mm
sigma0 = 200.0                  # MPa, initial yield
H = 500.0                       # MPa, linear hardening slope
frame_interval = 1.0e-6         # s, the output clock
n_intervals = 400               # 401 stored frames
dt_scale = 0.5                  # the stable increment scaled by 0.5; see the conformance document

[variables]                     # required: the sampled box, name = [low, high]
v0 = [1.0e5, 2.0e5]             # mm/s (100-200 m/s)

[splits.train]                  # a sampled split: n points of a seeded Sobol sequence
n = 8
seed = 1

[splits.pilot]                  # an explicit split: the preflight's cases; probe splits are not gated
points = [
  { v0 = 1.0e5 },
  { v0 = 1.5e5 },
  { v0 = 2.0e5 },
]
probe = true

[levels]                        # required: the mesh-level convention problem.mesh follows
refine_key = "refine"           # the reserved parameter the pipeline sets
production = "1"                # the level production runs at
pilot = ["1", "2"]              # the levels the preflight runs the pilots at

[pilot]                         # required: the preflight's targets
split = "pilot"
fine_cases = ["ACX-pilot-0001"] # pilots that also run at the finest level
min_free_gb = 20.0              # the runner refuses to start below this margin
accepted_gaps = ["solver_identity_complete"]  # verification rows this dataset accepts as known gaps

[qoi]                           # required: the keys problem.qoi returns, and their units
names = ["final_length", "face_radius"]
units = ["m", "m"]

[retention]                     # optional: which ODBs the archive keeps (defaults: 5 %, seed 0)
odb_fraction = 0.05
odb_seed = 0
odb_cases = ["ACX-pilot-0001"]
```

`src/structbench/datagen/examples/abaqus_conformance/problem.py`:

```python
"""The example problem: an axisymmetric rod striking a frictionless rigid wall.

The four hooks the pipeline calls are ``input_deck``, ``mesh``, ``qoi`` and
(optionally) ``feasible``. Everything the deck contains is written by
``structbench.datagen.abaqus.deck``; nothing is edited as text. Geometry: r
along 1, z along 2; the impact face is z = 0 and the rod moves in -z.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from structbench.core import Case
from structbench.datagen.abaqus import deck

ACROSS_RADIUS = 12  # elements across the radius at level 1
WALL_REACH = 10.0  # the wall spans r in [-0.1 R, WALL_REACH R]
HARDENING_TO = 10.0  # the yield table's last plastic strain


def mesh(params: dict[str, Any]) -> deck.QuadMesh:
    """Square elements of edge R / (12 * refine); level k nests level 1 exactly."""
    length, radius = float(params["L"]), float(params["D"]) / 2.0
    refine = int(params.get("refine", 1))
    rows = refine * round(ACROSS_RADIUS * length / radius)
    return deck.structured_quad_mesh(ACROSS_RADIUS * refine, rows, 0.0, radius, 0.0, length)


def feasible(params: dict[str, Any]) -> bool:
    """No limit in the example; a dataset states its own here (see the design)."""
    return True


def input_deck(params: dict[str, Any], variant: str | None) -> str:
    length, radius = float(params["L"]), float(params["D"]) / 2.0
    grid = mesh(params)
    across, n_axial = grid.nx, grid.ny
    rp = int(grid.node_labels[-1]) + 1
    sigma0, slope = float(params["sigma0"]), float(params["H"])
    table = [(sigma0, 0.0), (sigma0 + slope * HARDENING_TO, HARDENING_TO)]
    interval = float(params["frame_interval"])

    text = deck.heading("Abaqus conformance: rod on a rigid wall")
    text += deck.node_block(grid.node_labels, grid.coords)
    text += deck.node_block([rp], [[0.0, -radius]])
    text += deck.element_block("CAX4R", "ROD", grid.element_labels, grid.connectivity)
    text += deck.node_set("ROD_NODES", grid.node_labels)
    text += deck.node_set("AXIS", [grid.node(0, j) for j in range(n_axial + 1)])
    text += deck.node_set("RP", [rp])
    text += deck.element_set("IMPACT_ROW", [grid.element(i, 0) for i in range(across)])
    text += deck.element_set("OUTER_COL", [grid.element(across - 1, j) for j in range(n_axial)])
    text += deck.element_surface("ROD_SURF", [("IMPACT_ROW", "S1"), ("OUTER_COL", "S2")])
    text += deck.analytical_surface_2d("WALL", [(-0.1 * radius, 0.0), (WALL_REACH * radius, 0.0)])
    text += deck.rigid_body(rp, "WALL")
    text += deck.material(
        "ROD_MATERIAL",
        density=float(params["rho"]),
        youngs=float(params["E"]),
        poisson=float(params["nu"]),
        plastic=table,
    )
    text += deck.solid_section("ROD", "ROD_MATERIAL")
    text += deck.surface_interaction("SMOOTH")
    text += deck.boundary("AXIS", 1)
    text += deck.encastre("RP")
    text += deck.initial_velocity("ROD_NODES", 2, -float(params["v0"]))
    body = deck.contact_pair("ROD_SURF", "WALL", "SMOOTH")
    body += deck.standard_output(
        interval,
        node_set="ROD_NODES",
        node_vars=("U", "V", "A"),
        element_set="ROD",
        element_vars=("S", "PEEQ"),
        history_node_set="RP",
        history_node_vars=("RF2",),
    )
    period = float(f"{interval * int(params['n_intervals']):.12g}")
    scale = params.get("dt_scale")
    return text + deck.explicit_step(
        "IMPACT", period, body, scale_factor=None if scale is None else float(scale)
    )


def qoi(case: Case) -> dict[str, float]:
    """Final length and impact-face radius at the last frame, in the case's units (SI)."""
    assert case.response is not None
    x0 = case.nodes.coords
    used = np.unique(case.elements["solid"].connectivity)
    x = x0[used] + case.response.node["displacement"][-1][used]
    face = np.isclose(x0[used, 1], x0[used, 1].min())
    return {
        "final_length": float(np.ptp(x[:, 1])),
        "face_radius": float(x[face, 0].max()),
    }
```

`README.md` (the example's):

```markdown
# abaqus_conformance — the example dataset definition

One copper-like rod striking a frictionless rigid wall, the conformance case of
`docs/datagen/abaqus-conformance.md`. `structbench-datagen new` copies this
directory as the template a new dataset starts from.

    structbench-datagen check    <this dir>
    structbench-datagen generate --dataset <this dir> --work-root <runs>
    structbench-datagen run      --sweep <runs>/abaqus_conformance
    structbench-datagen export   --sweep <runs>/abaqus_conformance
    structbench-datagen convert  --sweep <runs>/abaqus_conformance
    structbench-datagen validate --sweep <runs>/abaqus_conformance --dataset <this dir>
    structbench-datagen archive  --sweep <runs>/abaqus_conformance --dataset <this dir> --data-root <tree>

`preflight`, `converge` and `card` arrive with parts two and three of ADR-0071.
```

`DATA_CARD.md` (the example's):

```markdown
# <dataset name> — data card

<!-- Sections marked GENERATED are rendered by `structbench-datagen card` (part three
of ADR-0071); the others are the author's. -->

## What it is
<!-- author -->

## Model and why
<!-- author -->

## Variables and splits
<!-- GENERATED from dataset.toml -->

## Stored data
<!-- GENERATED from the declaration and the canonical files -->

## Quantities of interest
<!-- GENERATED names and ranges; the author explains them -->

## Verification of the runs
<!-- GENERATED from the datacheck record -->

## Discretization uncertainty
<!-- GENERATED from converge -->

## Validation against experiments
<!-- author -->

## Known limitations
<!-- author -->

## Provenance and reproduction
<!-- GENERATED -->
```

Add to `pyproject.toml`:

```toml
[tool.setuptools.package-data]
structbench = ["py.typed", "datagen/examples/abaqus_conformance/*"]
```

- [ ] **Step 4: Run the example tests**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_example.py`
Expected: 3 passed.

- [ ] **Step 5: Gates and commit**

Run the four gates (mypy sees `examples/abaqus_conformance/problem.py`; it is typed). Expected: clean.

```bash
git add src/structbench/datagen/examples pyproject.toml tests/datagen/test_example.py
git commit -m "feat(datagen): the public example definition (the single-rod conformance case)"
```

---

### Task 6: `new` scaffolds a definition; `check` validates one

**Files:**
- Create: `src/structbench/datagen/template.py`
- Test: `tests/datagen/test_template.py` (new)

**Interfaces:**
- Consumes: `definition.*`, `sampling.sample_split`, the example directory.
- Produces: `template.EXAMPLE_DIR: Traversable`; `template.scaffold(target: Path, name: str, *, solver: str = "abaqus") -> list[Path]` (files written); `template.check_definition(dataset_dir: Path) -> list[str]` (problems; empty means ok); `template.mesh_nests(coarse: NDArray, fine: NDArray) -> bool`; `template.main_new(argv) -> int` and `template.main_check(argv) -> int` used by the CLI.

- [ ] **Step 1: Write the failing tests**

`tests/datagen/test_template.py`:

```python
"""`new` scaffolds a definition from the example; `check` validates one without a solver."""

from pathlib import Path

import pytest
from conftest import MINIMAL_PROBLEM, MINIMAL_TOML, write_definition

pytest.importorskip("scipy")

from structbench.datagen import template  # noqa: E402


def test_scaffold_writes_the_four_files_and_renames_the_dataset(tmp_path):
    written = template.scaffold(tmp_path / "my_set", "my_set")
    names = sorted(p.name for p in written)
    assert names == ["DATA_CARD.md", "README.md", "dataset.toml", "problem.py"]
    toml = (tmp_path / "my_set" / "dataset.toml").read_text(encoding="utf-8")
    assert 'name = "my_set"' in toml and "abaqus_conformance" not in toml.split("\n")[0]


def test_scaffold_writes_lf_endings(tmp_path):
    for path in template.scaffold(tmp_path / "s", "s"):
        assert b"\r\n" not in path.read_bytes()


def test_scaffold_refuses_a_non_empty_target(tmp_path):
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "note.txt").write_text("x")
    with pytest.raises(FileExistsError):
        template.scaffold(tmp_path / "busy", "busy")


def test_a_fresh_scaffold_passes_check(tmp_path):
    template.scaffold(tmp_path / "ok", "ok")
    assert template.check_definition(tmp_path / "ok") == []


def test_check_lists_definition_errors_instead_of_raising(tmp_path):
    ds = write_definition(tmp_path / "d", toml=MINIMAL_TOML.replace("[qoi]", "[qoi_]"))
    problems = template.check_definition(ds)
    assert len(problems) == 1 and "qoi" in problems[0]


def test_check_reports_an_unstable_deck(tmp_path):
    unstable = MINIMAL_PROBLEM.replace(
        'return deck.heading("toy")', 'import time\n    return deck.heading(str(time.perf_counter_ns()))'
    )
    ds = write_definition(tmp_path / "d", problem=unstable)
    assert any("byte-stable" in p for p in template.check_definition(ds))


def test_check_names_the_level_that_does_not_nest(tmp_path):
    skewed = MINIMAL_PROBLEM.replace(
        "mesh = deck.structured_quad_mesh(k, k,", "mesh = deck.structured_quad_mesh(k, 2 * k + 1,"
    ).replace(
        "return deck.structured_quad_mesh(k, k,", "return deck.structured_quad_mesh(k, 2 * k + 1,"
    )
    ds = write_definition(tmp_path / "d", problem=skewed)
    problems = template.check_definition(ds)
    assert any("nest" in p and "2" in p for p in problems)


def test_check_reports_qoi_keys_that_differ_from_the_declaration(tmp_path):
    wrong = MINIMAL_PROBLEM.replace('{"length":', '{"span":')
    ds = write_definition(tmp_path / "d", problem=wrong)
    assert any("qoi" in p and "span" in p for p in template.check_definition(ds))


def test_the_check_command_exits_one_on_problems_and_zero_when_clean(tmp_path, capsys):
    template.scaffold(tmp_path / "ok", "ok")
    assert template.main_check([str(tmp_path / "ok")]) == 0
    ds = write_definition(tmp_path / "bad", toml=MINIMAL_TOML.replace("[levels]", "[lvls]"))
    assert template.main_check([str(ds)]) == 1
    assert "levels" in capsys.readouterr().out
```

- [ ] **Step 2: Run to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_template.py 2>&1 | tail -3`
Expected: collection error `cannot import name 'template'`.

- [ ] **Step 3: Implement `template.py`**

```python
"""Scaffold a dataset definition (``new``) and check one (``check``), ADR-0071.

``scaffold`` copies the shipped example and renames it; ``check_definition``
runs every contract check that needs no solver: the tables, the problem's
hooks, a byte-stable deck, nesting mesh levels, and QoI names that match the
declaration. Problems come back as plain sentences, one per line.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from structbench.core import Case, ElementBlock, Metadata, Nodes, Response
from structbench.datagen import sampling
from structbench.datagen.definition import (
    DefinitionError,
    load_definition,
    load_problem,
)

EXAMPLE_DIR = resources.files("structbench.datagen") / "examples" / "abaqus_conformance"
SCAFFOLD_FILES = ("dataset.toml", "problem.py", "README.md", "DATA_CARD.md")


def scaffold(target: Path, name: str, *, solver: str = "abaqus") -> list[Path]:
    """Write the template into ``target`` (which must be empty or absent)."""
    if solver != "abaqus":
        raise ValueError(f"no template for solver {solver!r}")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty")
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for filename in SCAFFOLD_FILES:
        text = (EXAMPLE_DIR / filename).read_text(encoding="utf-8")
        text = text.replace("abaqus_conformance", name).replace("\r\n", "\n")
        if filename == "README.md":
            text = f"# {name}\n\nStarted from the example definition; edit dataset.toml and problem.py.\n\n" + text.split("\n", 2)[2]
        path = target / filename
        path.write_bytes(text.encode("utf-8"))
        written.append(path)
    return written


def mesh_nests(coarse: NDArray[np.float64], fine: NDArray[np.float64]) -> bool:
    """Every coarse node coincides with a fine node (to 1e-9 of the extent)."""
    from scipy.spatial import cKDTree

    scale = float(np.ptp(fine, axis=0).max()) or 1.0
    distance, _ = cKDTree(fine).query(coarse)
    return bool(distance.max() <= 1e-9 * scale)


def _synthetic_case(defn: Any, grid: Any) -> Case:
    n, e = len(grid.node_labels), len(grid.element_labels)
    return Case(
        metadata=Metadata(case_id="check", dimension=2, source_units=defn.units),
        nodes=Nodes(coords=np.asarray(grid.coords, float), node_id=np.asarray(grid.node_labels)),
        elements={
            "solid": ElementBlock(
                connectivity=np.asarray(grid.connectivity) - 1,
                element_id=np.asarray(grid.element_labels),
                part_id=np.ones(e, np.int64),
            )
        },
        materials=[],
        response=Response(
            time=np.array([0.0, 1.0]),
            node={"displacement": np.zeros((2, n, 2), np.float32)},
            element={"solid": {"effective_plastic_strain": np.zeros((2, e), np.float32)}},
            globals_={},
        ),
    )


def check_definition(dataset_dir: Path) -> list[str]:
    """Every contract problem in plain words; an empty list means the definition passes."""
    try:
        defn = load_definition(dataset_dir)
    except DefinitionError as exc:
        return [str(exc)]
    try:
        problem = load_problem(dataset_dir)
    except DefinitionError as exc:
        return [str(exc)]
    problems: list[str] = []
    pilot = defn.split(defn.pilot.split)
    points = sampling.sample_split(defn.variables, defn.regions, pilot)
    base = {**defn.fixed, **points[0].params}
    key = defn.levels.refine_key

    params = {**base, key: defn.levels.production}
    try:
        first = problem.input_deck(dict(params), None)
        second = problem.input_deck(dict(params), None)
    except Exception as exc:  # a broken hook is a problem to report, not a crash
        return problems + [f"problem.input_deck: raised {type(exc).__name__}: {exc}"]
    if hashlib.sha256(first.encode()).digest() != hashlib.sha256(second.encode()).digest():
        problems.append("problem.input_deck: not byte-stable (two calls with the same parameters differ)")

    coarse = None
    for level in defn.levels.pilot:
        grid = problem.mesh({**base, key: level})
        if coarse is not None and not mesh_nests(coarse, np.asarray(grid.coords, float)):
            problems.append(f"problem.mesh: level {level} does not nest level {defn.levels.pilot[0]}")
        coarse = np.asarray(grid.coords, float) if coarse is None else coarse

    grid = problem.mesh(params)
    try:
        out = problem.qoi(_synthetic_case(defn, grid))
    except Exception as exc:
        return problems + [f"problem.qoi: raised {type(exc).__name__}: {exc}"]
    if tuple(out) != defn.qoi.names:
        problems.append(f"problem.qoi: returns {sorted(out)} but qoi.names declares {list(defn.qoi.names)}")
    return problems


def main_new(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="structbench-datagen new", description=scaffold.__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("--solver", default="abaqus")
    args = parser.parse_args(argv)
    try:
        written = scaffold(args.target, args.target.name, solver=args.solver)
    except (FileExistsError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    for path in written:
        print(f"wrote {path}")
    print("next: edit dataset.toml and problem.py, then `structbench-datagen check`")
    return 0


def main_check(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="structbench-datagen check", description=check_definition.__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    problems = check_definition(args.dataset)
    for line in problems:
        print(line)
    if problems:
        return 1
    defn = load_definition(args.dataset)
    print(f"ok: {defn.name}: {len(defn.splits)} splits, {len(defn.variables)} variables, "
          f"pilot {defn.pilot.split!r} at levels {list(defn.levels.pilot)}")
    return 0
```

- [ ] **Step 4: Run the template tests**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_template.py`
Expected: 9 passed. If `test_check_names_the_level_that_does_not_nest` passes only because the skew also breaks stability, adjust the skew so both meshes stay byte-stable but do not nest (the given `2 * k + 1` rows keep stability and break nesting).

- [ ] **Step 5: Gates and commit**

Run the four gates. Expected: clean.

```bash
git add src/structbench/datagen/template.py tests/datagen/test_template.py
git commit -m "feat(datagen): new scaffolds a definition from the example; check validates one without a solver"
```

---

### Task 7: The console entry point

**Files:**
- Create: `src/structbench/datagen/cli.py`
- Modify: `pyproject.toml` (`[project.scripts]`), the module docstrings of `generate.py`, `run.py`, `convert.py`, `validate.py`, `archive.py`, `abaqus/odb_export.py` (usage lines)
- Test: `tests/datagen/test_cli.py` (new)

**Interfaces:**
- Consumes: `template.main_new`, `template.main_check`, `generate.main`, `run.main`, `convert.main`, `validate.main`, `archive.main`, the exporter's file path.
- Produces: `cli.main(argv: list[str] | None = None) -> int`; `cli.export_command(abaqus: str, sweep: Path, cases: list[str] | None) -> list[str]`; `cli.EXPORTER: Traversable`; console script `structbench-datagen`.

- [ ] **Step 1: Write the failing tests**

`tests/datagen/test_cli.py`:

```python
"""structbench-datagen: one entry point, one sub-command per stage."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")

from structbench.datagen import cli  # noqa: E402


@pytest.mark.parametrize("stage", ["new", "check", "generate", "run", "export", "convert", "validate", "archive"])
def test_every_stage_answers_help(stage, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main([stage, "--help"])
    assert stop.value.code == 0
    assert stage in capsys.readouterr().out


def test_an_unknown_stage_is_refused(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["frobnicate"])
    assert stop.value.code == 2


def test_new_then_check_round_trip(tmp_path):
    assert cli.main(["new", str(tmp_path / "d")]) == 0
    assert cli.main(["check", str(tmp_path / "d")]) == 0


def test_export_builds_the_abaqus_python_command(tmp_path):
    command = cli.export_command("abaqus", tmp_path, ["A-1", "A-2"])
    assert command[:2] == ["abaqus", "python"]
    assert Path(command[2]).name == "odb_export.py" and Path(command[2]).is_file()
    assert command[3:] == ["--sweep", str(tmp_path), "--cases", "A-1", "A-2"]


def test_export_without_abaqus_exits_two(tmp_path, capsys):
    rc = cli.main(["export", "--sweep", str(tmp_path), "--abaqus", "no-such-solver-xyz"])
    assert rc == 2
    assert "no-such-solver-xyz" in capsys.readouterr().err


def test_export_runs_a_stub_solver(tmp_path):
    stub = tmp_path / "stub.py"
    stub.write_text("import sys; print(' '.join(sys.argv[1:])); sys.exit(0)\n")
    rc = cli.main(["export", "--sweep", str(tmp_path), "--abaqus", sys.executable, "--exporter-args", str(stub)])
    assert rc == 0
```

- [ ] **Step 2: Run to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_cli.py 2>&1 | tail -3`
Expected: collection error `cannot import name 'cli'`.

- [ ] **Step 3: Implement `cli.py`**

```python
"""``structbench-datagen``: the data-generation stages behind one command (ADR-0071).

    structbench-datagen new      <dir>                       scaffold a definition
    structbench-datagen check    <dir>                       validate it, no solver
    structbench-datagen generate --dataset <dir> --work-root <runs> [...]
    structbench-datagen run      --sweep <runs>/<name> [...]
    structbench-datagen export   --sweep <runs>/<name> [--cases ID ...] [--abaqus EXE]
    structbench-datagen convert  --sweep <runs>/<name> [...]
    structbench-datagen validate --sweep <runs>/<name> --dataset <dir> [...]
    structbench-datagen archive  --sweep <runs>/<name> --dataset <dir> --data-root <tree> [...]

Each stage's own options are its module's; ``export`` hands the packaged
exporter to ``abaqus python`` because it must run under the solver's Python.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from importlib import resources
from pathlib import Path

from structbench.datagen import archive, convert, generate, run, template, validate

EXPORTER = resources.files("structbench.datagen.abaqus") / "odb_export.py"

STAGES: dict[str, Callable[[list[str] | None], int]] = {
    "new": template.main_new,
    "check": template.main_check,
    "generate": generate.main,
    "run": run.main,
    "convert": convert.main,
    "validate": validate.main,
    "archive": archive.main,
}


def export_command(abaqus: str, sweep: Path, cases: Sequence[str] | None) -> list[str]:
    """The ``abaqus python odb_export.py ...`` command line for a sweep."""
    with resources.as_file(EXPORTER) as path:
        command = [abaqus, "python", str(path), "--sweep", str(sweep)]
    if cases:
        command += ["--cases", *cases]
    return command


def _export(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(prog="structbench-datagen export", description="export ODBs to abaqus-npz/1 under Abaqus's Python")
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--abaqus", default="abaqus")
    parser.add_argument("--exporter-args", nargs="*", default=None, help=argparse.SUPPRESS)  # tests: a stub in place of 'python <exporter>'
    args = parser.parse_args(argv)
    exe = shutil.which(args.abaqus)
    if exe is None:
        print(f"abaqus executable {args.abaqus!r} not found", file=sys.stderr)
        return 2
    command = export_command(exe, args.sweep, args.cases)
    if args.exporter_args is not None:
        command = [exe, *args.exporter_args, "--sweep", str(args.sweep)]
    return subprocess.run(command).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="structbench-datagen", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("stage", choices=[*STAGES, "export"])
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    rest = args.rest
    if args.stage == "export":
        return _export(rest)
    return STAGES[args.stage](rest)


if __name__ == "__main__":
    sys.exit(main())
```

Add to `pyproject.toml` under `[project.scripts]`: `structbench-datagen = "structbench.datagen.cli:main"`. Reinstall the package in the venv so the script exists: `<venv>/python -m pip install -e . --no-deps -q` (editable installs are already how the venv is set up; this only refreshes the entry points).

In each stage module's docstring, replace the `python data_generation/abaqus/<script>.py` usage line with the `structbench-datagen <stage>` form (arguments unchanged); in `odb_export.py` the line becomes `structbench-datagen export --sweep <work-root>/<name> [--cases ID ...]` with a note that it expands to `abaqus python <this file>`.

- [ ] **Step 4: Run the CLI tests, then the entry point itself**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_cli.py`
Expected: 13 passed.

Run: `<venv>/structbench-datagen check src/structbench/datagen/examples/abaqus_conformance`
Expected: `ok: abaqus_conformance: 2 splits, 1 variables, pilot 'pilot' at levels ['1', '2']`, exit 0.

- [ ] **Step 5: Gates and commit**

Run the four gates. Expected: clean.

```bash
git add src/structbench/datagen pyproject.toml tests/datagen/test_cli.py
git commit -m "feat(datagen): the structbench-datagen entry point; export hands the packaged exporter to abaqus python"
```

---

### Task 8: Docs — layering, the conformance document's home, a guide stub

**Files:**
- Modify: `docs/ARCHITECTURE.md` (the `data_generation/` paragraph, a new `### datagen/` section, the layering sentence), `data_generation/README.md`
- Move: `data_generation/abaqus/STANDARD_INPUT_BLOCK.md` → `docs/datagen/abaqus-conformance.md`; update its mentions in `src/structbench/core/io/abaqus.py`, `src/structbench/core/io/abaqus_run.py`, `src/structbench/datagen/abaqus/deck.py`, `tests/core/test_abaqus_adapter.py`, `data_generation/README.md`, `decisions/0066-reference-data-verification.md`, `decisions/0068-abaqus-second-solver.md`, `decisions/0069-abaqus-data-pipeline.md` (text mentions only; `grep -rn STANDARD_INPUT_BLOCK` lists them)
- Create: `docs/DATA_GENERATION.md` (stub)
- Test: `tests/datagen/test_docs.py` (new)

**Interfaces:** none consumed by code.

- [ ] **Step 1: Write the failing test**

`tests/datagen/test_docs.py`:

```python
"""The documents that describe the pipeline exist where the design says."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_conformance_document_lives_under_docs_and_nothing_points_at_the_old_path():
    assert (ROOT / "docs" / "datagen" / "abaqus-conformance.md").is_file()
    assert not (ROOT / "data_generation" / "abaqus").exists()
    stale = [
        p for p in [*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/**/*.py"), *ROOT.glob("docs/*.md"), ROOT / "data_generation" / "README.md"]
        if "STANDARD_INPUT_BLOCK" in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert stale == []


def test_architecture_names_datagen_in_the_layering():
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "### `datagen/`" in text
    assert "datagen" in text.split("cli/` depends on most other modules")[0]


def test_the_guide_stub_exists():
    text = (ROOT / "docs" / "DATA_GENERATION.md").read_text(encoding="utf-8")
    assert "structbench-datagen" in text and "dataset.toml" in text
```

- [ ] **Step 2: Run to see it fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_docs.py`
Expected: 3 failed.

- [ ] **Step 3: Move the conformance document and repoint every mention**

```bash
mkdir -p docs/datagen
git mv data_generation/abaqus/STANDARD_INPUT_BLOCK.md docs/datagen/abaqus-conformance.md
rm -rf data_generation/abaqus   # only an untracked __pycache__ remains after the moves
```

Then, in every file `grep -rln STANDARD_INPUT_BLOCK src tests docs data_generation decisions` lists, replace `data_generation/abaqus/STANDARD_INPUT_BLOCK.md` and bare `STANDARD_INPUT_BLOCK.md` with `docs/datagen/abaqus-conformance.md` (in `decisions/*.md` add "(moved 2026-09-27; was `data_generation/abaqus/STANDARD_INPUT_BLOCK.md`)" at the first mention only, since ADRs are records). Do the replacements with a script that writes bytes (LF).

- [ ] **Step 4: Amend `ARCHITECTURE.md`**

Replace the `data_generation/` bullet under `config.py`'s "Solver-related code is split across two locations" with:

```markdown
- **`datagen/`** inside the package holds the data-generation pipeline (ADR-0069, ADR-0071): sampling, deck generation with provenance, the job runner, conversion, verification, archive, the dataset-definition contract with its scaffold and check, and per-solver subpackages (`datagen/abaqus/`: the deck writers and the ODB exporter, which runs under the solver's own Python). A dataset is a definition (`dataset.toml`, `problem.py`) that the pipeline consumes; definitions may live outside the repository. The package still depends on no solver: `datagen` shells out to one.
- **`data_generation/`** at repo root keeps only glue that is not importable: the LS-DYNA per-dataset collectors and converters that predate `datagen`.
```

Add, before `### \`cli/\``:

```markdown
### `datagen/`

The data-generation pipeline, `structbench-datagen` (ADR-0071). Solver-agnostic stages (`sampling`, `generate`, `run`, `convert`, `validate`, `archive`, `definition`, `template`, `cli`) and per-solver subpackages (`abaqus`: `deck`, `odb_export`). It depends on `core` (readers, adapter, schema) and `verification` (through `validate`), and nothing depends on it. `odb_export.py` is package data as much as code: Python 3.10, no imports from `structbench`, handed to `abaqus python` by the CLI.
```

And in the `cli/` section's layering sentence, after "It is the outermost layer.", add: "`datagen/` sits beside `eval/` and `benchmarks/`, above `verification/`; `cli/` and `datagen/cli` are both entry-point layers."

- [ ] **Step 5: `data_generation/README.md` and the guide stub**

In `data_generation/README.md` replace the first paragraph with:

```markdown
# data_generation/

Solver-specific **glue that is not importable**: per-dataset collectors and
converters for LS-DYNA archives that predate the package's pipeline. The
Abaqus data-generation pipeline lives in the package as `structbench.datagen`
(`structbench-datagen`, ADR-0071); its conformance record is
`docs/datagen/abaqus-conformance.md` and its guide `docs/DATA_GENERATION.md`.
```

Create `docs/DATA_GENERATION.md`:

```markdown
# Data generation with StructBench

*A stub. The full guide arrives with parts two and three of ADR-0071.*

StructBench generates reference data through `structbench-datagen`. A dataset is
a definition in a directory of its own, which may be private:

- `dataset.toml` — what the dataset is: identity, declaration, constants, the
  sampled box, splits, mesh levels, the pilot cases, quantities of interest,
  ODB retention.
- `problem.py` — how one case is built and read: `input_deck`, `feasible`,
  `mesh`, `qoi`.

Start from the template and check it before anything runs:

    structbench-datagen new my_dataset
    structbench-datagen check my_dataset

Then the stages, in order: `generate`, `run`, `export`, `convert`, `validate`,
`archive`. The shipped example (`structbench/datagen/examples/abaqus_conformance`)
is the single-rod conformance case described in `docs/datagen/abaqus-conformance.md`.
The design is `docs/plans/2026-09-27-abaqus-datagen-platform-design.md`.
```

- [ ] **Step 6: Run the docs test and the gates**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_docs.py`
Expected: 3 passed. Then the four gates: clean (the full suite includes `tests/core/test_abaqus_adapter.py`, whose repointed mention must still be valid text).

- [ ] **Step 7: Commit**

```bash
git add -A docs data_generation decisions src tests
git commit -m "docs: datagen in the layering; the conformance document under docs/datagen; a guide stub"
```

---

### Task 9: Migrate dataset A to the contract and regress its decks

**Files (the private dataset repository; `<dataset dir>` is the dataset's folder there):**
- Move: `<dataset dir>/sweep.toml` → `dataset.toml`; `<dataset dir>/model.py` → `problem.py`; `test_model.py` → `test_problem.py`
- Modify: `problem.py`, `dataset.toml`, `test_problem.py`, the dataset's notes (one dated note), the private lessons file's recipe section (commands)
- Test: `test_problem.py` (the regression over every deck that ran)

**Interfaces:**
- Consumes: `structbench.datagen.definition`, `structbench.datagen.generate.plan_cases`, `deck.explicit_step(scale_factor=)`, `deck.contact_damping`, `template.check_definition`.
- Produces: nothing public.

- [ ] **Step 1: Write the failing regression test**

Rename `test_model.py` to `test_problem.py` and give it three tests:

```python
"""Dataset A on the ADR-0071 contract: the definition checks, and every deck
that ran rebuilds byte for byte against the sha256 in its provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("scipy")

from structbench.datagen import definition, generate, template  # noqa: E402

HERE = Path(__file__).parent
RUNS = HERE.parents[1] / "runs" / HERE.name
PRODUCTION = ("<the production splits>",)
LEVELS = (("<R/12 convergence split>", "1"), ("<R/48 convergence split>", "4"))


def test_the_definition_passes_check():
    assert template.check_definition(HERE) == []


def test_production_and_convergence_levels_are_the_final_setup():
    d = definition.load_definition(HERE)
    problem = definition.load_problem(HERE)
    seen: dict[str, set[tuple[str, str, str]]] = {}
    for spec in generate.plan_cases(d, problem.feasible):
        p = spec.params
        key = (str(p.get("refine", "1")), str(p.get("contact", "kinematic")), str(p.get("dt_scale", "1")))
        seen.setdefault(spec.split, set()).add(key)
    for split in PRODUCTION:
        assert seen[split] == {("<production level>", "kinematic", "<increment scale>")}, split
    for split, level in LEVELS:
        assert seen[split] == {(level, "kinematic", "<increment scale>")}, split


@pytest.mark.skipif(not RUNS.is_dir(), reason="the runs are not in this checkout")
def test_every_deck_that_ran_rebuilds_byte_for_byte():
    problem = definition.load_problem(HERE)
    checked = 0
    for folder in sorted(RUNS.glob("*")):
        prov_path = folder / "provenance.json"
        if not prov_path.is_file():
            continue
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        if not (prov["split"] in PRODUCTION or prov["split"].startswith("<convergence prefix>")):
            continue
        text = problem.input_deck(dict(prov["params"]), prov["variant"])
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == prov["inp_sha256"], folder.name
        checked += 1
    assert checked == <the number of production and convergence decks>
```

- [ ] **Step 2: Run it to see it fail**

Run: `cd <private repo> && <venv>/python -m pytest -q -p no:cacheprovider <dataset dir>/test_problem.py 2>&1 | tail -5`
Expected: `DefinitionError: dataset.toml: not found` (three failures).

- [ ] **Step 3: Rename and adapt the definition**

`git mv` `sweep.toml` to `dataset.toml` and `model.py` to `problem.py`. In `dataset.toml`, with a script that writes bytes: add `solver = "abaqus"` to `[dataset]`; move the declared limit's constant (with its comment) out of `[fixed]` into a new `[limits]` table; add `probe = true` to every split that is not a production split; add `[levels]` (the dataset's refine key, its production level, its pilot levels), `[pilot]` (a new explicit pilot split of the convergence points, its fine cases, a disk margin, the accepted gaps) and `[qoi]` (the names and units `qoi()` returns), before `[variables]`.

In `problem.py`: `import deck` becomes `from structbench.datagen.abaqus import deck`; rename `build` to `input_deck` and the mesh helper to `mesh` (its local variable in `input_deck` becomes `grid`); replace the two string edits with `deck.contact_damping(...)` at the same position and `deck.explicit_step(..., scale_factor=...)`; add `qoi(case)` built on the dataset's QoI measures. `load_problem` executes `problem.py` by path with no `sys.path` entry for its folder, so a sibling module cannot be imported: copy the measures into `problem.py` and say in both files that part two removes the copy. `feasible` reads its constant from `params`, which `plan_cases` now supplies from `[limits]`; the deck never contained it, so no deck changes.

- [ ] **Step 4: Run the private tests**

Run: `cd <private repo> && <venv>/python -m pytest -q -p no:cacheprovider <dataset dir>/ 2>&1 | tail -3`
Expected: all pass, including `test_every_deck_that_ran_rebuilds_byte_for_byte` with every deck that ran checked. A single mismatch means a writer argument changed the text; compare the rebuilt deck with the stored `.inp` by `difflib` and fix the writer call, never the test.

- [ ] **Step 5: Also confirm the pipeline's own check and dry run**

Run: `<venv>/structbench-datagen check <dataset dir>`
Expected: `ok: <name>: ...` exit 0.

Run: `<venv>/structbench-datagen generate --dataset <dataset dir> --work-root <private repo>/runs --split <a production split> --dry-run`
Expected: `<split>: <n> cases`, the split's declared size.

- [ ] **Step 6: Record and commit in the private repository**

Append a dated note to the dataset's notes: the definition moved to the ADR-0071 contract; every deck that ran rebuilds byte for byte against its provenance; no run changed. Update the recipe in the private lessons file to the `structbench-datagen` commands. Commit there.

- [ ] **Step 7: Final gates in StructBench**

Run the four gates in `<repo>`. Expected: clean; no uncommitted changes.

---

## Self-review

- **Spec coverage (part one):** package layout (Task 1, 7, 8); the writer arguments (Task 2); the contract's tables and hooks (Task 3); the stages reading it and provenance/2 (Task 4); the public example (Task 5); `new` and `check` with every check the design lists that needs no solver (Task 6; the declaration's field names are validated through `declared_from_toml` when `validate` runs, and `check` validates the table's presence — the name vocabulary check is noted for part three's `card`, which needs the same table); the CLI (Task 7); the layering amendment, the conformance document's move and the guide stub (Task 8); A's migration with the byte-for-byte regression (Task 9).
- **Placeholders:** none; every step has its code or command and an `Expected:`.
- **Type consistency:** `Definition.split()`, `Definition.sha256()`, `problem_sha256`, `plan_cases(defn, feasible)`, `template.check_definition`, `template.scaffold`, `cli.export_command` are used with the same signatures in Tasks 3–9.
- **Review Focus:** items 1, 2 → Task 3 tests; 3, 4, 6 → Task 6 tests; 5 → Task 7 test.
