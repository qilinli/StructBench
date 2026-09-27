# Validation against experiments — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (the maintainer chose native execution). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `structbench.validation` — the `taylor_copper` reference set with provenance, measures shared by experiment and simulation, a comparison that reports deviations, a byte-stable record with Markdown, and the `structbench-validate` command — rename the datagen stage `validate` to `verify`, and regenerate the first dataset's private validation record with the public tool.

**Architecture:** A new package beside `verification`, depending on `core` only: `references/` (package data), `measures/taylor.py` (outline of a canonical case, L_f / R_f / W_f), `compare.py` (pairs → deviations), `report.py` (`validation-record/1` JSON and Markdown), `cli.py`. The digitiser that produced the reference set lives in `tools/validation/` and takes the source PDF's path as an argument. The datagen stage module `validate.py` becomes `verify.py`.

**Tech Stack:** Python 3.14 (venv at `<venv>`), numpy, h5py (through `core`), tomllib, pytest; no new dependencies.

**Spec:** `docs/plans/2026-09-27-validation-against-experiments-design.md` (ADR-0072 Proposed, `decisions/0072-validation-against-experiments.md`).

## Global Constraints

- Work on branch `feat/validation-1` of `<repo>` (cut from `main` at 6d4f641); `main` moves only on the maintainer's instruction; never push.
- Gates before every commit, chained with `&&` and `set -o pipefail`: `<venv>/ruff check .`, `<venv>/ruff format --check .`, `<venv>/mypy src`, and the FULL suite `<venv>/python -m pytest -q -p no:cacheprovider` (partial collection breaks on duplicate test basenames).
- TDD for every production change: the failing test first, watched to fail, then the minimal code.
- Files written by scripts are written as bytes with LF endings (`Path.write_text` writes CRLF on Windows).
- **No private dataset detail enters the repository**: no dataset name, split name, case id, constant, run path or private repository path. Task 6 works in the private repository and is written here with placeholders. The source PDF never enters the repository.
- `validation` imports from `core` only (and the standard library, numpy, tomllib); a test asserts the import boundary like `tests/test_verification_import_boundary.py` does for `verification`.
- Every Markdown the package ships or a task writes states its sources with DOIs where it quotes a measurement.

## Review Focus

1. A canonical case whose mesh boundary is not one loop (a hole, two bodies, a node no element uses on the boundary): `outline()` must raise a named error or ignore unused nodes as documented, never return a wrong loop silently — Task 2 tests unused nodes and a two-body mesh.
2. An outline with a horizontal segment lying exactly on a fraction height (z₁ = z₂ = h) or a vertex on it: `lateral_radii` must count the vertex once and take the largest r — Task 2's synthetic outline puts a vertex exactly on a fraction height.
3. A pair whose case file lacks `response/node/displacement`, is 3D, or does not exist: `compare` reports `missing` or raises a named error with the case path, never a traceback from h5py — Task 4 tests the missing path and a 3D case.
4. A `pairs.toml` naming an unknown test id, an unknown variant, or the same (variant, test) twice: refused with the name — Task 4.
5. The record must be byte-identical across platforms: LF, sorted keys, floats written by `json.dumps` (repr), no timestamps, `ensure_ascii=False` with UTF-8 — Task 4's byte-stability test regenerates twice and compares bytes.

---

### Task 1: Rename the datagen stage `validate` to `verify`

**Files:**
- Move: `src/structbench/datagen/validate.py` → `src/structbench/datagen/verify.py`; `tests/datagen/test_validate.py` → `tests/datagen/test_verify.py`
- Modify: `src/structbench/datagen/cli.py`, `src/structbench/datagen/verify.py` (docstring, `prog`), `tests/datagen/test_cli.py`, `tests/datagen/test_verify.py`, `tests/datagen/test_docs.py`, `docs/DATA_GENERATION.md`, `docs/datagen/abaqus-conformance.md`, `docs/ARCHITECTURE.md`, `docs/plans/2026-09-27-abaqus-datagen-platform-design.md`, `CLAUDE.md`, `decisions/0071-datagen-platform.md`, `src/structbench/datagen/archive.py` (docstring mentions `validate.py`)
- Test: `tests/datagen/test_cli.py`, `tests/datagen/test_verify.py`, `tests/datagen/test_docs.py`

**Interfaces:**
- Consumes: today's `structbench.datagen.validate.main(argv) -> int` and `STAGES` in `datagen/cli.py`.
- Produces: `structbench.datagen.verify.main(argv) -> int`; `cli.STAGES["verify"]`; `cli.main(["validate", ...])` returns 2 after printing `the stage is now "verify" (ADR-0072)` to stderr.

- [ ] **Step 1: Write the failing tests**

In `tests/datagen/test_cli.py`, change `STAGES` to name `verify` instead of `validate` and add:

```python
def test_the_old_stage_name_points_to_verify(capsys):
    assert cli.main(["validate", "--help"]) == 2
    err = capsys.readouterr().err
    assert "verify" in err and "ADR-0072" in err
```

In `tests/datagen/test_docs.py` add:

```python
def test_no_document_or_code_still_says_datagen_validate():
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    files = [*repo.glob("docs/**/*.md"), repo / "CLAUDE.md", repo / "data_generation/README.md",
             *repo.glob("src/structbench/datagen/*.py")]
    hits = [str(p.relative_to(repo)) for p in files
            if re.search(r"structbench-datagen validate|datagen[./]validate\b", p.read_text(encoding="utf-8"))]
    assert hits == [], hits
```

- [ ] **Step 2: Run them to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_cli.py tests/datagen/test_docs.py 2>&1 | tail -5`
Expected: `test_every_stage_answers_help[verify]` fails (`invalid choice`), `test_the_old_stage_name_points_to_verify` fails (SystemExit 0 from `--help`), `test_no_document_or_code_still_says_datagen_validate` fails listing the files.

- [ ] **Step 3: Rename and repoint**

`git mv src/structbench/datagen/validate.py src/structbench/datagen/verify.py` and `git mv tests/datagen/test_validate.py tests/datagen/test_verify.py`. In `verify.py` set `prog="structbench-datagen verify"` and reword the docstring's first line to `Verify a sweep's runs with the ADR-0066 instrument ...`. In `cli.py`:

```python
from structbench.datagen import archive, convert, generate, run, template, verify

STAGES: dict[str, Callable[[list[str] | None], int]] = {
    "new": template.main_new,
    "check": template.main_check,
    "generate": generate.main,
    "run": run.main,
    "convert": convert.main,
    "verify": verify.main,
    "archive": archive.main,
}
RENAMED = {"validate": "verify"}  # ADR-0072: validate means comparison with experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="structbench-datagen", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("stage", choices=[*STAGES, "export", *RENAMED])
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.stage in RENAMED:
        print(
            f'the stage is now "{RENAMED[args.stage]}" (ADR-0072): '
            f"structbench-datagen {RENAMED[args.stage]} ...",
            file=sys.stderr,
        )
        return 2
    if args.stage == "export":
        return _export(args.rest)
    return STAGES[args.stage](args.rest)
```

Update the module docstring's usage block (`validate` → `verify`). Repoint `tests/datagen/test_verify.py` (`from structbench.datagen import verify`, and any `validate.` reference). Replace `structbench-datagen validate` and `datagen/validate` in every file the Files list names (the design document's stage table row reads ``| `verify` | run the ADR-0066 instrument over a sweep | renamed from `validate` (ADR-0072) |``). Append to `decisions/0071-datagen-platform.md`:

```markdown
## Note 2026-09-27 — the `validate` stage is `verify`

ADR-0072 gives validation its meaning (comparison with experiment) and
renames this pipeline's `validate` stage, which runs the ADR-0066
verification instrument, to `verify`; the old name is refused with a
pointer. Clause 1's stage list reads accordingly.
```

- [ ] **Step 4: Run the tests**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 5: Gates and commit**

Run: `<venv>/ruff check . && <venv>/ruff format --check . && <venv>/mypy src && <venv>/python -m pytest -q -p no:cacheprovider 2>&1 | tail -2`
Expected: clean; the suite passes.

```bash
git add -A src/structbench/datagen tests/datagen docs CLAUDE.md decisions/0071-datagen-platform.md
git commit -m "refactor(datagen): the validate stage is verify (ADR-0072); validate now means comparison with experiment"
```

---

### Task 2: The package and the Taylor measures

**Files:**
- Create: `src/structbench/validation/__init__.py`, `src/structbench/validation/measures/__init__.py`, `src/structbench/validation/measures/taylor.py`
- Test: `tests/validation/test_taylor_measures.py`, `tests/validation/conftest.py`, `tests/test_validation_import_boundary.py`

**Interfaces:**
- Consumes: `structbench.core.Case` (`nodes.coords` (N, 2), `elements["solid"].connectivity` (E, 4) 0-indexed, `response.node["displacement"]` (T, N, 2), `response.time`).
- Produces:
  - `outline(case: Case, frame: int = -1) -> NDArray[np.float64]` — (K, 2) closed loop of (r, z), z from the impact face (min z at that frame), in the case's length unit; raises `OutlineError` when the boundary is not one loop.
  - `final_length(rz) -> float`, `largest_radius(rz) -> float`, `lateral_radii(rz, length, fractions) -> list[float]`.
  - `FRACTIONS = (0.2, 0.25, 1/3, 0.5, 2/3)`; `SETTLED_FRACTION = 0.75`.
  - `@dataclass(frozen=True) TaylorOutcome(length, length_band, largest_radius, lateral_radii: tuple[float, ...])` and `taylor_outcome(case, fractions=FRACTIONS) -> TaylorOutcome`.

- [ ] **Step 1: Write the failing tests**

`tests/validation/conftest.py`:

```python
"""A synthetic axisymmetric rod whose deformed outline is known exactly."""

import numpy as np
import pytest

from structbench.core import Case, ElementBlock, Metadata, Nodes, Response

pytest.importorskip("h5py")


def rod_case(nr=4, nz=10, radius=1.0, length=5.0, *, squash=0.8, flare=0.5, wobble=0.0, frames=5):
    """Map z' = squash z, r' = r (1 + flare (1 - z/L)); frame k adds wobble*(-1)**k to z'."""
    rs, zs = np.linspace(0.0, radius, nr + 1), np.linspace(0.0, length, nz + 1)
    gr, gz = np.meshgrid(rs, zs)
    x0 = np.column_stack([gr.ravel(), gz.ravel()])
    i, j = (a.ravel() for a in np.meshgrid(np.arange(nr), np.arange(nz)))
    row = nr + 1
    conn = np.column_stack([i + j * row, i + 1 + j * row, i + 1 + (j + 1) * row, i + (j + 1) * row])
    x0 = np.vstack([x0, [[9.0, 9.0]]])  # a node no element uses (a reference node)
    u = np.zeros((frames, len(x0), 2))
    for k in range(frames):
        u[k, :, 0] = x0[:, 0] * flare * (1 - x0[:, 1] / length)
        u[k, :, 1] = (squash - 1) * x0[:, 1] + wobble * (-1) ** k * x0[:, 1] / length
    n, e = len(x0), len(conn)
    return Case(
        metadata=Metadata(case_id="rod", dimension=2, source_units="t-mm-s"),
        nodes=Nodes(coords=x0, node_id=np.arange(1, n + 1)),
        elements={"solid": ElementBlock(connectivity=conn, element_id=np.arange(1, e + 1), part_id=np.ones(e, np.int64))},
        materials=[],
        response=Response(
            time=np.linspace(0.0, 1.0, frames),
            node={"displacement": u.astype(np.float32)},
            element={"solid": {"effective_plastic_strain": np.zeros((frames, e), np.float32)}},
            globals_={},
        ),
    )
```

`tests/validation/test_taylor_measures.py`:

```python
"""The Taylor measures: outline of a canonical case, L_f, R_f, W_f."""

import numpy as np
import pytest
from conftest import rod_case

from structbench.validation.measures import taylor


def test_outline_is_one_closed_loop_from_the_impact_face():
    rz = taylor.outline(rod_case())
    assert rz.ndim == 2 and rz.shape[1] == 2
    assert np.allclose(rz[0], rz[-1])  # closed
    assert rz[:, 1].min() == pytest.approx(0.0)  # z from the impact face
    assert not (rz == 9.0).all(axis=1).any()  # the unused node is not on it


def test_measures_of_the_known_deformation():
    # z' = 0.8 z, r' = r (1 + 0.5 (1 - z/L)): L_f = 4, R_f = 1.5, W_f(f) = 1 + 0.5 (1 - f)
    rz = taylor.outline(rod_case())
    assert taylor.final_length(rz) == pytest.approx(4.0)
    assert taylor.largest_radius(rz) == pytest.approx(1.5)
    got = taylor.lateral_radii(rz, 4.0, taylor.FRACTIONS)
    assert got == pytest.approx([1 + 0.5 * (1 - f) for f in taylor.FRACTIONS])


def test_a_vertex_exactly_on_a_fraction_height_counts_once():
    # nz = 10 puts deformed vertices at z' = 0.4 k; f = 0.5 -> h = 2.0 is a vertex row
    rz = taylor.outline(rod_case(nz=10))
    assert taylor.lateral_radii(rz, 4.0, (0.5,)) == pytest.approx([1.25])


def test_outcome_reports_the_settled_band_of_the_length():
    still = taylor.taylor_outcome(rod_case(wobble=0.0))
    assert still.length == pytest.approx(4.0) and still.length_band == 0.0
    moving = taylor.taylor_outcome(rod_case(wobble=0.1, frames=9))
    assert moving.length_band == pytest.approx(0.1)  # half the peak-to-peak over the last quarter
    assert moving.lateral_radii == pytest.approx([1 + 0.5 * (1 - f) for f in taylor.FRACTIONS], rel=0.05)


def test_two_bodies_are_refused():
    a, b = rod_case(), rod_case()
    import dataclasses
    from structbench.core import ElementBlock, Nodes

    coords = np.vstack([a.nodes.coords, b.nodes.coords + [5.0, 0.0]])
    conn = np.vstack([a.elements["solid"].connectivity, b.elements["solid"].connectivity + len(a.nodes.coords)])
    two = dataclasses.replace(
        a,
        nodes=Nodes(coords=coords, node_id=np.arange(1, len(coords) + 1)),
        elements={"solid": ElementBlock(connectivity=conn, element_id=np.arange(1, len(conn) + 1), part_id=np.ones(len(conn), np.int64))},
        response=dataclasses.replace(a.response, node={"displacement": np.zeros((5, len(coords), 2), np.float32)}),
    )
    with pytest.raises(taylor.OutlineError, match="one closed loop"):
        taylor.outline(two)
```

`tests/test_validation_import_boundary.py` (mirror `tests/test_verification_import_boundary.py`: import `structbench.validation` in a subprocess and assert no `torch`, `structbench.benchmarks`, `structbench.eval`, `structbench.datagen`, `structbench.verification` in `sys.modules`).

- [ ] **Step 2: Run them to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/validation tests/test_validation_import_boundary.py 2>&1 | tail -3`
Expected: `ModuleNotFoundError: structbench.validation`.

- [ ] **Step 3: Write the measures**

`src/structbench/validation/__init__.py`:

```python
"""Validation against experiments (ADR-0072): reference sets, shared measures,
a comparison that reports deviations, and a byte-stable record."""
```

`src/structbench/validation/measures/__init__.py` — a docstring naming the families (`taylor`).

`src/structbench/validation/measures/taylor.py`:

```python
"""Taylor-test measures from an outline: final length, largest radius, lateral radii.

The same functions measure an experiment's digitised outline and a canonical
case's deformed boundary, so the two sides of a comparison are measured alike
(ADR-0072 clause 4). Axisymmetric layout: column 0 is r (axis at r = 0),
column 1 is z, and the impact face is the lowest z.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from structbench.core import Case

FRACTIONS = (0.2, 0.25, 1 / 3, 0.5, 2 / 3)  # the source's heights for W_f
SETTLED_FRACTION = 0.75  # the length band is taken over the last quarter of the frames


class OutlineError(ValueError):
    """The mesh boundary is not one closed loop."""


def outline(case: Case, frame: int = -1) -> NDArray[np.float64]:
    """The closed boundary of the quad mesh at ``frame``, (K, 2), z from the impact face.

    Boundary edges are those one element uses; they are chained into a loop.
    Nodes no element uses are ignored.
    """
    conn = np.asarray(case.elements["solid"].connectivity, dtype=np.int64)
    if conn.shape[1] != 4:
        raise OutlineError(f"expected 4-node quads, got {conn.shape[1]} nodes per element")
    edges: Counter[tuple[int, int]] = Counter()
    for quad in conn:
        for a, b in zip(quad, np.roll(quad, -1)):
            edges[(min(a, b), max(a, b))] += 1
    boundary = [e for e, n in edges.items() if n == 1]
    nxt: defaultdict[int, list[int]] = defaultdict(list)
    for a, b in boundary:
        nxt[a].append(b)
        nxt[b].append(a)
    if any(len(v) != 2 for v in nxt.values()):
        raise OutlineError("the mesh boundary is not one closed loop")
    start = min(nxt)
    loop, prev, node = [start], -1, start
    while True:
        a, b = nxt[node]
        prev, node = node, (a if a != prev else b)
        if node == start:
            break
        loop.append(node)
        if len(loop) > len(boundary):
            raise OutlineError("the mesh boundary is not one closed loop")
    if len(loop) != len(boundary):
        raise OutlineError("the mesh boundary is not one closed loop (more than one)")
    x = np.asarray(case.nodes.coords, dtype=np.float64)
    u = np.asarray(case.response.node["displacement"][frame], dtype=np.float64)
    rz = (x + u)[loop + [start]]
    rz[:, 1] -= rz[:, 1].min()
    return rz


def final_length(rz: ArrayLike) -> float:
    return float(np.asarray(rz, dtype=np.float64)[:, 1].max())


def largest_radius(rz: ArrayLike) -> float:
    return float(np.asarray(rz, dtype=np.float64)[:, 0].max())


def lateral_radii(rz: ArrayLike, length: float, fractions: tuple[float, ...]) -> list[float]:
    """The largest r where the outline crosses the height f * length, per fraction.

    A vertex on the height counts once (the segment that ends on it), and a
    segment lying along the height contributes its two ends.
    """
    pts = np.asarray(rz, dtype=np.float64)
    out = []
    for f in fractions:
        h = f * length
        hits: list[float] = []
        for (r1, z1), (r2, z2) in zip(pts[:-1], pts[1:]):
            if z1 == z2:
                if z1 == h:
                    hits += [r1, r2]
                continue
            if min(z1, z2) < h <= max(z1, z2):
                hits.append(r1 + (h - z1) * (r2 - r1) / (z2 - z1))
        out.append(max(hits) if hits else float("nan"))
    return out


@dataclass(frozen=True)
class TaylorOutcome:
    length: float
    length_band: float  # half the peak-to-peak of the length over the settled frames
    largest_radius: float
    lateral_radii: tuple[float, ...]


def taylor_outcome(case: Case, fractions: tuple[float, ...] = FRACTIONS) -> TaylorOutcome:
    """The measures at the last frame, in the case's own length unit."""
    rz = outline(case)
    length = final_length(rz)
    n = len(case.response.time)
    settled = range(int(np.floor(SETTLED_FRACTION * (n - 1))), n)
    lengths = [final_length(outline(case, k)) for k in settled]
    return TaylorOutcome(
        length=length,
        length_band=float((max(lengths) - min(lengths)) / 2.0),
        largest_radius=largest_radius(rz),
        lateral_radii=tuple(lateral_radii(rz, length, fractions)),
    )
```

- [ ] **Step 4: Run the tests**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/validation tests/test_validation_import_boundary.py 2>&1 | tail -3`
Expected: all pass. If the vertex test gives 1.25 twice-counted or NaN, the crossing rule's inclusive end is wrong; fix `lateral_radii`, not the test.

- [ ] **Step 5: Gates and commit**

Run the four gates. Expected: clean.

```bash
git add src/structbench/validation tests/validation tests/test_validation_import_boundary.py
git commit -m "feat(validation): the package and the Taylor measures -- outline of a canonical case, L_f, R_f, W_f, the settled band"
```

---

### Task 3: The `taylor_copper` reference set and its digitiser

**Files:**
- Create: `tools/validation/digitize_zelepugin2023.py`, `src/structbench/validation/references/__init__.py`, `src/structbench/validation/references/taylor_copper.json`, `src/structbench/validation/references/taylor_copper.md`, `src/structbench/validation/reference.py`
- Modify: `pyproject.toml` (package data `validation/references/*`)
- Test: `tests/validation/test_reference_sets.py`

**Interfaces:**
- Consumes: Task 2's measure functions.
- Produces:
  - `structbench.validation.reference`: `REFERENCE_FORMAT = "validation-reference/1"`; `@dataclass(frozen=True) Test(id, source, via, material, L0_mm, D0_mm, v0_ms, T0_K, outline_rz_mm: NDArray, Lf_mm, Rf_mm, Wf_mm: tuple[float, ...])`; `@dataclass(frozen=True) ReferenceSet(name, family, title, units, sources: tuple[dict, ...], extraction: dict, fractions: tuple[float, ...], tests: tuple[Test, ...], caveats: tuple[str, ...], sha256: str)` with `.test(id)`; `load_reference(name: str) -> ReferenceSet` (shipped sets) and `load_reference_file(path) -> ReferenceSet`; `list_references() -> list[str]`.
  - The digitiser: `python tools/validation/digitize_zelepugin2023.py <pdf> <out.json>`.

- [ ] **Step 1: Write the failing tests**

`tests/validation/test_reference_sets.py`:

```python
"""The shipped reference sets: format, provenance, and measures that reproduce."""

import json
import re
from importlib import resources

import pytest

from structbench.validation import reference
from structbench.validation.measures import taylor

REFS = resources.files("structbench.validation") / "references"


def test_taylor_copper_is_listed_and_loads():
    assert "taylor_copper" in reference.list_references()
    ref = reference.load_reference("taylor_copper")
    assert ref.family == "taylor_rod" and ref.units["length"] == "mm"
    assert [t.id for t in ref.tests] == ["1", "2", "3", "4", "5", "6"]
    assert ref.test("3").v0_ms == 162 and ref.test("3").L0_mm == 34.5


def test_every_source_has_a_doi_and_says_whether_it_was_consulted():
    ref = reference.load_reference("taylor_copper")
    for s in ref.sources:
        assert re.fullmatch(r"10\.\d{4,9}/\S+", s["doi"]), s
        assert isinstance(s["consulted"], bool)
    assert any(s["licence"] == "CC BY 4.0" for s in ref.sources)


def test_the_stated_measures_are_what_the_functions_return_on_the_outline():
    ref = reference.load_reference("taylor_copper")
    for t in ref.tests:
        rz = t.outline_rz_mm
        assert t.Lf_mm == pytest.approx(taylor.final_length(rz), abs=1e-6)
        assert t.Rf_mm == pytest.approx(taylor.largest_radius(rz), abs=1e-6)
        assert list(t.Wf_mm) == pytest.approx(taylor.lateral_radii(rz, t.Lf_mm, ref.fractions), abs=1e-6)
        assert rz[:, 1].min() == pytest.approx(0.0, abs=1e-6)


def test_the_json_is_lf_sorted_and_names_its_extraction_tool():
    raw = (REFS / "taylor_copper.json").read_bytes()
    assert b"\r\n" not in raw
    data = json.loads(raw)
    assert data["format"] == reference.REFERENCE_FORMAT
    assert data["extraction"]["tool"] == "tools/validation/digitize_zelepugin2023.py"
    assert "source_simulation" not in raw.decode()


def test_the_companion_names_every_source():
    md = (REFS / "taylor_copper.md").read_text(encoding="utf-8")
    for s in reference.load_reference("taylor_copper").sources:
        assert s["doi"] in md, s["doi"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/validation/test_reference_sets.py 2>&1 | tail -3`
Expected: `ModuleNotFoundError: structbench.validation.reference`.

- [ ] **Step 3: Write `reference.py`**

```python
"""Reference-experiment sets (``validation-reference/1``): loading and access."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REFERENCE_FORMAT = "validation-reference/1"
_REFERENCES = resources.files("structbench.validation") / "references"


class ReferenceError(ValueError):
    """A reference set that does not meet the format."""


@dataclass(frozen=True)
class Test:
    id: str
    source: str
    via: str | None
    material: str
    L0_mm: float
    D0_mm: float
    v0_ms: float
    T0_K: float
    outline_rz_mm: NDArray[np.float64]
    Lf_mm: float
    Rf_mm: float
    Wf_mm: tuple[float, ...]


@dataclass(frozen=True)
class ReferenceSet:
    name: str
    family: str
    title: str
    units: dict[str, str]
    sources: tuple[dict[str, Any], ...]
    extraction: dict[str, Any]
    fractions: tuple[float, ...]
    tests: tuple[Test, ...]
    caveats: tuple[str, ...]
    sha256: str

    def test(self, test_id: str) -> Test:
        for t in self.tests:
            if t.id == test_id:
                return t
        raise KeyError(test_id)


def list_references() -> list[str]:
    return sorted(p.name[: -len(".json")] for p in _REFERENCES.iterdir() if p.name.endswith(".json"))


def load_reference(name: str) -> ReferenceSet:
    path = _REFERENCES / f"{name}.json"
    if not path.is_file():
        raise ReferenceError(f"no reference set {name!r}; shipped: {list_references()}")
    with resources.as_file(path) as p:
        return load_reference_file(p)


def load_reference_file(path: Path) -> ReferenceSet:
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("format") != REFERENCE_FORMAT:
        raise ReferenceError(f"{path.name}: format {data.get('format')!r} is not {REFERENCE_FORMAT}")
    tests = tuple(
        Test(
            id=str(t["id"]), source=t["source"], via=t.get("via"), material=t["material"],
            L0_mm=float(t["L0_mm"]), D0_mm=float(t["D0_mm"]), v0_ms=float(t["v0_ms"]), T0_K=float(t["T0_K"]),
            outline_rz_mm=np.asarray(t["outline_rz_mm"], dtype=np.float64),
            Lf_mm=float(t["Lf_mm"]), Rf_mm=float(t["Rf_mm"]), Wf_mm=tuple(float(w) for w in t["Wf_mm"]),
        )
        for t in data["tests"]
    )
    ids = [t.id for t in tests]
    if len(set(ids)) != len(ids):
        raise ReferenceError(f"{path.name}: duplicate test ids")
    return ReferenceSet(
        name=data["name"], family=data["family"], title=data["title"], units=dict(data["units"]),
        sources=tuple(data["sources"]), extraction=dict(data["extraction"]),
        fractions=tuple(float(f) for f in data["measures"]["fractions"]), tests=tests,
        caveats=tuple(data.get("caveats", ())), sha256=hashlib.sha256(raw).hexdigest(),
    )
```

- [ ] **Step 4: Write the digitiser**

`tools/validation/digitize_zelepugin2023.py` is the private digitiser generalised: the PDF path and the output path are arguments; it keeps `PANELS`, `TESTS`, `form_streams`, `parse`, `axis_map` as they are; `main` converts cm to mm, drops the red (source simulation) curves, shifts each outline so min z = 0, computes `Lf_mm`, `Rf_mm`, `Wf_mm` with `structbench.validation.measures.taylor`, and writes the JSON below with `json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False)` as UTF-8 bytes with LF. The metadata it writes:

```python
SOURCES = [
    {"id": "S1", "citation": "Zelepugin S.A., Cherepanov R.O., Pakhnutova N.V. Optimization of Johnson-Cook constitutive model parameters using the Nesterov gradient-descent method. Materials 2023, 16, 5452.", "doi": "10.3390/ma16155452", "licence": "CC BY 4.0", "consulted": True, "role": "test conditions (Table 1); measured outlines (the black curves of Figures 4 and 5)"},
    {"id": "S2", "citation": "Zelepugin S.A., Pakhnutova N.V., Shkoda O.A., Boyangin E.N. Experimental study of the microhardness and microstructure of a copper specimen using the Taylor impact test. Metals 2022, 12, 2186.", "doi": "10.3390/met12122186", "licence": None, "consulted": False, "role": "the original source of tests 3-6, as S1 reports them"},
    {"id": "S3", "citation": "Wilkins M.L., Guinan M.W. Impact of cylinders on a rigid boundary. J. Appl. Phys. 1973, 44, 1200-1206.", "doi": "10.1063/1.1662328", "licence": None, "consulted": False, "role": "the original source of test 1, as S1 reports it"},
    {"id": "S4", "citation": "Gust W.H. High impact deformation of metal cylinders at elevated temperatures. J. Appl. Phys. 1982, 53, 3566-3575.", "doi": "10.1063/1.331136", "licence": None, "consulted": False, "role": "the original source of test 2, as S1 reports it"},
]
EXTRACTION = {
    "tool": "tools/validation/digitize_zelepugin2023.py",
    "method": "The black polylines of S1's Figures 4 and 5 (vector form XObjects) are read from the PDF drawing commands and mapped to length by a least-squares fit to the major tick marks and their labels.",
    "checks": ["worst tick-map residual < 0.005 cm (0.0007 cm on the 2026-09-25 extraction)", "each test's curve is drawn in both figures and the two agree to 0.001 cm"],
    "date": "2026-09-25",
}
CAVEATS = [
    "The measured values reach this set through S1's figures; S1 does not tabulate them, and S2, S3 and S4 were not consulted directly.",
    "Test 1's measured top edge is drawn with a slight slope; L_f is its highest point.",
    "Test 2 is ETP copper at 718 K; a room-temperature comparison leaves it out.",
    "Test 6's Figure 5 copy has one extra vertex with the same top and the same largest radius.",
]
```

The DOIs of S3 and S4 must be verified against the publisher's record before the file is committed; if either cannot be verified in session, the field holds `null` and the test's DOI pattern applies to non-null values only (adjust `test_every_source_has_a_doi...` to `if s["doi"] is not None`). Never assert an unverified DOI.

- [ ] **Step 5: Produce the reference set and its companion**

Run: `<venv>/python tools/validation/digitize_zelepugin2023.py <private path to the source PDF> src/structbench/validation/references/taylor_copper.json`
Expected: `tick-map residual (worst) 7e-04 cm; Figs 4 and 5 experimental curves identical`, then six lines of `test k: Lf ... Rf ...`.

Check against the private extraction: `L_f/L_0` of tests 1, 3, 4, 5 read 0.630, 0.763, 0.743, 0.645 to three decimals (the private README's measured column).

Write `taylor_copper.md`: title; what the set is; the sources table (id, citation, DOI, consulted, role, licence); the tests table (id, material, L₀, D₀, v₀, T₀, source); how the outlines were extracted and checked; the measures' definitions and fractions; the caveats; how to regenerate (the tool and its arguments, with the note that the PDF is not in the repository).

Add `"validation/references/*"` to `[tool.setuptools.package-data]` in `pyproject.toml` and reinstall the entry points: `<venv>/python -m uv pip install -e . --no-deps -q --python <venv>/python` (or the `uv` binary if the venv has no `uv` module).

- [ ] **Step 6: Run the tests**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/validation 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 7: Gates and commit**

Run the four gates. Expected: clean.

```bash
git add tools/validation src/structbench/validation/references src/structbench/validation/reference.py pyproject.toml tests/validation/test_reference_sets.py
git commit -m "feat(validation): the taylor_copper reference set with provenance, its digitiser, and the reference loader"
```

---

### Task 4: Comparison, record, Markdown and the command

**Files:**
- Create: `src/structbench/validation/compare.py`, `src/structbench/validation/report.py`, `src/structbench/validation/cli.py`, `src/structbench/validation/__main__.py`
- Modify: `pyproject.toml` (`structbench-validate = "structbench.validation.cli:main"`)
- Test: `tests/validation/test_compare.py`, `tests/validation/test_report.py`, `tests/validation/test_validate_cli.py`

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces:
  - `compare.py`: `CANONICAL_TO_MM = 1e3`; `@dataclass(frozen=True) Pair(variant, test, case: Path | None, status: str | None, reason: str | None)`; `load_pairs(path) -> tuple[str, dict[str, str], list[Pair]]` (reference name, variant labels, pairs; raises `PairsError` naming an unknown key, a duplicate (variant, test), or a pair with neither `case` nor `status`); `@dataclass(frozen=True) Result(variant, test, case_id, status, length_mm, length_band_mm, largest_radius_mm, lateral_radii_mm, dev_length, dev_radius, dev_lateral_rms)`; `compare(ref, pairs, variants, *, case_loader=core.io.load_case) -> Comparison(results: tuple[Result, ...], summary: dict[variant, dict[measure, {min, median, max, n}]])`. Unknown test or variant → `PairsError`; a missing file → status `missing`; a case whose outline fails → status `unmeasurable` with the error text.
  - `report.py`: `RECORD_FORMAT = "validation-record/1"`; `Record(reference: {name, format, sha256}, setup: dict[str, str], caveats: tuple[str, ...], variants: dict[str, str], results, summary, structbench_version)`; `build_record(ref, comparison, variants, setup, caveats) -> Record`; `to_json(record) -> bytes` (sorted keys, indent 2, LF, UTF-8, trailing newline); `from_json(bytes) -> Record`; `render_markdown(record, ref) -> str`.
  - `cli.py`: `main(argv) -> int`; `structbench-validate --reference NAME --pairs FILE --setup FILE --out DIR` writes `DIR/<reference>.json` and `.md`; `--list` prints each shipped set with its tests; exit 2 on `PairsError`/`ReferenceError` with one line.

- [ ] **Step 1: Write the failing tests**

`tests/validation/test_compare.py`:

```python
"""compare: pairs against a reference set, deviations reported, gaps named."""

import numpy as np
import pytest
from conftest import rod_case

from structbench.validation import compare, reference
from structbench.validation.measures import taylor

PAIRS = """\
reference = "toy"

[variants]
a = "setup A"

[[pair]]
variant = "a"
test = "1"
case = "{case}"

[[pair]]
variant = "a"
test = "2"
status = "aborted"
reason = "distortion"
"""


def toy_reference(tmp_path):
    # the measured outline is the synthetic rod's own deformed outline, scaled to mm
    rz = taylor.outline(rod_case()) * 1e3
    tests = []
    for tid in ("1", "2"):
        tests.append({"id": tid, "source": "S1", "via": None, "material": "x", "L0_mm": 5000.0, "D0_mm": 2000.0,
                      "v0_ms": 100.0, "T0_K": 298.0, "outline_rz_mm": rz.tolist(),
                      "Lf_mm": taylor.final_length(rz), "Rf_mm": taylor.largest_radius(rz),
                      "Wf_mm": taylor.lateral_radii(rz, taylor.final_length(rz), taylor.FRACTIONS)})
    data = {"format": reference.REFERENCE_FORMAT, "name": "toy", "family": "taylor_rod", "title": "t",
            "units": {"length": "mm", "velocity": "m/s", "temperature": "K"},
            "sources": [{"id": "S1", "citation": "c", "doi": "10.1000/x", "licence": None, "consulted": True, "role": "r"}],
            "extraction": {"tool": "none", "method": "synthetic", "checks": [], "date": "2026-09-27"},
            "measures": {"fractions": list(taylor.FRACTIONS)}, "tests": tests, "caveats": []}
    import json
    p = tmp_path / "toy.json"
    p.write_bytes(json.dumps(data, sort_keys=True).encode())
    return reference.load_reference_file(p)


def write_case(tmp_path, case):
    from structbench.core.io import save_case

    p = tmp_path / "case.h5"
    save_case(case, p)
    return p


def test_an_identical_case_deviates_by_zero_and_an_aborted_pair_has_no_numbers(tmp_path):
    ref = toy_reference(tmp_path)
    case = rod_case()
    # the synthetic case is in "mm" already; scale its coordinates to metres so CANONICAL_TO_MM applies
    import dataclasses
    from structbench.core import Nodes
    si = dataclasses.replace(case, nodes=Nodes(coords=case.nodes.coords * 1e-3, node_id=case.nodes.node_id),
                             response=dataclasses.replace(case.response, node={"displacement": case.response.node["displacement"] * 1e-3}))
    path = write_case(tmp_path, si)
    pairs_file = tmp_path / "pairs.toml"
    pairs_file.write_bytes(PAIRS.format(case=path.as_posix()).encode())
    name, variants, pairs = compare.load_pairs(pairs_file)
    cmp = compare.compare(ref, pairs, variants)
    done, aborted = cmp.results
    assert done.status == "completed" and done.dev_length == pytest.approx(0.0, abs=1e-9)
    assert done.dev_radius == pytest.approx(0.0, abs=1e-9) and done.dev_lateral_rms == pytest.approx(0.0, abs=1e-9)
    assert aborted.status == "aborted" and aborted.length_mm is None
    assert cmp.summary["a"]["length"]["n"] == 1


def test_a_missing_case_file_is_reported_not_raised(tmp_path):
    ref = toy_reference(tmp_path)
    pairs = [compare.Pair("a", "1", tmp_path / "nowhere.h5", None, None)]
    (result,) = compare.compare(ref, pairs, {"a": "A"}).results
    assert result.status == "missing" and result.case_id is None


@pytest.mark.parametrize("bad, expected", [
    ('test = "1"', "test"), ('variant = "a"', "variant"),
])
def test_unknown_test_or_variant_is_refused(tmp_path, bad, expected):
    ref = toy_reference(tmp_path)
    text = PAIRS.format(case="x.h5").replace(bad, bad.replace('"', '"zz', 1).replace("zz", "zz") if False else bad.replace('1"', '9"').replace('a"', 'q"'))
    pairs_file = tmp_path / "pairs.toml"
    pairs_file.write_bytes(text.encode())
    with pytest.raises(compare.PairsError, match=expected):
        name, variants, pairs = compare.load_pairs(pairs_file)
        compare.compare(ref, pairs, variants)


def test_a_duplicate_pair_is_refused(tmp_path):
    text = PAIRS.format(case="x.h5").replace('test = "2"', 'test = "1"')
    pairs_file = tmp_path / "pairs.toml"
    pairs_file.write_bytes(text.encode())
    with pytest.raises(compare.PairsError, match="twice"):
        compare.load_pairs(pairs_file)
```

(The parametrised test's replacement line is convoluted; write it plainly in the file: for `test`, replace `test = "1"` with `test = "9"`; for `variant`, replace `variant = "a"\ntest = "1"` with `variant = "q"\ntest = "1"`.)

`tests/validation/test_report.py`:

```python
"""The record: byte-stable JSON, Markdown regenerated from it."""

from conftest import rod_case

from structbench.validation import compare, report
from test_compare import toy_reference, write_case  # noqa: F401  (same directory)


def _comparison(tmp_path):
    ref = toy_reference(tmp_path)
    pairs = [compare.Pair("a", "1", None, "aborted", "distortion"), compare.Pair("a", "2", tmp_path / "none.h5", None, None)]
    return ref, compare.compare(ref, pairs, {"a": "setup A"})


def test_the_record_is_byte_identical_on_regeneration(tmp_path):
    ref, cmp = _comparison(tmp_path)
    rec = report.build_record(ref, cmp, {"a": "setup A"}, {"solver": "toy"}, ("none",))
    first, second = report.to_json(rec), report.to_json(report.build_record(ref, compare.compare(ref, [compare.Pair("a", "1", None, "aborted", "distortion"), compare.Pair("a", "2", tmp_path / "none.h5", None, None)], {"a": "setup A"}), {"a": "setup A"}, {"solver": "toy"}, ("none",)))
    assert first == second and b"\r\n" not in first and first.endswith(b"\n")
    assert report.from_json(first) == rec


def test_markdown_renders_from_the_json_and_names_sources_and_gaps(tmp_path):
    ref, cmp = _comparison(tmp_path)
    rec = report.build_record(ref, cmp, {"a": "setup A"}, {"solver": "toy"}, ("none",))
    md = report.render_markdown(rec, ref)
    assert md == report.render_markdown(report.from_json(report.to_json(rec)), ref)
    assert "10.1000/x" in md and "aborted" in md and "missing" in md and "setup A" in md
    assert "%" in md  # deviations are shown as percentages
```

(`from test_compare import ...` requires `tests/validation` to have no `__init__.py`; the basenames `test_compare.py`, `test_report.py` must not exist elsewhere in `tests/` — check with `git ls-files tests | grep -E "test_(compare|report)\.py"`; if they do, name these `test_validation_compare.py` and `test_validation_report.py` and import accordingly.)

`tests/validation/test_validate_cli.py`:

```python
"""structbench-validate: --list, and a record written from pairs and setup."""

from structbench.validation import cli


def test_list_names_the_shipped_sets(capsys):
    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "taylor_copper" in out and "162" in out


def test_writes_json_and_markdown(tmp_path):
    (tmp_path / "pairs.toml").write_bytes(
        b'reference = "taylor_copper"\n\n[variants]\na = "A"\n\n[[pair]]\nvariant = "a"\ntest = "3"\nstatus = "aborted"\nreason = "none run"\n'
    )
    (tmp_path / "setup.toml").write_bytes(b'[setup]\nsolver = "none"\n\ncaveats = ["synthetic"]\n')
    rc = cli.main(["--reference", "taylor_copper", "--pairs", str(tmp_path / "pairs.toml"),
                   "--setup", str(tmp_path / "setup.toml"), "--out", str(tmp_path / "out")])
    assert rc == 0
    assert (tmp_path / "out" / "taylor_copper.json").is_file() and (tmp_path / "out" / "taylor_copper.md").is_file()


def test_a_bad_pairs_file_exits_two_with_one_line(tmp_path, capsys):
    (tmp_path / "pairs.toml").write_bytes(b'reference = "taylor_copper"\n[variants]\na = "A"\n[[pair]]\nvariant = "a"\ntest = "42"\nstatus = "aborted"\n')
    (tmp_path / "setup.toml").write_bytes(b"[setup]\n")
    rc = cli.main(["--reference", "taylor_copper", "--pairs", str(tmp_path / "pairs.toml"), "--setup", str(tmp_path / "setup.toml"), "--out", str(tmp_path / "o")])
    assert rc == 2 and "42" in capsys.readouterr().err
```

- [ ] **Step 2: Run them to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/validation 2>&1 | tail -3`
Expected: `ModuleNotFoundError` for `compare`, `report`, `cli`.

- [ ] **Step 3: Write `compare.py`**

```python
"""Pairs of (test, canonical case) against a reference set; deviations, never verdicts."""

from __future__ import annotations

import statistics
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from structbench.core import Case
from structbench.core.io import load_case
from structbench.validation.measures import taylor
from structbench.validation.reference import ReferenceSet

CANONICAL_TO_MM = 1e3  # canonical cases are SI; reference sets are in mm
MEASURES = ("length", "largest_radius", "lateral_rms")


class PairsError(ValueError):
    """A pairs file the comparison cannot use; says which entry."""


@dataclass(frozen=True)
class Pair:
    variant: str
    test: str
    case: Path | None
    status: str | None
    reason: str | None


@dataclass(frozen=True)
class Result:
    variant: str
    test: str
    case_id: str | None
    status: str  # completed | aborted | missing | unmeasurable
    reason: str | None
    length_mm: float | None
    length_band_mm: float | None
    largest_radius_mm: float | None
    lateral_radii_mm: tuple[float, ...] | None
    dev_length: float | None
    dev_radius: float | None
    dev_lateral_rms: float | None


@dataclass(frozen=True)
class Comparison:
    results: tuple[Result, ...]
    summary: dict[str, dict[str, dict[str, float | int]]]


def load_pairs(path: Path) -> tuple[str, dict[str, str], list[Pair]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    name = data.get("reference")
    if not isinstance(name, str):
        raise PairsError(f"{path.name}: reference: a set name is required")
    variants = {str(k): str(v) for k, v in data.get("variants", {}).items()}
    pairs, seen = [], set()
    for k, raw in enumerate(data.get("pair", [])):
        where = f"{path.name}: pair {k + 1}"
        variant, test = raw.get("variant"), raw.get("test")
        if variant not in variants:
            raise PairsError(f"{where}: variant {variant!r} is not in [variants]")
        if test is None:
            raise PairsError(f"{where}: test is required")
        if (variant, str(test)) in seen:
            raise PairsError(f"{where}: variant {variant!r} test {test!r} appears twice")
        seen.add((variant, str(test)))
        if "case" in raw:
            case: Path | None = (path.parent / raw["case"]).resolve() if not Path(raw["case"]).is_absolute() else Path(raw["case"])
            status = None
        elif "status" in raw:
            case, status = None, str(raw["status"])
        else:
            raise PairsError(f"{where}: give a case file or a status")
        pairs.append(Pair(str(variant), str(test), case, status, raw.get("reason")))
    return name, variants, pairs


def _measure(case: Case) -> taylor.TaylorOutcome:
    o = taylor.taylor_outcome(case)
    s = CANONICAL_TO_MM
    return taylor.TaylorOutcome(o.length * s, o.length_band * s, o.largest_radius * s, tuple(w * s for w in o.lateral_radii))


def compare(
    ref: ReferenceSet, pairs: list[Pair], variants: dict[str, str], *, case_loader: Callable[[Path], Case] = load_case
) -> Comparison:
    results = []
    for p in pairs:
        if p.variant not in variants:
            raise PairsError(f"variant {p.variant!r} is not in [variants]")
        try:
            t = ref.test(p.test)
        except KeyError:
            raise PairsError(f"test {p.test!r} is not in reference set {ref.name!r}") from None
        if p.case is None:
            results.append(Result(p.variant, p.test, None, p.status or "aborted", p.reason, *(None,) * 7))
            continue
        if not p.case.is_file():
            results.append(Result(p.variant, p.test, None, "missing", str(p.case), *(None,) * 7))
            continue
        case = case_loader(p.case)
        try:
            o = _measure(case)
        except (taylor.OutlineError, KeyError, ValueError) as exc:
            results.append(Result(p.variant, p.test, case.metadata.case_id, "unmeasurable", f"{type(exc).__name__}: {exc}", *(None,) * 7))
            continue
        dev_w = [(a - b) / b for a, b in zip(o.lateral_radii, t.Wf_mm)]
        results.append(
            Result(
                p.variant, p.test, case.metadata.case_id, "completed", None,
                o.length, o.length_band, o.largest_radius, o.lateral_radii,
                o.length / t.Lf_mm - 1.0, o.largest_radius / t.Rf_mm - 1.0, float(np.sqrt(np.mean(np.square(dev_w)))),
            )
        )
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for v in variants:
        done = [r for r in results if r.variant == v and r.status == "completed"]
        summary[v] = {}
        for key, attr in (("length", "dev_length"), ("largest_radius", "dev_radius"), ("lateral_rms", "dev_lateral_rms")):
            values = [getattr(r, attr) for r in done]
            summary[v][key] = (
                {"min": min(values), "median": statistics.median(values), "max": max(values), "n": len(values)}
                if values else {"n": 0}
            )
    return Comparison(tuple(results), summary)
```

- [ ] **Step 4: Write `report.py`**

`Record` is a frozen dataclass of plain types (`results` as tuples of dicts via `dataclasses.asdict`), `to_json` = `(json.dumps(asdict(record), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")`, `from_json` rebuilds it (tuples for sequences), `structbench_version` from `structbench.__version__`. `render_markdown(record, ref)` writes, in this order: the title (`# Validation of <setup solver or "the setup"> against <ref.title>`); an intro line naming the reference set and its sha256 prefix; **Sources** (a table: id, citation, DOI as a link, consulted yes/no, licence); **Tests used** (id, material, L₀, D₀, v₀, T₀); **Results** — one table per measure (final length L_f/L₀, largest radius R_f/R₀, lateral profile W_f RMS deviation) with a row per test and a column per variant, cells `value (+x.x %)` for completed, the status word otherwise, and the L_f band shown as `± b` where b/L₀ ≥ 0.0005; **Summary** (per variant: min / median / max deviation and n per measure); **Setup** (the table of strings); **Caveats** (the record's, then the reference set's); a closing line naming the record format and the StructBench version. Percentages with one decimal; ratios with three decimals; bounds never approximated. Tables render deterministic column order (variants in the record's order).

- [ ] **Step 5: Write `cli.py` and `__main__.py`, register the script**

```python
"""``structbench-validate``: compare canonical cases with a reference-experiment set (ADR-0072)."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from structbench.validation import compare, reference, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="structbench-validate", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--list", action="store_true", help="name the shipped reference sets and their tests")
    parser.add_argument("--reference")
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--setup", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.list:
        for name in reference.list_references():
            ref = reference.load_reference(name)
            print(f"{name}: {ref.title}")
            for t in ref.tests:
                print(f"  test {t.id}: {t.material}, L0 {t.L0_mm} mm, D0 {t.D0_mm} mm, v0 {t.v0_ms:g} m/s, T0 {t.T0_K:g} K")
        return 0
    if not (args.reference and args.pairs and args.setup and args.out):
        parser.error("--reference, --pairs, --setup and --out are required (or --list)")
    try:
        ref = reference.load_reference(args.reference)
        name, variants, pairs = compare.load_pairs(args.pairs)
        if name != args.reference:
            raise compare.PairsError(f"{args.pairs.name} names reference {name!r}, not {args.reference!r}")
        setup_raw = tomllib.loads(args.setup.read_text(encoding="utf-8"))
        setup = {str(k): str(v) for k, v in setup_raw.get("setup", {}).items()}
        caveats = tuple(str(c) for c in setup_raw.get("caveats", ()))
        comparison = compare.compare(ref, pairs, variants)
    except (reference.ReferenceError, compare.PairsError, OSError, tomllib.TOMLDecodeError) as exc:
        print(exc, file=sys.stderr)
        return 2
    record = report.build_record(ref, comparison, variants, setup, caveats)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{ref.name}.json").write_bytes(report.to_json(record))
    (args.out / f"{ref.name}.md").write_bytes(report.render_markdown(record, ref).encode("utf-8"))
    done = sum(r.status == "completed" for r in comparison.results)
    print(f"{ref.name}: {done} of {len(comparison.results)} pairs measured -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`__main__.py`: `from structbench.validation.cli import main; raise SystemExit(main())`. In `pyproject.toml` `[project.scripts]` add `structbench-validate = "structbench.validation.cli:main"`; reinstall the entry points as in Task 3.

- [ ] **Step 6: Run the tests**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/validation 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 7: Gates and commit**

Run the four gates. Expected: clean.

```bash
git add src/structbench/validation pyproject.toml tests/validation
git commit -m "feat(validation): compare pairs with a reference set, the validation-record/1 JSON and Markdown, and structbench-validate"
```

---

### Task 5: Documentation and the layering

**Files:**
- Modify: `docs/ARCHITECTURE.md` (a `validation/` section after `verification/`; the layering line gains `{verification, validation}`), `docs/DATA_GENERATION.md` (one paragraph: after `verify`, validation against experiments is `structbench-validate`, with a pointer to the reference sets), `README.md` (the repository tree and docs list, where `datachecks` is mentioned: add the validation package and `structbench-validate`), `CLAUDE.md` (a short paragraph in the project snapshot: ADR-0072 Proposed, what is built, the rename, what stays private)
- Test: `tests/datagen/test_docs.py` already guards the rename; add to `tests/validation/test_reference_sets.py`:

```python
def test_architecture_names_the_validation_layer():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "### `validation/`" in text and "structbench-validate" in text
```

- [ ] **Step 1: Write the failing test, run it**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/validation/test_reference_sets.py -k architecture 2>&1 | tail -2`
Expected: FAIL (no such heading).

- [ ] **Step 2: Write the documentation**

ARCHITECTURE.md `### validation/`: what it holds (reference sets as package data with provenance; measures shared by experiment and simulation; the comparison; the record and its Markdown; `structbench-validate`), what it depends on (`core` only), that it reports and does not judge (ADR-0072 clause 2), and that a dataset's record stays with the dataset until admission. Layering sentence: `core ← datasets ← {verification, validation} ← {eval, benchmarks, datagen} ← cli`.

CLAUDE.md paragraph (after the datagen paragraph): **Validation against experiments (ADR-0072, Proposed; built 2026-09-27)** — the package, the `taylor_copper` set (six copper Taylor tests from an open-access compilation, outlines read from vector figures, sources with DOIs, CC BY 4.0), the shared measures, the record that reports deviations and never judges, `structbench-validate`, the `validate → verify` rename, and that the first Abaqus dataset's record is private until admission; open: a card field and landing-page link with the first public record, an LS-DYNA Taylor record needing runs at the experiments' conditions.

- [ ] **Step 3: Run the docs tests and gates, commit**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/validation tests/datagen/test_docs.py 2>&1 | tail -2`, then the four gates.
Expected: clean.

```bash
git add docs/ARCHITECTURE.md docs/DATA_GENERATION.md README.md CLAUDE.md tests/validation/test_reference_sets.py
git commit -m "docs: the validation layer, structbench-validate, and the verify rename in ARCHITECTURE, the guide, README and the snapshot"
```

---

### Task 6: Regenerate the first dataset's private record with the public tool

**Files (the private dataset repository; `<dataset dir>` is its dataset folder, `<runs>` its run folder):**
- Create: `<dataset dir>/validation/pairs.toml`, `<dataset dir>/validation/setup.toml`
- Modify: `<dataset dir>/validation/README.md` (the results tables from the new record; a dated note on the regeneration and on any difference from the earlier numbers), `<dataset dir>/DATA_CARD.md` ("Validation against experiments", numbers if they changed; "Provenance and reproduction" command), the private lessons file's recipe (`verify`, `structbench-validate`), `<dataset dir>/NOTES.md` (one dated entry)
- Delete: `<dataset dir>/validation/compare.py` (retired; history keeps it)
- Test: the private suite

**Interfaces:**
- Consumes: `structbench-datagen run|export|convert|verify`, `structbench-validate`.
- Produces: nothing public.

- [ ] **Step 1: Finish the re-runs and bring the cases to canonical**

The re-runs were launched in session (the validation probes: the literature copper model at the dataset's mesh and at the coarser one, the rate-free table, the fitted linear law; the runs at the highest speed abort, as before). When they finish:

Run: `structbench-datagen export --sweep <runs>/<sweep> --cases <the completed cases>` then `structbench-datagen convert --sweep <runs>/<sweep> --split <the probe splits> --out <runs>/<sweep>/validation_canonical` (a destination of its own, so the archived cases' `canonical/` is untouched).
Expected: one canonical file per completed case; each deck's sha256 in its provenance equals the one the earlier record used (nothing changed in the definition).

- [ ] **Step 2: Write `pairs.toml` and `setup.toml`**

`pairs.toml`: `reference = "taylor_copper"`; `[variants]` with four keys and their labels (the dataset's mesh; the coarser mesh; the rate-free table; the fitted linear law); one `[[pair]]` per (variant, test) for tests 1, 3, 4, 5, 6 as the earlier `compare.py` mapping had them, `status = "aborted"` with `reason` for the highest-speed runs, and no pair where the earlier record had none. Paths relative to the file. `setup.toml`: the earlier README's "Our simulations" table as `[setup]` strings (solver, model, wall, loading, elastic constants, the copper model, the rate-free variant, the fitted law) and `caveats` (isothermal, frictionless, axisymmetric; the measured values reach us through the compilation's figures).

- [ ] **Step 3: Produce the record and compare it with the earlier numbers**

Run: `structbench-validate --reference taylor_copper --pairs <dataset dir>/validation/pairs.toml --setup <dataset dir>/validation/setup.toml --out <dataset dir>/validation/record`
Expected: `taylor_copper: <completed> of <all> pairs measured`, the aborted ones named.

Compare every L_f/L₀ and R_f/R₀ with the earlier README's tables. Expected: L_f/L₀ agrees within the record's stated length band; R_f/R₀ within 0.01; W_f RMS within 0.5 percentage points. Any larger difference is investigated before the README changes (`systematic-debugging`), never averaged away.

- [ ] **Step 4: Update the private documents, retire `compare.py`, run the private suite**

Replace the README's results tables with the record's Markdown tables (or link the record), add the dated note, update the reproduce block (`structbench-datagen ... verify`, `structbench-validate ...`), delete `compare.py`, update DATA_CARD.md and the lessons recipe, add the NOTES entry.

Run: `<venv>/python -m pytest -q -p no:cacheprovider <dataset dir> 2>&1 | tail -2`
Expected: all pass.

- [ ] **Step 5: Commit in the private repository**

```bash
git add -A <dataset dir>/validation <dataset dir>/DATA_CARD.md <dataset dir>/NOTES.md <lessons file>
git commit -m "<dataset>: validation record regenerated with structbench-validate; compare.py retired; verify stage in the recipe"
```

- [ ] **Step 6: Final gates in StructBench**

Run the four gates in `<repo>`. Expected: clean; no uncommitted changes.

---

## Self-review

- **Spec coverage:** reference set with provenance (Task 3); measures shared by both sides, outline from boundary edges, band (Task 2); comparison with `aborted`/`missing`/`unmeasurable` and the summary (Task 4); byte-stable record and Markdown, the command and `--list` (Task 4); the rename with refusal and docs (Task 1); layering and docs (Task 5); private migration with re-runs (Task 6). Open points of the design (a figure; a `/2` format) stay open.
- **Placeholders:** none; Task 6 uses placeholders by the global constraint, not for lack of content.
- **Type consistency:** `TaylorOutcome(length, length_band, largest_radius, lateral_radii)` is used by `compare._measure`; `ReferenceSet.test(id)`, `Test.Lf_mm/Rf_mm/Wf_mm` by `compare`; `Result` fields by `report`; `PairsError`/`ReferenceError` by `cli`.
- **Review Focus:** 1 → Task 2 (`test_two_bodies_are_refused`, unused node); 2 → Task 2 (`test_a_vertex_exactly_on_a_fraction_height_counts_once`); 3 → Task 4 (`test_a_missing_case_file_is_reported_not_raised`; add a 3D-case test there if `outline` does not already refuse `coords.shape[1] != 2` — it must, with `OutlineError`); 4 → Task 4 (unknown test/variant, duplicate); 5 → Task 4 (byte-identical, LF).
