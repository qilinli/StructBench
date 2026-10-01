# Datagen platform, part two (a): the convergence engine and the layering — implementation plan

> **Executed** — merged 2026-09-27 (`486d9be`). A historical record, not instructions: do not re-run it. See `docs/plans/README.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (the maintainer chose native execution). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the convergence engine into `structbench.verification.convergence` as solution verification, give the pipeline a generic `converge` stage that writes a byte-stable record and a Markdown report from a dataset's `[levels]` and `qoi()` alone, remove the one layering exception (`datagen` reaching `verification` through `cli/datacheck`), let a dataset's `problem.py` import its own siblings, and retire the private dataset's copies (its `convergence.py` and the QoI copy in `problem.py`) against a numerical regression.

**Architecture:** The engine (Richardson extrapolation with observed-order statuses, nested-node matching, restriction of element fields with axisymmetric or planar weights, pooled relative L2 per field) lives in `verification/convergence.py` and works on `Case` objects; the headline metric `relative_l2_pooled` moves to `verification/kernels.py` and `eval.metrics` re-exports it. `datagen/converge.py` does what only the pipeline knows — pairing runs across run roots by their parameters, reading canonical files, calling the dataset's `qoi()` — and writes `convergence-record/1`. The dataset-level measurement helpers (`declared_from_toml`, `measure_cases`, `input_facts_for`) move from `cli/datacheck.py` into `verification/dataset.py`; the card vocabulary they check against moves into `core`.

**Tech Stack:** Python 3.14 (venv at `<venv>`), numpy, h5py through `core`, scipy (the `datagen` extra; imported lazily by `match_nodes` only), tomllib, pytest; no new dependencies.

**Spec:** `docs/plans/2026-09-27-abaqus-datagen-platform-design.md` (ADR-0071 Proposed, `decisions/0071-datagen-platform.md`), sections "`converge`", "Package layout", "Migration of dataset A", the ADR-0071 note of 2026-09-27 (three resolutions), and plan 1's deferred review items 5 (the layering exception) and 11 (the private QoI copy).

**Part two (b)**, a separate plan after this one: `preflight` and its stamp with the three resolutions and the new `[pilot]` fields, `generate`'s gate, `run` hardening (free space, exit 3, the estimate) and `follow`.

## Global Constraints

- Work on branch `feat/datagen-platform-2a` of `<repo>` (cut from `main` at a3a9843); `main` moves only on the maintainer's instruction; never push.
- Gates before every commit, in one unbroken `&&` chain with `set -o pipefail` (a heredoc ends a chain — commit with `-m "$(cat <<'EOF' … EOF)"`): `<venv>/ruff check .`, `<venv>/ruff format --check .`, `<venv>/mypy src`, and the FULL suite `<venv>/python -m pytest -q -p no:cacheprovider` (duplicate test basenames break partial collection; check new basenames with `git ls-files tests | xargs -n1 basename | sort | uniq -d`).
- TDD for every production change: the failing test first, watched to fail, then the minimal code.
- Files written by scripts are bytes with LF endings; JSON records are `json.dumps(..., indent=2, sort_keys=True, ensure_ascii=False) + "\n"`, no timestamps, no absolute paths.
- **No private dataset detail enters the repository**: no dataset name, split name, case id, constant, count, run path or private repository path. Task 7 works in the private repository and is written here with placeholders.
- Layering after this plan, held by AST boundary tests: `verification` imports `core` and `datasets` only; `validation` imports `core` only; `datagen` imports `core`, `datasets`, `verification`, `validation` and itself — never `cli`, `benchmarks`, `eval`, `models`.
- Public names keep working: `structbench.eval.metrics.relative_l2_pooled`, `structbench.cli.datacheck.{declared_from_toml, measure_cases, input_facts_for, measure_dataset, declared_from_spec}`, `structbench.benchmarks.card.Discretisation`.

## Review Focus

1. A case with two runs at one level (a production run and a probe at the same level, or the same case in two run roots): `converge` must take one deterministically — the production run first, then root order — and say which in the record; never average or silently pick the last — Task 5 tests it.
2. Level labels that are not numbers, or three levels whose ratios differ (`"1", "2", "3"`): no Richardson extrapolation, a note naming the labels, field errors still computed — Task 5 tests it.
3. A finest-level case that lacks a field the coarser one stores (or the reverse): that field is skipped with a note, the other fields and QoIs still reported — Task 2 and Task 5 test it.
4. Meshes that do not nest (a probe at another geometry parameter paired by a mistaken key, or a dataset whose `mesh()` does not nest): refused per case with the reason in the record; the other cases proceed — Task 5 tests it.
5. The record must be byte-identical on regeneration and across machines: case ids only (no paths), sorted keys, LF; `render_markdown` deterministic — Task 5's byte-stability test regenerates twice and compares bytes.

---

### Task 1: The headline metric moves into `verification.kernels`; `eval` re-exports it

**Files:**
- Modify: `src/structbench/verification/kernels.py` (add `relative_l2_pooled`), `src/structbench/eval/metrics.py` (import and re-export it; delete the body)
- Test: `tests/verification/test_kernels.py` (add), `tests/eval/test_metrics.py` (unchanged, must still pass)

**Interfaces:**
- Consumes: nothing new.
- Produces: `structbench.verification.kernels.relative_l2_pooled(pred_field, gt_field, mask=None, eps=1e-12) -> float`, the same object as `structbench.eval.metrics.relative_l2_pooled`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/verification/test_kernels.py`:

```python
def test_relative_l2_pooled_pools_frames_points_and_components():
    # ‖pred − gt‖ / ‖gt‖ over everything: (3, 4, 5) against (0, 0, 12) → 5/13
    from structbench.verification.kernels import relative_l2_pooled

    gt = np.zeros((2, 3, 2))
    gt[1, 2, 1] = 12.0
    pred = gt.copy()
    pred[0, 0, 0], pred[0, 1, 1], pred[1, 0, 0] = 3.0, 4.0, 0.0
    assert relative_l2_pooled(pred, gt) == pytest.approx(5.0 / 13.0)


def test_eval_metrics_re_exports_the_kernel():
    from structbench.eval import metrics
    from structbench.verification import kernels

    assert metrics.relative_l2_pooled is kernels.relative_l2_pooled
```

- [ ] **Step 2: Run them to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/verification/test_kernels.py 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'relative_l2_pooled'`.

- [ ] **Step 3: Move the function**

Cut `relative_l2_pooled` (its docstring included, ADR-0055 references kept) from `eval/metrics.py` into `verification/kernels.py`, unchanged. In `eval/metrics.py`, where it stood:

```python
from ..verification.kernels import relative_l2_pooled  # the headline metric (ADR-0055) lives with verification
```

and keep the name in the module (ruff: add `# noqa: F401` only if nothing else in the module uses it; `eval/rollout.py` imports it from `.metrics`, which still works).

- [ ] **Step 4: Run the tests**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/verification/test_kernels.py tests/eval/test_metrics.py tests/test_verification_import_boundary.py 2>&1 | tail -3`
Expected: all pass (the boundary test: `verification` still imports only `core`/`datasets`; `eval` may import `verification`).

- [ ] **Step 5: Gates and commit**

```bash
git add src/structbench/verification/kernels.py src/structbench/eval/metrics.py tests/verification/test_kernels.py
git commit -m "refactor(verification): the pooled relative L2 (ADR-0055) lives in verification.kernels; eval re-exports it"
```

---

### Task 2: `structbench.verification.convergence` — the engine on `Case` objects

**Files:**
- Create: `src/structbench/verification/convergence.py`
- Test: `tests/verification/test_convergence.py`

**Interfaces:**
- Consumes: `structbench.core.Case`; `kernels.relative_l2_pooled`; scipy's `cKDTree` (lazy).
- Produces:
  - `RATIO_DEFAULT = 2.0`, `SAFETY_DEFAULT = 1.25`.
  - `@dataclass(frozen=True) Extrapolation(status: str, order: float | None, extrapolated: float | None, gci_fine: float | None, error_vs_extrapolated: dict[str, float] | None, coarsest_vs_finest: float)`; statuses `monotone | oscillatory | diverging | flat`.
  - `richardson(values: dict[str, float], *, ratio: float, safety: float = SAFETY_DEFAULT) -> Extrapolation` — `values` maps three level labels (coarse, medium, fine, in that order) to values.
  - `match_nodes(coarse_xy, fine_xy, *, tol=1e-9) -> NDArray[np.int64]`; raises `NestingError` (a `ValueError`).
  - `element_weights(coords, conn, *, axisymmetric: bool) -> NDArray[np.float64]` — per quad, ∫ r dA (axisymmetric) or ∫ dA (planar).
  - `class Restriction(coarse_xy, coarse_conn, fine_xy, fine_conn, *, axisymmetric=True)` with `.apply(values)`: `(T, n_fine, ...) -> (T, n_coarse, ...)`; raises `NestingError`.
  - `field_errors(coarse: Case, fine: Case, *, node_fields: Sequence[str] | None = None, element_fields: Sequence[str] | None = None, axisymmetric: bool = True) -> dict[str, float]` — pooled relative L2 per field of the coarse solution against the fine one, on the coarse mesh; fields default to those stored in both cases; raises `NestingError` or `ValueError` (time grids) with a sentence.

- [ ] **Step 1: Write the failing tests**

`tests/verification/test_convergence.py` (a synthetic nested-grid helper; no h5):

```python
"""The convergence engine: Richardson with statuses, nesting, restriction, field errors."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("scipy")

from structbench.core import Case, ElementBlock, Material, Metadata, Nodes, Response  # noqa: E402
from structbench.verification import convergence as cv  # noqa: E402


def grid(nr, nz, radius=1.0, length=2.0):
    rs, zs = np.linspace(0.0, radius, nr + 1), np.linspace(0.0, length, nz + 1)
    gr, gz = np.meshgrid(rs, zs)
    coords = np.column_stack([gr.ravel(), gz.ravel()])
    i, j = (a.ravel() for a in np.meshgrid(np.arange(nr), np.arange(nz)))
    row = nr + 1
    conn = np.column_stack([i + j * row, i + 1 + j * row, i + 1 + (j + 1) * row, i + (j + 1) * row])
    return coords, conn


def case(xy, conn, node=None, element=None, frames=3):
    n, e = len(xy), len(conn)
    node = node or {"displacement": np.zeros((frames, n, 2), np.float32)}
    element = element or {"stress": np.zeros((frames, e, 6), np.float32)}
    return Case(
        metadata=Metadata(case_id="c", dimension=2, source_units="t-mm-s"),
        nodes=Nodes(coords=xy, node_id=np.arange(1, n + 1)),
        elements={"solid": ElementBlock(connectivity=conn, element_id=np.arange(1, e + 1), part_id=np.ones(e, np.int64))},
        materials=[Material(material_id=1, source_model="toy", source_params={})],
        response=Response(time=np.linspace(0.0, 1.0, frames), node=node, element={"solid": element}, globals_={}),
    )


def fields(xy, conn, frames=3, scale=1.0):
    t = np.arange(1, frames + 1)[:, None, None]
    u = t * np.stack([xy[:, 0] * 0.1, -xy[:, 1] * 0.2], axis=-1)[None] * scale
    c = xy[conn].mean(axis=1)
    s = t * np.stack([c[:, 0], c[:, 1], c[:, 0] + c[:, 1], 0 * c[:, 0], 2 * c[:, 1], 3 * c[:, 0]], -1)[None]
    return {"displacement": u, "velocity": 2 * u}, {"stress": s, "effective_plastic_strain": t[..., 0] * (1 + c[:, 1])[None]}


# --- Richardson -----------------------------------------------------------------------

def test_second_order_data_give_order_two_and_the_exact_limit():
    # f = 1 + 0.04 (h/h0)^2 on h, h/2, h/4
    x = cv.richardson({"1": 1.04, "2": 1.01, "4": 1.0025}, ratio=2.0)
    assert x.status == "monotone" and x.order == pytest.approx(2.0)
    assert x.extrapolated == pytest.approx(1.0)
    assert x.error_vs_extrapolated == pytest.approx({"1": 0.04, "2": 0.01, "4": 0.0025})
    assert x.gci_fine == pytest.approx(1.25 * 0.0075 / 1.0025 / 3.0)
    assert x.coarsest_vs_finest == pytest.approx(0.0375 / 1.0025)


@pytest.mark.parametrize(
    "values, status",
    [
        ({"1": 1.04, "2": 0.99, "4": 1.0}, "oscillatory"),
        ({"1": 1.00, "2": 1.01, "4": 1.05}, "diverging"),
        ({"1": 1.02, "2": 1.0, "4": 1.0}, "flat"),
    ],
)
def test_data_without_an_order_say_why(values, status):
    x = cv.richardson(values, ratio=2.0)
    assert x.status == status and x.order is None and x.extrapolated is None
    assert x.error_vs_extrapolated is None and x.coarsest_vs_finest > 0


def test_richardson_needs_exactly_three_levels():
    with pytest.raises(ValueError, match="three"):
        cv.richardson({"1": 1.0, "2": 1.0}, ratio=2.0)


# --- nesting and restriction ----------------------------------------------------------

@pytest.mark.parametrize("k", [2, 4])
def test_every_coarse_node_is_found_in_a_nested_mesh(k):
    coarse, _ = grid(3, 5, 1.0, 2.1)
    fine, _ = grid(3 * k, 5 * k, 1.0, 2.1)
    idx = cv.match_nodes(coarse, fine)
    np.testing.assert_allclose(fine[idx], coarse, atol=1e-12)


def test_a_mesh_that_does_not_nest_is_refused():
    coarse, _ = grid(3, 5, 1.0, 2.1)
    fine, _ = grid(6, 11, 1.0, 2.1)
    with pytest.raises(cv.NestingError, match="nest"):
        cv.match_nodes(coarse, fine)


def test_axisymmetric_weights_are_the_first_moment_and_planar_the_area():
    xy, conn = grid(2, 1, 1.0, 1.0)  # two elements: r in [0, .5] and [.5, 1], z in [0, 1]
    assert cv.element_weights(xy, conn, axisymmetric=True) == pytest.approx([0.125, 0.375])
    assert cv.element_weights(xy, conn, axisymmetric=False) == pytest.approx([0.5, 0.5])


def test_restriction_is_the_weighted_average_of_the_children():
    # a field equal to r, averaged with weight r: exactly 2/3 (b^3 - a^3)/(b^2 - a^2)
    cxy, cconn = grid(2, 3, 1.0, 1.5)
    fxy, fconn = grid(8, 12, 1.0, 1.5)

    def r_mean(xy, conn):
        r = xy[conn][..., 0]
        a, b = r.min(axis=1), r.max(axis=1)
        return 2.0 / 3.0 * (b**3 - a**3) / (b**2 - a**2)

    values = np.stack([r_mean(fxy, fconn)] * 2)
    out = cv.Restriction(cxy, cconn, fxy, fconn, axisymmetric=True).apply(values)
    np.testing.assert_allclose(out, np.stack([r_mean(cxy, cconn)] * 2), rtol=1e-12)
    planar = cv.Restriction(cxy, cconn, fxy, fconn, axisymmetric=False).apply(values)
    mid = fxy[fconn][..., 0].mean(axis=1)  # plain mean of the children's r_mean == coarse r-midpoint mean
    assert planar.shape == (2, len(cconn))
    assert not np.allclose(planar, out)  # the two weightings differ off the axis


def test_restriction_keeps_trailing_components_and_refuses_a_stranger():
    cxy, cconn = grid(2, 2, 1.0, 1.0)
    fxy, fconn = grid(4, 4, 1.0, 1.0)
    values = np.ones((3, len(fconn), 6)) * np.arange(6)
    out = cv.Restriction(cxy, cconn, fxy, fconn).apply(values)
    assert out.shape == (3, len(cconn), 6)
    np.testing.assert_allclose(out, np.ones((3, len(cconn), 6)) * np.arange(6))
    oxy, oconn = grid(4, 4, 1.2, 1.0)  # wider: children outside every coarse element
    with pytest.raises(cv.NestingError):
        cv.Restriction(cxy, cconn, oxy, oconn)


# --- field errors on cases ------------------------------------------------------------

def _pair(scale_disp=1.0, drop=None):
    cxy, cconn = grid(2, 4)
    fxy, fconn = grid(4, 8)
    fn, fe = fields(fxy, fconn)
    idx = cv.match_nodes(cxy, fxy)
    R = cv.Restriction(cxy, cconn, fxy, fconn)
    cn = {k: v[:, idx] * (scale_disp if k == "displacement" else 1.0) for k, v in fn.items()}
    ce = {k: R.apply(v) for k, v in fe.items()}
    if drop:
        fn = {k: v for k, v in fn.items() if k != drop}
    return case(cxy, cconn, cn, ce), case(fxy, fconn, fn, fe)


def test_identical_solutions_have_zero_error_in_every_stored_field():
    errors = cv.field_errors(*_pair())
    assert set(errors) == {"displacement", "velocity", "stress", "effective_plastic_strain"}
    assert all(e == pytest.approx(0.0, abs=1e-6) for e in errors.values())


def test_a_ten_percent_displacement_error_reads_as_ten_percent():
    errors = cv.field_errors(*_pair(scale_disp=1.1))
    assert errors["displacement"] == pytest.approx(0.1, rel=1e-5)
    assert errors["stress"] == pytest.approx(0.0, abs=1e-6)


def test_a_field_missing_on_one_side_is_left_out_not_raised():
    errors = cv.field_errors(*_pair(drop="velocity"))
    assert "velocity" not in errors and "displacement" in errors


def test_different_time_grids_are_refused():
    coarse, fine = _pair()
    fine = dataclasses.replace(fine, response=dataclasses.replace(fine.response, time=fine.response.time * 2))
    with pytest.raises(ValueError, match="time"):
        cv.field_errors(coarse, fine)
```

- [ ] **Step 2: Run them to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/verification/test_convergence.py 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'convergence'`.

- [ ] **Step 3: Write the engine**

`src/structbench/verification/convergence.py`:

```python
"""Solution verification across nested mesh levels (ADR-0071 clause 4).

Richardson extrapolation with the observed order and its statuses, nested-node
matching, restriction of element fields onto the coarser mesh with
axisymmetric or planar volume weights, and the pooled relative L2 of each
field (ADR-0055) of a coarser solution against the finest. Everything works
on ``Case`` objects; pairing runs and reading files is the pipeline's job.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from structbench.core import Case
from structbench.verification.kernels import relative_l2_pooled

RATIO_DEFAULT = 2.0
SAFETY_DEFAULT = 1.25  # Roache's factor for a three-level study


class NestingError(ValueError):
    """Two meshes that are not one the refinement of the other."""


@dataclass(frozen=True)
class Extrapolation:
    status: str  # monotone | oscillatory | diverging | flat
    order: float | None
    extrapolated: float | None
    gci_fine: float | None
    error_vs_extrapolated: dict[str, float] | None  # per level label
    coarsest_vs_finest: float


def richardson(values: dict[str, float], *, ratio: float, safety: float = SAFETY_DEFAULT) -> Extrapolation:
    """Observed order and extrapolated value from three levels, coarse to fine."""
    if len(values) != 3:
        raise ValueError(f"Richardson extrapolation needs exactly three levels, got {len(values)}")
    (lc, fc), (lm, fm), (lf, ff) = values.items()
    e32, e21 = fc - fm, fm - ff
    coarsest_vs_finest = abs(fc - ff) / abs(ff)
    if e21 == 0.0:
        status = "flat"
    elif e32 / e21 <= 0.0:
        status = "oscillatory"
    elif abs(e32) <= abs(e21):
        status = "diverging"
    else:
        status = "monotone"
    if status != "monotone":
        return Extrapolation(status, None, None, None, None, coarsest_vs_finest)
    p = math.log(e32 / e21) / math.log(ratio)
    gain = ratio**p - 1.0
    extrapolated = ff + (ff - fm) / gain
    errors = {label: abs(v - extrapolated) / abs(extrapolated) for label, v in values.items()}
    return Extrapolation(status, p, extrapolated, safety * abs(e21 / ff) / gain, errors, coarsest_vs_finest)


def match_nodes(coarse_xy: ArrayLike, fine_xy: ArrayLike, *, tol: float = 1e-9) -> NDArray[np.int64]:
    """For each coarse node, the index of the fine node at the same place."""
    from scipy.spatial import cKDTree  # the datagen extra; only this function needs it

    coarse, fine = np.asarray(coarse_xy, float), np.asarray(fine_xy, float)
    scale = float(np.ptp(fine, axis=0).max()) or 1.0
    distance, index = cKDTree(fine).query(coarse)
    if distance.max() > tol * scale:
        raise NestingError(
            f"the meshes do not nest: a coarse node is {distance.max():.3g} from the nearest fine node"
        )
    return np.asarray(index, dtype=np.int64)


def element_weights(coords: ArrayLike, conn: ArrayLike, *, axisymmetric: bool) -> NDArray[np.float64]:
    """Per element: the integral of r dA (axisymmetric) or dA (planar) over the quad."""
    xy = np.asarray(coords, float)[np.asarray(conn, np.int64)]  # (E, 4, 2)
    x, y = xy[..., 0], xy[..., 1]
    xn, yn = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    cross = x * yn - xn * y
    if axisymmetric:
        return np.abs((cross * (x + xn)).sum(axis=1)) / 6.0
    return np.abs(cross.sum(axis=1)) / 2.0


class Restriction:
    """Average fine element values onto the coarse elements that contain them.

    Each fine element goes to the coarse element whose axis-aligned box holds
    its centroid (structured meshes), weighted by ``element_weights``.
    """

    def __init__(self, coarse_xy, coarse_conn, fine_xy, fine_conn, *, axisymmetric: bool = True) -> None:
        cxy, cconn = np.asarray(coarse_xy, float), np.asarray(coarse_conn, np.int64)
        fxy, fconn = np.asarray(fine_xy, float), np.asarray(fine_conn, np.int64)
        boxes = cxy[cconn]
        lo, hi = boxes.min(axis=1), boxes.max(axis=1)
        centroid = fxy[fconn].mean(axis=1)
        r_lines, z_lines = np.unique(lo[:, 0]), np.unique(lo[:, 1])
        col = np.searchsorted(r_lines, centroid[:, 0], side="right") - 1
        row = np.searchsorted(z_lines, centroid[:, 1], side="right") - 1
        owner = {
            (int(i), int(j)): e
            for e, (i, j) in enumerate(zip(np.searchsorted(r_lines, lo[:, 0]), np.searchsorted(z_lines, lo[:, 1]), strict=True))
        }
        parent = np.array([owner.get((int(i), int(j)), -1) for i, j in zip(col, row, strict=True)])
        safe = parent.clip(0)
        inside = (parent >= 0) & np.all((centroid >= lo[safe]) & (centroid <= hi[safe]), axis=1)
        if not inside.all():
            raise NestingError(f"{int((~inside).sum())} fine elements lie in no coarse element")
        counts = np.bincount(parent, minlength=len(cconn))
        if counts.min() == 0:
            raise NestingError("a coarse element has no fine children: the meshes do not nest")
        weight = element_weights(fxy, fconn, axisymmetric=axisymmetric)
        order = np.argsort(parent, kind="stable")
        self._order = order
        self._starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        total = np.bincount(parent, weights=weight, minlength=len(cconn))
        self._weight = (weight / total[parent])[order]
        self.n_coarse = len(cconn)

    def apply(self, values: ArrayLike) -> NDArray[np.float64]:
        """``(T, n_fine, ...)`` -> ``(T, n_coarse, ...)``."""
        v = np.asarray(values, float)
        frames, rest = v.shape[0], v.shape[2:]
        flat = v.reshape(frames, v.shape[1], -1)[:, self._order, :] * self._weight[None, :, None]
        out = np.add.reduceat(flat, self._starts, axis=1)
        return out.reshape(frames, self.n_coarse, *rest)


def _mesh(case: Case) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    return np.asarray(case.nodes.coords, float), np.asarray(case.elements["solid"].connectivity, np.int64)


def field_errors(
    coarse: Case,
    fine: Case,
    *,
    node_fields: Sequence[str] | None = None,
    element_fields: Sequence[str] | None = None,
    axisymmetric: bool = True,
) -> dict[str, float]:
    """Pooled relative L2 of the coarse solution against the fine one, per stored field.

    Node fields are compared at the coarse nodes, which the fine mesh contains;
    element fields on the coarse elements, the fine children restricted with
    volume weights. Fields default to those both cases store; one stored on a
    single side is left out.
    """
    if coarse.response is None or fine.response is None:
        raise ValueError("both cases must store a response")
    tc, tf = np.asarray(coarse.response.time), np.asarray(fine.response.time)
    if tc.shape != tf.shape or not np.allclose(tc, tf, rtol=1e-9, atol=0.0):
        raise ValueError("the two runs are not on the same time grid")
    cxy, cconn = _mesh(coarse)
    fxy, fconn = _mesh(fine)
    used = np.unique(cconn)  # a rigid body's reference node is no mesh node
    index = match_nodes(cxy[used], fxy)
    cn, fn = coarse.response.node, fine.response.node
    ce, fe = coarse.response.element.get("solid", {}), fine.response.element.get("solid", {})
    names_n = [n for n in (node_fields or cn) if n in cn and n in fn]
    names_e = [n for n in (element_fields or ce) if n in ce and n in fe]
    errors = {n: float(relative_l2_pooled(np.asarray(cn[n])[:, used], np.asarray(fn[n])[:, index])) for n in names_n}
    if names_e:
        restrict = Restriction(cxy, cconn, fxy, fconn, axisymmetric=axisymmetric)
        for n in names_e:
            errors[n] = float(relative_l2_pooled(np.asarray(ce[n]), restrict.apply(fe[n])))
    return errors
```

- [ ] **Step 4: Run the tests**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/verification/test_convergence.py tests/test_verification_import_boundary.py 2>&1 | tail -3`
Expected: all pass. (`test_restriction_is_the_weighted_average...` has an unused `mid`; delete that line if ruff flags it.)

- [ ] **Step 5: Gates and commit**

```bash
git add src/structbench/verification/convergence.py tests/verification/test_convergence.py
git commit -m "feat(verification): the convergence engine -- Richardson with statuses, nested-node matching, weighted restriction, field errors on cases"
```

---

### Task 3: The dataset-level measurement helpers move into `verification.dataset`; the layering exception goes

**Files:**
- Create: `src/structbench/verification/dataset.py`
- Modify: `src/structbench/core/evidence.py` (add `CardDiscretisation = Literal["SPH", "FEM", "coupled"]` and export it), `src/structbench/benchmarks/card.py` (`from ..core.evidence import CardDiscretisation as Discretisation`), `src/structbench/cli/datacheck.py` (import `declared_from_toml`, `input_facts_for`, `measure_cases` from `verification.dataset`; keep `declared_from_spec`, `measure_dataset`, the CLI), `src/structbench/datagen/verify.py` (import from `verification.dataset`), `docs/ARCHITECTURE.md` (the exception paragraph in `datagen/` goes; `verification/` gains `dataset.py`), `CLAUDE.md` (the exception clause goes)
- Test: `tests/test_datagen_import_boundary.py` (create), `tests/cli/test_datacheck_declaration.py` (unchanged, must pass), `tests/verification/test_dataset_helpers.py` (create: the helpers import from their new home; `CardDiscretisation` and `benchmarks.card.Discretisation` are the same object)

**Interfaces:**
- Consumes: `core.io.{read_case, lsdyna_run.read_input_facts, abaqus_run.read_abaqus_input_facts}`, `core.evidence.{DeclaredFacts, RunEvidence, AbsenceReason, InputFacts, CardDiscretisation}`, `verification.{measures.measure_case, quantities.CATALOGUE, results.*}`.
- Produces: `structbench.verification.dataset.{declared_from_toml, input_facts_for, measure_cases}` with today's signatures; `structbench.core.evidence.CardDiscretisation`.

- [ ] **Step 1: Write the failing tests**

`tests/test_datagen_import_boundary.py` (the verification boundary test's resolver, `_ALLOWED = {"core", "datasets", "verification", "validation", "datagen"}`, over `src/structbench/datagen/**/*.py`). `tests/verification/test_dataset_helpers.py`:

```python
"""The dataset-level helpers live in verification; the CLI and datagen import them from there."""


def test_the_helpers_import_from_verification_and_the_cli_re_exports_them():
    from structbench.cli import datacheck
    from structbench.verification import dataset

    assert datacheck.declared_from_toml is dataset.declared_from_toml
    assert datacheck.measure_cases is dataset.measure_cases
    assert datacheck.input_facts_for is dataset.input_facts_for


def test_the_card_vocabulary_is_one_object_in_core():
    from typing import get_args

    from structbench.benchmarks.card import Discretisation
    from structbench.core.evidence import CardDiscretisation

    assert Discretisation is CardDiscretisation
    assert set(get_args(CardDiscretisation)) == {"SPH", "FEM", "coupled"}
```

- [ ] **Step 2: Run them to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/test_datagen_import_boundary.py tests/verification/test_dataset_helpers.py 2>&1 | tail -3`
Expected: the boundary test fails naming `datagen/verify.py: ['cli']`; the helper test fails on `ImportError`.

- [ ] **Step 3: Move the code**

Create `verification/dataset.py` with `declared_from_toml`, `input_facts_for`, `measure_cases`, `_unreadable`, `_sha256`, `_normalise_solver`, `_INPUT_READERS`, `_UNIT_SYSTEM`, `_DECLARATION_KEYS`, `_DISCRETISATIONS = frozenset(get_args(CardDiscretisation))` — moved verbatim from `cli/datacheck.py` with their docstrings; module docstring: "Measuring a set of canonical cases against a declaration (ADR-0066): what a benchmark's spec or a dataset's `[declaration]` says, applied to every case file, with the solver input read by the reader for its solver." In `cli/datacheck.py` replace the moved code by `from ..verification.dataset import declared_from_toml, input_facts_for, measure_cases` (and `_INPUT_READERS` if a test imports it from the CLI: `tests/cli/test_datacheck_solver_gate.py:149` does — re-export it too). `datagen/verify.py`: `from structbench.verification.dataset import declared_from_toml, measure_cases`. `core/evidence.py`: add `CardDiscretisation = Literal["SPH", "FEM", "coupled"]` beside the existing `Discretisation`, with a comment that the first is the card's vocabulary and the second the run trait, and add it to `__all__`. `benchmarks/card.py`: `from ..core.evidence import CardDiscretisation as Discretisation` replacing the local Literal.

ARCHITECTURE.md: delete the sentence "One exception stands today: … removes the edge." from the `datagen/` section, and add `dataset.py` to the `verification/` section's list ("`dataset.py`: `declared_from_toml`, `measure_cases` and `input_facts_for`, the dataset-level entry the CLI and `datagen verify` share"). CLAUDE.md: delete "with one stated exception: `verify` takes `declared_from_toml` … until part two moves them into `verification`" and say "(`datagen` depends on `core`, `verification` and `validation` only, held by an import-boundary test since part two)".

- [ ] **Step 4: Run the tests**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/test_datagen_import_boundary.py tests/test_verification_import_boundary.py tests/verification tests/cli tests/datagen/test_verify.py 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 5: Gates and commit**

```bash
git add src/structbench/verification/dataset.py src/structbench/core/evidence.py src/structbench/benchmarks/card.py src/structbench/cli/datacheck.py src/structbench/datagen/verify.py docs/ARCHITECTURE.md CLAUDE.md tests/test_datagen_import_boundary.py tests/verification/test_dataset_helpers.py
git commit -m "refactor(verification): declared_from_toml, measure_cases and input_facts_for live in verification.dataset; the card vocabulary in core; datagen no longer imports cli"
```

---

### Task 4: `problem.py` may import its siblings; `[levels].symmetry`

**Files:**
- Modify: `src/structbench/datagen/definition.py` (`load_problem` puts the dataset directory first on `sys.path` while the module executes, and removes it after; `Levels` gains `symmetry: str = "axisymmetric"`, parsed from `[levels].symmetry`, one of `axisymmetric | planar`), `src/structbench/datagen/examples/abaqus_conformance/dataset.toml` (`symmetry = "axisymmetric"` with a comment), `docs/DATA_GENERATION.md` (one line in the table description)
- Test: `tests/datagen/test_definition.py`

**Interfaces:**
- Produces: `Definition.levels.symmetry`; a `problem.py` that does `from helpers import f` for a sibling `helpers.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/datagen/test_definition.py`:

```python
def test_a_problem_may_import_a_sibling_module(definition_dir):
    (definition_dir / "helpers.py").write_bytes(b"SCALE = 3.0\n")
    problem = MINIMAL_PROBLEM.replace(
        "from structbench.datagen.abaqus import deck",
        "from structbench.datagen.abaqus import deck\nfrom helpers import SCALE",
    ).replace('return {"length":', 'return {"length": SCALE * 0 +')
    (definition_dir / "problem.py").write_bytes(problem.encode())
    module = definition.load_problem(definition_dir)
    assert module.SCALE == 3.0
    import sys

    assert str(definition_dir) not in sys.path  # the entry is removed afterwards


def test_levels_symmetry_defaults_to_axisymmetric_and_refuses_others(tmp_path):
    d = definition.load_definition(write_definition(tmp_path / "a"))
    assert d.levels.symmetry == "axisymmetric"
    planar = MINIMAL_TOML.replace('refine_key = "refine"', 'refine_key = "refine"\nsymmetry = "planar"')
    assert definition.load_definition(write_definition(tmp_path / "b", toml=planar)).levels.symmetry == "planar"
    odd = MINIMAL_TOML.replace('refine_key = "refine"', 'refine_key = "refine"\nsymmetry = "spherical"')
    with pytest.raises(definition.DefinitionError, match="levels.symmetry"):
        definition.load_definition(write_definition(tmp_path / "c", toml=odd))
```

- [ ] **Step 2: Run them to see them fail**

Run: `<venv>/python -m pytest -q -p no:cacheprovider tests/datagen/test_definition.py 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'helpers'` (wrapped as `DefinitionError`) and `AttributeError: symmetry`.

- [ ] **Step 3: Implement**

In `load_problem`, around `exec_module`: `sys.path.insert(0, str(dataset_dir))` before, and in the `finally` remove the first occurrence of that entry. In `Levels`: `symmetry: str = "axisymmetric"`; in `load_definition`: `symmetry = lv.get("symmetry", "axisymmetric")`; refuse anything but `axisymmetric` / `planar` with `DefinitionError("levels.symmetry: ...")`. The example's `[levels]` gets `symmetry = "axisymmetric"  # the volume weights converge uses: axisymmetric (r dA) or planar (dA)`.

- [ ] **Step 4: Run the tests, gates, commit**

```bash
git add src/structbench/datagen/definition.py src/structbench/datagen/examples/abaqus_conformance/dataset.toml docs/DATA_GENERATION.md tests/datagen/test_definition.py
git commit -m "feat(datagen): a problem.py may import its siblings; [levels].symmetry names the volume weights"
```

---

### Task 5: `structbench-datagen converge` — pairing, the record, the report

**Files:**
- Create: `src/structbench/datagen/converge.py`
- Modify: `src/structbench/datagen/cli.py` (stage `converge`), `docs/DATA_GENERATION.md` (the stage list)
- Test: `tests/datagen/test_converge.py`, `tests/datagen/test_cli.py` (STAGES gains `converge`)

**Interfaces:**
- Consumes: `definition.{load_definition, load_problem, Definition}`, `verification.convergence`, `core.io.read_case`.
- Produces:
  - `RECORD_FORMAT = "convergence-record/1"`.
  - `@dataclass(frozen=True) LevelRun(case_id: str, split: str, level: str, root_index: int, production: bool, canonical: Path | None)`.
  - `pair_levels(defn, roots: Sequence[Path]) -> tuple[dict[str, dict[str, LevelRun]], list[str]]` — production case id → level label → the chosen run, plus notes (duplicates resolved, runs without the refine key skipped). Pairing key: the run's params without the refine key and without `[limits]` keys, plus the variant, `json.dumps(sort_keys=True)`. A production case is a run of a non-probe split. For one level, the production run wins, then root order, then case id.
  - `converge(defn, problem, roots, *, out_note=None) -> dict[str, Any]` (the record: `format`, `dataset`, `refine_key`, `symmetry`, `levels` (labels with counts), `cases` (per production case: `levels` {label: case_id}, `qoi` {name: {label: value}}, `extrapolation` {name: asdict(Extrapolation) | None with `ratio` and `production_level`}, `fields` {field: {label: error vs finest}}, `notes`), `summary` (per QoI: status counts and the spread of `error_vs_extrapolated` at the production level; per field and level: min/median/max), `notes`, `structbench_version`).
  - `to_json(record) -> bytes`, `render_markdown(record) -> str`, `main(argv) -> int`: `structbench-datagen converge --dataset <dir> --sweep <runs>/<name> [--root <dir> ...] [--out <dir>]` writes `<out>/convergence.json` and `convergence.md` (`--out` defaults to `<sweep>/converge`); exit 0, or 2 on a refused definition, or 1 when no case has two levels.

- [ ] **Step 1: Write the failing tests**

`tests/datagen/test_converge.py` builds a toy sweep from `MINIMAL_TOML` (`levels.pilot = ["1", "2", "4"]`; a non-probe `train` split; a probe split `conv` with `categorical = { refine = ["1", "4"] }` at the train points) with hand-written `provenance.json` files and canonical cases written by `write_case` from the `MINIMAL_PROBLEM`'s own mesh (`problem.mesh({**params, "refine": k})`, k × k quads, which nest), with fields sampled from a smooth function so that the coarse solutions restrict exactly (zero field error) or with a known 10 % scaling; QoI `length` from the problem's `qoi`. Tests:

```python
def test_levels_are_paired_by_parameters_and_the_production_run_supplies_its_level(...)
def test_a_duplicate_run_at_one_level_is_resolved_production_first_then_root_order_and_named(...)   # Review Focus 1
def test_labels_without_a_constant_ratio_get_no_extrapolation_but_field_errors(...)                  # Review Focus 2
def test_a_field_stored_on_one_level_only_is_skipped_with_a_note(...)                                # Review Focus 3
def test_a_case_whose_levels_do_not_nest_is_reported_and_the_others_proceed(...)                     # Review Focus 4
def test_the_record_is_byte_identical_and_names_no_path(...)                                         # Review Focus 5
def test_runs_without_the_refine_key_are_skipped_and_noted(...)
def test_second_order_toy_data_extrapolate_to_the_limit_at_the_production_level(...)                 # length = L + c h^2
def test_the_command_writes_json_and_markdown_and_exits_one_without_pairs(...)
```

(Each as a full test in the file: the fixture helper writes `<root>/<case>/provenance.json` with `params`, `split`, `variant` and `<root>/canonical/<case>.h5`.)

- [ ] **Step 2: Run them to see them fail**

Expected: `ImportError: cannot import name 'converge'`.

- [ ] **Step 3: Write `converge.py`, the stage, the docs line**

Implement per the Interfaces above; `render_markdown` writes: the title (`# Convergence of <dataset> across mesh levels`), the levels and how many cases hold each, one table per QoI (case, value per level, status, order, extrapolated, error at the production level, GCI), one table per field (case, error per coarser level against the finest, in percent with one decimal), the summary tables, the notes, the closing line with the format and version. Ratios in four significant figures, percentages one decimal. Add `"converge": converge.main` to `STAGES` after `verify`, and the stage to the docstrings and `docs/DATA_GENERATION.md`.

- [ ] **Step 4: Run the tests, gates, commit**

```bash
git add src/structbench/datagen/converge.py src/structbench/datagen/cli.py docs/DATA_GENERATION.md tests/datagen/test_converge.py tests/datagen/test_cli.py
git commit -m "feat(datagen): the converge stage -- runs paired across roots by their parameters, QoIs and fields across levels, the convergence-record/1 JSON and Markdown"
```

---

### Task 6: Documentation

**Files:**
- Modify: `docs/ARCHITECTURE.md` (`verification/` gains `convergence.py` and the metric's home; the layering sentence unchanged), `docs/DATA_GENERATION.md` (a short "Convergence" paragraph: what `converge` needs from a definition — `[levels]`, `symmetry`, `qoi()` — and what it writes), `docs/plans/2026-09-27-abaqus-datagen-platform-design.md` (stage table: `converge` "built (plan 2a)"), `decisions/0071-datagen-platform.md` (a dated note: part 2a delivered clause 4 and the layering; 2b owes preflight, the gate, run hardening and follow), `CLAUDE.md` (the snapshot's datagen paragraph: converge built, the engine's home, the layering held by tests; owed list shortened), `src/structbench/core/io/abaqus.py` (line 3: the stale `data_generation/abaqus/odb_export.py` becomes `structbench/datagen/abaqus/odb_export.py`)
- Test: `tests/datagen/test_docs.py` (add: ARCHITECTURE names `convergence.py` under `verification/` and the stale path is gone from `core/io/abaqus.py`)

- [ ] Steps: failing test → docs → tests → gates → commit `docs: converge and the convergence engine in ARCHITECTURE, the guide, the design, ADR-0071 (note) and the snapshot; a stale path in the adapter`.

---

### Task 7: The private dataset retires its copies against a numerical regression

**Files (the private dataset repository; `<dataset dir>` its dataset folder, `<runs>` its run folder):**
- Create: `<dataset dir>/test_converge_record.py`
- Modify: `<dataset dir>/problem.py` (import the QoI measures from the sibling `qoi.py`; delete the copy), `<dataset dir>/qoi.py` (docstring: no longer copied), `<dataset dir>/NOTES.md` (dated entry), `<dataset dir>/DATA_CARD.md` ("Discretization uncertainty": regenerated numbers and the record's path; correct the level any earlier statement was made at), the private lessons file's recipe (`converge`)
- Delete: `<dataset dir>/convergence.py`, `<dataset dir>/test_convergence.py` (history keeps them; the public engine's tests cover the algorithms)
- Test: the private suite

- [ ] **Step 1: Write the failing regression test**

`test_converge_record.py`: skip unless the runs are present; run `structbench-datagen converge --dataset <dataset dir> --sweep <runs>/<sweep> --out <tmp>`; load the record and the dataset's earlier convergence CSV (the one produced by the private `convergence.py` on the final setup); for every case and QoI, the values per level agree to 1e-9 relative, `order`, `extrapolated`, `gci_fine` and `coarsest_vs_finest` to 1e-8, and every field error to 1e-8; the record's production level is the dataset's `[levels].production`. Also: `problem.py` no longer defines the measures (`assert problem.taylor_qoi is qoi.taylor_qoi` after the import change).

- [ ] **Step 2: Run it to see it fail** — `converge` runs, but `problem.taylor_qoi is qoi.taylor_qoi` fails (the copy).

- [ ] **Step 3: Retire the copies** — `problem.py` imports from `qoi.py` (Task 4 made that possible); `git rm` the private engine and its tests; regenerate the record into `<dataset dir>/convergence/` (committed: `convergence.json`, `convergence.md`).

- [ ] **Step 4: Check the numbers** — compare the record with the earlier CSV as the test does; **note the semantic difference**: the earlier CSV's `production_error` was the *coarsest* level's error against the extrapolated value (it was written when production ran at the coarsest level), whereas the record reports the error at the dataset's declared production level. Read the data card's "Discretization uncertainty" and correct any sentence that quoted the coarsest level's error as the production error; record the correction in NOTES.

- [ ] **Step 5: Private suite, commit there; final gates in StructBench.**

---

## Self-review

- **Spec coverage:** clause 4 of ADR-0071 (the engine in `verification.convergence`, datasets supply only `qoi()` and `[levels]`) → Tasks 2, 4, 5; the layering exception → Task 3; the private copies retired → Task 7; the headline metric's home → Task 1 (required by the layering: `verification` may not import `eval`); docs → Task 6. The stress breakdown named in the design's `converge` paragraph (contact-phase / post-release / time-shift / block-averaged) is **not** in this plan: it was a diagnostic of one study, and a generic form needs a definition of "release" the contract does not have; it is listed in the ADR-0071 note as open, and the record's per-field errors are the platform's number.
- **Placeholders:** Task 5's tests are named with their intent rather than written out; the executor writes each from the fixture helper described. Task 7 uses placeholders by the global constraint.
- **Type consistency:** `Extrapolation` fields used by `converge`'s record; `NestingError` caught per case in `converge`; `Levels.symmetry` read by `converge` and passed to `field_errors(axisymmetric=...)`.
- **Review Focus:** 1 → Task 5; 2 → Task 5; 3 → Tasks 2 and 5; 4 → Task 5; 5 → Task 5.
