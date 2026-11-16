"""Solution verification across nested mesh levels (ADR-0071 clause 4).

Richardson extrapolation with the observed order and its statuses, nested-node
matching, restriction of element fields onto the coarser mesh with
axisymmetric or planar volume weights, and the pooled relative L2 of each
field (ADR-0055) of a coarser solution against the finest. Everything works
on ``Case`` objects; pairing runs and reading files is the pipeline's job
(``structbench.datagen.converge``).
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
STATUSES = ("monotone", "oscillatory", "diverging", "flat")


class NestingError(ValueError):
    """Two meshes of which one is not a refinement of the other, or elements the
    restriction cannot own."""


@dataclass(frozen=True)
class Extrapolation:
    """Richardson extrapolation of one quantity over three levels.

    ``status`` says whether an observed order exists: ``monotone`` when the
    differences between levels have one sign and shrink; ``oscillatory``,
    ``diverging`` or ``flat`` otherwise, and then the other fields are None
    except ``coarsest_vs_finest``, which is reported whatever the status.
    A relative quantity whose denominator is zero is None rather than a crash.
    """

    status: str
    order: float | None
    extrapolated: float | None
    gci_fine: float | None
    error_vs_extrapolated: dict[str, float] | None  # by level label
    coarsest_vs_finest: float | None


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else abs(numerator) / abs(denominator)


def richardson(
    values: dict[str, float], *, ratio: float, safety: float = SAFETY_DEFAULT
) -> Extrapolation:
    """Observed order and extrapolated value from three levels, coarse to fine.

    ``values`` maps the three level labels, coarsest first, to the quantity;
    ``ratio`` is the refinement ratio between consecutive levels. Non-finite
    values are refused by name.
    """
    if len(values) != 3:
        raise ValueError(
            f"Richardson extrapolation needs exactly three levels, got {len(values)}"
        )
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError(
            f"Richardson extrapolation needs finite values, got {dict(values)}"
        )
    (_lc, fc), (_lm, fm), (_lf, ff) = values.items()
    e32, e21 = fc - fm, fm - ff
    coarsest_vs_finest = _ratio(fc - ff, ff)
    if e21 == 0.0:
        status = "flat"
    elif e32 == 0.0:
        status = "diverging"  # no change at the first step, a change at the second
    elif e32 / e21 < 0.0:
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
    errors = (
        None
        if extrapolated == 0.0
        else {
            label: abs(v - extrapolated) / abs(extrapolated)
            for label, v in values.items()
        }
    )
    gci = None if ff == 0.0 else safety * abs(e21 / ff) / gain
    return Extrapolation(status, p, extrapolated, gci, errors, coarsest_vs_finest)


def match_nodes(
    coarse_xy: ArrayLike, fine_xy: ArrayLike, *, tol: float = 1e-9
) -> NDArray[np.int64]:
    """For each coarse node, the index of the fine node at the same place."""
    from scipy.spatial import cKDTree  # the datagen extra; only this function needs it

    coarse, fine = np.asarray(coarse_xy, float), np.asarray(fine_xy, float)
    scale = float(np.ptp(fine, axis=0).max()) or 1.0
    distance, index = cKDTree(fine).query(coarse)
    if distance.max() > tol * scale:
        raise NestingError(
            f"the meshes do not nest: a coarse node is {distance.max():.3g} from "
            "the nearest fine node"
        )
    return np.asarray(index, dtype=np.int64)


def element_weights(
    coords: ArrayLike, conn: ArrayLike, *, axisymmetric: bool
) -> NDArray[np.float64]:
    """Per element: the integral of r dA (axisymmetric) or of dA (planar) over the quad.

    Column 0 of ``coords`` is r. The shoelace sums make the sign irrelevant.
    """
    xy = np.asarray(coords, float)[np.asarray(conn, np.int64)]  # (E, 4, 2)
    x, y = xy[..., 0], xy[..., 1]
    xn, yn = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    cross = x * yn - xn * y
    if axisymmetric:
        return np.abs((cross * (x + xn)).sum(axis=1)) / 6.0
    return np.abs(cross.sum(axis=1)) / 2.0


def _not_rectangles(xy: NDArray[np.float64], conn: NDArray[np.int64]) -> int:
    """How many quads are not axis-aligned rectangles (two x values, two y values)."""
    quads = xy[conn]  # (E, 4, 2)
    scale = float(np.ptp(xy, axis=0).max()) or 1.0
    xs, ys = np.sort(quads[..., 0], axis=1), np.sort(quads[..., 1], axis=1)
    tol = 1e-9 * scale
    ok_x = (np.abs(xs[:, 1] - xs[:, 0]) <= tol) & (np.abs(xs[:, 3] - xs[:, 2]) <= tol)
    ok_y = (np.abs(ys[:, 1] - ys[:, 0]) <= tol) & (np.abs(ys[:, 3] - ys[:, 2]) <= tol)
    return int((~(ok_x & ok_y)).sum())


_STEP = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])  # across local edge 0..3
_CORNER = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # of local node 0..3


def _grid_indices(conn: NDArray[np.int64], n_nodes: int) -> NDArray[np.int64]:
    """Each node's ``(i, j)`` on the structured grid a quad mesh forms.

    Read from the connectivity alone, so the quads may have any shape and the
    elements any order or starting corner. The frame is one of the grid's four
    rotations, fixed by the first element. A mesh that is not one structured
    grid of consistently oriented quads is refused.
    """
    n = len(conn)
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for e in range(n):
        for k in range(4):
            a, b = int(conn[e, k]), int(conn[e, (k + 1) % 4])
            edges.setdefault((min(a, b), max(a, b)), []).append((e, k))
    if any(len(v) > 2 for v in edges.values()):
        raise NestingError("not a structured quad grid: an edge joins three elements")
    cell = np.zeros((n, 2), dtype=np.int64)
    turn = np.full(n, -1, dtype=np.int64)  # local edge k faces direction k + turn
    turn[0] = 0
    queue = [0]
    while queue:
        e = queue.pop()
        for k in range(4):
            a, b = int(conn[e, k]), int(conn[e, (k + 1) % 4])
            for other, m in edges[(min(a, b), max(a, b))]:
                if other == e or turn[other] >= 0:
                    continue
                facing = (turn[e] + k) % 4
                turn[other] = (facing + 2 - m) % 4
                cell[other] = cell[e] + _STEP[facing]
                queue.append(other)
    if (turn < 0).any():
        raise NestingError("not a structured quad grid: the elements are not connected")
    node_ij = np.full((n_nodes, 2), np.iinfo(np.int64).min, dtype=np.int64)
    for k in range(4):
        ij = cell + _CORNER[(k + turn) % 4]
        seen = node_ij[conn[:, k]]
        if np.any((seen != ij) & (seen != np.iinfo(np.int64).min)):
            raise NestingError("not a structured quad grid: a node has two places")
        node_ij[conn[:, k]] = ij
    used = np.unique(conn)
    node_ij[used] -= node_ij[used].min(axis=0)
    nx, ny = (node_ij[used].max(axis=0)).tolist()
    if nx * ny != n or len(used) != (nx + 1) * (ny + 1):
        raise NestingError("not a structured quad grid: it has holes or branches")
    return node_ij


def _parents_by_index(
    cconn: NDArray[np.int64],
    fconn: NDArray[np.int64],
    fine_of_coarse: NDArray[np.int64],
    n_coarse_nodes: int,
    n_fine_nodes: int,
) -> NDArray[np.int64]:
    """For each fine element, the coarse element it refines, by grid index.

    ``fine_of_coarse`` maps every coarse node to the fine node at its place.
    The two grids' frames may differ by a rotation; the matched nodes fix it,
    and with it the refinement ratio along each direction.
    """
    cij, fij = _grid_indices(cconn, n_coarse_nodes), _grid_indices(fconn, n_fine_nodes)
    c_used, f_used = np.unique(cconn), np.unique(fconn)
    c_at = cij[c_used]
    for quarter_turns in range(4):
        rotated = fij.copy()
        for _ in range(quarter_turns):
            rotated = np.column_stack([-rotated[:, 1], rotated[:, 0]])
        rotated[f_used] -= rotated[f_used].min(axis=0)
        f_at = rotated[fine_of_coarse[c_used]]
        c_span, f_span = c_at.max(axis=0), rotated[f_used].max(axis=0)
        if np.any(f_span % c_span):
            continue
        ratio = f_span // c_span
        if np.array_equal(f_at, c_at * ratio):
            break
    else:
        raise NestingError(
            "the meshes do not nest by index: the coarse nodes are not every "
            "r-th node of the fine grid"
        )
    coarse_cell = cij[cconn].min(axis=1)  # each element's lower-left grid node
    fine_cell = rotated[fconn].min(axis=1) // ratio
    owner = {(int(i), int(j)): e for e, (i, j) in enumerate(coarse_cell)}
    return np.array([owner[(int(i), int(j))] for i, j in fine_cell], dtype=np.int64)


class Restriction:
    """Average fine element values onto the coarse elements that contain them.

    Meshes of axis-aligned rectangular quads are matched by place: the owner
    of a fine element is the coarse box holding its centroid. Any other
    structured quad mesh is matched by index when ``fine_of_coarse`` is given
    (the fine node at each coarse node's place): each coarse element owns the
    ``r x r`` fine elements of its grid cell, whatever the quads' shape.
    Weighted by ``element_weights`` either way.
    """

    def __init__(
        self,
        coarse_xy: ArrayLike,
        coarse_conn: ArrayLike,
        fine_xy: ArrayLike,
        fine_conn: ArrayLike,
        *,
        axisymmetric: bool = True,
        fine_of_coarse: ArrayLike | None = None,
    ) -> None:
        cxy, cconn = np.asarray(coarse_xy, float), np.asarray(coarse_conn, np.int64)
        fxy, fconn = np.asarray(fine_xy, float), np.asarray(fine_conn, np.int64)
        for name, xy, conn in (("coarse", cxy, cconn), ("fine", fxy, fconn)):
            bad = _not_rectangles(xy, conn)
            if bad and fine_of_coarse is None:
                raise NestingError(
                    "restriction supports axis-aligned rectangular elements only; "
                    f"{bad} {name} elements are not"
                )
            if bad:
                parent = _parents_by_index(
                    cconn,
                    fconn,
                    np.asarray(fine_of_coarse, np.int64),
                    len(cxy),
                    len(fxy),
                )
                break
        else:
            parent = self._parents_by_place(cxy, cconn, fxy, fconn)
        counts = np.bincount(parent, minlength=len(cconn))
        if counts.min() == 0:
            raise NestingError(
                "a coarse element has no fine children: the meshes do not nest"
            )
        weight = element_weights(fxy, fconn, axisymmetric=axisymmetric)
        order = np.argsort(parent, kind="stable")
        total = np.bincount(parent, weights=weight, minlength=len(cconn))
        self._order = order
        self._starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self._weight = (weight / total[parent])[order]
        self.n_coarse = len(cconn)

    @staticmethod
    def _parents_by_place(
        cxy: NDArray[np.float64],
        cconn: NDArray[np.int64],
        fxy: NDArray[np.float64],
        fconn: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Rectangles: the coarse box holding each fine element's centroid."""
        boxes = cxy[cconn]
        lo, hi = boxes.min(axis=1), boxes.max(axis=1)
        centroid = fxy[fconn].mean(axis=1)
        r_lines, z_lines = np.unique(lo[:, 0]), np.unique(lo[:, 1])
        col = np.searchsorted(r_lines, centroid[:, 0], side="right") - 1
        row = np.searchsorted(z_lines, centroid[:, 1], side="right") - 1
        corners = zip(
            np.searchsorted(r_lines, lo[:, 0]),
            np.searchsorted(z_lines, lo[:, 1]),
            strict=True,
        )
        owner = {(int(i), int(j)): e for e, (i, j) in enumerate(corners)}
        parent = np.array(
            [owner.get((int(i), int(j)), -1) for i, j in zip(col, row, strict=True)]
        )
        safe = parent.clip(0)
        inside = (parent >= 0) & np.all(
            (centroid >= lo[safe]) & (centroid <= hi[safe]), axis=1
        )
        if not inside.all():
            raise NestingError(
                f"{int((~inside).sum())} fine elements lie in no coarse element"
            )
        return parent

    def apply(self, values: ArrayLike) -> NDArray[np.float64]:
        """``(T, n_fine, ...)`` -> ``(T, n_coarse, ...)``."""
        v = np.asarray(values, float)
        frames, rest = v.shape[0], v.shape[2:]
        flat = v.reshape(frames, v.shape[1], -1)[:, self._order, :]
        weighted = flat * self._weight[None, :, None]
        out = np.add.reduceat(weighted, self._starts, axis=1)
        return out.reshape(frames, self.n_coarse, *rest)


def _mesh(case: Case) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    coords = np.asarray(case.nodes.coords, float)
    conn = np.asarray(case.elements["solid"].connectivity, np.int64)
    return coords, conn


def field_errors(
    coarse: Case,
    fine: Case,
    *,
    node_fields: Sequence[str] | None = None,
    element_fields: Sequence[str] | None = None,
    axisymmetric: bool = True,
    notes: list[str] | None = None,
) -> dict[str, float]:
    """Pooled relative L2 of the coarse solution against the fine one, per field.

    Node fields are compared at the coarse nodes, which the fine mesh contains;
    element fields on the coarse elements, the fine children restricted with
    volume weights (by place for rectangles, by grid index otherwise). Fields
    default to those both cases store; one stored on a single side is left
    out. Raises ``NestingError`` when the nodes do not nest and ``ValueError``
    when the time grids differ. Element fields that cannot be restricted raise
    ``NestingError`` too, unless ``notes`` is given: the reason is then
    appended to it and the node fields are returned alone.
    """
    if coarse.response is None or fine.response is None:
        raise ValueError("both cases must store a response")
    tc, tf = np.asarray(coarse.response.time), np.asarray(fine.response.time)
    if tc.shape != tf.shape or not np.allclose(tc, tf, rtol=1e-9, atol=0.0):
        raise ValueError("the two runs are not on the same time grid")
    cxy, cconn = _mesh(coarse)
    fxy, fconn = _mesh(fine)
    # A rigid body's reference node is no mesh node, on either side: it may
    # even stand where a mesh node does, and must not be taken for it.
    used, fine_used = np.unique(cconn), np.unique(fconn)
    index = fine_used[match_nodes(cxy[used], fxy[fine_used])]
    cn, fn = coarse.response.node, fine.response.node
    ce = coarse.response.element.get("solid", {})
    fe = fine.response.element.get("solid", {})
    names_n = [n for n in (node_fields or cn) if n in cn and n in fn]
    names_e = [n for n in (element_fields or ce) if n in ce and n in fe]
    errors = {}
    for n in names_n:
        a, b = np.asarray(cn[n])[:, used], np.asarray(fn[n])[:, index]
        errors[n] = float(relative_l2_pooled(a, b))
    if names_e:
        fine_of_coarse = np.zeros(len(cxy), dtype=np.int64)
        fine_of_coarse[used] = index
        try:
            restrict = Restriction(
                cxy,
                cconn,
                fxy,
                fconn,
                axisymmetric=axisymmetric,
                fine_of_coarse=fine_of_coarse,
            )
        except NestingError as exc:
            if notes is None:
                raise
            notes.append(f"element fields not compared: {exc}")
            return errors
        for n in names_e:
            fine_on_coarse = restrict.apply(fe[n])
            errors[n] = float(relative_l2_pooled(np.asarray(ce[n]), fine_on_coarse))
    return errors
