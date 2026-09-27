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


class Restriction:
    """Average fine element values onto the coarse elements that contain them.

    The meshes must be structured from axis-aligned rectangular quads (the
    owner of a fine element is the coarse box holding its centroid); any other
    element is refused by name. Weighted by ``element_weights``.
    """

    def __init__(
        self,
        coarse_xy: ArrayLike,
        coarse_conn: ArrayLike,
        fine_xy: ArrayLike,
        fine_conn: ArrayLike,
        *,
        axisymmetric: bool = True,
    ) -> None:
        cxy, cconn = np.asarray(coarse_xy, float), np.asarray(coarse_conn, np.int64)
        fxy, fconn = np.asarray(fine_xy, float), np.asarray(fine_conn, np.int64)
        for name, xy, conn in (("coarse", cxy, cconn), ("fine", fxy, fconn)):
            bad = _not_rectangles(xy, conn)
            if bad:
                raise NestingError(
                    "restriction supports axis-aligned rectangular elements only; "
                    f"{bad} {name} elements are not"
                )
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
) -> dict[str, float]:
    """Pooled relative L2 of the coarse solution against the fine one, per field.

    Node fields are compared at the coarse nodes, which the fine mesh contains;
    element fields on the coarse elements, the fine children restricted with
    volume weights. Fields default to those both cases store; one stored on a
    single side is left out. Raises ``NestingError`` when the meshes do not
    nest (or are not rectangular quads) and ``ValueError`` when the time grids
    differ.
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
    ce = coarse.response.element.get("solid", {})
    fe = fine.response.element.get("solid", {})
    names_n = [n for n in (node_fields or cn) if n in cn and n in fn]
    names_e = [n for n in (element_fields or ce) if n in ce and n in fe]
    errors = {}
    for n in names_n:
        a, b = np.asarray(cn[n])[:, used], np.asarray(fn[n])[:, index]
        errors[n] = float(relative_l2_pooled(a, b))
    if names_e:
        restrict = Restriction(cxy, cconn, fxy, fconn, axisymmetric=axisymmetric)
        for n in names_e:
            fine_on_coarse = restrict.apply(fe[n])
            errors[n] = float(relative_l2_pooled(np.asarray(ce[n]), fine_on_coarse))
    return errors
