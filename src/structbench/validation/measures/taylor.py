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

#: The heights at which the source reports the lateral radius, as fractions of L_f.
FRACTIONS = (0.2, 0.25, 1 / 3, 0.5, 2 / 3)
#: The length band is taken over the last quarter of the stored frames.
SETTLED_FRACTION = 0.75


class OutlineError(ValueError):
    """The case has no single closed boundary an outline can be read from."""


def outline(case: Case, frame: int = -1) -> NDArray[np.float64]:
    """The closed boundary of the quad mesh at ``frame``: (K, 2), z from the face.

    Boundary edges are those exactly one element uses; they are chained into
    one loop, which is returned closed (its first point repeated at the end).
    Nodes no element uses are ignored. The loop is deformed by the frame's
    displacement and shifted so that its lowest z is 0.
    """
    if case.response is None:
        raise OutlineError("the case stores no response")
    coords = np.asarray(case.nodes.coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise OutlineError(
            "expected an axisymmetric (r, z) case with 2-column coordinates"
        )
    conn = np.asarray(case.elements["solid"].connectivity, dtype=np.int64)
    if conn.ndim != 2 or conn.shape[1] != 4:
        raise OutlineError(
            f"expected 4-node quads, got {conn.shape[-1]} nodes per element"
        )
    edges: Counter[tuple[int, int]] = Counter()
    for quad in conn:
        for a, b in zip(quad, np.roll(quad, -1), strict=True):
            edges[(int(min(a, b)), int(max(a, b)))] += 1
    boundary = [edge for edge, n in edges.items() if n == 1]
    neighbours: defaultdict[int, list[int]] = defaultdict(list)
    for a, b in boundary:
        neighbours[a].append(b)
        neighbours[b].append(a)
    if not boundary or any(len(v) != 2 for v in neighbours.values()):
        raise OutlineError("the mesh boundary is not one closed loop")
    start = min(neighbours)
    loop, previous, node = [start], -1, start
    while True:
        a, b = neighbours[node]
        previous, node = node, (a if a != previous else b)
        if node == start:
            break
        loop.append(node)
        if len(loop) > len(boundary):
            raise OutlineError("the mesh boundary is not one closed loop")
    if len(loop) != len(boundary):
        raise OutlineError("the mesh boundary is not one closed loop (it is several)")
    u = np.asarray(case.response.node["displacement"][frame], dtype=np.float64)
    rz = (coords + u)[loop + [start]]
    rz[:, 1] -= rz[:, 1].min()
    return rz


def final_length(rz: ArrayLike) -> float:
    """L_f: the outline's extent along the axis (its lowest z is the face)."""
    pts = np.asarray(rz, dtype=np.float64)
    return float(pts[:, 1].max() - pts[:, 1].min())


def largest_radius(rz: ArrayLike) -> float:
    """R_f: the largest radius anywhere on the outline."""
    return float(np.asarray(rz, dtype=np.float64)[:, 0].max())


def lateral_radii(
    rz: ArrayLike, length: float, fractions: tuple[float, ...]
) -> list[float]:
    """W_f: the largest r where the outline crosses the height f * length, per fraction.

    A vertex on the height counts once (through the segment that ends on it,
    whether the outline crosses or only touches the height there);
    a segment lying along the height contributes both ends; a height the
    outline never reaches gives NaN.
    """
    pts = np.asarray(rz, dtype=np.float64)
    base = pts[:, 1].min()
    out = []
    for f in fractions:
        h = base + f * length
        hits: list[float] = []
        for (r1, z1), (r2, z2) in zip(pts[:-1], pts[1:], strict=True):
            if z1 == z2:
                if z1 == h:
                    hits += [float(r1), float(r2)]
                continue
            if z2 == h:  # a vertex on the height: the segment arriving at it
                hits.append(float(r2))
            elif min(z1, z2) < h < max(z1, z2):
                hits.append(float(r1 + (h - z1) * (r2 - r1) / (z2 - z1)))
        out.append(max(hits) if hits else float("nan"))
    return out


@dataclass(frozen=True)
class TaylorOutcome:
    """The measures of one case at its last frame, in the case's own length unit."""

    length: float
    length_band: float  # half the peak-to-peak of the length over the settled frames
    largest_radius: float
    lateral_radii: tuple[float, ...]


def taylor_outcome(
    case: Case, fractions: tuple[float, ...] = FRACTIONS
) -> TaylorOutcome:
    """L_f, R_f and W_f at the last frame, with the length's band over the tail."""
    if case.response is None:
        raise OutlineError("the case stores no response")
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
