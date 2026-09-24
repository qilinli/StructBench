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
    "ALLAE",
    "ALLCD",
    "ALLFD",
    "ALLIE",
    "ALLKE",
    "ALLPD",
    "ALLSE",
    "ALLVD",
    "ALLWK",
    "ETOTAL",
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
    lines = [f"START, {num(x0)}, {num(y0)}"]
    lines += [f"LINE, {num(x)}, {num(y)}" for x, y in rest]
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
    body = "".join(f"{int(e)}, {num(v)}\n" for e, v in zip(labels, vals, strict=True))
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
