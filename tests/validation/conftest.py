"""Synthetic axisymmetric rods whose deformed outlines are known exactly."""

import numpy as np
import pytest

from structbench.core import Case, ElementBlock, Material, Metadata, Nodes, Response

pytest.importorskip("h5py")


def _grid(nr, nz, radius, length):
    rs, zs = np.linspace(0.0, radius, nr + 1), np.linspace(0.0, length, nz + 1)
    gr, gz = np.meshgrid(rs, zs)
    coords = np.column_stack([gr.ravel(), gz.ravel()])
    i, j = (a.ravel() for a in np.meshgrid(np.arange(nr), np.arange(nz)))
    row = nr + 1
    conn = np.column_stack(
        [i + j * row, i + 1 + j * row, i + 1 + (j + 1) * row, i + (j + 1) * row]
    )
    return coords, conn


def _case(coords, conn, u):
    n, e = len(coords), len(conn)
    return Case(
        metadata=Metadata(case_id="rod", dimension=2, source_units="t-mm-s"),
        nodes=Nodes(coords=coords, node_id=np.arange(1, n + 1)),
        elements={
            "solid": ElementBlock(
                connectivity=conn,
                element_id=np.arange(1, e + 1),
                part_id=np.ones(e, np.int64),
            )
        },
        materials=[Material(material_id=1, source_model="toy", source_params={})],
        response=Response(
            time=np.linspace(0.0, 1.0, len(u)),
            node={"displacement": u.astype(np.float32)},
            element={
                "solid": {"effective_plastic_strain": np.zeros((len(u), e), np.float32)}
            },
            globals_={},
        ),
    )


def rod_case(
    nr=4, nz=10, radius=1.0, length=5.0, *, squash=0.8, flare=0.5, wobble=0.0, frames=5
):
    """z' = squash z, r' = r (1 + flare (1 - z/L)); frame k adds (-1)^k wobble z/L.

    Known outline at any frame with wobble 0: L_f = squash L, R_f = (1 + flare) R,
    W_f(f) = R (1 + flare (1 - f)). A node no element uses sits at (9, 9).
    """
    coords, conn = _grid(nr, nz, radius, length)
    coords = np.vstack([coords, [[9.0, 9.0]]])
    u = np.zeros((frames, len(coords), 2))
    z = coords[:, 1]
    for k in range(frames):
        u[k, :, 0] = coords[:, 0] * flare * (1 - z / length)
        u[k, :, 1] = (squash - 1) * z + wobble * (-1) ** k * z / length
    return _case(coords, conn, u)


def two_body_case():
    """Two disjoint rods in one mesh: the boundary is two loops."""
    a_coords, a_conn = _grid(2, 3, 1.0, 3.0)
    b_coords, b_conn = _grid(2, 3, 1.0, 3.0)
    coords = np.vstack([a_coords, b_coords + [5.0, 0.0]])
    conn = np.vstack([a_conn, b_conn + len(a_coords)])
    return _case(coords, conn, np.zeros((3, len(coords), 2)))
