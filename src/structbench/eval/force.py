"""Force read from the state: the internal-force operator and what follows.

A simulator that predicts positions and stress has predicted every force:
the nodal internal force is the stress integrated against the gradient of
the shape functions on the deformed mesh, the operation a finite-element
code performs each increment. This module holds that operator, one
implementation per element formulation a benchmark can declare
(:data:`~structbench.benchmarks.constitutive.ELEMENT_FORMULATIONS`), and
the quantities built on it: the reaction at a named node set, the force a
rigid plane applies through the nodes on it, the force through a section,
and the reductions of such a series to one number. Two reported properties
sit beside them: the residual of the balance on a quasi-static problem and
the penetration of a declared rigid plane.

Everything is numpy over a stored rollout. Units follow the working frame:
positions in mm and stress in MPa give force in N; an axisymmetric force is
that of the full revolution. The first coordinate of an axisymmetric case is
the radius, as in the stored stress layout ``(s_rr, s_zz, s_tt, s_rz)``; a
plane-strain case reads the same four slots as ``(s_xx, s_yy, s_zz, s_xy)``
and its force is per unit thickness (N/mm).

The elements are four-node quadrilaterals integrated at one point (the
centroid), the formulation of the reduced-integration elements the shipped
element-site data was produced with. Another element type is refused by
name; it gets its own implementation when a benchmark arrives with it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import QoiFn, QoiInputs, SeriesFn

__all__ = [
    "FORCE_FORMULATIONS",
    "SeriesFn",
    "body_force",
    "element_mean",
    "force_balance_residual",
    "internal_force",
    "lumped_mass",
    "plane_penetration",
    "reaction_force",
    "section_force",
    "series_impulse",
    "series_mean",
    "series_peak",
]

#: The element formulations the operator implements.
FORCE_FORMULATIONS: tuple[str, ...] = ("axisymmetric", "plane_strain")


class _PlaneLike(Protocol):
    """What this module reads of a declared rigid plane
    (:class:`~structbench.benchmarks.registry.RigidPlane`)."""

    @property
    def normal(self) -> tuple[float, ...]: ...
    @property
    def offset(self) -> float | None: ...
    @property
    def node_type(self) -> int | None: ...


def _check(formulation: str, cells: NDArray, dim: int) -> None:
    if formulation not in FORCE_FORMULATIONS:
        raise ValueError(
            f"internal force: formulation {formulation!r} is not implemented; "
            f"one of {FORCE_FORMULATIONS}"
        )
    if cells.ndim != 2 or cells.shape[1] != 4 or dim != 2:
        raise ValueError(
            "internal force: implemented for four-node quadrilaterals in two "
            f"dimensions; got cells of shape {cells.shape} in {dim} dimensions"
        )


def _element_forces(
    positions: NDArray[np.float64],
    stress: NDArray[np.float64],
    cells: NDArray[np.int64],
    formulation: str,
) -> NDArray[np.float64]:
    """Each element's four nodal forces, ``(..., E, 4, 2)``, unassembled."""
    x = positions[..., cells, :]  # (..., E, 4, 2)
    d31 = x[..., 2, :] - x[..., 0, :]
    d42 = x[..., 3, :] - x[..., 1, :]
    signed = 0.5 * (d31[..., 0] * d42[..., 1] - d42[..., 0] * d31[..., 1])
    area = np.abs(signed)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(area > 0.0, 0.5 / signed, 0.0)
    # centroid gradients of the four bilinear shape functions
    grad_x = np.stack([-d42[..., 1], d31[..., 1], d42[..., 1], -d31[..., 1]], axis=-1)
    grad_y = np.stack([d42[..., 0], -d31[..., 0], -d42[..., 0], d31[..., 0]], axis=-1)
    grad_x = grad_x * inv[..., None]
    grad_y = grad_y * inv[..., None]
    s11, s22, s33, s12 = (stress[..., i][..., None] for i in range(4))
    if formulation == "axisymmetric":
        radius = x[..., 0].mean(axis=-1)
        volume = 2.0 * np.pi * radius * area
        hoop = 0.25 * s33 * (2.0 * np.pi * area)[..., None]
    else:
        volume = area
        hoop = np.zeros_like(s33)
    fx = (grad_x * s11 + grad_y * s12) * volume[..., None] + hoop
    fy = (grad_x * s12 + grad_y * s22) * volume[..., None]
    return np.stack([fx, fy], axis=-1)


def _assemble(
    element_forces: NDArray[np.float64], cells: NDArray[np.int64], n_nodes: int
) -> NDArray[np.float64]:
    lead = element_forces.shape[:-3]
    flat = element_forces.reshape((-1, *element_forces.shape[-3:]))
    out = np.zeros((flat.shape[0], n_nodes, 2), dtype=np.float64)
    frames = np.arange(flat.shape[0])[:, None]
    for a in range(4):
        np.add.at(out, (frames, cells[None, :, a]), flat[:, :, a])
    return out.reshape((*lead, n_nodes, 2))


def internal_force(
    positions: ArrayLike,
    element_stress: ArrayLike,
    cells: ArrayLike,
    formulation: str,
) -> NDArray[np.float64]:
    """Nodal internal force of an element stress field on the deformed mesh.

    ``f_a = ∫ B_aᵀ σ dV`` with one integration point per element. In
    equilibrium it equals the external force at the node, so its sum over a
    supported node set is the reaction there, and it vanishes at a free node
    of a static problem. Summed over every node of a free body the in-plane
    components cancel (axisymmetric: the axial one; the radial one is held
    by the hoop stress).

    Parameters
    ----------
    positions : array_like, shape (..., N, 2)
        Current nodal coordinates (mm).
    element_stress : array_like, shape (..., E, 4)
        ``(s11, s22, s33, s12)`` per element (MPa), 33 out of plane.
    cells : array_like, shape (E, 4)
        Node indices of each quadrilateral, in order around it.
    formulation : str
        ``"axisymmetric"`` or ``"plane_strain"``.

    Returns
    -------
    numpy.ndarray
        Shape ``(..., N, 2)``: N for a full revolution, N/mm in plane strain.
    """
    x = np.asarray(positions, dtype=np.float64)
    s = np.asarray(element_stress, dtype=np.float64)
    c = np.asarray(cells, dtype=np.int64)
    _check(formulation, c, x.shape[-1])
    if s.shape[-2:] != (c.shape[0], 4) or s.shape[:-2] != x.shape[:-2]:
        raise ValueError(
            f"internal force: stress of shape {s.shape} does not match "
            f"positions {x.shape} and {c.shape[0]} elements of 4 components"
        )
    return _assemble(_element_forces(x, s, c, formulation), c, x.shape[-2])


def lumped_mass(
    reference_coords: ArrayLike,
    cells: ArrayLike,
    formulation: str,
    density: float | ArrayLike,
) -> NDArray[np.float64]:
    """Lumped nodal mass: each element's mass shared equally by its nodes.

    Parameters
    ----------
    reference_coords : array_like, shape (N, 2)
        Undeformed coordinates (mm).
    cells : array_like, shape (E, 4)
    formulation : str
    density : float or array_like, shape (E,)
        Mass per unit volume in the working frame (tonne/mm³ with mm, MPa
        and seconds).

    Returns
    -------
    numpy.ndarray
        Shape ``(N,)``.
    """
    x = np.asarray(reference_coords, dtype=np.float64)
    c = np.asarray(cells, dtype=np.int64)
    _check(formulation, c, x.shape[-1])
    corners = x[c]
    d31 = corners[:, 2] - corners[:, 0]
    d42 = corners[:, 3] - corners[:, 1]
    volume = np.abs(0.5 * (d31[:, 0] * d42[:, 1] - d42[:, 0] * d31[:, 1]))
    if formulation == "axisymmetric":
        volume = volume * 2.0 * np.pi * corners[:, :, 0].mean(axis=1)
    share = 0.25 * volume * np.asarray(density, dtype=np.float64)
    mass = np.zeros(x.shape[0], dtype=np.float64)
    for a in range(4):
        np.add.at(mass, c[:, a], share)
    return mass


def element_mean(nodal: ArrayLike, cells: ArrayLike) -> NDArray[np.float64]:
    """A nodal field averaged to the elements: the mean over each element's
    nodes, ``(..., N, C)`` to ``(..., E, C)``. How a nodal arm's stress
    reaches the operator; the record says when it was used."""
    values = np.asarray(nodal, dtype=np.float64)
    c = np.asarray(cells, dtype=np.int64)
    return values[..., c, :].mean(axis=-2)


def _forces(inputs: QoiInputs) -> NDArray[np.float64] | None:
    """The nodal internal force of every frame, or ``None`` when the caller
    gave no stress, mesh or formulation."""
    if (
        inputs.element_stress is None
        or inputs.cells is None
        or inputs.element_formulation is None
    ):
        return None
    return internal_force(
        inputs.positions,
        inputs.element_stress,
        inputs.cells,
        inputs.element_formulation,
    )


def _undefined(inputs: QoiInputs) -> NDArray[np.float64]:
    return np.full(np.asarray(inputs.positions).shape[0], np.nan)


def _unit(direction: ArrayLike) -> NDArray[np.float64]:
    d = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        raise ValueError("a force quantity needs a non-zero direction")
    return d / norm


def reaction_force(node_set: str, direction: ArrayLike) -> SeriesFn:
    """Series factory: the reaction at a named node set, along a direction.

    The internal force summed over the set's nodes: what the support (or
    whatever holds those nodes) applies to the body. Every frame is ``nan``
    when the caller gave no stress; an unknown set is an error.

    Parameters
    ----------
    node_set : str
        A key of the definition's ``node_sets``.
    direction : array_like, shape (2,)
        The component read; normalised here.
    """
    d = _unit(direction)

    def series(inputs: QoiInputs) -> NDArray[np.float64]:
        forces = _forces(inputs)
        if forces is None:
            return _undefined(inputs)
        sets = inputs.node_sets or {}
        if node_set not in sets:
            raise KeyError(
                f"reaction_force: the definition declares no node set "
                f"{node_set!r} (it has {sorted(sets)})"
            )
        return forces[:, np.asarray(sets[node_set], dtype=bool)].sum(axis=1) @ d

    return series


def _plane_gap(
    plane: _PlaneLike, positions: NDArray, particle_type: NDArray | None
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Signed distance of every node from the plane on every frame, ``(T,
    N)``, positive on the material's side, and the mask of the nodes that
    are not the plane's own."""
    normal = np.asarray(plane.normal, dtype=np.float64)
    along = np.asarray(positions, dtype=np.float64) @ normal  # (T, N)
    other = np.ones(along.shape[1], dtype=bool)
    if plane.node_type is None:
        return along - float(plane.offset or 0.0), other
    if particle_type is None:
        raise ValueError("a moving rigid plane needs the node types of the case")
    own = np.asarray(particle_type) == plane.node_type
    if own.sum() != 1:
        raise ValueError(
            f"a moving rigid plane needs exactly one node of type "
            f"{plane.node_type}; the case has {int(own.sum())}"
        )
    return along - along[:, own], ~own


def body_force(plane: _PlaneLike, tolerance: float) -> SeriesFn:
    """Series factory: the force a rigid plane applies to the body, along
    the plane's normal.

    On each frame the nodes on the body are the element nodes within
    ``tolerance`` of the plane (or past it); the internal force summed over
    them, along the outward normal, is what the body pushes with (positive
    in compression). The plane's own node, and nodes of no element, carry no
    internal force and do not enter.

    Parameters
    ----------
    plane :
        A declared :class:`~structbench.benchmarks.registry.RigidPlane`.
    tolerance : float
        Distance from the plane within which a node counts as on it (mm).
    """
    normal = np.asarray(plane.normal, dtype=np.float64)

    def series(inputs: QoiInputs) -> NDArray[np.float64]:
        forces = _forces(inputs)
        if forces is None:
            return _undefined(inputs)
        gap, other = _plane_gap(plane, inputs.positions, inputs.particle_type)
        on_body = (gap <= tolerance) & other[None, :]
        return ((forces @ normal) * on_body).sum(axis=1)

    return series


def section_force(axis: int, coordinate: float, direction: ArrayLike) -> SeriesFn:
    """Series factory: the force carried through a section of the body.

    The section is the surface ``x[axis] = coordinate`` in the frame-0
    geometry, followed with the material. Elements whose frame-0 centroid
    lies above it form one side; what they apply to the nodes they share
    with the other side (the reverse of their internal force there), along
    ``direction``, is the force the upper part applies to the lower through
    the section: against the axis in compression.

    Parameters
    ----------
    axis : int
        The coordinate the section is normal to.
    coordinate : float
        Where the section cuts, in frame-0 coordinates (mm).
    direction : array_like, shape (2,)
        The component read; normalised here.
    """
    d = _unit(direction)

    def series(inputs: QoiInputs) -> NDArray[np.float64]:
        if (
            inputs.element_stress is None
            or inputs.cells is None
            or inputs.element_formulation is None
        ):
            return _undefined(inputs)
        positions = np.asarray(inputs.positions, dtype=np.float64)
        cells = np.asarray(inputs.cells, dtype=np.int64)
        _check(inputs.element_formulation, cells, positions.shape[-1])
        centroid = positions[0][cells][:, :, axis].mean(axis=1)
        upper = centroid > coordinate
        on_lower = np.zeros(positions.shape[1], dtype=bool)
        on_lower[cells[~upper].ravel()] = True
        stress = np.asarray(inputs.element_stress, dtype=np.float64)
        partial = _assemble(
            _element_forces(
                positions,
                stress[:, upper],
                cells[upper],
                inputs.element_formulation,
            ),
            cells[upper],
            positions.shape[1],
        )
        return -(partial[:, on_lower].sum(axis=1) @ d)

    return series


def _scored(series: SeriesFn, inputs: QoiInputs) -> NDArray[np.float64]:
    return np.asarray(series(inputs), dtype=np.float64)[inputs.init :]


def series_peak(series: SeriesFn, *, absolute: bool = True) -> QoiFn:
    """Quantity of interest: the peak of a series over the scored span
    (of its magnitude unless ``absolute`` is false)."""

    def qoi(inputs: QoiInputs) -> float:
        values = _scored(series, inputs)
        return float(np.abs(values).max() if absolute else values.max())

    return qoi


def series_mean(series: SeriesFn) -> QoiFn:
    """Quantity of interest: the time mean of a series over the scored span
    (trapezoid over the frames' own times, so uneven frames are weighted)."""

    def qoi(inputs: QoiInputs) -> float:
        values = _scored(series, inputs)
        time = np.asarray(inputs.time, dtype=np.float64)[inputs.init :]
        span = float(time[-1] - time[0]) if time.size > 1 else 0.0
        if span <= 0.0:
            return float(values.mean())
        return float(np.trapezoid(values, time) / span)

    return qoi


def series_impulse(series: SeriesFn) -> QoiFn:
    """Quantity of interest: the time integral of a series over the scored
    span (N s for a force in N and time in seconds)."""

    def qoi(inputs: QoiInputs) -> float:
        values = _scored(series, inputs)
        time = np.asarray(inputs.time, dtype=np.float64)[inputs.init :]
        return float(np.trapezoid(values, time))

    return qoi


def force_balance_residual(
    positions: ArrayLike,
    element_stress: ArrayLike,
    cells: ArrayLike,
    formulation: str,
    free: ArrayLike,
) -> float:
    """Out-of-balance force at the free nodes of a static problem, normalised.

    ``‖f_int at the free nodes‖₂ / ‖the unassembled element forces‖₂``,
    pooled over the frames given: zero for a stress field in equilibrium
    with no load at those nodes, of order one for a field whose element
    forces do not cancel at all. ``nan`` when the stress is not finite or
    carries no force.

    Parameters
    ----------
    positions : array_like, shape (T, N, 2)
    element_stress : array_like, shape (T, E, 4)
    cells : array_like, shape (E, 4)
    formulation : str
    free : array_like of bool, shape (N,)
        The nodes with no external force: not supported, not prescribed and
        not in contact.
    """
    x = np.asarray(positions, dtype=np.float64)
    s = np.asarray(element_stress, dtype=np.float64)
    c = np.asarray(cells, dtype=np.int64)
    _check(formulation, c, x.shape[-1])
    if not (np.isfinite(x).all() and np.isfinite(s).all()):
        return float("nan")
    per_element = _element_forces(x, s, c, formulation)
    scale = float(np.linalg.norm(per_element.ravel()))
    if scale == 0.0:
        return float("nan")
    nodal = _assemble(per_element, c, x.shape[-2])
    mask = np.asarray(free, dtype=bool)
    return float(np.linalg.norm(nodal[:, mask].ravel()) / scale)


def plane_penetration(
    plane: _PlaneLike,
    positions: ArrayLike,
    particle_type: ArrayLike | None,
    nodes: ArrayLike | None = None,
) -> float:
    """Largest depth by which a node passes a rigid plane, over the frames
    given (mm); zero when none does. ``nan`` for positions that are not
    finite.

    Parameters
    ----------
    plane :
        A declared :class:`~structbench.benchmarks.registry.RigidPlane`.
    positions : array_like, shape (T, N, dim)
    particle_type : array_like, shape (N,), optional
        Needed for a plane that moves with a node.
    nodes : array_like of bool, shape (N,), optional
        The nodes examined (the deformable ones); every node but the plane's
        own when omitted.
    """
    x = np.asarray(positions, dtype=np.float64)
    if not np.isfinite(x).all():
        return float("nan")
    types = None if particle_type is None else np.asarray(particle_type)
    gap, other = _plane_gap(plane, x, types)
    keep = other if nodes is None else other & np.asarray(nodes, dtype=bool)
    if not keep.any():
        return 0.0
    return float(max(0.0, -gap[:, keep].min()))
