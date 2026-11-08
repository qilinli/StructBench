"""Two-site state mechanics shared by the model families (ADR-0075 clause 3).

A mesh case's state has two sites: kinematics at the nodes and constitutive
state at the integration points (one per four-node reduced-integration
quad, so "element" below). This module holds what every family needs to
carry that state in its decoder rather than inside its network:

- :func:`cell_weights` and :func:`project_elements_to_nodes` — the same
  area-weighted mean the loader applies (``datasets.canonical``), in torch
  and differentiable, so the nodal view a network reads is the projection
  of the element state it carries;
- :func:`corner_mean` — the unconstrained two-site head: an element value
  as the mean of its corner nodes' values;
- :func:`quad_increment` — the centroid strain increment and in-plane
  rotation increment of four-node quads between two configurations (mid-
  configuration gradients, Hughes–Winget rotation; the axisymmetric
  formulation adds the hoop strain ``Δu_r / r``);
- :func:`radial_return` — the J2 radial return with per-element law rows
  (shear and bulk moduli, ``(peeq, sigma_y)`` knots with ``np.interp``
  end-clamp semantics, ADR-0064's contract), a short Newton solve on the
  plastic multiplier that is exact for linear hardening;
- :func:`return_map_decode` — the composition the direct return-map arm
  uses (it did not train on a metal, plan 3a note 2026-11-08: stress derived
  from a surrogate's kinematics amplifies displacement error by
  ``E / std sigma``);
- :func:`admissible_decode` — the admissible arm in **admissible
  coordinates** (ADR-0075 clause 3 note, 2026-11-08): the head's five
  channels are read as a trace-free deviator direction and radius, a
  plastic-strain increment and a free mean stress, so every point of the
  head's range is an admissible state with no constitutive call — the
  element-site port of the ADR-0064 structured heads;
- :func:`project_trial` — the admissible arm as **trial state + projection**:
  the head's five channels are an unconstrained element state that the
  radial return projects onto the admissible set inside the training graph
  (closed form, differentiable almost everywhere with the ReLU-like kink at
  the elastic/plastic switch, non-expansive in the law's own metric).

Neither admissible form reads the predicted kinematics; both read the
CARRIED anchor state (never the network's noised anchor input), so training
noise cannot ratchet the monotone plastic strain.

Stress lives in the four-component layout ``(s11, s22, s33, s12)`` the
``stress_axisymmetric`` extractor stores (``s_rr, s_zz, s_tt, s_rz`` on an
axisymmetric model, ``xx, yy, zz, xy`` on a plane one), in the working stress
unit; coordinates are in the working length unit, and ``E`` in the stress
unit, so no scale enters. Everything here is plain torch on plain tensors:
no module, no benchmark import.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

#: Element formulations the quad kinematics support.
FORMULATIONS: tuple[str, ...] = ("axisymmetric", "plane_strain")

#: The decoders a two-site simulator may carry (``TransolverConfig.two_site_decoder``).
TWO_SITE_DECODERS: tuple[str, ...] = (
    "off",
    "return_map",
    "corner_mean",
    "admissible_coords",
    "projected_trial",
)

#: The decoders that read the case's constitutive law and so need the
#: canonical :data:`RETURN_MAP_CHANNELS` layout.
LAW_DECODERS: tuple[str, ...] = ("return_map", "admissible_coords", "projected_trial")

#: Largest displacement-gradient and hoop-strain component one increment may
#: carry; beyond it the geometry a prediction implies is unphysical for a
#: frame or a segment, and the clamp keeps the trial stress finite.
INCREMENT_CAP = 1.0

#: The state-channel layout the law-reading decoders emit and consume:
#: the four stress components then the equivalent plastic strain.
RETURN_MAP_CHANNELS: tuple[str, ...] = (
    "stress_axisymmetric",
    "effective_plastic_strain",
)
RETURN_MAP_WIDTH = 5

# dN_i/dxi (row 0) and dN_i/deta (row 1) of the bilinear quad at its
# centroid, corners in counter-clockwise order (-1,-1), (1,-1), (1,1), (-1,1).
_DN_DXI = (
    torch.tensor([[-1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]], dtype=torch.float64)
    * 0.25
)


def cell_weights(coords: Tensor, cells: Tensor) -> Tensor:
    """Positive per-cell weights for the element-to-node mean.

    Polygon area in 2D (shoelace over the corners in connectivity order),
    tetrahedron volume for 3D four-node cells, 1 otherwise; equal weights
    when any cell is degenerate — the loader's rule (``_cell_weights``), so
    the two projections agree.
    """
    pts = coords[cells]  # (E, k, dim)
    dim, k = pts.shape[-1], pts.shape[1]
    if dim == 2:
        x, y = pts[..., 0], pts[..., 1]
        w = 0.5 * (x * y.roll(-1, dims=1) - x.roll(-1, dims=1) * y).sum(dim=1).abs()
    elif dim == 3 and k == 4:
        a, b, c = (pts[:, i] - pts[:, 0] for i in (1, 2, 3))
        w = (a * torch.cross(b, c, dim=-1)).sum(dim=-1).abs() / 6.0
    else:
        w = coords.new_ones(cells.shape[0])
    if not bool((w > 0).all()):
        w = torch.ones_like(w)
    return w


def project_elements_to_nodes(
    values: Tensor, cells: Tensor, weights: Tensor, n_nodes: int
) -> Tensor:
    """Weighted mean of each node's adjacent elements, ``(E, S) -> (P, S)``.

    A node no cell touches reads 0 (the loader's rule). Differentiable in
    ``values``; ``weights`` are treated as constants of the mesh.
    """
    w = weights.to(values.dtype).detach()
    num = values.new_zeros((n_nodes, values.shape[-1]))
    den = values.new_zeros((n_nodes,))
    for corner in range(cells.shape[1]):
        idx = cells[:, corner]
        num = num.index_add(0, idx, w[:, None] * values)
        den = den.index_add(0, idx, w)
    return num / den.clamp(min=torch.finfo(values.dtype).tiny)[:, None]


def corner_mean(node_values: Tensor, cells: Tensor) -> Tensor:
    """The unconstrained two-site head: ``(P, S) -> (E, S)`` as the corner mean."""
    return node_values[cells].mean(dim=1)


def quad_increment(
    x_anchor: Tensor,
    x_query: Tensor,
    cells: Tensor,
    formulation: str,
    du: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Centroid strain and rotation increments of four-node quads.

    Parameters
    ----------
    x_anchor, x_query:
        ``(P, 2)`` node positions in the two configurations (the carried
        state's configuration and the one the state is being advanced to).
    cells:
        ``(E, 4)`` int64 connectivity, corners counter-clockwise.
    formulation:
        ``"axisymmetric"`` (first coordinate the radius; the hoop strain
        ``Δu_r / r`` at the centroid enters the third diagonal slot) or
        ``"plane_strain"`` (third slot zero).
    du:
        Optional ``(P, 2)`` displacement increment ``x_query - x_anchor``
        supplied directly, for a caller that holds the two displacements
        from rest and can form their difference without the cancellation
        of the reference coordinates in single precision.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(deps (E, 3, 3), rot (E, 2, 2))`` in float64: the symmetric strain
        increment with the out-of-plane slot filled per the formulation, and
        the Hughes–Winget rotation increment of the in-plane block — the
        displacement gradient is taken in the mid-configuration, which is
        exact in strain for a rigid rotation and second-order otherwise.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(
            f"formulation must be one of {FORMULATIONS}, got {formulation!r}"
        )
    if cells.shape[1] != 4 or x_anchor.shape[-1] != 2:
        raise ValueError(
            "quad_increment needs four-node cells on 2-D coordinates; got "
            f"cells {tuple(cells.shape)} and coordinates {tuple(x_anchor.shape)}"
        )
    du_all = (x_query - x_anchor) if du is None else du
    xa = x_anchor[cells].to(torch.float64)  # (E, 4, 2)
    du = du_all[cells].to(torch.float64)  # (E, 4, 2)
    xm = xa + 0.5 * du
    dn = _DN_DXI.to(dtype=xm.dtype, device=xm.device)  # (2, 4)
    jac = torch.einsum("ri,eid->erd", dn, xm)  # J[r, d] = d x_d / d xi_r
    # Guard against degenerate predicted geometry (an element collapsed or
    # inverted in the mid-configuration): fall back to the anchor
    # configuration's gradient there, so no inverse explodes. A trained
    # model never reaches this; an untrained one does on its first steps.
    jac_anchor = torch.einsum("ri,eid->erd", dn, xa)
    degenerate = torch.linalg.det(jac).abs() < 1e-6 * torch.linalg.det(jac_anchor).abs()
    jac = torch.where(degenerate[:, None, None], jac_anchor, jac)
    jinv = torch.linalg.inv(jac)  # (J^-1)[d, r] = d xi_r / d x_d
    grad_n = torch.einsum("edr,ri->eid", jinv, dn)  # dN_i / dx_d, (E, 4, 2)
    disp_grad = torch.einsum(
        "eia,eib->eab", du, grad_n
    )  # L[a, b] = sum_i du_i[a] dN_i/dx_b
    # The same guard on magnitude: an increment beyond INCREMENT_CAP per
    # component is unphysical for one frame or segment and would only carry
    # a wild prediction into an unbounded trial stress.
    disp_grad = disp_grad.clamp(-INCREMENT_CAP, INCREMENT_CAP)
    sym = 0.5 * (disp_grad + disp_grad.transpose(1, 2))
    skew = 0.5 * (disp_grad - disp_grad.transpose(1, 2))
    eye = torch.eye(2, dtype=xm.dtype, device=xm.device).expand_as(skew)
    rot = torch.linalg.solve(eye - 0.5 * skew, eye + 0.5 * skew)
    n_cells = cells.shape[0]
    deps = xm.new_zeros((n_cells, 3, 3))
    deps[:, :2, :2] = sym
    if formulation == "axisymmetric":
        r_mid = xm[..., 0].mean(dim=1)
        du_r = du[..., 0].mean(dim=1)
        hoop = du_r / r_mid.clamp(min=torch.finfo(xm.dtype).eps)
        deps[:, 2, 2] = hoop.clamp(-INCREMENT_CAP, INCREMENT_CAP)
    return deps, rot


@dataclass(frozen=True)
class LawRows:
    """Per-element rows of an isotropic J2 law with tabulated hardening.

    ``shear`` and ``bulk`` are ``(E,)`` moduli in the stress unit;
    ``knots_peeq`` and ``knots_sy`` are ``(E, K)`` with the plastic-strain
    knots strictly increasing per row (``np.interp`` end-clamp semantics,
    ADR-0064).
    """

    shear: Tensor
    bulk: Tensor
    knots_peeq: Tensor
    knots_sy: Tensor

    def to(self, device: torch.device | str) -> LawRows:
        return LawRows(
            self.shear.to(device),
            self.bulk.to(device),
            self.knots_peeq.to(device),
            self.knots_sy.to(device),
        )


def law_rows(
    youngs_modulus: Tensor, poisson_ratio: Tensor, knots_peeq: Tensor, knots_sy: Tensor
) -> LawRows:
    """Build :class:`LawRows` from per-row elastic constants and knots."""
    e, nu = youngs_modulus, poisson_ratio
    return LawRows(
        shear=e / (2.0 * (1.0 + nu)),
        bulk=e / (3.0 * (1.0 - 2.0 * nu)),
        knots_peeq=knots_peeq,
        knots_sy=knots_sy,
    )


def sigma_y_rows(
    peeq: Tensor, knots_peeq: Tensor, knots_sy: Tensor
) -> tuple[Tensor, Tensor]:
    """Tabulated yield stress and local hardening slope, per row.

    ``peeq`` is ``(E,)``, the knots ``(E, K)``. Linear between knots and
    clamped to the end knots outside them, where the slope is 0 — exactly
    ``np.interp`` (the ADR-0064 contract), differentiable in ``peeq``.
    """
    knots_peeq = knots_peeq.contiguous()
    lo, hi = knots_peeq[:, 0], knots_peeq[:, -1]
    x = torch.minimum(torch.maximum(peeq, lo), hi)
    idx = torch.searchsorted(knots_peeq, x[:, None].contiguous(), right=True).clamp(
        1, knots_peeq.shape[1] - 1
    )
    x0 = knots_peeq.gather(1, idx - 1)[:, 0]
    x1 = knots_peeq.gather(1, idx)[:, 0]
    y0 = knots_sy.gather(1, idx - 1)[:, 0]
    y1 = knots_sy.gather(1, idx)[:, 0]
    slope = (y1 - y0) / (x1 - x0)
    sy = y0 + slope * (x - x0)
    inside = (peeq >= lo) & (peeq <= hi)
    return sy, torch.where(inside, slope, torch.zeros_like(slope))


def voigt4_to_tensor(stress: Tensor) -> Tensor:
    """``(E, 4)`` ``(s11, s22, s33, s12)`` -> ``(E, 3, 3)`` symmetric tensor."""
    s11, s22, s33, s12 = stress.unbind(-1)
    zero = torch.zeros_like(s11)
    rows = (
        torch.stack([s11, s12, zero], dim=-1),
        torch.stack([s12, s22, zero], dim=-1),
        torch.stack([zero, zero, s33], dim=-1),
    )
    return torch.stack(rows, dim=-2)


def tensor_to_voigt4(sigma: Tensor) -> Tensor:
    """``(E, 3, 3)`` -> ``(E, 4)`` ``(s11, s22, s33, s12)``."""
    return torch.stack(
        [sigma[:, 0, 0], sigma[:, 1, 1], sigma[:, 2, 2], sigma[:, 0, 1]], dim=-1
    )


def radial_return(
    stress: Tensor,
    peeq: Tensor,
    deps: Tensor,
    rot: Tensor,
    law: LawRows,
    newton_steps: int = 3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Advance a J2 state by one increment: rotate, trial, radial return.

    Parameters
    ----------
    stress:
        ``(E, 4)`` carried stress ``(s11, s22, s33, s12)``.
    peeq:
        ``(E,)`` carried equivalent plastic strain.
    deps, rot:
        The increments from :func:`quad_increment`.
    law:
        Per-element law rows.
    newton_steps:
        Corrections of the plastic multiplier after the closed-form first
        estimate; exact after one for linear hardening, three cover a
        knot crossing.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(stress (E, 4), peeq (E,), dgamma (E,))``. By construction the
        returned von Mises stress is at or below ``sigma_y(peeq)`` wherever
        the trial lay outside, ``peeq`` never decreases and the deviator
        stays traceless (the mean stress is untouched by the return).
    """
    eye3 = torch.eye(3, dtype=stress.dtype, device=stress.device)
    rot3 = eye3.expand(rot.shape[0], 3, 3).clone()
    rot3[:, :2, :2] = rot
    sigma = rot3 @ voigt4_to_tensor(stress) @ rot3.transpose(1, 2)
    tr_eps = deps.diagonal(dim1=1, dim2=2).sum(-1)
    dev_eps = deps - (tr_eps / 3.0)[:, None, None] * eye3
    g, k = law.shear, law.bulk
    trial = (
        sigma + (k * tr_eps)[:, None, None] * eye3 + (2.0 * g)[:, None, None] * dev_eps
    )
    p = trial.diagonal(dim1=1, dim2=2).sum(-1) / 3.0
    s = trial - p[:, None, None] * eye3
    q = torch.sqrt((1.5 * (s * s).sum(dim=(1, 2))).clamp(min=1e-24))
    sy, slope = sigma_y_rows(peeq, law.knots_peeq, law.knots_sy)
    f = q - sy
    plastic = f > 0
    dg = f.clamp(min=0.0) / (3.0 * g + slope)
    for _ in range(newton_steps):
        sy_n, slope_n = sigma_y_rows(peeq + dg, law.knots_peeq, law.knots_sy)
        residual = q - 3.0 * g * dg - sy_n
        dg = (dg + residual / (3.0 * g + slope_n)).clamp(min=0.0)
    dg = torch.where(plastic, dg, torch.zeros_like(dg))
    scale = 1.0 - 3.0 * g * dg / q.clamp(min=1e-6)
    s_new = scale[:, None, None] * s
    sigma_new = s_new + p[:, None, None] * eye3
    return tensor_to_voigt4(sigma_new), peeq + dg, dg


def return_map_decode(
    state_anchor: Tensor,
    x_anchor: Tensor,
    x_query: Tensor,
    cells: Tensor,
    law: LawRows,
    formulation: str,
    du: Tensor | None = None,
) -> Tensor:
    """The admissible arm's decoder: carried element state + kinematics -> state.

    ``state_anchor`` is ``(E, 5)`` in the :data:`RETURN_MAP_CHANNELS` layout
    (four stress components, then peeq); the result has the same layout at
    the query configuration, in float64: the arithmetic runs in double (a
    frame's strain increment can sit at the single-precision floor of the
    stored positions, and a deviator of a few hundred MPa must survive
    beside a mean stress a thousand times larger), and a simulator carries
    the state in double between anchors.
    """
    if state_anchor.shape[-1] != RETURN_MAP_WIDTH:
        raise ValueError(
            f"return_map_decode expects {RETURN_MAP_WIDTH} state channels "
            f"{RETURN_MAP_CHANNELS}; got {state_anchor.shape[-1]}"
        )
    deps, rot = quad_increment(x_anchor, x_query, cells, formulation, du=du)
    law64 = _law64(law)
    anchor64 = state_anchor.to(torch.float64)
    stress, peeq, _ = radial_return(anchor64[:, :4], anchor64[:, 4], deps, rot, law64)
    return torch.cat([stress, peeq[:, None]], dim=-1)


def _law64(law: LawRows) -> LawRows:
    return LawRows(
        law.shear.to(torch.float64),
        law.bulk.to(torch.float64),
        law.knots_peeq.to(torch.float64),
        law.knots_sy.to(torch.float64),
    )


def deviator_from_direction(v: Tensor) -> Tensor:
    """``(E, 3)`` ``(a, b, c)`` -> ``(E, 4)`` deviator ``(a, b, -(a + b), c)``.

    The three free coordinates of a trace-free symmetric tensor in the
    ``(s11, s22, s33, s12)`` layout; the third diagonal component is formed
    from the first two, so the trace ``(s11 + s22) + s33`` is zero exactly,
    not to rounding.
    """
    a, b, c = v.unbind(-1)
    return torch.stack([a, b, -(a + b), c], dim=-1)


def deviator_norm(v: Tensor) -> Tensor:
    """Von Mises norm of the direction coordinates ``(a, b, c)``.

    ``sqrt(1.5 * (a^2 + b^2 + (a + b)^2) + 3 c^2)``: the von Mises stress of
    :func:`deviator_from_direction`. The zero point is regularised so the
    gradient stays finite at ``v = 0``; the value error is below ``1e-12``.
    """
    a, b, c = v.unbind(-1)
    q = 1.5 * (a * a + b * b + (a + b) ** 2) + 3.0 * c * c
    return torch.sqrt(q.clamp(min=1e-24))


def assemble_stress(deviator: Tensor, mean_stress: Tensor) -> Tensor:
    """``(E, 4)`` deviator and ``(E,)`` mean stress -> ``(E, 4)`` stress."""
    p = mean_stress[:, None]
    return torch.cat([deviator[:, :3] + p, deviator[:, 3:4]], dim=-1)


def split_stress(stress: Tensor) -> tuple[Tensor, Tensor]:
    """``(E, 4)`` stress -> (``(E, 4)`` deviator with an exactly zero trace,
    ``(E,)`` mean stress). The deviator's third diagonal component is formed
    from the first two (:func:`deviator_from_direction`)."""
    s11, s22, s33, s12 = stress.unbind(-1)
    p = (s11 + s22 + s33) / 3.0
    return deviator_from_direction(torch.stack([s11 - p, s22 - p, s12], dim=-1)), p


def admissible_parts(
    head: Tensor,
    peeq_anchor: Tensor,
    law: LawRows,
    mean_stress_loc: Tensor | float,
    mean_stress_scale: Tensor | float,
) -> tuple[Tensor, Tensor, Tensor]:
    """The admissible-coordinates decode as (deviator, mean stress, peeq).

    Parameters
    ----------
    head:
        ``(E, 5)`` the head's element values in the NETWORK's output space
        (not stresses): ``(a, b, c)`` the deviator direction and radius,
        ``d`` the plastic-strain increment coordinate, ``m`` the mean-stress
        coordinate.
    peeq_anchor:
        ``(E,)`` the CARRIED anchor plastic strain (the state the simulator
        hands across the anchor interface, never the network's noised aux).
    law:
        Per-element law rows; only the hardening knots are read.
    mean_stress_loc, mean_stress_scale:
        Location and scale of the free mean-stress channel, so ``m`` of
        order one is a stress of the data's size (a scale, not a statistic:
        the network absorbs the rest).

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(deviator (E, 4), mean_stress (E,), peeq (E,))`` in float64 with,
        for ANY head value: ``peeq = peeq_anchor + softplus(d) >= peeq_anchor``
        exactly; the deviator ``sigma_y(peeq) * tanh(|v|) / |v| * v`` with
        von Mises ``sigma_y(peeq) * tanh(|v|) <= sigma_y(peeq)`` (the norm
        is 1-homogeneous; smooth at ``v -> 0``); the deviator trace zero
        exactly (:func:`deviator_from_direction`).
    """
    if head.shape[-1] != RETURN_MAP_WIDTH:
        raise ValueError(
            f"admissible_decode expects {RETURN_MAP_WIDTH} head channels; got "
            f"{head.shape[-1]}"
        )
    h = head.to(torch.float64)
    law64 = _law64(law)
    v, d, m = h[:, :3], h[:, 3], h[:, 4]
    peeq = peeq_anchor.to(torch.float64) + F.softplus(d)
    sy, _ = sigma_y_rows(peeq, law64.knots_peeq, law64.knots_sy)
    q = deviator_norm(v)
    radius = sy * torch.tanh(q) / q
    deviator = deviator_from_direction(radius[:, None] * v)
    mean_stress = torch.as_tensor(mean_stress_loc, dtype=h.dtype, device=h.device) + (
        torch.as_tensor(mean_stress_scale, dtype=h.dtype, device=h.device) * m
    )
    return deviator, mean_stress, peeq


def admissible_decode(
    head: Tensor,
    peeq_anchor: Tensor,
    law: LawRows,
    mean_stress_loc: Tensor | float,
    mean_stress_scale: Tensor | float,
) -> Tensor:
    """The admissible-coordinates decoder: ``(E, 5)`` head -> ``(E, 5)`` state
    in the :data:`RETURN_MAP_CHANNELS` layout, float64 (:func:`admissible_parts`)."""
    deviator, mean_stress, peeq = admissible_parts(
        head, peeq_anchor, law, mean_stress_loc, mean_stress_scale
    )
    return torch.cat([assemble_stress(deviator, mean_stress), peeq[:, None]], dim=-1)


def project_trial_parts(
    state_trial: Tensor, peeq_anchor: Tensor, law: LawRows
) -> tuple[Tensor, Tensor, Tensor]:
    """Project an unconstrained element state onto the admissible set.

    The admissible set at an anchor is ``{(sigma, peeq): peeq >= peeq_anchor,
    vm(dev sigma) <= sigma_y(peeq)}``. The projection is the composition of
    the plastic-strain clamp ``peeq_0 = max(peeq_trial, peeq_anchor)`` and
    the J2 radial return of the trial stress from ``peeq_0`` with a zero
    kinematic increment (:func:`radial_return`): its plastic multiplier
    ``dgamma`` gives ``peeq = peeq_0 + dgamma`` and the trial deviator is
    scaled onto the surface at that plastic strain, ``min(1, sigma_y(peeq) /
    vm_trial)`` (the Newton residual only approaches the surface after a
    knot crossing; the scale puts the state on it). The mean stress passes
    through untouched.

    Closed form and differentiable almost everywhere: one ReLU-like kink at
    the elastic/plastic switch (and one at the clamp). For linear hardening
    ``H`` it is the Euclidean projection onto a convex set in the metric
    ``|dev|_F^2 / (2 G) + H peeq^2`` (the generalised-stress metric in which
    the return map is a closest-point projection), hence non-expansive
    there: its Jacobian has operator norm at most one in that metric.

    Parameters
    ----------
    state_trial:
        ``(E, 5)`` unconstrained state in the :data:`RETURN_MAP_CHANNELS`
        layout, in physical units.
    peeq_anchor:
        ``(E,)`` the CARRIED anchor plastic strain.
    law:
        Per-element law rows.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(deviator (E, 4), mean_stress (E,), peeq (E,))`` in float64 with,
        for ANY trial: ``peeq >= peeq_anchor`` and ``peeq >= peeq_trial``;
        von Mises at or below ``sigma_y(peeq)``; the deviator trace zero
        exactly; and the identity on an admissible trial.
    """
    if state_trial.shape[-1] != RETURN_MAP_WIDTH:
        raise ValueError(
            f"project_trial expects {RETURN_MAP_WIDTH} state channels "
            f"{RETURN_MAP_CHANNELS}; got {state_trial.shape[-1]}"
        )
    st = state_trial.to(torch.float64)
    law64 = _law64(law)
    stress_trial, peeq_trial = st[:, :4], st[:, 4]
    peeq_0 = torch.maximum(peeq_trial, peeq_anchor.to(torch.float64))
    n = st.shape[0]
    deps = st.new_zeros((n, 3, 3))
    rot = torch.eye(2, dtype=st.dtype, device=st.device).expand(n, 2, 2)
    _, peeq, _ = radial_return(stress_trial, peeq_0, deps, rot, law64)
    deviator_trial, mean_stress = split_stress(stress_trial)
    q = deviator_norm(deviator_trial[:, [0, 1, 3]])
    sy, _ = sigma_y_rows(peeq, law64.knots_peeq, law64.knots_sy)
    scale = torch.minimum(torch.ones_like(q), sy / q)
    deviator = deviator_from_direction(scale[:, None] * deviator_trial[:, [0, 1, 3]])
    return deviator, mean_stress, peeq


def project_trial(state_trial: Tensor, peeq_anchor: Tensor, law: LawRows) -> Tensor:
    """The trial-state + projection decoder: ``(E, 5)`` unconstrained state ->
    ``(E, 5)`` admissible state in the :data:`RETURN_MAP_CHANNELS` layout,
    float64 (:func:`project_trial_parts`)."""
    deviator, mean_stress, peeq = project_trial_parts(state_trial, peeq_anchor, law)
    return torch.cat([assemble_stress(deviator, mean_stress), peeq[:, None]], dim=-1)


def von_mises4(stress: Tensor) -> Tensor:
    """Von Mises stress of ``(E, 4)`` ``(s11, s22, s33, s12)`` components."""
    s11, s22, s33, s12 = stress.unbind(-1)
    return torch.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * s12**2
    )
