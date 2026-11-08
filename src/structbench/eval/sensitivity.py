"""The gradient task family: sensitivities of quantities of interest to a
design parameter (ADR-0076).

Two sides, both pure functions over arrays:

* **Reference.** :func:`reference_sensitivities` takes a quantity of
  interest sampled along a uniform parameter grid (a sweep line) and returns
  one :class:`SensitivityPoint` per grid value: the central difference across
  the neighbouring cases (one-sided at the ends), a truncation-error estimate
  from the same difference at twice the step, and a measurement-error term
  propagated from the quantity's own noise. A point is *judged* when both
  errors are small against the sensitivity, otherwise *reported*.
* **Surrogate.** :func:`free_flight_seed` constructs the seed frames of a
  case at a perturbed parameter value; :func:`model_sensitivity` differences
  the surrogate's own quantities at ``±δ``; :func:`gradient_metrics` scores
  them against the reference in the ADR-0055 relative form.

:func:`reference_record` and :func:`render_reference_markdown` build and
render the committed reference record (``docs/gradients/<benchmark>.*``), so
a reader sees the reference's own uncertainty before any model's number.
Nothing here is fitted to a dataset: the step, the noise terms and the judged
rule are declared by the benchmark's :class:`GradientFamily`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from ..datasets.canonical import CaseTrajectory

#: ``(trajectory, value_from, value_to, seed_frames) -> trajectory`` with the
#: seed frames rebuilt at ``value_to``.
SeedConstructor = Callable[[CaseTrajectory, float, float, int], CaseTrajectory]

__all__ = [
    "GradientFamily",
    "QoiNoise",
    "SensitivityPoint",
    "free_flight_seed",
    "gradient_floor",
    "gradient_metrics",
    "model_sensitivity",
    "reference_record",
    "reference_sensitivities",
    "render_reference_markdown",
    "sensitivities_from_record",
]

#: A reference point is judged when its larger error term is at most this
#: fraction of its sensitivity (ADR-0076 clause 4).
JUDGED_FRACTION = 0.5


@dataclass(frozen=True)
class SensitivityPoint:
    """One reference sensitivity at one grid value.

    Parameters
    ----------
    value : float
        The parameter value.
    sensitivity : float
        The difference estimate of ``dQ/dp`` at ``value``.
    truncation_error : float
        ``|s_h − s_2h|``: the same difference at twice the step. For a central
        difference this is three times the leading error term (the error
        scales with ``h²``); for a one-sided end it is the leading term
        itself. An estimate, not a bound.
    measurement_error : float
        The quantity's own noise propagated through the difference:
        ``√2·σ_Q / 2h`` (central) or ``√2·σ_Q / h`` (one-sided).
    one_sided : bool
        Whether the point is an end of the line.
    """

    value: float
    sensitivity: float
    truncation_error: float
    measurement_error: float
    one_sided: bool

    @property
    def error(self) -> float:
        """The larger of the two error terms."""
        return max(self.truncation_error, self.measurement_error)

    @property
    def judged(self) -> bool:
        """Whether the point enters the pooled headline (ADR-0076 clause 4)."""
        return self.error <= JUDGED_FRACTION * abs(self.sensitivity)


@dataclass(frozen=True)
class QoiNoise:
    """The declared measurement noise of one quantity of interest.

    ``σ_Q = absolute + relative·|Q|`` at each grid point. Both terms are
    declared by the benchmark (the card's floor, the particle spacing, the
    output interval) and ``basis`` says why, in one sentence.
    """

    absolute: float = 0.0
    relative: float = 0.0
    basis: str = ""

    def sigma(self, q: float) -> float:
        return self.absolute + self.relative * abs(q)


@dataclass(frozen=True)
class GradientFamily:
    """A benchmark's declaration of its gradient task (ADR-0076 clause 1).

    Parameters
    ----------
    parameter : str
        The scalar design parameter, named with its unit
        (``"impact_velocity_m_s"``).
    parameter_of : callable
        Case id to parameter value.
    line_of : callable
        Case id to the label of the sweep line it lies on (the other factors
        held fixed; Taylor: the bar length). Sensitivities are taken along a
        line.
    noise : mapping of str to QoiNoise
        Declared noise per quantity of interest, every scored quantity named.
    deltas : tuple of float
        The two perturbations the surrogate is differenced at; the first is
        the registered one, their disagreement the model's floor.
    perturb_seed : callable or None
        ``(trajectory, value_from, value_to, seed_frames) -> trajectory``
        with the seed frames rebuilt at ``value_to``. ``None`` means the
        family has no construction and the surrogate side cannot run.
    """

    parameter: str
    parameter_of: Callable[[str], float]
    line_of: Callable[[str], str]
    noise: Mapping[str, QoiNoise]
    deltas: tuple[float, float] = (1.0, 2.0)
    perturb_seed: SeedConstructor | None = None

    def __post_init__(self) -> None:
        if len(self.deltas) != 2 or self.deltas[0] <= 0 or self.deltas[1] <= 0:
            raise ValueError("deltas must be two positive perturbations")
        if self.deltas[0] == self.deltas[1]:
            raise ValueError("the two deltas must differ to measure the model's floor")


def _uniform_step(values: np.ndarray) -> float:
    steps = np.diff(values)
    h = float(steps[0])
    if h <= 0 or not np.allclose(steps, h, rtol=1e-9, atol=0.0):
        raise ValueError(
            f"grid values must be strictly increasing and uniform: {values}"
        )
    return h


def reference_sensitivities(
    values: Sequence[float],
    quantities: Sequence[float],
    *,
    noise: QoiNoise | None = None,
) -> tuple[SensitivityPoint, ...]:
    """Reference sensitivities along one uniform sweep line (ADR-0076 clause 2).

    Parameters
    ----------
    values : sequence of float
        Parameter values, any order; must form a uniform grid of at least
        three points.
    quantities : sequence of float
        The reference quantity at each value, same order.
    noise : QoiNoise or None
        The quantity's declared noise; ``None`` means none.

    Returns
    -------
    tuple of SensitivityPoint
        One per value, in increasing parameter order. Interior points use
        the central difference at the step ``h``; the truncation estimate
        uses the central difference at ``2h`` where both ``±2h`` neighbours
        exist and the one-sided ``2h`` difference on the available side
        otherwise (a conservative stand-in). The ends use one-sided
        differences at ``h`` and ``2h``.

    Raises
    ------
    ValueError
        If fewer than three points, or the grid is not uniform.
    """
    noise = noise or QoiNoise()
    v = np.asarray(values, dtype=float)
    q = np.asarray(quantities, dtype=float)
    if v.shape != q.shape or v.ndim != 1:
        raise ValueError("values and quantities must be one-dimensional and aligned")
    if v.size < 3:
        raise ValueError("a sweep line needs at least three points")
    order = np.argsort(v)
    v, q = v[order], q[order]
    h = _uniform_step(v)
    n = v.size
    points = []
    for i in range(n):
        if i == 0:
            s_h = (q[1] - q[0]) / h
            s_2h = (q[2] - q[0]) / (2 * h)
            one_sided = True
        elif i == n - 1:
            s_h = (q[n - 1] - q[n - 2]) / h
            s_2h = (q[n - 1] - q[n - 3]) / (2 * h)
            one_sided = True
        else:
            s_h = (q[i + 1] - q[i - 1]) / (2 * h)
            if 2 <= i <= n - 3:
                s_2h = (q[i + 2] - q[i - 2]) / (4 * h)
            else:
                # No ±2h stencil at the second and second-to-last points: the
                # leading error of the central difference, h²·q‴/6, is
                # estimated from the third difference over the four available
                # points, scaled by three so it equals |s_h − s_2h| for a
                # cubic (the same quantity the 2h rule reports).
                if i == 1:
                    q3 = (q[i + 2] - 3 * q[i + 1] + 3 * q[i] - q[i - 1]) / h**3
                else:  # i == n - 2
                    q3 = (q[i + 1] - 3 * q[i] + 3 * q[i - 1] - q[i - 2]) / h**3
                s_2h = s_h + h * h * q3 / 2.0
            one_sided = False
        sigma = noise.sigma(float(q[i]))
        e_meas = math.sqrt(2.0) * sigma / (h if one_sided else 2 * h)
        points.append(
            SensitivityPoint(
                value=float(v[i]),
                sensitivity=float(s_h),
                truncation_error=float(abs(s_h - s_2h)),
                measurement_error=float(e_meas),
                one_sided=one_sided,
            )
        )
    return tuple(points)


def model_sensitivity(q_minus: float, q_plus: float, delta: float) -> float:
    """The surrogate's central difference at ``±delta`` (ADR-0076 clause 3)."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    return (q_plus - q_minus) / (2.0 * delta)


def gradient_floor(s_small: float, s_large: float) -> float:
    """The model's own gradient floor: the disagreement of its two deltas."""
    return abs(s_small - s_large)


def gradient_metrics(
    reference: Sequence[SensitivityPoint],
    model: Sequence[float],
    floor: Sequence[float] | None = None,
) -> dict[str, float]:
    """Score one quantity's model sensitivities against the reference.

    Parameters
    ----------
    reference : sequence of SensitivityPoint
        The reference points, in order.
    model : sequence of float
        The model's registered sensitivity at each point, same order.
    floor : sequence of float or None
        The model's floor at each point, same order.

    Returns
    -------
    dict
        ``rel_l2`` — pooled relative L2 over the judged points,
        ``‖s_model − s_ref‖₂ / ‖s_ref‖₂`` (the ADR-0055 form); ``sign`` — the
        fraction of judged points whose signs agree; ``floor`` — the pooled
        model floor against ``‖s_ref‖₂`` (``nan`` when not given);
        ``n_judged`` and ``n_reported``. ``rel_l2`` and ``sign`` are ``nan``
        when no point is judged.
    """
    if len(model) != len(reference):
        raise ValueError("model sensitivities must align with the reference points")
    if floor is not None and len(floor) != len(reference):
        raise ValueError("floors must align with the reference points")
    judged = np.array([p.judged for p in reference], dtype=bool)
    ref = np.array([p.sensitivity for p in reference], dtype=float)
    mod = np.asarray(model, dtype=float)
    out: dict[str, float] = {
        "n_judged": float(judged.sum()),
        "n_reported": float((~judged).sum()),
        "rel_l2": float("nan"),
        "sign": float("nan"),
        "floor": float("nan"),
    }
    if not judged.any():
        return out
    r, m = ref[judged], mod[judged]
    denom = float(np.linalg.norm(r))
    out["rel_l2"] = float(np.linalg.norm(m - r) / denom) if denom > 0 else float("nan")
    out["sign"] = float(np.mean(np.sign(m) == np.sign(r)))
    if floor is not None:
        f = np.asarray(floor, dtype=float)[judged]
        out["floor"] = float(np.linalg.norm(f) / denom) if denom > 0 else float("nan")
    return out


def free_flight_seed(
    trajectory: CaseTrajectory,
    *,
    velocity_from: float,
    velocity_to: float,
    seed_frames: int,
    axis: int = 0,
    length_scale: float = 1e3,
    kinematic_types: tuple[int, ...] = (),
    tolerance: float = 1e-3,
) -> CaseTrajectory:
    """Rebuild a case's seed frames at another rigid-body velocity.

    For a case whose seed frames are free flight — every moving particle
    translates at the stored velocity and nothing else happens — the seed at
    any other velocity is the frame-0 configuration translated at that
    velocity: ``x_k = x_0 + sign · v · (t_k − t_0)``. The direction and sign
    are the stored case's. Frames from ``seed_frames`` on are returned
    unchanged (a rollout predicts them; the stored ones are not the truth at
    the new velocity and must not be scored against).

    Parameters
    ----------
    trajectory : CaseTrajectory
        The stored case, positions in the working frame (mm by default).
    velocity_from, velocity_to : float
        The stored and the new rigid-body speeds, in m/s (positive).
    seed_frames : int
        The protocol's observed prefix (``card.input_frames``).
    axis : int
        The axis of motion.
    length_scale : float
        Working-frame length per metre (1e3 for mm).
    kinematic_types : tuple of int
        Particle types that do not move (a wall); left as stored.
    tolerance : float
        Largest deviation, in working-frame length, of the stored seed from
        free flight at ``velocity_from`` before the construction refuses.

    Raises
    ------
    ValueError
        If the stored seed frames are not free flight at ``velocity_from``
        within ``tolerance`` — the construction is exact only then.
    """
    if seed_frames < 2 or seed_frames > trajectory.positions.shape[0]:
        raise ValueError("seed_frames must lie within the trajectory")
    pos = np.array(trajectory.positions, dtype=np.float64, copy=True)
    t0 = float(trajectory.time[0])
    t = np.asarray(trajectory.time[:seed_frames], dtype=float) - t0
    fixed = np.asarray(kinematic_types, dtype=np.int64)
    moving = ~np.isin(trajectory.particle_type, fixed)
    x0 = pos[0, moving, axis]
    motion = pos[seed_frames - 1, moving, axis] - x0
    sign = -1.0 if float(np.mean(motion)) < 0 else 1.0
    expected = x0[None, :] + sign * velocity_from * length_scale * t[:, None]
    deviation = float(np.abs(pos[:seed_frames][:, moving, axis] - expected).max())
    other_axes = [a for a in range(pos.shape[2]) if a != axis]
    if other_axes:
        seed = pos[:seed_frames][:, moving][..., other_axes]
        drift = seed - pos[0, moving][:, other_axes]
        deviation = max(deviation, float(np.abs(drift).max()))
    if deviation > tolerance:
        raise ValueError(
            f"{trajectory.case_id}: stored seed frames deviate from free flight at "
            f"{velocity_from} m/s by {deviation:.3g} (tolerance {tolerance})"
        )
    new = x0[None, :] + sign * velocity_to * length_scale * t[:, None]
    pos[:seed_frames, moving, axis] = new
    return replace(trajectory, positions=pos.astype(np.float32))


# --- the committed reference record ------------------------------------------


def reference_record(
    family: GradientFamily,
    *,
    benchmark: str,
    quantities: Mapping[str, Mapping[str, float]],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the reference record for one benchmark (ADR-0076 Consequences).

    Parameters
    ----------
    family : GradientFamily
        The benchmark's declaration.
    benchmark : str
        The registry name.
    quantities : mapping
        ``case_id -> {qoi_name: value}`` for every case on the sweep.
    source : mapping
        Provenance of the quantities (archive, revision, date, tool).

    Returns
    -------
    dict
        JSON-serialisable: the declaration (parameter, deltas, noise with its
        basis), the source, and per line the ordered values, case ids,
        quantities and sensitivities per quantity of interest.
    """
    qoi_names = tuple(family.noise)
    lines: dict[str, list[str]] = {}
    for cid in quantities:
        lines.setdefault(family.line_of(cid), []).append(cid)
    record: dict[str, Any] = {
        "benchmark": benchmark,
        "parameter": family.parameter,
        "deltas": list(family.deltas),
        "judged_fraction": JUDGED_FRACTION,
        "noise": {name: asdict(n) for name, n in family.noise.items()},
        "source": dict(source),
        "lines": {},
    }
    for line, cids in sorted(lines.items()):
        cids = sorted(cids, key=family.parameter_of)
        values = [family.parameter_of(c) for c in cids]
        per_qoi = {
            name: [float(quantities[c][name]) for c in cids] for name in qoi_names
        }
        sens = {
            name: [
                asdict(p) | {"judged": p.judged}
                for p in reference_sensitivities(
                    values, per_qoi[name], noise=family.noise[name]
                )
            ]
            for name in qoi_names
        }
        record["lines"][line] = {
            "cases": cids,
            "values": values,
            "quantities": per_qoi,
            "sensitivities": sens,
        }
    return record


Sensitivities = dict[str, dict[str, tuple[SensitivityPoint, ...]]]


def sensitivities_from_record(record: Mapping[str, Any]) -> Sensitivities:
    """Recompute every line's sensitivities from the record's stored quantities.

    The drift test compares the result with the record's stored sensitivities,
    so the committed numbers are checkable without the archive.
    """
    noise = {name: QoiNoise(**n) for name, n in record["noise"].items()}
    out: Sensitivities = {}
    for line, body in record["lines"].items():
        out[line] = {
            name: reference_sensitivities(
                body["values"], body["quantities"][name], noise=noise[name]
            )
            for name in noise
        }
    return out


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def render_reference_markdown(record: Mapping[str, Any]) -> str:
    """Render the reference record as the committed Markdown page."""
    lines = [
        "<!-- generated by tools/gradient_reference.py; do not edit by hand -->",
        "",
        f"# {record['benchmark']} — reference sensitivities (ADR-0076)",
        "",
        f"Sensitivity of each quantity of interest to `{record['parameter']}`, taken "
        "by central differences across the neighbouring cases of the sweep "
        "(one-sided at the ends of a line). Each value carries two error terms: "
        "*trunc*, the same difference at twice the step (three times the leading "
        "error of a central difference; an estimate, not a bound), and *meas*, the "
        "quantity's declared noise propagated through the difference. A point is "
        f"**judged** when the larger term is at most {record['judged_fraction']:g} "
        "of the sensitivity and enters a model's pooled headline; otherwise it is "
        "*reported* (shown in italics) and a model's number there is read against "
        "an uncertain reference. Nothing here is fitted to the data.",
        "",
        "## Declared noise",
        "",
        "| Quantity | σ absolute | σ relative | Basis |",
        "|---|---|---|---|",
    ]
    for name, n in record["noise"].items():
        lines.append(
            f"| `{name}` | {_fmt(n['absolute'])} | {_fmt(n['relative'])} "
            f"| {n['basis']} |"
        )
    src = record["source"]
    source = "; ".join(f"{k} `{v}`" for k, v in src.items())
    lines += ["", f"Source: {source}.", ""]
    for line, body in record["lines"].items():
        lines += [f"## Line {line}", ""]
        values = body["values"]
        header = "| Quantity | " + " | ".join(f"{v:g}" for v in values) + " |"
        lines += [header, "|---|" + "---|" * len(values)]
        for name, pts in body["sensitivities"].items():
            cells = []
            for p in pts:
                cell = (
                    f"{_fmt(p['sensitivity'])} (±{_fmt(p['truncation_error'])}, "
                    f"±{_fmt(p['measurement_error'])})"
                )
                cells.append(cell if p["judged"] else f"*{cell}*")
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Cells: sensitivity (±trunc, ±meas) in the quantity's unit per unit of "
            f"`{record['parameter']}`; italics mark reported, not judged. Quantities "
            "at each value: "
            + "; ".join(
                f"`{name}` " + ", ".join(_fmt(q) for q in qs)
                for name, qs in body["quantities"].items()
            )
            + ".",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"
