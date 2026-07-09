"""FEM-postprocessor-style fringe plots for particle physics fields.

Rendering follows the conventions structural engineers know from LS-PrePost
and Abaqus/CAE (ADR-0022): the jet rainbow color code (blue = low, red =
high), a fringe bar with evenly spaced labelled levels, physical units in
the ML working frame (mm, MPa), equal-aspect axes, and the rigid wall drawn
at its plane. Every figure showing a physics quantity goes through this
module so runs, docs, and the README stay visually consistent.

Matplotlib is deliberately an optional dependency: install the ``viz``
extra (``pip install structbench[viz]``). Import of this module succeeds
without it; plotting calls raise with that instruction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..core import read_case
from ..datasets.canonical import (
    max_principal_strain_from_voigt,
    n_valid_frames,
    von_mises_from_voigt,
)

if TYPE_CHECKING:  # matplotlib types only for annotations; import stays lazy
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure


def _plt() -> Any:
    """Import pyplot lazily, with an actionable error when it is absent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "matplotlib is required for structbench.viz - install the viz "
            "extra: pip install structbench[viz]"
        ) from exc
    return plt


@dataclass(frozen=True)
class FieldSpec:
    """Display contract for one physics quantity.

    Parameters
    ----------
    key:
        Canonical field name (matches the case-schema response field).
    label:
        Human-readable name shown on the fringe bar.
    unit:
        Unit in the ML working frame; ``None`` for dimensionless fields.
    fmt:
        ``str.format`` spec for fringe-level tick labels.
    """

    key: str
    label: str
    unit: str | None
    fmt: str = "{:.1f}"

    @property
    def bar_label(self) -> str:
        """Fringe-bar caption, e.g. ``"von Mises stress (MPa)"``."""
        return f"{self.label} ({self.unit})" if self.unit else self.label


#: Known quantities, keyed by canonical field name (working frame: mm, MPa).
FIELDS: dict[str, FieldSpec] = {
    spec.key: spec
    for spec in (
        FieldSpec("von_mises_stress", "von Mises stress", "MPa"),
        FieldSpec("axial_stress", "axial stress", "MPa"),
        FieldSpec(
            "effective_plastic_strain", "effective plastic strain", None, "{:.3f}"
        ),
        FieldSpec("max_principal_strain", "max principal strain", None, "{:.4f}"),
        FieldSpec("pressure", "pressure", "MPa"),
        FieldSpec("density", "density", "kg/m³", "{:.0f}"),
        FieldSpec("displacement_magnitude", "displacement magnitude", "mm", "{:.2f}"),
        FieldSpec("velocity_magnitude", "velocity magnitude", "m/s"),
    )
}

#: Default number of labelled fringe levels (README/LS-PrePost look).
FRINGE_LEVELS = 7


def _resolve(field: str | FieldSpec) -> FieldSpec:
    if isinstance(field, FieldSpec):
        return field
    try:
        return FIELDS[field]
    except KeyError:
        known = ", ".join(sorted(FIELDS))
        raise KeyError(f"unknown field {field!r}; known fields: {known}") from None


def _cmap(bands: int | None) -> Colormap:
    """The FEM fringe color code: jet, optionally discretised into bands."""
    plt = _plt()
    return plt.get_cmap("jet", bands) if bands else plt.get_cmap("jet")


def _limits(
    values: NDArray[np.floating], vmin: float | None, vmax: float | None
) -> tuple[float, float]:
    """FEM fringe defaults: true max; zero floor for non-negative fields."""
    lo = float(values.min())
    if vmin is None:
        vmin = 0.0 if lo >= 0.0 else lo
    if vmax is None:
        vmax = float(values.max())
    if vmax <= vmin:  # constant field - keep a non-degenerate bar
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def fringe_scatter(
    ax: Axes,
    positions: NDArray[np.floating],
    values: NDArray[np.floating],
    *,
    field: str | FieldSpec = "von_mises_stress",
    vmin: float | None = None,
    vmax: float | None = None,
    bands: int | None = None,
    size: float = 2.0,
) -> PathCollection:
    """Draw one frame of particles as an FEM-style fringe scatter.

    Composition primitive: adds only the mark to ``ax``. Use
    :func:`snapshot` / :func:`compare_rollout` for complete figures.

    Parameters
    ----------
    ax:
        Target axes.
    positions:
        Particle coordinates ``(P, dim)`` in mm; 3D particles are drawn
        as their x-y projection.
    values:
        Per-particle field values ``(P,)``.
    field:
        Field key from :data:`FIELDS`, or a custom :class:`FieldSpec`.
    vmin, vmax:
        Fringe range. Defaults per FEM convention: 0 (or the true minimum
        when negative) to the true maximum of ``values``.
    bands:
        When given, discretise the fringe into this many bands
        (Abaqus/LS-PrePost banded look); otherwise continuous.
    size:
        Marker size in points².

    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter, ready for a shared fringe bar.
    """
    _resolve(field)
    vmin, vmax = _limits(np.asarray(values), vmin, vmax)
    return ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=np.clip(values, vmin, vmax),
        cmap=_cmap(bands),
        vmin=vmin,
        vmax=vmax,
        s=size,
        linewidths=0,
        rasterized=True,
    )


def _fringe_bar(
    fig: Figure,
    mappable: PathCollection,
    spec: FieldSpec,
    *,
    axes: Any,
    levels: int = FRINGE_LEVELS,
) -> Colorbar:
    """Attach the labelled fringe bar (evenly spaced levels, unit caption)."""
    vmin, vmax = mappable.get_clim()
    ticks = np.linspace(vmin, vmax, levels)
    bar = fig.colorbar(mappable, ax=axes, ticks=ticks, shrink=0.85, pad=0.02)
    bar.ax.set_yticklabels([spec.fmt.format(t) for t in ticks])
    bar.set_label(spec.bar_label)
    return bar


def _draw_wall(ax: Axes, wall_x: float, *, width: float = 1.5) -> None:
    """Gray rigid-wall band at ``wall_x`` (mm), labelled like the README."""
    ax.axvspan(wall_x - width, wall_x, color="0.45", zorder=0)
    ax.text(
        wall_x - width / 2,
        0.98,
        "rigid wall",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha="center",
        va="top",
        fontsize=7,
        color="1.0",
        clip_on=True,
    )


def snapshot(
    positions: NDArray[np.floating],
    values: NDArray[np.floating],
    *,
    field: str | FieldSpec = "von_mises_stress",
    title: str = "",
    time_us: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    bands: int | None = None,
    wall_x: float | None = None,
    size: float = 2.0,
) -> Figure:
    """One labelled fringe frame: particles, wall, axes in mm, fringe bar.

    Parameters
    ----------
    positions, values, field, vmin, vmax, bands, size:
        As in :func:`fringe_scatter`.
    title:
        Figure title; the frame time is appended when ``time_us`` is given.
    time_us:
        Frame time in microseconds.
    wall_x:
        Rigid-wall plane x (mm); drawn when given.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _plt()
    spec = _resolve(field)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    sc = fringe_scatter(
        ax, positions, values, field=spec, vmin=vmin, vmax=vmax, bands=bands, size=size
    )
    if wall_x is not None:
        _draw_wall(ax, wall_x)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    stamp = f"   (t = {time_us:.1f} µs)" if time_us is not None else ""
    fig.suptitle(f"{title}{stamp}" if title else stamp.strip(), fontsize=11)
    _fringe_bar(fig, sc, spec, axes=ax)
    return fig


def compare_rollout(
    gt_positions: NDArray[np.floating],
    gt_values: NDArray[np.floating],
    pred_positions: NDArray[np.floating],
    pred_values: NDArray[np.floating],
    *,
    field: str | FieldSpec = "von_mises_stress",
    frames: Sequence[int],
    times_us: NDArray[np.floating] | None = None,
    title: str = "",
    row_labels: tuple[str, str] = ("Ground truth", "Prediction"),
    vmin: float | None = None,
    vmax: float | None = None,
    bands: int | None = None,
    wall_x: float | None = None,
    size: float = 1.6,
) -> Figure:
    """Ground-truth vs prediction fringe grid over selected frames.

    The fringe range is set by the *ground-truth* frames (FEM convention:
    the reference solution defines the scale; predictions are clipped into
    it), and all panels share the same spatial extent so deformation is
    directly comparable.

    Parameters
    ----------
    gt_positions, pred_positions:
        Trajectories ``(T, P, dim)`` in mm; 3D trajectories are drawn as
        their x-y projection.
    gt_values, pred_values:
        Per-particle fields ``(T, P)``.
    frames:
        Frame indices to show as columns.
    times_us:
        Per-frame times in microseconds for column titles.
    title:
        Figure title.
    row_labels:
        Labels for the two rows.
    field, vmin, vmax, bands, size, wall_x:
        As in :func:`snapshot`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _plt()
    spec = _resolve(field)
    frames = list(frames)
    vmin, vmax = _limits(np.asarray(gt_values[frames]), vmin, vmax)

    shown = np.concatenate(
        [
            gt_positions[frames][..., :2].reshape(-1, 2),
            pred_positions[frames][..., :2].reshape(-1, 2),
        ]
    )
    (x0, y0), (x1, y1) = shown.min(axis=0) - 2.0, shown.max(axis=0) + 2.0

    fig, axes = plt.subplots(
        2,
        len(frames),
        figsize=(3.1 * len(frames) + 1.6, 6.2),
        constrained_layout=True,
        squeeze=False,
    )
    if title:
        fig.suptitle(title, fontsize=12)
    sc = None
    rows = (
        (row_labels[0], gt_positions, gt_values),
        (row_labels[1], pred_positions, pred_values),
    )
    for row, (label, positions, values) in enumerate(rows):
        for col, frame in enumerate(frames):
            ax = axes[row, col]
            sc = fringe_scatter(
                ax,
                positions[frame],
                values[frame],
                field=spec,
                vmin=vmin,
                vmax=vmax,
                bands=bands,
                size=size,
            )
            if wall_x is not None:
                _draw_wall(ax, wall_x)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7.5)
            if row == 0 and times_us is not None:
                ax.set_title(f"t = {times_us[frame]:.0f} µs", fontsize=9.5)
            if col == 0:
                ax.set_ylabel(f"{label}\ny (mm)", fontsize=9.5)
            else:
                ax.set_yticklabels([])
            if row == 1:
                ax.set_xlabel("x (mm)", fontsize=9.5)
    assert sc is not None
    _fringe_bar(fig, sc, spec, axes=axes)
    return fig


def animate_rollout(
    positions: NDArray[np.floating],
    values: NDArray[np.floating],
    out_path: str | Path,
    *,
    field: str | FieldSpec = "von_mises_stress",
    times_us: NDArray[np.floating] | None = None,
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    bands: int | None = None,
    wall_x: float | None = None,
    size: float = 2.0,
    fps: int = 15,
    dpi: int = 100,
) -> Path:
    """Write a fringe animation of a full trajectory (README-style GIF).

    Parameters
    ----------
    positions, values:
        Trajectory ``(T, P, dim)`` in mm (3D drawn as the x-y projection)
        and field ``(T, P)``.
    out_path:
        Output file; the extension picks the writer (``.gif`` uses pillow).
    times_us, title, field, vmin, vmax, bands, wall_x, size:
        As in :func:`compare_rollout`; the fringe range defaults to the
        global min/max over all frames so the bar stays fixed.
    fps, dpi:
        Animation frame rate and raster resolution.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    plt = _plt()
    from matplotlib.animation import FuncAnimation, PillowWriter

    spec = _resolve(field)
    vmin, vmax = _limits(np.asarray(values), vmin, vmax)
    xy = positions[..., :2].reshape(-1, 2)
    (x0, y0), (x1, y1) = xy.min(axis=0) - 2.0, xy.max(axis=0) + 2.0

    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    sc = fringe_scatter(
        ax,
        positions[0],
        values[0],
        field=spec,
        vmin=vmin,
        vmax=vmax,
        bands=bands,
        size=size,
    )
    if wall_x is not None:
        _draw_wall(ax, wall_x)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    _fringe_bar(fig, sc, spec, axes=ax)

    def _stamp(frame: int) -> str:
        if times_us is None:
            return title
        return f"{title}   (t = {times_us[frame]:.1f} µs)" if title else ""

    text = fig.suptitle(_stamp(0), fontsize=10)

    def _update(frame: int) -> tuple[Any, ...]:
        sc.set_offsets(positions[frame][:, :2])
        sc.set_array(np.clip(values[frame], vmin, vmax))
        text.set_text(_stamp(frame))
        return (sc, text)

    anim = FuncAnimation(fig, _update, frames=positions.shape[0], blit=False)
    out = Path(out_path)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out


def animate_comparison(
    gt_positions: NDArray[np.floating],
    gt_values: NDArray[np.floating],
    pred_positions: NDArray[np.floating],
    pred_values: NDArray[np.floating],
    out_path: str | Path,
    *,
    field: str | FieldSpec = "von_mises_stress",
    times_us: NDArray[np.floating] | None = None,
    titles: tuple[str, str] = ("Ground truth", "CGN prediction"),
    vmin: float | None = None,
    vmax: float | None = None,
    bands: int | None = None,
    wall_x: float | None = None,
    size: float = 2.0,
    fps: int = 15,
    dpi: int = 100,
) -> Path:
    """Write a side-by-side ground-truth vs prediction fringe animation (GIF).

    The animated companion to :func:`compare_rollout`: two panels
    (ground truth left, prediction right) sharing one fringe scale — set by
    the *ground truth*, FEM convention — and one spatial extent, so the
    deformation and stress fields are directly comparable frame by frame.

    Parameters
    ----------
    gt_positions, gt_values, pred_positions, pred_values:
        Ground-truth and predicted trajectories ``(T, P, dim)`` in mm and
        fields ``(T, P)``. Animated over ``min`` of the two frame counts.
    out_path:
        Output file; ``.gif`` uses the pillow writer.
    field, vmin, vmax, bands, wall_x, size:
        As in :func:`compare_rollout` / :func:`fringe_scatter`.
    times_us:
        Per-frame times (microseconds) for the shared time stamp.
    titles:
        Per-panel titles.
    fps, dpi:
        Animation frame rate and raster resolution.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    plt = _plt()
    from matplotlib.animation import FuncAnimation, PillowWriter

    spec = _resolve(field)
    vmin, vmax = _limits(np.asarray(gt_values), vmin, vmax)
    shown = np.concatenate(
        [
            gt_positions[..., :2].reshape(-1, 2),
            pred_positions[..., :2].reshape(-1, 2),
        ]
    )
    (x0, y0), (x1, y1) = shown.min(axis=0) - 2.0, shown.max(axis=0) + 2.0

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    panels = (
        (titles[0], gt_positions, gt_values),
        (titles[1], pred_positions, pred_values),
    )
    scatters = []
    for ax, (label, positions, values) in zip(axes, panels, strict=True):
        sc = fringe_scatter(
            ax,
            positions[0],
            values[0],
            field=spec,
            vmin=vmin,
            vmax=vmax,
            bands=bands,
            size=size,
        )
        if wall_x is not None:
            _draw_wall(ax, wall_x)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_title(label, fontsize=10)
        scatters.append(sc)
    axes[0].set_ylabel("y (mm)")
    _fringe_bar(fig, scatters[0], spec, axes=list(axes))

    def _stamp(frame: int) -> str:
        return f"t = {times_us[frame]:.1f} µs" if times_us is not None else ""

    text = fig.suptitle(_stamp(0), fontsize=11)
    trajectories = ((gt_positions, gt_values), (pred_positions, pred_values))

    def _update(frame: int) -> tuple[Any, ...]:
        for sc, (positions, values) in zip(scatters, trajectories, strict=True):
            sc.set_offsets(positions[frame][:, :2])
            sc.set_array(np.clip(values[frame], vmin, vmax))
        text.set_text(_stamp(frame))
        return (*scatters, text)

    n_frames = min(gt_positions.shape[0], pred_positions.shape[0])
    anim = FuncAnimation(fig, _update, frames=n_frames, blit=False)
    out = Path(out_path)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out


@dataclass(frozen=True)
class CaseField:
    """One case's trajectory of a single physics field (working frame)."""

    case_id: str
    field: FieldSpec
    positions: NDArray[np.float32]  # (T, P, dim), mm
    values: NDArray[np.float64]  # (T, P)
    time: NDArray[np.float64]  # (T,), s

    @property
    def times_us(self) -> NDArray[np.float64]:
        """Frame times in microseconds."""
        return self.time * 1e6


def load_case_field(h5_path: str | Path, field: str | FieldSpec) -> CaseField:
    """Load one field of a canonical case as an SPH-particle trajectory.

    Mirrors :func:`structbench.datasets.canonical.load_case_trajectory`
    (SPH particles only, mm/MPa working frame) but extracts an arbitrary
    known field rather than only von Mises stress.

    Parameters
    ----------
    h5_path:
        Path to a canonical ``.h5`` case.
    field:
        Field key from :data:`FIELDS`, or a custom :class:`FieldSpec` whose
        key matches an SPH element response field.

    Returns
    -------
    CaseField

    Raises
    ------
    KeyError
        If the case does not carry the response data the field needs.
    """
    spec = _resolve(field)
    case = read_case(h5_path)
    if case.response is None:
        raise KeyError(f"case {case.metadata.case_id} has no response data")
    sph = case.elements["sph"]
    idx = sph.connectivity[:, 0]
    dim = case.metadata.dimension
    # Trim the terminal solver-output dt artifact exactly like the training
    # loader, so GT frames stay aligned with rollout .npz files (ADR-0028).
    n_frames = n_valid_frames(np.asarray(case.response.time))

    coords0 = case.nodes.coords[idx][:, :dim]
    disp = case.response.node["displacement"][:n_frames, idx, :]
    positions = ((coords0[None] + disp) * 1e3).astype(np.float32)

    values = _extract_field(case, spec, idx)[:n_frames].astype(np.float64)
    return CaseField(
        case_id=case.metadata.case_id,
        field=spec,
        positions=positions,
        values=values,
        time=np.asarray(case.response.time[:n_frames], dtype=np.float64),
    )


#: SI -> working-frame scale for element-response fields read verbatim.
_ELEMENT_FIELD_SCALE = {
    "effective_plastic_strain": 1.0,
    "pressure": 1e-6,  # Pa -> MPa
    "density": 1.0,
}


def _extract_field(case: Any, spec: FieldSpec, idx: NDArray[np.int64]) -> NDArray[Any]:
    """Extract ``spec`` for the SPH particles as ``(T, P)`` working-frame."""
    element = case.response.element.get("sph", {})
    node = case.response.node
    if spec.key == "von_mises_stress":
        if "stress" not in element:
            raise KeyError(f"case {case.metadata.case_id} has no SPH stress")
        return np.asarray(von_mises_from_voigt(element["stress"]) * 1e-6)
    if spec.key == "axial_stress":
        if "stress" not in element:
            raise KeyError(f"case {case.metadata.case_id} has no SPH stress")
        return np.asarray(element["stress"], dtype=np.float64)[..., 0] * 1e-6
    if spec.key == "max_principal_strain":
        if "strain" not in element:
            raise KeyError(f"case {case.metadata.case_id} has no SPH strain")
        return np.asarray(max_principal_strain_from_voigt(element["strain"]))
    if spec.key in _ELEMENT_FIELD_SCALE:
        if spec.key not in element:
            raise KeyError(f"case {case.metadata.case_id} has no SPH {spec.key}")
        return (
            np.asarray(element[spec.key], dtype=np.float64)
            * _ELEMENT_FIELD_SCALE[spec.key]
        )
    if spec.key == "displacement_magnitude":
        return np.linalg.norm(node["displacement"][:, idx, :], axis=-1) * 1e3
    if spec.key == "velocity_magnitude":
        if "velocity" not in node:
            raise KeyError(f"case {case.metadata.case_id} has no nodal velocity")
        return np.linalg.norm(node["velocity"][:, idx, :], axis=-1)
    raise KeyError(f"no extractor for field {spec.key!r}")
