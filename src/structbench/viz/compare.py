"""Several runs beside the reference on one case, and the cases to look at.

The qualitative look of a round of runs: for one case, the reference and
each run's stored rollout in a panel of its own, on one fringe scale set by
the reference, with each run's displacement error in its title
(:func:`compare_runs`); and the cases worth fixing for such a look, chosen
once from the runs' own errors (:func:`pick_cases`).

Like the ``__main__`` entry this module reads run directories, so it reaches
``benchmarks/`` and ``config.py``; the drawing itself is
:func:`structbench.viz.fringe.compare_panels`, which sees arrays alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..benchmarks import BenchmarkSpec, get_benchmark
from ..config import read_run_record
from ..datasets import aux_channel_labels, aux_channel_units
from ..datasets.canonical import CaseTrajectory, load_case_trajectory
from .fringe import FIELDS, FieldSpec, Panel, PlaneTrack, compare_panels

#: The error a panel's title and :func:`pick_cases` read from a metrics file.
ERROR_KEY = "rollout_rel_l2_displacement"


@dataclass(frozen=True)
class RunView:
    """What a comparison needs of one run directory."""

    path: Path
    label: str
    spec: BenchmarkSpec
    record: dict[str, Any]

    @property
    def aux_fields(self) -> tuple[str, ...]:
        """The run's nodal channel selection (the benchmark's when unset)."""
        chosen = tuple(self.record.get("train", {}).get("aux_fields") or ())
        return chosen or (self.spec.aux_field,)

    @property
    def aux_labels(self) -> tuple[str, ...]:
        """The labels of ``predicted_aux``'s channels."""
        return aux_channel_labels(self.aux_fields)

    @property
    def state_labels(self) -> tuple[str, ...]:
        """The labels of ``predicted_state``'s channels, when the run has one."""
        return aux_channel_labels(self.spec.state_channels)


def open_runs(
    runs: Sequence[str | Path], labels: Sequence[str] | None = None
) -> list[RunView]:
    """Read the run directories of one comparison.

    Raises
    ------
    ValueError
        If the labels do not match the runs one to one, or the runs are not
        all of one benchmark.
    """
    if not runs:
        raise ValueError("a comparison needs at least one run")
    if labels is not None and len(labels) != len(runs):
        raise ValueError(f"{len(labels)} labels for {len(runs)} runs")
    views = []
    for i, run in enumerate(runs):
        path = Path(run)
        record = read_run_record(path / "config.json")
        views.append(
            RunView(
                path,
                labels[i] if labels is not None else path.name,
                get_benchmark(record["run"]["benchmark"]),
                record,
            )
        )
    names = {str(v.record["run"]["benchmark"]) for v in views}
    if len(names) > 1:
        raise ValueError(f"the runs are of different benchmarks: {sorted(names)}")
    return views


def figure_span(
    pred: Mapping[str, Any], declared_bound: int | None, full: bool
) -> tuple[int, int]:
    """``(input_frames, end)``: the frames ``[input_frames, end)`` of a saved
    rollout a figure shows (ADR-0078).

    By default ``end`` is the case's scored bound, so panels are spaced over
    the scored span and an animation ends at the bound; ``full`` shows every
    saved frame (a rollout evaluated with the tail). The bound is the one
    the rollout file carries (``scored_frames``, -1 for none), so a
    benchmark loaded by path needs nothing extra; a file from before the
    field falls back to ``declared_bound``, the benchmark's own.
    """
    n_saved = int(pred["predicted_positions"].shape[0])
    input_frames = (
        int(pred["input_frames"])
        if "input_frames" in pred
        else n_saved - len(pred["position_rmse"])
    )
    if "scored_frames" in pred:
        carried = int(pred["scored_frames"])
        bound = None if carried < 0 else carried
    else:
        bound = declared_bound
    end = n_saved if full or bound is None else min(bound, n_saved)
    return input_frames, end


def _case_errors(run: Path, split: str) -> dict[str, float]:
    """Each case's displacement error from the run's metrics file of ``split``."""
    path = run / f"metrics-{split}.json"
    if not path.is_file():
        return {}
    cases = json.loads(path.read_text(encoding="utf-8")).get("cases", {})
    return {
        case: float(row[ERROR_KEY])
        for case, row in cases.items()
        if row.get(ERROR_KEY) is not None
    }


def pick_cases(
    runs: Sequence[str | Path], splits: Sequence[str], quantile: float = 0.5
) -> list[tuple[str, str]]:
    """Per split, the case at ``quantile`` of the runs' mean displacement error.

    A study fixes its cases once with this and reuses them in later rounds,
    so the rounds stay comparable. The mean is over the given runs, on the
    cases every one of them scored; the case returned is the one at that
    rank (0 the best, 1 the worst, 0.5 the median), ties broken by case id.

    Raises
    ------
    ValueError
        If ``quantile`` is outside ``[0, 1]``, or no case of a split was
        scored by every run.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    picked = []
    for split in splits:
        per_run = [_case_errors(Path(run), split) for run in runs]
        common = set(per_run[0]).intersection(*per_run[1:]) if per_run else set()
        if not common:
            raise ValueError(
                f"split {split!r}: no case with {ERROR_KEY} in every run's "
                f"metrics-{split}.json"
            )
        mean = {c: float(np.mean([errors[c] for errors in per_run])) for c in common}
        order = sorted(mean, key=lambda c: (mean[c], c))
        picked.append((split, order[min(len(order) - 1, int(quantile * len(order)))]))
    return picked


def _field_spec(label: str, unit: str) -> FieldSpec:
    """The display contract of a channel: the known one, else its label."""
    return FIELDS.get(label) or FieldSpec(label, label, unit or None, "{:.3g}")


def _element_mean(
    nodal: NDArray[np.floating], cells: NDArray[np.integer]
) -> NDArray[np.floating]:
    """``(T, N)`` nodal values as ``(T, E)``: the mean over each element's nodes."""
    return np.asarray(nodal)[:, cells].mean(axis=-1)


def _plane_tracks(
    spec: BenchmarkSpec,
    positions: NDArray[np.floating],
    node_types: NDArray[np.integer],
) -> tuple[PlaneTrack, ...]:
    """The definition's rigid planes where ``positions`` put them."""
    tracks = []
    for plane in spec.rigid_planes:
        normal = (plane.normal[0], plane.normal[1])
        if plane.node_type is None:
            assert plane.offset is not None
            tracks.append(PlaneTrack(normal, plane.offset, plane.label))
            continue
        nodes = np.flatnonzero(node_types == plane.node_type)
        if len(nodes) != 1:
            raise ValueError(
                f"rigid plane {plane.label!r} moves with the node of type "
                f"{plane.node_type}; the case has {len(nodes)} such nodes"
            )
        travel = np.asarray(positions[:, nodes[0], :2], np.float64) @ np.asarray(normal)
        tracks.append(PlaneTrack(normal, travel, plane.label))
    return tuple(tracks)


def _reference(
    views: Sequence[RunView], data_root: Path, case: str, channel: str
) -> tuple[CaseTrajectory, NDArray[np.floating], str, bool]:
    """The stored case, its values of ``channel``, the unit, and whether the
    values are per element (the stored element state) or per node."""
    spec = views[0].spec
    if spec.state_channels and channel in views[0].state_labels:
        stored = load_case_trajectory(
            data_root / f"{case}.h5",
            aux_field=spec.aux_field,
            state_fields=spec.state_channels,
        )
        assert stored.state is not None
        index = views[0].state_labels.index(channel)
        unit = aux_channel_units(spec.state_channels)[index]
        return stored, stored.state[..., index], unit, True
    for view in views:
        if channel in view.aux_labels:
            stored = load_case_trajectory(
                data_root / f"{case}.h5", aux_field=view.aux_fields
            )
            index = view.aux_labels.index(channel)
            unit = aux_channel_units(view.aux_fields)[index]
            return stored, stored.aux[..., index], unit, False
    known = sorted({label for v in views for label in v.aux_labels})
    raise ValueError(f"no run carries the channel {channel!r}; the runs carry {known}")


def _run_values(
    view: RunView,
    rollout: Any,
    channel: str,
    n_nodes: int,
    cells: NDArray[np.integer] | None,
) -> NDArray[np.floating]:
    """One run's values of ``channel``: its own element state when it stored
    one, else its nodal values (their element means when drawn by element)."""
    if (
        cells is not None
        and "predicted_state" in rollout.files
        and channel in view.state_labels
    ):
        return np.asarray(
            rollout["predicted_state"][..., view.state_labels.index(channel)]
        )
    if channel not in view.aux_labels:
        raise ValueError(
            f"run {view.label!r} carries no channel {channel!r} "
            f"(it has {', '.join(view.aux_labels)})"
        )
    aux = rollout["predicted_aux"]
    nodal = aux if aux.ndim == 2 else aux[..., view.aux_labels.index(channel)]
    nodal = nodal[:, :n_nodes]
    return nodal if cells is None else _element_mean(nodal, cells)


def compare_runs(
    views: Sequence[RunView],
    split: str,
    case: str,
    data_root: str | Path,
    out_dir: str | Path,
    *,
    channel: str | None = None,
    full: bool = False,
    bands: int | None = None,
    vmax: float | None = None,
    vmax_percentile: float | None = None,
    max_frames: int | None = 130,
    axis_labels: tuple[str, str] = ("x (mm)", "y (mm)"),
    fps: int = 12,
    dpi: int = 80,
) -> tuple[Path, Path]:
    """Draw the reference and each run's stored rollout of ``case``, side by side.

    Reads ``<run>/rollouts/<split>-<case>.npz`` of every run and the stored
    case, and writes ``<out_dir>/<case>.gif`` and ``<out_dir>/<case>-final.png``.
    A case with a mesh is drawn by element: a run that stored its own element
    state shows it, any other the mean of its nodal values over each
    element's nodes; the reference shows the stored element state when the
    benchmark declares one. The benchmark's rigid planes are drawn in every
    panel. The clock runs over the scored span unless ``full``.

    Parameters
    ----------
    views:
        The runs, from :func:`open_runs`.
    split, case:
        The scored set and the case in it.
    data_root:
        Directory of the canonical cases.
    out_dir:
        Where the two files are written; created if absent.
    channel:
        The channel's label; default the benchmark's headline field when a
        run carries it, else the first run's first channel.
    full:
        Show every saved frame, past the scored bound.
    bands, max_frames, axis_labels, fps, dpi:
        As in :func:`~structbench.viz.fringe.compare_panels`.
    vmax, vmax_percentile:
        The top of the fringe scale: a value, or a percentile of the
        reference at the last frame shown; default the reference's maximum.

    Returns
    -------
    tuple of pathlib.Path
        The animation and the final-frame figure.
    """
    spec = views[0].spec
    if channel is None:
        headline = aux_channel_labels((spec.aux_field,))[0]
        carried = any(headline in view.aux_labels for view in views)
        channel = headline if carried else views[0].aux_labels[0]
    stored, reference, unit, by_element = _reference(
        views, Path(data_root), case, channel
    )
    n_nodes = stored.positions.shape[1]
    cells = stored.cells
    typed = stored
    if cells is not None and spec.mesh_transform is not None:
        typed = spec.mesh_transform(stored)
    node_types = np.asarray(typed.particle_type)[:n_nodes]
    if cells is not None and not by_element:
        reference = _element_mean(reference, cells)

    drawn: list[tuple[str, NDArray[np.floating], NDArray[np.floating]]] = []
    end = stored.positions.shape[0]
    for view in views:
        path = view.path / "rollouts" / f"{split}-{case}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"run {view.label!r} has no stored rollout {path}")
        rollout = np.load(path)
        _, run_end = figure_span(rollout, spec.scored_frames_for(case), full)
        end = min(end, run_end)
        error = _case_errors(view.path, split).get(case)
        title = (
            view.label if error is None else f"{view.label}\ndisp. error {error:.3f}"
        )
        drawn.append(
            (
                title,
                rollout["predicted_positions"][:, :n_nodes],
                _run_values(view, rollout, channel, n_nodes, cells),
            )
        )
    panels = [
        Panel(
            "Reference",
            stored.positions[:end],
            reference[:end],
            _plane_tracks(spec, stored.positions[:end], node_types),
        )
    ]
    panels += [
        Panel(title, pos[:end], val[:end], _plane_tracks(spec, pos[:end], node_types))
        for title, pos, val in drawn
    ]
    if vmax is None and vmax_percentile is not None:
        vmax = float(np.percentile(reference[end - 1], vmax_percentile))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gif, final = out / f"{case}.gif", out / f"{case}-final.png"
    compare_panels(
        panels,
        gif,
        cells=cells,
        field=_field_spec(channel, unit),
        times_us=stored.time[:end] * 1e6,
        title=f"{case} ({split})",
        vmax=vmax,
        bands=bands,
        final_path=final,
        max_frames=max_frames,
        axis_labels=axis_labels,
        fps=fps,
        dpi=dpi,
    )
    return gif, final
