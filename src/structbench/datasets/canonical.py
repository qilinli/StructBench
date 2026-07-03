"""Read a canonical case into a model-ready particle trajectory.

The ML layer works in millimetres and megapascals (ADR-0019); canonical
storage is strict SI, so positions are scaled by ``length_scale`` (m->mm) and
stress by ``stress_scale`` (Pa->MPa) here. Only SPH particles are returned;
visualization shell nodes are dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..core import read_case


def n_valid_frames(time: NDArray[np.floating]) -> int:
    """Frames to keep after dropping a terminal solver-output dt artifact.

    LS-DYNA writes a final d3plot state at the exact termination time, which
    can land a fraction of the regular output interval after the previous
    state (measured 0.077 µs vs ~2 µs on the Taylor cases). Index-based
    velocity/acceleration targets assume uniform dt, so that terminal frame
    injects a spurious deceleration into training targets and biases
    final-frame metrics (ADR-0028). The frame is dropped when the final
    interval is under half the median interval.

    Parameters
    ----------
    time:
        Frame times in seconds, shape ``(T,)``.

    Returns
    -------
    int
        ``T`` or ``T - 1``.
    """
    t = np.asarray(time, dtype=np.float64)
    if t.shape[0] >= 3:
        intervals = np.diff(t)
        if intervals[-1] < 0.5 * float(np.median(intervals)):
            return t.shape[0] - 1
    return t.shape[0]


def von_mises_from_voigt(stress: NDArray[np.floating]) -> NDArray[np.float64]:
    """von Mises stress from a Voigt tensor ``[xx, yy, zz, xy, yz, zx]``.

    Parameters
    ----------
    stress:
        Array with last axis of length 6.

    Returns
    -------
    numpy.ndarray
        Same leading shape as ``stress`` with the last axis removed.
    """
    s = np.asarray(stress, dtype=np.float64)
    sx, sy, sz, sxy, syz, szx = (s[..., i] for i in range(6))
    return np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + szx**2)
    )


AuxExtractor = Callable[
    [Mapping[str, NDArray[np.floating]], float], NDArray[np.float32]
]
"""Maps (mapping of SPH response fields, stress_scale) to a (T, P) aux array."""


def _aux_von_mises(
    sph: Mapping[str, NDArray[np.floating]], stress_scale: float
) -> NDArray[np.float32]:
    """von Mises stress derived from the 6-component Voigt stress, scaled.

    Parameters
    ----------
    sph:
        Mapping of SPH response fields with a ``"stress"`` key holding a
        ``(T, P, 6)`` array (Pa).
    stress_scale:
        Multiplier applied to SI stress (e.g. 1e-6: Pa -> MPa).

    Returns
    -------
    numpy.ndarray
        Shape ``(T, P)``, dtype ``float32``.
    """
    vm = von_mises_from_voigt(sph["stress"][...])
    return (vm * stress_scale).astype(np.float32)


def _aux_axial_stress(
    sph: Mapping[str, NDArray[np.floating]], stress_scale: float
) -> NDArray[np.float32]:
    """Axial stress: Voigt component 0 (sigma_xx), scaled to the working frame.

    Parameters
    ----------
    sph:
        Mapping of SPH response fields with a ``"stress"`` key holding a
        ``(T, P, 6)`` Voigt array (Pa).
    stress_scale:
        Multiplier to the working stress unit (1e-6 for Pa -> MPa).

    Returns
    -------
    numpy.ndarray
        Shape ``(T, P)``, float32, working-frame units (MPa by default).
    """
    return (sph["stress"][...][..., 0] * stress_scale).astype(np.float32)


_AUX_EXTRACTORS: dict[str, AuxExtractor] = {
    "von_mises_stress": _aux_von_mises,
    "axial_stress": _aux_axial_stress,
}


def available_aux_fields() -> frozenset[str]:
    """Names accepted by :func:`load_case_trajectory`'s ``aux_field``.

    Returns
    -------
    frozenset of str
        The set of valid ``aux_field`` strings.
    """
    return frozenset(_AUX_EXTRACTORS)


@dataclass
class CaseTrajectory:
    """One case as a particle trajectory in the ML working frame (mm, MPa)."""

    case_id: str
    positions: NDArray[np.float32]  # (T, P, dim), mm
    particle_type: NDArray[np.int64]  # (P,)
    aux: NDArray[np.float32]  # (T, P); units depend on aux_field
    time: NDArray[np.float64]  # (T,), s


def load_case_trajectory(
    h5_path: str | Path,
    *,
    aux_field: str = "von_mises_stress",
    length_scale: float = 1e3,
    stress_scale: float = 1e-6,
) -> CaseTrajectory:
    """Load a canonical case into a :class:`CaseTrajectory` (SPH particles only).

    Parameters
    ----------
    h5_path:
        Path to a canonical ``.h5`` case.
    aux_field:
        Name of the auxiliary extraction strategy to apply.  Must be one of
        :func:`available_aux_fields`.  Stress-like extractors receive
        ``stress_scale`` to convert from SI to the working unit.  Defaults to
        ``"von_mises_stress"``.
    length_scale:
        Multiplier applied to SI positions (default 1e3: m -> mm).
    stress_scale:
        Multiplier applied to SI stress (default 1e-6: Pa -> MPa).

    Returns
    -------
    CaseTrajectory

    Raises
    ------
    KeyError
        If ``aux_field`` is not in :func:`available_aux_fields`.
    ValueError
        If the case has no response data.
    """
    try:
        extractor = _AUX_EXTRACTORS[aux_field]
    except KeyError:
        raise KeyError(
            f"unknown aux_field {aux_field!r}; available: "
            f"{', '.join(sorted(_AUX_EXTRACTORS))}"
        ) from None

    case = read_case(h5_path)
    if case.response is None:
        raise ValueError(f"case {case.metadata.case_id} has no response")
    sph = case.elements["sph"]
    idx = sph.connectivity[:, 0]  # node indices of the SPH particles
    dim = case.metadata.dimension
    n_frames = n_valid_frames(np.asarray(case.response.time))

    coords0 = case.nodes.coords[idx][:, :dim]  # (P, dim) SI
    disp = case.response.node["displacement"][:n_frames, idx, :]  # (T, P, dim) SI
    positions = ((coords0[None] + disp) * length_scale).astype(np.float32)

    # The extractor sees the full response; the terminal-artifact trim
    # (ADR-0028) is applied to its output alongside positions and time.
    aux = extractor(case.response.element["sph"], stress_scale)[:n_frames]

    return CaseTrajectory(
        case_id=case.metadata.case_id,
        positions=positions,
        particle_type=np.asarray(sph.part_id, dtype=np.int64),
        aux=aux,
        time=np.asarray(case.response.time[:n_frames], dtype=np.float64),
    )
