"""Read a canonical case into a model-ready particle trajectory.

The ML layer works in millimetres and megapascals (ADR-0019); canonical
storage is strict SI, so positions are scaled by ``length_scale`` (m->mm) and
stress by ``stress_scale`` (Pa->MPa) here. Two loading paths exist: SPH cases
(``"sph" in case.elements``) return SPH particles only, dropping
visualization shell nodes; mesh (nodal-FE) cases return every node as a
particle, with connectivity and reference coordinates alongside (ADR-0043).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..core import Case, read_case


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


def max_principal_strain_from_voigt(
    strain: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Maximum principal strain from a Voigt tensor ``[xx, yy, zz, xy, yz, zx]``.

    Engineering shear components (indices 3-5) are halved to form the symmetric
    strain tensor before the eigenvalue solve (ADR-0029). Dimensionless.

    Parameters
    ----------
    strain:
        Array with last axis of length 6.

    Returns
    -------
    numpy.ndarray
        Same leading shape as ``strain`` with the last axis removed; the
        largest eigenvalue (principal strain) of each tensor.
    """
    voigt = np.asarray(strain, dtype=np.float64)
    tensor = np.zeros((*voigt.shape[:-1], 3, 3), dtype=np.float64)
    tensor[..., 0, 0] = voigt[..., 0]
    tensor[..., 1, 1] = voigt[..., 1]
    tensor[..., 2, 2] = voigt[..., 2]
    tensor[..., 0, 1] = tensor[..., 1, 0] = voigt[..., 3] / 2
    tensor[..., 1, 2] = tensor[..., 2, 1] = voigt[..., 4] / 2
    tensor[..., 0, 2] = tensor[..., 2, 0] = voigt[..., 5] / 2
    return np.asarray(np.linalg.eigvalsh(tensor)[..., -1])


AuxExtractor = Callable[
    [Mapping[str, NDArray[np.floating]], float], NDArray[np.float32]
]
"""Maps (mapping of SPH response fields, stress_scale) to a ``(T, P)`` or
``(T, P, k)`` aux array — single-channel extractors return ``(T, P)`` and are
lifted to one channel by the loader (ADR-0059)."""


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


def _aux_damage(
    sph: Mapping[str, NDArray[np.floating]], stress_scale: float
) -> NDArray[np.float32]:
    """K&C scaled damage measure from the effective-plastic-strain slot.

    For ``*MAT_CONCRETE_DAMAGE_REL3`` the d3plot effective-plastic-strain
    slot records the scaled damage measure (0..2), unitless — so
    ``stress_scale`` is ignored (ADR-0026).

    Parameters
    ----------
    sph:
        Mapping of SPH response fields with an
        ``"effective_plastic_strain"`` key holding a ``(T, P)`` array.
    stress_scale:
        Unused; present for the :data:`AuxExtractor` signature.

    Returns
    -------
    numpy.ndarray
        Shape ``(T, P)``, float32, unitless.
    """
    del stress_scale
    return sph["effective_plastic_strain"][...].astype(np.float32)


def _aux_max_principal_strain(
    sph: Mapping[str, NDArray[np.floating]], stress_scale: float
) -> NDArray[np.float32]:
    """Maximum principal strain from the 6-component Voigt strain tensor.

    Engineering shear components (indices 3-5) are halved to form the
    symmetric tensor before the eigenvalue solve. Dimensionless, so
    ``stress_scale`` is ignored (ADR-0029).

    Parameters
    ----------
    sph:
        Mapping of SPH response fields with a ``"strain"`` key holding a
        ``(T, P, 6)`` Voigt array ``[xx, yy, zz, xy, yz, zx]``.
    stress_scale:
        Unused; present for the :data:`AuxExtractor` signature.

    Returns
    -------
    numpy.ndarray
        Shape ``(T, P)``, float32, dimensionless.
    """
    del stress_scale
    return max_principal_strain_from_voigt(sph["strain"][...]).astype(np.float32)


def _aux_deviatoric_stress_2d(
    sph: Mapping[str, NDArray[np.floating]], stress_scale: float
) -> NDArray[np.float32]:
    """Deviatoric stress components ``(s_xx, s_yy, s_xy)``, scaled (ADR-0059).

    The trace is removed from the 6-component Voigt Cauchy stress. Under
    plane strain (``s_yz = s_zx = 0``, ``s_zz = -(s_xx + s_yy)``) these three
    components carry the full deviator, so von Mises recovers exactly as
    ``sqrt(3/2 s:s)`` — the composed-stress route of the complete-state plan
    (C1). For non-plane-strain cases the three components are still valid
    deviator entries but no longer determine the tensor.

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
        Shape ``(T, P, 3)``, float32, working-frame units.
    """
    sig = np.asarray(sph["stress"][...], dtype=np.float64)
    tr3 = (sig[..., 0] + sig[..., 1] + sig[..., 2]) / 3.0
    dev = np.stack([sig[..., 0] - tr3, sig[..., 1] - tr3, sig[..., 3]], axis=-1)
    return (dev * stress_scale).astype(np.float32)


def _aux_effective_plastic_strain(
    sph: Mapping[str, NDArray[np.floating]], stress_scale: float
) -> NDArray[np.float32]:
    """Effective plastic strain, raw slot value, dimensionless (ADR-0059).

    Unlike :func:`_aux_damage` (which documents the K&C reuse of this d3plot
    slot as a damage measure), this extractor is for materials where the slot
    holds the true accumulated plastic strain (e.g. Taylor's
    ``*MAT_ELASTIC_PLASTIC_HYDRO``).

    Parameters
    ----------
    sph:
        Mapping of SPH response fields with an
        ``"effective_plastic_strain"`` key holding a ``(T, P)`` array.
    stress_scale:
        Unused; present for the :data:`AuxExtractor` signature.

    Returns
    -------
    numpy.ndarray
        Shape ``(T, P)``, float32, dimensionless.
    """
    del stress_scale
    return sph["effective_plastic_strain"][...].astype(np.float32)


def _aux_internal_energy(
    sph: Mapping[str, NDArray[np.floating]], stress_scale: float
) -> NDArray[np.float32]:
    """Internal energy per particle, SI joules, unscaled (ADR-0059).

    Verified on Taylor: ``sum(E)`` reproduces ``global/internal_energy``
    exactly, so the slot is total energy per particle in J (not per unit
    mass or volume). Left in SI — energy has no working-frame convention and
    per-channel normalization (ADR-0059) handles the magnitude.

    Parameters
    ----------
    sph:
        Mapping of SPH response fields with an ``"internal_energy"`` key
        holding a ``(T, P)`` array (J).
    stress_scale:
        Unused; present for the :data:`AuxExtractor` signature.

    Returns
    -------
    numpy.ndarray
        Shape ``(T, P)``, float32, J.
    """
    del stress_scale
    return sph["internal_energy"][...].astype(np.float32)


def _aux_density(
    sph: Mapping[str, NDArray[np.floating]], stress_scale: float
) -> NDArray[np.float32]:
    """Density per particle, SI kg/m^3, unscaled (ADR-0059).

    Left in SI for the same reason as internal energy; the FORM=12
    continuity-equation density cannot be recomputed from positions
    (summation density measured ~9% median error on Taylor), so it must be
    carried as state.

    Parameters
    ----------
    sph:
        Mapping of SPH response fields with a ``"density"`` key holding a
        ``(T, P)`` array (kg/m^3).
    stress_scale:
        Unused; present for the :data:`AuxExtractor` signature.

    Returns
    -------
    numpy.ndarray
        Shape ``(T, P)``, float32, kg/m^3.
    """
    del stress_scale
    return sph["density"][...].astype(np.float32)


@dataclass(frozen=True)
class AuxFieldInfo:
    """Registry metadata for one auxiliary field name (ADR-0059).

    Parameters
    ----------
    extractor:
        The SPH-path extraction function.
    labels:
        Per-channel labels; ``len(labels)`` is the field's channel count.
        Single-channel fields label the channel with the field name itself.
    unit:
        Working-frame unit shared by the field's channels (``"MPa"``,
        ``"-"`` for dimensionless, ``"J"``, ``"kg/m^3"``).
    """

    extractor: AuxExtractor
    labels: tuple[str, ...]
    unit: str

    @property
    def n_channels(self) -> int:
        return len(self.labels)


_AUX_FIELDS: dict[str, AuxFieldInfo] = {
    "von_mises_stress": AuxFieldInfo(_aux_von_mises, ("von_mises_stress",), "MPa"),
    "axial_stress": AuxFieldInfo(_aux_axial_stress, ("axial_stress",), "MPa"),
    "damage": AuxFieldInfo(_aux_damage, ("damage",), "-"),
    "max_principal_strain": AuxFieldInfo(
        _aux_max_principal_strain, ("max_principal_strain",), "-"
    ),
    "deviatoric_stress_2d": AuxFieldInfo(
        _aux_deviatoric_stress_2d, ("s_xx", "s_yy", "s_xy"), "MPa"
    ),
    "effective_plastic_strain": AuxFieldInfo(
        _aux_effective_plastic_strain, ("effective_plastic_strain",), "-"
    ),
    "internal_energy": AuxFieldInfo(_aux_internal_energy, ("internal_energy",), "J"),
    "density": AuxFieldInfo(_aux_density, ("density",), "kg/m^3"),
}

#: Backwards-compatible view used by pre-0059 call sites/tests.
_AUX_EXTRACTORS: dict[str, AuxExtractor] = {
    name: info.extractor for name, info in _AUX_FIELDS.items()
}


def as_aux_fields(aux_field: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize the ``aux_field`` argument to a non-empty name tuple.

    A bare string is the ``C = 1``-per-field shorthand (ADR-0059). Raises
    ``ValueError`` on an empty sequence — a trajectory always carries at
    least one auxiliary channel.
    """
    names = (aux_field,) if isinstance(aux_field, str) else tuple(aux_field)
    if not names:
        raise ValueError("aux_field must name at least one auxiliary field")
    return names


def aux_channel_labels(aux_field: str | Sequence[str]) -> tuple[str, ...]:
    """Flattened per-channel labels for a field selection (ADR-0059).

    Names in the SPH registry contribute their declared labels (a
    multi-channel field like ``deviatoric_stress_2d`` contributes three);
    unknown names (the mesh ``response.node`` path) contribute themselves as
    a single channel.
    """
    labels: list[str] = []
    for name in as_aux_fields(aux_field):
        info = _AUX_FIELDS.get(name)
        labels.extend(info.labels if info is not None else (name,))
    return tuple(labels)


def aux_channel_count(aux_field: str | Sequence[str]) -> int:
    """Total channel count ``C`` for a field selection (ADR-0059)."""
    return len(aux_channel_labels(aux_field))


def aux_channel_units(aux_field: str | Sequence[str]) -> tuple[str, ...]:
    """Per-channel working-frame units, aligned with :func:`aux_channel_labels`.

    Unknown (mesh-path) names report the stress working unit ``"MPa"`` —
    every current mesh aux is a stress read with ``stress_scale`` applied.
    """
    units: list[str] = []
    for name in as_aux_fields(aux_field):
        info = _AUX_FIELDS.get(name)
        if info is not None:
            units.extend([info.unit] * info.n_channels)
        else:
            units.append("MPa")
    return tuple(units)


def available_aux_fields() -> frozenset[str]:
    """Names accepted by :func:`load_case_trajectory`'s ``aux_field`` on the SPH path.

    This registry governs SPH cases only; the mesh path reads ``aux_field``
    directly as a ``response.node`` key instead (ADR-0043).
    :class:`~structbench.benchmarks.registry.BenchmarkSpec` still validates
    every benchmark's ``aux_field`` against this set, which
    ``"von_mises_stress"`` satisfies for both paths — accepted debt until a
    mesh-only aux name arrives.

    Returns
    -------
    frozenset of str
        The set of valid SPH-path ``aux_field`` strings.
    """
    return frozenset(_AUX_EXTRACTORS)


@dataclass
class CaseTrajectory:
    """One case as a particle trajectory in the ML working frame (mm, MPa)."""

    case_id: str
    positions: NDArray[np.float32]  # (T, P, dim), mm
    particle_type: NDArray[np.int64]  # (P,)
    aux: NDArray[np.float32]  # (T, P, C); per-channel units per aux_field (ADR-0059)
    time: NDArray[np.float64]  # (T,), s
    cells: NDArray[np.int64] | None = None  # (n_cells, nodes_per_cell); mesh only
    reference_coords: NDArray[np.float32] | None = None  # (P, dim), mm; mesh only

    def __post_init__(self) -> None:
        # ADR-0059 compat lift: a legacy (T, P) aux is one channel. Keeps the
        # internal representation single — every consumer sees (T, P, C).
        if self.aux.ndim == 2:
            self.aux = self.aux[..., None]


def load_case_trajectory(
    h5_path: str | Path,
    *,
    aux_field: str | Sequence[str] = "von_mises_stress",
    length_scale: float = 1e3,
    stress_scale: float = 1e-6,
) -> CaseTrajectory:
    """Load a canonical case into a :class:`CaseTrajectory`.

    Dispatches on the case's element types: SPH cases (``"sph" in
    case.elements``) return SPH particles only, dropping visualization shell
    nodes; mesh (nodal-FE) cases return every node as a particle, with
    ``cells``/``reference_coords`` populated (ADR-0043).

    Parameters
    ----------
    h5_path:
        Path to a canonical ``.h5`` case.
    aux_field:
        One name or a sequence of names; the resulting channels are
        concatenated in declaration order into ``aux``'s trailing axis
        (ADR-0059 — a bare string is the ``C = 1``-per-field shorthand).
        SPH path: each name is an extraction strategy from
        :func:`available_aux_fields` and receives ``stress_scale`` to convert
        stress-like values from SI to the working unit (multi-channel
        strategies like ``"deviatoric_stress_2d"`` contribute several
        channels). Mesh path: each name is a ``response.node`` key read
        directly (``stress_scale`` still applies). Defaults to
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
        SPH path: if ``aux_field`` is not in :func:`available_aux_fields`.
    ValueError
        If the case has no response data; or (mesh path) if
        ``nodes.node_type`` is absent, the case has more than one element
        type, or ``aux_field`` is not a ``response.node`` key.
    """
    case = read_case(h5_path)
    if case.response is None:
        raise ValueError(f"case {case.metadata.case_id} has no response")

    aux_names = as_aux_fields(aux_field)
    if "sph" in case.elements:
        infos = []
        for name in aux_names:
            try:
                infos.append(_AUX_FIELDS[name])
            except KeyError:
                raise KeyError(
                    f"unknown aux_field {name!r}; available: "
                    f"{', '.join(sorted(_AUX_FIELDS))}"
                ) from None

        sph = case.elements["sph"]
        idx = sph.connectivity[:, 0]  # node indices of the SPH particles
        dim = case.metadata.dimension
        n_frames = n_valid_frames(np.asarray(case.response.time))

        coords0 = case.nodes.coords[idx][:, :dim]  # (P, dim) SI
        disp = case.response.node["displacement"][:n_frames, idx, :]  # (T, P, dim) SI
        positions = ((coords0[None] + disp) * length_scale).astype(np.float32)

        # Each extractor sees the full response; the terminal-artifact trim
        # (ADR-0028) is applied to its output alongside positions and time.
        # Single-channel extractors return (T, P) and are lifted to one
        # channel; channels concatenate in declaration order (ADR-0059).
        blocks = []
        for info in infos:
            arr = info.extractor(case.response.element["sph"], stress_scale)
            arr = np.asarray(arr)[:n_frames]
            blocks.append(arr[..., None] if arr.ndim == 2 else arr)
        aux = np.concatenate(blocks, axis=-1).astype(np.float32)

        return CaseTrajectory(
            case_id=case.metadata.case_id,
            positions=positions,
            particle_type=np.asarray(sph.part_id, dtype=np.int64),
            aux=aux,
            time=np.asarray(case.response.time[:n_frames], dtype=np.float64),
        )

    return _load_mesh_trajectory(
        case,
        aux_fields=aux_names,
        length_scale=length_scale,
        stress_scale=stress_scale,
    )


def _load_mesh_trajectory(
    case: Case,
    *,
    aux_fields: tuple[str, ...],
    length_scale: float,
    stress_scale: float,
) -> CaseTrajectory:
    """Load a nodal-FE (mesh) case: nodes are the particles (ADR-0043)."""
    nodes = case.nodes
    if nodes.node_type is None:
        raise ValueError(
            f"mesh case {case.metadata.case_id!r} has no nodes.node_type; "
            "mesh benchmarks require it (schema 0.2.0, ADR-0042)"
        )
    if len(case.elements) != 1:
        raise ValueError(
            f"mesh case {case.metadata.case_id!r} has element types "
            f"{sorted(case.elements)}; exactly one is supported"
        )
    response = case.response
    assert response is not None  # guaranteed: shared response-None check runs first
    for name in aux_fields:
        if name not in response.node:
            raise ValueError(
                f"aux field {name!r} not in response.node "
                f"(available: {sorted(response.node)})"
            )
    n = n_valid_frames(np.asarray(response.time))  # same trim as the SPH path
    disp = response.node["displacement"][:n].astype(np.float64)
    positions = ((nodes.coords[None, :, :] + disp) * length_scale).astype(np.float32)
    aux = np.stack(
        [
            response.node[name][:n, :, 0].astype(np.float64) * stress_scale
            for name in aux_fields
        ],
        axis=-1,
    ).astype(np.float32)
    (block,) = case.elements.values()
    reference = (
        (nodes.reference_coords * length_scale).astype(np.float32)
        if nodes.reference_coords is not None
        else None
    )
    return CaseTrajectory(
        case_id=case.metadata.case_id,
        positions=positions,
        particle_type=nodes.node_type.astype(np.int64),
        aux=aux,
        time=np.asarray(response.time[:n], dtype=np.float64),
        cells=block.connectivity.astype(np.int64),
        reference_coords=reference,
    )
