"""Ten-field particle state for the Taylor 2D SPH benchmark.

Stage-1 probe (see ``README.md``). Extracts the state that closes
``*MAT_ELASTIC_PLASTIC_HYDRO`` + ``*EOS_GRUNEISEN`` under SPH ``FORM=12``,
verifies it against the case file, and normalises it by physical constants
rather than dataset statistics.

Exploratory scratch code -- not part of the ``structbench`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from structbench.datasets.canonical import n_valid_frames, von_mises_from_voigt

# --- Physical constants -------------------------------------------------
# Normalisation divisors. Chosen from the deck and the material model, not
# from dataset statistics: dataset stats move when cases are added, physical
# constants do not, and the yield bound stays expressible in normalised units.

SIGMA_Y_MAX = 422.2e6  # Pa, top of the hardening table
S_SCALE = SIGMA_Y_MAX * np.sqrt(2.0 / 3.0)  # 3.447e8 Pa, max |s| on the surface
PEEQ_SCALE = 1.5  # hardening-table knot
E_SCALE = 0.05  # J, round number above the observed per-particle max
RHO0 = 8900.0  # kg/m^3, reference density (centred, not divided)
V_SCALE = 200.0  # m/s, top of the swept band
X_SCALE = 0.1  # m, longest bar; fixed so geometries stay comparable

#: ``*MAT_ELASTIC_PLASTIC_HYDRO`` hardening curve from the deck.
#: Note ``es[1] > es[2]`` -- a 0.2 MPa non-monotonicity present in the source
#: deck, kept verbatim rather than smoothed.
HARDEN_EPS = np.array(
    [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 15, 20], float
)
HARDEN_SY = 1e6 * np.array(
    [
        199.3, 251.1, 250.9, 321.4, 344.8, 362.8, 376.6, 387.7,
        395.4, 406.4, 415.9, 416.7, 419.0, 421.1, 422.2, 422.2,
    ],
    float,
)

#: Field name -> component count. Order defines the packed state layout.
FIELDS: tuple[tuple[str, int], ...] = (
    ("x", 2),
    ("v", 2),
    ("s", 3),
    ("peeq", 1),
    ("E", 1),
    ("rho", 1),
)
STATE_DIM = sum(n for _, n in FIELDS)  # 10


@dataclass(frozen=True)
class CaseState:
    """One case as a ten-field particle state trajectory, SI units.

    Attributes
    ----------
    x, v : (T, P, 2)
        Position (m) and velocity (m/s).
    s : (T, P, 3)
        Deviatoric stress ``(s_xx, s_yy, s_xy)`` in Pa. Plane strain makes
        ``s_yz = s_zx = 0`` and ``s_zz = -(s_xx + s_yy)``, so three
        components carry the full tensor.
    peeq : (T, P)
        Effective plastic strain, dimensionless.
    E : (T, P)
        Internal energy per particle, J (total, not per unit mass).
    rho : (T, P)
        Density, kg/m^3.
    time : (T,)
        Frame times, s.
    """

    case_id: str
    x: NDArray[np.float32]
    v: NDArray[np.float32]
    s: NDArray[np.float32]
    peeq: NDArray[np.float32]
    E: NDArray[np.float32]
    rho: NDArray[np.float32]
    time: NDArray[np.float64]

    @property
    def n_frames(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_particles(self) -> int:
        return int(self.x.shape[1])


def von_mises_from_deviatoric(s: NDArray[np.floating]) -> NDArray[np.float64]:
    """``sqrt(3/2 s:s)`` from the three stored deviatoric components.

    Uses ``s_zz = -(s_xx + s_yy)`` and ``s_yz = s_zx = 0`` (plane strain), so
    ``s:s = s_xx^2 + s_yy^2 + s_zz^2 + 2 s_xy^2``.
    """
    s = np.asarray(s, float)
    s_zz = -(s[..., 0] + s[..., 1])
    ss = s[..., 0] ** 2 + s[..., 1] ** 2 + s_zz**2 + 2.0 * s[..., 2] ** 2
    return np.sqrt(1.5 * ss)


def yield_stress(peeq: NDArray[np.floating]) -> NDArray[np.float64]:
    """Flow stress ``sigma_y(peeq)`` in Pa, linearly interpolated on the deck table."""
    return np.interp(np.asarray(peeq, float), HARDEN_EPS, HARDEN_SY)


def load_state(h5_path: str | Path, *, check: bool = True) -> CaseState:
    """Load one canonical Taylor case as a ten-field state trajectory.

    Reads the eight datasets this probe needs directly rather than going
    through ``read_case``, which materialises every field in the file
    (``strain``, ``strain_rate``, ``pressure``, ``mass``, ``radius``,
    ``n_neighbors``, ``deletion``, ``acceleration``) -- ~96 MB per case
    against the ~38 MB actually used. On the OneDrive-backed data root that
    difference dominates load time.

    Parameters
    ----------
    h5_path:
        Path to a canonical ``.h5`` case.
    check:
        Run the plane-strain, tracelessness and von Mises round-trip gates.

    Raises
    ------
    AssertionError
        If ``check`` and any gate fails -- the state extraction is wrong for
        this case and downstream numbers would be meaningless.
    """
    with h5py.File(h5_path, "r") as f:
        time = np.asarray(f["response/time/t"][:], float)
        # Drops the terminal solver-output artifact frame (ADR-0028): its dt
        # is 0.077 us against a 2.006 us median.
        n = n_valid_frames(time)
        # SPH connectivity is a 0-based ROW INDEX into nodes/coords, not a
        # node id. There are 4804 nodes against 4800 particles (4 belong to a
        # dummy shell), so node arrays must never be indexed by particle
        # index directly.
        idx = f["elements/sph/connectivity"][:, 0]
        dim = 2
        coords0 = f["nodes/coords"][:, :dim][idx]
        disp = f["response/node/displacement"][:n, :, :dim][:, idx]
        v = f["response/node/velocity"][:n, :, :dim][:, idx]

        el = "response/element/sph"
        sig = np.asarray(f[f"{el}/stress"][:n], float)  # (T, P, 6) Pa, Cauchy
        peeq = np.asarray(f[f"{el}/effective_plastic_strain"][:n], float)
        energy = np.asarray(f[f"{el}/internal_energy"][:n], float)
        rho = np.asarray(f[f"{el}/density"][:n], float)
        case_id = Path(h5_path).stem

    x = coords0[None] + disp

    # sph/stress is Cauchy, not deviatoric: tr(sigma)/3 tracks -pressure.
    tr3 = (sig[..., 0] + sig[..., 1] + sig[..., 2]) / 3.0
    s = np.stack([sig[..., 0] - tr3, sig[..., 1] - tr3, sig[..., 3]], axis=-1)

    if check:
        _check_extraction(sig, tr3, s)

    return CaseState(
        case_id=case_id,
        x=x.astype(np.float32),
        v=v.astype(np.float32),
        s=s.astype(np.float32),
        peeq=peeq.astype(np.float32),
        E=energy.astype(np.float32),
        rho=rho.astype(np.float32),
        time=time[:n],
    )


def _check_extraction(
    sig: NDArray[np.floating], tr3: NDArray[np.floating], s: NDArray[np.floating]
) -> None:
    """Plane-strain, tracelessness and von Mises round-trip gates."""
    scale = float(np.abs(sig).max()) or 1.0

    assert np.abs(sig[..., 4]).max() == 0.0, "s_yz != 0: not plane strain"
    assert np.abs(sig[..., 5]).max() == 0.0, "s_zx != 0: not plane strain"

    s_zz = sig[..., 2] - tr3
    trace = np.abs(s[..., 0] + s[..., 1] + s_zz).max()
    assert trace / scale < 1e-6, f"deviator not traceless: {trace:.3e} Pa"

    vm_ref = von_mises_from_voigt(sig)
    vm_new = von_mises_from_deviatoric(s)
    denom = np.linalg.norm(vm_ref) or 1.0
    rel = float(np.linalg.norm(vm_new - vm_ref) / denom)
    assert rel < 1e-12, f"von Mises round-trip failed: rel {rel:.3e}"


def normalise(st: CaseState) -> dict[str, NDArray[np.float32]]:
    """Map a state to O(1) training units using the physical constants above.

    ``rho`` is centred on ``RHO0`` -- a raw divide puts every value near 1.0
    and leaves nothing for the network to learn.
    """
    return {
        "x": (st.x / X_SCALE).astype(np.float32),
        "v": (st.v / V_SCALE).astype(np.float32),
        "s": (st.s / S_SCALE).astype(np.float32),
        "peeq": (st.peeq / PEEQ_SCALE).astype(np.float32),
        "E": (st.E / E_SCALE).astype(np.float32),
        "rho": ((st.rho - RHO0) / RHO0).astype(np.float32),
    }


#: Divisor per field, for converting normalised errors back to SI.
UNIT_SCALE: dict[str, float] = {
    "x": X_SCALE,
    "v": V_SCALE,
    "s": S_SCALE,
    "peeq": PEEQ_SCALE,
    "E": E_SCALE,
    "rho": RHO0,
}
