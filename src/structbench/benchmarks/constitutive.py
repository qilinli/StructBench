"""The constitutive law a benchmark declares per case (ADR-0075 clause 3).

A two-site benchmark whose admissible arm derives the element state by the
material class's return map supplies the case's own law through
``BenchmarkSpec.constitutive_law``: a callable from case id to a
:class:`ConstitutiveLaw`. The law is an input of the case, so a checkpoint
stores no table (the ADR-0064 one-table hook is the per-benchmark form this
replaces for element-site benchmarks).

Units follow the working frame: moduli and yield stresses in the aux stress
unit (MPa), plastic strain dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Element formulations a two-site benchmark may declare
#: (``BenchmarkSpec.element_formulation``); the decoder's kinematics
#: (``models.common.two_site``) support exactly these.
ELEMENT_FORMULATIONS: tuple[str, ...] = ("axisymmetric", "plane_strain")

#: The state-channel layout the law-reading decoders consume: the four stress
#: components of a 2-D continuum, then the equivalent plastic strain.
RETURN_MAP_STATE_CHANNELS: tuple[str, ...] = (
    "stress_axisymmetric",
    "effective_plastic_strain",
)

#: Valid values of ``TransolverConfig.two_site_decoder`` (the model-side
#: vocabulary ``models.common.two_site.TWO_SITE_DECODERS`` is the same tuple).
TWO_SITE_DECODERS: tuple[str, ...] = (
    "off",
    "return_map",
    "corner_mean",
    "admissible_coords",
    "projected_trial",
)

#: The decoders that read the case's law (``BenchmarkSpec.constitutive_law``)
#: and so need the :data:`RETURN_MAP_STATE_CHANNELS` layout.
LAW_DECODERS: tuple[str, ...] = ("return_map", "admissible_coords", "projected_trial")


@dataclass(frozen=True)
class ConstitutiveLaw:
    """Isotropic J2 plasticity with tabulated isotropic hardening.

    Parameters
    ----------
    youngs_modulus:
        Young's modulus in the working stress unit.
    poisson_ratio:
        Poisson's ratio, in ``[0, 0.5)``.
    hardening:
        ``(peeq_knots, sigma_y_knots)``: yield stress against equivalent
        plastic strain, at least two knots, knots strictly increasing,
        ``np.interp`` end-clamped semantics (ADR-0064). A linear law is two
        knots; put the second far beyond the plastic strains the data reach
        so the clamp never binds.
    """

    youngs_modulus: float
    poisson_ratio: float
    hardening: tuple[tuple[float, ...], tuple[float, ...]]

    def __post_init__(self) -> None:
        if not self.youngs_modulus > 0:
            raise ValueError(f"youngs_modulus must be > 0, got {self.youngs_modulus}")
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError(
                f"poisson_ratio must be in [0, 0.5), got {self.poisson_ratio}"
            )
        peeq, sy = self.hardening
        if len(peeq) < 2 or len(peeq) != len(sy):
            raise ValueError(
                "hardening needs >= 2 (peeq, sigma_y) knot pairs of equal length; "
                f"got {len(peeq)}/{len(sy)}"
            )
        if any(b <= a for a, b in zip(peeq[:-1], peeq[1:], strict=True)):
            raise ValueError("hardening peeq knots must be strictly increasing")
        if any(v <= 0 for v in sy):
            raise ValueError("hardening sigma_y knots must be > 0")
        object.__setattr__(
            self,
            "hardening",
            (tuple(float(v) for v in peeq), tuple(float(v) for v in sy)),
        )

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def bulk_modulus(self) -> float:
        return self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))

    @property
    def n_knots(self) -> int:
        return len(self.hardening[0])
