"""Benchmark card for the Taylor 2D impact benchmark (ADR-0025)."""

from ..card import BenchmarkCard
from .benchmark import AUX_FIELD, QOIS, TEST_EXTRAP, TEST_INTERP, TRAIN, VAL

CARD = BenchmarkCard(
    name="Taylor2D-Impact",
    version="0.1",
    description=(
        "Autoregressive next-step surrogate of a 2D SPH copper bar under "
        "Taylor impact against a rigid wall (ADR-0019)."
    ),
    provenance=(
        "LS-DYNA parametric sweep (3 bar lengths x 11 impact velocities) "
        "produced by Curtin collaborators; benchmark protocol per ADR-0019. "
        "One extra Convergence run is held aside for a mesh-resolution check."
    ),
    data_license="CC BY 4.0",
    solver="LS-DYNA",
    discretisation="SPH",
    materials=("*MAT_ELASTIC_PLASTIC_HYDRO", "*EOS_GRUNEISEN"),
    erosion=False,
    loading="rigid-wall impact; initial velocity 100-200 m/s",
    source_units="g-mm-ms",
    geometry="2D bar, 20 mm x {60, 80, 100} mm",
    n_cases=len(TRAIN) + len(VAL) + len(TEST_INTERP) + len(TEST_EXTRAP),
    splits={
        "train": len(TRAIN),
        "val": len(VAL),
        "test_interp": len(TEST_INTERP),
        "test_extrap": len(TEST_EXTRAP),
    },
    task="autoregressive transition (ADR-0019)",
    aux_field=AUX_FIELD,
    aux_unit="MPa",
    qois=tuple(QOIS),
    fields=(
        "positions",
        "velocity",
        "acceleration",
        "stress",
        "effective_plastic_strain",
    ),
    particles_per_case="4804-8004",
    n_frames=152,
    output_dt_ms=0.002,
)
