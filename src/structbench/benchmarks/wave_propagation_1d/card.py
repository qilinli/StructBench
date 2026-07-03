"""Benchmark card for the wave-1d benchmark (ADR-0027)."""

from ..card import BenchmarkCard
from .benchmark import AUX_FIELD, QOIS, TEST_INTERP, TRAIN, VAL

CARD = BenchmarkCard(
    name="Wave1D-Propagation",
    version="0.1",
    description=(
        "Autoregressive next-step surrogate of an elastic stress wave in a "
        "2D SPH bar strip under initial-velocity excitation (ADR-0025). "
        "Entry tier: onboarding, tutorial, and fast CI."
    ),
    provenance=(
        "LS-DYNA parametric sweep (4 bar lengths x 4 initial velocities) "
        "produced by Curtin collaborators; benchmark protocol per ADR-0025."
    ),
    data_license="CC BY 4.0",
    solver="LS-DYNA",
    discretisation="SPH",
    materials=("*MAT_ELASTIC",),
    erosion=False,
    loading="initial velocity 1-8 mm/ms; elastic wave propagation",
    source_units="g-mm-ms",
    geometry="2D strip, 5 particle rows, {200, 300, 400, 500} mm x 8 mm",
    n_cases=len(TRAIN) + len(VAL) + len(TEST_INTERP),
    splits={
        "train": len(TRAIN),
        "val": len(VAL),
        "test_interp": len(TEST_INTERP),
    },
    task="autoregressive transition (ADR-0025)",
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
    particles_per_case="500-1250",
    n_frames=302,
    output_dt_ms=0.1,
)
