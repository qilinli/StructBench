"""Benchmark card for the notch-beam impact benchmark (ADR-0027)."""

from ..card import BenchmarkCard
from .benchmark import AUX_FIELD, PROBE, QOIS, TEST_INTERP, TRAIN, VAL

# PROVISIONAL — particles_per_case is based on 320-span impact cases (4264)
# only; 640-span cases may have higher counts. Task 10 will rescan
# h5_canonical after all 221 cases are converted and update this range.
CARD = BenchmarkCard(
    name="NotchBeam2D-Impact",
    version="0.1",
    description=(
        "Autoregressive next-step surrogate of a 2D SPH notched concrete beam "
        "under drop-weight impact (ADR-0026). "
        "Covers 3 spans, 3 impactor shapes, 3 notch positions, and 4 velocities."
    ),
    provenance=(
        "LS-DYNA parametric sweep (3 spans x 3 shapes x 3 notches x 4 velocities) "
        "produced by Curtin collaborators; benchmark protocol per ADR-0026."
    ),
    data_license="CC BY 4.0",
    solver="LS-DYNA",
    discretisation="SPH",
    materials=(
        "*MAT_CONCRETE_DAMAGE_REL3 (K&C; scaled density 2.4e-6 g/mm3)",
        "*MAT_PLASTIC_KINEMATIC",
    ),
    erosion=False,
    loading=(
        "drop-weight impact, initial velocity 40-160 mm/s,"
        " impactor shapes Bullet/Rectangular/Sphere"
    ),
    source_units="g-mm-ms",
    geometry="2D SPH notched beam, H80 x span {320,480,640} mm",
    n_cases=len(TRAIN) + len(VAL) + len(TEST_INTERP) + len(PROBE),
    splits={
        "train": len(TRAIN),
        "val": len(VAL),
        "test_interp": len(TEST_INTERP),
        "probe": len(PROBE),
    },
    task="autoregressive transition (ADR-0026)",
    aux_field=AUX_FIELD,
    aux_unit="-",
    qois=tuple(QOIS),
    fields=(
        "positions",
        "velocity",
        "acceleration",
        "stress",
        "effective_plastic_strain",
    ),
    # PROVISIONAL; Task 10 will correct after full batch
    particles_per_case="4264-4264",
    n_frames=502,
    output_dt_ms=1.0,
)
