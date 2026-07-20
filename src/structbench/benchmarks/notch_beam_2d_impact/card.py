"""Benchmark card for the notch-beam impact benchmark (ADR-0027)."""

from ..card import BenchmarkCard
from .benchmark import AUX_FIELD, PROBE, QOIS, TEST_INTERP, TRAIN, VAL

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
        "*MAT_CONCRETE_DAMAGE_REL3 (K&C; density 2.4e-6 kg/mm3)",
        "*MAT_PLASTIC_KINEMATIC",
    ),
    erosion=False,
    loading=(
        "drop-weight impact, initial velocity 40-160 m/s,"
        " impactor shapes Bullet/Rectangular/Sphere"
    ),
    source_units="kg-mm-ms",
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
        "node/displacement",
        "node/velocity",
        "node/acceleration",
        "sph/stress",
        "sph/strain",
        "sph/strain_rate",
        "sph/effective_plastic_strain",
        "sph/pressure",
        "sph/density",
        "sph/internal_energy",
        "sph/mass",
        "sph/radius",
        "sph/n_neighbors",
        "sph/deletion",
        "global/kinetic_energy",
        "global/internal_energy",
        "global/total_energy",
    ),
    particles_per_case="4264-12966",
    n_frames=502,
    output_dt_ms=0.001,
    input_frames=6,
    protocol_rationale=(
        "Provisional (ADR-0035): input_frames = 6 gives C = 5 input velocities "
        "(input_frames - 1), the GNS reference history length; the mandatory "
        "GT timeline analysis has not yet run for this dataset (ingested data "
        "lives on the ingestion machine), so 6 is not yet confirmed to sit "
        "before the onset of non-rigid motion. Confirm before the first "
        "trained baseline. Scored horizon (ADR-0039): rollout metrics and "
        "QoIs are scored on frames [input_frames, 250] (250 µs). Internal "
        "energy reaches 99% of its final value by frame 77-213 "
        "(span-dependent); the remaining frames are ballistic separation and "
        "elastic ringing, which dominated full-horizon RMSE (half the final "
        "error accrued after frame 301 in baseline rollouts) while adding no "
        "fracture physics. The full 502-frame error curve remains a "
        "non-leaderboard long-horizon diagnostic."
    ),
    size_gb=24.9,
)
