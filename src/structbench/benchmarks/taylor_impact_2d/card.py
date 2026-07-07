"""Benchmark card for the Taylor 2D impact benchmark (ADR-0027)."""

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
    particles_per_case="4800-8000",
    n_frames=152,
    output_dt_ms=0.002,
    input_frames=6,
    protocol_rationale=(
        "input_frames = 6 gives the model C = 5 input velocities "
        "(input_frames - 1), the GNS reference history length "
        "(Sanchez-Gonzalez et al. 2023); under ADR-0035 the model observes "
        "exactly these 6 ground-truth frames to seed the rollout, with no "
        "constant-velocity history backfill. GT timeline analysis over all 33 "
        "cases (2026-07-05, python -m structbench.benchmarks.timeline; evidence "
        "table in docs/timelines/taylor_impact_2d.md): the rod is in free "
        "flight until first wall contact near frame 7, so a 6-frame observed "
        "prefix takes in 0.0% of the impact in every case, while the historical "
        "init = 11 handed models the shock onset (up to 10.6% of total KE "
        "already dissipated; nonzero in every case above 100 m/s). 99% "
        "displacement settlement lands as late as 296 us of the 300 us record "
        "and the last fifth of the horizon retains 1.6-8.1% of peak mean "
        "acceleration (elastic ringing), so the full horizon is dynamically "
        "active. n_frames = 152 counts stored frames; the working trajectory "
        "drops the terminal solver-output artifact frame (ADR-0028), giving a "
        "151-frame / 300 us protocol horizon and a scored span of frames "
        "[6, 151) -- 145 predicted frames. Predictions are scored at the native "
        "2 us output times; peak_von_mises/t_peak_von_mises (peak of the "
        "particle-mean field, e.g. 191 MPa at 44 us in T-20-80-150 ground "
        "truth) penalize temporally coarse surrogates."
    ),
    size_gb=2.4,
)
