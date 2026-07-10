"""Benchmark card for the wave-1d benchmark (ADR-0027)."""

from ..card import BenchmarkCard, BenchmarkFigure
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
    materials=("*MAT_ELASTIC (scaled toy constants: E=0.01 GPa, rho=2e-6 kg/mm3)",),
    erosion=False,
    loading=(
        "initial velocity 1-8 mm/ms; elastic wave propagation"
        "; wave speed ~70.7 mm/ms (~10 traversals per trajectory)"
    ),
    source_units="kg-mm-ms",
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
    particles_per_case="500-1250",
    n_frames=302,
    output_dt_ms=0.1,
    input_frames=6,
    protocol_rationale=(
        "input_frames = 6 (ADR-0035): C = 5 input velocities (input_frames - "
        "1), the GNS reference history length; the model observes exactly "
        "these 6 ground-truth frames (indices 0-5) to seed the rollout, with "
        "no constant-velocity backfill. GT timeline analysis run 2026-07-06 "
        "(docs/timelines/wave_propagation_1d.md): a 6-frame observed prefix "
        "takes in 14.8% of initial KE worst-case (3.7% at 3 frames), and at "
        "the measured front speed ~70.7 mm/ms the wave reaches the first (25%) "
        "gauge about 7 frames in -- after the observed prefix -- so the "
        "arrival_time QoI is predicted, not observed. 6 is near the ceiling "
        "for this benchmark: a larger input_frames would risk seeding past "
        "first arrival."
    ),
    size_gb=0.23,
    figures=(
        BenchmarkFigure(
            path="assets/wave_rollout.gif",
            caption=(
                "Ground truth (top) vs CGN prediction (bottom) on held-out "
                "W1D-300-4 (test_interp): a 300 mm bar at 4 mm/ms initial "
                "velocity, coloured by axial stress, y-axis exaggerated x8. "
                "The surrogate tracks the compression front, the free-end "
                "reflections, and the cycle timing over the 30 ms rollout; "
                "degradation concentrates in the final ~5 ms."
            ),
            alt=(
                "Stacked animation of ground-truth and CGN-predicted axial "
                "stress waves in a slender bar."
            ),
        ),
        BenchmarkFigure(
            path="assets/wave_axial_interp_400_2.png",
            caption=(
                "In-distribution (test_interp, 400 mm bar at 2 mm/ms): ground "
                "truth (top) vs CGN prediction (bottom), axial stress at "
                "t = 0.6 / 10.4 / 20.2 / 30.0 ms (y x8). The prediction "
                "reproduces the wavefront position and reflection cycles; "
                "late-horizon fields roughen and overshoot near the impact "
                "end (rollout position RMSE 0.95 mm)."
            ),
            alt=(
                "Prediction-vs-truth axial-stress snapshots for the 400 mm "
                "bar, in-distribution."
            ),
        ),
        BenchmarkFigure(
            path="assets/wave_rollout_error_vs_time.png",
            caption=(
                "Rollout error vs time for the CGN baseline (fleet run "
                "x1-s1): position RMSE (top) and axial-stress RMSE (bottom) "
                "for each eval case. Error is concentrated in the final ~5 ms "
                "of the 30 ms horizon; the held-out test_interp cases match "
                "the val cases (no interpolation cliff)."
            ),
            alt=(
                "Line charts of rollout position and axial-stress error over "
                "time for four cases."
            ),
        ),
    ),
)
