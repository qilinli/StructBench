"""Benchmark card for the Taylor 2D impact benchmark (ADR-0027)."""

from ..card import BenchmarkCard, BenchmarkFigure
from .benchmark import AUX_FIELD, QOIS, TEST_EXTRAP, TEST_INTERP, TRAIN, VAL

# Landing-page narrative (ADR-0036). Non-derivable prose; the structured facts,
# splits and baseline numbers are rendered from the card + results registry.
_OVERVIEW = """\
## The problem

A copper bar strikes a rigid wall head-on and *mushrooms*: the impact face
spreads outward while a plastic wave runs back up the bar. The **Taylor impact
test** is a classic high-strain-rate experiment for calibrating elasto-plastic
material models, and it makes a demanding learned-surrogate target — large
plastic deformation, a travelling stress front, strain-hardening flow with a
pressure–volume equation of state, and a moving contact boundary, all inside
~300 microseconds.

StructBench ships the 2D SPH version: an LS-DYNA `*MAT_ELASTIC_PLASTIC_HYDRO`
+ `*EOS_GRUNEISEN` copper bar, 20 mm wide and 60 / 80 / 100 mm long, fired at
a rigid wall at 100–200 m/s. The task is an **autoregressive next-step
surrogate** — from a short ground-truth prefix the model advances the particle
state one output step at a time to the end of the trajectory, predicting both
position and the per-particle von Mises stress.

## Interpolation vs. extrapolation

The split varies **only the impact velocity** across the three fixed
geometries, cleanly separating the two regimes a surrogate should be judged
on: `test_interp` (130 / 170 m/s) sits inside the training band, while
`test_extrap` (200 m/s) sits beyond it; `val` (150 m/s) only picks each run's
checkpoint. Everything is scored in physical units — position RMSE in mm, the
von Mises field in MPa — and four quantities of interest read the engineering
outcome directly: final bar length, mushroom width, and the peak mean von
Mises stress with its timing. The reference CGN baseline is strong in
interpolation and degrades honestly at 200 m/s; the numbers are below."""

_FIGURES = (
    BenchmarkFigure(
        path="assets/taylor_rollout.gif",
        caption=(
            "Ground-truth LS-DYNA SPH rollout: a 20x80 mm copper bar "
            "mushrooming against the wall at 200 m/s, coloured by von Mises "
            "stress."
        ),
        alt="Animation of a copper bar mushrooming against a rigid wall.",
    ),
    BenchmarkFigure(
        path="assets/taylor_vms_val_150.png",
        caption=(
            "In-distribution (val, 150 m/s): CGN prediction (bottom) vs "
            "ground truth (top), von Mises stress at 12 / 108 / 204 / 300 us. "
            "The mushroom head, bar shortening, and impact-face stress band "
            "are reproduced."
        ),
        alt="Prediction-vs-truth von Mises snapshots at 150 m/s.",
    ),
    BenchmarkFigure(
        path="assets/taylor_vms_extrap_200.png",
        caption=(
            "Extrapolation (test_extrap, 200 m/s): the same comparison. The "
            "model under-flares the mushroom rim and smears the localized "
            "high-stress band — the visible face of the ~6x rollout-position "
            "degradation beyond the training range."
        ),
        alt="Prediction-vs-truth von Mises snapshots at 200 m/s showing degradation.",
    ),
    BenchmarkFigure(
        path="assets/taylor_rollout_error_vs_time.png",
        caption=(
            "Rollout error vs time for the four training seeds: position RMSE "
            "(top) and von Mises RMSE (bottom) across val / interp / extrap. "
            "Position error accumulates monotonically; the von Mises error "
            "spikes at first wall contact (~20-40 us) then re-grows. "
            "Extrapolation is where it blows up."
        ),
        alt="Line charts of rollout position and von Mises error over time.",
    ),
)

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
    overview=_OVERVIEW,
    figures=_FIGURES,
)
