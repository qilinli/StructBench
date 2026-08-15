"""Benchmark card for the notch-beam impact benchmark (ADR-0027)."""

from ..card import BenchmarkCard, BenchmarkFigure
from .benchmark import AUX_FIELD, PROBE, QOIS, TEST_INTERP, TRAIN, VAL

CARD = BenchmarkCard(
    name="NotchBeam2D-Impact",
    version="0.1",
    description=(
        "Autoregressive next-step surrogate of a 2D SPH notched concrete beam "
        "under drop-weight impact (ADR-0026). "
        "Covers 3 spans, 3 impactor shapes, 3 notch positions, and 4 velocities. "
        "Three bodies: the K&C concrete beam (part 1) is the predicted "
        "deformable; the steel impactor (part 2) and the two support blocks "
        "(part 3) are protocol-kinematic (ADR-0026) — driven by ground truth "
        "during rollout (both move: the impactor decelerates from its case "
        "velocity to ~10-20% on contact, the supports displace a few mm), "
        "excluded from the training loss and from position/strain metrics, "
        "with both QoIs restricted to concrete particles."
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
    horizon="frames [6, 250) of 502 scored (250 µs, ADR-0039); full-length diagnostic",
    protocol_rationale=(
        "Confirmed (maintainer, 2026-07-20): input_frames = 6 gives C = 5 "
        "input velocities (input_frames - 1), the GNS reference history "
        "length — the velocity budget is the criterion, not a rigid prefix. "
        "The timeline analysis (2026-07-20, on the DUG data copy) shows "
        "impact contact from frame 0, so the observed window takes in the "
        "first 6 us of contact; accepted. Scored horizon (ADR-0039): "
        "rollout metrics and "
        "QoIs are scored on frames [input_frames, 250) (250 µs). Internal "
        "energy reaches 99% of its final value by frame 77-213 "
        "(span-dependent); the remaining frames are ballistic separation and "
        "elastic ringing, which dominated full-horizon RMSE (half the final "
        "error accrued after frame 301 in baseline rollouts) while adding no "
        "fracture physics. The full 502-frame error curve remains a "
        "non-leaderboard long-horizon diagnostic. The cracked_fraction QoI "
        "threshold 0.01 is a declared protocol definition (ADR-0029, "
        "amended 2026-08-06): the SPH source model has no erosion or crack "
        "criterion; a 221-case sweep shows the GT fraction shifts ~0.05 "
        "mean per case across the factor-2 band [0.005, 0.02], and "
        "frame-249 vs frame-501 fractions are nearly identical (0.305 vs "
        "0.317 mean), corroborating the 250 us horizon. "
        "Probe split (characterisation, 2026-08-15): the probe cases are "
        "out-of-distribution on THREE axes at once — span (400/800 mm) and "
        "impactor velocity both outside the training grids ({320,480,640} mm; "
        "{40,80,120,160} m/s), and, decisively, an OFF-CENTRE impact. All 108 "
        "train/val/test_interp cases are struck exactly at midspan (impact "
        "offset 0.0 mm, every notch a/b/c variant and span); the probe impacts "
        "land ~6% off-centre — a loading mode absent from training entirely. "
        "Probe scores therefore measure graceful failure on a genuinely new "
        "loading configuration, not ordinary interpolation: global-attention "
        "operators mis-localise the response to the learned midspan prior, "
        "while relative-position message-passing (MGN/CGN) degrades more "
        "gracefully."
    ),
    size_gb=24.9,
    figures=(
        BenchmarkFigure(
            path="assets/notch_impact_rollout.gif",
            caption=(
                "Ground truth (top) vs CGN prediction (bottom) on held-out "
                "NB-I-640-Sphere-c-120 (test_interp): a 640 mm span beam "
                "under 120 m/s sphere impact, coloured by max principal "
                "strain (fringe capped at 0.05, 5x the 1% crack threshold). "
                "The surrogate tracks the impact wedge and beam deflection "
                "through the 250 µs scored window; the marked frames beyond "
                "it are the unscored long-horizon diagnostic, where the "
                "prediction visibly degrades."
            ),
            alt=(
                "Stacked animation of ground-truth and CGN-predicted strain "
                "fringes on a notched concrete beam under drop-weight impact."
            ),
        ),
        BenchmarkFigure(
            path="assets/notch_impact_strain_interp_640_c_120.png",
            caption=(
                "In-distribution snapshots (test_interp, 640 mm span, sphere "
                "at 120 m/s): ground truth (top) vs CGN baseline (bottom) at "
                "four scored-window times plus the beyond-horizon diagnostic "
                "frame. The model follows the central shear wedge and the "
                "deflection but diffuses the discrete flexural cracks into "
                "streaky bands and over-counts cracked fraction (0.39 vs "
                "0.29 at 250 µs) — the damage field, not the kinematics, is "
                "the open gap."
            ),
            alt=(
                "Grid of strain fringe snapshots comparing ground truth and "
                "CGN prediction at five times."
            ),
        ),
        BenchmarkFigure(
            path="assets/notch_impact_rollout_error_vs_time.png",
            caption=(
                "Per-frame rollout position RMSE for the 12 test_interp "
                "cases (gray) and their mean (blue). Error grows smoothly to "
                "~0.7 mm at the 250 µs scored horizon (dashed) and keeps "
                "growing to ~2.3 mm over the full 502-frame record — the "
                "ballistic-separation and ringing tail the ADR-0039 horizon "
                "deliberately excludes from scoring."
            ),
            alt=(
                "Line plot of rollout position error versus time for twelve "
                "test cases with the scored horizon marked at 250 "
                "microseconds."
            ),
        ),
    ),
)
