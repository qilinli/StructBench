"""Benchmark card for the notch-beam impact benchmark (ADR-0027)."""

from ..card import BenchmarkCard, BenchmarkFigure
from .benchmark import AUX_FIELD, PROBE, QOIS, TEST_INTERP, TRAIN, VAL

# Landing-page narrative (ADR-0036). Non-derivable prose; the structured facts,
# splits and baseline numbers are rendered from the card + results registry.
_OVERVIEW = """\
## The problem

A steel weight drops onto a **notched concrete beam** and the beam cracks: a
shear wedge punches down under the impactor while flexural cracks open from the
notch and race up through the section. Concrete under high-rate impact is a
brutal surrogate target — a quasi-brittle material that softens as it damages,
discrete cracks that localise strain, and a fracture pattern that *is* the
scientific quantity of interest rather than a by-product of the motion.

StructBench ships the 2D SPH version: an LS-DYNA `*MAT_CONCRETE_DAMAGE_REL3`
(K&C) notched beam, 80 mm deep and 320 / 480 / 640 mm span, struck at midspan by
a drop weight (Bullet / Rectangular / Sphere) at 40–160 m/s, with no element
erosion. The task is an **autoregressive next-step surrogate** — from a short
ground-truth prefix the model advances the SPH particle state one output step at
a time, predicting both position and the per-particle **max principal strain**,
the field that carries the crack pattern.

![Schematic of the notch-beam impact setup: a drop weight above a simply-supported notched concrete beam, with the swept parameter ranges.](../../assets/problem_notch_beam_impact.png)

*Problem setup: a drop weight (three shapes) strikes the notched concrete beam
at midspan; beam height 80 mm, spans 320 / 480 / 640 mm, notch positions
a / b / c.*

## Interpolation vs. the off-centre probe

`test_interp` holds out interior combinations of span, impactor shape, notch
position and velocity — every factor level still appears in training, so it
measures ordinary interpolation. The separate **probe** set is deliberately
harder: it is out-of-distribution on *three* axes at once — a new span, an
out-of-range velocity, and, decisively, an **off-centre impact** (every
in-distribution case is struck exactly at midspan). It measures graceful failure
on a genuinely new loading mode, not interpolation — and it is where the method
ordering flips (a global-attention operator that wins in-distribution
mis-localises the response there, while relative-position message passing
degrades more gracefully). Everything is scored over the 250 µs window (ADR-0039)
in physical units — position RMSE in mm, strain RMSE — plus two quantities of
interest: peak mid-span deflection and the end-state cracked fraction. The
numbers, and the cross-method comparison, are below."""

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
    overview=_OVERVIEW,
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
            path="assets/notch_rollout_methods.gif",
            caption=(
                "Ground truth vs the three baselines (MGN, CGN, Transolver) on "
                "held-out NB-I-640-Sphere-c-120 (test_interp): a 640 mm span "
                "beam under 120 m/s sphere impact, coloured by max principal "
                "strain (fringe capped at 0.05, 5x the 1% crack threshold) over "
                "the 250 µs scored window. Transolver (time-conditioned) tracks "
                "the central shear wedge and the discrete flexural cracks most "
                "closely (rollout strain RMSE 0.004); CGN diffuses them into "
                "streaky bands (0.013); MGN is the diffusest (0.020)."
            ),
            alt=(
                "Stacked animation of ground-truth, MGN, CGN, and Transolver "
                "strain fringes on a notched concrete beam under drop-weight "
                "impact."
            ),
        ),
        BenchmarkFigure(
            path="assets/notch_strain_methods_640_c_120.png",
            caption=(
                "In-distribution max-principal-strain snapshots (test_interp, "
                "640 mm span, sphere at 120 m/s) at 12 / 72 / 132 / 192 / "
                "249 µs across the scored window: ground truth vs MGN vs CGN vs "
                "Transolver. Transolver (rollout strain RMSE 0.004) reproduces "
                "the shear wedge and the discrete flexural cracks closely, while "
                "CGN (0.013) and MGN (0.020) smear them into diffuse streaky "
                "bands — the damage field, not the kinematics, is the open gap. "
                "MGN and Transolver are provisional native baselines "
                "(ADR-0044/0045)."
            ),
            alt=(
                "Grid of strain fringe snapshots comparing ground truth, MGN, "
                "CGN, and Transolver at five times."
            ),
        ),
    ),
)
