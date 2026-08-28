"""Wave-1d benchmark: entry-tier elastic wave propagation (ADR-0025)."""

from ..registry import BenchmarkSpec
from ..results import BaselineResult
from .benchmark import (
    ALL_BENCHMARK_CASES,
    AUX_FIELD,
    QOIS,
    TEST_INTERP,
    TRAIN,
    VAL,
    _initial_velocity,
    native_mesh_transform,
)
from .card import CARD

__all__ = [
    "ALL_BENCHMARK_CASES",
    "AUX_FIELD",
    "CARD",
    "QOIS",
    "SPEC",
    "TEST_INTERP",
    "TRAIN",
    "VAL",
    "native_mesh_transform",
]

#: Official baseline results (ADR-0033). Transcribed from the ``mean`` block of
#: each run's held-out ``metrics-test_interp.json`` at 4 significant figures
#: (QoI MAEs from the same file's per-case signed errors); full precision,
#: per-case numbers and the fleet spread stay in the run directory. ``val``
#: selects the checkpoint, so it is not a number to beat and is omitted here.
#:
#: SCHEME MATRIX EXTENSION (2026-08-28, maintainer-directed in-session): like
#: the other three benchmarks, wave now tables one row per family x prediction
#: scheme, from the 2026-08-28 multi-method fleet (commit e6482f2, jobs
#: 63076010-17, 50k steps each at the CGN-matched budget). The autoregressive
#: MGN/Transolver/GeoFLARE rows are DELIBERATE NEGATIVE RESULTS - every
#: mesh-native AR family fails on wave (displacement relative L2 > 1, worse
#: than predicting zero) with healthy one-step errors and converged train
#: loss, i.e. rollout instability under sustained oscillation, not a recipe
#: artifact (the arms carry the full ADR-0049 repair, and a low-noise
#: Transolver sibling, wave-transolver-ar-n02-s1, fails identically). They
#: are tabled so the leaderboard shows the scheme x regime interaction
#: directly: time-conditioning is the only operator scheme that survives
#: this benchmark's oscillatory physics, and CGN is the only stable
#: autoregressive family.
RESULTS: tuple[BaselineResult, ...] = (
    BaselineResult(
        family="cgn",
        label="CGN",
        scheme="autoregressive",
        reference=(
            "Li, Q., Wang, Z., Li, L., Hao, H., Chen, W., & Shao, Y. (2023). "
            "Machine learning prediction of structural dynamic responses using "
            "graph neural networks. *Computers & Structures*, 289, 107188. "
            "https://doi.org/10.1016/j.compstruc.2023.107188"
        ),
        run_commit="48046ea",
        run_date="2026-07-10",
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 0.3507,
                "rollout_rel_l2_aux": 0.9025,
                "rollout_pos_rmse_mm": 0.8750,
                "rollout_axial_rmse_mpa": 0.1676,
                "one_step_pos_rmse_mm": 0.004882,
                "one_step_axial_rmse_mpa": 0.01547,
                "qoi_arrival_time_25_mae_ms": 0.1007,
                "qoi_arrival_time_50_mae_ms": 0.05045,
                "qoi_arrival_time_75_mae_ms": 0.1006,
                "qoi_peak_stress_mae_mpa": 0.9665,
            },
        },
        checkpoint="models/wave_propagation_1d/cgn-48046ea/model-best-050000.pt",
        checkpoint_sha256=(
            "2139335fb0cb2f6cccaf9be69e69cced369deb6eda80f6970b292deeba07dc0a"
        ),
        notes=(
            "Single-scale CGN (ADR-0034) on the round-2 capacity recipe "
            "(hidden 128 / 10 MP steps / 2-layer node MLP, noise_std 0.06) at "
            "50k steps, batch 32; seed 1 of the X1 arm (seeds 1-2) of the "
            "2026-07-10 17-run recipe fleet, val-selected checkpoint "
            "model-best-050000.pt (50k), one A100-80GB, ~3.9 h. The winning "
            "arm beats the shipped-config control (64/5/1, noise 0.02) by "
            "~2-3x on both rollout channels at half the step budget; blessed "
            "from the round-2 winner on maintainer instruction without the "
            "pre-declared 4-seed confirmation fleet. Caveats: test_interp is "
            "a 2-case split; rollout RMSE is dominated by the final ~5 ms of "
            "the 30 ms horizon; the pointwise-max peak_stress QoI "
            "overshoots in both held-out cases (pred 1.738/1.481 MPa vs true "
            "0.860/0.426 MPa) - arrival-time QoIs are the trustworthy wave "
            "quantities (all within ~1 output frame). Relative L2 "
            "(rollout_rel_l2_disp/aux) is the pooled space+time headline "
            "(ADR-0055), added 2026-08-16 from a re-eval on this checkpoint; "
            "RMSE reproduced to <1%, so the blessed RMSE/QoI values are "
            "unchanged. Standing after the 2026-08-28 multi-method fleet: the "
            "ONLY autoregressive family that is stable on wave (every "
            "mesh-native AR arm lands at relative L2 > 1) - the "
            "relative-displacement particle-graph formulation survives the "
            "sustained oscillation that breaks the mesh-native adaptations - "
            "though both time-conditioned operators beat it ~3x on the fields."
        ),
    ),
    BaselineResult(
        family="mgn",
        label="MGN",
        scheme="autoregressive",
        reference=(
            "Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. W. "
            "(2021). Learning Mesh-Based Simulation with Graph Networks. *ICLR*. "
            "https://arxiv.org/abs/2010.03409"
        ),
        provisional=True,
        run_commit="e6482f2",
        run_date="2026-08-28",
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 1.566,
                "rollout_rel_l2_aux": 2.079,
                "rollout_pos_rmse_mm": 5.363,
                "rollout_axial_rmse_mpa": 0.5561,
                "one_step_pos_rmse_mm": 0.01385,
                "one_step_axial_rmse_mpa": 0.05301,
                "qoi_arrival_time_25_mae_ms": 1.200,
                "qoi_arrival_time_50_mae_ms": 0.7055,
                "qoi_arrival_time_75_mae_ms": 0.2518,
                "qoi_peak_stress_mae_mpa": 0.9974,
            },
        },
        notes=(
            "Native MeshGraphNets, DELIBERATE NEGATIVE ROW (see module "
            "header): FAILS on wave - displacement relative L2 1.566 (> 1, "
            "worse than predicting zero), stress 2.079. Not a recipe gap: the "
            "arm carries the full ADR-0049 repair (velocity history 5, "
            "working-frame noise 0.06 = the blessed wave CGN value, stretch "
            "gate off, world-edge radius below the lattice spacing so world "
            "edges act only as a contact detector), one-step is healthy "
            "(0.0139 mm) and train loss converges, but the val rollout "
            "oscillated 5-80 mm all training long - rollout instability under "
            "the 30 ms sustained reverberation, the regime the decaying "
            "impact transients (taylor/notch) never enter. Run wave-mgn-s1, "
            "50k steps, val-selected model-best-038000.pt, single seed. "
            "PROVISIONAL (ADR-0044/0046). Pooled relative L2 headline "
            "(ADR-0055)."
        ),
    ),
    BaselineResult(
        family="transolver",
        label="Transolver",
        scheme="autoregressive",
        reference=(
            "Wu, H., Luo, H., Wang, H., Wang, J., & Long, M. (2024). Transolver: "
            "A Fast Transformer Solver for PDEs on General Geometries. *ICML*. "
            "https://arxiv.org/abs/2402.02366"
        ),
        provisional=True,
        run_commit="e6482f2",
        run_date="2026-08-28",
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 1.261,
                "rollout_rel_l2_aux": 1.013,
                "rollout_pos_rmse_mm": 4.180,
                "rollout_axial_rmse_mpa": 0.2819,
                "one_step_pos_rmse_mm": 0.01329,
                "one_step_axial_rmse_mpa": 0.03836,
                "qoi_arrival_time_25_mae_ms": 0.4035,
                "qoi_arrival_time_50_mae_ms": 1.251,
                "qoi_arrival_time_75_mae_ms": 1.351,
                "qoi_peak_stress_mae_mpa": 0.5225,
            },
        },
        notes=(
            "Native autoregressive Transolver, DELIBERATE NEGATIVE ROW (see "
            "module header): FAILS on wave - displacement relative L2 1.261 "
            "(> 1), stress 1.013 - reproducing the 2026-08-21 diverged "
            "control (1.26/1.01) whose artifacts were lost with the EMI26 "
            "worktree. The failure is INTRINSIC, not a noise-transfer "
            "artifact: the low-noise sibling arm wave-transolver-ar-n02-s1 "
            "(noise 0.02, the taylor-validated transolver-AR value, vs this "
            "arm's 0.06 CGN transfer) fails identically (1.395, best "
            "checkpoint already at step 2k). One-step is healthy (0.0133 mm); "
            "the same repaired recipe (ADR-0049 velocity history + noise) is "
            "registered and functional on taylor (0.0213) and notch (0.0641) "
            "- wave's sustained oscillation is the one regime where errors "
            "recirculate instead of decaying. Run wave-transolver-ar-s1, 50k "
            "steps, val-selected model-best-030000.pt, single seed. "
            "PROVISIONAL (ADR-0044/0046). Pooled relative L2 headline "
            "(ADR-0055)."
        ),
    ),
    BaselineResult(
        family="transolver",
        label="Transolver",
        scheme="time-conditioned",
        reference=(
            "Wu, H., Luo, H., Wang, H., Wang, J., & Long, M. (2024). Transolver: "
            "A Fast Transformer Solver for PDEs on General Geometries. *ICML*. "
            "https://arxiv.org/abs/2402.02366"
        ),
        provisional=True,
        run_commit="e6482f2",
        run_date="2026-08-28",
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 0.1133,
                "rollout_rel_l2_aux": 0.2120,
                "rollout_pos_rmse_mm": 0.3990,
                "rollout_axial_rmse_mpa": 0.06464,
                "qoi_arrival_time_25_mae_ms": 0.1512,
                "qoi_arrival_time_50_mae_ms": 0.0906,
                "qoi_arrival_time_75_mae_ms": 0.1006,
                "qoi_peak_stress_mae_mpa": 0.09222,
            },
        },
        notes=(
            "Native time-conditioned Transolver (ADR-0054): history-free, each "
            "frame predicted independently from the rest lattice + normalized "
            "query time + initial-velocity scalar (ADR-0051 B), no rollout "
            "accumulation, so one-step is N/A. The STRONGEST wave baseline: "
            "beats the blessed CGN ~3.1x on displacement relative L2 (0.113 "
            "vs 0.351), ~4.3x on the stress field (0.212 vs 0.903) and ~10x "
            "on the peak-stress QoI (0.092 vs 0.966 MPa - the QoI the CGN "
            "blessing flags as its overshoot); arrival times sit at the "
            "0.05 ms frame-tick floor for both (2-3 ticks vs CGN's 1-2). "
            "Because it never rolls out, it survives the oscillatory regime "
            "that fails every mesh-native autoregressive family (see module "
            "header). Re-establishes with registry-grade artifacts the "
            "2026-08-21 EMI26 result whose runs were lost (F-006). Seed 1 of "
            "the s1-s2 pair, val-selected (val relative L2 0.0973 vs seed "
            "2's 0.1238; seed 2 test displacement 0.1342), run "
            "wave-transolver-timecond-s1, 50k steps, model-best-042000.pt, "
            "~25 min A100. PROVISIONAL (ADR-0044/0046): not validated "
            "against a published wave-Transolver number; 2-case test split "
            "(benchmark caveat). Pooled relative L2 headline (ADR-0055)."
        ),
    ),
    BaselineResult(
        family="geoflare",
        label="GeoFLARE",
        scheme="autoregressive",
        reference=(
            "Adams, R., et al. (NVIDIA). GeoTransolver. arXiv:2512.20399; with "
            "Puri, R., et al. FLARE: Fast Low-rank Attention Routing Engine. "
            "arXiv:2508.12594. GeoFLARE is GeoTransolver with the FLARE attention "
            "backend (attention_type GALE_FA; ADR-0045)."
        ),
        provisional=True,
        run_commit="e6482f2",
        run_date="2026-08-28",
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 1.071,
                "rollout_rel_l2_aux": 0.9798,
                "rollout_pos_rmse_mm": 3.798,
                "rollout_axial_rmse_mpa": 0.2781,
                "one_step_pos_rmse_mm": 0.01854,
                "one_step_axial_rmse_mpa": 0.04536,
                "qoi_arrival_time_25_mae_ms": 1.199,
                "qoi_arrival_time_50_mae_ms": 0.7349,
                "qoi_arrival_time_75_mae_ms": 0.1510,
                "qoi_peak_stress_mae_mpa": 0.6025,
            },
        },
        notes=(
            "Native autoregressive GeoFLARE, DELIBERATE NEGATIVE ROW (see "
            "module header): FAILS on wave - displacement relative L2 1.071 "
            "(> 1), the third mesh-native autoregressive family to fail on "
            "this benchmark (least badly of the three). One-step is healthy "
            "(0.0185 mm); the rollout is unstable under the 30 ms sustained "
            "reverberation. Same repaired recipe as the taylor GeoFLARE arm "
            "(ADR-0049 velocity history 5 + the blessed wave CGN noise 0.06). "
            "Run wave-geoflare-ar-s1, 50k steps, val-selected "
            "model-best-032000.pt, single seed. PROVISIONAL (ADR-0045/0046). "
            "Pooled relative L2 headline (ADR-0055)."
        ),
    ),
    BaselineResult(
        family="geoflare",
        label="GeoFLARE",
        scheme="time-conditioned",
        reference=(
            "Adams, R., et al. (NVIDIA). GeoTransolver. arXiv:2512.20399; with "
            "Puri, R., et al. FLARE: Fast Low-rank Attention Routing Engine. "
            "arXiv:2508.12594. GeoFLARE is GeoTransolver with the FLARE attention "
            "backend (attention_type GALE_FA; ADR-0045)."
        ),
        provisional=True,
        run_commit="e6482f2",
        run_date="2026-08-28",
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 0.1320,
                "rollout_rel_l2_aux": 0.3475,
                "rollout_pos_rmse_mm": 0.5007,
                "rollout_axial_rmse_mpa": 0.09531,
                "qoi_arrival_time_25_mae_ms": 0.2519,
                "qoi_arrival_time_50_mae_ms": 0.1005,
                "qoi_arrival_time_75_mae_ms": 0.3525,
                "qoi_peak_stress_mae_mpa": 0.1320,
            },
        },
        notes=(
            "Native GeoFLARE under the time-conditioned scheme (ADR-0054): "
            "history-free independent-time-query with the initial-velocity "
            "scalar (ADR-0051 B), no rollout, so one-step is N/A. Confirms "
            "the scheme effect on a second operator family: beats the blessed "
            "CGN ~2.7x on displacement relative L2 (0.132 vs 0.351), ~2.6x on "
            "the stress field and ~7x on peak stress (0.132 vs 0.966 MPa), "
            "while its autoregressive sibling above fails outright - on wave "
            "the scheme, not the architecture, is what separates success from "
            "failure. Behind the time-conditioned Transolver on every metric, "
            "as on notch and DeformingPlate. Seed 2 of the s1-s2 pair, "
            "val-selected (val relative L2 0.1290 vs seed 1's 0.1448; seed 1 "
            "test displacement 0.1512), run wave-geoflare-timecond-s2, 50k "
            "steps, model-best-048000.pt, ~36 min A100. PROVISIONAL "
            "(ADR-0045/0046): 2-case test split (benchmark caveat). Pooled "
            "relative L2 headline (ADR-0055)."
        ),
    ),
)

SPEC = BenchmarkSpec(
    card=CARD,
    results=RESULTS,
    splits={
        "train": tuple(TRAIN),
        "val": tuple(VAL),
        "test_interp": tuple(TEST_INTERP),
    },
    eval_splits=("val", "test_interp"),
    aux_field=AUX_FIELD,
    qois=dict(QOIS),
    boundary_feature_fn=None,
    dataset_id="1D-Wave-Propagation",
    # Mesh-native wiring (the ADR-0047 mechanism, extended to wave for the
    # ADR-0054 time-conditioned Transolver): the cgn path is untouched by all
    # three fields. No kinematic parts exist (the arrest BC lives inside the
    # part-1 particle set), so scripted_types must be pinned to the empty
    # tuple — the simulators' family default (1,) would script every bar
    # particle and fail the scripted-subset check against the empty
    # kinematic set.
    mesh_transform=native_mesh_transform,
    scripted_types=(),
    loading_scalar=_initial_velocity,
)
