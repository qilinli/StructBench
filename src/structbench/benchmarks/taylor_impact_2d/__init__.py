"""v0.1 Taylor 2D impact benchmark (ADR-0019)."""

from ..registry import BenchmarkSpec
from ..results import BaselineResult
from .benchmark import (
    ALL_BENCHMARK_CASES,
    AUX_FIELD,
    HELD_ASIDE,
    QOIS,
    TEST_EXTRAP,
    TEST_INTERP,
    TRAIN,
    VAL,
    WALL_NODE_SPACING_MM,
    WALL_NODE_TYPE,
    WALL_SPAN_MM,
    WALL_X_MM,
    native_mesh_transform,
    wall_distance_feature,
)
from .card import CARD

#: Official baseline results (ADR-0033). Transcribed from the ``mean`` blocks of
#: the blessed run's held-out ``metrics-{test_interp,test_extrap}.json`` at 4
#: significant figures; full precision, per-case numbers and the seed spread stay
#: in the run directory. ``val`` selects the checkpoint, so it is not a number to
#: beat and is omitted here.
RESULTS: tuple[BaselineResult, ...] = (
    BaselineResult(
        family="cgn",
        label="CGN",
        scheme="autoregressive",
        run_commit="7be9d4b",
        run_date="2026-07-08",
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 0.1299,
                "rollout_rel_l2_aux": 0.3223,
                "rollout_pos_rmse_mm": 1.274,
                "rollout_vm_rmse_mpa": 52.57,
                "one_step_pos_rmse_mm": 0.003244,
                "one_step_vm_rmse_mpa": 36.09,
                "qoi_final_length_mae_mm": 3.083,
                "qoi_mushroom_width_mae_mm": 4.754,
                "qoi_peak_vm_mae_mpa": 2.865,
                "qoi_t_peak_vm_mae_ms": 0.003993,
            },
            "test_extrap": {
                "rollout_rel_l2_disp": 0.5547,
                "rollout_rel_l2_aux": 0.4531,
                "rollout_pos_rmse_mm": 7.645,
                "rollout_vm_rmse_mpa": 79.46,
                "one_step_pos_rmse_mm": 0.004649,
                "one_step_vm_rmse_mpa": 40.43,
                "qoi_final_length_mae_mm": 3.198,
                "qoi_mushroom_width_mae_mm": 11.59,
                "qoi_peak_vm_mae_mpa": 19.21,
                "qoi_t_peak_vm_mae_ms": 0.2293,
            },
        },
        checkpoint="models/taylor_impact_2d/cgn-7be9d4b/model-best-096000.pt",
        checkpoint_sha256=(
            "4950d798a774be111d7096a16a112d38385d44e7832c7577bb1fa82eb7b76060"
        ),
        notes=(
            "Single-scale CGN (ADR-0034) on the ADR-0028 recipe at 100k steps, "
            "seed 1 of the s0-s3 fleet; val-selected checkpoint "
            "model-best-096000.pt (96k), one A100-80GB, ~22.4 h. s1 is the best "
            "von Mises seed (lowest rollout aux RMSE on val and test_interp) and "
            "the seed behind the published qualitative rollouts; on rollout "
            "position it is the best of four on test_interp and the most "
            "conservative (highest) on test_extrap. Extrapolation to 200 m/s is "
            "the benchmark's honest failure mode: rollout position degrades ~6x "
            "against test_interp. Relative L2 (rollout_rel_l2_disp/aux) is the "
            "pooled space+time headline (ADR-0055), added 2026-08-16 from a "
            "re-eval on this checkpoint; RMSE reproduced to <1%, so the blessed "
            "RMSE/QoI values above are unchanged."
        ),
    ),
    BaselineResult(
        family="transolver",
        label="Transolver",
        scheme="time-conditioned",
        provisional=True,
        run_commit="59d5786",
        run_date="2026-08-16",
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 0.009383,
                "rollout_rel_l2_aux": 0.1749,
                "rollout_pos_rmse_mm": 0.1043,
                "rollout_vm_rmse_mpa": 28.56,
                "qoi_final_length_mae_mm": 0.5557,
                "qoi_mushroom_width_mae_mm": 0.3532,
                "qoi_peak_vm_mae_mpa": 2.164,
                "qoi_t_peak_vm_mae_ms": 0.001986,
            },
            "test_extrap": {
                "rollout_rel_l2_disp": 0.01637,
                "rollout_rel_l2_aux": 0.1982,
                "rollout_pos_rmse_mm": 0.2344,
                "rollout_vm_rmse_mpa": 35.23,
                "qoi_final_length_mae_mm": 1.810,
                "qoi_mushroom_width_mae_mm": 3.148,
                "qoi_peak_vm_mae_mpa": 2.297,
                "qoi_t_peak_vm_mae_ms": 0.01064,
            },
        },
        notes=(
            "Native time-conditioned Transolver (ADR-0054): each frame is "
            "predicted independently from the reference geometry + normalized "
            "query time + impact-velocity scalar, history-free with no rollout "
            "accumulation, so one-step is N/A (the faithful thuml structural "
            "scheme, not autoregressive). Seed 2 of the s1-s2 pair, val-selected "
            "model-best-100000.pt, run taylor-transolver-tc-s2. PROVISIONAL "
            "(ADR-0044/0045): a best-effort native implementation, not validated "
            "against a published Taylor-Transolver number. On Taylor it is the "
            "strongest baseline by a wide margin (~14x lower displacement "
            "relative L2 than CGN on test_interp) because avoiding rollout "
            "accumulation dominates on this smooth-kinematics task. Pooled "
            "relative L2 headline (ADR-0055) from the 2026-08-16 re-eval."
        ),
    ),
    BaselineResult(
        family="mgn",
        label="MGN",
        scheme="autoregressive",
        provisional=True,
        run_commit="d838606",
        run_date="2026-08-16",
        metrics={
            "test_interp": {
                "rollout_rel_l2_disp": 0.2157,
                "rollout_rel_l2_aux": 0.3456,
                "rollout_pos_rmse_mm": 2.272,
                "rollout_vm_rmse_mpa": 55.79,
                "one_step_pos_rmse_mm": 0.004279,
                "one_step_vm_rmse_mpa": 31.32,
                "qoi_final_length_mae_mm": 9.108,
                "qoi_mushroom_width_mae_mm": 6.097,
                "qoi_peak_vm_mae_mpa": 3.639,
                "qoi_t_peak_vm_mae_ms": 0.001335,
            },
            "test_extrap": {
                "rollout_rel_l2_disp": 0.3233,
                "rollout_rel_l2_aux": 0.4327,
                "rollout_pos_rmse_mm": 4.313,
                "rollout_vm_rmse_mpa": 75.08,
                "one_step_pos_rmse_mm": 0.005979,
                "one_step_vm_rmse_mpa": 36.08,
                "qoi_final_length_mae_mm": 14.06,
                "qoi_mushroom_width_mae_mm": 6.482,
                "qoi_peak_vm_mae_mpa": 6.766,
                "qoi_t_peak_vm_mae_ms": 0.1087,
            },
        },
        notes=(
            "Native MeshGraphNets (ADR-0047 baseline, ADR-0049 repair): "
            "autoregressive next-step on the SPH particle set as a mesh, "
            "val-selected model-best-066000.pt, seed 1 of the s1-s2 pair, run "
            "taylor-mgn-base-s1. PROVISIONAL (ADR-0044/0045): not validated "
            "against a published Taylor-MGN number. On Taylor it TRAILS the CGN "
            "baseline (rollout position 2.27 vs 1.27 mm on test_interp; "
            "displacement relative L2 0.22 vs 0.13) - the known MGN-on-Taylor "
            "difficulty (large mesh stretch, noise-scale sensitivity; ADR-0047). "
            "Pooled relative L2 headline (ADR-0055) from the 2026-08-16 re-eval."
        ),
    ),
)


def _impact_velocity(case_id: str) -> float:
    """Impact speed (m/s) from a Taylor case id ``T-20-<geom>-<velocity>``."""
    return float(case_id.rsplit("-", 1)[1])


SPEC = BenchmarkSpec(
    card=CARD,
    results=RESULTS,
    splits={
        "train": tuple(TRAIN),
        "val": tuple(VAL),
        "test_interp": tuple(TEST_INTERP),
        "test_extrap": tuple(TEST_EXTRAP),
    },
    eval_splits=("val", "test_interp", "test_extrap"),
    aux_field=AUX_FIELD,
    qois=dict(QOIS),
    boundary_feature_fn=wall_distance_feature,
    dataset_id="2D-Copper-Bar-Taylor-Impact",
    kinematic_types=(WALL_NODE_TYPE,),
    mesh_transform=native_mesh_transform,
    scripted_types=(WALL_NODE_TYPE,),
    loading_scalar=_impact_velocity,
)

__all__ = [
    "TRAIN",
    "VAL",
    "TEST_INTERP",
    "TEST_EXTRAP",
    "HELD_ASIDE",
    "ALL_BENCHMARK_CASES",
    "AUX_FIELD",
    "QOIS",
    "WALL_NODE_SPACING_MM",
    "WALL_NODE_TYPE",
    "WALL_SPAN_MM",
    "WALL_X_MM",
    "native_mesh_transform",
    "wall_distance_feature",
    "CARD",
    "SPEC",
]
