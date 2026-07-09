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
    WALL_X_MM,
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
        label="CGN baseline",
        run_commit="7be9d4b",
        run_date="2026-07-08",
        metrics={
            "test_interp": {
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
        notes=(
            "Single-scale CGN (ADR-0034) on the ADR-0028 recipe at 100k steps, "
            "seed 1 of the s0-s3 fleet; val-selected checkpoint "
            "model-best-096000.pt (96k), one A100-80GB, ~22.4 h. s1 is the best "
            "von Mises seed (lowest rollout aux RMSE on val and test_interp) and "
            "the seed behind the published qualitative rollouts; on rollout "
            "position it is the best of four on test_interp and the most "
            "conservative (highest) on test_extrap. Extrapolation to 200 m/s is "
            "the benchmark's honest failure mode: rollout position degrades ~6x "
            "against test_interp."
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
        "test_extrap": tuple(TEST_EXTRAP),
    },
    eval_splits=("val", "test_interp", "test_extrap"),
    aux_field=AUX_FIELD,
    qois=dict(QOIS),
    boundary_feature_fn=wall_distance_feature,
    dataset_id="2D-Copper-Bar-Taylor-Impact",
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
    "WALL_X_MM",
    "wall_distance_feature",
    "CARD",
    "SPEC",
]
