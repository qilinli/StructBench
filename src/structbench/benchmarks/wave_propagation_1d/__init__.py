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
]

#: Official baseline results (ADR-0033). Transcribed from the ``mean`` block of
#: the blessed run's held-out ``metrics-test_interp.json`` at 4 significant
#: figures; full precision, per-case numbers and the fleet spread stay in the
#: run directory. ``val`` selects the checkpoint, so it is not a number to beat
#: and is omitted here.
RESULTS: tuple[BaselineResult, ...] = (
    BaselineResult(
        family="cgn",
        label="CGN baseline",
        run_commit="48046ea",
        run_date="2026-07-10",
        metrics={
            "test_interp": {
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
            "quantities (all within ~1 output frame)."
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
)
