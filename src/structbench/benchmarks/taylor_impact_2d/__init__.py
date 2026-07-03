"""v0.1 Taylor 2D impact benchmark (ADR-0019)."""

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
from ..registry import BenchmarkSpec

SPEC = BenchmarkSpec(
    card=CARD,
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
