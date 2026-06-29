"""v0.1 Taylor 2D impact benchmark (ADR-0019)."""

from .benchmark import (
    ALL_BENCHMARK_CASES,
    AUX_FIELD,
    HELD_ASIDE,
    TEST_EXTRAP,
    TEST_INTERP,
    TRAIN,
    VAL,
    WALL_X_MM,
    wall_distance_feature,
)

__all__ = [
    "TRAIN", "VAL", "TEST_INTERP", "TEST_EXTRAP", "HELD_ASIDE",
    "ALL_BENCHMARK_CASES", "AUX_FIELD", "WALL_X_MM", "wall_distance_feature",
]
