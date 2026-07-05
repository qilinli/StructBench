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

#: Official baseline results (ADR-0033); empty until a run is blessed.
RESULTS: tuple[BaselineResult, ...] = ()

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
