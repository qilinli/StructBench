"""Notch-beam 3-point bend benchmark (ADR-0026)."""

from ..registry import BenchmarkSpec
from ..results import BaselineResult
from .benchmark import (
    AUX_FIELD,
    CONCRETE_TYPE,
    PIN_TYPE,
    PROBE,
    QOIS,
    SUPPORT_TYPE,
    TEST_INTERP,
    TRAIN,
    VAL,
)
from .card import CARD

__all__ = [
    "AUX_FIELD",
    "CARD",
    "CONCRETE_TYPE",
    "PIN_TYPE",
    "PROBE",
    "QOIS",
    "SPEC",
    "SUPPORT_TYPE",
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
        "probe": tuple(PROBE),
    },
    eval_splits=("val", "test_interp", "probe"),
    aux_field=AUX_FIELD,
    qois=dict(QOIS),
    boundary_feature_fn=None,
    dataset_id="2D-Notched-Beam",
    kinematic_types=(PIN_TYPE, SUPPORT_TYPE),
)
