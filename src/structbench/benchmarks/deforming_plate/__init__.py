"""DeformingPlate benchmark: MeshGraphNets quasi-static rollout (ADR-0043)."""

from ..registry import BenchmarkSpec
from ..results import BaselineResult
from .benchmark import AUX_FIELD, KINEMATIC_TYPES, QOIS, TEST, TRAIN, VAL
from .card import CARD

__all__ = [
    "AUX_FIELD",
    "CARD",
    "KINEMATIC_TYPES",
    "QOIS",
    "SPEC",
    "TEST",
    "TRAIN",
    "VAL",
]

#: No blessed baseline yet (ADR-0033).
RESULTS: tuple[BaselineResult, ...] = ()

SPEC = BenchmarkSpec(
    card=CARD,
    results=RESULTS,
    splits={"train": tuple(TRAIN), "val": tuple(VAL), "test": tuple(TEST)},
    eval_splits=("val", "test"),
    aux_field=AUX_FIELD,
    qois=dict(QOIS),
    boundary_feature_fn=None,
    dataset_id="deforming_plate",
    kinematic_types=KINEMATIC_TYPES,
)
