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
    # No blessed or provisional result yet, so _quickstart_family (ADR-0046)
    # falls back to this default. MGN is the blessed target family for this
    # benchmark (ADR-0043) and configs/deforming_plate/mgn.toml is committed;
    # the base-class default "cgn" has no grouped config here (ADR-0046
    # comparison-table plan, config-path-exists regression guard).
    quickstart_family="mgn",
)
