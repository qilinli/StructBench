"""Native Transolver family (ADR-0041 step 2; ADR-0044)."""

from .network import PhysicsAttentionIrregularMesh, TransolverNet
from .simulator import (
    FlowMapAnchor,
    TransolverSimulator,
    anchor_from_predictions,
    hardening_sigma_y,
    plane_strain_vm,
)

__all__ = [
    "FlowMapAnchor",
    "anchor_from_predictions",
    "PhysicsAttentionIrregularMesh",
    "TransolverNet",
    "TransolverSimulator",
    "hardening_sigma_y",
    "plane_strain_vm",
]
