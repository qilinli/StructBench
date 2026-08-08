"""MGN — MeshGraphNet baseline for mesh-based simulation (ADR-0043 §8)."""

from .collate import MeshStatic, collate_mesh_samples, mesh_static_from_trajectory
from .network import MGNet
from .simulator import MeshSimulator

__all__ = [
    "MGNet",
    "MeshSimulator",
    "MeshStatic",
    "collate_mesh_samples",
    "mesh_static_from_trajectory",
]
