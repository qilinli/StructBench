"""MGN — MeshGraphNet baseline for mesh-based simulation (ADR-0043 §8)."""

from .network import MGNet
from .simulator import MeshSimulator

__all__ = ["MGNet", "MeshSimulator"]
