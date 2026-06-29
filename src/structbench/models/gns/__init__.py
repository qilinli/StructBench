"""Single-scale Graph Network Simulator (ported from the sgnn reference)."""

from .graph_network import EncodeProcessDecode
from .simulator import LearnedSimulator

__all__ = ["EncodeProcessDecode", "LearnedSimulator"]
