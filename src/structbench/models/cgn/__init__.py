"""CGN — Concrete Graph Network, the StructBench reference baseline.

The model of Li, Q., Wang, Z., Li, L., Hao, H., Chen, W., & Shao, Y. (2023),
"Machine learning prediction of structural dynamic responses using graph
neural networks", Computers & Structures 289, 107188
(https://doi.org/10.1016/j.compstruc.2023.107188) — developed and validated
on the Taylor-bar and notch-beam datasets StructBench ships (ADR-0034).

Architecturally it builds on the encode-process-decode Graph Network
Simulator of Sanchez-Gonzalez et al. (2020); this implementation is the
lineage of the paper's own code (ported from the sgnn reference).
"""

from .graph_network import EncodeProcessDecode
from .simulator import LearnedSimulator

__all__ = ["EncodeProcessDecode", "LearnedSimulator"]
