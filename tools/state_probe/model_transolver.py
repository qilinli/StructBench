"""Transolver trunk for the state-sufficiency probe (``--model transolver``).

Wraps the package's native :class:`TransolverNet` (the Stage-3 model family —
Physics-Attention over the particle set, ADR-0044) behind the probe's
operator contract: node features in, eight increments out.

One structural difference from the message-passing operator: Physics-
Attention has no edges, so geometry cannot ride on relative displacements.
Normalised position is prepended to the node features instead — the native
Transolver treatment (positions are input channels), at the cost of the MP
operator's built-in translation invariance. Both arms get the same position
channels, so the ablation stays fair.

Exploratory scratch code -- not part of the ``structbench`` package.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from structbench.models.transolver.network import TransolverNet


class TransolverOperator(nn.Module):
    """``F: z_n -> dz`` with a Physics-Attention trunk.

    Parameters
    ----------
    node_in:
        Arm feature width (8 full / 2 kinematic). Two position channels are
        added internally, so the net sees ``node_in + 2``.
    hidden, n_layers, n_heads, slice_num:
        Forwarded to :class:`TransolverNet`. Defaults sized to budget-match
        the MP probe at the A100 configuration (hidden 128, 6 blocks).
    out_dim:
        Predicted increment width (8, as the MP operator).
    """

    def __init__(
        self,
        node_in: int,
        *,
        hidden: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        slice_num: int = 32,
        out_dim: int = 8,
    ) -> None:
        super().__init__()
        self.net = TransolverNet(
            node_in + 2,
            out_dim,
            hidden_dim=hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            slice_num=slice_num,
        )

    def forward(self, node_feat: Tensor, pos_norm: Tensor, npp: Tensor) -> Tensor:
        """Predict increments for a flat multi-example point set.

        Parameters
        ----------
        node_feat:
            ``(P, node_in)`` arm features, examples concatenated.
        pos_norm:
            ``(P, 2)`` positions normalised by ``X_SCALE``.
        npp:
            ``(B,)`` particles per example, concatenation order.
        """
        return self.net(torch.cat([pos_norm, node_feat], dim=-1), npp)
