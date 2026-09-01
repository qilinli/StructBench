"""Neighbourhood operator for the stage-1 state-sufficiency probe.

SPH dynamics are a *spatial* discretisation: the update at particle ``i``
needs velocity gradients over its neighbours, so a per-particle MLP on ten
scalars cannot work no matter how complete the state is. Everything here is a
neighbourhood operator for that reason -- a per-particle control would fail
for reasons unrelated to state sufficiency and would be misread.

Encode-process-decode in pure torch (no PyG message passing, so it runs on
MPS); the neighbour search reuses the package's native ``radius_graph``.

Exploratory scratch code -- not part of the ``structbench`` package.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """Two-hidden-layer MLP, optionally LayerNorm-terminated (GNS convention)."""

    def __init__(
        self, in_dim: int, hidden: int, out_dim: int, *, layer_norm: bool = True
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        ]
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class InteractionBlock(nn.Module):
    """One message-passing step with residual node and edge updates."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.edge_mlp = MLP(3 * hidden, hidden, hidden)
        self.node_mlp = MLP(2 * hidden, hidden, hidden)

    def forward(
        self, node: Tensor, edge: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        recv, send = edge_index[0], edge_index[1]
        e = self.edge_mlp(torch.cat([edge, node[recv], node[send]], dim=-1))
        agg = torch.zeros_like(node).index_add_(0, recv, e)
        n = self.node_mlp(torch.cat([node, agg], dim=-1))
        return node + n, edge + e


class StateOperator(nn.Module):
    """``F: z_n -> dz`` over a particle neighbourhood.

    Predicts *increments*, not absolute state -- the solver integrates rates,
    and increment targets keep the yield and monotonicity structure reachable
    by later variants.

    Parameters
    ----------
    node_in:
        Input node feature width. This is what the ablation varies: the full
        arm feeds ``v, s, peeq, E, rho``; the kinematic arm feeds ``v`` only.
    out_dim:
        Predicted increment width (default 8: ``dv`` 2, ``ds`` 3, ``dpeeq`` 1,
        ``dE`` 1, ``drho`` 1). Position is integrated from velocity, never
        predicted directly.
    """

    def __init__(
        self,
        node_in: int,
        *,
        edge_in: int = 3,
        hidden: int = 128,
        n_steps: int = 6,
        out_dim: int = 8,
    ) -> None:
        super().__init__()
        self.node_encoder = MLP(node_in, hidden, hidden)
        self.edge_encoder = MLP(edge_in, hidden, hidden)
        self.blocks = nn.ModuleList(InteractionBlock(hidden) for _ in range(n_steps))
        self.decoder = MLP(hidden, hidden, out_dim, layer_norm=False)

    def forward(
        self, node_feat: Tensor, edge_feat: Tensor, edge_index: Tensor
    ) -> Tensor:
        node = self.node_encoder(node_feat)
        edge = self.edge_encoder(edge_feat)
        for block in self.blocks:
            node, edge = block(node, edge, edge_index)
        return self.decoder(node)


def edge_features(pos: Tensor, edge_index: Tensor, radius: float) -> Tensor:
    """Relative displacement and distance per edge, scaled by the radius.

    Relative rather than absolute position: the operator must be translation
    invariant, and absolute coordinates would tie it to where the bar happens
    to sit.
    """
    recv, send = edge_index[0], edge_index[1]
    delta = (pos[send] - pos[recv]) / radius
    dist = delta.norm(dim=-1, keepdim=True)
    return torch.cat([delta, dist], dim=-1)
