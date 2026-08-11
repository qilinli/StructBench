"""Two-edge-set encode-process-decode network for the MGN baseline (ADR-0043 §8).

Pure-``torch`` MeshGraphNets core: message passing over a mesh-edge set and a
world-edge set, aggregated per-node via ``index_add_``. Deliberately
independent of ``torch_geometric`` (unlike the CGN baseline in
``structbench.models.cgn``), so MGN's core stays PyG-free.

Edge-index convention: ``edge_index[0]`` is the sender, ``edge_index[1]`` is
the receiver; a message on edge ``(i, j)`` is aggregated into node ``j``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def build_mlp(
    in_size: int,
    hidden: int,
    n_hidden: int,
    out_size: int,
    *,
    layer_norm: bool,
) -> nn.Sequential:
    """Build a ReLU multilayer perceptron.

    Parameters
    ----------
    in_size:
        Input feature width.
    hidden:
        Width of each hidden layer.
    n_hidden:
        Number of hidden layers (each ``in_size``/``hidden`` -> ``hidden``,
        followed by ReLU).
    out_size:
        Output feature width of the final linear layer.
    layer_norm:
        If ``True``, append ``nn.LayerNorm(out_size)`` after the final linear
        layer.

    Returns
    -------
    nn.Sequential
        Alternating ``Linear``/``ReLU`` layers, ending in a bare ``Linear``
        (no trailing activation), with ``LayerNorm(out_size)`` appended iff
        ``layer_norm``.
    """
    sizes = [in_size] + [hidden] * n_hidden + [out_size]
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if layer_norm:
        layers.append(nn.LayerNorm(out_size))
    return nn.Sequential(*layers)


class MGNet(nn.Module):
    """Two-edge-set encode-process-decode graph network (MeshGraphNets core).

    Node features and two independent edge feature sets (mesh edges and
    world edges) are each encoded into a shared ``latent``-wide space, then
    refined over ``mp_steps`` processor blocks. Each block updates the
    mesh-edge latents and world-edge latents (edge MLP on
    ``[e, v_sender, v_receiver]``), aggregates both edge sets into their
    receiver nodes by ``sum``, and updates the node latents (node MLP on
    ``[v, agg_mesh, agg_world]``). Every edge and node update is residual
    (``x' = x + MLP(...)``). A final decoder MLP (no ``LayerNorm``) maps the
    processed node latents to the output size.

    The world-edge set may be empty (``Ew == 0``): its encoder and per-step
    edge MLP run on zero-row tensors, and its aggregation contributes an
    all-zero term to every node update.

    Parameters
    ----------
    node_in:
        Node input feature width.
    mesh_edge_in:
        Mesh-edge input feature width.
    world_edge_in:
        World-edge input feature width.
    out_size:
        Output feature width (e.g. the number of predicted quantities per
        node).
    latent:
        Shared latent width used by every encoder, processor MLP, and the
        decoder's input.
    mp_steps:
        Number of processor blocks. Each block has its own parameters (MGN
        does not share weights across steps).
    n_hidden:
        Number of hidden layers in every sub-MLP (encoders, processor edge
        and node MLPs, decoder).
    recompute_activation:
        If ``True`` (default), wrap each processor block in
        :func:`torch.utils.checkpoint.checkpoint` (non-reentrant) during
        training, recomputing the block's activations in the backward pass
        instead of retaining them. Peak activation memory drops from all
        ``mp_steps`` blocks to roughly one, at the cost of one extra forward
        per block. This is output- and gradient-identical (no dropout;
        deterministic ops), so the ADR-0043 §8 recipe's *results* are
        unchanged -- only the memory/compute profile differs. It is the
        PhysicsNeMo ``recompute_activation`` parity knob and exists because
        the world-edge (contact) count is data-dependent: a contact-heavy
        frame can blow up the ``(Ew, latent)`` processor activations and OOM
        an 80 GB GPU without it (observed at ~760 k steps on the
        ``deforming-mgn-v03`` blessing attempt). Checkpointing is skipped
        automatically when gradients are disabled (inference/eval), where
        there is nothing to retain.
    """

    def __init__(
        self,
        node_in: int,
        mesh_edge_in: int,
        world_edge_in: int,
        out_size: int,
        latent: int = 128,
        mp_steps: int = 15,
        n_hidden: int = 2,
        recompute_activation: bool = True,
    ) -> None:
        super().__init__()
        self._latent = latent
        self._recompute_activation = recompute_activation

        self._node_encoder = build_mlp(
            node_in, latent, n_hidden, latent, layer_norm=True
        )
        self._mesh_edge_encoder = build_mlp(
            mesh_edge_in, latent, n_hidden, latent, layer_norm=True
        )
        self._world_edge_encoder = build_mlp(
            world_edge_in, latent, n_hidden, latent, layer_norm=True
        )

        # Per-step processor MLPs: unshared parameters across mp_steps, per
        # the MGN architecture. Edge MLPs take [e, v_sender, v_receiver]
        # (3 * latent); the node MLP takes [v, agg_mesh, agg_world]
        # (3 * latent).
        self._mesh_edge_mlps = nn.ModuleList(
            [
                build_mlp(3 * latent, latent, n_hidden, latent, layer_norm=True)
                for _ in range(mp_steps)
            ]
        )
        self._world_edge_mlps = nn.ModuleList(
            [
                build_mlp(3 * latent, latent, n_hidden, latent, layer_norm=True)
                for _ in range(mp_steps)
            ]
        )
        self._node_mlps = nn.ModuleList(
            [
                build_mlp(3 * latent, latent, n_hidden, latent, layer_norm=True)
                for _ in range(mp_steps)
            ]
        )

        self._decoder = build_mlp(latent, latent, n_hidden, out_size, layer_norm=False)

    def forward(
        self,
        node_feats: Tensor,
        mesh_edge_index: Tensor,
        mesh_edge_feats: Tensor,
        world_edge_index: Tensor,
        world_edge_feats: Tensor,
    ) -> Tensor:
        """Run the encode-process-decode forward pass.

        Parameters
        ----------
        node_feats:
            ``(P, node_in)`` node input features.
        mesh_edge_index:
            ``(2, Em)`` int64 mesh-edge index; row 0 is sender, row 1 is
            receiver.
        mesh_edge_feats:
            ``(Em, mesh_edge_in)`` mesh-edge input features.
        world_edge_index:
            ``(2, Ew)`` int64 world-edge index (same sender/receiver
            convention); ``Ew`` may be 0.
        world_edge_feats:
            ``(Ew, world_edge_in)`` world-edge input features.

        Returns
        -------
        Tensor
            ``(P, out_size)`` decoded per-node output.
        """
        n_nodes = node_feats.shape[0]
        mesh_sender, mesh_receiver = mesh_edge_index[0], mesh_edge_index[1]
        world_sender, world_receiver = world_edge_index[0], world_edge_index[1]

        node_latent = self._node_encoder(node_feats)
        mesh_latent = self._mesh_edge_encoder(mesh_edge_feats)
        world_latent = self._world_edge_encoder(world_edge_feats)

        def processor_block(
            node_latent: Tensor,
            mesh_latent: Tensor,
            world_latent: Tensor,
            mesh_mlp: nn.Module,
            world_mlp: nn.Module,
            node_mlp: nn.Module,
        ) -> tuple[Tensor, Tensor, Tensor]:
            """One residual message-passing block (edges -> aggregate -> node).

            Factored out so the loop can run it either directly or under
            :func:`torch.utils.checkpoint.checkpoint`. The per-step MLPs are
            explicit arguments (passed through checkpoint), not closed-over
            loop variables, so backward-pass recomputation binds the correct
            step's parameters. Closes only over the loop-invariant edge
            indices and ``n_nodes``.
            """
            mesh_in = torch.cat(
                [
                    mesh_latent,
                    node_latent[mesh_sender],
                    node_latent[mesh_receiver],
                ],
                dim=-1,
            )
            new_mesh_latent = mesh_latent + mesh_mlp(mesh_in)

            world_in = torch.cat(
                [
                    world_latent,
                    node_latent[world_sender],
                    node_latent[world_receiver],
                ],
                dim=-1,
            )
            new_world_latent = world_latent + world_mlp(world_in)

            dtype, device = node_latent.dtype, node_latent.device
            agg_mesh = torch.zeros(n_nodes, self._latent, dtype=dtype, device=device)
            agg_mesh.index_add_(0, mesh_receiver, new_mesh_latent)
            agg_world = torch.zeros(n_nodes, self._latent, dtype=dtype, device=device)
            agg_world.index_add_(0, world_receiver, new_world_latent)

            node_in = torch.cat([node_latent, agg_mesh, agg_world], dim=-1)
            new_node_latent = node_latent + node_mlp(node_in)
            return new_node_latent, new_mesh_latent, new_world_latent

        # Recompute activations in backward only when there is a graph to build
        # (training); under torch.no_grad() (eval/rollout) run the block plainly.
        checkpointing = self._recompute_activation and torch.is_grad_enabled()

        for mesh_mlp, world_mlp, node_mlp in zip(
            self._mesh_edge_mlps,
            self._world_edge_mlps,
            self._node_mlps,
            strict=True,
        ):
            if checkpointing:
                node_latent, mesh_latent, world_latent = checkpoint(
                    processor_block,
                    node_latent,
                    mesh_latent,
                    world_latent,
                    mesh_mlp,
                    world_mlp,
                    node_mlp,
                    use_reentrant=False,
                )
            else:
                node_latent, mesh_latent, world_latent = processor_block(
                    node_latent,
                    mesh_latent,
                    world_latent,
                    mesh_mlp,
                    world_mlp,
                    node_mlp,
                )

        return self._decoder(node_latent)
