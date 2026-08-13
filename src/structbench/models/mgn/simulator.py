"""``MeshSimulator``: MGN's stateful predict-only wrapper (ADR-0043 §8).

The full per-case-binding statefulness contract (bind per case, reset before
each eval pass, step pointer, tripwire semantics) is inherited from
:class:`~structbench.models.common.simulator_base.CaseBoundSimulator` and
documented in that module's docstring — it is not repeated here. This module
covers what is MGN-specific: network sizing and the graph feature builder.

**``world_edge_radius``.** Given in the WORKING FRAME — the same units as
the positions passed to ``bind_case``/``predict_positions`` (mm, for a
metre-native source, per ADR-0043 §8's ``0.03 x f_length x 1e3``
conversion). The default ``30.0`` is CONFIRMED by the full-dataset unit
measurement (metre-native source; ADR-0042 §2b dated note); the training
config remains the source of truth for a blessed run.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..common import CaseBoundSimulator
from .mesh_ops import cells_to_edges, world_edges
from .network import MGNet
from .normalizers import OnlineNormalizer


class MeshSimulator(CaseBoundSimulator):
    """MGN encode-process-decode simulator with per-case GT binding.

    See :class:`~structbench.models.common.simulator_base.CaseBoundSimulator`'s
    module docstring for the full statefulness contract (bind per case,
    reset before each eval pass, tripwire semantics) — it is not repeated
    here.

    Parameters
    ----------
    dim:
        Spatial dimensionality of node positions.
    latent:
        Shared latent width forwarded to :class:`~.network.MGNet`.
    mp_steps:
        Number of message-passing steps forwarded to
        :class:`~.network.MGNet`.
    n_hidden:
        Hidden-layer count per sub-MLP, forwarded to
        :class:`~.network.MGNet`.
    node_type_size:
        Width of the one-hot node-type encoding (``NodeType.SIZE`` in the
        source MeshGraphNets framework).
    kinematic_types:
        Node-type codes whose motion is prescribed by ground truth; their
        rows anchor and verify the step pointer (the tripwire).
    scripted_types:
        Node-type codes whose next-step ground-truth velocity is fed as a
        node input feature (a subset of ``kinematic_types`` in the ADR-0043
        recipe: OBSTACLE is scripted, HANDLE is not).
    world_edge_radius:
        Radius (working-frame units) for :func:`~.mesh_ops.world_edges`,
        recomputed from the current positions on every call. See the module
        docstring's units note (measured-SI confirmed).
    history_velocities:
        Number of finite-difference window velocities appended to the node
        features (ADR-0049); ``0`` keeps the reference feature builder.
    mesh_edge_max_stretch:
        Drop mesh-edge messages whose current length exceeds this multiple
        of their rest length (ADR-0049); dropped (torn) pairs regain
        world-edge eligibility. ``0.0`` disables the gate (the reference
        behaviour). Applied identically in training and rollout, inside
        :meth:`_graph_features`.
    device:
        Device the network and normalizer buffers are moved to at
        construction time.
    """

    def __init__(
        self,
        dim: int = 3,
        latent: int = 128,
        mp_steps: int = 15,
        n_hidden: int = 2,
        node_type_size: int = 9,
        kinematic_types: tuple[int, ...] = (1, 3),
        scripted_types: tuple[int, ...] = (1,),
        world_edge_radius: float = 30.0,
        history_velocities: int = 0,
        mesh_edge_max_stretch: float = 0.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            dim=dim,
            node_type_size=node_type_size,
            kinematic_types=kinematic_types,
            scripted_types=scripted_types,
            history_velocities=history_velocities,
            device=device,
        )
        self._world_edge_radius = world_edge_radius
        self._mesh_edge_max_stretch = mesh_edge_max_stretch

        node_in = node_type_size + dim + history_velocities * dim
        self._net = MGNet(
            node_in=node_in,
            mesh_edge_in=2 * dim + 2,
            world_edge_in=dim + 1,
            out_size=dim + 1,
            latent=latent,
            mp_steps=mp_steps,
            n_hidden=n_hidden,
        )
        self._node_normalizer = OnlineNormalizer(node_in)
        self._mesh_edge_normalizer = OnlineNormalizer(2 * dim + 2)
        self._world_edge_normalizer = OnlineNormalizer(dim + 1)
        self._target_normalizer = OnlineNormalizer(dim + 1)

        # MGN-specific per-case binding, populated by _on_bind_case() (called
        # at the end of the inherited bind_case()). The rest of the per-case
        # bind state lives on CaseBoundSimulator -- see its module docstring.
        # A PLAIN attribute, like the base's: never registered as a buffer/
        # parameter, so it never enters state_dict().
        self._mesh_edge_index: Tensor | None = None

        self.to(device)

    def _on_bind_case(self, cells: Tensor) -> None:
        """Build the mesh edge index for the newly bound case.

        Parameters
        ----------
        cells:
            ``(n_cells, nodes_per_cell)`` int64 element connectivity.
        """
        self._mesh_edge_index = cells_to_edges(cells)

    def predict_positions(
        self,
        current_positions: Tensor,
        nparticles_per_example: Tensor,
        particle_types: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict the next positions and de-normalized stress.

        Parameters
        ----------
        current_positions:
            ``(P, F, dim)`` position window; ``current_positions[:, -1]`` is
            the most recent (current) frame.
        nparticles_per_example:
            Unused: ``MeshSimulator`` serves one bound case (a single
            example) at a time. Accepted for ``_SimulatorLike`` protocol
            compatibility with :mod:`structbench.eval`.
        particle_types:
            Unused directly: the bound case's one-hot types and kinematic
            mask, cached by :meth:`bind_case`, are used instead. The caller
            is expected to pass the same ``particle_types`` that were bound.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(next_positions (P, dim), aux (P, 1))``. ``aux`` is the
            de-normalized predicted stress, kept 2-D per the
            ``_SimulatorLike`` contract.

        Raises
        ------
        RuntimeError
            If called before :meth:`bind_case`, or if the input window's
            kinematic rows do not match the bound ground truth at the
            current pointer position (the tripwire; see
            :class:`~structbench.models.common.simulator_base.CaseBoundSimulator`'s
            module docstring).
        """
        del nparticles_per_example, particle_types  # see docstring: unused

        mesh_edge_index = self._mesh_edge_index
        reference_coords = self._reference_coords
        node_type_onehot = self._node_type_onehot
        kin_mask = self._kin_mask
        scripted_mask = self._scripted_mask
        gt_positions = self._gt_positions
        if (
            mesh_edge_index is None
            or reference_coords is None
            or node_type_onehot is None
            or kin_mask is None
            or scripted_mask is None
            or gt_positions is None
        ):
            raise RuntimeError(
                "MeshSimulator.predict_positions() called before bind_case(); "
                "bind_case() must be called with the case being evaluated "
                "before any prediction."
            )

        x_t = current_positions[:, -1].contiguous()
        n_frames = current_positions.shape[1]

        self._advance_pointer(x_t, n_frames)
        scripted_velocity = self._eval_scripted_velocity(x_t)
        velocity_history = (
            self._window_velocity_history(current_positions)
            if self._history_velocities > 0
            else None
        )

        (
            node_feats,
            mesh_edge_index,
            mesh_edge_feats,
            world_edge_index,
            world_edge_feats,
        ) = self._graph_features(
            x_t,
            node_type_onehot,
            scripted_velocity,
            mesh_edge_index,
            reference_coords,
            None,
            accumulate=False,
            velocity_history=velocity_history,
        )

        out = self._net(
            node_feats,
            mesh_edge_index,
            mesh_edge_feats,
            world_edge_index,
            world_edge_feats,
        )
        # Inverse-normalize the FULL (P, dim+1) output first -- slicing
        # before inverse would broadcast the dim-wide velocity slice against
        # the (dim+1)-wide std/mean buffers.
        out = self._target_normalizer.inverse(out)
        velocity = out[:, : self._dim]
        stress = out[:, self._dim :]

        next_positions = x_t + velocity
        return next_positions, stress

    def _graph_features(
        self,
        x_last: Tensor,
        one_hot: Tensor,
        scripted_velocity: Tensor,
        mesh_edge_index: Tensor,
        reference_coords: Tensor,
        n_particles_per_example: Tensor | None,
        *,
        accumulate: bool,
        velocity_history: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Build the five normalized network inputs shared by train and eval.

        Parameters
        ----------
        x_last:
            ``(P, dim)`` current world positions (working-frame units, e.g.
            mm).
        one_hot:
            ``(P, node_type_size)`` float32 one-hot node-type encoding.
        scripted_velocity:
            ``(P, dim)`` scripted-actuator velocity node feature; zero on
            non-scripted rows.
        mesh_edge_index:
            ``(2, Em)`` int64 mesh-edge index. For a batched call (multiple
            collated examples) this is batched with per-example node offsets
            already applied.
        reference_coords:
            ``(P, dim)`` mesh-space (rest/reference) coordinates.
        n_particles_per_example:
            ``(B,)`` int64 per-example node counts, or ``None`` for a single
            example (the inference fast path).
        accumulate:
            Forwarded to all four ``OnlineNormalizer`` calls made here (node,
            mesh-edge, world-edge); the caller's target normalizer is a
            separate call outside this method.
        velocity_history:
            ``(P, history_velocities * dim)`` flattened window velocities
            (ADR-0049); required exactly when the simulator was built with
            ``history_velocities > 0``, ``None`` otherwise.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            ``(node_f_norm, mesh_edge_index, mesh_ef_norm, world_edge_index,
            world_ef_norm)`` -- exactly the five positional arguments
            :class:`~.network.MGNet` expects, in order. ``mesh_edge_index``
            is passed through unchanged (returned for the caller's
            convenience, since it flows straight into the network call).

        Raises
        ------
        ValueError
            If ``n_particles_per_example`` is given and ``mesh_edge_index``
            contains an edge whose sender and receiver fall in different
            example blocks (a malformed batch). This is checked explicitly
            rather than trusted: :func:`~.mesh_ops.world_edges` never
            *indexes* with ``mesh_edge_index`` -- it only builds an
            arithmetic hash key from it (``sender * n + receiver``) to
            exclude mesh-connected pairs from the radius query -- so a
            cross-example edge would NOT raise on its own. It would instead
            run to completion on finite-but-wrong output, and an
            out-of-range local key can even spuriously collide with a valid
            candidate pair (phantom exclusion of a real world edge). Silent
            corruption, not a crash, hence the explicit check.

        Notes
        -----
        **Batched world-edge partition (load-bearing):** when
        ``n_particles_per_example`` is given, world edges are computed PER
        EXAMPLE -- positions are sliced per sample, :func:`~.mesh_ops.
        world_edges` is run on each slice (with mesh edges restricted to and
        re-indexed into that slice's local node range), and the resulting
        local world-edge index has the example's node offset added back
        before concatenation. This is never one whole-tensor radius query
        over the collated batch, which would invent cross-example edges
        between unrelated cases whose node coordinates happen to be close
        (or, worst case, coincide) after collation. ``None`` selects the
        single-example fast path used by inference (single example, so the
        cross-boundary check above does not apply).
        """
        node_parts = [one_hot, scripted_velocity]
        if self._history_velocities > 0:
            if velocity_history is None:
                raise ValueError(
                    "simulator was built with history_velocities="
                    f"{self._history_velocities} but no velocity_history "
                    "feature was supplied"
                )
            node_parts.append(velocity_history)
        node_feats_raw = torch.cat(node_parts, dim=-1)
        node_feats = self._node_normalizer(node_feats_raw, accumulate=accumulate)

        sender, receiver = mesh_edge_index[0], mesh_edge_index[1]
        u_ij = reference_coords[sender] - reference_coords[receiver]
        u_norm = torch.linalg.norm(u_ij, dim=-1, keepdim=True)
        x_ij = x_last[sender] - x_last[receiver]
        x_norm = torch.linalg.norm(x_ij, dim=-1, keepdim=True)

        # Stretch gate (ADR-0049): a mesh edge stretched past the threshold
        # is torn — its message is dropped and, because the world-edge query
        # below excludes only the GATED index, the pair regains world-edge
        # eligibility if it comes back within the radius.
        if self._mesh_edge_max_stretch > 0:
            keep = (x_norm <= self._mesh_edge_max_stretch * u_norm).squeeze(-1)
            mesh_edge_index = mesh_edge_index[:, keep]
            sender, receiver = mesh_edge_index[0], mesh_edge_index[1]
            u_ij, u_norm = u_ij[keep], u_norm[keep]
            x_ij, x_norm = x_ij[keep], x_norm[keep]

        mesh_edge_feats_raw = torch.cat([u_ij, u_norm, x_ij, x_norm], dim=-1)
        mesh_edge_feats = self._mesh_edge_normalizer(
            mesh_edge_feats_raw, accumulate=accumulate
        )

        if n_particles_per_example is None:
            world_edge_index = world_edges(
                x_last, self._world_edge_radius, mesh_edge_index
            )
        else:
            example_of = torch.repeat_interleave(
                torch.arange(
                    n_particles_per_example.numel(), device=mesh_edge_index.device
                ),
                n_particles_per_example,
            )
            if (example_of[sender] != example_of[receiver]).any():
                raise ValueError(
                    "mesh_edge_index crosses example boundaries — collate "
                    "must offset per-trajectory edges"
                )

            world_parts: list[Tensor] = []
            offset = 0
            for count in n_particles_per_example.tolist():
                end = offset + count
                pos_slice = x_last[offset:end]
                in_range = (sender >= offset) & (sender < end)
                local_mesh_edges = mesh_edge_index[:, in_range] - offset
                local_world_edges = world_edges(
                    pos_slice, self._world_edge_radius, local_mesh_edges
                )
                world_parts.append(local_world_edges + offset)
                offset = end
            world_edge_index = (
                torch.cat(world_parts, dim=1)
                if world_parts
                else mesh_edge_index.new_zeros((2, 0))
            )

        w_sender, w_receiver = world_edge_index[0], world_edge_index[1]
        wx_ij = x_last[w_sender] - x_last[w_receiver]
        wx_norm = torch.linalg.norm(wx_ij, dim=-1, keepdim=True)
        world_edge_feats_raw = torch.cat([wx_ij, wx_norm], dim=-1)
        world_edge_feats = self._world_edge_normalizer(
            world_edge_feats_raw, accumulate=accumulate
        )

        return (
            node_feats,
            mesh_edge_index,
            mesh_edge_feats,
            world_edge_index,
            world_edge_feats,
        )

    def forward_train(
        self,
        x_last: Tensor,
        next_positions: Tensor,
        next_aux: Tensor,
        particle_types: Tensor,
        mesh_edge_index: Tensor,
        reference_coords: Tensor,
        n_particles_per_example: Tensor,
        *,
        accumulate: bool,
        velocity_history: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """One training forward pass: normalized network output and target.

        Parameters
        ----------
        x_last:
            ``(P, dim)`` current world positions; noise (if any) has ALREADY
            been applied by the caller before this method is called.
        next_positions:
            ``(P, dim)`` ground-truth next-frame world positions.
        next_aux:
            ``(P,)`` ground-truth stress (working-frame units, e.g. MPa) at
            the next frame.
        particle_types:
            ``(P,)`` int64 node-type codes.
        mesh_edge_index:
            ``(2, Em)`` int64 mesh-edge index (batched with per-example node
            offsets is fine).
        reference_coords:
            ``(P, dim)`` mesh-space (rest/reference) coordinates.
        n_particles_per_example:
            ``(B,)`` int64 per-example node counts from the collate step;
            drives the batched world-edge partition in :meth:`_graph_features`.
        accumulate:
            If ``True``, this call's features are folded into all four
            ``OnlineNormalizer`` running statistics (node, mesh-edge,
            world-edge -- via :meth:`_graph_features` -- and target, here).
        velocity_history:
            ``(P, history_velocities * dim)`` flattened window velocities
            computed by the caller from the NOISY position window
            (ADR-0049); required exactly when the simulator was built with
            ``history_velocities > 0``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(pred_norm, target_norm)``, each ``(P, dim + 1)``: the raw
            network output (already in normalized/target space, matching
            what :meth:`predict_positions` inverse-normalizes) and the
            normalized ground-truth target ``cat([next_positions - x_last,
            next_aux[:, None]], dim=1)``.

        Notes
        -----
        The one-hot node-type encoding is built here; the scripted-velocity
        node input is delegated to the inherited
        :meth:`~structbench.models.common.simulator_base.CaseBoundSimulator._train_scripted_velocity`
        (``next_positions - x_last`` on scripted rows, zero elsewhere) since
        its source differs from the eval path's bound ground truth; the rest
        of the feature assembly is shared with :meth:`predict_positions` via
        :meth:`_graph_features`.

        gamma = 1.0 falls out of the construction here: the caller is
        expected to have already added noise to ``x_last``, so the velocity
        target ``next_positions - x_last`` is measured from the noisy
        position, matching the source MeshGraphNets training recipe without
        a separate noise-correction term. On the velocity-history path
        (ADR-0049) the caller instead passes the noise-ADJUSTED next
        position (``next + noise[:, -1]``, the GNS reference convention),
        so the same subtraction yields the CLEAN next velocity — velocity
        noise is corrected exactly, the accumulated random-walk position
        offset deliberately is not (see ``_mesh_family_noise``).
        """
        one_hot = F.one_hot(particle_types, num_classes=self._node_type_size).to(
            torch.float32
        )
        scripted_velocity = self._train_scripted_velocity(
            x_last, next_positions, particle_types
        )

        (
            node_feats,
            mesh_edge_index,
            mesh_edge_feats,
            world_edge_index,
            world_edge_feats,
        ) = self._graph_features(
            x_last,
            one_hot,
            scripted_velocity,
            mesh_edge_index,
            reference_coords,
            n_particles_per_example,
            accumulate=accumulate,
            velocity_history=velocity_history,
        )

        pred_norm = self._net(
            node_feats,
            mesh_edge_index,
            mesh_edge_feats,
            world_edge_index,
            world_edge_feats,
        )

        target_raw = torch.cat([next_positions - x_last, next_aux[:, None]], dim=1)
        target_norm = self._target_normalizer(target_raw, accumulate=accumulate)

        return pred_norm, target_norm
