"""``MeshSimulator``: MGN's stateful predict-only wrapper (ADR-0043 §8).

**STATEFULNESS CONTRACT — READ BEFORE USE.**

``MeshSimulator`` carries two kinds of state that must not be confused:

1. Model parameters and the four
   :class:`~structbench.models.mgn.normalizers.OnlineNormalizer` buffers.
   These live in ``state_dict()`` and travel with a checkpoint via
   :meth:`save`/:meth:`load`. They are **case-independent**.
2. Per-case binding, set by :meth:`bind_case` and held as PLAIN (non-buffer,
   non-parameter) attributes, so a checkpoint never carries case data.
   Binding a case caches: the mesh edge index, the mesh-space reference
   coordinates, the one-hot node types, the kinematic-particle mask, and the
   bound case's FULL ground-truth position trajectory ``(T, P, dim)`` — of
   which only the KINEMATIC rows are ever read (the scripted-actuator
   velocity input, and the tripwire below). Binding also resets the
   autoregressive step pointer.

``predict_positions`` is the ONLY method ``structbench.eval`` calls
(:func:`~structbench.eval.rollout.rollout`,
:func:`~structbench.eval.rollout.one_step_position_rmse`,
:func:`~structbench.eval.rollout.one_step_aux_rmse`) — none of them pass any
per-case context. Consequently:

* Call :meth:`bind_case` before the first ``predict_positions`` call of an
  eval pass on a given trajectory.
* Call :meth:`reset_rollout` before **EACH** separate eval pass over the
  SAME bound case. A fresh rollout and a fresh one-step sweep are two
  separate eval passes, and each needs its own ``reset_rollout()`` — the
  step pointer has no way to know an eval pass has ended on its own.

**Step pointer — deterministic, never search-based.** The first
``predict_positions`` call after ``bind_case``/``reset_rollout`` anchors the
pointer at ``t = F`` (``F`` = the input window's frame count), because every
eval entry point — rollout's autoregressive loop and both one-step
teacher-forced sweeps — makes its first call with a window whose last frame
is ground-truth frame ``F - 1``. Every subsequent call advances the pointer
by 1.

**Tripwire — verification only, never anchoring.** At every call (when
kinematic particles are bound), the input window's kinematic rows are
compared against the bound ground truth at frame ``t - 1`` (``atol=1e-4``).
A mismatch raises ``RuntimeError`` naming both likely causes: a stale
rollout pointer (call ``reset_rollout()`` before each eval pass) or a
case/trajectory mismatch (``bind_case`` was not (re)bound to the trajectory
under evaluation). This check stays exact even when a kinematic particle is
stationary across consecutive frames (e.g. a HANDLE node paused
mid-actuation) — the exact failure mode a search-based "find this window in
the GT" anchor would silently mishandle. With no kinematic particles bound,
the pointer/tripwire logic is skipped entirely and the scripted-velocity
input is always zero.

**``world_edge_radius``.** Given in the WORKING FRAME — the same units as
the positions passed to ``bind_case``/``predict_positions`` (mm, for a
metre-native source, per ADR-0043 §8's ``0.03 x f_length x 1e3``
conversion). The default ``30.0`` is CONFIRMED by the full-dataset unit
measurement (metre-native source; ADR-0042 §2b dated note); the training
config remains the source of truth for a blessed run.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .mesh_ops import cells_to_edges, world_edges
from .network import MGNet
from .normalizers import OnlineNormalizer


class MeshSimulator(nn.Module):
    """MGN encode-process-decode simulator with per-case GT binding.

    See the module docstring for the full statefulness contract (bind per
    case, reset before each eval pass, tripwire semantics) — it is not
    repeated here.

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
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._dim = dim
        self._node_type_size = node_type_size
        self._kinematic_types = kinematic_types
        self._scripted_types = scripted_types
        self._world_edge_radius = world_edge_radius

        if not set(scripted_types) <= set(kinematic_types):
            raise ValueError(
                "scripted_types must be a subset of kinematic_types (the "
                "NORMAL-only noise mask relies on it): "
                f"scripted_types={scripted_types!r}, "
                f"kinematic_types={kinematic_types!r}"
            )

        self._net = MGNet(
            node_in=node_type_size + dim,
            mesh_edge_in=2 * dim + 2,
            world_edge_in=dim + 1,
            out_size=dim + 1,
            latent=latent,
            mp_steps=mp_steps,
            n_hidden=n_hidden,
        )
        self._node_normalizer = OnlineNormalizer(node_type_size + dim)
        self._mesh_edge_normalizer = OnlineNormalizer(2 * dim + 2)
        self._world_edge_normalizer = OnlineNormalizer(dim + 1)
        self._target_normalizer = OnlineNormalizer(dim + 1)

        # Per-case binding (populated by bind_case). These are PLAIN
        # attributes -- never passed through register_buffer/register_
        # parameter -- so they never enter state_dict(): a saved checkpoint
        # is case-independent, and predict_positions before bind_case fails
        # loudly instead of running on stale/absent data.
        self._mesh_edge_index: Tensor | None = None
        self._reference_coords: Tensor | None = None
        self._node_type_onehot: Tensor | None = None
        self._kin_mask: Tensor | None = None
        self._scripted_mask: Tensor | None = None
        self._gt_positions: Tensor | None = None
        self._has_kinematic: bool = False
        self._n_gt_frames: int = 0
        self._t: int | None = None

        self.to(device)

    def bind_case(
        self,
        cells: Tensor,
        reference_coords: Tensor,
        particle_types: Tensor,
        kinematic_positions: Tensor,
    ) -> None:
        """Bind one case's static mesh and GT trajectory; reset the pointer.

        Parameters
        ----------
        cells:
            ``(n_cells, nodes_per_cell)`` int64 element connectivity.
        reference_coords:
            ``(P, dim)`` mesh-space (rest/reference) coordinates.
        particle_types:
            ``(P,)`` int64 node-type codes.
        kinematic_positions:
            ``(T, P, dim)`` full ground-truth world-space trajectory of this
            case. Only rows whose type is in ``kinematic_types`` are ever
            read (the scripted-velocity node feature, restricted further to
            ``scripted_types``, and the pointer tripwire).
        """
        self._mesh_edge_index = cells_to_edges(cells)
        self._reference_coords = reference_coords
        self._node_type_onehot = F.one_hot(
            particle_types, num_classes=self._node_type_size
        ).to(torch.float32)

        dtype, device = particle_types.dtype, particle_types.device
        kinematic = torch.tensor(self._kinematic_types, dtype=dtype, device=device)
        scripted = torch.tensor(self._scripted_types, dtype=dtype, device=device)
        self._kin_mask = torch.isin(particle_types, kinematic)
        self._scripted_mask = torch.isin(particle_types, scripted)
        self._has_kinematic = bool(self._kin_mask.any())

        self._gt_positions = kinematic_positions
        self._n_gt_frames = kinematic_positions.shape[0]
        self._t = None

    def reset_rollout(self) -> None:
        """Reset the step pointer; the next call re-anchors at ``t = F``."""
        self._t = None

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
            current pointer position (the tripwire; see the module
            docstring).
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

        t: int | None
        if self._has_kinematic:
            t = n_frames if self._t is None else self._t + 1
            gt_prev = gt_positions[t - 1]
            if not torch.allclose(x_t[kin_mask], gt_prev[kin_mask], atol=1e-4):
                raise RuntimeError(
                    "MeshSimulator's kinematic input rows are out of sync "
                    f"with the bound ground-truth trajectory at frame {t - 1}. "
                    "This means either: call reset_rollout() before each "
                    "eval pass (rollout / one_step_position_rmse / "
                    "one_step_aux_rmse), or bind_case() was not (re)bound to "
                    "the trajectory currently being evaluated."
                )
            self._t = t
        else:
            t = None

        # Scripted-actuator velocity input: GT[t][scripted] - x_t[scripted],
        # zero elsewhere and zero past the bound trajectory's final frame
        # (final-frame guard) and whenever no kinematic rows are bound.
        scripted_velocity = torch.zeros(
            x_t.shape[0], self._dim, dtype=x_t.dtype, device=x_t.device
        )
        if t is not None and t < self._n_gt_frames:
            gt_t = gt_positions[t]
            scripted_velocity[scripted_mask] = gt_t[scripted_mask] - x_t[scripted_mask]

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
        node_feats_raw = torch.cat([one_hot, scripted_velocity], dim=-1)
        node_feats = self._node_normalizer(node_feats_raw, accumulate=accumulate)

        sender, receiver = mesh_edge_index[0], mesh_edge_index[1]
        u_ij = reference_coords[sender] - reference_coords[receiver]
        u_norm = torch.linalg.norm(u_ij, dim=-1, keepdim=True)
        x_ij = x_last[sender] - x_last[receiver]
        x_norm = torch.linalg.norm(x_ij, dim=-1, keepdim=True)
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
        The one-hot node-type encoding and the scripted-velocity node input
        are built here (``next_positions - x_last`` on scripted rows, zero
        elsewhere) since their sources differ from the eval path's bound
        ground truth; the rest of the feature assembly is shared with
        :meth:`predict_positions` via :meth:`_graph_features`.

        gamma = 1.0 falls out of the construction here: the caller is
        expected to have already added noise to ``x_last``, so the velocity
        target ``next_positions - x_last`` is measured from the noisy
        position, matching the source MeshGraphNets training recipe without
        a separate noise-correction term.
        """
        one_hot = F.one_hot(particle_types, num_classes=self._node_type_size).to(
            torch.float32
        )

        dtype, device = particle_types.dtype, particle_types.device
        scripted = torch.tensor(self._scripted_types, dtype=dtype, device=device)
        scripted_mask = torch.isin(particle_types, scripted)

        scripted_velocity = torch.zeros_like(x_last)
        scripted_velocity[scripted_mask] = (next_positions - x_last)[scripted_mask]

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

    def save(self, path: str | Path) -> None:
        """Save the model's ``state_dict`` (network + normalizer buffers).

        Parameters
        ----------
        path:
            Destination path.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load a ``state_dict`` saved by :meth:`save` (mapped to CPU).

        Parameters
        ----------
        path:
            Source path of the saved state dict.
        """
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
