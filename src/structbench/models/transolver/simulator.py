"""``TransolverSimulator``: Transolver's stateful rollout wrapper (ADR-0041/0043/0044).

The full per-case-binding statefulness contract (bind per case, reset before
each eval pass, step pointer, tripwire semantics) is inherited from
:class:`~structbench.models.common.simulator_base.CaseBoundSimulator` and
documented in that module's docstring — it is not repeated here. This module
covers what is Transolver-specific: network sizing and the point-set feature
builder.

**Edge-free sibling of MGN.** Unlike
:class:`~structbench.models.mgn.simulator.MeshSimulator`, there are no
graphs and no edges here: node features are
``cat([one_hot, scripted_velocity, x_t, reference_coords], -1)`` =
``node_type_size + 3 * dim`` channels (18 for the ADR-0043 recipe's
``node_type_size=9``, ``dim=3``), and :class:`~.network.TransolverNet` is
called with ``n_particles_per_example`` (batched training) or ``None``
(single bound-case eval, the inference fast path). A velocity-history run
(ADR-0049, ``history_velocities > 0``) appends the window's flattened
finite-difference velocities, adding ``history_velocities * dim`` channels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..common import CaseBoundSimulator
from ..mgn.normalizers import OnlineNormalizer
from .network import TransolverNet


class TransolverSimulator(CaseBoundSimulator):
    """Transolver (Physics-Attention) simulator with per-case GT binding.

    See :class:`~structbench.models.common.simulator_base.CaseBoundSimulator`'s
    module docstring for the full statefulness contract (bind per case,
    reset before each eval pass, tripwire semantics) — it is not repeated
    here.

    Parameters
    ----------
    dim:
        Spatial dimensionality of node positions.
    hidden_dim:
        Channel width, forwarded to :class:`~.network.TransolverNet`.
    n_layers:
        Number of Transolver blocks, forwarded to
        :class:`~.network.TransolverNet`.
    n_heads:
        Number of Physics-Attention heads per block, forwarded to
        :class:`~.network.TransolverNet`.
    slice_num:
        Number of Physics-Attention slice tokens, forwarded to
        :class:`~.network.TransolverNet`.
    mlp_ratio:
        FFN hidden-width multiplier inside each block, forwarded to
        :class:`~.network.TransolverNet`.
    dropout:
        Dropout probability, forwarded to :class:`~.network.TransolverNet`.
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
    history_velocities:
        Number of finite-difference window velocities appended to the node
        features (ADR-0049); ``0`` keeps the reference Markovian-in-position
        feature builder.
    frames_per_call:
        Frames the decoder emits per forward call (ADR-0050/0051 ``k``),
        already resolved to a concrete count (the k=T sentinel ``0`` is
        resolved by :func:`~structbench.cli.train.build_transolver_simulator`
        before construction). ``1`` (default) is the byte-identical
        autoregressive scheme; ``k>1`` widens the decoder head to
        ``k*(dim+1)`` and makes :meth:`predict_positions` / :meth:`forward_train`
        emit ``k`` per-frame velocities (integrated on eval). The 1<k<T
        training-seam pushforward lives in the training loop, not here.
    device:
        Device the network and normalizer buffers are moved to at
        construction time.
    """

    def __init__(
        self,
        dim: int = 3,
        hidden_dim: int = 128,
        n_layers: int = 8,
        n_heads: int = 8,
        slice_num: int = 64,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
        node_type_size: int = 9,
        kinematic_types: tuple[int, ...] = (1, 3),
        scripted_types: tuple[int, ...] = (1,),
        history_velocities: int = 0,
        frames_per_call: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            dim=dim,
            node_type_size=node_type_size,
            kinematic_types=kinematic_types,
            scripted_types=scripted_types,
            history_velocities=history_velocities,
            # The base only uses this to call self.to(device) before any
            # parameters of THIS subclass exist yet (see the final
            # self.to(device) below, which does the real work); str(device)
            # keeps the base's str-only signature untouched while accepting
            # this subclass's wider str | torch.device parameter.
            device=str(device),
        )
        # frames_per_call is the k of the ADR-0050/0051 prediction-scheme axis,
        # already resolved to a concrete count (the k=T sentinel 0 is resolved
        # by build_transolver_simulator BEFORE construction, so a 0 reaching
        # here is an unresolved-sentinel bug). k>1 widens the decoder head to
        # k*(dim+1); the (P, k*(dim+1)) -> (P, k, dim+1) reshape is a
        # simulator-side convention (network.py is k-agnostic) and a no-op at
        # k=1, so k=1 is byte-identical to the pre-0051 recipe.
        if frames_per_call < 1:
            raise ValueError(
                "frames_per_call must be >= 1 at construction "
                f"(got {frames_per_call}); the k=T sentinel (0) must be "
                "resolved to a concrete horizon before the simulator is built"
            )
        self._k = frames_per_call
        node_in = node_type_size + 3 * dim + history_velocities * dim

        self._net = TransolverNet(
            node_in=node_in,
            out_size=frames_per_call * (dim + 1),
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            slice_num=slice_num,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self._node_normalizer = OnlineNormalizer(node_in)
        self._target_normalizer = OnlineNormalizer(dim + 1)

        self.to(device)

    @property
    def frames_per_call(self) -> int:
        """Resolved ``k`` (frames emitted per forward call; ADR-0050/0051).

        Read by :mod:`structbench.eval` to decide whether a returned bundle is
        a single frame (``k=1``, 2-D) or a ``(P, k, dim)`` stack, and to switch
        the one-step sweep into teacher-forced (pointer-by-1) mode for ``k>1``.
        Families that never bundle simply do not expose this attribute.
        """
        return self._k

    def _features(
        self,
        one_hot: Tensor,
        scripted_velocity: Tensor,
        x_t: Tensor,
        reference_coords: Tensor,
        velocity_history: Tensor | None = None,
    ) -> Tensor:
        """Build the raw (pre-normalization) node feature tensor.

        ``one_hot`` is an EXPLICIT parameter (mirrors MGN's
        ``_graph_features`` signature) rather than read off ``self``:
        :meth:`predict_positions` supplies the bound
        ``self._node_type_onehot``; :meth:`forward_train` builds its own
        from its ``particle_types`` argument, since the training path is
        never bound to a case via :meth:`~.CaseBoundSimulator.bind_case`
        (batched collated data instead).

        Parameters
        ----------
        one_hot:
            ``(P, node_type_size)`` float32 one-hot node-type encoding.
        scripted_velocity:
            ``(P, dim)`` scripted-actuator velocity node feature; zero on
            non-scripted rows.
        x_t:
            ``(P, dim)`` current world positions (working-frame units).
        reference_coords:
            ``(P, dim)`` mesh-space (rest/reference) coordinates.
        velocity_history:
            ``(P, history_velocities * dim)`` flattened window velocities
            (ADR-0049); required exactly when the simulator was built with
            ``history_velocities > 0``, ``None`` otherwise.

        Returns
        -------
        Tensor
            ``(P, node_type_size + (3 + history_velocities) * dim)`` raw
            node features: ``cat([one_hot, scripted_velocity, x_t,
            reference_coords, velocity_history?], -1)``.
        """
        parts = [one_hot, scripted_velocity, x_t, reference_coords]
        if self._history_velocities > 0:
            if velocity_history is None:
                raise ValueError(
                    "simulator was built with history_velocities="
                    f"{self._history_velocities} but no velocity_history "
                    "feature was supplied"
                )
            parts.append(velocity_history)
        return torch.cat(parts, dim=-1)

    def predict_positions(
        self,
        current_positions: Tensor,
        nparticles_per_example: Tensor,
        particle_types: Tensor,
        *,
        teacher_forced: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Predict the next ``k`` positions and de-normalized stress.

        Parameters
        ----------
        current_positions:
            ``(P, F, dim)`` position window; ``current_positions[:, -1]`` is
            the most recent (current) frame.
        nparticles_per_example:
            Unused: ``TransolverSimulator`` serves one bound case (a single
            example) at a time. Accepted for ``_SimulatorLike`` protocol
            compatibility with :mod:`structbench.eval`.
        particle_types:
            Unused directly: the bound case's one-hot types and kinematic
            mask, cached by :meth:`~.CaseBoundSimulator.bind_case`, are used
            instead. The caller is expected to pass the same
            ``particle_types`` that were bound.
        teacher_forced:
            Pointer-advance mode (ADR-0050/0051). ``False`` (default) is the
            rollout regime: the pointer advances by ``k`` per call, matching
            the ``k`` frames a bundle consumes. ``True`` is the teacher-forced
            one-step sweep, where the caller re-feeds a ground-truth window
            slid by ONE frame each call and scores only the first predicted
            frame, so the pointer must advance by 1. A no-op at ``k=1`` (the
            advance is 1 either way).

        Returns
        -------
        tuple[Tensor, Tensor]
            At ``k=1`` (byte-identical to the pre-0051 recipe):
            ``(next_positions (P, dim), aux (P, 1))``. At ``k>1``:
            ``(next_positions (P, k, dim), aux (P, k, 1))`` — the ``k`` bundled
            frames, positions obtained by integrating the ``k`` predicted
            per-frame velocities from ``x_t`` (``cumsum``). ``aux`` is the
            de-normalized predicted stress.

        Raises
        ------
        RuntimeError
            If called before :meth:`~.CaseBoundSimulator.bind_case`, or if
            the input window's kinematic rows do not match the bound ground
            truth at the current pointer position (the tripwire; see
            :class:`~structbench.models.common.simulator_base.CaseBoundSimulator`'s
            module docstring).
        """
        del nparticles_per_example, particle_types  # see docstring: unused

        reference_coords = self._reference_coords
        node_type_onehot = self._node_type_onehot
        kin_mask = self._kin_mask
        scripted_mask = self._scripted_mask
        gt_positions = self._gt_positions
        if (
            reference_coords is None
            or node_type_onehot is None
            or kin_mask is None
            or scripted_mask is None
            or gt_positions is None
        ):
            raise RuntimeError(
                "TransolverSimulator.predict_positions() called before "
                "bind_case(); bind_case() must be called with the case "
                "being evaluated before any prediction."
            )

        x_t = current_positions[:, -1].contiguous()
        n_frames = current_positions.shape[1]

        # Rollout advances the pointer by the bundle size k; the teacher-forced
        # one-step sweep slides the GT window by 1 and so advances by 1. At k=1
        # both are 1 (byte-identical). The scripted-velocity INPUT feature stays
        # a single (immediate-next) actuator velocity for all k.
        self._advance_pointer(x_t, n_frames, step=1 if teacher_forced else self._k)
        scripted_velocity = self._eval_scripted_velocity(x_t)
        velocity_history = (
            self._window_velocity_history(current_positions)
            if self._history_velocities > 0
            else None
        )

        node_feats_raw = self._features(
            node_type_onehot,
            scripted_velocity,
            x_t,
            reference_coords,
            velocity_history,
        )
        node_feats = self._node_normalizer(node_feats_raw, accumulate=False)

        out = self._net(node_feats, None)  # (P, k*(dim+1))
        if self._k == 1:
            # Byte-identical k=1 path. Inverse-normalize the FULL (P, dim+1)
            # output first -- slicing before inverse would broadcast the
            # dim-wide velocity slice against the (dim+1)-wide std/mean buffers.
            out = self._target_normalizer.inverse(out)
            velocity = out[:, : self._dim]
            stress = out[:, self._dim :]
            next_positions = x_t + velocity
            return next_positions, stress

        # k>1: reshape to (P, k, dim+1), inverse-normalize on the row-folded
        # (P*k, dim+1) view (same width-(dim+1) stats as k=1), then integrate
        # the k per-frame velocities from x_t to k absolute positions.
        p, k, dim = out.shape[0], self._k, self._dim
        out = self._target_normalizer.inverse(out.reshape(p * k, dim + 1))
        out = out.reshape(p, k, dim + 1)
        velocity = out[:, :, :dim]  # (P, k, dim)
        stress = out[:, :, dim:]  # (P, k, 1)
        next_positions = x_t.unsqueeze(1) + torch.cumsum(velocity, dim=1)
        return next_positions, stress

    def forward_train(
        self,
        x_last: Tensor,
        next_positions: Tensor,
        next_aux: Tensor,
        particle_types: Tensor,
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
            Ground-truth next-frame world positions: ``(P, dim)`` at ``k=1``;
            ``(P, k, dim)`` at ``k>1`` (the k bundled target frames).
        next_aux:
            Ground-truth stress (working-frame units, e.g. MPa): ``(P,)`` at
            ``k=1``; ``(P, k)`` at ``k>1``.
        particle_types:
            ``(P,)`` int64 node-type codes.
        reference_coords:
            ``(P, dim)`` mesh-space (rest/reference) coordinates, as
            produced by
            :func:`~structbench.models.mgn.collate.collate_mesh_samples`.
        n_particles_per_example:
            ``(B,)`` int64 per-example node counts from the collate step;
            forwarded to :class:`~.network.TransolverNet` to drive its
            per-example segment loop (ragged-batch Physics-Attention).
        accumulate:
            If ``True``, this call's features are folded into both
            ``OnlineNormalizer`` running statistics (node and target).
        velocity_history:
            ``(P, history_velocities * dim)`` flattened window velocities
            computed by the caller from the NOISY position window
            (ADR-0049); required exactly when the simulator was built with
            ``history_velocities > 0``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(pred_norm, target_norm)``: the raw network output (already in
            normalized/target space, matching what :meth:`predict_positions`
            inverse-normalizes) and the normalized ground-truth target. Each is
            ``(P, dim + 1)`` at ``k=1`` (target
            ``cat([next_positions - x_last, next_aux[:, None]], dim=1)``) and
            ``(P, k, dim + 1)`` at ``k>1`` (per-frame running-integration
            velocity targets ``cat([Δpositions, next_aux[..., None]], -1)``,
            normalized on the row-folded ``(P*k, dim+1)`` view).

        Notes
        -----
        The one-hot node-type encoding is built here; the scripted-velocity
        node input is delegated to the inherited
        :meth:`~structbench.models.common.simulator_base.CaseBoundSimulator._train_scripted_velocity`
        (``next_positions - x_last`` on scripted rows, zero elsewhere) since
        its source differs from the eval path's bound ground truth; the rest
        of the feature assembly is shared with :meth:`predict_positions` via
        :meth:`_features`.

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
        # The scripted-velocity INPUT feature stays a single (immediate-next)
        # actuator velocity for all k, so at k>1 it reads the FIRST bundle
        # frame; node_in is unchanged (ADR-0051).
        first_next = next_positions if self._k == 1 else next_positions[:, 0]
        scripted_velocity = self._train_scripted_velocity(
            x_last, first_next, particle_types
        )

        node_feats_raw = self._features(
            one_hot, scripted_velocity, x_last, reference_coords, velocity_history
        )
        node_feats = self._node_normalizer(node_feats_raw, accumulate=accumulate)

        pred_norm = self._net(node_feats, n_particles_per_example)  # (P, k*(dim+1))

        if self._k == 1:
            # Byte-identical k=1 path.
            target_raw = torch.cat([next_positions - x_last, next_aux[:, None]], dim=1)
            target_norm = self._target_normalizer(target_raw, accumulate=accumulate)
            return pred_norm, target_norm

        # k>1: per-frame velocity targets via running integration
        # (v_0 = next[:,0] - x_last; v_j = next[:,j] - next[:,j-1]), so every
        # one of the k targets stays one-step scaled and the width-(dim+1)
        # target normalizer stays valid (fed the row-folded (P*k, dim+1) view).
        p, k, dim = x_last.shape[0], self._k, self._dim
        running = torch.cat([x_last.unsqueeze(1), next_positions], dim=1)
        velocities = running[:, 1:] - running[:, :-1]  # (P, k, dim)
        target_raw = torch.cat([velocities, next_aux.unsqueeze(-1)], dim=-1)
        target_norm = self._target_normalizer(
            target_raw.reshape(p * k, dim + 1), accumulate=accumulate
        ).reshape(p, k, dim + 1)
        pred_norm = pred_norm.reshape(p, k, dim + 1)
        return pred_norm, target_norm
