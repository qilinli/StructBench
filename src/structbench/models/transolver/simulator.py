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
        ``k*(dim+C)`` and makes :meth:`predict_positions` / :meth:`forward_train`
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
        impact_velocity_feature: bool = False,
        time_conditioned: bool = False,
        adaptive_temperature: bool = False,
        slice_reparam: bool = False,
        n_aux: int = 1,
        aux_input: bool = False,
        flow_map: bool = False,
        flow_map_anchor_time: bool = True,
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
        # k*(dim+C); the (P, k*(dim+C)) -> (P, k, dim+C) reshape is a
        # simulator-side convention (network.py is k-agnostic) and a no-op at
        # k=1, so k=1 is byte-identical to the pre-0051 recipe.
        if frames_per_call < 1:
            raise ValueError(
                "frames_per_call must be >= 1 at construction "
                f"(got {frames_per_call}); the k=T sentinel (0) must be "
                "resolved to a concrete horizon before the simulator is built"
            )
        self._k = frames_per_call
        # ADR-0051 B: one extra global node channel carrying the case's scalar
        # loading parameter (impact velocity), broadcast to every node. Off by
        # default (byte-identical). Distinct from velocity_history (per-node
        # velocity window); this is the operator-learning/one-shot convention.
        self._impact_velocity_feature = impact_velocity_feature
        # ADR-0054: faithful thuml time-conditioned (Time_Input) scheme. The
        # model maps (static geometry, node types, scalar impact velocity?,
        # scripted BC displacement at t, query time t) -> absolute state at t;
        # it is history-free and non-autoregressive, so it is mutually
        # exclusive with the velocity-history window and the k>1 bundling axis.
        self._time_conditioned = time_conditioned
        if time_conditioned:
            if frames_per_call != 1:
                raise ValueError(
                    "time_conditioned=True requires frames_per_call=1 "
                    f"(got {frames_per_call}); the time-conditioned and "
                    "k-frames-per-call schemes are mutually exclusive (ADR-0054)"
                )
            if history_velocities != 0:
                raise ValueError(
                    "time_conditioned=True requires history_velocities=0 "
                    f"(got {history_velocities}); the time-conditioned scheme "
                    "is history-free (ADR-0054)"
                )
            # TC input = one_hot + reference coords + scripted-BC displacement
            # at t (+ scalar impact velocity). No current-position or velocity
            # window channels: the geometry is static and time enters additively.
            node_in = node_type_size + 2 * dim + (1 if impact_velocity_feature else 0)
            if flow_map:
                # ADR-0062 anchor block: displacement-from-rest + FD velocity
                # at the anchor frame (2*dim), the anchor aux state (n_aux),
                # and optionally the anchor's normalized time (1). The time
                # embedding then carries Δt from the anchor, not absolute t.
                node_in += 2 * dim + n_aux + (1 if flow_map_anchor_time else 0)
        else:
            node_in = (
                node_type_size
                + 3 * dim
                + history_velocities * dim
                + (1 if impact_velocity_feature else 0)
                # ADR-0060: the aux state block at the last input frame.
                + (n_aux if aux_input else 0)
            )

        self._n_aux = n_aux
        # ADR-0060 state-feedback surface (guarded at config load too;
        # defense in depth for programmatic construction). ADR-0062 narrows
        # the TC rejection: the anchored flow map consumes the anchor state
        # through the TC formulation, so the combination is valid exactly
        # when flow_map is on.
        if aux_input and time_conditioned and not flow_map:
            raise ValueError(
                "aux_input is incompatible with time_conditioned "
                "unless flow_map=True (ADR-0060/ADR-0062)"
            )
        if aux_input and frames_per_call != 1:
            raise ValueError("aux_input requires frames_per_call=1 (ADR-0060)")
        if flow_map and not time_conditioned:
            raise ValueError("flow_map requires time_conditioned=True (ADR-0062)")
        if flow_map and not aux_input:
            raise ValueError("flow_map requires aux_input=True (ADR-0062)")
        self._aux_input = aux_input
        self._flow_map = flow_map
        self._flow_map_anchor_time = flow_map_anchor_time
        # ADR-0062 anchor cache (eval path): set via set_anchor(), cleared on
        # bind_case()/reset_rollout(). The training path passes anchor parts
        # per batch instead and never touches the cache.
        self._anchor_disp: Tensor | None = None
        self._anchor_vel: Tensor | None = None
        self._anchor_aux: Tensor | None = None
        self._anchor_t_norm: float | None = None
        #: rollout state-input mode: "self" feeds back the model's own
        #: predicted aux; "oracle" reads ground truth each step (ADR-0060
        #: accumulation-isolation mode). The evaluator switches this.
        self.aux_feedback: str = "self"
        self._aux_state: Tensor | None = None  # last predicted aux (self mode)
        self._aux_t: int | None = None  # frame the NEXT predict call targets
        self._net = TransolverNet(
            node_in=node_in,
            out_size=frames_per_call * (dim + n_aux),  # ADR-0059
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            slice_num=slice_num,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            time_conditioned=time_conditioned,
            adaptive_temperature=adaptive_temperature,
            slice_reparam=slice_reparam,
        )
        self._node_normalizer = OnlineNormalizer(node_in)
        self._target_normalizer = OnlineNormalizer(dim + n_aux)

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

    @property
    def time_conditioned(self) -> bool:
        """Whether this simulator uses the ADR-0054 time-conditioned scheme.

        Read by :mod:`structbench.eval` and :mod:`structbench.cli.train` to
        route to the independent-query eval/training path (no autoregressive
        rollout, no one-step teacher forcing) instead of :meth:`predict_positions`.
        """
        return self._time_conditioned

    @property
    def flow_map(self) -> bool:
        """Whether this simulator uses the ADR-0062 anchored-flow-map mode.

        Introspection parity with :attr:`time_conditioned`; the routing
        decision itself is made from the run config (``cfg.flow_map``) in
        :mod:`structbench.cli.train`, before the simulator is built.
        """
        return self._flow_map

    def _features(
        self,
        one_hot: Tensor,
        scripted_velocity: Tensor,
        x_t: Tensor,
        reference_coords: Tensor,
        velocity_history: Tensor | None = None,
        loading_feature: Tensor | None = None,
        aux_state: Tensor | None = None,
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
        aux_state:
            ``(P, C)`` aux state block at the last input frame (ADR-0060);
            required exactly when the simulator was built with
            ``aux_input=True``, forbidden-by-ignore otherwise.

        Returns
        -------
        Tensor
            ``(P, node_type_size + (3 + history_velocities) * dim [+ 1]
            [+ C])`` raw node features: ``cat([one_hot, scripted_velocity,
            x_t, reference_coords, velocity_history?, loading_feature?,
            aux_state?], -1)``.
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
        if self._impact_velocity_feature:
            if loading_feature is None:
                raise ValueError(
                    "simulator was built with impact_velocity_feature=True but "
                    "no loading_feature (case impact-velocity scalar) was "
                    "supplied"
                )
            parts.append(loading_feature)
        if self._aux_input:
            if aux_state is None:
                raise ValueError(
                    "simulator was built with aux_input=True but no aux_state "
                    "block was supplied (ADR-0060)"
                )
            parts.append(aux_state)
        return torch.cat(parts, dim=-1)

    def train_output_aux(self, pred_norm: Tensor) -> Tensor:
        """Raw-unit aux block of a ``k=1`` ``forward_train`` output.

        ADR-0061 pushforward helper: step B of the state-channel chain feeds
        step A's predicted state back as ``input_aux``, which lives in raw
        working units — this inverts the target normalizer and slices the
        trailing aux block. ``(P, dim+C)`` normalized -> ``(P, C)`` raw.
        """
        return self._target_normalizer.inverse(pred_norm)[..., self._dim :]

    def train_output_state(self, pred_norm: Tensor) -> tuple[Tensor, Tensor]:
        """Raw-unit ``(displacement, aux)`` blocks of a TC/FM training output.

        ADR-0063 chain helper: rebuilding a self-anchor from a step-A
        ``forward_train_tc`` prediction needs BOTH raw blocks — the anchor
        displacement/FD-velocity from the displacement slice and the anchor
        state from the aux slice (``train_output_aux`` covers only the
        latter). Inverts the target normalizer once and splits:
        ``(P, dim+C)`` normalized -> (``(P, dim)`` raw displacement-from-rest,
        ``(P, C)`` raw aux).
        """
        raw = self._target_normalizer.inverse(pred_norm)
        return raw[..., : self._dim], raw[..., self._dim :]

    def reset_rollout(self) -> None:
        """Reset the step pointer, the ADR-0060 state cache, and the anchor."""
        super().reset_rollout()
        self._aux_state = None
        self._aux_t = None
        self._clear_anchor()

    def _on_bind_case(self, cells: Tensor) -> None:
        """Clear the ADR-0060 state cache and ADR-0062 anchor for the new case."""
        del cells  # no static connectivity to derive (operator family)
        self._aux_state = None
        self._aux_t = None
        self._clear_anchor()

    def _clear_anchor(self) -> None:
        """Drop the ADR-0062 anchor cache (stale-anchor tripwire support)."""
        self._anchor_disp = None
        self._anchor_vel = None
        self._anchor_aux = None
        self._anchor_t_norm = None

    def set_anchor(
        self,
        frame: int,
        prev_positions: Tensor,
        positions: Tensor,
        aux: Tensor,
        anchor_t_norm: float | None = None,
    ) -> None:
        """Bind the ADR-0062 flow-map anchor for subsequent queries.

        The anchor is the scheme's only inter-segment interface: subsequent
        :meth:`predict_state_at` calls condition on the state cached here (and
        on the Δt the caller passes as ``t_norm``). The re-anchoring rollout
        calls this at every segment boundary — with the model's own
        predictions (self-anchored) or ground truth (oracle-anchored).

        Parameters
        ----------
        frame:
            The anchor frame index ``t0`` (used only for the kinematic-row
            clamp below; the time feature comes from ``anchor_t_norm``).
        prev_positions:
            ``(P, dim)`` world positions at ``t0 - 1`` — the FD-velocity
            partner frame (the AR families' ``time_diff`` convention,
            un-divided by dt; ADR-0062 anchor contract).
        positions:
            ``(P, dim)`` world positions at ``t0``.
        aux:
            ``(P, C)`` aux state at ``t0``. KINEMATIC rows are overwritten
            with the bound GT aux at ``t0`` (the ADR-0060 house clamp: those
            rows receive no aux training signal, so a predicted value there
            is untargeted decoder output, not state).
        anchor_t_norm:
            The anchor's normalized time ``t0 / (time_ref - 1)``; required
            exactly when built with ``flow_map_anchor_time=True``.

        Raises
        ------
        RuntimeError
            If called before :meth:`~.CaseBoundSimulator.bind_case`, or on a
            non-flow-map simulator.
        """
        if not self._flow_map:
            raise RuntimeError("set_anchor() requires flow_map=True (ADR-0062)")
        reference_coords = self._reference_coords
        if reference_coords is None:
            raise RuntimeError(
                "set_anchor() called before bind_case(); bind_case() must be "
                "called with the case being evaluated first"
            )
        if self._flow_map_anchor_time and anchor_t_norm is None:
            raise ValueError(
                "simulator was built with flow_map_anchor_time=True but "
                "set_anchor() received no anchor_t_norm"
            )
        anchor_aux = aux.detach()
        if self._has_kinematic and self._kin_mask is not None:
            # The clamp is REQUIRED, not best-effort: kinematic rows carry no
            # aux training signal, so silently anchoring on a predicted value
            # there would corrupt every query in the segment with no error.
            if self._gt_aux is None or frame >= self._gt_aux.shape[0]:
                raise RuntimeError(
                    "set_anchor() cannot clamp kinematic aux rows: bind_case()"
                    f" supplied no gt_aux covering frame {frame} (the "
                    "ADR-0060 house clamp is mandatory under flow_map; "
                    "ADR-0062)"
                )
            anchor_aux = anchor_aux.clone()
            anchor_aux[self._kin_mask] = self._gt_aux[frame][self._kin_mask]
        self._anchor_disp = (positions - reference_coords).detach()
        self._anchor_vel = (positions - prev_positions).detach()
        self._anchor_aux = anchor_aux
        self._anchor_t_norm = (
            float(anchor_t_norm) if anchor_t_norm is not None else None
        )

    def _loading_feature(self, ref: Tensor) -> Tensor | None:
        """Broadcast the bound case's scalar loading parameter to ``(P, 1)``.

        Returns ``None`` when ``impact_velocity_feature`` is off (ADR-0051 B).
        Used on the eval path, where the scalar comes from
        :meth:`~.CaseBoundSimulator.bind_case`; the training path passes its
        own collated ``loading_feature`` instead.
        """
        if not self._impact_velocity_feature:
            return None
        if self._loading_scalar is None:
            raise RuntimeError(
                "impact_velocity_feature is on but bind_case() supplied no "
                "loading_scalar for this case"
            )
        return torch.full(
            (ref.shape[0], 1),
            float(self._loading_scalar),
            dtype=ref.dtype,
            device=ref.device,
        )

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
            ``(next_positions (P, dim), aux (P, C))``. At ``k>1``:
            ``(next_positions (P, k, dim), aux (P, k, C))`` — the ``k`` bundled
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
        loading_feature = self._loading_feature(x_t)

        aux_state: Tensor | None = None
        if self._aux_input:
            # ADR-0060 state input for the frame this call predicts. The
            # counter anchors at the window frame count F on the first call
            # (predicting frame F needs the state at F-1) and advances by 1
            # per call — valid for k=1 (enforced) in both the rollout and the
            # teacher-forced one-step sweep, which slides the window by 1.
            gt_aux = self._gt_aux
            if gt_aux is None:
                raise RuntimeError(
                    "aux_input=True but bind_case() supplied no gt_aux "
                    "trajectory for this case (ADR-0060)"
                )
            if self._aux_t is None:
                self._aux_t = n_frames
            if self._aux_t - 1 >= gt_aux.shape[0]:
                # Loud-failure parity with the position pointer's tripwire
                # (which no-ops on kinematic-free cases): a counter past the
                # bound trajectory means a missing reset between eval passes.
                raise RuntimeError(
                    "aux-state counter is past the bound gt_aux trajectory "
                    f"(frame {self._aux_t} of {gt_aux.shape[0]}): call "
                    "reset_rollout() before each eval pass, or re-bind_case() "
                    "the trajectory being evaluated (ADR-0060)"
                )
            if self.aux_feedback == "oracle":
                aux_state = gt_aux[self._aux_t - 1]
            elif self.aux_feedback == "self":
                # seed from the GT state at the last seed frame; thereafter
                # the model's own prediction feeds back.
                aux_state = (
                    self._aux_state
                    if self._aux_state is not None
                    else gt_aux[n_frames - 1]
                )
            else:
                raise ValueError(
                    f"aux_feedback must be 'self' or 'oracle', got "
                    f"{self.aux_feedback!r}"
                )

        node_feats_raw = self._features(
            node_type_onehot,
            scripted_velocity,
            x_t,
            reference_coords,
            velocity_history,
            loading_feature,
            aux_state,
        )
        node_feats = self._node_normalizer(node_feats_raw, accumulate=False)

        out = self._net(node_feats, None)  # (P, k*(dim+C))
        if self._k == 1:
            # k=1 path. Inverse-normalize the FULL (P, dim+C)
            # output first -- slicing before inverse would broadcast the
            # dim-wide velocity slice against the (dim+C)-wide std/mean buffers.
            out = self._target_normalizer.inverse(out)
            velocity = out[:, : self._dim]
            stress = out[:, self._dim :]
            next_positions = x_t + velocity
            if self._aux_input:
                # ADR-0060: cache the prediction for the self-fed mode's next
                # step, then advance the state counter. KINEMATIC rows are
                # overwritten with the GT state at the just-predicted frame —
                # the exact analog of the rollout loop's GT position override:
                # those rows receive no aux training signal (their loss is
                # masked), so the decoder's output there is untargeted, and
                # feeding it back would contaminate the self-fed mode with a
                # train/rollout distribution shift unrelated to state-channel
                # accumulation (the effect modes 2 vs 3 exist to isolate).
                cached = stress.detach()
                t_pred = self._aux_t if self._aux_t is not None else n_frames
                if (
                    self._has_kinematic
                    and self._gt_aux is not None
                    and self._kin_mask is not None
                    and t_pred < self._gt_aux.shape[0]
                ):
                    cached = cached.clone()
                    cached[self._kin_mask] = self._gt_aux[t_pred][self._kin_mask]
                self._aux_state = cached
                self._aux_t = t_pred + 1
            return next_positions, stress

        # k>1: reshape to (P, k, dim+C), inverse-normalize on the row-folded
        # (P*k, dim+C) view (same width-(dim+C) stats as k=1), then integrate
        # the k per-frame velocities from x_t to k absolute positions.
        return self._decode_positions(out, x_t)

    def _decode_positions(
        self, pred_norm: Tensor, x_last: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Integrate a normalized k-frame output into ``k`` absolute positions.

        Shared by :meth:`predict_positions` (eval) and the 1<k<T pushforward
        seam in the training loop: inverse-normalize the ``(P, k*(dim+C))``
        network output on the row-folded ``(P*k, dim+C)`` view (the same
        width-(dim+C) target stats k=1 uses), then integrate the ``k``
        per-frame velocities from ``x_last`` via ``cumsum``.

        Parameters
        ----------
        pred_norm:
            ``(P, k*(dim+C))`` raw network output (normalized/target space).
        x_last:
            ``(P, dim)`` positions to integrate the velocities from.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(positions (P, k, dim), aux (P, k, C))``.
        """
        p, k, dim = pred_norm.shape[0], self._k, self._dim
        w = dim + self._n_aux
        out = self._target_normalizer.inverse(pred_norm.reshape(p * k, w))
        out = out.reshape(p, k, w)
        velocity = out[:, :, :dim]  # (P, k, dim)
        stress = out[:, :, dim:]  # (P, k, 1)
        next_positions = x_last.unsqueeze(1) + torch.cumsum(velocity, dim=1)
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
        loading_feature: Tensor | None = None,
        input_aux: Tensor | None = None,
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
            Ground-truth aux channels (working-frame units, e.g. MPa;
            ADR-0059): ``(P, C)`` at ``k=1``; ``(P, k, C)`` at ``k>1``.
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
            ``(P, dim + C)`` at ``k=1`` (target
            ``cat([next_positions - x_last, next_aux], dim=1)``) and
            ``(P, k, dim + C)`` at ``k>1`` (per-frame running-integration
            velocity targets ``cat([Δpositions, next_aux], -1)``,
            normalized on the row-folded ``(P*k, dim+C)`` view).

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
            one_hot,
            scripted_velocity,
            x_last,
            reference_coords,
            velocity_history,
            loading_feature,
            input_aux,
        )
        node_feats = self._node_normalizer(node_feats_raw, accumulate=accumulate)

        pred_norm = self._net(node_feats, n_particles_per_example)  # (P, k*(dim+C))

        if self._k == 1:
            # Byte-identical k=1 path.
            target_raw = torch.cat([next_positions - x_last, next_aux], dim=1)
            target_norm = self._target_normalizer(target_raw, accumulate=accumulate)
            return pred_norm, target_norm

        # k>1: per-frame velocity targets via running integration
        # (v_0 = next[:,0] - x_last; v_j = next[:,j] - next[:,j-1]), so every
        # one of the k targets stays one-step scaled and the width-(dim+C)
        # target normalizer stays valid (fed the row-folded (P*k, dim+C) view).
        p, k, dim = x_last.shape[0], self._k, self._dim
        running = torch.cat([x_last.unsqueeze(1), next_positions], dim=1)
        velocities = running[:, 1:] - running[:, :-1]  # (P, k, dim)
        w = dim + self._n_aux
        target_raw = torch.cat([velocities, next_aux], dim=-1)
        target_norm = self._target_normalizer(
            target_raw.reshape(p * k, w), accumulate=accumulate
        ).reshape(p, k, w)
        pred_norm = pred_norm.reshape(p, k, w)
        return pred_norm, target_norm

    # --- ADR-0054 time-conditioned scheme -----------------------------------

    def _kinematic_bc(
        self, gt_position: Tensor, reference_coords: Tensor, kin_mask: Tensor
    ) -> Tensor:
        """Scripted-boundary input channel: kinematic-node displacement at t.

        The faithful thuml time-conditioned scheme (ADR-0054, decision A)
        feeds the prescribed boundary state at the queried time ``t`` as an
        input channel — the analog of Plasticity's die state. Here that is the
        kinematic (prescribed-motion) nodes' displacement from rest at frame
        ``t``: ``gt_position - reference_coords`` on kinematic rows, zero on
        every free (NORMAL) row. Kinematic positions are known boundary
        conditions, so this is legitimate conditioning, not leakage.

        Parameters
        ----------
        gt_position:
            ``(P, dim)`` ground-truth world positions at the queried frame.
        reference_coords:
            ``(P, dim)`` mesh-space (rest/reference) coordinates.
        kin_mask:
            ``(P,)`` bool mask, ``True`` on kinematic (prescribed) rows.

        Returns
        -------
        Tensor
            ``(P, dim)`` displacement on kinematic rows, zero elsewhere.
        """
        bc = torch.zeros_like(reference_coords)
        bc[kin_mask] = gt_position[kin_mask] - reference_coords[kin_mask]
        return bc

    def _features_tc(
        self,
        one_hot: Tensor,
        reference_coords: Tensor,
        kinematic_bc: Tensor,
        loading_feature: Tensor | None,
        anchor_disp: Tensor | None = None,
        anchor_vel: Tensor | None = None,
        anchor_aux: Tensor | None = None,
        anchor_time_feature: Tensor | None = None,
    ) -> Tensor:
        """Build the raw (pre-normalization) time-conditioned node features.

        ``cat([one_hot, reference_coords, kinematic_bc, loading_feature?,
        anchor_disp?, anchor_vel?, anchor_aux?, anchor_time_feature?])``
        (ADR-0054 / ADR-0062): the static geometry (node type + rest coords),
        the prescribed boundary displacement at the queried time, optionally
        the case's scalar impact velocity, and — under ``flow_map`` — the
        ADR-0062 anchor block (displacement-from-rest ``(P, dim)``, FD
        velocity ``(P, dim)``, and aux state ``(P, C)`` at the anchor frame,
        plus the anchor's normalized time as a ``(P, 1)`` broadcast scalar
        when ``flow_map_anchor_time`` is on). No current-position or velocity
        channels of the QUERY frame — time (Δt from the anchor, under
        ``flow_map``) enters additively inside the network, not here.
        """
        parts = [one_hot, reference_coords, kinematic_bc]
        if self._impact_velocity_feature:
            if loading_feature is None:
                raise ValueError(
                    "simulator was built with impact_velocity_feature=True but "
                    "no loading_feature (case impact-velocity scalar) was supplied"
                )
            parts.append(loading_feature)
        if self._flow_map:
            if anchor_disp is None or anchor_vel is None or anchor_aux is None:
                raise ValueError(
                    "simulator was built with flow_map=True but the anchor "
                    "block (anchor_disp/anchor_vel/anchor_aux) was not "
                    "supplied (ADR-0062)"
                )
            parts.extend([anchor_disp, anchor_vel, anchor_aux])
            if self._flow_map_anchor_time:
                if anchor_time_feature is None:
                    raise ValueError(
                        "simulator was built with flow_map_anchor_time=True "
                        "but no anchor_time_feature was supplied (ADR-0062)"
                    )
                parts.append(anchor_time_feature)
        # NB: no last-input-frame aux_state part here — plain-TC aux_input is
        # rejected (ADR-0060); the flow map consumes state via the anchor
        # block above (ADR-0062).
        return torch.cat(parts, dim=-1)

    def forward_train_tc(
        self,
        gt_position: Tensor,
        gt_aux: Tensor,
        particle_types: Tensor,
        reference_coords: Tensor,
        n_particles_per_example: Tensor,
        t_norm: Tensor,
        *,
        accumulate: bool,
        loading_feature: Tensor | None = None,
        anchor_disp: Tensor | None = None,
        anchor_vel: Tensor | None = None,
        anchor_aux: Tensor | None = None,
        anchor_time_feature: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """One time-conditioned training forward pass (ADR-0054).

        Maps ``(static geometry, node types, scalar impact velocity?, scripted
        BC at t, query time t) -> displacement-from-rest at t`` plus aux, with
        no history and no autoregressive feedback. The regressed target is the
        rest-frame displacement (``gt_position - reference_coords``), a
        well-conditioned parameterisation of the absolute state at ``t``
        (position is recovered as ``reference_coords + displacement``); the aux
        channel is the raw GT aux at ``t``.

        Parameters
        ----------
        gt_position:
            ``(P, dim)`` ground-truth world positions at the queried frames.
        gt_aux:
            ``(P, C)`` ground-truth aux channels at the queried frames (ADR-0059).
        particle_types:
            ``(P,)`` int64 node-type codes.
        reference_coords:
            ``(P, dim)`` mesh-space (rest/reference) coordinates.
        n_particles_per_example:
            ``(B,)`` int64 per-example node counts (ragged-batch segments and
            the per-example time broadcast).
        t_norm:
            ``(B,)`` per-example normalized query time ``t ∈ [0, 1]`` — or,
            under ``flow_map`` (ADR-0062), the normalized OFFSET from the
            anchor, ``Δt / (time_ref - 1)``.
        accumulate:
            Fold this call's features into the online node/target normalizers.
        loading_feature:
            ``(P, 1)`` scalar impact-velocity channel; required exactly when
            ``impact_velocity_feature`` is on.
        anchor_disp, anchor_vel, anchor_aux:
            ADR-0062 anchor block at each sample's anchor frame ``t0``
            (``(P, dim)`` displacement-from-rest, ``(P, dim)`` FD velocity,
            ``(P, C)`` aux state — the aux ADR-0061-noised by the caller
            when the knob is on); required exactly when built with
            ``flow_map=True``.
        anchor_time_feature:
            ``(P, 1)`` broadcast anchor time ``t0 / (time_ref - 1)``;
            required exactly when built with ``flow_map_anchor_time=True``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(pred_norm, target_norm)``, each ``(P, dim + C)`` in the target
            normalizer's space, for the standard position(+aux) RMSE loss.
        """
        one_hot = F.one_hot(particle_types, num_classes=self._node_type_size).to(
            torch.float32
        )
        kinematic = torch.as_tensor(
            self._kinematic_types,
            dtype=particle_types.dtype,
            device=particle_types.device,
        )
        kin_mask = torch.isin(particle_types, kinematic)
        kinematic_bc = self._kinematic_bc(gt_position, reference_coords, kin_mask)

        node_feats_raw = self._features_tc(
            one_hot,
            reference_coords,
            kinematic_bc,
            loading_feature,
            anchor_disp=anchor_disp,
            anchor_vel=anchor_vel,
            anchor_aux=anchor_aux,
            anchor_time_feature=anchor_time_feature,
        )
        node_feats = self._node_normalizer(node_feats_raw, accumulate=accumulate)

        pred_norm = self._net(node_feats, n_particles_per_example, t=t_norm)

        target_raw = torch.cat([gt_position - reference_coords, gt_aux], dim=1)
        target_norm = self._target_normalizer(target_raw, accumulate=accumulate)
        return pred_norm, target_norm

    def predict_state_at(self, frame: int, t_norm: float) -> tuple[Tensor, Tensor]:
        """Predict the absolute state at a single scored frame (ADR-0054 eval).

        Independent-query, history-free: for the bound case's queried frame,
        assemble the static TC features (rest coords + node types + scalar
        impact velocity? + scripted BC at ``frame``), run the network at the
        normalized time ``t_norm``, reconstruct absolute positions
        (``reference_coords + predicted displacement``), then OVERRIDE the
        kinematic rows with their ground-truth positions at ``frame`` (their
        motion is prescribed). No step pointer / tripwire is used — every
        frame is an independent query.

        Parameters
        ----------
        frame:
            Ground-truth frame index to query (supplies the scripted BC input
            and the kinematic override).
        t_norm:
            Normalized query time for ``frame`` (``frame`` over the scored
            horizon; ADR-0054), a Python float. Under ``flow_map``
            (ADR-0062) the caller instead passes the normalized OFFSET from
            the current anchor, ``(frame - t0) / (time_ref - 1)``, and the
            query additionally conditions on the anchor cached by
            :meth:`set_anchor` (required — a missing anchor raises).

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(positions (P, dim), aux (P, C))`` at ``frame``.

        Raises
        ------
        RuntimeError
            If called before :meth:`~.CaseBoundSimulator.bind_case`.
        """
        reference_coords = self._reference_coords
        node_type_onehot = self._node_type_onehot
        kin_mask = self._kin_mask
        gt_positions = self._gt_positions
        if (
            reference_coords is None
            or node_type_onehot is None
            or kin_mask is None
            or gt_positions is None
        ):
            raise RuntimeError(
                "TransolverSimulator.predict_state_at() called before "
                "bind_case(); bind_case() must be called with the case being "
                "evaluated before any prediction."
            )

        gt_frame = gt_positions[frame]
        kinematic_bc = self._kinematic_bc(gt_frame, reference_coords, kin_mask)
        loading_feature = self._loading_feature(reference_coords)

        anchor_time_feature: Tensor | None = None
        if self._flow_map:
            if self._anchor_aux is None:
                raise RuntimeError(
                    "flow_map=True but no anchor is set: call set_anchor() "
                    "before querying (the re-anchoring rollout does this; "
                    "ADR-0062)"
                )
            if self._flow_map_anchor_time:
                assert self._anchor_t_norm is not None  # set_anchor enforces
                anchor_time_feature = torch.full(
                    (reference_coords.shape[0], 1),
                    self._anchor_t_norm,
                    dtype=reference_coords.dtype,
                    device=reference_coords.device,
                )
        node_feats_raw = self._features_tc(
            node_type_onehot,
            reference_coords,
            kinematic_bc,
            loading_feature,
            anchor_disp=self._anchor_disp,
            anchor_vel=self._anchor_vel,
            anchor_aux=self._anchor_aux,
            anchor_time_feature=anchor_time_feature,
        )
        node_feats = self._node_normalizer(node_feats_raw, accumulate=False)

        t = torch.tensor(
            [[float(t_norm)]],
            dtype=reference_coords.dtype,
            device=reference_coords.device,
        )
        out = self._net(node_feats, None, t=t)  # (P, dim+C)
        out = self._target_normalizer.inverse(out)
        displacement = out[:, : self._dim]
        stress = out[:, self._dim :]
        positions = (reference_coords + displacement).clone()
        positions[kin_mask] = gt_frame[kin_mask]
        return positions, stress
