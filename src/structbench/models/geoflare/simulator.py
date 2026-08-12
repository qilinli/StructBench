"""``GeoFlareSimulator``: GeoFLARE rollout wrapper (ADR-0041/0043; ADR-0045, draft).

The full per-case-binding statefulness contract (bind per case, reset before
each eval pass, step pointer, tripwire semantics) is inherited from
:class:`~structbench.models.common.simulator_base.CaseBoundSimulator` and
documented in that module's docstring — it is not repeated here. This module
covers what is GeoFLARE-specific: network sizing and the point-set feature
builder, both a straight sibling of
:class:`~structbench.models.transolver.simulator.TransolverSimulator` (read
that module first) with one architectural difference worth flagging up
front.

**Edge-free, like Transolver.** Node features are ``cat([one_hot,
scripted_velocity, x_t, reference_coords], -1)`` = ``node_type_size + 3 *
dim`` channels (18 for the ADR-0043 recipe's ``node_type_size=9``,
``dim=3``) — identical to Transolver's feature builder.

**Coords are threaded to the net separately from the feature tensor, and
stay RAW.** Unlike Transolver's :class:`~structbench.models.transolver
.network.TransolverNet`, :class:`~structbench.models.geoflare.network
.GeoFlareNet` takes a THIRD positional argument: the point coordinates,
forwarded verbatim to its internal
:class:`~structbench.models.geoflare.context.MultiScaleContext` for the
ball-query geometry pathway. Every call site here passes the SAME raw
working-frame (mm) ``x_t``/``x_last`` used to build the ``x_t``/
``reference_coords`` slices of the node-feature tensor — coordinates are
NOT standardized here and NOT run through either
:class:`~structbench.models.mgn.normalizers.OnlineNormalizer`.
Standardization is internal to ``MultiScaleContext``, is per-example (it
cannot be precomputed once for a ragged batch), and the ball-query radii
(``radii``/``neighbors`` below) are defined against that per-example
standardized frame, not the raw mm frame passed in here. All calls to
``self._net`` are made ALL-POSITIONAL (``(node_feats, coords,
n_particles_per_example)``), matching the keyword-free calling convention
:class:`~structbench.models.geoflare.network.GeoFlareNet` was written to.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..common import CaseBoundSimulator
from ..mgn.normalizers import OnlineNormalizer
from .network import GeoFlareNet


class GeoFlareSimulator(CaseBoundSimulator):
    """GeoFLARE (GALE_FA) simulator with per-case GT binding.

    See :class:`~structbench.models.common.simulator_base.CaseBoundSimulator`'s
    module docstring for the full statefulness contract (bind per case,
    reset before each eval pass, tripwire semantics) — it is not repeated
    here. Sibling of
    :class:`~structbench.models.transolver.simulator.TransolverSimulator`;
    see this module's docstring for the one architectural difference
    (coords threaded to the net as a separate, always-raw argument).

    Parameters
    ----------
    dim:
        Spatial dimensionality of node positions.
    n_hidden:
        Latent channel width, forwarded to
        :class:`~.network.GeoFlareNet` (its preprocess output width and
        the geometry context pathway's ``dim_head_ctx`` derivation — NOT
        the block stack's own, wider effective hidden width).
    n_layers:
        Number of stacked :class:`~.network.GeoFlareBlock`, forwarded to
        :class:`~.network.GeoFlareNet`.
    n_heads:
        Number of attention heads per block, forwarded to
        :class:`~.network.GeoFlareNet`.
    slice_num:
        Number of FLARE global queries / context slice tokens, forwarded
        to :class:`~.network.GeoFlareNet`.
    mlp_ratio:
        FFN hidden-width multiplier inside each block, forwarded to
        :class:`~.network.GeoFlareNet`.
    dropout:
        Dropout probability, forwarded to :class:`~.network.GeoFlareNet`.
    n_hidden_local:
        Per-scale local-feature width produced by the ball-query geometric
        feature processor, forwarded to :class:`~.network.GeoFlareNet`.
    radii:
        ``(near, far)`` ball-query radii, in per-example STANDARDIZED
        coordinate units (:func:`~structbench.models.geoflare.geo_ops
        .standardize_coords`), forwarded to :class:`~.network.GeoFlareNet`.
    neighbors:
        ``(near, far)`` neighbour caps, paired with ``radii`` by position,
        forwarded to :class:`~.network.GeoFlareNet`.
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
    device:
        Device the network and normalizer buffers are moved to at
        construction time.
    """

    def __init__(
        self,
        dim: int = 3,
        n_hidden: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        slice_num: int = 128,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        n_hidden_local: int = 32,
        radii: tuple[float, float] = (0.05, 0.25),
        neighbors: tuple[int, int] = (8, 32),
        node_type_size: int = 9,
        kinematic_types: tuple[int, ...] = (1, 3),
        scripted_types: tuple[int, ...] = (1,),
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            dim=dim,
            node_type_size=node_type_size,
            kinematic_types=kinematic_types,
            scripted_types=scripted_types,
            # The base only uses this to call self.to(device) before any
            # parameters of THIS subclass exist yet (see the final
            # self.to(device) below, which does the real work); str(device)
            # keeps the base's str-only signature untouched while accepting
            # this subclass's wider str | torch.device parameter.
            device=str(device),
        )
        node_in = node_type_size + 3 * dim

        self._net = GeoFlareNet(
            node_in=node_in,
            out_size=dim + 1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            slice_num=slice_num,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            n_hidden_local=n_hidden_local,
            radii=radii,
            neighbors=neighbors,
            dim=dim,
        )
        self._node_normalizer = OnlineNormalizer(node_in)
        self._target_normalizer = OnlineNormalizer(dim + 1)

        self.to(device)

    def _features(
        self,
        one_hot: Tensor,
        scripted_velocity: Tensor,
        x_t: Tensor,
        reference_coords: Tensor,
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

        Returns
        -------
        Tensor
            ``(P, node_type_size + 3 * dim)`` raw node features:
            ``cat([one_hot, scripted_velocity, x_t, reference_coords], -1)``.
        """
        return torch.cat([one_hot, scripted_velocity, x_t, reference_coords], dim=-1)

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
            Unused: ``GeoFlareSimulator`` serves one bound case (a single
            example) at a time. Accepted for ``_SimulatorLike`` protocol
            compatibility with :mod:`structbench.eval`.
        particle_types:
            Unused directly: the bound case's one-hot types and kinematic
            mask, cached by :meth:`~.CaseBoundSimulator.bind_case`, are used
            instead. The caller is expected to pass the same
            ``particle_types`` that were bound.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(next_positions (P, dim), aux (P, 1))``. ``aux`` is the
            de-normalized predicted stress, kept 2-D per the
            ``_SimulatorLike`` contract.

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
                "GeoFlareSimulator.predict_positions() called before "
                "bind_case(); bind_case() must be called with the case "
                "being evaluated before any prediction."
            )

        x_t = current_positions[:, -1].contiguous()
        n_frames = current_positions.shape[1]

        self._advance_pointer(x_t, n_frames)
        scripted_velocity = self._eval_scripted_velocity(x_t)

        node_feats_raw = self._features(
            node_type_onehot, scripted_velocity, x_t, reference_coords
        )
        node_feats = self._node_normalizer(node_feats_raw, accumulate=False)

        # Coords are the RAW mm-frame x_t, NOT the normalized feats slice
        # and NOT pre-standardized: GeoFlareNet's internal MultiScaleContext
        # standardizes per example (see module docstring).
        out = self._net(node_feats, x_t, None)
        # Inverse-normalize the FULL (P, dim+1) output first -- slicing
        # before inverse would broadcast the dim-wide velocity slice against
        # the (dim+1)-wide std/mean buffers.
        out = self._target_normalizer.inverse(out)
        velocity = out[:, : self._dim]
        stress = out[:, self._dim :]

        next_positions = x_t + velocity
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
        reference_coords:
            ``(P, dim)`` mesh-space (rest/reference) coordinates, as
            produced by
            :func:`~structbench.models.mgn.collate.collate_mesh_samples`.
        n_particles_per_example:
            ``(B,)`` int64 per-example node counts from the collate step;
            forwarded to :class:`~.network.GeoFlareNet` to drive its
            per-example segment loop (ragged-batch geometry context +
            attention).
        accumulate:
            If ``True``, this call's features are folded into both
            ``OnlineNormalizer`` running statistics (node and target).

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
        :meth:`_features`.

        gamma = 1.0 falls out of the construction here: the caller is
        expected to have already added noise to ``x_last``, so the velocity
        target ``next_positions - x_last`` is measured from the noisy
        position, matching the source MeshGraphNets training recipe without
        a separate noise-correction term.
        """
        one_hot = F.one_hot(particle_types, num_classes=self._node_type_size).to(
            torch.float32
        )
        scripted_velocity = self._train_scripted_velocity(
            x_last, next_positions, particle_types
        )

        node_feats_raw = self._features(
            one_hot, scripted_velocity, x_last, reference_coords
        )
        node_feats = self._node_normalizer(node_feats_raw, accumulate=accumulate)

        # Coords are the RAW mm-frame x_last, same rule as predict_positions.
        pred_norm = self._net(node_feats, x_last, n_particles_per_example)

        target_raw = torch.cat([next_positions - x_last, next_aux[:, None]], dim=1)
        target_norm = self._target_normalizer(target_raw, accumulate=accumulate)

        return pred_norm, target_norm
