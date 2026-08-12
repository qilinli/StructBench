"""``GaleFlareAttention`` / ``GeoFlareBlock`` / ``GeoFlareNet`` -- the GALE_FA core.

ADR-0041 step 3; ADR-0045 (draft). The self-attention half of NVIDIA
PhysicsNeMo's ``GALE_FA`` (Apache-2.0 License, Copyright (c) NVIDIA
CORPORATION & AFFILIATES) is FLARE (*Fast Low-rank Attention Routing
Engine*, arXiv:2508.12594): a small set of learnable global queries
``q_global`` attend over every point's keys/values to encode ``M`` global
tokens, then every point's own key vector is REUSED as a query to decode
those ``M`` tokens back out per point -- two standard attention calls,
``O(NM)`` instead of ``O(N^2)``. The cross-attention half is GALE's
geometry context pathway (:class:`structbench.models.geoflare.context
.MultiScaleContext`, Task 3), retained unchanged, mixed with the
FLARE-encode/decode output via a single learnable scalar gate. Both halves
and the mix are pinned to the upstream ``GALE_FA`` module (grounding
``scratch/2026-08-09-geoflare-grounding.md`` SS3-SS4, SS10) -- see
:class:`GaleFlareAttention` for the exact math.

**Attention is MANUAL, not ``F.scaled_dot_product_attention``.** The house
style elsewhere in this family (``models/transolver/network.py``) already
hand-writes attention as ``softmax(q @ k.transpose(-1, -2) * scale) @ v``;
this module follows the same style, but here it is also load-bearing:
PyTorch's ``scaled_dot_product_attention`` only gained its ``scale=``
keyword in torch 2.1, above this repo's ``torch>=2.0`` floor (``pyproject
.toml``), and the upstream reference pins a NON-default scale of ``1.0``
(see :class:`GaleFlareAttention`) -- SDPA's own default scale
(``dim_head**-0.5``) would silently produce the wrong math on a torch 2.0
install with no error. Manual softmax has no such version gate.

**The per-segment loop wraps the WHOLE block stack, not each attention
call.** Contrast ``models/transolver/network.py``'s ``TransolverNet``,
which computes Physics-Attention's point-wise slice weights ONCE over the
full ragged batch and only loops per segment inside the parts that sum
across points (token aggregation/attention/deslice) -- because Physics-
Attention's per-point work (``in_project_x``/``in_project_slice``) is cheap
and shared. GeoFLARE cannot reuse that shape: the GALE geometry context
(:class:`structbench.models.geoflare.context.MultiScaleContext`) is a
PER-EXAMPLE ball-query construction over one example's own coordinate
cloud -- it has no meaning computed jointly across multiple examples'
concatenated points (a ball query would find "neighbours" across unrelated
meshes), and it is expensive relative to a GALE_FA block (dominated by
``torch.cdist``/``topk`` ball queries, not by the ``O(NM)`` attention).
Since the context must already be rebuilt per example, and it is reused
UNCHANGED by every block in the stack (grounding SS4.5, "same context
every layer"), the simplest correct shape is to loop segments ONCE around
the entire ``preprocess -> blocks -> decoder`` pipeline, rather than
threading segments through every individual attention call the way
``TransolverNet`` does. At ``deforming_plate`` scale (``N`` ~1.3k nodes per
example), the per-block attention cost this forgoes reusing is small next
to the context-construction cost it cannot avoid paying per example
regardless -- simplicity over reusing the inner-loop pattern for its own
sake (see :class:`GeoFlareNet`).

Reuses ``build_mlp_2layer``/``_segments`` from
``structbench.models.transolver.network`` (established cross-family-
utility precedent, per this task's brief) rather than redefining them --
``_segments`` is private-by-convention to its own module (``PRINCIPLES.md``:
the ``_``-prefix boundary), and this cross-module import is a deliberate,
brief-directed exception to that convention, not an oversight; ADR-0045
(Task 8) is the place of record for it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from structbench.models.geoflare.context import MultiScaleContext
from structbench.models.transolver.network import _segments, build_mlp_2layer

# Upstream ``GALE_FA``/``FLARE`` comment (grounding SS4.1 c4, quoted
# verbatim in spirit): "recommended by the FLARE authors to use scale=1 if
# dim_head<=8 else dim_head**-0.5 but we use 1.0 because the recommended
# scaling is not tested yet." StructBench follows the reference's actual
# behaviour (the flat 1.0), not its own commented-out recommendation --
# faithfully reproducing a known, flagged upstream deviation from the
# FLARE paper's own guidance, applied at all three attention sites (FLARE
# encode, FLARE decode, GALE cross-attention).
_SCALE = 1.0


class GaleFlareAttention(nn.Module):
    """GALE_FA, single-example (grounding SS4.1-SS4.2, SS10).

    Three attention passes per call, all sharing the same ``(heads,
    dim_head)`` projection width and the same manual
    ``softmax(q @ k.T * scale) @ v`` primitive (:meth:`_attend`):

    1. **FLARE encode** -- the ``slice_num`` learnable global queries
       (``q_global``) attend over every point's projected key/value
       (``self_k``/``self_v`` of ``x``); softmax over the ``N`` (point)
       axis; produces ``M`` global tokens ``z``.
    2. **FLARE decode** -- every point's own key vector is REUSED AS A
       QUERY against ``q_global`` (now playing the KEY role) to read back
       out of ``z`` (the VALUE); softmax over the ``M`` (global-token)
       axis; produces one output per point, ``y_self``.
    3. **GALE cross-attention** -- a fresh query projection of every point
       (``cross_q``) attends over the externally supplied geometry
       ``context`` (``cross_k``/``cross_v``); softmax over the context's
       own token axis; produces ``y_cross``.

    The two outputs are combined with a single learnable scalar gate
    (``state_mixing``, ``sigmoid(0) = 0.5`` at init, i.e. balanced 50/50):
    ``y = w * y_self + (1 - w) * y_cross``. No batch dimension anywhere --
    like :class:`structbench.models.geoflare.context.MultiScaleContext`,
    this module is single-example; ragged multi-example batches are handled
    by the caller looping segments (:class:`GeoFlareNet`).

    Parameters
    ----------
    dim:
        Input/output channel width (the block's hidden width).
    heads:
        Number of attention heads ``H``.
    dim_head:
        Per-head channel width ``D``.
    context_dim:
        Last-dim width of the ``context`` tensor consumed by cross-
        attention (:attr:`structbench.models.geoflare.context
        .MultiScaleContext.context_dim`).
    slice_num:
        Number of FLARE global queries ``M`` (dual-purpose with the
        context pathway's own slice count upstream, but this module takes
        it as a plain constructor argument, not shared state).
    dropout:
        Dropout probability applied to the final output projection only
        -- FLARE encode/decode and GALE cross-attention have no internal
        dropout site upstream (contrast
        ``PhysicsAttentionIrregularMesh.attn_dropout``, which GALE_FA has
        no equivalent of).
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        context_dim: int,
        slice_num: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        # Plain torch.randn, std ~1.0 -- NOT trunc_normal_/orthogonal_, and
        # never re-initialized by any global pass (grounding SS4.3 c22: no
        # explicit weight init upstream). GeoFlareNet must not add one.
        self.q_global = nn.Parameter(torch.randn(heads, slice_num, dim_head))
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.self_k = nn.Linear(dim_head, dim_head)
        self.self_v = nn.Linear(dim_head, dim_head)
        self.cross_q = nn.Linear(dim_head, dim_head)
        self.cross_k = nn.Linear(context_dim, dim_head)
        self.cross_v = nn.Linear(context_dim, dim_head)
        self.state_mixing = nn.Parameter(torch.tensor(0.0))
        self.out_linear = nn.Linear(inner_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    @staticmethod
    def _attend(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Manual scaled-dot-product attention, scale pinned to ``_SCALE``.

        Deliberately NOT ``torch.nn.functional.scaled_dot_product_attention``
        -- see the module docstring: SDPA's ``scale=`` keyword requires
        torch>=2.1, above this repo's torch>=2.0 floor, and the reference
        pins a non-default scale (see ``_SCALE``) that SDPA's own default
        would silently get wrong.

        Parameters
        ----------
        q:
            ``(..., n_q, d)`` queries.
        k:
            ``(..., n_kv, d)`` keys.
        v:
            ``(..., n_kv, d_v)`` values.

        Returns
        -------
        Tensor
            ``(..., n_q, d_v)``.
        """
        weights = torch.softmax(q @ k.transpose(-1, -2) * _SCALE, dim=-1)
        return weights @ v

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """Run FLARE encode/decode self-attention plus GALE cross-attention.

        Parameters
        ----------
        x:
            ``(N, dim)`` point features for one example.
        context:
            ``(heads, S_ctx, context_dim)`` geometry context tokens for the
            SAME example (:meth:`structbench.models.geoflare.context
            .MultiScaleContext.forward`).

        Returns
        -------
        Tensor
            ``(N, dim)`` updated point features.
        """
        n = x.shape[0]
        x_mid = self.in_project_x(x).view(n, self.heads, self.dim_head)
        x_mid = x_mid.permute(1, 0, 2)  # (H, N, D)

        k = self.self_k(x_mid)
        v = self.self_v(x_mid)
        z = self._attend(self.q_global, k, v)  # FLARE encode: (H, M, D)
        y_self = self._attend(k, self.q_global, z)  # FLARE decode: (H, N, D)

        q_c = self.cross_q(x_mid)
        k_c = self.cross_k(context)
        v_c = self.cross_v(context)
        y_cross = self._attend(q_c, k_c, v_c)  # (H, N, D)

        w = torch.sigmoid(self.state_mixing)
        y = w * y_self + (1.0 - w) * y_cross
        out = y.permute(1, 0, 2).reshape(n, self.heads * self.dim_head)
        return self.out_dropout(self.out_linear(out))


class GeoFlareBlock(nn.Module):
    """One pre-LN GALE_FA block: attention + FFN, both residual (grounding SS4.4).

    ``fx = attn(ln_1(fx), context) + fx``; ``fx = ffn(fx) + fx``, where
    ``ffn`` is its OWN ``LayerNorm -> Linear -> GELU -> Linear`` stack
    (contrast ``models/transolver/network.py``'s ``TransolverBlock``,
    which keeps a separate ``ln_2`` outside its ``mlp``). No last-layer
    special head -- unlike ``TransolverBlock``, which folds the decoder
    into its final block, every ``GeoFlareBlock`` is architecturally
    identical; the decoder is a wholly external module
    (:class:`GeoFlareNet`).

    Parameters
    ----------
    hidden_dim:
        Channel width (the block's effective hidden width, i.e.
        ``n_hidden + n_hidden_local * len(radii)`` at the driver level --
        this module itself does not compute that sum).
    heads:
        Number of attention heads ``H``.
    dim_head:
        Per-head channel width for THIS block's attention -- computed by
        the caller as ``hidden_dim // heads``, independent of the context
        pathway's own ``dim_head_ctx`` (grounding SS10: "TWO different
        dim_heads").
    context_dim:
        Last-dim width of the geometry context tensor consumed by
        cross-attention.
    slice_num:
        Number of FLARE global queries, forwarded to
        :class:`GaleFlareAttention`.
    mlp_ratio:
        FFN hidden-width multiplier (``hidden_dim -> hidden_dim *
        mlp_ratio -> hidden_dim``).
    dropout:
        Dropout probability, forwarded to the attention sub-module.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        dim_head: int,
        context_dim: int,
        slice_num: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = GaleFlareAttention(
            hidden_dim, heads, dim_head, context_dim, slice_num, dropout
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, fx: Tensor, context: Tensor) -> Tensor:
        """Apply the block.

        Parameters
        ----------
        fx:
            ``(N, hidden_dim)`` node latents for one example.
        context:
            ``(heads, S_ctx, context_dim)`` geometry context for the SAME
            example, unchanged across every block in the stack (see
            :class:`GeoFlareNet`).

        Returns
        -------
        Tensor
            ``(N, hidden_dim)`` updated latents.
        """
        fx = self.attn(self.ln_1(fx), context) + fx
        fx = self.ffn(fx) + fx
        return fx


class GeoFlareNet(nn.Module):
    """Full GeoFLARE model: preprocess + geometry context + block stack + decoder.

    Segments the ragged multi-example input ONCE (:func:`structbench.models
    .transolver.network._segments`) and, per example, rebuilds the geometry
    context (:class:`structbench.models.geoflare.context.MultiScaleContext`)
    and runs the ENTIRE ``preprocess -> blocks -> decoder`` pipeline before
    moving to the next example -- see the module docstring for why the loop
    wraps the whole stack rather than each attention call individually.

    Parameters
    ----------
    node_in:
        Input node feature width.
    out_size:
        Output feature width.
    n_hidden:
        Latent channel width fed to :attr:`preprocess` and to the geometry
        context pathway's ``dim_head_ctx`` derivation (``n_hidden //
        n_heads``) -- NOT the block stack's own hidden width, which is
        wider (see ``effective`` below).
    n_layers:
        Number of stacked :class:`GeoFlareBlock`.
    n_heads:
        Number of attention heads per block, shared by the context
        pathway and every block.
    slice_num:
        Number of FLARE global queries / context slice tokens (dual
        purpose, per the upstream reference -- grounding SS4.3 c17,
        c19).
    mlp_ratio:
        FFN hidden-width multiplier inside each block.
    dropout:
        Dropout probability, forwarded to the context pathway and every
        block.
    n_hidden_local:
        Per-scale local-feature width produced by the ball-query geometric
        feature processor (:class:`structbench.models.geoflare.context
        .MultiScaleContext`); concatenated onto :attr:`preprocess`'s
        output BEFORE the first block, widening the block stack's own
        hidden width to ``effective = n_hidden + n_hidden_local * 2``.
    radii:
        ``(near, far)`` ball-query radii, in per-example standardized
        coordinate units (:func:`structbench.models.geoflare.geo_ops
        .standardize_coords`).
    neighbors:
        ``(near, far)`` neighbour caps, paired with ``radii`` by position.
    dim:
        Spatial dimensionality of the geometry coordinates (3 for
        deforming_plate, 2 for taylor; ADR-0047).
    """

    def __init__(
        self,
        node_in: int,
        out_size: int,
        n_hidden: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        slice_num: int = 128,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        n_hidden_local: int = 32,
        radii: tuple[float, float] = (0.05, 0.25),
        neighbors: tuple[int, int] = (8, 32),
        dim: int = 3,
    ) -> None:
        super().__init__()
        self.preprocess = build_mlp_2layer(node_in, n_hidden * 2, n_hidden)
        self.context_builder = MultiScaleContext(
            n_hidden,
            n_heads,
            n_hidden_local,
            slice_num,
            radii,
            neighbors,
            dropout,
            dim=dim,
        )
        # Effective block-stack width: n_hidden (preprocess output) plus
        # one n_hidden_local-wide local feature per scale (2 fixed scales).
        effective = n_hidden + n_hidden_local * 2
        # The BLOCK's own dim_head -- deliberately NOT
        # self.context_builder.dim_head_ctx (n_hidden // n_heads); the two
        # are numerically different at every non-degenerate config
        # (grounding SS10: 40 vs 32 at defaults).
        dim_head_block = effective // n_heads
        self.blocks = nn.ModuleList(
            [
                GeoFlareBlock(
                    effective,
                    n_heads,
                    dim_head_block,
                    self.context_builder.context_dim,
                    slice_num,
                    mlp_ratio,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(effective), nn.Linear(effective, out_size)
        )
        # NO global weight-init pass here (contrast TransolverNet's
        # ._initialize_weights()/.apply()): the upstream GALE_FA reference
        # uses plain PyTorch defaults everywhere, including q_global's
        # torch.randn (std ~1.0) -- an .apply(trunc_normal_(std=0.02)) pass
        # would rescale q_global by ~50x, a large unintended change to the
        # learned global queries (grounding SS9 gap 5, ADR-0045).

    def forward(
        self,
        node_feats: Tensor,
        coords: Tensor,
        n_particles_per_example: Tensor | None,
    ) -> Tensor:
        """Run the full forward pass.

        Parameters
        ----------
        node_feats:
            ``(P, node_in)`` node input features, examples concatenated
            along dim 0.
        coords:
            ``(P, 3)`` RAW per-node coordinates, same concatenation order
            as ``node_feats``. Standardized internally, per example, by
            :class:`structbench.models.geoflare.context.MultiScaleContext`
            -- callers must NOT pre-standardize.
        n_particles_per_example:
            ``(B,)`` particle counts per example, in concatenation order,
            or ``None`` for a single example.

        Returns
        -------
        Tensor
            ``(P, out_size)`` decoded per-node output.
        """
        segments = _segments(node_feats.shape[0], n_particles_per_example)
        outs: list[Tensor] = []
        for start, end in segments:
            context, local = self.context_builder(coords[start:end])
            fx = torch.cat([self.preprocess(node_feats[start:end]), local], dim=-1)
            for block in self.blocks:
                fx = block(fx, context)
            outs.append(self.decoder(fx))
        return torch.cat(outs, dim=0)
