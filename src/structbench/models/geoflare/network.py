"""``GaleFlareAttention`` -- the GALE_FA attention core.

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

The block/driver (:class:`GeoFlareBlock`/:class:`GeoFlareNet`) that wraps
this attention module into a full model, including the rationale for why
its per-example ragged-batch loop wraps the WHOLE block stack rather than
each attention call individually, land in a later addition to this module.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

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
    this module is single-example; ragged multi-example batches are the
    caller's concern (a later addition to this module).

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
