"""Multi-scale ball-query context pathway (ADR-0041 step 3; ADR-0045, draft).

The three pieces of NVIDIA PhysicsNeMo's GALE geometry-context pathway
(Apache-2.0 License, Copyright (c) NVIDIA CORPORATION & AFFILIATES),
ported to a single-example (flat ``N``, no batch dim) convention --
callers loop segments the same way ``TransolverNet`` does for
``PhysicsAttentionIrregularMesh`` (``models/transolver/network.py``):

- :class:`ContextTokenizer` -- PhysicsNeMo's ``ContextProjector``: the
  classic Transolver-style slice tokenizer (project, slice-softmax,
  weighted aggregation) reused purely as a TOKENIZER here, with no
  self-attention among the resulting tokens (grounding
  ``scratch/2026-08-09-geoflare-grounding.md`` SS4.6/S10) -- contrast
  ``models/transolver/network.py``'s ``PhysicsAttentionIrregularMesh``,
  which attends among its slice tokens. Deliberately differs from that
  sibling port in two numeric details -- per-family fidelity to each
  family's OWN upstream reference, not a shared convention: temperature
  is CLAMPED to ``[0.5, 5]`` (Transolver's irregular-mesh variant is
  unclamped) and the slice-norm epsilon is ``1e-2`` (Transolver's is
  ``1e-5``).
- :class:`GeometricFeatureProcessor` -- PhysicsNeMo's single-scale
  ``MultiScaleFeatureExtractor`` branch: ball query -> flatten -> 3-linear
  MLP -> ``tanh``, feeding both a per-scale LOCAL feature (concatenated
  onto the input token before the first block, a later task) and, via its
  own :class:`ContextTokenizer`, a per-scale CONTEXT feature.
- :class:`MultiScaleContext` -- the full per-example assembly (PhysicsNeMo's
  ``build_context``): one geometry tokenizer over raw (standardized)
  coordinates, plus one ``(GeometricFeatureProcessor, ContextTokenizer)``
  pair per scale, concatenated into a single context tensor meant to be
  computed once and reused unchanged by every GALE block (a later task).

Part order is CODE-faithful, not paper-faithful: the raw ``build_context``
runs its local-extractor loop (``context_parts.extend(...)``) BEFORE
appending the geometry context (``context_parts.append(...)``), i.e.
``[scale_1, scale_2, geometry]`` -- while the paper's Eq (10) lists geometry
first. This port matches the CODE. The order is functionally absorbed by
the downstream cross-attention's ``Linear`` over the full context width
either way; the paper-vs-code discrepancy is recorded in ADR-0045.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from structbench.models.geoflare.geo_ops import ball_query, standardize_coords


class ContextTokenizer(nn.Module):
    """PhysicsNeMo's ``ContextProjector``, single-example (grounding S10).

    Projects ``N`` points into ``slice_num`` tokens per head via a learned
    soft assignment, with no attention among the resulting tokens -- used
    purely as a tokenizer feeding the GALE cross-attention context, not as
    a self-contained attention block.

    Parameters
    ----------
    dim:
        Input feature width.
    heads:
        Number of heads ``H``.
    dim_head:
        Per-head channel width ``D``.
    slice_num:
        Number of slice tokens ``S`` produced per head.
    dropout:
        Accepted for constructor-signature parity with the upstream
        reference; unused here -- a pure tokenizer has no
        attention/dropout site (grounding SS4.6: "no attention inside").
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        slice_num: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        # NO orthogonal init (contrast PhysicsAttentionIrregularMesh's
        # in_project_slice) -- PhysicsNeMo's ContextProjector does not
        # orthogonal-init this layer (grounding S4.3 c22).
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        # (H, 1, 1): broadcasts against the head-first (H, N, S) logits
        # computed in forward(). Upstream's own layout is batch-first
        # (B, N, H, S) with a (1, 1, H, 1) temperature; the per-head
        # SEMANTICS is what's pinned here, not that literal shape, which
        # belongs to a batch-first layout this port does not use.
        self.temperature = nn.Parameter(torch.full((heads, 1, 1), 0.5))

    def forward(self, x: Tensor) -> Tensor:
        """Tokenize ``N`` points into ``(heads, slice_num, dim_head)`` tokens.

        Parameters
        ----------
        x:
            ``(N, dim)`` point features for one example.

        Returns
        -------
        Tensor
            ``(heads, slice_num, dim_head)`` context tokens.
        """
        n = x.shape[0]
        x_mid = self.in_project_x(x).view(n, self.heads, self.dim_head).permute(1, 0, 2)
        fx_mid = (
            self.in_project_fx(x).view(n, self.heads, self.dim_head).permute(1, 0, 2)
        )
        logits = self.in_project_slice(x_mid)  # (H, N, S)
        temperature = torch.clamp(self.temperature, 0.5, 5.0)  # CLAMPED
        w = torch.softmax(logits / temperature, dim=-1)  # (H, N, S)
        norm = w.sum(dim=1) + 1e-2  # (H, S); eps 1e-2
        # Equivalent to the reference's matmul-with-permutes aggregation
        # (grounding SS8/C8) -- documented as einsum-equivalent, never as a
        # byte-level port of that matmul call.
        token = torch.einsum("hns,hnd->hsd", w, fx_mid) / norm.unsqueeze(-1)
        return token


class GeometricFeatureProcessor(nn.Module):
    """PhysicsNeMo's single-scale ``MultiScaleFeatureExtractor`` branch.

    Ball-queries up to ``k`` neighbours within ``radius`` around every
    point, flattens the ``(N, k, dim)`` result to ``(N, dim*k)``, and passes it
    through a 3-linear MLP with ``tanh`` applied to the FINAL output --
    OUTSIDE the linear/GELU stack, not as the stack's own last layer --
    bounding every output channel to ``(-1, 1)``.

    Parameters
    ----------
    radius:
        Ball-query radius, in per-example STANDARDIZED coordinate units.
    k:
        Neighbour cap of the ball query.
    n_hidden_local:
        Output width, and the MLP's first hidden width; the second hidden
        width is ``n_hidden_local // 2`` (dims ``[dim*k, n_hidden_local,
        n_hidden_local // 2, n_hidden_local]``, GELU between layers, none
        after the last).
    dim:
        Spatial dimensionality of the coordinates (3 for deforming_plate,
        2 for taylor; ADR-0047).
    """

    def __init__(
        self, radius: float, k: int, n_hidden_local: int = 32, dim: int = 3
    ) -> None:
        super().__init__()
        self.radius = radius
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(dim * k, n_hidden_local),
            nn.GELU(),
            nn.Linear(n_hidden_local, n_hidden_local // 2),
            nn.GELU(),
            nn.Linear(n_hidden_local // 2, n_hidden_local),
        )

    def forward(self, g_std: Tensor) -> Tensor:
        """Compute per-point local geometric features for one scale.

        Parameters
        ----------
        g_std:
            ``(N, dim)`` per-example STANDARDIZED coordinates (see
            :func:`structbench.models.geoflare.geo_ops.standardize_coords`).

        Returns
        -------
        Tensor
            ``(N, n_hidden_local)`` features, every element in ``(-1, 1)``.
        """
        n = g_std.shape[0]
        neighbors = ball_query(g_std, self.radius, self.k)  # (N, k, dim)
        flat = neighbors.reshape(n, -1)  # (N, dim*k)
        return torch.tanh(self.mlp(flat))


class MultiScaleContext(nn.Module):
    """Full per-example GALE context assembly (grounding S10).

    Produces a ``[scale_1, scale_2, geometry]`` context tensor (part order
    is CODE-faithful, see module docstring) plus a ``[scale_1_local,
    scale_2_local]`` local-feature tensor, both derived from a single set
    of raw coordinates that this module standardizes internally.

    Parameters
    ----------
    n_hidden:
        Latent channel width of the surrounding GALE blocks; used only to
        derive the shared context ``dim_head`` (``n_hidden // n_heads``),
        NOT as any tokenizer's own input feature width.
    n_heads:
        Number of heads, shared by every tokenizer.
    n_hidden_local:
        Output width of each :class:`GeometricFeatureProcessor`, and the
        input feature width of each per-scale :class:`ContextTokenizer`.
    slice_num:
        Number of slice tokens produced by every tokenizer.
    radii:
        ``(near, far)`` ball-query radii, in per-example STANDARDIZED
        coordinate units.
    neighbors:
        ``(near, far)`` neighbour caps, paired with ``radii`` by position.
    dropout:
        Accepted for constructor-signature parity; unused (see
        :class:`ContextTokenizer`).
    dim:
        Spatial dimensionality of the standardized coordinates the
        geometry tokenizer and the per-scale processors consume (3 for
        deforming_plate, 2 for taylor; ADR-0047).
    """

    def __init__(
        self,
        n_hidden: int,
        n_heads: int,
        n_hidden_local: int,
        slice_num: int,
        radii: tuple[float, float],
        neighbors: tuple[int, int],
        dropout: float = 0.0,
        dim: int = 3,
    ) -> None:
        super().__init__()
        self.dim_head_ctx = n_hidden // n_heads
        self.geometry_tokenizer = ContextTokenizer(
            dim, n_heads, self.dim_head_ctx, slice_num
        )
        self.processors = nn.ModuleList(
            [
                GeometricFeatureProcessor(r, k, n_hidden_local, dim=dim)
                for r, k in zip(radii, neighbors, strict=True)
            ]
        )
        self.scale_tokenizers = nn.ModuleList(
            [
                ContextTokenizer(n_hidden_local, n_heads, self.dim_head_ctx, slice_num)
                for _ in radii
            ]
        )

    @property
    def context_dim(self) -> int:
        """Last-dim width of :meth:`forward`'s context output.

        ``3 * dim_head_ctx``: geometry + 2 scales, no global (``radii``/
        ``neighbors`` are pinned 2-tuples -- see the constructor).
        """
        return 3 * self.dim_head_ctx

    def forward(self, coords_raw: Tensor) -> tuple[Tensor, Tensor]:
        """Build the context and local-feature tensors for one example.

        Parameters
        ----------
        coords_raw:
            ``(N, 3)`` RAW (un-standardized) per-example coordinates.
            Standardization runs INSIDE this call, so callers must NOT
            pre-standardize -- the configured ball-query radii are defined
            against this call's own standardized frame, not whatever units
            ``coords_raw`` happens to be in.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``context``: ``(heads, slice_num, context_dim)``, part order
            ``[scale_1, scale_2, geometry]`` (see module docstring).
            ``local``: ``(N, 2 * n_hidden_local)``, scale order.
        """
        g = standardize_coords(coords_raw)
        local_parts: list[Tensor] = []
        context_parts: list[Tensor] = []
        for processor, tokenizer in zip(
            self.processors, self.scale_tokenizers, strict=True
        ):
            h_s = processor(g)
            local_parts.append(h_s)
            context_parts.append(tokenizer(h_s))
        context_parts.append(self.geometry_tokenizer(g))  # geometry LAST
        context = torch.cat(context_parts, dim=-1)
        local = torch.cat(local_parts, dim=-1)
        return context, local
