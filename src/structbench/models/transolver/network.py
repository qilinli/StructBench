"""Physics-Attention network for the native Transolver family.

ADR-0041 step 2; recipe pins in ADR-0044.

Architecture follows Wu et al., ICML 2024 (arXiv:2402.02366); reference
implementation github.com/thuml/Transolver, MIT License, Copyright (c) 2024
THUML @ Tsinghua University. Ported to pure ``torch`` (no ``einops``, no
``timm``): the reference's sole ``einops.rearrange`` call (a head-merge,
``'b h n d -> b n (h d)'``) is subsumed here by choosing the output
subscript order of the head-attention ``torch.einsum`` call directly
(``"hmd,nhm->nhd"``, i.e. ``(H, M, dim_head), (P, H, M) -> (P, H, dim_head)``)
followed by a plain ``reshape``; its ``timm.trunc_normal_`` is
``torch.nn.init.trunc_normal_``, native since torch 1.10. Math follows
Eqs (1)-(4) (Physics-Attention: slice, token aggregation, token attention,
deslice) and the pre-LN block of Eq (6); every code-only detail (learnable
per-head slice temperature, orthogonal-then-overwritten init, ``mlp_ratio=1``
no-expansion FFN, the unconditional placeholder vector, and the
``fx_mid``-vs-``x_mid`` aggregation subtlety) is faithful to the released
irregular-mesh variant, not just the paper. See
``scratch/2026-08-08-transolver-grounding.md`` SS2-SS3 for the verified
citation trail.

The one piece of novel engineering here (not present upstream, where the
irregular-mesh variant only ever ran at batch=1) is ragged-``N`` batching:
StructBench batches variable-size meshes as one flat ``(sum_P, C)`` tensor
plus ``n_particles_per_example``. ``PhysicsAttentionIrregularMesh`` computes
the row-wise slice weights once over the whole flat tensor (softmax over the
slice axis is per-point and does not mix rows), then loops over each
example's contiguous particle segment for the parts that DO sum across
points — token aggregation, token attention, and deslice — so the batched
result is exactly the per-example result (mathematically, not just
approximately; see ``tests/models/transolver/test_network.py::
test_batched_forward_matches_per_example``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def build_mlp_2layer(in_size: int, hidden: int, out_size: int) -> nn.Sequential:
    """Build the thuml ``MLP(..., n_layers=0)`` shape: a plain 2-layer MLP.

    Parameters
    ----------
    in_size:
        Input feature width.
    hidden:
        Width of the single hidden layer.
    out_size:
        Output feature width.

    Returns
    -------
    nn.Sequential
        ``Linear(in_size, hidden) -> GELU -> Linear(hidden, out_size)``.
    """
    return nn.Sequential(
        nn.Linear(in_size, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_size),
    )


def _segments(total: int, n_per: Tensor | None) -> list[tuple[int, int]]:
    """Split a flat ``(total, ...)`` tensor into per-example ``[start, end)`` ranges.

    Parameters
    ----------
    total:
        Total number of rows in the flat tensor (``sum_P``).
    n_per:
        ``(B,)`` particle counts per example, in concatenation order; ``None``
        for the single-example case (one segment covering the whole tensor).

    Returns
    -------
    list[tuple[int, int]]
        Contiguous ``(start, end)`` index pairs, one per example.
    """
    if n_per is None:
        return [(0, total)]
    ends = torch.cumsum(n_per, dim=0).tolist()
    starts = [0, *ends[:-1]]
    return list(zip(starts, ends, strict=True))


class PhysicsAttentionIrregularMesh(nn.Module):
    """Physics-Attention, irregular-mesh (point-cloud) variant.

    Implements Eqs (1)-(4) of Wu et al. 2024: project each point to a set of
    ``slice_num`` soft-assignment weights (Eq 1); aggregate the projected
    features into ``slice_num`` tokens weighted by those assignments (Eq 2);
    run standard scaled dot-product self-attention among the tokens only
    (Eq 3); broadcast the attended tokens back to every point using the SAME
    slice weights, unnormalized (Eq 4). Complexity is linear in the number of
    points ``P`` since ``slice_num`` is a small constant.

    Faithful to the released code (not just the paper): the slice logits are
    divided by a learnable per-head ``temperature`` (unclamped, matching the
    irregular-mesh variant — the structured-mesh variants clamp it);
    ``in_project_slice`` is initialized with ``orthogonal_`` here but that
    init is overwritten later by ``TransolverNet``'s global
    ``trunc_normal_(std=0.02)`` pass, faithfully reproducing a thuml
    ordering quirk (see ``TransolverNet._initialize_weights``); token
    aggregation (Eq 2) pools ``in_project_fx`` features (``fx_mid``), NOT the
    ``in_project_x`` features used to derive the slice weights (``x_mid``) —
    a subtlety present in the released code but easy to miss from the paper
    alone.

    Parameters
    ----------
    dim:
        Input/output channel width ``C``.
    heads:
        Number of attention heads ``H``.
    dim_head:
        Per-head channel width. ``heads * dim_head`` is the inner width used
        for the token projections and multi-head concatenation.
    slice_num:
        Number of slice tokens ``M`` each head projects the mesh onto.
    dropout:
        Dropout probability applied to the token-attention weights and to
        the output projection.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        slice_num: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        # "Principled initialization" per the paper; overwritten by
        # TransolverNet's later global trunc_normal_ pass — kept anyway to
        # faithfully reproduce the released ordering (grounding SS3.3).
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        # Per-head, unclamped (Irregular_Mesh variant); broadcasts against
        # the (P, H, M) slice logits.
        self.temperature = nn.Parameter(torch.full((heads, 1), 0.5))
        self.scale = dim_head**-0.5
        self.attn_dropout = nn.Dropout(dropout)

    def _slice_weights(self, x: Tensor) -> Tensor:
        """Compute the Eq (1) slice-assignment weights, row-wise.

        Point-wise (each row of ``x`` is independent of every other row), so
        this is safe to call on a flat multi-example ``(P, dim)`` tensor
        without a segment loop.

        Parameters
        ----------
        x:
            ``(P, dim)`` point features.

        Returns
        -------
        Tensor
            ``(P, heads, slice_num)`` slice weights; softmax-normalized over
            the last (slice) axis per point per head.
        """
        x_mid = self.in_project_x(x).reshape(-1, self.heads, self.dim_head)
        logits = self.in_project_slice(x_mid) / self.temperature
        return torch.softmax(logits, dim=-1)

    def forward(self, x: Tensor, n_particles_per_example: Tensor | None) -> Tensor:
        """Run Physics-Attention over a (possibly multi-example) flat point set.

        Parameters
        ----------
        x:
            ``(P, dim)`` point features, examples concatenated along dim 0.
        n_particles_per_example:
            ``(B,)`` particle counts per example, in concatenation order, or
            ``None`` to treat ``x`` as a single example.

        Returns
        -------
        Tensor
            ``(P, dim)`` updated point features.
        """
        fx_mid = self.in_project_fx(x).reshape(-1, self.heads, self.dim_head)
        w = self._slice_weights(x)  # (P, H, M)
        outs = []
        for start, end in _segments(x.shape[0], n_particles_per_example):
            w_e, fx_e = w[start:end], fx_mid[start:end]
            norm = w_e.sum(dim=0)  # (H, M)
            token = torch.einsum("nhd,nhm->hmd", fx_e, w_e)
            token = token / (norm + 1e-5).unsqueeze(-1)  # Eq (2), eps per thuml
            q, k, v = self.to_q(token), self.to_k(token), self.to_v(token)
            dots = q @ k.transpose(-1, -2) * self.scale  # (H, M, M), Eq (3)
            token_out = self.attn_dropout(torch.softmax(dots, dim=-1)) @ v
            out_e = torch.einsum("hmd,nhm->nhd", token_out, w_e)  # Eq (4)
            outs.append(out_e.reshape(end - start, self.heads * self.dim_head))
        return self.to_out(torch.cat(outs, dim=0))


class TransolverBlock(nn.Module):
    """One pre-LN Transolver block (Eq 6): Physics-Attention + FFN, both residual.

    The final block additionally carries the decoder head (``ln_3`` +
    ``mlp2``), applied WITHOUT a residual connection — faithful to the
    released code, which folds the output projection into the last block
    rather than using a separate decoder module.

    Parameters
    ----------
    hidden_dim:
        Channel width ``C``.
    heads:
        Number of Physics-Attention heads ``H``.
    slice_num:
        Number of Physics-Attention slice tokens ``M``.
    mlp_ratio:
        FFN hidden-width multiplier (``mlp_ratio=1`` means no expansion:
        hidden -> hidden -> hidden).
    dropout:
        Dropout probability, forwarded to the attention sub-module.
    last_layer:
        If ``True``, this block also applies the decoder head after the FFN
        residual (no residual on the head itself).
    out_size:
        Output feature width of the decoder head (only used when
        ``last_layer``).
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        slice_num: int,
        mlp_ratio: int,
        dropout: float,
        last_layer: bool,
        out_size: int,
    ) -> None:
        super().__init__()
        self.last_layer = last_layer
        dim_head = hidden_dim // heads
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttentionIrregularMesh(
            hidden_dim,
            heads=heads,
            dim_head=dim_head,
            slice_num=slice_num,
            dropout=dropout,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = build_mlp_2layer(hidden_dim, hidden_dim * mlp_ratio, hidden_dim)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_size)

    def forward(self, fx: Tensor, n_particles_per_example: Tensor | None) -> Tensor:
        """Apply the block.

        Parameters
        ----------
        fx:
            ``(P, hidden_dim)`` node latents.
        n_particles_per_example:
            ``(B,)`` particle counts per example, or ``None`` for a single
            example; forwarded unchanged to ``attn``.

        Returns
        -------
        Tensor
            ``(P, hidden_dim)`` updated latents, or ``(P, out_size)`` if this
            is the last block (decoder head applied, no residual).
        """
        fx = self.attn(self.ln_1(fx), n_particles_per_example) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


def _init_weights(module: nn.Module) -> None:
    """thuml's ``initialize_weights()`` leaf-module init, applied via ``.apply()``.

    ``Linear`` weights get ``trunc_normal_(std=0.02)`` and zero bias;
    ``LayerNorm`` is reset to the identity affine (weight 1, bias 0). Run
    LAST (after all submodules, including ``in_project_slice``'s
    ``orthogonal_`` init, already exist) so it overwrites that orthogonal
    init — a released-code ordering quirk, reproduced faithfully rather than
    "corrected" (grounding SS3.3, ADR-0044).

    Parameters
    ----------
    module:
        A single leaf (or container) module, as passed by ``nn.Module.apply``.
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class TransolverNet(nn.Module):
    """Full Transolver model: preprocess MLP + placeholder, then ``n_layers`` blocks.

    Parameters
    ----------
    node_in:
        Input node feature width.
    out_size:
        Output feature width (produced only by the last block's decoder
        head).
    hidden_dim:
        Channel width ``C`` (default: the paper's irregular-mesh Elasticity
        configuration, Table 8).
    n_layers:
        Number of Transolver blocks ``L``.
    n_heads:
        Number of Physics-Attention heads ``H`` per block.
    slice_num:
        Number of Physics-Attention slice tokens ``M``.
    mlp_ratio:
        FFN hidden-width multiplier inside each block.
    dropout:
        Dropout probability, forwarded to every block.
    """

    def __init__(
        self,
        node_in: int,
        out_size: int,
        hidden_dim: int = 128,
        n_layers: int = 8,
        n_heads: int = 8,
        slice_num: int = 64,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.preprocess = build_mlp_2layer(node_in, hidden_dim * 2, hidden_dim)
        # thuml Irregular_Mesh: added UNCONDITIONALLY (unlike the Structured
        # variants, where it is nested inside an `if fx is None` branch and
        # so is dead code on their fx-conditioned tasks). Grounding SS3.3.
        self.placeholder = nn.Parameter((1 / hidden_dim) * torch.rand(hidden_dim))
        self.blocks = nn.ModuleList(
            [
                TransolverBlock(
                    hidden_dim,
                    n_heads,
                    slice_num,
                    mlp_ratio,
                    dropout,
                    last_layer=(i == n_layers - 1),
                    out_size=out_size,
                )
                for i in range(n_layers)
            ]
        )
        self._initialize_weights()

    def forward(
        self, node_feats: Tensor, n_particles_per_example: Tensor | None
    ) -> Tensor:
        """Run the full forward pass.

        Parameters
        ----------
        node_feats:
            ``(P, node_in)`` node input features, examples concatenated
            along dim 0.
        n_particles_per_example:
            ``(B,)`` particle counts per example, in concatenation order, or
            ``None`` for a single example.

        Returns
        -------
        Tensor
            ``(P, out_size)`` decoded per-node output.
        """
        fx = self.preprocess(node_feats) + self.placeholder
        for block in self.blocks:
            fx = block(fx, n_particles_per_example)
        return fx

    def _initialize_weights(self) -> None:
        """Apply thuml's global ``Linear``/``LayerNorm`` re-init, last."""
        self.apply(_init_weights)
