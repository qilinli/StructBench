"""TransolverNet / Physics-Attention (Wu et al. 2024; ADR-0044).

Reference math: Eqs (1)-(4) and pre-LN block Eq (6) of arXiv:2402.02366;
implementation details follow thuml/Transolver's irregular-mesh variant.
"""

import torch

from structbench.models.transolver.network import (
    PhysicsAttentionIrregularMesh,
    TransolverNet,
)


def test_forward_shape_single_example() -> None:
    net = TransolverNet(
        node_in=7, out_size=4, hidden_dim=16, n_layers=2, n_heads=2, slice_num=4
    )
    out = net(torch.randn(11, 7), None)
    assert out.shape == (11, 4)


def test_batched_forward_matches_per_example() -> None:
    # THE ragged-batching correctness test: segment computation must equal
    # running each example alone (thuml batch=1 semantics). eval() + dropout=0
    # make both paths deterministic; atol loosened above the float32 GEMM
    # tiling noise floor (different matrix sizes accumulate differently).
    torch.manual_seed(0)
    net = TransolverNet(
        node_in=7, out_size=4, hidden_dim=16, n_layers=2, n_heads=2, slice_num=4
    )
    net.eval()
    a, b = torch.randn(11, 7), torch.randn(5, 7)
    with torch.no_grad():
        batched = net(torch.cat([a, b]), torch.tensor([11, 5]))
        singles = torch.cat([net(a, None), net(b, None)])
    assert torch.allclose(batched, singles, atol=1e-5)


def test_slice_weights_softmax_over_slices() -> None:
    torch.manual_seed(0)
    attn = PhysicsAttentionIrregularMesh(dim=16, heads=2, dim_head=8, slice_num=4)
    w = attn._slice_weights(torch.randn(11, 16))  # (P, H, M)
    assert w.shape == (11, 2, 4)
    assert torch.allclose(w.sum(dim=-1), torch.ones(11, 2), atol=1e-6)
    # Non-degeneracy at init: weights must VARY across nodes (a constant
    # projection would silently give average pooling — the Transolver++
    # collapse risk, ADR-0044 ledgered; deeper non-collapse is NOT tested).
    assert w.std(dim=0).max() > 1e-4


def test_temperature_learnable_init() -> None:
    attn = PhysicsAttentionIrregularMesh(dim=16, heads=2, dim_head=8, slice_num=4)
    assert attn.temperature.requires_grad
    assert attn.temperature.shape == (2, 1)  # per-head, broadcasts vs (P, H, M)
    assert torch.allclose(attn.temperature, torch.full_like(attn.temperature, 0.5))


def test_reference_parameter_count() -> None:
    # Pins the reference architecture (L=8, C=128, H=8, M=64, ratio=1,
    # node_in=18, out=4) against the structural formula from grounding §2-§3.
    net = TransolverNet(node_in=18, out_size=4)
    n = sum(p.numel() for p in net.parameters())
    c, h, dh, m, node_in, out = 128, 8, 16, 64, 18, 4
    preprocess = (node_in + 1) * 2 * c + (2 * c + 1) * c
    attn = 2 * ((c + 1) * c) + (dh + 1) * m + 3 * dh * dh + (c + 1) * c + h
    block = 2 * 2 * c + attn + 2 * ((c + 1) * c)  # ln_1+ln_2, attn, mlp(ratio=1)
    last_extra = 2 * c + (c + 1) * out  # ln_3 + mlp2
    assert n == preprocess + c + 8 * block + last_extra  # +c = placeholder


def test_trunc_normal_init_applied() -> None:
    # Faithful to released code: the global trunc_normal_(std=0.02) + zero-bias
    # pass runs LAST, overwriting the orthogonal in_project_slice init (thuml
    # initialize_weights() ordering quirk, ADR-0044). Assertions must be
    # DISCRIMINATING — PyTorch's default Linear init has nonzero uniform bias
    # and weight std ~0.14 at these widths, so each check below fails if the
    # init pass is omitted.
    net = TransolverNet(node_in=18, out_size=4)
    lin = net.preprocess[0]
    assert isinstance(lin, torch.nn.Linear)
    assert torch.all(lin.bias == 0.0)
    assert abs(float(lin.weight.std()) - 0.02) < 0.006
    # The overwrite happened: slice projection is trunc_normal, NOT orthonormal
    # (orthogonal_ on the (slice_num=64, dim_head=16) weight gives Wᵀ W = I₁₆).
    w = net.blocks[0].attn.in_project_slice.weight
    assert abs(float(w.std()) - 0.02) < 0.006
    assert not torch.allclose(w.T @ w, torch.eye(w.shape[1]), atol=0.1)
    for mod in net.modules():  # secondary: LayerNorm resets
        if isinstance(mod, torch.nn.LayerNorm):
            assert torch.all(mod.weight == 1.0) and torch.all(mod.bias == 0.0)
