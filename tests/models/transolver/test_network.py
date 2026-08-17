"""TransolverNet / Physics-Attention (Wu et al. 2024; ADR-0044).

Reference math: Eqs (1)-(4) and pre-LN block Eq (6) of arXiv:2402.02366;
implementation details follow thuml/Transolver's irregular-mesh variant.
"""

import pytest
import torch

from structbench.models.transolver.network import (
    PhysicsAttentionIrregularMesh,
    TransolverNet,
    timestep_embedding,
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


def test_physics_attention_slice1_pins_fx_mid_aggregation() -> None:
    """Eq (2) token aggregation must pool ``fx_mid``, NOT ``x_mid``.

    With ``slice_num=1`` the Eq (1) softmax is exactly 1.0 for every node
    regardless of the slice logits (softmax of a single element always
    normalizes to itself), so ``in_project_x`` cannot affect the result
    through the slice weights at all. The only way ``in_project_x``'s
    projection (``x_mid``) could still leak into the output is if Eq (2)
    mistakenly aggregated ``x_mid`` instead of ``fx_mid`` -- exactly the
    subtlety documented on ``PhysicsAttentionIrregularMesh``.

    ``in_project_fx`` is set to the identity map and ``in_project_x`` to a
    DIFFERENT, distinguishable affine map; ``to_q``/``to_k``/``to_v``/
    ``to_out`` are set to (near-)identity with zero bias so the whole
    forward collapses to the hand-derivable Eq (2) token itself:
    ``token = sum_n(fx_mid[n]) / (N + 1e-5)``, broadcast to every node
    (deslice with an all-ones weight, Eq (4), is a no-op broadcast here).
    We assert the forward output equals this hand computation exactly
    (atol 1e-6), and -- as evidence the fixture is sensitive to the
    fx-vs-x mutation -- that the value that would result from aggregating
    ``x_mid`` instead is a materially different number.
    """
    torch.manual_seed(0)
    attn = PhysicsAttentionIrregularMesh(
        dim=2, heads=1, dim_head=2, slice_num=1, dropout=0.0
    )
    attn.eval()
    with torch.no_grad():
        attn.in_project_fx.weight.copy_(torch.eye(2))
        attn.in_project_fx.bias.zero_()
        attn.in_project_x.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
        attn.in_project_x.bias.copy_(torch.tensor([10.0, 20.0]))
        attn.to_q.weight.copy_(torch.eye(2))
        attn.to_k.weight.copy_(torch.eye(2))
        attn.to_v.weight.copy_(torch.eye(2))
        attn.to_out[0].weight.copy_(torch.eye(2))
        attn.to_out[0].bias.zero_()

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    n = x.shape[0]
    fx_mid = x  # in_project_fx is identity
    x_mid = 2 * x + torch.tensor([10.0, 20.0])  # in_project_x, for the mutation check
    expected_token = fx_mid.sum(dim=0) / (n + 1e-5)
    mutated_token = x_mid.sum(dim=0) / (n + 1e-5)
    assert not torch.allclose(expected_token, mutated_token), (
        "fixture not sensitive to fx-vs-x mutation"
    )

    out = attn(x, [(0, n)])
    assert torch.allclose(out, expected_token.expand(n, -1), atol=1e-6)
    assert not torch.allclose(out, mutated_token.expand(n, -1), atol=1e-6)


def test_physics_attention_slice2_pins_raw_deslice_weights() -> None:
    """Eq (4) deslice must reuse the RAW Eq (1) weights, not renormalize them.

    Per-node, the Eq (1) softmax weights already sum to 1, so renormalizing
    the deslice weights BY NODE is a no-op and can't be caught this way.
    The real mutation risk is the plausible copy-paste of Eq (2)'s
    normalizer -- dividing by ``norm``, the per-SLICE mass summed across
    nodes -- into the Eq (4) deslice step.

    Three nodes split 2-1 across two slice "modes" (``x=[1,0]`` twice,
    ``x=[0,1]`` once) so the two slices accumulate unequal mass:
    ``norm0 = 1+a``, ``norm1 = 2-a`` where ``a = softmax([1,0])[0] =
    e/(1+e) != 0.5``, so ``norm0 != norm1``. ``to_q``/``to_k`` are zeroed
    so the Eq (3) token-attention logits are exactly 0 for every pair,
    making its softmax exactly ``[0.5, 0.5]`` by construction (no
    hand-arithmetic needed for that step) -- so both attended tokens equal
    the same value, ``mean_token = 0.5*(token0 + token1)``.

    Because Eq (1) weights sum to 1 per node and both attended tokens are
    identical, the CORRECT Eq (4) deslice (``sum_m token_out[m] * w[n,
    m]``, raw weights) collapses to ``mean_token`` for EVERY node,
    regardless of that node's own slice split -- so the correct forward
    output is IDENTICAL across all 3 nodes despite unequal per-node slice
    weights. A mutation dividing the deslice weights by the per-slice mass
    (``norm``) would break this: since ``norm0 != norm1``, the
    renormalized per-node factor ``w[n,:] . (1/norm)`` is not constant
    across the two node modes, so the two modes would no longer produce
    the same output. We assert the forward output matches the
    hand-derived ``mean_token`` for every node, and -- as evidence of
    mutation-sensitivity -- that the renormalized-by-slice-mass factor
    genuinely differs between the two node modes.
    """
    torch.manual_seed(0)
    attn = PhysicsAttentionIrregularMesh(
        dim=2, heads=1, dim_head=2, slice_num=2, dropout=0.0
    )
    attn.eval()
    eye2 = torch.eye(2)
    with torch.no_grad():
        attn.in_project_x.weight.copy_(eye2)
        attn.in_project_x.bias.zero_()
        attn.in_project_fx.weight.copy_(eye2)
        attn.in_project_fx.bias.zero_()
        attn.in_project_slice.weight.copy_(eye2)
        attn.in_project_slice.bias.zero_()
        attn.temperature.fill_(1.0)
        attn.to_q.weight.zero_()
        attn.to_k.weight.zero_()
        attn.to_v.weight.copy_(eye2)
        attn.to_out[0].weight.copy_(eye2)
        attn.to_out[0].bias.zero_()

    # 2 nodes in "mode A" (x=[1,0]), 1 node in "mode B" (x=[0,1]).
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    n = x.shape[0]

    # Hand-derive Eq (1)-(2): x_mid = x (identity in_project_x), so slice
    # logits = x_mid exactly (identity in_project_slice, temperature=1).
    a = torch.softmax(torch.tensor([1.0, 0.0]), dim=-1)[0]
    w_mode_a = torch.stack([a, 1 - a])
    w_mode_b = torch.stack([1 - a, a])
    norm0 = 2 * a + (1 - a)  # slice 0 mass: two mode-A nodes + one mode-B node
    norm1 = 2 * (1 - a) + a  # slice 1 mass
    fx_mode_a, fx_mode_b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    token0 = (2 * a * fx_mode_a + (1 - a) * fx_mode_b) / (norm0 + 1e-5)
    token1 = (2 * (1 - a) * fx_mode_a + a * fx_mode_b) / (norm1 + 1e-5)
    # Eq (3): to_q = to_k = 0 => logits are exactly 0 => softmax = [0.5, 0.5]
    # regardless of token values, so both attended tokens equal the mean.
    mean_token = 0.5 * (token0 + token1)

    out = attn(x, [(0, n)])
    assert torch.allclose(out, mean_token.expand(n, -1), atol=1e-6)

    # Sensitivity check: a norm-renormalized deslice would multiply
    # mean_token by a per-node factor w[n, :] . (1/norm) instead of by
    # w[n, :].sum() == 1; that factor is not constant across node modes
    # since norm0 != norm1, so the uniform-output result above would break.
    norm = torch.stack([norm0, norm1])
    factor_mode_a = (w_mode_a / norm).sum()
    factor_mode_b = (w_mode_b / norm).sum()
    assert abs(float(norm0 - norm1)) > 1e-3
    assert abs(float(factor_mode_a - factor_mode_b)) > 1e-3


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
    assert abs(float(lin.weight.detach().std()) - 0.02) < 0.006
    # The overwrite happened: slice projection is trunc_normal, NOT orthonormal
    # (orthogonal_ on the (slice_num=64, dim_head=16) weight gives Wᵀ W = I₁₆).
    w = net.blocks[0].attn.in_project_slice.weight
    assert abs(float(w.detach().std()) - 0.02) < 0.006
    assert not torch.allclose(w.T @ w, torch.eye(w.shape[1]), atol=0.1)
    for mod in net.modules():  # secondary: LayerNorm resets
        if isinstance(mod, torch.nn.LayerNorm):
            assert torch.all(mod.weight == 1.0) and torch.all(mod.bias == 0.0)


# --- ADR-0053 time-conditioning (thuml Time_Input scheme) ---


def test_timestep_embedding_shape_and_parity() -> None:
    # thuml sinusoidal embedding: (B, dim), cos|sin halves; odd dim gets a
    # trailing zero column.
    emb = timestep_embedding(torch.tensor([0.0, 0.5, 1.0]), 16)
    assert emb.shape == (3, 16)
    odd = timestep_embedding(torch.tensor([0.5]), 15)
    assert odd.shape == (1, 15)
    assert float(odd[0, -1]) == 0.0
    # distinct times -> distinct embeddings (else time cannot condition)
    assert not torch.allclose(emb[0], emb[2])


def test_time_conditioned_false_is_byte_identical() -> None:
    # Same seed, same params (no time_fc), and identical forward output: the
    # default network is unchanged by the ADR-0053 addition.
    torch.manual_seed(0)
    a = TransolverNet(
        node_in=7, out_size=4, hidden_dim=16, n_layers=2, n_heads=2, slice_num=4
    )
    torch.manual_seed(0)
    b = TransolverNet(
        node_in=7,
        out_size=4,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        slice_num=4,
        time_conditioned=False,
    )
    assert set(a.state_dict()) == set(b.state_dict())
    assert not any("time_fc" in k for k in a.state_dict())
    x = torch.randn(11, 7)
    assert torch.allclose(a(x, None), b(x, None))


def test_time_conditioned_registers_time_fc() -> None:
    net = TransolverNet(
        node_in=7,
        out_size=4,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        slice_num=4,
        time_conditioned=True,
    )
    assert any("time_fc" in k for k in net.state_dict())


def test_time_conditioned_requires_t() -> None:
    net = TransolverNet(
        node_in=7,
        out_size=4,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        slice_num=4,
        time_conditioned=True,
    )
    with pytest.raises(ValueError, match="time_conditioned=True"):
        net(torch.randn(5, 7), None)  # t missing


def test_time_conditioned_grad_flows_to_time_fc() -> None:
    net = TransolverNet(
        node_in=7,
        out_size=4,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        slice_num=4,
        time_conditioned=True,
    )
    net.train()
    out = net(torch.randn(11, 7), None, t=torch.tensor([[0.3]]))
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()
    g = net.time_fc[0].weight.grad
    assert g is not None and float(g.abs().sum()) > 0.0


def test_two_different_t_produce_different_outputs() -> None:
    net = TransolverNet(
        node_in=7,
        out_size=4,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        slice_num=4,
        time_conditioned=True,
    )
    net.eval()
    x = torch.randn(11, 7)
    with torch.no_grad():
        o1 = net(x, None, t=torch.tensor([[0.1]]))
        o2 = net(x, None, t=torch.tensor([[0.9]]))
    assert not torch.allclose(o1, o2)


def test_time_conditioned_batched_time_broadcast() -> None:
    # Two ragged examples get per-example time embeddings; the batched forward
    # matches running each example alone with its own scalar t.
    torch.manual_seed(0)
    net = TransolverNet(
        node_in=7,
        out_size=4,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        slice_num=4,
        time_conditioned=True,
    )
    net.eval()
    a, b = torch.randn(11, 7), torch.randn(5, 7)
    with torch.no_grad():
        batched = net(
            torch.cat([a, b]), torch.tensor([11, 5]), t=torch.tensor([[0.2], [0.8]])
        )
        singles = torch.cat(
            [
                net(a, None, t=torch.tensor([[0.2]])),
                net(b, None, t=torch.tensor([[0.8]])),
            ]
        )
    assert torch.allclose(batched, singles, atol=1e-5)
