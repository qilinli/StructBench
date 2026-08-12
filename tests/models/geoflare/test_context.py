"""``ContextTokenizer`` / ``GeometricFeatureProcessor`` / ``MultiScaleContext``.

Semantics pinned to NVIDIA PhysicsNeMo's ``ContextProjector`` /
``MultiScaleFeatureExtractor`` / ``build_context`` (see
``scratch/2026-08-09-geoflare-grounding.md`` S10): clamped temperature
``[0.5, 5]``, slice-norm eps ``1e-2``, code-faithful part order
``[scale_1, scale_2, geometry]``.
"""

import torch

from structbench.models.geoflare.context import (
    ContextTokenizer,
    GeometricFeatureProcessor,
    MultiScaleContext,
)
from structbench.models.geoflare.geo_ops import ball_query


def test_multi_scale_context_shapes_at_defaults() -> None:
    # (a) GeoFlareConfig defaults (n_hidden=256, n_heads=8, n_hidden_local=32,
    # slice_num=128, radii=(0.05, 0.25), neighbors=(8, 32)) ->
    # dim_head_ctx = 256 // 8 = 32; context (H=8, S=128, 3*32=96);
    # local (N, 2*32=64).
    ctx = MultiScaleContext(
        n_hidden=256,
        n_heads=8,
        n_hidden_local=32,
        slice_num=128,
        radii=(0.05, 0.25),
        neighbors=(8, 32),
    )
    coords = torch.randn(20, 3)
    context, local = ctx(coords)
    assert context.shape == (8, 128, 96)
    assert local.shape == (20, 64)
    assert ctx.context_dim == context.shape[-1]


def test_context_tokenizer_slice1_pins_eps_and_fx_aggregation() -> None:
    # (b) slice_num=1 degenerate softmax: with only 1 slice, softmax over
    # the slice axis is exactly 1.0 for every node regardless of the slice
    # logits (softmax of a single element always normalizes to itself), so
    # in_project_x cannot affect the result through the slice weights.
    # w=1 for every (h, n) pair collapses the whole tokenizer to
    # ``token = sum_n(fx_mid[n]) / (N + 1e-2)`` -- this pins BOTH the eps
    # 1e-2 (an eps-1e-5 mutation shifts the value measurably at N=3) AND
    # fx-vs-x aggregation (a mutation aggregating x_mid instead of fx_mid
    # would leak in_project_x's DIFFERENT affine map).
    torch.manual_seed(0)
    tok = ContextTokenizer(dim=2, heads=1, dim_head=2, slice_num=1)
    tok.eval()
    with torch.no_grad():
        tok.in_project_fx.weight.copy_(torch.eye(2))
        tok.in_project_fx.bias.zero_()
        tok.in_project_x.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
        tok.in_project_x.bias.copy_(torch.tensor([10.0, 20.0]))

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    n = x.shape[0]
    fx_mid = x  # in_project_fx is identity
    x_mid = 2 * x + torch.tensor([10.0, 20.0])  # in_project_x, for the mutation check
    expected = fx_mid.sum(dim=0) / (n + 1e-2)
    eps_mutant = fx_mid.sum(dim=0) / (n + 1e-5)
    x_mutant = x_mid.sum(dim=0) / (n + 1e-2)
    assert not torch.allclose(expected, eps_mutant, atol=1e-4), (
        "fixture not sensitive to the eps mutation"
    )
    assert not torch.allclose(expected, x_mutant, atol=1e-4), (
        "fixture not sensitive to the fx-vs-x mutation"
    )

    out = tok(x)
    assert out.shape == (1, 1, 2)
    assert torch.allclose(out[0, 0], expected, atol=1e-6)
    assert not torch.allclose(out[0, 0], eps_mutant, atol=1e-6)
    assert not torch.allclose(out[0, 0], x_mutant, atol=1e-6)


def test_context_tokenizer_temperature_clamp_pinned() -> None:
    # (c) Temperature clamp pin: slice_num=2 (NOT (b)'s slice_num=1, where
    # any temperature softmaxes to 1.0 and this pin would be vacuous) with
    # DISTINCT per-slice logits. temperature.data is set to 0.1, below the
    # clamp floor of 0.5 -- a correct (clamped) implementation must behave
    # as though temperature were 0.5; an unclamped port would use 0.1 and
    # produce a measurably different softmax (the smaller the effective
    # temperature, the more the softmax saturates toward one-hot).
    torch.manual_seed(0)
    tok = ContextTokenizer(dim=1, heads=1, dim_head=1, slice_num=2)
    tok.eval()
    with torch.no_grad():
        tok.in_project_x.weight.copy_(torch.tensor([[1.0]]))
        tok.in_project_x.bias.zero_()
        tok.in_project_fx.weight.copy_(torch.tensor([[1.0]]))
        tok.in_project_fx.bias.zero_()
        # logits = x_mid * [1, -1] = [v, -v] -- distinct per-slice logits
        # whenever v != 0.
        tok.in_project_slice.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        tok.in_project_slice.bias.zero_()
        tok.temperature.fill_(0.1)

    x = torch.tensor([[1.0]])  # N=1 -> v=1, logits=[1, -1]
    logits = torch.tensor([1.0, -1.0])

    clamped_w = torch.softmax(logits / torch.clamp(torch.tensor(0.1), 0.5, 5.0), dim=-1)
    clamped_norm = clamped_w + 1e-2  # N=1
    expected = (clamped_w / clamped_norm).unsqueeze(-1)  # (S=2, D=1)

    unclamped_w = torch.softmax(logits / torch.tensor(0.1), dim=-1)
    unclamped_norm = unclamped_w + 1e-2
    mutant = (unclamped_w / unclamped_norm).unsqueeze(-1)

    assert not torch.allclose(expected, mutant, atol=1e-3), (
        "fixture not sensitive to the temperature-clamp mutation"
    )

    out = tok(x)
    assert out.shape == (1, 2, 1)
    assert torch.allclose(out[0], expected, atol=1e-5)
    assert not torch.allclose(out[0], mutant, atol=1e-5)


def test_multi_scale_context_part_order_scale1_scale2_geometry() -> None:
    # (d) Part-order pin: hand-set weights make each of the 3 tokenizers
    # output a distinct constant (in_project_fx weight zeroed, bias set to
    # a distinct vector, slice_num=1 so w=1 for every node and the token
    # collapses to ``N * bias / (N + 1e-2)`` regardless of the actual
    # input -- see the (b) derivation). Asserts the concat layout is
    # exactly [scale1, scale2, geometry] along the last dim.
    ctx = MultiScaleContext(
        n_hidden=2,
        n_heads=1,
        n_hidden_local=4,
        slice_num=1,
        radii=(10.0, 10.0),
        neighbors=(2, 2),
    )
    ctx.eval()

    def pin_constant(tok: ContextTokenizer, bias: list[float]) -> None:
        with torch.no_grad():
            tok.in_project_fx.weight.zero_()
            tok.in_project_fx.bias.copy_(torch.tensor(bias))

    pin_constant(ctx.scale_tokenizers[0], [1.0, 2.0])
    pin_constant(ctx.scale_tokenizers[1], [10.0, 20.0])
    pin_constant(ctx.geometry_tokenizer, [100.0, 200.0])

    coords = torch.randn(5, 3)
    n = coords.shape[0]
    with torch.no_grad():
        context, _ = ctx(coords)

    scale = n / (n + 1e-2)
    expected_scale1 = scale * torch.tensor([1.0, 2.0])
    expected_scale2 = scale * torch.tensor([10.0, 20.0])
    expected_geo = scale * torch.tensor([100.0, 200.0])

    assert context.shape == (1, 1, 6)
    assert torch.allclose(context[0, 0, 0:2], expected_scale1, atol=1e-4)
    assert torch.allclose(context[0, 0, 2:4], expected_scale2, atol=1e-4)
    assert torch.allclose(context[0, 0, 4:6], expected_geo, atol=1e-4)
    # Mutation-sensitivity: a geometry-first order would swap slot 0 and
    # slot 4, so asserting slot 0 == scale1 (not geometry) already fails
    # under that mutation; make the sensitivity explicit too.
    assert not torch.allclose(context[0, 0, 0:2], expected_geo, atol=1e-4)


def test_geometric_feature_processor_tanh_bounds_large_preactivation() -> None:
    # (e) tanh boundedness, made a real (non-vacuous) pin: force a huge
    # pre-tanh value via the last linear layer's bias so that, if tanh were
    # dropped or misapplied (e.g. INSIDE the stack before the final
    # Linear, or omitted), the output would blow up far outside (-1, 1).
    proc = GeometricFeatureProcessor(radius=1.0, k=2, n_hidden_local=4)
    proc.eval()
    with torch.no_grad():
        last_linear = proc.mlp[-1]
        last_linear.bias.fill_(1000.0)

    g = torch.randn(3, 3)
    out = proc(g)
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()
    assert (out.abs() <= 1.0).all()
    assert torch.allclose(out, torch.ones_like(out), atol=1e-3)  # tanh(~1000) ~= 1


def test_geometric_feature_processor_zero_pad_propagates_no_nan() -> None:
    # (e) Zero-pad propagation: 3 points, k=4 > n=3, so slot index 3 is
    # STRUCTURALLY zero-padded for every query row (ball_query's
    # eff_k = min(k, n) contract, Task 2) regardless of radius. Deliberately
    # NOT the "one real neighbour, rest padding" shape: with 3 distinct
    # non-origin points, 3 different (non-degenerate) neighbour slots carry
    # real, DIFFERENT-valued coordinates -- if only one slot were nonzero
    # (as in an earlier draft of this fixture, a 2-point line with a
    # query point sitting AT the origin), a flatten-order bug permuting
    # the k-axis before reshape (e.g. axis-major [x0,x1,..,y0,y1,..] instead
    # of interleaved [x0,y0,z0,x1,y1,z1,..]) would be INVISIBLE: the single
    # nonzero value would land in position 0 under either ordering. This
    # fixture's 3 distinct nonzero slots make the two orderings produce a
    # genuinely different flattened vector, so the cross-check below (an
    # independent manual flatten of ball_query's own output through the
    # same submodule) actually pins the flatten order, not just re-derives
    # whatever order forward() happens to use. The flattened zero row
    # (slot 3) must propagate through the MLP as ordinary zero input, not
    # NaN, and the whole output must still land in (-1, 1).
    torch.manual_seed(0)
    radius, k, n_hidden_local = 0.2, 4, 8
    proc = GeometricFeatureProcessor(radius=radius, k=k, n_hidden_local=n_hidden_local)
    proc.eval()
    g = torch.tensor([[5.0, 6.0, 7.0], [5.1, 6.0, 7.0], [5.0, 6.15, 7.0]])
    with torch.no_grad():
        out = proc(g)
        neighbors = ball_query(g, radius, k)  # (3, k, 3); slot 3 is exact zero
        assert torch.equal(neighbors[:, 3], torch.zeros(3, 3)), (
            "fixture does not actually exercise structural zero-padding"
        )
        expected = torch.tanh(proc.mlp(neighbors.reshape(3, -1)))

    assert out.shape == (3, n_hidden_local)
    assert torch.isfinite(out).all()
    assert (out.abs() <= 1.0).all()
    assert torch.allclose(out, expected, atol=1e-6)


def test_multi_scale_context_standardizes_coords_internally() -> None:
    # (f) Standardization-inside-builder invariance pin: MultiScaleContext
    # must call standardize_coords itself. A raw-coords mutation (skipping
    # internal standardization) would silently break neighbour finding at
    # mm/large-offset scale while passing every shape test -- so pin
    # invariance to a LARGE combined offset+scale, not just a small one.
    torch.manual_seed(0)
    ctx = MultiScaleContext(
        n_hidden=8,
        n_heads=2,
        n_hidden_local=4,
        slice_num=3,
        radii=(0.5, 1.5),
        neighbors=(3, 5),
    )
    ctx.eval()
    coords = torch.randn(12, 3)
    scaled = coords * 1000.0 + 5000.0
    with torch.no_grad():
        c0, l0 = ctx(coords)
        c1, l1 = ctx(scaled)
    assert torch.allclose(c0, c1, atol=1e-4)
    assert torch.allclose(l0, l1, atol=1e-4)


def test_geometric_feature_processor_supports_2d():
    # ADR-0047: the local-geometry MLP width is dim*k, not a hardcoded 3*k.
    import torch

    from structbench.models.geoflare.context import GeometricFeatureProcessor

    proc = GeometricFeatureProcessor(radius=1.0, k=3, n_hidden_local=8, dim=2)
    out = proc(torch.randn(11, 2))
    assert out.shape == (11, 8)
    assert out.abs().max() <= 1.0
