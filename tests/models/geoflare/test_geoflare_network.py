"""``GaleFlareAttention`` / ``GeoFlareBlock`` / ``GeoFlareNet`` (ADR-0041; ADR-0045).

FLARE encode/decode + GALE cross-attention math pinned against the
upstream ``GALE_FA`` reference (see
``scratch/2026-08-09-geoflare-grounding.md`` SS3-SS4, SS10).

No segment-leak pin lives in the ``GaleFlareAttention``-only tests below:
unlike ``PhysicsAttentionIrregularMesh``, ``GaleFlareAttention`` takes no
``segments`` argument and does no internal batching at all -- it is a pure
single-example function of ``(x, context)``. There is nothing inside it
that could "leak" across a ragged batch to pin against; that risk lives
entirely in ``GeoFlareNet``'s per-segment loop, where the geometry context
must be rebuilt per example -- the segment-leak/batched-equals-per-example
pin is written against ``GeoFlareNet`` below instead.
"""

import math

import torch

from structbench.models.geoflare.network import GaleFlareAttention, GeoFlareNet


def test_q_global_shape_grad_and_std() -> None:
    # (e) Pins the no-trunc_normal/no-orthogonal init decision: q_global is
    # plain torch.randn (std ~1.0). A 0.02-std trunc_normal_ init (the
    # convention TransolverNet's global init pass would apply) would fail
    # the std bound below.
    attn = GaleFlareAttention(
        dim=8, heads=4, dim_head=6, context_dim=8, slice_num=5, dropout=0.0
    )
    assert attn.q_global.shape == (4, 5, 6)
    assert attn.q_global.requires_grad
    std = float(attn.q_global.detach().std())
    assert 0.8 <= std <= 1.2


def test_decode_reuses_k_as_query_pin_m2() -> None:
    """FLARE decode must use ``k`` (not ``v``) as the decode query. M=2.

    M=1 would be vacuous: with a single global slot, the decode softmax is
    exactly 1.0 regardless of the query tensor, so a k-vs-v swap could never
    be detected. H=1, M=2, N=2, D=1: ``in_project_x``/``self_k``/``self_v``
    are set to the identity/an affine map so ``k``, ``v`` are simple,
    DISTINCT known scalars per node, and ``q_global``'s two rows are
    distinct known scalars, making the encode (Eq 8a) and decode (Eq 8b)
    formulas hand-computable in closed form via plain ``torch.softmax``
    calls (not the module) -- mirroring the ``test_context.py``
    hand-derivation style, not literal typed-out decimals.

    Sensitivity: the plausible bug is decoding with ``v`` (the OTHER
    per-point projection) as the query instead of ``k`` -- i.e.
    ``softmax(v_n . g_m)`` instead of ``softmax(k_n . g_m)``. Verified on
    paper before writing the assertion: ``v = k + 5`` here (both nodes'
    values become large and POSITIVE, 6 and 4, vs k's 1 and -1), so under
    the mutation both nodes' decode logits are dominated by the SAME global
    query row (``g_1 = 1``) regardless of node identity, collapsing both
    mutant outputs toward ``z_1`` alone -- a qualitatively different (and
    numerically confirmed distinct) result from the correct, node-dependent
    ``y``.
    """
    torch.manual_seed(0)
    attn = GaleFlareAttention(
        dim=1, heads=1, dim_head=1, context_dim=1, slice_num=2, dropout=0.0
    )
    attn.eval()
    with torch.no_grad():
        attn.in_project_x.weight.copy_(torch.tensor([[1.0]]))
        attn.in_project_x.bias.zero_()
        attn.self_k.weight.copy_(torch.tensor([[1.0]]))
        attn.self_k.bias.zero_()  # k_n = x_n
        attn.self_v.weight.copy_(torch.tensor([[1.0]]))
        attn.self_v.bias.copy_(torch.tensor([5.0]))  # v_n = x_n + 5, != k_n
        attn.q_global.copy_(torch.tensor([[[1.0], [-1.0]]]))  # (H=1,M=2,D=1)
        # Isolate the decode pin from the cross-attention branch entirely:
        # zero BOTH cross_v.weight and .bias (bias alone is nonzero by
        # default init) so y_cross = 0 exactly, and use an identity
        # out_linear so the module's output IS y = w*y_self (w=0.5 at
        # init) with no extra affine map to invert.
        attn.cross_v.weight.zero_()
        attn.cross_v.bias.zero_()
        attn.out_linear.weight.copy_(torch.tensor([[1.0]]))
        attn.out_linear.bias.zero_()

    x = torch.tensor([[1.0], [-1.0]])  # k = [1, -1], v = [6, 4]
    k = torch.tensor([1.0, -1.0])
    v = torch.tensor([6.0, 4.0])
    g = torch.tensor([1.0, -1.0])

    # Encode (Eq 8a): z_m = sum_n softmax(g_m . k_n)_n * v_n, softmax over N.
    z = torch.stack([(torch.softmax(g[m] * k, dim=-1) * v).sum() for m in range(2)])

    # Decode (Eq 8b), CORRECT: y_n = sum_m softmax(k_n . g_m)_m * z_m.
    y_correct = torch.stack(
        [(torch.softmax(k[n] * g, dim=-1) * z).sum() for n in range(2)]
    )
    # Decode, MUTATED (v as query instead of k):
    # y'_n = sum_m softmax(v_n . g_m)_m * z_m.
    y_mutant = torch.stack(
        [(torch.softmax(v[n] * g, dim=-1) * z).sum() for n in range(2)]
    )
    assert not torch.allclose(y_correct, y_mutant, atol=1e-2), (
        "fixture not sensitive to the k-vs-v decode-query mutation"
    )

    with torch.no_grad():
        # Context content is irrelevant: cross_v is zeroed, so y_cross = 0
        # regardless of what cross_q/cross_k compute from it.
        out = attn(x, torch.zeros(1, 1, 1))

    w = 0.5  # sigmoid(state_mixing) at init (state_mixing untouched here)
    expected = (w * y_correct).unsqueeze(-1)
    mutant_expected = (w * y_mutant).unsqueeze(-1)
    assert torch.allclose(out, expected, atol=1e-5)
    assert not torch.allclose(out, mutant_expected, atol=1e-3)


def test_mix_direction_pin_weighted_toward_self() -> None:
    """GALE mix must be ``w*self + (1-w)*cross``, not the reverse.

    ``cross_v.weight`` AND ``.bias`` are BOTH zeroed (zeroing only the
    weight would leave ``y_cross = bias != 0``, since ``nn.Linear``'s
    default bias init is nonzero) so ``y_cross = 0`` exactly regardless of
    ``context``. ``state_mixing.data`` is set so ``sigmoid -> 0.9``. With
    ``y_cross = 0``, the correct mix collapses to ``0.9 * y_self`` and the
    SWAPPED mix (``w*cross + (1-w)*self``) collapses to ``0.1 * y_self`` --
    both pushed through the SAME ``out_linear`` (weight+bias identical
    either way), so the 0.9-vs-0.1 factor on ``y_self`` is what
    distinguishes them. ``y_self`` itself is recomputed directly from the
    module's own submodules (``self_k``/``self_v``/``q_global``, via
    ``_attend``) -- the real encode/decode formula, not a call to
    ``forward()`` -- so this is a genuine hand-derivation, not a
    tautological self-comparison.
    """
    torch.manual_seed(0)
    attn = GaleFlareAttention(
        dim=3, heads=1, dim_head=2, context_dim=3, slice_num=2, dropout=0.0
    )
    attn.eval()
    # At init, sigmoid(state_mixing) == 0.5 (balanced 50/50 mix).
    assert torch.allclose(
        torch.sigmoid(attn.state_mixing), torch.tensor(0.5), atol=1e-6
    )

    with torch.no_grad():
        attn.cross_v.weight.zero_()
        attn.cross_v.bias.zero_()
        attn.state_mixing.data = torch.log(torch.tensor(9.0))  # sigmoid -> 0.9

    x = torch.randn(4, 3)
    context = torch.randn(1, 5, 3)  # (heads, S_ctx, context_dim)

    with torch.no_grad():
        n = x.shape[0]
        x_mid = attn.in_project_x(x).view(n, attn.heads, attn.dim_head)
        x_mid = x_mid.permute(1, 0, 2)
        k = attn.self_k(x_mid)
        v = attn.self_v(x_mid)
        z = GaleFlareAttention._attend(attn.q_global, k, v)
        y_self = GaleFlareAttention._attend(k, attn.q_global, z)
        y_self_flat = y_self.permute(1, 0, 2).reshape(n, attn.heads * attn.dim_head)

        w = 0.9
        expected_correct = attn.out_linear(w * y_self_flat)
        expected_swapped = attn.out_linear((1.0 - w) * y_self_flat)
        out = attn(x, context)

    assert torch.allclose(
        torch.sigmoid(attn.state_mixing), torch.tensor(0.9), atol=1e-6
    )
    assert torch.allclose(out, expected_correct, atol=1e-5)
    assert not torch.allclose(out, expected_swapped, atol=1e-5)


def test_scale_1_0_pin_encode_softmax_d4() -> None:
    """``_SCALE = 1.0`` pins the encode softmax, its OWN case at D=4.

    NOT the D=1 case used by the decode pin above: at ``dim_head=1``,
    ``1.0`` and ``dim_head**-0.5 = 1.0`` COINCIDE, so a D=1 fixture cannot
    distinguish the two. At D=4, ``dim_head**-0.5 = 0.5`` halves the encode
    logits, which is confirmed below (both by direct assertion and by the
    ~0.035-per-channel gap between the scale-1.0 and scale-0.5 hand
    derivations, computed via ``torch.softmax`` directly -- not the module
    -- mirroring the encode half of Eq 8a) to be a real, non-vacuous
    difference before it is used as an expected value.
    """
    torch.manual_seed(0)
    attn = GaleFlareAttention(
        dim=4, heads=1, dim_head=4, context_dim=4, slice_num=1, dropout=0.0
    )
    attn.eval()
    with torch.no_grad():
        attn.in_project_x.weight.copy_(torch.eye(4))
        attn.in_project_x.bias.zero_()
        attn.self_k.weight.copy_(torch.eye(4))
        attn.self_k.bias.zero_()
        attn.self_v.weight.copy_(torch.eye(4))
        attn.self_v.bias.zero_()
        attn.q_global.copy_(torch.ones(1, 1, 4))

    x = torch.tensor([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]])
    with torch.no_grad():
        n = x.shape[0]
        x_mid = attn.in_project_x(x).view(n, attn.heads, attn.dim_head)
        x_mid = x_mid.permute(1, 0, 2)
        k = attn.self_k(x_mid)
        v = attn.self_v(x_mid)

    # Hand derivation: dots[0, 0, n] = q_global . k_n = +-4 (dot of two
    # length-4 all +-1 vectors).
    dots = torch.tensor([4.0, -4.0])
    expected_w_scale1 = torch.softmax(dots * 1.0, dim=-1)
    expected_z_scale1 = expected_w_scale1 @ v[0]

    mutant_w = torch.softmax(dots * (4**-0.5), dim=-1)  # dim_head**-0.5 = 0.5
    mutant_z = mutant_w @ v[0]
    assert not torch.allclose(expected_z_scale1, mutant_z, atol=1e-2), (
        "fixture not sensitive to the scale=dim_head**-0.5 mutation"
    )

    with torch.no_grad():
        z = GaleFlareAttention._attend(attn.q_global, k, v)  # (H, M, D)
    assert torch.allclose(z[0, 0], expected_z_scale1, atol=1e-5)
    assert not torch.allclose(z[0, 0], mutant_z, atol=1e-3)


def test_attention_forward_shape_single_example() -> None:
    attn = GaleFlareAttention(
        dim=8, heads=2, dim_head=4, context_dim=6, slice_num=3, dropout=0.0
    )
    out = attn(torch.randn(5, 8), torch.randn(2, 7, 6))
    assert out.shape == (5, 8)


def test_scale_constant_value() -> None:
    # Sanity companion to the (g) pin: the module-level constant itself is
    # the flat 1.0 the upstream comment describes, not a math.isclose
    # coincidence with any other formula.
    from structbench.models.geoflare.network import _SCALE

    assert _SCALE == 1.0
    assert not math.isclose(_SCALE, 4**-0.5)


# ---------------------------------------------------------------------------
# GeoFlareBlock / GeoFlareNet -- integration pins (a), (b), (f).
# ---------------------------------------------------------------------------

_TINY = dict(n_hidden=16, n_layers=2, n_heads=2, slice_num=4, n_hidden_local=4)


def test_geoflare_net_forward_shape_tiny_config() -> None:
    # (a) Tiny config from the brief: effective = 16 + 4*2 = 24,
    # dim_head_block = 24 // 2 = 12, ctx dim_head = 16 // 2 = 8,
    # context_dim = 3*8 = 24.
    net = GeoFlareNet(node_in=7, out_size=4, **_TINY)
    assert net.context_builder.dim_head_ctx == 8
    assert net.context_builder.context_dim == 24
    coords = torch.randn(11, 3)
    feats = torch.randn(11, 7)
    out = net(feats, coords, None)
    assert out.shape == (11, 4)


def test_geoflare_net_batched_matches_per_example_segment_leak_pin() -> None:
    """THE segment-leak killer pin: batched forward == per-example forwards.

    Two ragged examples, concatenated through ``n_particles_per_example``,
    must produce EXACTLY the same output as running each example alone --
    this is only true if BOTH the geometry context AND the FLARE
    encode/decode softmaxes are computed per-example (segmented), not once
    over the concatenated ``sum_P`` tensor. A global (non-segmented) FLARE
    encode softmax over the concatenated ``N`` would silently mix examples
    together and fail this at both examples simultaneously; a
    non-segmented context build would additionally corrupt the ball query
    (neighbours found across unrelated meshes). No far-apart-coordinate
    construction is needed to expose this -- the FLARE encode operates on
    PROJECTED FEATURES, not raw coordinates, so ordinary random coordinates
    for both examples are already sufficient; segmenting correctly is
    exactly what is under test, not geometric separation.
    """
    torch.manual_seed(0)
    net = GeoFlareNet(node_in=7, out_size=4, **_TINY)
    net.eval()
    feats_a, feats_b = torch.randn(9, 7), torch.randn(6, 7)
    coords_a, coords_b = torch.randn(9, 3), torch.randn(6, 3)
    with torch.no_grad():
        batched = net(
            torch.cat([feats_a, feats_b]),
            torch.cat([coords_a, coords_b]),
            torch.tensor([9, 6]),
        )
        single_a = net(feats_a, coords_a, None)
        single_b = net(feats_b, coords_b, None)
    assert torch.allclose(batched[:9], single_a, atol=1e-5)
    assert torch.allclose(batched[9:], single_b, atol=1e-5)


def _tokenizer_params(dim_in: int, heads: int, dim_head: int, slice_num: int) -> int:
    return (
        2 * (dim_in + 1) * heads * dim_head  # in_project_x + in_project_fx
        + (dim_head + 1) * slice_num  # in_project_slice
        + heads  # temperature
    )


def _processor_params(k: int, n_hidden_local: int) -> int:
    h1 = n_hidden_local
    h2 = n_hidden_local // 2
    return (3 * k + 1) * h1 + (h1 + 1) * h2 + (h2 + 1) * h1  # 3-linear MLP


def _context_params(
    n_hidden: int,
    n_heads: int,
    n_hidden_local: int,
    slice_num: int,
    neighbors: tuple[int, int],
) -> int:
    dim_head_ctx = n_hidden // n_heads
    total = _tokenizer_params(3, n_heads, dim_head_ctx, slice_num)  # geometry
    for k in neighbors:
        total += _processor_params(k, n_hidden_local)
        total += _tokenizer_params(n_hidden_local, n_heads, dim_head_ctx, slice_num)
    return total


def _attn_params(
    effective: int, n_heads: int, dim_head: int, context_dim: int, slice_num: int
) -> int:
    return (
        n_heads * slice_num * dim_head  # q_global
        + (effective + 1) * n_heads * dim_head  # in_project_x
        + 3 * (dim_head + 1) * dim_head  # self_k, self_v, cross_q
        + 2 * (context_dim + 1) * dim_head  # cross_k, cross_v
        + 1  # state_mixing
        + (n_heads * dim_head + 1) * effective  # out_linear
    )


def _ffn_params(effective: int, mlp_ratio: int) -> int:
    hidden = effective * mlp_ratio
    return 2 * effective + (effective + 1) * hidden + (hidden + 1) * effective


def _block_params(
    effective: int,
    n_heads: int,
    dim_head: int,
    context_dim: int,
    slice_num: int,
    mlp_ratio: int,
) -> int:
    return (
        2 * effective  # ln_1
        + _attn_params(effective, n_heads, dim_head, context_dim, slice_num)
        + _ffn_params(effective, mlp_ratio)
    )


def _decoder_params(effective: int, out_size: int) -> int:
    return 2 * effective + (effective + 1) * out_size


def _formula(
    node_in: int,
    out_size: int,
    n_hidden: int,
    n_layers: int,
    n_heads: int,
    slice_num: int,
    mlp_ratio: int,
    n_hidden_local: int,
    neighbors: tuple[int, int],
) -> int:
    effective = n_hidden + n_hidden_local * 2
    dim_head_block = effective // n_heads
    dim_head_ctx = n_hidden // n_heads
    context_dim = 3 * dim_head_ctx
    preprocess = (node_in + 1) * (2 * n_hidden) + (2 * n_hidden + 1) * n_hidden
    context = _context_params(n_hidden, n_heads, n_hidden_local, slice_num, neighbors)
    block = _block_params(
        effective, n_heads, dim_head_block, context_dim, slice_num, mlp_ratio
    )
    decoder = _decoder_params(effective, out_size)
    return preprocess + context + n_layers * block + decoder


def test_geoflare_net_parameter_count_formula_tiny_and_defaults() -> None:
    # (f) Structural formula derived term-by-term (preprocess + context
    # pathway + per-block attention/FFN/LNs + decoder) from the module's
    # OWN structure above, checked at the tiny config (brief's exact
    # values) AND at defaults with node_in=18, out=4 (Transolver-suite
    # precedent, test_reference_parameter_count).
    cases = [
        (dict(node_in=7, out_size=4, **_TINY), (8, 32)),
        (dict(node_in=18, out_size=4), (8, 32)),  # every other kwarg default
    ]
    for kwargs, neighbors in cases:
        net = GeoFlareNet(**kwargs)
        n = sum(p.numel() for p in net.parameters())
        expected = _formula(
            node_in=kwargs["node_in"],
            out_size=kwargs["out_size"],
            n_hidden=kwargs.get("n_hidden", 256),
            n_layers=kwargs.get("n_layers", 6),
            n_heads=kwargs.get("n_heads", 8),
            slice_num=kwargs.get("slice_num", 128),
            mlp_ratio=kwargs.get("mlp_ratio", 4),
            n_hidden_local=kwargs.get("n_hidden_local", 32),
            neighbors=neighbors,
        )
        assert n == expected
