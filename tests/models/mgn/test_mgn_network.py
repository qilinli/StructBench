import torch

from structbench.models.mgn.network import MGNet, build_mlp


def _net(**kw):
    torch.manual_seed(0)
    defaults = dict(
        node_in=12,
        mesh_edge_in=8,
        world_edge_in=4,
        out_size=4,
        latent=16,
        mp_steps=2,
        n_hidden=2,
    )
    defaults.update(kw)
    return MGNet(**defaults)


def test_forward_shapes():
    net = _net()
    P, Em, Ew = 5, 12, 4
    out = net(
        torch.randn(P, 12),
        torch.randint(0, P, (2, Em)),
        torch.randn(Em, 8),
        torch.randint(0, P, (2, Ew)),
        torch.randn(Ew, 4),
    )
    assert out.shape == (P, 4)


def test_forward_empty_world_edges():
    net = _net()
    P, Em = 5, 12
    out = net(
        torch.randn(P, 12),
        torch.randint(0, P, (2, Em)),
        torch.randn(Em, 8),
        torch.empty(2, 0, dtype=torch.int64),
        torch.empty(0, 4),
    )
    assert out.shape == (P, 4)


def test_message_passing_propagates_information():
    # node 0's input feature must influence node 1's output via the edge 0->1
    net = _net(mp_steps=1)
    nf = torch.zeros(2, 12)
    mesh = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    ef = torch.zeros(2, 8)
    base = net(nf, mesh, ef, torch.empty(2, 0, dtype=torch.int64), torch.empty(0, 4))
    nf2 = nf.clone()
    nf2[0, 0] = 10.0
    pert = net(nf2, mesh, ef, torch.empty(2, 0, dtype=torch.int64), torch.empty(0, 4))
    assert not torch.allclose(base[1], pert[1])


def test_build_mlp_layernorm_toggle():
    with_ln = build_mlp(3, 8, 2, 5, layer_norm=True)
    without = build_mlp(3, 8, 2, 5, layer_norm=False)
    assert isinstance(list(with_ln.children())[-1], torch.nn.LayerNorm)
    assert not isinstance(list(without.children())[-1], torch.nn.LayerNorm)


def test_recompute_activation_default_on():
    # Checkpointing is the default so the blessing run gets it without a config
    # knob; it is output-identical, so no result changes (ADR-0043 §8).
    assert _net()._recompute_activation is True


def test_recompute_activation_matches_plain():
    # Activation checkpointing must be output- and gradient-identical to the
    # plain forward loop -- it only trades memory for a backward recompute.
    net = _net(mp_steps=3)
    P, Em, Ew = 6, 14, 5
    torch.manual_seed(1)
    nf = torch.randn(P, 12)
    mesh = torch.randint(0, P, (2, Em))
    ef = torch.randn(Em, 8)
    world = torch.randint(0, P, (2, Ew))
    wf = torch.randn(Ew, 4)

    def run(flag):
        net._recompute_activation = flag
        net.zero_grad(set_to_none=True)
        out = net(nf, mesh, ef, world, wf)
        out.pow(2).sum().backward()
        grads = {n: p.grad.detach().clone() for n, p in net.named_parameters()}
        return out.detach().clone(), grads

    out_plain, grads_plain = run(False)
    out_ckpt, grads_ckpt = run(True)

    # The value returned is the initial forward pass, so it is exact.
    torch.testing.assert_close(out_ckpt, out_plain, rtol=0, atol=0)
    # Gradients come from a recompute; equal up to float noise on CPU.
    for name, g_plain in grads_plain.items():
        torch.testing.assert_close(grads_ckpt[name], g_plain, rtol=1e-5, atol=1e-6)


def test_recompute_activation_empty_world_edges_backward():
    # The Ew == 0 path must survive the checkpointed backward (empty world
    # tensors flow through the recomputed block).
    net = _net()  # default: checkpointing on
    P, Em = 5, 12
    out = net(
        torch.randn(P, 12),
        torch.randint(0, P, (2, Em)),
        torch.randn(Em, 8),
        torch.empty(2, 0, dtype=torch.int64),
        torch.empty(0, 4),
    )
    out.pow(2).sum().backward()
    assert out.shape == (P, 4)


def test_recompute_activation_eval_path_no_grad():
    # Under no_grad the block runs plainly (nothing to checkpoint); output shape
    # is unaffected by the default-on flag.
    net = _net()
    P, Em, Ew = 5, 12, 4
    with torch.no_grad():
        out = net(
            torch.randn(P, 12),
            torch.randint(0, P, (2, Em)),
            torch.randn(Em, 8),
            torch.randint(0, P, (2, Ew)),
            torch.randn(Ew, 4),
        )
    assert out.shape == (P, 4)
