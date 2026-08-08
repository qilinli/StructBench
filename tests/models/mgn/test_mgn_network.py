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
