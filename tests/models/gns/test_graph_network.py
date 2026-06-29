import torch

from structbench.models.gns.graph_network import EncodeProcessDecode


def test_encode_process_decode_output_shape_and_finite():
    n, e = 6, 10
    net = EncodeProcessDecode(
        nnode_in_features=7,
        nnode_out_features=3,
        nedge_in_features=3,
        latent_dim=16,
        nmessage_passing_steps=2,
        nmlp_layers=1,
        mlp_hidden_dim=16,
    )
    x = torch.randn(n, 7)
    edge_index = torch.randint(0, n, (2, e))
    edge_features = torch.randn(e, 3)
    out = net(x, edge_index, edge_features)
    assert out.shape == (n, 3)
    assert torch.isfinite(out).all()
