import torch

from structbench.models.cgn.graph_network import (
    EncodeProcessDecode,
    InteractionNetwork,
)


def test_interaction_network_edge_stream_is_input_doubled():
    """PIN: the edge output is the residual (2x input), not the edge-MLP output.

    PyG's update() receives the ORIGINAL edge_features kwarg (the message()
    output only feeds node aggregation), so forward returns
    edge_features + edge_features_residual = 2 * input. This is faithful to the
    trained sgnn lineage (ADR-0034); rewiring it to propagate the edge-MLP
    output is a behavioural change needing an ADR + retrain, so it is pinned.
    """
    torch.manual_seed(0)
    d = 8
    net = InteractionNetwork(
        nnode_in=d,
        nnode_out=d,
        nedge_in=d,
        nedge_out=d,
        nmlp_layers=1,
        mlp_hidden_dim=d,
    )
    n, e = 5, 12
    x = torch.randn(n, d)
    edge_index = torch.randint(0, n, (2, e))
    edge_features = torch.randn(e, d)
    _, edge_out = net(x, edge_index, edge_features)
    torch.testing.assert_close(edge_out, 2.0 * edge_features)


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
