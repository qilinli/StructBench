"""Tests for the ADR-0052 hybrid local+global branch (TC adaptation).

A gated MGN-style MP sublayer over a reference-config radius graph on middle
blocks. Off (hybrid_mp=False) is byte-identical; the gate inits to 0 so the
branch starts as a no-op.
"""

from __future__ import annotations

import torch

from structbench.models.transolver.network import (
    LocalMessagePassing,
    TransolverNet,
    hybrid_block_indices,
)


def test_hybrid_block_indices():
    assert hybrid_block_indices(8, 0) == set()
    assert hybrid_block_indices(8, 2) == {3, 4}  # middle, never 0 or 7
    assert hybrid_block_indices(4, 2) == {1, 2}
    assert min(hybrid_block_indices(8, 6)) >= 1 and max(hybrid_block_indices(8, 6)) <= 6
    try:
        hybrid_block_indices(4, 3)  # only 2 middle blocks available
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_local_message_passing_shape_and_mean_norm():
    mp = LocalMessagePassing(hidden_dim=8, edge_in=3)
    fx = torch.randn(4, 8)
    # node 3 is isolated (no incoming edges) -> its agg is zeros -> node_mlp([fx,0])
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    ef = torch.randn(3, 3)
    out = mp(fx, ei, ef)
    assert out.shape == (4, 8) and torch.isfinite(out).all()
    ref_iso = mp.node_mlp(torch.cat([fx[3], torch.zeros(8)]))
    assert torch.allclose(out[3], ref_iso, atol=1e-6)


def _net(**kw):
    torch.manual_seed(0)
    return TransolverNet(
        node_in=8, out_size=3, hidden_dim=16, n_layers=6, n_heads=2, slice_num=4, **kw
    )


def test_off_has_no_hybrid_params():
    net = _net()
    assert not any("mp_gate" in n or "ln_local" in n for n, _ in net.named_parameters())
    assert torch.isfinite(net.eval()(torch.randn(6, 8), None)).all()  # no graph needed


def _graph(p=6, e=8):
    torch.manual_seed(1)
    ei = torch.randint(0, p, (2, e))
    return ei, torch.randn(e, 3)


def test_on_blocks_guard_and_gate_init():
    net = _net(hybrid_mp=True, hybrid_edge_in=3, hybrid_blocks=2).eval()
    mp_blocks = [i for i, b in enumerate(net.blocks) if getattr(b, "hybrid_mp", False)]
    assert mp_blocks == [2, 3]  # middle of 6
    assert all(float(net.blocks[i].mp_gate) == 0.0 for i in mp_blocks)  # no-op start
    out = net(torch.randn(6, 8), None, graph=_graph())
    assert out.shape == (6, 3) and torch.isfinite(out).all()
    try:
        net(torch.randn(6, 8), None)  # hybrid on, no graph
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_gate_controls_mp_contribution():
    net = _net(hybrid_mp=True, hybrid_edge_in=3, hybrid_blocks=2).eval()
    x = torch.randn(6, 8)
    g1, g2 = _graph(e=8), _graph(e=5)
    # gate=0 (init): the graph must not affect the output at all
    a = net(x, None, graph=g1)
    b = net(x, None, graph=g2)
    assert torch.equal(a, b)
    # open the gates: now different graphs give different outputs
    for blk in net.blocks:
        if getattr(blk, "hybrid_mp", False):
            torch.nn.init.constant_(blk.mp_gate, 1.0)
    assert not torch.equal(net(x, None, graph=g1), net(x, None, graph=g2))
