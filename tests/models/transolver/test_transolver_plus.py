"""Tests for the ADR-0057 Transolver++ eidetic-state edits.

Two independently-gated edits inside Physics-Attention's slice-weight path:
``adaptive_temperature`` (per-point/per-head MLP temperature) and
``slice_reparam`` (train-only Gumbel noise). Both default off and MUST be
byte-identical to the pre-0057 network when off.
"""

from __future__ import annotations

import torch

from structbench.models.transolver.network import (
    PhysicsAttentionIrregularMesh,
    TransolverNet,
)


def _attn(**kw) -> PhysicsAttentionIrregularMesh:
    torch.manual_seed(0)
    return PhysicsAttentionIrregularMesh(dim=16, heads=2, dim_head=8, slice_num=4, **kw)


def _x(p: int = 5) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(p, 16)


# --- byte-identical when both flags are off ----------------------------------


def test_defaults_are_off():
    a = _attn()
    assert a.adaptive_temperature is False
    assert a.slice_reparam is False
    # default param set unchanged: has the scalar temperature, no MLP params
    assert hasattr(a, "temperature")
    assert not hasattr(a, "proj_temperature")
    assert not hasattr(a, "temperature_bias")


def test_off_path_matches_reference_formula():
    """flags off => softmax(in_project_slice(x_mid) / temperature) exactly."""
    a = _attn().eval()
    x = _x()
    got = a._slice_weights(x)
    x_mid = a.in_project_x(x).reshape(-1, a.heads, a.dim_head)
    ref = torch.softmax(a.in_project_slice(x_mid) / a.temperature, dim=-1)
    assert torch.equal(got, ref)  # bit-for-bit, not just allclose


def test_slice_weights_are_a_distribution():
    for kw in ({}, {"adaptive_temperature": True}, {"slice_reparam": True}):
        w = _attn(**kw).eval()._slice_weights(_x())  # (P, H, M)
        assert w.shape == (5, 2, 4)
        assert torch.allclose(w.sum(-1), torch.ones(5, 2), atol=1e-6)
        assert (w >= 0).all()


# --- adaptive temperature -----------------------------------------------------


def test_adaptive_temperature_param_set():
    a = _attn(adaptive_temperature=True)
    assert hasattr(a, "proj_temperature")
    assert hasattr(a, "temperature_bias")
    assert not hasattr(a, "temperature")  # scalar replaced, state_dict stays clean
    # per-point/per-head positive temperature, clamped >= 0.01
    x_mid = a.in_project_x(_x()).reshape(-1, a.heads, a.dim_head)
    temp = torch.clamp(a.proj_temperature(x_mid) + a.temperature_bias, min=0.01)
    assert temp.shape == (5, 2, 1)
    assert (temp >= 0.01).all()


# --- slice reparam: train-only Gumbel (ADR-0057 D2) ---------------------------


def test_reparam_is_stochastic_in_train_mode():
    a = _attn(slice_reparam=True).train()
    x = _x()
    assert not torch.equal(a._slice_weights(x), a._slice_weights(x))


def test_reparam_is_deterministic_and_noop_in_eval_mode():
    a = _attn(slice_reparam=True).eval()
    x = _x()
    w1, w2 = a._slice_weights(x), a._slice_weights(x)
    assert torch.equal(w1, w2)  # deterministic at eval
    # and identical to the no-noise off-path (Gumbel gated off at eval)
    x_mid = a.in_project_x(x).reshape(-1, a.heads, a.dim_head)
    ref = torch.softmax(a.in_project_slice(x_mid) / a.temperature, dim=-1)
    assert torch.equal(w1, ref)


# --- threading through the full network --------------------------------------


def _net(**kw) -> TransolverNet:
    torch.manual_seed(0)
    return TransolverNet(
        node_in=8, out_size=3, hidden_dim=16, n_layers=2, n_heads=2, slice_num=4, **kw
    )


def test_net_forward_finite_with_both_edits():
    torch.manual_seed(2)
    feats = torch.randn(6, 8)
    out = _net(adaptive_temperature=True, slice_reparam=True).eval()(feats, None)
    assert out.shape == (6, 3)
    assert torch.isfinite(out).all()


def test_net_eval_is_deterministic_with_reparam():
    feats = torch.randn(6, 8)
    net = _net(slice_reparam=True).eval()
    assert torch.equal(net(feats, None), net(feats, None))
