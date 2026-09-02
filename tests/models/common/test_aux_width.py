"""ADR-0059: mesh-family head widths and target assembly at C > 1."""

import numpy as np
import pytest
import torch

from structbench.models.geoflare import GeoFlareSimulator
from structbench.models.mgn import MeshSimulator
from structbench.models.transolver import TransolverSimulator


def _tiny_transolver(**kwargs):
    return TransolverSimulator(
        dim=3, hidden_dim=8, n_layers=1, n_heads=2, slice_num=2, **kwargs
    )


def _tiny_geoflare(**kwargs):
    return GeoFlareSimulator(
        dim=3,
        n_hidden=8,
        n_layers=1,
        n_heads=2,
        slice_num=2,
        n_hidden_local=4,
        **kwargs,
    )


def _tiny_mgn(**kwargs):
    return MeshSimulator(
        dim=3, latent=8, mp_steps=1, n_hidden=1, world_edge_radius=0.5, **kwargs
    )


_BUILDERS = [_tiny_transolver, _tiny_geoflare, _tiny_mgn]

_C = 3
_DIM = 3
_P = 5


def _bind(sim, T=6, P=_P, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
    ref = torch.tensor(rng.random((P, _DIM)), dtype=torch.float32)
    types = torch.tensor([0, 0, 1, 3, 0], dtype=torch.int64)
    gt = torch.tensor(rng.random((T, P, _DIM)), dtype=torch.float32).cumsum(0)
    sim.bind_case(cells, ref, types, gt)
    return gt, types


@pytest.mark.parametrize("build", _BUILDERS)
def test_forward_train_widths_at_c3(build):
    torch.manual_seed(1)
    sim = build(n_aux=_C)
    gt, types = _bind(sim)
    rng = np.random.default_rng(1)
    x_last = gt[0]
    next_positions = gt[1]
    next_aux = torch.tensor(rng.random((_P, _C)), dtype=torch.float32)
    ref = torch.tensor(rng.random((_P, _DIM)), dtype=torch.float32)
    npp = torch.tensor([_P], dtype=torch.int64)
    if isinstance(sim, MeshSimulator):  # MGN additionally takes mesh edges
        edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.int64).T
        pred, target = sim.forward_train(
            x_last, next_positions, next_aux, types, edges, ref, npp, accumulate=True
        )
    else:
        pred, target = sim.forward_train(
            x_last, next_positions, next_aux, types, ref, npp, accumulate=True
        )
    assert pred.shape == (_P, _DIM + _C)
    assert target.shape == (_P, _DIM + _C)
    # the trailing block of the target is the (normalized) aux block, so the
    # RAW target must reproduce next_aux exactly before normalization — check
    # via the normalizer inverse.
    raw = sim._target_normalizer.inverse(target)
    torch.testing.assert_close(raw[:, _DIM:], next_aux, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("build", _BUILDERS)
def test_predict_positions_aux_block_at_c3(build):
    torch.manual_seed(2)
    sim = build(n_aux=_C)
    gt, types = _bind(sim)
    window = gt[:2].permute(1, 0, 2).contiguous()  # (P, input_frames, dim)
    npp = torch.tensor([_P], dtype=torch.int64)
    with torch.no_grad():
        next_pos, aux = sim.predict_positions(window, npp, types)
    assert next_pos.shape == (_P, _DIM)
    assert aux.shape == (_P, _C)
