"""ADR-0060: the aux-input (state-feedback) surface on the AR Transolver."""

import numpy as np
import pytest
import torch

from structbench.models.transolver import TransolverSimulator

_DIM = 3
_P = 5
_C = 2
_T = 7
_F = 2  # input frames / seed


def _sim(**kwargs):
    torch.manual_seed(0)
    return TransolverSimulator(
        dim=_DIM,
        hidden_dim=8,
        n_layers=1,
        n_heads=2,
        slice_num=2,
        n_aux=_C,
        **kwargs,
    )


def _bind(sim, with_aux=True, seed=0):
    rng = np.random.default_rng(seed)
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
    ref = torch.tensor(rng.random((_P, _DIM)), dtype=torch.float32)
    types = torch.zeros(_P, dtype=torch.int64)  # no kinematic rows (tripwire)
    gt = torch.tensor(rng.random((_T, _P, _DIM)), dtype=torch.float32).cumsum(0)
    gt_aux = (
        torch.tensor(rng.random((_T, _P, _C)), dtype=torch.float32)
        if with_aux
        else None
    )
    sim.bind_case(cells, ref, types, gt, gt_aux=gt_aux)
    return gt, gt_aux, types


def test_ctor_rejects_incompatible_schemes():
    with pytest.raises(ValueError, match="time_conditioned"):
        _sim(aux_input=True, time_conditioned=True)
    with pytest.raises(ValueError, match="frames_per_call"):
        _sim(aux_input=True, frames_per_call=4)


def test_input_width_grows_by_c():
    base = _sim(aux_input=False)
    fed = _sim(aux_input=True)
    assert (
        fed._node_normalizer._sum.shape[0] == base._node_normalizer._sum.shape[0] + _C
    )


def test_forward_train_requires_input_aux_when_on():
    sim = _sim(aux_input=True)
    rng = np.random.default_rng(1)
    x = torch.tensor(rng.random((_P, _DIM)), dtype=torch.float32)
    nxt = torch.tensor(rng.random((_P, _DIM)), dtype=torch.float32)
    aux = torch.tensor(rng.random((_P, _C)), dtype=torch.float32)
    types = torch.tensor([0, 0, 1, 3, 0], dtype=torch.int64)
    ref = torch.tensor(rng.random((_P, _DIM)), dtype=torch.float32)
    npp = torch.tensor([_P], dtype=torch.int64)
    with pytest.raises(ValueError, match="aux_state"):
        sim.forward_train(x, nxt, aux, types, ref, npp, accumulate=True)
    pred, target = sim.forward_train(
        x, nxt, aux, types, ref, npp, accumulate=True, input_aux=aux
    )
    assert pred.shape == (_P, _DIM + _C)
    assert target.shape == (_P, _DIM + _C)


def test_predict_requires_gt_aux_binding():
    sim = _sim(aux_input=True)
    gt, _, types = _bind(sim, with_aux=False)
    npp = torch.tensor([_P], dtype=torch.int64)
    window = gt[:_F].permute(1, 0, 2).contiguous()
    with pytest.raises(RuntimeError, match="gt_aux"):
        sim.predict_positions(window, npp, types)


def _two_step_rollout(sim, gt, types, mode):
    """Two self-rolled position steps; returns the two aux predictions."""
    sim.reset_rollout()
    sim.aux_feedback = mode
    npp = torch.tensor([_P], dtype=torch.int64)
    window = gt[:_F].permute(1, 0, 2).contiguous()
    with torch.no_grad():
        p1, a1 = sim.predict_positions(window, npp, types)
        window2 = torch.cat([window[:, 1:], p1.unsqueeze(1)], dim=1)
        p2, a2 = sim.predict_positions(window2, npp, types)
    return (a1, a2)


def test_oracle_and_self_agree_at_first_step_then_diverge():
    sim = _sim(aux_input=True)
    gt, gt_aux, types = _bind(sim)
    a1_self, a2_self = _two_step_rollout(sim, gt, types, "self")
    a1_orac, a2_orac = _two_step_rollout(sim, gt, types, "oracle")
    # step 1: both modes consume the GT state at the last seed frame
    torch.testing.assert_close(a1_self, a1_orac)
    # step 2: self consumes its own prediction, oracle consumes GT -> differ
    assert not torch.allclose(a2_self, a2_orac)


def test_reset_rollout_clears_feedback_state():
    sim = _sim(aux_input=True)
    gt, _, types = _bind(sim)
    a1_first, _ = _two_step_rollout(sim, gt, types, "self")
    a1_again, _ = _two_step_rollout(sim, gt, types, "self")
    torch.testing.assert_close(a1_first, a1_again)


def test_off_knob_is_inert():
    """aux_input=False ignores gt_aux entirely and needs no input_aux."""
    sim = _sim(aux_input=False)
    gt, _, types = _bind(sim)
    npp = torch.tensor([_P], dtype=torch.int64)
    window = gt[:_F].permute(1, 0, 2).contiguous()
    with torch.no_grad():
        pos, aux = sim.predict_positions(window, npp, types)
    assert pos.shape == (_P, _DIM)
    assert aux.shape == (_P, _C)


def _bind_with_kinematic(sim, seed=0):
    """Bind a case whose row 2 is kinematic (type 1)."""
    rng = np.random.default_rng(seed)
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
    ref = torch.tensor(rng.random((_P, _DIM)), dtype=torch.float32)
    types = torch.tensor([0, 0, 1, 0, 0], dtype=torch.int64)
    gt = torch.tensor(rng.random((_T, _P, _DIM)), dtype=torch.float32).cumsum(0)
    gt_aux = torch.tensor(rng.random((_T, _P, _C)), dtype=torch.float32)
    sim.bind_case(cells, ref, types, gt, gt_aux=gt_aux)
    return gt, gt_aux, types


def test_self_fed_cache_clamps_kinematic_rows_to_gt():
    """ADR-0060 amendment: kinematic rows of the fed-back state are GT —
    the analog of the rollout's GT position override (their decoder output
    is untargeted by the masked loss)."""
    sim = _sim(aux_input=True)
    gt, gt_aux, types = _bind_with_kinematic(sim)
    sim.aux_feedback = "self"
    npp = torch.tensor([_P], dtype=torch.int64)
    window = gt[:_F].permute(1, 0, 2).contiguous()
    with torch.no_grad():
        # predicting frame _F: kinematic rows must be overwritten with GT
        # position per the rollout contract before re-feeding
        p1, a1 = sim.predict_positions(window, npp, types)
    cached = sim._aux_state
    kin = types == 1
    torch.testing.assert_close(cached[kin], gt_aux[_F][kin])
    torch.testing.assert_close(cached[~kin], a1.detach()[~kin])


def test_counter_overrun_fails_loud_with_reset_guidance():
    """ADR-0060: parity with the position tripwire on kinematic-free cases."""
    sim = _sim(aux_input=True)
    gt, _, types = _bind(sim)  # kinematic-free
    sim.aux_feedback = "oracle"
    npp = torch.tensor([_P], dtype=torch.int64)
    with torch.no_grad():
        for t in range(_F, _T):
            window = gt[t - _F : t].permute(1, 0, 2).contiguous()
            sim.predict_positions(window, npp, types)
        # a further pass WITHOUT reset_rollout(): the counter cannot know
        # the true frame, so (like the position tripwire, which only fires
        # on kinematic rows) the first stale read passes — but the counter
        # then overruns the trajectory and must fail LOUDLY with guidance.
        window = gt[:_F].permute(1, 0, 2).contiguous()
        sim.predict_positions(window, npp, types)  # stale but in-bounds
        with pytest.raises(RuntimeError, match="reset_rollout"):
            sim.predict_positions(window, npp, types)


def test_off_knob_output_equality_with_and_without_gt_aux():
    """aux_input=False output is IDENTICAL whether gt_aux is bound or not."""
    outs = []
    for with_aux in (True, False):
        sim = _sim(aux_input=False)  # same torch seed inside _sim
        gt, _, types = _bind(sim, with_aux=with_aux)
        npp = torch.tensor([_P], dtype=torch.int64)
        window = gt[:_F].permute(1, 0, 2).contiguous()
        with torch.no_grad():
            outs.append(sim.predict_positions(window, npp, types))
    torch.testing.assert_close(outs[0][0], outs[1][0])
    torch.testing.assert_close(outs[0][1], outs[1][1])
