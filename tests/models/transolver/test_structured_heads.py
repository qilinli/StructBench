"""ADR-0064 constitutively-structured admissible heads: ctor guards, the
hardening-curve interp contract, the decode's by-construction guarantees
(D2 = 0, peeq >= its fed anchor), smoothness at v = 0, checkpoint-key
parity when off, and the soft-hinge comparator helper."""

import numpy as np
import pytest
import torch

from structbench.cli.train import _fm_admissibility_hinge
from structbench.models.transolver import (
    TransolverSimulator,
    hardening_sigma_y,
    plane_strain_vm,
)

DIM = 2
C = 6  # the canonical state block: deviator(3) + peeq + energy + density
T = 6
P = 5

#: Taylor's *MAT_ELASTIC_PLASTIC_HYDRO table (deck-verbatim knots, MPa) —
#: the same values the benchmark spec registers (ADR-0064).
HE = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0)
HS = (
    199.3,
    251.1,
    250.9,
    321.4,
    344.8,
    362.8,
    376.6,
    387.7,
    395.4,
    406.4,
    415.9,
    416.7,
    419.0,
    421.1,
    422.2,
    422.2,
)

#: Float-only slack over the exact mathematical guarantee (the analysis
#: instrument's tolerance is 1.001; measured worst-case here is ~1 + 2e-7).
D2_FLOAT_TOL = 1 + 1e-5


def _sim(**over):
    kwargs = dict(
        dim=DIM,
        hidden_dim=16,
        n_layers=1,
        n_heads=2,
        slice_num=4,
        node_type_size=3,
        kinematic_types=(2,),
        scripted_types=(2,),
        n_aux=C,
        time_conditioned=True,
        aux_input=True,
        flow_map=True,
        flow_map_structured_heads=True,
        hardening_curve=(HE, HS),
    )
    kwargs.update(over)
    return TransolverSimulator(**kwargs)


def _bind(sim, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    positions = torch.rand((T, P, DIM), generator=g)
    aux = torch.rand((T, P, C), generator=g)
    reference = torch.rand((P, DIM), generator=g)
    ptype = torch.tensor([0, 0, 0, 0, 2], dtype=torch.long)
    cells = torch.zeros((1, 3), dtype=torch.long)
    sim.bind_case(cells, reference, ptype, positions, gt_aux=aux)
    return positions, aux, reference


def test_ctor_guards():
    with pytest.raises(ValueError, match="requires flow_map=True"):
        _sim(time_conditioned=False, aux_input=False, flow_map=False)
    with pytest.raises(ValueError, match="6-channel"):
        _sim(n_aux=4)
    with pytest.raises(ValueError, match="hardening_curve"):
        _sim(hardening_curve=None)
    with pytest.raises(ValueError, match="knot pairs"):
        _sim(hardening_curve=((0.0,), (199.3,)))
    with pytest.raises(ValueError, match="strictly increasing"):
        _sim(hardening_curve=((0.0, 1.0, 1.0), (199.3, 250.0, 260.0)))
    _sim()  # the full combination constructs


def test_checkpoint_keys_off_path_unchanged():
    """The hardening buffers exist ONLY when the knob is on, so every
    existing (unstructured) checkpoint state_dict is untouched (ADR-0064
    byte-identity surface)."""
    torch.manual_seed(0)
    off = _sim(flow_map_structured_heads=False, hardening_curve=None)
    torch.manual_seed(0)
    on = _sim()
    extra = set(on.state_dict()) - set(off.state_dict())
    assert extra == {"_hardening_peeq", "_hardening_sy"}
    for k in off.state_dict():
        torch.testing.assert_close(off.state_dict()[k], on.state_dict()[k])


def test_hardening_sigma_y_matches_np_interp():
    """The spec-hook contract: ``np.interp`` semantics — linear between
    knots, END-CLAMPED outside, non-monotone sigma_y knots verbatim."""
    he, hs = torch.tensor(HE), torch.tensor(HS)
    x = torch.cat([torch.linspace(-2.0, 25.0, 4001), torch.tensor(HE)])
    ours = hardening_sigma_y(x, he, hs)
    ref = np.interp(x.numpy(), HE, HS)
    np.testing.assert_allclose(ours.numpy(), ref, atol=1e-3)  # float32
    # end clamps exactly (to the float32 knot values)
    assert float(hardening_sigma_y(torch.tensor(-5.0), he, hs)) == float(hs[0])
    assert float(hardening_sigma_y(torch.tensor(99.0), he, hs)) == float(hs[-1])
    # differentiable, with the local hardening slope as gradient
    xg = torch.tensor([0.25], requires_grad=True)
    hardening_sigma_y(xg, he, hs).backward()
    np.testing.assert_allclose(
        float(xg.grad), (HS[1] - HS[0]) / (HE[1] - HE[0]), rtol=1e-5
    )


def test_decode_invariants():
    """D2 = 0 and peeq >= anchor BY CONSTRUCTION on arbitrary raw outputs;
    the unconstrained slices equal the plain normalizer inverse."""
    sim = _sim()
    he, hs = torch.tensor(HE), torch.tensor(HS)
    g = torch.Generator().manual_seed(3)
    net_out = (torch.rand((500, DIM + C), generator=g) - 0.5) * 8
    anchor_peeq = torch.rand(500, generator=g) * 3
    raw = sim._decode_structured(net_out, anchor_peeq)
    assert raw.shape == (500, DIM + C)
    dev, peeq = raw[:, DIM : DIM + 3], raw[:, DIM + 3]
    vm = plane_strain_vm(dev)
    sy = hardening_sigma_y(peeq, he, hs)
    assert bool((vm <= sy * D2_FLOAT_TOL).all())  # D2 = 0
    assert bool((peeq >= anchor_peeq).all())  # hand-off-level D3 = 0
    lin = sim._target_normalizer.inverse(net_out)
    torch.testing.assert_close(raw[:, :DIM], lin[:, :DIM])  # displacement
    torch.testing.assert_close(raw[:, DIM + 4 :], lin[:, DIM + 4 :])  # E, rho
    # gradients flow through the whole decode (sigma_y coupling included)
    net_out.requires_grad_(True)
    sim._decode_structured(net_out, anchor_peeq).sum().backward()
    assert bool(torch.isfinite(net_out.grad).all())


def test_decode_smooth_at_zero_deviator():
    """tanh(x)/x -> 1: v = 0 exactly gives dev = 0 with finite gradients
    (the ADR-0064 rationale for the tanh form over an eps-guarded unit
    direction)."""
    sim = _sim()
    z = torch.zeros((3, DIM + C), requires_grad=True)
    raw = sim._decode_structured(z, torch.zeros(3))
    assert float(raw[:, DIM : DIM + 3].abs().max()) == 0.0
    raw.sum().backward()
    assert bool(torch.isfinite(z.grad).all())


def test_forward_train_and_predict_share_the_decode():
    """Both surfaces emit admissible states against the anchor actually
    FED (train: the anchor_aux kwarg; eval: the set_anchor cache), and the
    training pred_norm round-trips to the structured raw state."""
    he, hs = torch.tensor(HE), torch.tensor(HS)
    sim = _sim()
    positions, aux, reference = _bind(sim)
    ptype = torch.tensor([0, 0, 0, 0, 2], dtype=torch.long)
    anchor_aux = aux[1] * 3  # off-GT values, as a noised anchor would be
    pred, target = sim.forward_train_tc(
        positions[3],
        aux[3],
        ptype,
        reference,
        torch.tensor([P]),
        torch.tensor([0.4]),
        accumulate=True,
        anchor_disp=positions[1] - reference,
        anchor_vel=positions[1] - positions[0],
        anchor_aux=anchor_aux,
        anchor_time_feature=torch.full((P, 1), 0.2),
    )
    assert pred.shape == target.shape == (P, DIM + C)
    _disp, raw_aux = sim.train_output_state(pred)
    vm = plane_strain_vm(raw_aux[:, :3])
    sy = hardening_sigma_y(raw_aux[:, 3], he, hs)
    assert bool((vm <= sy * D2_FLOAT_TOL).all())
    # normalize/inverse round trip: allow float slack on the exact floor
    assert bool((raw_aux[:, 3] >= anchor_aux[:, 3] - 1e-4).all())

    sim.set_anchor(1, positions[0], positions[1], aux[1], anchor_t_norm=0.2)
    sim.eval()
    with torch.no_grad():
        pos, out_aux = sim.predict_state_at(3, 0.4)
    assert pos.shape == (P, DIM) and out_aux.shape == (P, C)
    vm = plane_strain_vm(out_aux[:, :3])
    sy = hardening_sigma_y(out_aux[:, 3], he, hs)
    assert bool((vm <= sy * D2_FLOAT_TOL).all())
    # eval path is the exact decode (no round trip): exact floor, except
    # kinematic rows, which set_anchor clamps to GT aux (house rule).
    free = ptype != 2
    assert bool((out_aux[free, 3] >= aux[1][free, 3]).all())


def test_hinge_helper_values_and_gradients():
    """ADR-0064 knob 2: zero on admissible states, positive (and
    differentiable) on yield or irreversibility violations."""
    knots = (torch.tensor(HE), torch.tensor(HS))
    scale = torch.tensor(0.5)
    # admissible: tiny deviator, peeq above its anchor
    ok = torch.tensor([[10.0, -5.0, 3.0, 1.0, 0.2, 8900.0]])
    h = _fm_admissibility_hinge(ok, torch.tensor([0.5]), knots, scale)
    assert float(h) == 0.0
    # yield violation: vm far above sigma_y(0) = 199.3
    bad_vm = torch.tensor([[400.0, -400.0, 0.0, 0.0, 0.2, 8900.0]])
    bad_vm.requires_grad_(True)
    h = _fm_admissibility_hinge(bad_vm, torch.tensor([0.0]), knots, scale)
    assert float(h) > 0
    h.sum().backward()
    assert bool(torch.isfinite(bad_vm.grad).all())
    assert float(bad_vm.grad.abs().sum()) > 0
    # irreversibility violation: peeq below the fed anchor
    bad_peeq = torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.2, 8900.0]])
    h = _fm_admissibility_hinge(bad_peeq, torch.tensor([0.6]), knots, scale)
    np.testing.assert_allclose(float(h), (0.6 - 0.1) / 0.5, rtol=1e-6)
