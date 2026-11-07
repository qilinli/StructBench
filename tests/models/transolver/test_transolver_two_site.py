"""ADR-0075 two-site decoder on the anchored flow map: constructor guards,
the binding and anchor contract, the training forward's shapes and
gradients, and the by-construction admissibility of the return-map arm."""

import pytest
import torch

from structbench.models.common.two_site import (
    law_rows,
    project_elements_to_nodes,
    sigma_y_rows,
    von_mises4,
)
from structbench.models.transolver import TransolverSimulator

DIM = 2
S = 5  # (s11, s22, s33, s12, peeq)
T = 6
P = 6
E = 2
REFERENCE = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [1.0, 1.0], [3.0, 1.0]]
)
CELLS = torch.tensor([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=torch.long)
PTYPE = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)  # top-right row kinematic


def _law(n: int = E):
    return law_rows(
        torch.full((n,), 124_000.0),
        torch.full((n,), 0.34),
        torch.tensor([[0.0, 10.0]] * n),
        torch.tensor([[200.0, 10_200.0]] * n),
    )


def _sim(**over):
    kwargs = dict(
        dim=DIM,
        hidden_dim=16,
        n_layers=1,
        n_heads=2,
        slice_num=4,
        node_type_size=2,
        kinematic_types=(1,),
        scripted_types=(1,),
        n_aux=S,
        time_conditioned=True,
        aux_input=True,
        flow_map=True,
        two_site_decoder="return_map",
        n_state=S,
        element_formulation="plane_strain",
    )
    kwargs.update(over)
    return TransolverSimulator(**kwargs)


def _case(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    positions = REFERENCE[None] + torch.rand((T, P, DIM), generator=g) * 0.01
    state = torch.zeros((T, E, S))
    state[..., 4] = torch.rand((T, E), generator=g) * 0.01
    weights = torch.tensor([1.0, 2.0])
    aux = torch.stack(
        [project_elements_to_nodes(state[t], CELLS, weights, P) for t in range(T)]
    )
    return positions, aux, state


def _bind(sim, law=None):
    positions, aux, state = _case()
    sim.bind_case(
        CELLS, REFERENCE, PTYPE, positions, gt_aux=aux, gt_state=state, law=law
    )
    return positions, aux, state


def test_ctor_guards():
    with pytest.raises(ValueError, match="requires flow_map"):
        _sim(flow_map=False, aux_input=False)
    with pytest.raises(ValueError, match="n_state == n_aux"):
        _sim(n_state=4)
    with pytest.raises(ValueError, match="5-channel"):
        _sim(n_aux=4, n_state=4)
    with pytest.raises(ValueError, match="element_formulation"):
        _sim(element_formulation="shell")
    with pytest.raises(ValueError, match="two_site_decoder must be one of"):
        _sim(two_site_decoder="magic")
    # The off path registers no state normalizer: checkpoints stay untouched.
    off = _sim(two_site_decoder="off", n_state=0, element_formulation=None)
    assert not any(k.startswith("_state_normalizer") for k in off.state_dict())
    assert any(k.startswith("_state_normalizer") for k in _sim().state_dict())
    assert _sim().two_site == "return_map"


def test_bind_and_anchor_need_the_state_and_the_law():
    sim = _sim()
    positions, aux, state = _case()
    with pytest.raises(ValueError, match="gt_state"):
        sim.bind_case(CELLS, REFERENCE, PTYPE, positions, gt_aux=aux)
    with pytest.raises(ValueError, match="law rows"):
        sim.bind_case(CELLS, REFERENCE, PTYPE, positions, gt_aux=aux, gt_state=state)
    _bind(sim, _law())
    with pytest.raises(ValueError, match="element state"):
        sim.set_anchor(1, positions[0], positions[1], aux[1], anchor_t_norm=0.2)
    sim.set_anchor(
        1, positions[0], positions[1], aux[1], anchor_t_norm=0.2, state=state[1]
    )
    assert torch.equal(sim._anchor_state, state[1])
    sim.reset_rollout()
    assert sim._anchor_state is None
    # corner_mean needs the state but no law.
    sim2 = _sim(two_site_decoder="corner_mean")
    _bind(sim2)


def test_forward_train_two_site_shapes_and_gradients():
    for decoder in ("return_map", "corner_mean"):
        sim = _sim(two_site_decoder=decoder)
        positions, aux, state = _case()
        weights = torch.tensor([1.0, 2.0])
        pred, target, state_pred, state_target = sim.forward_train_two_site(
            positions[3],
            aux[3],
            state[3],
            PTYPE,
            REFERENCE,
            torch.tensor([P]),
            torch.tensor([0.5]),
            accumulate=True,
            cells=CELLS,
            weights=weights,
            law=_law() if decoder == "return_map" else None,
            anchor_disp=positions[1] - REFERENCE,
            anchor_vel=positions[1] - positions[0],
            anchor_aux=aux[1],
            anchor_state=state[1],
            anchor_time_feature=torch.full((P, 1), 0.2),
        )
        assert pred.shape == (P, DIM + S) and target.shape == (P, DIM + S)
        assert state_pred.shape == (E, S) and state_target.shape == (E, S)
        loss = ((pred - target) ** 2).mean() + ((state_pred - state_target) ** 2).mean()
        loss.backward()
        grads = [p.grad for p in sim.parameters() if p.grad is not None]
        assert grads and any(float(g.abs().sum()) > 0 for g in grads)


def test_return_map_arm_emits_admissible_state_and_its_projection():
    sim = _sim()
    positions, aux, state = _bind(sim, _law())
    sim.set_anchor(
        1, positions[0], positions[1], aux[1], anchor_t_norm=0.2, state=state[1]
    )
    sim.eval()
    with torch.no_grad():
        pos, aux_out, st = sim.predict_two_site_at(3, 0.4)
    assert pos.shape == (P, DIM) and aux_out.shape == (P, S) and st.shape == (E, S)
    law = _law()
    sy, _ = sigma_y_rows(st[:, 4], law.knots_peeq, law.knots_sy)
    assert bool((von_mises4(st[:, :4]) <= sy * (1 + 1e-5)).all())
    assert bool((st[:, 4] >= state[1][:, 4]).all())
    # The emitted nodal aux is the area-weighted projection of the state.
    expected = project_elements_to_nodes(st, CELLS, torch.tensor([1.0, 2.0]), P)
    assert torch.allclose(aux_out, expected, atol=1e-5)
    # Kinematic rows are overridden with ground truth, as before.
    assert torch.allclose(pos[4:], positions[3][4:])


def test_corner_mean_arm_emits_the_corner_mean_of_the_network_aux():
    sim = _sim(two_site_decoder="corner_mean")
    positions, aux, state = _bind(sim)
    sim.set_anchor(
        1, positions[0], positions[1], aux[1], anchor_t_norm=0.2, state=state[1]
    )
    sim.eval()
    with torch.no_grad():
        _pos, aux_out, st = sim.predict_two_site_at(3, 0.4)
    expected = project_elements_to_nodes(st, CELLS, torch.tensor([1.0, 2.0]), P)
    assert torch.allclose(aux_out, expected, atol=1e-5)
    assert st.shape == (E, S)
