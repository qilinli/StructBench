"""ADR-0062 anchored-flow-map simulator surface: ctor guards, anchor cache,
feature widening, leakage tripwire, and the Markov feature drop."""

import pytest
import torch

from structbench.models.transolver import TransolverSimulator

DIM = 2
C = 2
T = 6
P = 3


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
    )
    kwargs.update(over)
    return TransolverSimulator(**kwargs)


def _bind(sim, seed: int = 0):
    """Bind a tiny deterministic case; returns (positions, aux, reference).

    The reference frame is DISTINCT from frame-0 positions so a
    displacement/velocity swap bug cannot cancel out in the anchor asserts.
    """
    g = torch.Generator().manual_seed(seed)
    positions = torch.rand((T, P, DIM), generator=g)
    aux = torch.rand((T, P, C), generator=g)
    reference = torch.rand((P, DIM), generator=g)
    ptype = torch.tensor([0, 0, 2], dtype=torch.long)  # last row kinematic
    cells = torch.zeros((1, 3), dtype=torch.long)
    sim.bind_case(cells, reference, ptype, positions, gt_aux=aux)
    return positions, aux, reference


def test_ctor_guards():
    with pytest.raises(ValueError, match="flow_map requires time_conditioned"):
        _sim(time_conditioned=False, aux_input=False, flow_map=True)
    with pytest.raises(ValueError, match="flow_map requires aux_input"):
        _sim(aux_input=False)
    # The ADR-0060 rejection survives for plain TC (no flow_map).
    with pytest.raises(ValueError, match="aux_input is incompatible"):
        _sim(flow_map=False)
    # The narrowed combination constructs.
    _sim()


def test_node_in_widens_by_anchor_block():
    def width(sim):
        return int(sim._node_normalizer._sum.numel())

    base = width(_sim(flow_map=False, aux_input=False))
    full = width(_sim())
    markov = width(_sim(flow_map_anchor_time=False))
    # anchor block = disp (DIM) + FD velocity (DIM) + aux (C) + time scalar.
    assert full == base + 2 * DIM + C + 1
    assert markov == base + 2 * DIM + C


def test_set_anchor_requires_flow_map_and_bind():
    sim = _sim(flow_map=False, aux_input=False)
    with pytest.raises(RuntimeError, match="set_anchor.*flow_map"):
        sim.set_anchor(1, torch.zeros(P, DIM), torch.zeros(P, DIM), torch.zeros(P, C))
    sim = _sim()
    with pytest.raises(RuntimeError, match="before bind_case"):
        sim.set_anchor(
            1,
            torch.zeros(P, DIM),
            torch.zeros(P, DIM),
            torch.zeros(P, C),
            anchor_t_norm=0.2,
        )


def test_set_anchor_requires_t_norm_iff_anchor_time():
    sim = _sim()
    positions, aux, reference = _bind(sim)
    with pytest.raises(ValueError, match="anchor_t_norm"):
        sim.set_anchor(1, positions[0], positions[1], aux[1])
    sim2 = _sim(flow_map_anchor_time=False)
    positions, aux, _reference = _bind(sim2)
    sim2.set_anchor(1, positions[0], positions[1], aux[1])  # no t_norm needed


def test_set_anchor_clamps_kinematic_aux_rows():
    sim = _sim()
    positions, aux, reference = _bind(sim)
    fed = aux[1] + 100.0  # "predicted" aux, wrong everywhere
    sim.set_anchor(1, positions[0], positions[1], fed, anchor_t_norm=0.2)
    # Kinematic row (index 2) clamped to GT; free rows keep the fed values.
    assert torch.equal(sim._anchor_aux[2], aux[1][2])
    assert torch.equal(sim._anchor_aux[:2], fed[:2])
    # Kinematics stored as displacement-from-rest and FD velocity —
    # reference != positions[0], so a disp/vel swap cannot pass.
    assert torch.allclose(sim._anchor_disp, positions[1] - reference)
    assert torch.allclose(sim._anchor_vel, positions[1] - positions[0])


def test_predict_requires_anchor_and_clears_on_reset():
    sim = _sim()
    positions, aux, reference = _bind(sim)
    with pytest.raises(RuntimeError, match="no anchor is set"):
        sim.predict_state_at(2, 0.2)
    sim.set_anchor(1, positions[0], positions[1], aux[1], anchor_t_norm=0.2)
    sim.eval()
    with torch.no_grad():
        sim.predict_state_at(2, 0.2)
    sim.reset_rollout()
    with pytest.raises(RuntimeError, match="no anchor is set"):
        sim.predict_state_at(2, 0.2)
    # Re-binding also clears the anchor (stale-anchor tripwire).
    sim.set_anchor(1, positions[0], positions[1], aux[1], anchor_t_norm=0.2)
    _bind(sim, seed=1)
    with pytest.raises(RuntimeError, match="no anchor is set"):
        sim.predict_state_at(2, 0.2)


def test_query_ignores_future_free_rows():
    """Leakage tripwire (ADR-0062): with the anchor fixed, a query must not
    depend on GT free-row positions at any frame — only the kinematic rows
    (prescribed BC + override) may enter."""
    sim = _sim()
    positions, aux, reference = _bind(sim)
    sim.set_anchor(1, positions[0], positions[1], aux[1], anchor_t_norm=0.2)
    sim.eval()
    with torch.no_grad():
        pos_a, aux_a = sim.predict_state_at(4, 0.5)
    # Perturb FREE rows of the bound GT everywhere after the anchor.
    perturbed = positions.clone()
    perturbed[2:, :2] += 7.0
    ptype = torch.tensor([0, 0, 2], dtype=torch.long)
    sim.bind_case(
        torch.zeros((1, 3), dtype=torch.long), reference, ptype, perturbed, gt_aux=aux
    )
    sim.set_anchor(1, positions[0], positions[1], aux[1], anchor_t_norm=0.2)
    with torch.no_grad():
        pos_b, aux_b = sim.predict_state_at(4, 0.5)
    assert torch.equal(pos_a, pos_b)
    assert torch.equal(aux_a, aux_b)


def test_forward_train_requires_anchor_parts():
    sim = _sim()
    ptype = torch.tensor([0, 0, 2], dtype=torch.long)
    ref = torch.rand(P, DIM)
    with pytest.raises(ValueError, match="anchor block"):
        sim.forward_train_tc(
            torch.rand(P, DIM),
            torch.rand(P, C),
            ptype,
            ref,
            torch.tensor([P]),
            torch.tensor([0.3]),
            accumulate=False,
        )


def test_markov_variant_trains_and_queries_without_time_inputs():
    sim = _sim(flow_map_anchor_time=False)
    positions, aux, reference = _bind(sim)
    ptype = torch.tensor([0, 0, 2], dtype=torch.long)
    pred, target = sim.forward_train_tc(
        positions[3],
        aux[3],
        ptype,
        positions[0],
        torch.tensor([P]),
        torch.tensor([0.4]),
        accumulate=True,
        anchor_disp=positions[1] - positions[0],
        anchor_vel=positions[1] - positions[0],
        anchor_aux=aux[1],
    )
    assert pred.shape == target.shape == (P, DIM + C)
    sim.set_anchor(1, positions[0], positions[1], aux[1])
    sim.eval()
    with torch.no_grad():
        pos, out_aux = sim.predict_state_at(3, 0.4)
    assert pos.shape == (P, DIM)
    assert out_aux.shape == (P, C)


def test_set_anchor_requires_gt_aux_for_kinematic_clamp():
    """The ADR-0060 house clamp is mandatory: with kinematic rows bound but
    no gt_aux, set_anchor must fail loud, not silently anchor on untargeted
    decoder output."""
    sim = _sim()
    g = torch.Generator().manual_seed(0)
    positions = torch.rand((T, P, DIM), generator=g)
    aux = torch.rand((T, P, C), generator=g)
    ptype = torch.tensor([0, 0, 2], dtype=torch.long)
    sim.bind_case(
        torch.zeros((1, 3), dtype=torch.long),
        torch.rand((P, DIM), generator=g),
        ptype,
        positions,
    )
    with pytest.raises(RuntimeError, match="gt_aux"):
        sim.set_anchor(1, positions[0], positions[1], aux[1], anchor_t_norm=0.2)


def test_train_output_state_splits_raw_blocks():
    """ADR-0063 helper: inverse-normalized (displacement, aux) split matches
    the ADR-0061 aux helper and the target-normalizer inverse."""
    sim = _sim()
    g = torch.Generator().manual_seed(5)
    pred = torch.rand((P, DIM + C), generator=g)
    disp, aux = sim.train_output_state(pred)
    torch.testing.assert_close(aux, sim.train_output_aux(pred))
    raw = sim._target_normalizer.inverse(pred)
    torch.testing.assert_close(disp, raw[:, :DIM])
    assert disp.shape == (P, DIM) and aux.shape == (P, C)
