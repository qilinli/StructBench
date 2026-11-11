"""The two-hop pushforward chain with the element state carried.

ADR-0063's chain for the two-site head decoders (ADR-0075 note 2026-11-11):
the mid-chain anchor is built from the first hop's own emitted state by the
function the rollout's hand-off calls, and carries no gradient.
"""

from __future__ import annotations

import pytest
import torch

from structbench.cli.train import _fm_two_site_chain
from structbench.models.common.two_site import (
    LAW_DECODERS,
    law_rows,
    project_elements_to_nodes,
    sigma_y_rows,
    von_mises4,
)
from structbench.models.transolver import TransolverSimulator

CHAIN_DECODERS = ("corner_mean", "admissible_coords", "projected_trial")
ADMISSIBLE_FORMS = ("admissible_coords", "projected_trial")

DIM, S, T, P, E = 2, 5, 8, 6, 2
REFERENCE = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [1.0, 1.0], [3.0, 1.0]]
)
CELLS = torch.tensor([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=torch.long)
PTYPE = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)  # two kinematic nodes
WEIGHTS = torch.tensor([1.0, 2.0])
T0, T1, T2 = 1, 4, 6  # the chain: anchor, mid-chain anchor, target
TIME_REF = T


def _law():
    return law_rows(
        torch.full((E,), 124_000.0),
        torch.full((E,), 0.34),
        torch.tensor([[0.0, 10.0]] * E),
        torch.tensor([[200.0, 10_200.0]] * E),
    )


def _sim(decoder: str) -> TransolverSimulator:
    torch.manual_seed(0)
    return TransolverSimulator(
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
        two_site_decoder=decoder,
        n_state=S,
        element_formulation="plane_strain",
    )


def _case(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    positions = REFERENCE[None] + torch.rand((T, P, DIM), generator=g) * 0.01
    state = torch.zeros((T, E, S))
    state[..., :4] = torch.randn((T, E, 4), generator=g) * 20.0
    state[..., 4] = torch.cumsum(torch.rand((T, E), generator=g) * 0.01, dim=0)
    aux = torch.stack(
        [project_elements_to_nodes(state[t], CELLS, WEIGHTS, P) for t in range(T)]
    )
    return positions, aux, state


class _Wild(torch.nn.Module):
    """Stands in for the network: random values of the right width that
    depend on a parameter, so gradients exist to be cut."""

    def __init__(self, width: int, scale: float, seed: int):
        super().__init__()
        self.width, self.scale = width, scale
        self.g = torch.Generator().manual_seed(seed)
        self.p = torch.nn.Parameter(torch.ones(()))

    def forward(self, feats, n_particles_per_example=None, t=None):
        noise = torch.randn((feats.shape[0], self.width), generator=self.g)
        return self.p * noise * self.scale + 0.0 * feats.sum()


def _chain(sim, decoder, *, accumulate=True, capture=None):
    """One chain on a batch of one case; ``capture`` collects what each
    clean query emitted."""
    positions, aux, state = _case()
    if capture is not None:
        forward = sim.forward_train_two_site_emitted

        def spy(*args, **kwargs):
            out = forward(*args, **kwargs)
            capture.append((out[4], kwargs.get("peeq_floor")))
            return out

        sim.forward_train_two_site_emitted = spy
    frames = (T1 - 1, T1, T2)
    out = _fm_two_site_chain(
        sim,
        torch.stack([positions[f] for f in frames], dim=1),
        torch.stack([aux[f] for f in frames], dim=1),
        torch.stack([state[f] for f in frames], dim=1),
        PTYPE,
        REFERENCE,
        torch.tensor([P]),
        torch.tensor([T0]),
        torch.tensor([T1]),
        torch.tensor([T2]),
        TIME_REF,
        accumulate=accumulate,
        cells=CELLS,
        weights=WEIGHTS,
        law=_law() if decoder in LAW_DECODERS else None,
        anchor_disp=positions[T0] - REFERENCE,
        anchor_vel=positions[T0] - positions[T0 - 1],
        anchor_aux=aux[T0],
        anchor_state=state[T0],
        is_kinematic=PTYPE == 1,
        loading_feature=None,
        anchor_time_feature=torch.full((P, 1), T0 / (TIME_REF - 1)),
        anchor_time=True,
    )
    return (positions, aux, state), out


@pytest.mark.parametrize("decoder", CHAIN_DECODERS)
def test_no_gradient_reaches_the_first_hop_through_the_anchor(decoder):
    """The mid-chain anchor is detached: none of its parts carries a graph,
    and the dirty query's loss has no gradient with respect to anything the
    first hop emitted. The clean queries' own losses still train the first
    hop."""
    sim = _sim(decoder)
    sim.train()
    emitted: list = []
    _case_, (pred, target, st_pred, st_target, anchor) = _chain(
        sim, decoder, capture=emitted
    )
    assert pred.shape == (P, 3, DIM + S) and st_pred.shape == (E, 3, S)
    for part in anchor:
        assert part is not None
        assert not part.requires_grad and part.grad_fn is None
    assert anchor.state.dtype == torch.float64

    # Three queries ran; the first two are the first hop.
    assert len(emitted) == 3
    first_hop = [t for out, _floor in emitted[:2] for t in out]
    assert all(t.requires_grad for t in first_hop)
    dirty = ((pred[:, 2] - target[:, 2]) ** 2).mean() + (
        (st_pred[:, 2] - st_target[:, 2]) ** 2
    ).mean()
    grads = torch.autograd.grad(dirty, first_hop, retain_graph=True, allow_unused=True)
    # Absent from the graph, or reached only through the stack the losses
    # are sliced from, which passes an exact zero.
    assert all(g is None or not bool(g.any()) for g in grads)
    # The dirty query trains the network through its own forward, and the
    # clean queries reach what the first hop emitted.
    params = [p for p in sim.parameters() if p.requires_grad]
    assert any(
        g is not None and bool(g.abs().sum() > 0)
        for g in torch.autograd.grad(
            dirty, params, retain_graph=True, allow_unused=True
        )
    )
    clean = ((st_pred[:, 1] - st_target[:, 1]) ** 2).mean()
    assert any(
        g is not None and bool(g.any())
        for g in torch.autograd.grad(clean, first_hop, allow_unused=True)
    )


@pytest.mark.parametrize("decoder", CHAIN_DECODERS)
def test_the_training_anchor_is_the_rollout_anchor_for_the_same_predictions(decoder):
    """Handed the state the first hop emitted at ``t1 - 1`` and ``t1``, the
    rollout's hand-off (``set_anchor``) caches exactly the anchor the
    training step built: displacement, finite-difference velocity, nodal aux
    with the stored kinematic rows, and the element state in double."""
    sim = _sim(decoder)
    sim.train()
    emitted: list = []
    (positions, aux, state), out = _chain(sim, decoder, capture=emitted)
    anchor = out[4]
    (pos_a1, _aux_a1, _state_a1), _ = emitted[0]
    (pos_a2, aux_a2, state_a2), _ = emitted[1]

    sim.bind_case(
        CELLS,
        REFERENCE,
        PTYPE,
        positions,
        gt_aux=aux,
        gt_state=state,
        law=_law() if decoder in LAW_DECODERS else None,
    )
    sim.set_anchor(
        T1,
        pos_a1.detach(),
        pos_a2.detach(),
        aux_a2.detach(),
        anchor_t_norm=T1 / (TIME_REF - 1),
        state=state_a2.detach(),
    )
    assert torch.equal(sim._anchor_disp, anchor.disp)
    assert torch.equal(sim._anchor_vel, anchor.vel)
    assert torch.equal(sim._anchor_aux, anchor.aux)
    assert torch.equal(sim._anchor_state, anchor.state)
    # What it holds: the emitted state, with the prescribed rows stored.
    kin = PTYPE == 1
    assert torch.equal(anchor.aux[kin], aux[T1][kin])
    assert torch.equal(anchor.aux[~kin], aux_a2.detach()[~kin])
    assert torch.equal(anchor.disp[kin], (positions[T1] - REFERENCE)[kin])
    assert torch.equal(anchor.vel[kin], (positions[T1] - positions[T1 - 1])[kin])
    assert torch.equal(anchor.state, state_a2.detach().double())


@pytest.mark.parametrize("decoder", CHAIN_DECODERS)
def test_normalizers_accumulate_on_the_clean_queries_only(decoder):
    """Three queries, two accumulations: the dirty query never warms a
    normalizer (ADR-0063), nodal, target or element state."""
    sim = _sim(decoder)
    sim.train()
    _chain(sim, decoder, accumulate=True)
    for normalizer in (
        sim._node_normalizer,
        sim._target_normalizer,
        sim._state_normalizer,
    ):
        assert int(normalizer._n_accumulations) == 2
    _chain(sim, decoder, accumulate=False)
    assert int(sim._state_normalizer._n_accumulations) == 2


@pytest.mark.parametrize("decoder", ADMISSIBLE_FORMS)
@pytest.mark.parametrize("scale", [1.0, 40.0])
def test_the_mid_chain_anchor_is_admissible_and_monotone(decoder, scale):
    """Whatever the network emits, the state the second hop is anchored on
    has a plastic strain at or above the stored anchor's and at or above
    what the chain emitted for the frame before it (the second clean query
    is floored by the first), and sits on or inside the yield surface at
    that plastic strain."""
    sim = _sim(decoder)
    sim._net = _Wild(DIM + S, scale, seed=3)
    sim.train()
    emitted: list = []
    (_positions, _aux, state), out = _chain(sim, decoder, capture=emitted)
    anchor_state = out[4].state
    (_p1, _a1, state_a1), floor_1 = emitted[0]
    (_p2, _a2, _state_a2), floor_2 = emitted[1]
    assert floor_1 is None
    assert torch.equal(floor_2, state_a1[:, 4].detach()) and not floor_2.requires_grad
    assert bool((anchor_state[:, 4] >= state_a1[:, 4].detach()).all())
    assert bool((anchor_state[:, 4] >= state[T0][:, 4].double()).all())
    law = _law()
    sy, _ = sigma_y_rows(
        anchor_state[:, 4], law.knots_peeq.double(), law.knots_sy.double()
    )
    assert bool((von_mises4(anchor_state[:, :4]) <= sy * (1 + 1e-12)).all())
