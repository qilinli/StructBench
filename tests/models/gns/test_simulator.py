import torch

from structbench.models.gns.simulator import LearnedSimulator


def _stats(dim=2):
    z, o = torch.zeros(dim), torch.ones(dim)
    return {"velocity": {"mean": z, "std": o},
            "acceleration": {"mean": z, "std": o}}


def _sim(n_aux=1, boundary_feature_fn=None, window=3):
    nnode_in = (window - 1) * 2  # velocities only; +embedding handled internally
    return LearnedSimulator(
        particle_dimensions=2, nnode_in=nnode_in, nedge_in=3, latent_dim=16,
        nmessage_passing_steps=2, nmlp_layers=1, mlp_hidden_dim=16,
        connectivity_radius=5.0, normalization_stats=_stats(),
        nparticle_types=1, particle_type_embedding_size=4,
        n_aux=n_aux, boundary_feature_fn=boundary_feature_fn, device="cpu",
    )


def test_predict_positions_shapes():
    sim = _sim(n_aux=1)
    P, window = 4, 3
    pos_seq = torch.randn(P, window, 2)
    npp = torch.tensor([P])
    ptype = torch.zeros(P, dtype=torch.long)
    next_pos, aux = sim.predict_positions(pos_seq, npp, ptype)
    assert next_pos.shape == (P, 2)
    assert aux.shape == (P, 1)


def test_boundary_feature_fn_changes_node_input_width():
    # With a boundary fn adding 1 feature, the encoder must accept nnode_in+1.
    def wall(pos):  # (P, dim) -> (P, 1)
        return pos[:, 0:1].clamp(min=0.0, max=5.0)
    sim = LearnedSimulator(
        particle_dimensions=2, nnode_in=(3 - 1) * 2 + 1, nedge_in=3, latent_dim=16,
        nmessage_passing_steps=1, nmlp_layers=1, mlp_hidden_dim=16,
        connectivity_radius=5.0, normalization_stats=_stats(),
        nparticle_types=1, particle_type_embedding_size=4,
        n_aux=1, boundary_feature_fn=wall, device="cpu",
    )
    out, _ = sim.predict_positions(torch.randn(4, 3, 2), torch.tensor([4]),
                                   torch.zeros(4, dtype=torch.long))
    assert out.shape == (4, 2)
