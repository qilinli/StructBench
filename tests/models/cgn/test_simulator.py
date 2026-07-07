import torch

from structbench.models.cgn.simulator import LearnedSimulator


def _stats(dim=2, aux_mean=0.0, aux_std=1.0):
    z, o = torch.zeros(dim), torch.ones(dim)
    return {
        "velocity": {"mean": z, "std": o},
        "acceleration": {"mean": z, "std": o},
        "aux": {
            "mean": torch.tensor([aux_mean]),
            "std": torch.tensor([aux_std]),
        },
    }


def _sim(n_aux=1, boundary_feature_fn=None, input_frames=3, aux_mean=0.0, aux_std=1.0):
    nnode_in = (input_frames - 1) * 2  # velocities only; +embedding handled internally
    return LearnedSimulator(
        particle_dimensions=2,
        nnode_in=nnode_in,
        nedge_in=3,
        latent_dim=16,
        nmessage_passing_steps=2,
        nmlp_layers=1,
        mlp_hidden_dim=16,
        connectivity_radius=5.0,
        normalization_stats=_stats(aux_mean=aux_mean, aux_std=aux_std),
        nparticle_types=1,
        particle_type_embedding_size=4,
        n_aux=n_aux,
        boundary_feature_fn=boundary_feature_fn,
        device="cpu",
    )


def test_predict_positions_shapes():
    sim = _sim(n_aux=1)
    P, input_frames = 4, 3
    pos_seq = torch.randn(P, input_frames, 2)
    npp = torch.tensor([P])
    ptype = torch.zeros(P, dtype=torch.long)
    next_pos, aux = sim.predict_positions(pos_seq, npp, ptype)
    assert next_pos.shape == (P, 2)
    assert aux.shape == (P, 1)


def test_predict_positions_denormalizes_aux():
    # predict_positions must return aux in raw (MPa) units, i.e. the normalized
    # decoder output (as returned by predict_accelerations) scaled by std and
    # shifted by mean. With identical inputs (zero training noise) the two
    # forward passes produce the same decoder output, so we can assert the
    # exact denormalization transform.
    mean, std = 5.0, 2.0
    sim = _sim(n_aux=1, aux_mean=mean, aux_std=std)
    sim.eval()
    P, input_frames = 4, 3
    pos_seq = torch.randn(P, input_frames, 2)
    npp = torch.tensor([P])
    ptype = torch.zeros(P, dtype=torch.long)

    _, _, pred_aux_norm = sim.predict_accelerations(
        next_positions=torch.zeros(P, 2),
        position_sequence_noise=torch.zeros(P, input_frames, 2),
        position_sequence=pos_seq,
        nparticles_per_example=npp,
        particle_types=ptype,
    )
    _, aux = sim.predict_positions(pos_seq, npp, ptype)

    expected = pred_aux_norm * std + mean
    torch.testing.assert_close(aux, expected)


def test_boundary_feature_fn_changes_node_input_width():
    # With a boundary fn adding 1 feature, the encoder must accept nnode_in+1.
    def wall(pos):  # (P, dim) -> (P, 1)
        return pos[:, 0:1].clamp(min=0.0, max=5.0)

    sim = LearnedSimulator(
        particle_dimensions=2,
        nnode_in=(3 - 1) * 2 + 1,
        nedge_in=3,
        latent_dim=16,
        nmessage_passing_steps=1,
        nmlp_layers=1,
        mlp_hidden_dim=16,
        connectivity_radius=5.0,
        normalization_stats=_stats(),
        nparticle_types=1,
        particle_type_embedding_size=4,
        n_aux=1,
        boundary_feature_fn=wall,
        device="cpu",
    )
    out, _ = sim.predict_positions(
        torch.randn(4, 3, 2), torch.tensor([4]), torch.zeros(4, dtype=torch.long)
    )
    assert out.shape == (4, 2)
