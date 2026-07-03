import torch

from structbench.cli.train import GNSConfig, TrainConfig, build_simulator


def test_train_config_benchmark_defaults_to_taylor(tmp_path):
    cfg = TrainConfig()
    assert cfg.benchmark == "taylor_impact_2d"


def test_train_config_benchmark_from_toml(tmp_path):
    toml = tmp_path / "cfg.toml"
    toml.write_text('benchmark = "taylor_impact_2d"\nbatch_size = 4\n')
    cfg = TrainConfig.from_toml(toml)
    assert cfg.benchmark == "taylor_impact_2d"
    assert cfg.batch_size == 4


def _stats_dict():
    return {
        "velocity": {"mean": torch.zeros(2), "std": torch.ones(2)},
        "acceleration": {"mean": torch.zeros(2), "std": torch.ones(2)},
        "aux": {"mean": torch.tensor([5.0]), "std": torch.tensor([2.0])},
    }


def test_train_config_from_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("batch_size = 8\nlr_init = 0.0005\n", encoding="utf-8")
    cfg = TrainConfig.from_toml(p)
    assert cfg.batch_size == 8 and cfg.lr_init == 0.0005
    assert cfg.training_steps == 100000  # default preserved


def test_build_simulator_node_input_width():
    gns = GNSConfig()
    sim = build_simulator(
        _stats_dict(),
        gns,
        n_particle_types=2,
        boundary_feature_fn=lambda p: p[:, 0:1],
        device="cpu",
    )
    # (window-1)*dim + 1 boundary + embedding(9) = 10*2 + 1 + 9 = 30
    out, aux = sim.predict_positions(
        torch.randn(5, gns.window, 2),
        torch.tensor([5]),
        torch.zeros(5, dtype=torch.long),
    )
    assert out.shape == (5, 2) and aux.shape == (5, 1)


def test_build_simulator_includes_aux_stats():
    gns = GNSConfig()
    sim = build_simulator(
        _stats_dict(),
        gns,
        n_particle_types=2,
        boundary_feature_fn=lambda p: p[:, 0:1],
        device="cpu",
    )
    aux_stats = sim._normalization_stats["aux"]
    # Aux carries no training-noise inflation, so mean/std pass through verbatim.
    torch.testing.assert_close(aux_stats["mean"], torch.tensor([5.0]))
    torch.testing.assert_close(aux_stats["std"], torch.tensor([2.0]))
