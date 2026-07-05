"""Grouped run-config loading (ADR-0032): strict validation and dispatch."""

import pytest
import torch

from structbench.cli.train import GNSConfig, TrainConfig, build_simulator
from structbench.config import ConfigError, load_run_config

#: A complete, valid grouped config; tests below perturb it.
VALID = """\
[run]
benchmark = "taylor_impact_2d"
seed = 7

[model]
family = "gns"
window = 11
connectivity_radius = 1.5
hidden_dim = 64
message_passing_steps = 5
nmlp_layers = 1
particle_type_embedding_size = 9
noise_std = 0.02
dim = 2
max_neighbors = 48

[train]
batch_size = 8
lr_init = 1e-4
lr_decay = 0.1
lr_decay_steps = 30000
training_steps = 100
val_every = 50
w_pos = 1.0
w_aux = 1.0
"""


def _write(tmp_path, text):
    p = tmp_path / "run.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_run_config_happy_path(tmp_path):
    rc = load_run_config(_write(tmp_path, VALID))
    assert rc.family == "gns"
    assert isinstance(rc.model, GNSConfig)
    assert isinstance(rc.train, TrainConfig)
    assert rc.train.benchmark == "taylor_impact_2d"
    assert rc.train.seed == 7  # [run].seed lands on TrainConfig
    assert rc.train.batch_size == 8
    assert rc.model.window == 11
    assert rc.protocol_override is None


def test_load_run_config_rejects_flat_configs(tmp_path):
    p = _write(tmp_path, 'benchmark = "taylor_impact_2d"\nbatch_size = 4\n')
    with pytest.raises(ConfigError, match="flat configs are no longer supported"):
        load_run_config(p)


def test_load_run_config_rejects_unknown_key(tmp_path):
    # The classic silent-typo footgun: noise_st instead of noise_std.
    bad = VALID.replace("noise_std = 0.02", "noise_st = 0.02")
    with pytest.raises(ConfigError, match="unknown keys: noise_st"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_missing_key(tmp_path):
    bad = VALID.replace("lr_init = 1e-4\n", "")
    with pytest.raises(ConfigError, match="missing keys: lr_init"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_benchmark_in_train(tmp_path):
    bad = VALID.replace("[train]\n", '[train]\nbenchmark = "taylor_impact_2d"\n')
    with pytest.raises(ConfigError, match="belong in \\[run\\]"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_unknown_family(tmp_path):
    bad = VALID.replace('family = "gns"', 'family = "transformer"')
    with pytest.raises(ConfigError, match="unknown family"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_unknown_section(tmp_path):
    bad = VALID + "\n[extras]\nfoo = 1\n"
    with pytest.raises(ConfigError, match="unknown sections: extras"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_protocol_override(tmp_path):
    rc = load_run_config(_write(tmp_path, VALID + "\n[protocol]\ninit_frames = 11\n"))
    assert rc.protocol_override is not None
    assert rc.protocol_override.init_frames == 11
    with pytest.raises(ConfigError, match="init_frames must be an int >= 2"):
        load_run_config(_write(tmp_path, VALID + "\n[protocol]\ninit_frames = 1\n"))


def _stats_dict():
    return {
        "velocity": {"mean": torch.zeros(2), "std": torch.ones(2)},
        "acceleration": {"mean": torch.zeros(2), "std": torch.ones(2)},
        "aux": {"mean": torch.tensor([5.0]), "std": torch.tensor([2.0])},
    }


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
