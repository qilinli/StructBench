"""Grouped run-config loading (ADR-0032): strict validation and dispatch."""

import pytest
import torch

from structbench.cli.train import CGNConfig, TrainConfig, build_simulator
from structbench.config import ConfigError, load_run_config

#: A complete, valid grouped config; tests below perturb it.
VALID = """\
[run]
benchmark = "taylor_impact_2d"
seed = 7

[model]
family = "cgn"
input_frames = 6
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
    assert rc.family == "cgn"
    assert isinstance(rc.model, CGNConfig)
    assert isinstance(rc.train, TrainConfig)
    assert rc.train.benchmark == "taylor_impact_2d"
    assert rc.train.seed == 7  # [run].seed lands on TrainConfig
    assert rc.train.batch_size == 8
    assert rc.train.lr_decay_steps == 38  # derived: round(100 * 30000/80000)
    assert rc.model.input_frames == 6


def test_legacy_gns_family_alias_still_resolves(tmp_path):
    # Pre-ADR-0034 run configs and run-dir records say family = "gns";
    # the alias keeps them loadable and re-evaluable.
    legacy = VALID.replace('family = "cgn"', 'family = "gns"')
    rc = load_run_config(_write(tmp_path, legacy))
    assert rc.family == "gns"
    assert isinstance(rc.model, CGNConfig)


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
    bad = VALID.replace('family = "cgn"', 'family = "transformer"')
    with pytest.raises(ConfigError, match="unknown family"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_unknown_section(tmp_path):
    bad = VALID + "\n[extras]\nfoo = 1\n"
    with pytest.raises(ConfigError, match="unknown sections: extras"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_input_frames_off_card(tmp_path):
    # ADR-0035: the model observes exactly the frames it inputs, so a run's
    # input_frames must equal its benchmark's protocol (taylor card = 6).
    bad = VALID.replace("input_frames = 6", "input_frames = 11")
    with pytest.raises(ConfigError, match="must equal benchmark"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_protocol_section(tmp_path):
    # The [protocol] research override of ADR-0032 §4 was removed by ADR-0035;
    # a leftover section is now an unknown-section error.
    bad = VALID + "\n[protocol]\ninput_frames = 6\n"
    with pytest.raises(ConfigError, match="unknown sections: protocol"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_derives_lr_decay_steps(tmp_path):
    # lr_decay_steps is not in [train]; it is derived from training_steps to hold
    # the reference anneal depth. 40000 * 30000/80000 = 15000 (the value the
    # 2026-07-06 fleet should have used instead of the inherited 30000).
    cfg = VALID.replace("training_steps = 100", "training_steps = 40000")
    rc = load_run_config(_write(tmp_path, cfg))
    assert rc.train.lr_decay_steps == 15000


def test_derived_lr_decay_steps_reproduces_reference(tmp_path):
    # The 80k Taylor budget must reproduce the validated ADR-0028 lr_decay_steps
    # (30000) exactly, so the flagship recipe's schedule is byte-for-byte unchanged.
    cfg = VALID.replace("training_steps = 100", "training_steps = 80000")
    rc = load_run_config(_write(tmp_path, cfg))
    assert rc.train.lr_decay_steps == 30000


def test_load_run_config_rejects_explicit_lr_decay_steps(tmp_path):
    # Setting it by hand is exactly the footgun this derivation removes; reject it
    # with a clear message rather than silently honoring a mis-scaled value.
    bad = VALID.replace(
        "training_steps = 100", "lr_decay_steps = 30000\ntraining_steps = 100"
    )
    with pytest.raises(ConfigError, match="lr_decay_steps is derived"):
        load_run_config(_write(tmp_path, bad))


def _stats_dict():
    return {
        "velocity": {"mean": torch.zeros(2), "std": torch.ones(2)},
        "acceleration": {"mean": torch.zeros(2), "std": torch.ones(2)},
        "aux": {"mean": torch.tensor([5.0]), "std": torch.tensor([2.0])},
    }


def test_build_simulator_node_input_width():
    cgn = CGNConfig()
    sim = build_simulator(
        _stats_dict(),
        cgn,
        n_particle_types=2,
        boundary_feature_fn=lambda p: p[:, 0:1],
        device="cpu",
    )
    # (input_frames-1)*dim + 1 boundary + embedding(9) = 5*2 + 1 + 9 = 20
    out, aux = sim.predict_positions(
        torch.randn(5, cgn.input_frames, 2),
        torch.tensor([5]),
        torch.zeros(5, dtype=torch.long),
    )
    assert out.shape == (5, 2) and aux.shape == (5, 1)


def test_build_simulator_includes_aux_stats():
    cgn = CGNConfig()
    sim = build_simulator(
        _stats_dict(),
        cgn,
        n_particle_types=2,
        boundary_feature_fn=lambda p: p[:, 0:1],
        device="cpu",
    )
    aux_stats = sim._normalization_stats["aux"]
    # Aux carries no training-noise inflation, so mean/std pass through verbatim.
    torch.testing.assert_close(aux_stats["mean"], torch.tensor([5.0]))
    torch.testing.assert_close(aux_stats["std"], torch.tensor([2.0]))
