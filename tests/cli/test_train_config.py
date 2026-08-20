"""Grouped run-config loading (ADR-0032): strict validation and dispatch."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from structbench.cli.train import CGNConfig, TrainConfig, build_simulator
from structbench.config import (
    ConfigError,
    GeoFlareConfig,
    MGNConfig,
    TransolverConfig,
    load_run_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

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
aux_transform = "none"
aux_transform_scale = 0.01

[train]
batch_size = 8
lr_init = 1e-4
lr_decay = 0.1
training_steps = 100
val_every = 50
w_pos = 1.0
w_aux = 1.0
aux_tail_weight = 0.0
train_frames = 0
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
    assert rc.train.lr_decay_steps == 40  # derived: round(100 * 40000/100000)
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
    # the reference anneal depth. 40000 * 40000/100000 = 16000.
    cfg = VALID.replace("training_steps = 100", "training_steps = 40000")
    rc = load_run_config(_write(tmp_path, cfg))
    assert rc.train.lr_decay_steps == 16000


def test_derived_lr_decay_steps_reproduces_reference(tmp_path):
    # The 100k Taylor baseline budget reproduces the reference lr_decay_steps
    # (40000) exactly — clean decade drops at 40k/80k (re-pinned from 80k/30000).
    cfg = VALID.replace("training_steps = 100", "training_steps = 100000")
    rc = load_run_config(_write(tmp_path, cfg))
    assert rc.train.lr_decay_steps == 40000


def test_load_run_config_rejects_explicit_lr_decay_steps(tmp_path):
    # Setting it by hand is exactly the footgun this derivation removes; reject it
    # with a clear message rather than silently honoring a mis-scaled value.
    bad = VALID.replace(
        "training_steps = 100", "lr_decay_steps = 30000\ntraining_steps = 100"
    )
    with pytest.raises(ConfigError, match="lr_decay_steps is derived"):
        load_run_config(_write(tmp_path, bad))


#: A complete, valid MGN grouped config (deforming_plate card input_frames=2).
VALID_MGN = """\
[run]
benchmark = "deforming_plate"
seed = 7

[model]
family = "mgn"
input_frames = 2
dim = 3
hidden_dim = 128
message_passing_steps = 15
nmlp_layers = 2
node_type_size = 9
world_edge_radius = 30.0
noise_std = 0.003
normalizer_warmup_steps = 1000
history_frames = 0
mesh_edge_max_stretch = 0.0

[train]
batch_size = 8
lr_init = 1e-4
lr_decay = 0.1
training_steps = 100
val_every = 50
w_pos = 1.0
w_aux = 1.0
aux_tail_weight = 0.0
train_frames = 0
"""


def test_load_run_config_mgn_happy_path(tmp_path):
    rc = load_run_config(_write(tmp_path, VALID_MGN))
    assert rc.family == "mgn"
    assert isinstance(rc.model, MGNConfig)
    assert isinstance(rc.train, TrainConfig)
    assert rc.train.benchmark == "deforming_plate"
    assert rc.model.input_frames == 2


def test_load_run_config_rejects_mgn_input_frames_off_card(tmp_path):
    # ADR-0035: input_frames must equal the benchmark card's (deforming_plate = 2).
    bad = VALID_MGN.replace("input_frames = 2", "input_frames = 6")
    with pytest.raises(ConfigError, match="must equal benchmark"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_mgn_unknown_key(tmp_path):
    bad = VALID_MGN.replace("noise_std = 0.003", "noise_st = 0.003")
    with pytest.raises(ConfigError, match="unknown keys: noise_st"):
        load_run_config(_write(tmp_path, bad))


def test_load_deforming_plate_mgn_config():
    # The ADR-0043 §8 reference config: the strict loader accepts it and the
    # deforming_plate card's input_frames=2 check passes.
    rc = load_run_config(REPO_ROOT / "configs" / "deforming_plate" / "mgn.toml")
    assert rc.family == "mgn"
    assert isinstance(rc.model, MGNConfig)
    assert rc.model.input_frames == 2
    assert rc.model.hidden_dim == 128
    assert rc.model.message_passing_steps == 15
    assert rc.train.training_steps == 1_000_000


def test_load_deforming_plate_mgn_smoke_config():
    rc = load_run_config(REPO_ROOT / "configs" / "deforming_plate" / "mgn_smoke.toml")
    assert rc.family == "mgn"
    assert isinstance(rc.model, MGNConfig)
    assert rc.model.input_frames == 2
    assert rc.model.hidden_dim == 16
    assert rc.train.training_steps == 50


#: A complete, valid Transolver grouped config (deforming_plate card input_frames=2).
VALID_TRANSOLVER = """\
[run]
benchmark = "deforming_plate"
seed = 7

[model]
family = "transolver"
input_frames = 2
dim = 3
hidden_dim = 128
n_layers = 8
n_heads = 8
slice_num = 64
mlp_ratio = 1
dropout = 0.0
node_type_size = 9
noise_std = 0.003
normalizer_warmup_steps = 1000
weight_decay = 1e-5
max_grad_norm = 0.1
history_frames = 0
frames_per_call = 1
impact_velocity_feature = false
time_conditioned = false
adaptive_temperature = false
slice_reparam = false

[train]
batch_size = 8
lr_init = 1e-4
lr_decay = 0.1
training_steps = 100
val_every = 50
w_pos = 1.0
w_aux = 1.0
aux_tail_weight = 0.0
train_frames = 0
"""


def test_load_run_config_transolver_happy_path(tmp_path):
    rc = load_run_config(_write(tmp_path, VALID_TRANSOLVER))
    assert rc.family == "transolver"
    assert isinstance(rc.model, TransolverConfig)
    assert isinstance(rc.train, TrainConfig)
    assert rc.train.benchmark == "deforming_plate"
    # Every TOML value equals the dataclass default, so equality is a full
    # per-field round-trip check.
    assert rc.model == TransolverConfig()


def test_load_run_config_transolver_frames_per_call_roundtrips(tmp_path):
    # The config layer accepts any int frames_per_call (the k=T sentinel 0 and
    # bundling k>1); the simulator resolves/gates it later (ADR-0050/0051).
    for value in (0, 5):
        cfg = VALID_TRANSOLVER.replace(
            "frames_per_call = 1", f"frames_per_call = {value}"
        )
        rc = load_run_config(_write(tmp_path, cfg))
        assert rc.model.frames_per_call == value


def test_load_run_config_transolver_impact_velocity_feature_roundtrips(tmp_path):
    # ADR-0051 B: default off; explicit on round-trips.
    assert (
        load_run_config(
            _write(tmp_path, VALID_TRANSOLVER)
        ).model.impact_velocity_feature
        is False
    )
    on = VALID_TRANSOLVER.replace(
        "impact_velocity_feature = false", "impact_velocity_feature = true"
    )
    assert load_run_config(_write(tmp_path, on)).model.impact_velocity_feature is True


def test_load_run_config_time_conditioned_roundtrips(tmp_path):
    # ADR-0053: default off; explicit on round-trips (with the required guards
    # already satisfied by the default frames_per_call=1 / velocity_history=false).
    assert (
        load_run_config(_write(tmp_path, VALID_TRANSOLVER)).model.time_conditioned
        is False
    )
    on = VALID_TRANSOLVER.replace("time_conditioned = false", "time_conditioned = true")
    assert load_run_config(_write(tmp_path, on)).model.time_conditioned is True


def test_load_run_config_adaptive_temperature_roundtrips(tmp_path):
    # ADR-0057: default off; explicit on round-trips.
    assert (
        load_run_config(_write(tmp_path, VALID_TRANSOLVER)).model.adaptive_temperature
        is False
    )
    on = VALID_TRANSOLVER.replace(
        "adaptive_temperature = false", "adaptive_temperature = true"
    )
    assert load_run_config(_write(tmp_path, on)).model.adaptive_temperature is True


def test_load_run_config_slice_reparam_roundtrips(tmp_path):
    # ADR-0057: default off; explicit on round-trips.
    assert (
        load_run_config(_write(tmp_path, VALID_TRANSOLVER)).model.slice_reparam is False
    )
    on = VALID_TRANSOLVER.replace("slice_reparam = false", "slice_reparam = true")
    assert load_run_config(_write(tmp_path, on)).model.slice_reparam is True


def test_load_run_config_rejects_time_conditioned_with_history(tmp_path):
    # ADR-0054 (on the ADR-0053 history_frames schema): the time-conditioned
    # scheme is history-free, so a nonzero history_frames is rejected.
    bad = VALID_TRANSOLVER.replace("history_frames = 0", "history_frames = 1").replace(
        "time_conditioned = false", "time_conditioned = true"
    )
    with pytest.raises(ConfigError, match="requires history_frames=0"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_time_conditioned_with_kframes(tmp_path):
    # ADR-0053: time-conditioning and k-frames-per-call are mutually exclusive.
    bad = VALID_TRANSOLVER.replace(
        "frames_per_call = 1", "frames_per_call = 4"
    ).replace("time_conditioned = false", "time_conditioned = true")
    with pytest.raises(ConfigError, match="requires frames_per_call=1"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_negative_frames_per_call(tmp_path):
    # A typo'd negative k is rejected at load with a clear message (0 is the
    # one-shot sentinel; k>=1 is concrete), not deep in simulator construction.
    bad = VALID_TRANSOLVER.replace("frames_per_call = 1", "frames_per_call = -3")
    with pytest.raises(ConfigError, match="frames_per_call must be >= 0"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_transolver_history_frames_roundtrips(tmp_path):
    # ADR-0053: history_frames is decoupled from the input_frames seed; any
    # value in [0, input_frames-1] round-trips (dp card input_frames=2 -> {0,1}).
    for value in (0, 1):
        cfg = VALID_TRANSOLVER.replace(
            "history_frames = 0", f"history_frames = {value}"
        )
        rc = load_run_config(_write(tmp_path, cfg))
        assert rc.model.history_frames == value


def test_load_run_config_rejects_history_frames_over_seed(tmp_path):
    # history_frames > input_frames-1 is unsatisfiable (a velocity needs a
    # preceding frame); rejected at load with a clear message (ADR-0053).
    bad = VALID_TRANSOLVER.replace("history_frames = 0", "history_frames = 2")
    with pytest.raises(ConfigError, match="history_frames must be in"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_negative_history_frames(tmp_path):
    bad = VALID_TRANSOLVER.replace("history_frames = 0", "history_frames = -1")
    with pytest.raises(ConfigError, match="history_frames must be in"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_transolver_unknown_key(tmp_path):
    bad = VALID_TRANSOLVER.replace("noise_std = 0.003", "noise_st = 0.003")
    with pytest.raises(ConfigError, match="unknown keys: noise_st"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_transolver_missing_key(tmp_path):
    bad = VALID_TRANSOLVER.replace("dropout = 0.0\n", "")
    with pytest.raises(ConfigError, match="missing keys: dropout"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_transolver_input_frames_off_card(tmp_path):
    # ADR-0035: input_frames must equal the benchmark card's (deforming_plate = 2).
    bad = VALID_TRANSOLVER.replace("input_frames = 2", "input_frames = 6")
    with pytest.raises(ConfigError, match="must equal benchmark"):
        load_run_config(_write(tmp_path, bad))


def test_load_deforming_plate_transolver_config():
    # The ADR-0044 reference config: the strict loader accepts it and the
    # deforming_plate card's input_frames=2 check passes.
    rc = load_run_config(REPO_ROOT / "configs" / "deforming_plate" / "transolver.toml")
    assert rc.family == "transolver"
    assert isinstance(rc.model, TransolverConfig)
    assert rc.model.input_frames == 2
    assert rc.model.hidden_dim == 128
    assert rc.model.n_layers == 8
    assert rc.train.training_steps == 2_000_000


def test_load_deforming_plate_transolver_smoke_config():
    rc = load_run_config(
        REPO_ROOT / "configs" / "deforming_plate" / "transolver_smoke.toml"
    )
    assert rc.family == "transolver"
    assert isinstance(rc.model, TransolverConfig)
    assert rc.model.input_frames == 2
    assert rc.model.hidden_dim == 16
    assert rc.train.training_steps == 50


#: A complete, valid GeoFLARE grouped config (deforming_plate card input_frames=2).
#: n_layers and slice_num are diverged from GeoFlareConfig's defaults (6, 128) so
#: the round-trip assertion below distinguishes plumbed-through values from
#: silently-defaulted ones (the Transolver-branch deferred-minor lesson).
VALID_GEOFLARE = """\
[run]
benchmark = "deforming_plate"
seed = 7

[model]
family = "geoflare"
input_frames = 2
dim = 3
n_hidden = 256
n_layers = 5
n_heads = 8
slice_num = 64
mlp_ratio = 4
dropout = 0.0
n_hidden_local = 32
radius_near = 0.05
radius_far = 0.25
neighbors_near = 8
neighbors_far = 32
node_type_size = 9
noise_std = 0.003
normalizer_warmup_steps = 1000
weight_decay = 1e-4
max_grad_norm = 0.0
history_frames = 0

[train]
batch_size = 8
lr_init = 1e-4
lr_decay = 0.1
training_steps = 100
val_every = 50
w_pos = 1.0
w_aux = 1.0
aux_tail_weight = 0.0
train_frames = 0
"""


def test_load_run_config_geoflare_happy_path(tmp_path):
    rc = load_run_config(_write(tmp_path, VALID_GEOFLARE))
    assert rc.family == "geoflare"
    assert isinstance(rc.model, GeoFlareConfig)
    assert isinstance(rc.train, TrainConfig)
    assert rc.train.benchmark == "deforming_plate"
    # Full per-field round-trip: every value equals the dataclass default
    # except the two deliberately diverged fields, so a typo anywhere in
    # load_run_config's [model] plumbing would be caught by the mismatch.
    assert rc.model == replace(GeoFlareConfig(), n_layers=5, slice_num=64)


def test_load_run_config_rejects_geoflare_unknown_key(tmp_path):
    bad = VALID_GEOFLARE.replace("noise_std = 0.003", "noise_st = 0.003")
    with pytest.raises(ConfigError, match="unknown keys: noise_st"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_geoflare_missing_key(tmp_path):
    bad = VALID_GEOFLARE.replace("dropout = 0.0\n", "")
    with pytest.raises(ConfigError, match="missing keys: dropout"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_rejects_geoflare_input_frames_off_card(tmp_path):
    # ADR-0035: input_frames must equal the benchmark card's (deforming_plate = 2).
    bad = VALID_GEOFLARE.replace("input_frames = 2", "input_frames = 6")
    with pytest.raises(ConfigError, match="must equal benchmark"):
        load_run_config(_write(tmp_path, bad))


def test_load_deforming_plate_geoflare_config():
    # The ADR-0045 reference config: the strict loader accepts it and the
    # deforming_plate card's input_frames=2 check passes.
    rc = load_run_config(REPO_ROOT / "configs" / "deforming_plate" / "geoflare.toml")
    assert rc.family == "geoflare"
    assert isinstance(rc.model, GeoFlareConfig)
    assert rc.model.input_frames == 2
    assert rc.model.n_hidden == 256
    assert rc.model.n_layers == 6
    assert rc.train.training_steps == 2_000_000


def test_load_deforming_plate_geoflare_smoke_config():
    rc = load_run_config(
        REPO_ROOT / "configs" / "deforming_plate" / "geoflare_smoke.toml"
    )
    assert rc.family == "geoflare"
    assert isinstance(rc.model, GeoFlareConfig)
    assert rc.model.input_frames == 2
    assert rc.model.n_hidden == 16
    assert rc.train.training_steps == 50


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


def test_load_run_config_rejects_unknown_aux_transform(tmp_path):
    bad = VALID.replace('aux_transform = "none"', 'aux_transform = "rank"')
    with pytest.raises(ConfigError, match="unknown aux_transform"):
        load_run_config(_write(tmp_path, bad))


def test_load_run_config_requires_the_aux_knobs_explicitly(tmp_path):
    # ADR-0032 exact-keys: the new strain-channel knobs are recipe record
    # entries like any other; a config omitting them must not load silently.
    bad = VALID.replace("aux_tail_weight = 0.0\n", "")
    with pytest.raises(ConfigError, match="missing keys: aux_tail_weight"):
        load_run_config(_write(tmp_path, bad))


def test_load_taylor_native_configs():
    # ADR-0047: the three mesh-native provisional configs load through the
    # strict loader and satisfy the taylor card's input_frames=6 check
    # (ADR-0035) with the CGN-matched 100k budget.
    for name, cls in (
        ("mgn", MGNConfig),
        ("transolver", TransolverConfig),
        ("geoflare", GeoFlareConfig),
    ):
        rc = load_run_config(
            REPO_ROOT / "configs" / "taylor_impact_2d" / f"{name}.toml"
        )
        assert rc.family == name
        assert isinstance(rc.model, cls)
        assert rc.model.input_frames == 6
        assert rc.model.dim == 2
        assert rc.model.node_type_size == 3
        assert rc.train.training_steps == 100_000


def test_load_taylor_native_smoke_configs():
    for name, cls in (
        ("mgn", MGNConfig),
        ("transolver", TransolverConfig),
        ("geoflare", GeoFlareConfig),
    ):
        rc = load_run_config(
            REPO_ROOT / "configs" / "taylor_impact_2d" / f"{name}_smoke.toml"
        )
        assert rc.family == name
        assert isinstance(rc.model, cls)
        assert rc.model.input_frames == 6
        assert rc.train.training_steps == 50
