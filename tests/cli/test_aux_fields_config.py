"""ADR-0059: the train.aux_fields selection and per-channel knob loading."""

import pytest

from structbench.config import ConfigError, load_run_config

_BASE = """
[run]
benchmark = "taylor_impact_2d"
seed = 1

[model]
family = "cgn"
input_frames = 6
connectivity_radius = 1.5
hidden_dim = 8
message_passing_steps = 2
nmlp_layers = 1
particle_type_embedding_size = 9
noise_std = 0.02
dim = 2
max_neighbors = 32
aux_transform = {aux_transform}
aux_transform_scale = {aux_transform_scale}

[train]
batch_size = 2
lr_init = 1e-4
lr_decay = 0.1
training_steps = 10
val_every = 10
w_pos = 1.0
w_aux = 1.0
aux_tail_weight = {aux_tail_weight}
train_frames = 0
{aux_fields_line}
"""


def _load(
    tmp_path,
    *,
    aux_transform='"none"',
    aux_transform_scale="0.01",
    aux_tail_weight="0.0",
    aux_fields_line="",
):
    p = tmp_path / "run.toml"
    p.write_text(
        _BASE.format(
            aux_transform=aux_transform,
            aux_transform_scale=aux_transform_scale,
            aux_tail_weight=aux_tail_weight,
            aux_fields_line=aux_fields_line,
        )
    )
    return load_run_config(p)


def test_aux_fields_key_is_optional_and_defaults_none(tmp_path):
    assert _load(tmp_path).train.aux_fields is None


def test_aux_fields_list_normalizes_to_tuple(tmp_path):
    train_cfg = _load(
        tmp_path,
        aux_fields_line='aux_fields = ["deviatoric_stress_2d", "density"]',
        aux_transform='["none", "none", "none", "none"]',
        aux_transform_scale="[0.01, 0.01, 0.01, 0.01]",
        aux_tail_weight="[0.0, 0.0, 0.0, 0.0]",
    ).train
    assert train_cfg.aux_fields == ("deviatoric_stress_2d", "density")
    assert train_cfg.aux_tail_weight == (0.0, 0.0, 0.0, 0.0)


def test_unknown_aux_field_rejected_at_load(tmp_path):
    with pytest.raises(ConfigError, match="unknown aux_fields"):
        _load(tmp_path, aux_fields_line='aux_fields = ["not_a_field"]')


def test_per_channel_knob_length_mismatch_rejected(tmp_path):
    # canonical taylor selection is 1 channel; a 3-entry knob cannot apply
    with pytest.raises(ConfigError, match="aux_tail_weight"):
        _load(tmp_path, aux_tail_weight="[0.0, 1.0, 2.0]")


def test_unknown_transform_in_per_channel_list_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown aux_transform"):
        _load(
            tmp_path,
            aux_transform='["nope"]',
            aux_transform_scale="[0.01]",
        )
