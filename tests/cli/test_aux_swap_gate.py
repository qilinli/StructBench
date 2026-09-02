"""The ADR-0059 aux_fields selection replaced the env-gated E-X swap."""

from structbench.config import TrainConfig


def test_env_gate_is_retired():
    import structbench.cli.train as train_mod

    assert not hasattr(train_mod, "_env_aux_field_override")
    assert not hasattr(train_mod, "_ENV_AUX_SWAPS")


def test_train_config_carries_aux_fields_default_none():
    assert TrainConfig().aux_fields is None
