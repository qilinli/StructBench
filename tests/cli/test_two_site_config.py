"""Load-time rules of ``two_site_decoder`` (ADR-0075 clause 3)."""

from types import SimpleNamespace

import pytest

from structbench.benchmarks import (
    RETURN_MAP_STATE_CHANNELS,
    TWO_SITE_DECODERS,
    ConstitutiveLaw,
)
from structbench.config import (
    ConfigError,
    TrainConfig,
    TransolverConfig,
    _validate_two_site,
)
from structbench.models.common.two_site import TWO_SITE_DECODERS as MODEL_DECODERS


def _spec(**over):
    base = dict(
        state_site="element",
        state_channels=RETURN_MAP_STATE_CHANNELS,
        aux_field="von_mises_stress",
        constitutive_law=lambda cid: ConstitutiveLaw(
            1.0, 0.3, ((0.0, 1.0), (1.0, 2.0))
        ),
    )
    base.update(over)
    return SimpleNamespace(**base)


def _train(aux_fields=RETURN_MAP_STATE_CHANNELS):
    return TrainConfig(benchmark="x", aux_fields=aux_fields)


def test_the_two_decoder_vocabularies_agree():
    assert TWO_SITE_DECODERS == MODEL_DECODERS


def test_off_passes_without_a_spec():
    _validate_two_site(
        TransolverConfig(two_site_decoder="off"), _train(None), None, "x"
    )


def test_unknown_value_is_rejected():
    with pytest.raises(ConfigError, match="must be one of"):
        _validate_two_site(
            TransolverConfig(two_site_decoder="magic"), _train(), _spec(), "x"
        )


@pytest.mark.parametrize(
    "knob",
    [
        {"flow_map_pushforward": True},
        {"flow_map_structured_heads": True},
        {"flow_map_consistency_hinge": 0.1},
    ],
)
def test_incompatible_knobs(knob):
    cfg = TransolverConfig(two_site_decoder="return_map", **knob)
    with pytest.raises(ConfigError, match="incompatible"):
        _validate_two_site(cfg, _train(), _spec(), "x")


def test_needs_an_element_site_and_the_state_layout():
    cfg = TransolverConfig(two_site_decoder="return_map")
    with pytest.raises(ConfigError, match="state_site='element'"):
        _validate_two_site(cfg, _train(), _spec(state_site="node"), "x")
    with pytest.raises(ConfigError, match="train.aux_fields"):
        _validate_two_site(cfg, _train(("effective_plastic_strain",)), _spec(), "x")
    with pytest.raises(ConfigError, match="requires the state layout"):
        _validate_two_site(
            cfg,
            _train(("effective_plastic_strain",)),
            _spec(state_channels=("effective_plastic_strain",)),
            "x",
        )
    with pytest.raises(ConfigError, match="constitutive_law"):
        _validate_two_site(cfg, _train(), _spec(constitutive_law=None), "x")
    _validate_two_site(cfg, _train(), _spec(), "x")
    # corner_mean needs the layout match but no law.
    cm = TransolverConfig(two_site_decoder="corner_mean")
    _validate_two_site(cm, _train(), _spec(constitutive_law=None), "x")
