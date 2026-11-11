"""Load-time rules of ``two_site_decoder`` (ADR-0075 clause 3)."""

from types import SimpleNamespace

import pytest

from structbench.benchmarks import (
    LAW_DECODERS,
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
from structbench.models.common.two_site import LAW_DECODERS as MODEL_LAW_DECODERS
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
    assert LAW_DECODERS == MODEL_LAW_DECODERS
    assert set(LAW_DECODERS) == {"return_map", "admissible_coords", "projected_trial"}


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


@pytest.mark.parametrize("decoder", ["admissible_coords", "projected_trial"])
def test_the_admissible_forms_need_the_law_and_the_layout(decoder):
    cfg = TransolverConfig(two_site_decoder=decoder)
    _validate_two_site(cfg, _train(), _spec(), "x")
    with pytest.raises(ConfigError, match="constitutive_law"):
        _validate_two_site(cfg, _train(), _spec(constitutive_law=None), "x")
    other = ("von_mises_stress", "effective_plastic_strain")
    with pytest.raises(ConfigError, match="state layout"):
        _validate_two_site(
            cfg, _train(aux_fields=other), _spec(state_channels=other), "x"
        )
    with pytest.raises(ConfigError, match="incompatible"):
        _validate_two_site(
            TransolverConfig(two_site_decoder=decoder, flow_map_structured_heads=True),
            _train(),
            _spec(),
            "x",
        )


@pytest.mark.parametrize(
    "decoder", ["corner_mean", "admissible_coords", "projected_trial"]
)
def test_the_two_hop_chain_is_accepted_for_the_head_decoders(decoder):
    """ADR-0075 note 2026-11-11: ``flow_map_pushforward`` loads with the
    three decoders that read the state off the head; the generation
    curriculum stays nodal, and the other 3a refusals stand."""
    from structbench.benchmarks.constitutive import TWO_SITE_CHAIN_DECODERS

    assert decoder in TWO_SITE_CHAIN_DECODERS
    spec = _spec() if decoder != "corner_mean" else _spec()
    _validate_two_site(
        TransolverConfig(two_site_decoder=decoder, flow_map_pushforward=True),
        _train(),
        spec,
        "x",
    )
    with pytest.raises(ConfigError, match="generations"):
        _validate_two_site(
            TransolverConfig(
                two_site_decoder=decoder,
                flow_map_pushforward=True,
                flow_map_pushforward_generations=2,
            ),
            _train(),
            spec,
            "x",
        )
    for knob in (
        {"flow_map_structured_heads": True},
        {"flow_map_consistency_hinge": 0.1},
    ):
        with pytest.raises(ConfigError, match="incompatible"):
            _validate_two_site(
                TransolverConfig(
                    two_site_decoder=decoder, flow_map_pushforward=True, **knob
                ),
                _train(),
                spec,
                "x",
            )


def test_the_direct_return_map_has_no_chain():
    with pytest.raises(ConfigError, match="flow_map_pushforward"):
        _validate_two_site(
            TransolverConfig(two_site_decoder="return_map", flow_map_pushforward=True),
            _train(),
            _spec(),
            "x",
        )
