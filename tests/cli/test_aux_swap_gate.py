"""E-X env-gated aux-target swap screens (train-process only)."""

from structbench.cli.train import _env_aux_field_override
from structbench.datasets import available_aux_fields


def test_unset_env_is_a_no_op(monkeypatch):
    monkeypatch.delenv("STRUCTBENCH_TAYLOR_AUX_MPSTRAIN", raising=False)
    monkeypatch.delenv("STRUCTBENCH_NOTCH_AUX_VM", raising=False)
    for bench in (
        "taylor_impact_2d",
        "notch_beam_2d_impact",
        "wave_propagation_1d",
        "deforming_plate",
    ):
        assert _env_aux_field_override(bench) is None


def test_empty_env_value_is_a_no_op(monkeypatch):
    # os.environ.get truthiness: an empty string does NOT arm the gate.
    monkeypatch.setenv("STRUCTBENCH_TAYLOR_AUX_MPSTRAIN", "")
    assert _env_aux_field_override("taylor_impact_2d") is None


def test_taylor_gate_swaps_to_max_principal_strain(monkeypatch):
    monkeypatch.setenv("STRUCTBENCH_TAYLOR_AUX_MPSTRAIN", "1")
    assert _env_aux_field_override("taylor_impact_2d") == "max_principal_strain"
    # The gate is benchmark-scoped: no other benchmark is touched.
    assert _env_aux_field_override("notch_beam_2d_impact") is None
    assert _env_aux_field_override("wave_propagation_1d") is None


def test_notch_gate_swaps_to_von_mises(monkeypatch):
    monkeypatch.setenv("STRUCTBENCH_NOTCH_AUX_VM", "1")
    assert _env_aux_field_override("notch_beam_2d_impact") == "von_mises_stress"
    assert _env_aux_field_override("taylor_impact_2d") is None


def test_swapped_fields_are_registered_loader_extractors():
    # The swap must reuse the canonical extractors (same definitions the
    # benchmarks already use), never new math: both targets resolve in the
    # loader registry.
    for env_field in ("max_principal_strain", "von_mises_stress"):
        assert env_field in available_aux_fields()
