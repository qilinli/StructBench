"""BenchmarkCard invariants (ADR-0027)."""

import json

import pytest

from structbench.benchmarks.card import BenchmarkCard


def _kwargs(**overrides):
    base = dict(
        name="Demo-Bench",
        version="0.1",
        description="A demo benchmark.",
        provenance="Synthetic, for tests.",
        data_license="CC BY 4.0",
        solver="LS-DYNA",
        discretisation="SPH",
        materials=("*MAT_ELASTIC",),
        erosion=False,
        loading="rigid-wall impact",
        source_units="g-mm-ms",
        geometry="2D bar",
        n_cases=3,
        splits={"train": 2, "val": 1},
        task="autoregressive transition",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=("final_length",),
        fields=("positions", "stress"),
        particles_per_case="100",
        n_frames=10,
        output_dt_ms=0.1,
    )
    base.update(overrides)
    return base


def test_card_accepts_consistent_splits():
    card = BenchmarkCard(**_kwargs())
    assert card.n_cases == 3
    assert card.size_gb is None


def test_card_rejects_split_sum_mismatch():
    with pytest.raises(ValueError, match="n_cases"):
        BenchmarkCard(**_kwargs(n_cases=99))


def test_card_json_dict_serializes():
    card = BenchmarkCard(**_kwargs())
    payload = json.dumps(card.to_json_dict())
    assert "Demo-Bench" in payload
