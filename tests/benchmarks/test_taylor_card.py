"""The Taylor card's ML stats are computed from the split constants."""

from structbench.benchmarks.taylor_impact_2d.benchmark import (
    AUX_FIELD,
    QOIS,
    TEST_EXTRAP,
    TEST_INTERP,
    TRAIN,
    VAL,
)
from structbench.benchmarks.taylor_impact_2d.card import CARD


def test_ml_stats_are_computed_from_split_constants():
    assert CARD.n_cases == len(TRAIN) + len(VAL) + len(TEST_INTERP) + len(TEST_EXTRAP)
    assert CARD.splits == {
        "train": len(TRAIN),
        "val": len(VAL),
        "test_interp": len(TEST_INTERP),
        "test_extrap": len(TEST_EXTRAP),
    }
    assert CARD.aux_field == AUX_FIELD
    assert set(CARD.qois) == set(QOIS)


def test_physics_facts_match_adr_0019():
    assert CARD.discretisation == "SPH"
    assert CARD.erosion is False
    assert CARD.source_units == "g-mm-ms"
