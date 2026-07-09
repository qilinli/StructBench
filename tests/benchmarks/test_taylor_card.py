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


def test_overview_cites_the_actual_split_velocities_and_geometry():
    # The landing-page overview (ADR-0036) is non-derivable prose, so it hand-
    # names split velocities and bar lengths. Guard it against silent drift: if
    # a split velocity or geometry changes (a new benchmark version), the prose
    # must be updated or this fails. Case ids are "T-20-<geom>-<velocity>".
    overview = CARD.overview
    velocities = {cid.split("-")[3] for cid in TEST_INTERP + TEST_EXTRAP + VAL}
    geometries = {cid.split("-")[2] for cid in TRAIN + VAL + TEST_INTERP + TEST_EXTRAP}
    for vel in velocities:
        assert vel in overview, f"overview omits held-out/val velocity {vel} m/s"
    for geom in geometries:
        assert geom in overview, f"overview omits geometry length {geom} mm"
