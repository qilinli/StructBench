import torch

from structbench.benchmarks.taylor_impact_2d import (
    HELD_ASIDE,
    TEST_EXTRAP,
    TEST_INTERP,
    TRAIN,
    VAL,
    wall_distance_feature,
)


def test_split_partitions_the_33_parametric_cases():
    splits = [TRAIN, VAL, TEST_INTERP, TEST_EXTRAP]
    sizes = [len(s) for s in splits]
    assert sizes == [21, 3, 6, 3]
    all_cases = [c for s in splits for c in s]
    assert len(all_cases) == len(set(all_cases)) == 33   # no overlap
    assert HELD_ASIDE == ["T-20-80-Convergence"]


def test_split_velocities_match_adr_0019():
    def vels(split):
        return sorted({int(c.split("-")[3]) for c in split})
    assert vels(VAL) == [150]
    assert vels(TEST_INTERP) == [130, 170]
    assert vels(TEST_EXTRAP) == [200]


def test_wall_distance_feature_clamps_to_radius():
    pos = torch.tensor([[-2.0, 0.0], [-1.5, 0.0], [10.0, 0.0]])  # mm
    feat = wall_distance_feature(pos, radius=0.6)
    assert feat.shape == (3, 1)
    torch.testing.assert_close(feat[:, 0], torch.tensor([0.0, 0.5, 0.6]))
