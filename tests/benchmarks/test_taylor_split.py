import numpy as np
import torch

from structbench.benchmarks.taylor_impact_2d import (
    HELD_ASIDE,
    QOIS,
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
    assert len(all_cases) == len(set(all_cases)) == 33  # no overlap
    assert HELD_ASIDE == ["T-20-80-Convergence"]


def test_split_velocities_match_adr_0019():
    def vels(split):
        return sorted({int(c.split("-")[3]) for c in split})

    assert vels(VAL) == [150]
    assert vels(TEST_INTERP) == [130, 170]
    assert vels(TEST_EXTRAP) == [200]


def test_wall_distance_feature_is_signed_and_radius_normalized():
    # ADR-0024: clamp((x - wall)/R, -1, 1) — penetration must read negative.
    pos = torch.tensor(
        [[-3.0, 0.0], [-2.3, 0.0], [-2.0, 0.0], [-1.7, 0.0], [10.0, 0.0]]
    )  # mm
    feat = wall_distance_feature(pos, radius=0.6)
    assert feat.shape == (5, 1)
    torch.testing.assert_close(feat[:, 0], torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]))


def test_qois_bind_the_adr_0019_quantities():
    # ADR-0019 §5: final bar length and mushroom width, evaluated on the last
    # frame of a (T, P, dim) trajectory.
    assert set(QOIS) == {"final_length", "mushroom_width"}
    pos = np.zeros((2, 4, 2))
    pos[-1, :, 0] = [0.0, 10.0, 5.0, 5.0]
    pos[-1, :, 1] = [-3.0, 3.0, 0.0, 0.0]
    assert QOIS["final_length"](pos) == 10.0
    assert QOIS["mushroom_width"](pos) == 6.0
