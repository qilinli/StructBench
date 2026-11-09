"""The coverage kernel (ADR-0074 clause 2; open-evaluation plan §4)."""

import numpy as np
import pytest

from structbench.eval.coverage import (
    Coverage,
    coverage,
    coverage_bin,
    median_spacing,
)

# Taylor's training design: lengths {60, 80, 100} x velocities {100 … 190}
# minus the interior holdouts; box = the training design's.
_TRAIN_V = (100, 110, 120, 140, 160, 180, 190)
TAYLOR_TRAIN = np.array([(L, v) for L in (60, 80, 100) for v in _TRAIN_V], float)
TAYLOR_BOX = [(60.0, 100.0), (100.0, 190.0)]


def test_taylor_numbers_from_the_plan():
    scored = np.array(
        [
            (80, 130),  # interior test velocity: one velocity step from 120/140
            (80, 200),  # extrapolation: one step past 190, beyond the box
            (70, 150),  # a new bar length, a quarter of the length range off
            (60, 100),  # a training point itself
        ],
        float,
    )
    cov = coverage(TAYLOR_TRAIN, scored, TAYLOR_BOX)
    # the unit is the velocity step (10 of 90): four of seven training
    # velocities per line have a neighbour one step away
    assert cov[0].within and cov[0].distance == pytest.approx(1.0)
    assert not cov[1].within and cov[1].distance == pytest.approx(1.0)
    # sqrt(0.25² + (1/9)²) / (1/9)
    assert cov[2].within and cov[2].distance == pytest.approx(2.462, abs=0.001)
    assert cov[3].within and cov[3].distance == 0.0
    assert [coverage_bin(c) for c in cov] == [
        "within, ≤ 1",
        "beyond, ≤ 1",
        "within, > 2",
        "within, ≤ 1",
    ]


def test_box_edges_are_inclusive():
    cov = coverage(
        TAYLOR_TRAIN, np.array([(100.0, 190.0), (100.0, 190.0 + 1e-6)]), TAYLOR_BOX
    )
    assert cov[0].within
    assert not cov[1].within


def test_fixed_axis_is_dropped_from_the_distance_and_moving_it_is_beyond():
    train = np.array([(80, 100), (80, 110), (80, 120)], float)
    box = [(80.0, 80.0), (100.0, 120.0)]
    cov = coverage(train, np.array([(80, 115), (100, 115)], float), box)
    assert cov[0].within and cov[0].distance == pytest.approx(0.5)
    assert not cov[1].within and cov[1].distance == pytest.approx(0.5)


def test_sobol_like_cloud_reads_in_units_of_its_own_spacing():
    rng = np.random.default_rng(0)
    train = rng.uniform(0.0, 1.0, size=(256, 3))
    box = [(0.0, 1.0)] * 3
    unit = median_spacing(train)
    far = np.array([[2.0, 2.0, 2.0]])
    cov = coverage(train, far, box)
    assert not cov[0].within
    expected = np.linalg.norm(train - far[0], axis=1).min() / unit
    assert cov[0].distance == pytest.approx(expected)
    assert coverage_bin(cov[0]) == "beyond, > 2"


def test_novelty_is_declared_and_shown_apart():
    cov = coverage(
        TAYLOR_TRAIN, np.array([(80, 130)], float), TAYLOR_BOX, novelty=[True]
    )
    assert cov[0].novelty and coverage_bin(cov[0]) == "novel"
    assert coverage_bin(Coverage(True, 1.5, 0)) == "within, 1–2"


def test_kernel_refuses_malformed_inputs():
    with pytest.raises(ValueError, match="parameter count"):
        coverage(TAYLOR_TRAIN, np.array([(80.0,)]), TAYLOR_BOX)
    with pytest.raises(ValueError, match="low <= high"):
        coverage(
            TAYLOR_TRAIN, np.array([(80, 130)], float), [(100.0, 60.0), (100.0, 190.0)]
        )
    with pytest.raises(ValueError, match="at least two"):
        coverage(TAYLOR_TRAIN[:1], np.array([(80, 130)], float), TAYLOR_BOX)
    # repeated numeric points (categorical twins) are one distinct point
    with pytest.raises(ValueError, match="at least two distinct"):
        coverage(
            np.array([(80, 100), (80, 100)], float),
            np.array([(80, 130)], float),
            TAYLOR_BOX,
        )


def test_categorical_twins_do_not_collapse_the_spacing():
    # Two impactor shapes at every grid point: the numeric points repeat, the
    # spacing is still the grid step, and a scored case sharing a grid point
    # with a training case of another shape reads distance 0, within.
    grid = [(w, v) for w in (320, 480, 640) for v in (40, 80, 120, 160)]
    train = np.array(grid + grid, float)
    box = [(320.0, 640.0), (40.0, 160.0)]
    cov = coverage(train, np.array([(480, 80), (480, 100)], float), box)
    assert cov[0].within and cov[0].distance == 0.0
    # the unit is the velocity step (40 of 120); 100 m/s is half a step off
    assert cov[1].within and cov[1].distance == pytest.approx(0.5)
    with pytest.raises(ValueError, match="novelty"):
        coverage(
            TAYLOR_TRAIN,
            np.array([(80, 130)], float),
            TAYLOR_BOX,
            novelty=[True, False],
        )
