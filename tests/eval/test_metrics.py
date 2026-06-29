import numpy as np

from structbench.eval.metrics import (
    field_rmse,
    final_length,
    mushroom_width,
    position_rmse,
)


def test_position_rmse_per_frame():
    true = np.zeros((2, 3, 2))
    pred = np.zeros((2, 3, 2))
    pred[1] = 3.0  # every component off by 3 at frame 1 -> rmse sqrt(mean(9))=3
    out = position_rmse(pred, true)
    np.testing.assert_allclose(out, [0.0, 3.0])


def test_field_rmse_per_frame():
    true = np.zeros((2, 4))
    pred = np.array([[0, 0, 0, 0], [2, 2, 2, 2]], dtype=float)
    np.testing.assert_allclose(field_rmse(pred, true), [0.0, 2.0])


def test_qois_use_last_frame_extents():
    pos = np.zeros((1, 4, 2))
    pos[0, :, 0] = [0, 10, 5, 5]   # x extent 10
    pos[0, :, 1] = [-3, 3, 0, 0]   # y extent 6
    assert final_length(pos) == 10.0
    assert mushroom_width(pos) == 6.0
