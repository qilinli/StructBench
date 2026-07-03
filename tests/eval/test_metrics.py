import numpy as np

from structbench.eval.metrics import (
    QoiInputs,
    field_rmse,
    final_length,
    mushroom_width,
    position_rmse,
)


def _inputs(positions):
    """Wrap a raw positions array in a QoiInputs with zero aux and integer time."""
    t = positions.shape[0]
    return QoiInputs(
        time=np.arange(t, dtype=float),
        positions=np.asarray(positions, np.float32),
        aux=np.zeros(positions.shape[:2], np.float32),
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
    pos[0, :, 0] = [0, 10, 5, 5]  # x extent 10
    pos[0, :, 1] = [-3, 3, 0, 0]  # y extent 6
    assert final_length(_inputs(pos)) == 10.0
    assert mushroom_width(_inputs(pos)) == 6.0
