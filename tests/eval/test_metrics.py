import numpy as np
import pytest

from structbench.eval.metrics import (
    QoiInputs,
    arrival_time,
    field_rmse,
    final_length,
    mushroom_width,
    peak_stress,
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


def _plane_wave_inputs():
    """A |stress| front moving +x at 1 station per frame; bar x in [0, 100]."""
    t = np.linspace(0.0, 0.01, 11)  # 11 frames, seconds
    x = np.linspace(0.0, 100.0, 101, dtype=np.float32)
    positions = np.zeros((11, 101, 2), np.float32)
    positions[:, :, 0] = x  # static bar
    aux = np.zeros((11, 101), np.float32)
    for frame in range(11):
        front = frame * 10.0  # front position in mm at this frame
        aux[frame, x <= front] = 5.0  # 5 MPa behind the front
    return QoiInputs(time=t, positions=positions, aux=aux)


def test_arrival_time_reads_the_front_crossing():
    inputs = _plane_wave_inputs()
    # station 0.5 -> x = 50 mm; front reaches it at frame 5 -> t = 0.005 s = 5 ms
    assert arrival_time(0.5)(inputs) == pytest.approx(5.0)
    assert arrival_time(0.25)(inputs) == pytest.approx(2.5, abs=0.51)  # frame 3


def test_arrival_time_saturates_when_no_crossing():
    inputs = _plane_wave_inputs()
    quiet = QoiInputs(
        time=inputs.time, positions=inputs.positions, aux=np.zeros_like(inputs.aux)
    )
    assert arrival_time(0.5)(quiet) == pytest.approx(inputs.time[-1] * 1e3)


def test_peak_stress_reads_late_half():
    inputs = _plane_wave_inputs()
    # frames 0-10; late half = 5-10; front fills to 5.0 by frame 5 and stays
    assert peak_stress(inputs) == pytest.approx(5.0)


def test_peak_stress_ignores_early_only_spike():
    base = _plane_wave_inputs()
    aux = base.aux.copy()
    aux[:] = 0.0
    aux[0, :] = 99.0  # early-only spike, inside the would-be seeded frames
    aux[8, 3] = 2.0  # late-half signal
    spiked = QoiInputs(time=base.time, positions=base.positions, aux=aux)
    assert peak_stress(spiked) == pytest.approx(2.0)


def test_position_rmse_keep_mask_excludes_particles():
    pred = np.zeros((2, 3, 2), np.float32)
    true = np.zeros((2, 3, 2), np.float32)
    true[:, 2, :] = 10.0  # particle 2 is wildly wrong
    keep = np.array([True, True, False])
    full = position_rmse(pred, true)
    masked = position_rmse(pred, true, keep=keep)
    assert full[0] > 0 and np.allclose(masked, 0.0)


def test_field_rmse_keep_mask():
    pred = np.zeros((2, 3), np.float32)
    true = np.zeros((2, 3), np.float32)
    true[:, 0] = 4.0
    assert np.allclose(field_rmse(pred, true, keep=np.array([False, True, True])), 0.0)
