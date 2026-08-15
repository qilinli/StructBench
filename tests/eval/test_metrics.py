import numpy as np
import pytest

from structbench.eval.metrics import (
    QoiInputs,
    arrival_time,
    cracked_fraction,
    field_rmse,
    final_length,
    midspan_deflection_peak,
    mushroom_width,
    peak_nodal_aux,
    peak_stress,
    position_rmse,
    relative_l2,
    terminal_peak_displacement,
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


def test_relative_l2_scalar_field_analytic():
    """Per-frame ‖û−u‖₂ / ‖u‖₂ on a hand-built scalar field (ADR-0055)."""
    gt = np.array([[3.0, 4.0], [6.0, 8.0]])  # ‖·‖ = 5 and 10 per frame
    pred = np.array([[0.0, 0.0], [6.0, 8.0]])  # frame 0 all wrong, frame 1 exact
    # frame 0: ‖err‖=5, ‖gt‖=5 -> 1.0; frame 1: err 0 -> 0.0
    np.testing.assert_allclose(relative_l2(pred, gt), [1.0, 0.0])


def test_relative_l2_vector_field_norms_over_particles_and_dims():
    """A (T, P, dim) field is normed over particles and components jointly."""
    gt = np.zeros((1, 2, 2))
    gt[0, 0] = [3.0, 4.0]  # ‖gt‖ = 5
    pred = np.zeros((1, 2, 2))
    pred[0, 0] = [3.0, 0.0]  # err = [0, -4] -> ‖err‖ = 4
    np.testing.assert_allclose(relative_l2(pred, gt), [0.8])


def test_relative_l2_mask_excludes_kinematic_rows():
    """The particle mask drops kinematic rows from both norms (ADR-0026)."""
    gt = np.array([[3.0, 4.0, 0.0]])
    pred = np.array([[0.0, 0.0, 99.0]])  # particle 2 is wildly wrong
    mask = np.array([True, True, False])  # ... but excluded
    # Kept particles {0,1}: ‖err‖=5, ‖gt‖=5 -> 1.0; the 99 does not leak in.
    np.testing.assert_allclose(relative_l2(pred, gt, mask), [1.0])
    assert relative_l2(pred, gt)[0] > 10.0  # unmasked is dominated by the 99


def test_relative_l2_zero_gt_frame_does_not_divide_by_zero():
    """A fully static reference frame is floored by eps, never inf/nan."""
    gt = np.zeros((2, 2))  # ‖gt‖ = 0 on both frames
    pred = np.array([[1e-3, 0.0], [0.0, 0.0]])
    out = relative_l2(pred_field=pred, gt_field=gt)
    assert np.all(np.isfinite(out))
    # frame 0: 1e-3 / 1e-8 = 1e5; frame 1: 0 / 1e-8 = 0
    np.testing.assert_allclose(out, [1e5, 0.0])


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


def test_arrival_time_respects_scored_span():
    """A gauge crossing in the seeded prefix must not leak (ADR-0032 §4)."""
    t = np.arange(7, dtype=float) * 1e-3  # seconds; time[f]*1e3 == frame index
    x = np.linspace(0.0, 100.0, 101, dtype=np.float32)
    positions = np.zeros((7, 101, 2), np.float32)
    positions[:, :, 0] = x
    aux = np.zeros((7, 101), np.float32)
    aux[1, 50] = 10.0  # spurious spike at the gauge, inside the seeded prefix
    aux[5, 50] = 10.0  # true arrival, in the scored span
    inputs = QoiInputs(time=t, positions=positions, aux=aux, init=3)
    assert arrival_time(0.5)(inputs) == pytest.approx(5.0)  # not 1.0


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


def _beam_inputs():
    """Static 3-particle 'beam' on x in {0, 50, 100}; middle particle sags."""
    t = np.linspace(0.0, 0.5, 6)
    positions = np.zeros((6, 3, 2), np.float32)
    positions[:, :, 0] = np.array([0.0, 50.0, 100.0], np.float32)
    positions[:, 1, 1] = -np.array([0, 1, 2, 4, 3, 2], np.float32)  # sag, peak 4mm
    aux = np.zeros((6, 3), np.float32)
    # Final-frame principal strains: particles 0 and 2 exceed 0.01 threshold.
    aux[-1] = np.array([0.02, 0.005, 0.02], np.float32)
    ptype = np.array([1, 1, 2], np.int64)  # particle 2 is not concrete (type 1)
    return QoiInputs(time=t, positions=positions, aux=aux, particle_type=ptype)


def test_midspan_deflection_peak_reads_the_sag():
    qoi = midspan_deflection_peak(gauge_halfwidth=5.0)(_beam_inputs())
    assert qoi == pytest.approx(4.0)


def test_midspan_deflection_peak_respects_scored_span():
    """A large sag in the seeded prefix must not leak (ADR-0032 §4)."""
    t = np.arange(7, dtype=float)
    positions = np.zeros((7, 3, 2), np.float32)
    positions[:, :, 0] = np.array([0.0, 50.0, 100.0], np.float32)
    # middle particle: 100 mm sag in the prefix, 2 mm peak sag in the scored span
    positions[:, 1, 1] = -np.array([0, 100, 100, 1, 1, 2, 1], np.float32)
    aux = np.zeros((7, 3), np.float32)
    inputs = QoiInputs(time=t, positions=positions, aux=aux, init=3)
    qoi = midspan_deflection_peak(gauge_halfwidth=5.0)(inputs)
    assert qoi == pytest.approx(2.0)  # not 100.0


def test_cracked_fraction_final_frame():
    inputs = _beam_inputs()
    # All 3 particles: 2 with strain >= 0.01 -> 2/3
    assert cracked_fraction()(inputs) == pytest.approx(2.0 / 3.0)
    # concrete_type=1 restricts to particles 0 and 1: only particle 0 >= 0.01 -> 0.5
    qoi = cracked_fraction(concrete_type=1)(inputs)
    assert qoi == pytest.approx(0.5)


def test_peak_mean_aux_and_time_closed_form():
    """Peak of the particle-mean aux and its time, against hand-computed values."""
    from structbench.eval.metrics import peak_mean_aux, t_peak_mean_aux

    # 4 frames, 2 particles: particle means are [1.0, 5.0, 3.0, 2.0];
    # a single-particle outlier at frame 3 (9.0) must NOT move the peak.
    aux = np.array([[1.0, 1.0], [4.0, 6.0], [3.0, 3.0], [-5.0, 9.0]], dtype=np.float32)
    positions = np.zeros((4, 2, 2), dtype=np.float32)
    time = np.array([0.0, 1e-5, 2e-5, 3e-5])  # seconds
    inputs = QoiInputs(time=time, positions=positions, aux=aux)
    assert peak_mean_aux(inputs) == 5.0
    assert t_peak_mean_aux(inputs) == 1e-5 * 1e3  # frame 1, in ms


def test_peak_mean_aux_respects_scored_span():
    """A GT-seeded prefix peak must not leak into the QoI (ADR-0032 §4)."""
    from structbench.eval.metrics import peak_mean_aux, t_peak_mean_aux

    aux = np.array([[9.0, 9.0], [1.0, 1.0], [4.0, 4.0], [2.0, 2.0]], dtype=np.float32)
    positions = np.zeros((4, 2, 2), dtype=np.float32)
    time = np.array([0.0, 1e-5, 2e-5, 3e-5])
    inputs = QoiInputs(time=time, positions=positions, aux=aux, init=1)
    # Frame 0's value 9.0 is seeded ground truth; the scored peak is 4.0 @ frame 2.
    assert peak_mean_aux(inputs) == 4.0
    assert t_peak_mean_aux(inputs) == 2e-5 * 1e3


def test_peak_nodal_aux_masks_excluded_types():
    aux = np.zeros((4, 3), dtype=np.float32)
    aux[2, 0] = 5.0  # NORMAL node peak
    aux[3, 2] = 99.0  # kinematic node — must be ignored
    inp = QoiInputs(
        time=np.arange(4, dtype=np.float64),
        positions=np.zeros((4, 3, 3), dtype=np.float32),
        aux=aux,
        particle_type=np.array([0, 0, 1], dtype=np.int64),
        init=2,
    )
    qoi = peak_nodal_aux(exclude_types=(1, 3))
    assert qoi(inp) == pytest.approx(5.0)


def test_peak_nodal_aux_respects_init():
    aux = np.zeros((4, 2), dtype=np.float32)
    aux[0, 0] = 50.0  # before init — must be ignored
    aux[3, 1] = 2.0
    inp = QoiInputs(
        time=np.arange(4, dtype=np.float64),
        positions=np.zeros((4, 2, 3), dtype=np.float32),
        aux=aux,
        particle_type=None,
        init=2,
    )
    assert peak_nodal_aux()(inp) == pytest.approx(2.0)


def test_terminal_peak_displacement_masks_and_measures():
    pos = np.zeros((3, 3, 3), dtype=np.float32)
    pos[-1, 0] = [3.0, 4.0, 0.0]  # NORMAL: |disp| = 5
    pos[-1, 2] = [100.0, 0.0, 0.0]  # kinematic: ignored
    inp = QoiInputs(
        time=np.arange(3, dtype=np.float64),
        positions=pos,
        aux=np.zeros((3, 3), dtype=np.float32),
        particle_type=np.array([0, 0, 3], dtype=np.int64),
        init=2,
    )
    qoi = terminal_peak_displacement(exclude_types=(1, 3))
    assert qoi(inp) == pytest.approx(5.0)
