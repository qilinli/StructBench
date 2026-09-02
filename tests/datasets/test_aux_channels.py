"""ADR-0059: multi-channel auxiliary blocks through the datasets layer."""

import numpy as np
import pytest

from structbench.core import (
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
    write_case,
)
from structbench.datasets import (
    WindowDataset,
    as_aux_fields,
    aux_channel_count,
    aux_channel_labels,
    aux_channel_units,
    compute_stats,
    expand_aux_knob,
    load_case_trajectory,
)
from structbench.datasets.canonical import CaseTrajectory
from structbench.datasets.sph_mesh import append_wall_nodes


def _state_case(tmp_path):
    """Two-frame, three-particle SPH case carrying the full state fields."""
    coords = np.array([[0.0, 0.0], [1e-3, 0.0], [0.0, 1e-3]])
    disp = np.zeros((2, 3, 2), dtype=np.float32)
    stress = np.zeros((2, 3, 6), dtype=np.float32)
    stress[1, :, 0] = 300e6  # sigma_xx
    stress[1, :, 1] = 60e6  # sigma_yy
    stress[1, :, 3] = 30e6  # sigma_xy
    peeq = np.zeros((2, 3), dtype=np.float32)
    peeq[1] = 0.4
    energy = np.zeros((2, 3), dtype=np.float32)
    energy[1] = 0.02
    density = np.full((2, 3), 8900.0, dtype=np.float32)
    density[1] += 45.0
    case = Case(
        metadata=Metadata(case_id="T-state", dimension=2, source_units="g-mm-ms"),
        nodes=Nodes(coords=coords, node_id=np.arange(1, 4, dtype=np.int64)),
        elements={
            "sph": ElementBlock(
                connectivity=np.arange(3, dtype=np.int64).reshape(3, 1),
                element_id=np.arange(1, 4, dtype=np.int64),
                part_id=np.ones(3, dtype=np.int64),
            ),
        },
        materials=[Material(1, "MAT_ELASTIC_PLASTIC_HYDRO", {"data": [[1]]}, None)],
        response=Response(
            time=np.array([0.0, 2e-6]),
            node={
                "displacement": disp,
                "velocity": np.zeros((2, 3, 2), dtype=np.float32),
            },
            element={
                "sph": {
                    "stress": stress,
                    "effective_plastic_strain": peeq,
                    "internal_energy": energy,
                    "density": density,
                }
            },
        ),
    )
    path = tmp_path / "T-state.h5"
    write_case(case, path)
    return path


def test_multi_field_load_concatenates_declared_channels(tmp_path):
    path = _state_case(tmp_path)
    tr = load_case_trajectory(
        path,
        aux_field=["von_mises_stress", "deviatoric_stress_2d", "density"],
    )
    assert tr.aux.shape == (2, 3, 5)
    # deviator channels (MPa): tr3 = (300+60+0)/3 = 120
    np.testing.assert_allclose(tr.aux[1, 0, 1:4], [180.0, -60.0, 30.0], rtol=1e-5)
    # vm channel is consistent with the deviator: sqrt(3/2 s:s), s_zz = -tr3
    s = tr.aux[1, 0, 1:4].astype(float)
    s_zz = -(s[0] + s[1])
    vm = np.sqrt(1.5 * (s[0] ** 2 + s[1] ** 2 + s_zz**2 + 2 * s[2] ** 2))
    np.testing.assert_allclose(tr.aux[1, 0, 0], vm, rtol=1e-5)
    # density is unscaled SI
    np.testing.assert_allclose(tr.aux[1, :, 4], 8945.0, rtol=1e-6)


def test_single_string_stays_one_channel(tmp_path):
    tr = load_case_trajectory(_state_case(tmp_path), aux_field="internal_energy")
    assert tr.aux.shape == (2, 3, 1)
    np.testing.assert_allclose(tr.aux[1, :, 0], 0.02, rtol=1e-6)


def test_channel_helpers_flatten_labels_units():
    fields = ["deviatoric_stress_2d", "effective_plastic_strain"]
    assert as_aux_fields("von_mises_stress") == ("von_mises_stress",)
    assert aux_channel_count(fields) == 4
    assert aux_channel_labels(fields) == (
        "s_xx",
        "s_yy",
        "s_xy",
        "effective_plastic_strain",
    )
    assert aux_channel_units(fields) == ("MPa", "MPa", "MPa", "-")
    # mesh-path (unregistered) names pass through as one stress channel
    assert aux_channel_labels(["von_mises_stress_node"]) == ("von_mises_stress_node",)
    assert aux_channel_units(["von_mises_stress_node"]) == ("MPa",)
    with pytest.raises(ValueError, match="at least one"):
        as_aux_fields([])


def _traj(aux):
    T, P = aux.shape[:2]
    return CaseTrajectory(
        case_id="synth",
        positions=np.linspace(0, 1, T * P * 2, dtype=np.float32).reshape(T, P, 2),
        particle_type=np.zeros(P, dtype=np.int64),
        aux=np.asarray(aux, dtype=np.float32),
        time=np.arange(T, dtype=np.float64),
    )


def test_per_channel_stats_match_per_field_scalars():
    rng = np.random.default_rng(3)
    aux = rng.normal([[10.0, -3.0]], [[2.0, 0.5]], size=(6, 4, 2))
    tr = _traj(aux)
    stats = compute_stats([tr])
    assert stats.aux_mean.shape == (2,)
    for c in range(2):
        solo = compute_stats([_traj(aux[..., c : c + 1])])
        np.testing.assert_allclose(stats.aux_mean[c], solo.aux_mean[0])
        np.testing.assert_allclose(stats.aux_std[c], solo.aux_std[0])


def test_per_channel_transform_applies_channelwise():
    aux = np.stack([np.full((4, 3), 5.0), np.full((4, 3), 7.0)], axis=-1)
    stats = compute_stats(
        [_traj(aux)],
        aux_transform=("asinh", "none"),
        aux_transform_scale=(1.0, 1.0),
    )
    np.testing.assert_allclose(stats.aux_mean[0], np.arcsinh(5.0), rtol=1e-12)
    np.testing.assert_allclose(stats.aux_mean[1], 7.0, rtol=1e-12)


def test_expand_aux_knob_scalar_and_length_check():
    assert expand_aux_knob(0.5, 3) == (0.5, 0.5, 0.5)
    assert expand_aux_knob("asinh", 2) == ("asinh", "asinh")
    assert expand_aux_knob((1.0, 2.0), 2) == (1.0, 2.0)
    with pytest.raises(ValueError, match="2 entries for 3"):
        expand_aux_knob((1.0, 2.0), 3)


def test_window_dataset_yields_channel_blocks():
    aux = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    tr = _traj(aux)
    single = WindowDataset([tr], input_frames=2)[0]
    assert single["next_aux"].shape == (4, 3)
    bundled = WindowDataset([tr], input_frames=2, target_frames=2)[0]
    assert bundled["next_aux"].shape == (4, 2, 3)
    np.testing.assert_allclose(bundled["next_aux"][:, 0], aux[2])
    np.testing.assert_allclose(bundled["next_aux"][:, 1], aux[3])


def test_wall_append_zero_fills_every_channel():
    T, P, C = 3, 4, 3
    tr = CaseTrajectory(
        case_id="wally",
        positions=np.tile(
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], np.float32),
            (T, 1, 1),
        ),
        particle_type=np.zeros(P, dtype=np.int64),
        aux=np.ones((T, P, C), dtype=np.float32),
        time=np.arange(T, dtype=np.float64),
        cells=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        reference_coords=np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], np.float32
        ),
    )
    out = append_wall_nodes(
        tr, plane_x=-2.0, y_min=0.0, y_max=1.0, spacing=0.5, node_type=2
    )
    n_wall = 3
    assert out.aux.shape == (T, P + n_wall, C)
    np.testing.assert_allclose(out.aux[:, P:], 0.0)
    np.testing.assert_allclose(out.aux[:, :P], 1.0)


def test_stats_npz_round_trip_preserves_channel_shape(tmp_path):
    """ADR-0059: (C,) aux stats survive NormalizationStats save/load."""
    from structbench.datasets import NormalizationStats

    aux = np.stack([np.full((4, 3), 5.0), np.full((4, 3), 7.0)], axis=-1)
    stats = compute_stats(
        [_traj(aux)],
        aux_transform=("asinh", "none"),
        aux_transform_scale=(1.0, 1.0),
    )
    path = tmp_path / "stats.npz"
    stats.save(path)
    loaded = NormalizationStats.load(path)
    assert loaded.aux_mean.shape == (2,)
    np.testing.assert_allclose(loaded.aux_mean, stats.aux_mean)
    np.testing.assert_allclose(loaded.aux_std, stats.aux_std)
