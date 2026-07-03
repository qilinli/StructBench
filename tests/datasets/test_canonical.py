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
from structbench.datasets.canonical import (
    CaseTrajectory,
    load_case_trajectory,
    von_mises_from_voigt,
)


def test_von_mises_uniaxial_equals_axial_stress():
    s = np.zeros((1, 6))
    s[0, 0] = 250.0  # pure sigma_xx
    np.testing.assert_allclose(von_mises_from_voigt(s), [250.0], rtol=1e-6)


def _sph_case(tmp_path):
    # 3 SPH particles + 1 shell node, 2 frames, SI units.
    coords = np.array([[0.0, 0.0], [1e-3, 0.0], [0.0, 1e-3], [5e-3, 5e-3]])
    disp = np.zeros((2, 4, 2), dtype=np.float32)
    disp[1, :3, 0] = 2e-3  # +2 mm in x at frame 1, SPH particles only
    stress = np.zeros((2, 3, 6), dtype=np.float32)
    stress[1, :, 0] = 300e6  # 300 MPa sigma_xx at frame 1
    effective_plastic_strain = np.zeros((2, 3), dtype=np.float32)
    effective_plastic_strain[1, :] = 0.15  # K&C scaled damage measure
    case = Case(
        metadata=Metadata(case_id="T-test", dimension=2, source_units="g-mm-ms"),
        nodes=Nodes(coords=coords, node_id=np.arange(1, 5, dtype=np.int64)),
        elements={
            "sph": ElementBlock(
                connectivity=np.arange(3, dtype=np.int64).reshape(3, 1),
                element_id=np.arange(1, 4, dtype=np.int64),
                part_id=np.ones(3, dtype=np.int64),
            ),
            "shell": ElementBlock(
                connectivity=np.array([[3, 3, 3, 3]], dtype=np.int64),
                element_id=np.array([99], dtype=np.int64),
                part_id=np.array([2], dtype=np.int64),
            ),
        },
        materials=[Material(2, "MAT_ELASTIC_PLASTIC_HYDRO", {"data": [[2]]}, None)],
        response=Response(
            time=np.array([0.0, 2e-6]),
            node={"displacement": disp},
            element={
                "sph": {
                    "stress": stress,
                    "effective_plastic_strain": effective_plastic_strain,
                }
            },
        ),
    )
    path = tmp_path / "case.h5"
    write_case(case, path)
    return path


def test_load_case_trajectory_sph_only_in_mm_and_mpa(tmp_path):
    traj = load_case_trajectory(_sph_case(tmp_path))
    assert isinstance(traj, CaseTrajectory)
    assert traj.positions.shape == (2, 3, 2)  # SPH particles only
    np.testing.assert_allclose(traj.positions[0, 1], [1.0, 0.0])  # 1 mm
    np.testing.assert_allclose(traj.positions[1, 0], [2.0, 0.0])  # +2 mm disp
    assert traj.aux.shape == (2, 3)
    np.testing.assert_allclose(traj.aux[1], [300.0, 300.0, 300.0])  # MPa
    np.testing.assert_array_equal(traj.particle_type, [1, 1, 1])


def test_n_valid_frames_drops_terminal_dt_artifact():
    from structbench.datasets.canonical import n_valid_frames

    uniform = np.array([0.0, 2e-6, 4e-6, 6e-6])
    assert n_valid_frames(uniform) == 4
    # LS-DYNA termination state a fraction of an interval after the last dump:
    artifact = np.array([0.0, 2e-6, 4e-6, 4.077e-6])
    assert n_valid_frames(artifact) == 3
    # too short to judge — keep everything
    assert n_valid_frames(np.array([0.0, 2e-6])) == 2


def test_load_case_trajectory_default_aux_is_von_mises(tmp_path):
    h5_path = _sph_case(tmp_path)
    tr = load_case_trajectory(h5_path)
    assert tr.aux.shape == tr.positions.shape[:2]


def test_load_case_trajectory_rejects_unknown_aux_field(tmp_path):
    h5_path = _sph_case(tmp_path)
    with pytest.raises(KeyError, match="von_mises_stress"):
        load_case_trajectory(h5_path, aux_field="no_such_field")


def test_available_aux_fields_lists_von_mises():
    from structbench.datasets import available_aux_fields

    assert "von_mises_stress" in available_aux_fields()


def test_axial_stress_extractor_takes_voigt_xx(tmp_path):
    h5_path = _sph_case(tmp_path)
    tr_axial = load_case_trajectory(h5_path, aux_field="axial_stress")
    import h5py

    with h5py.File(h5_path) as f:
        sxx_pa = f["response/element/sph/stress"][...][..., 0]
    np.testing.assert_allclose(tr_axial.aux, sxx_pa * 1e-6, rtol=1e-6)  # Pa -> MPa


def test_available_aux_fields_lists_axial_stress():
    from structbench.datasets import available_aux_fields

    assert "axial_stress" in available_aux_fields()


def test_damage_extractor_reads_eff_plastic_strain_unscaled(tmp_path):
    h5_path = _sph_case(tmp_path)
    tr = load_case_trajectory(h5_path, aux_field="damage")
    import h5py

    with h5py.File(h5_path) as f:
        expected = f["response/element/sph/effective_plastic_strain"][...]
    np.testing.assert_allclose(tr.aux, expected, rtol=1e-6)  # NO stress scaling


def test_available_aux_fields_lists_damage():
    from structbench.datasets import available_aux_fields

    assert "damage" in available_aux_fields()
