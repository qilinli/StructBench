import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from structbench.core import (  # noqa: E402
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
    write_case,
)
from structbench.viz import (  # noqa: E402
    FIELDS,
    FieldSpec,
    animate_rollout,
    compare_rollout,
    fringe_scatter,
    load_case_field,
    snapshot,
)
from structbench.viz.__main__ import snapshot_frames, split_and_case  # noqa: E402

RNG = np.random.default_rng(0)
T, P = 3, 40
POSITIONS = RNG.uniform(0.0, 10.0, size=(T, P, 2))
VALUES = RNG.uniform(0.0, 300.0, size=(T, P))


def test_field_registry_is_self_consistent():
    for key, spec in FIELDS.items():
        assert spec.key == key
        assert spec.label
    assert FIELDS["von_mises_stress"].bar_label == "von Mises stress (MPa)"
    assert FIELDS["effective_plastic_strain"].bar_label == "effective plastic strain"


def test_unknown_field_raises_with_known_list():
    with pytest.raises(KeyError, match="von_mises_stress"):
        snapshot(POSITIONS[0], VALUES[0], field="not_a_field")


def test_snapshot_uses_fem_fringe_conventions():
    fig = snapshot(POSITIONS[0], VALUES[0], title="case", time_us=12.0, wall_x=-2.0)
    sc = fig.axes[0].collections[0]
    assert sc.get_cmap().name == "jet"
    assert sc.get_clim()[0] == 0.0  # zero floor for a non-negative field
    assert fig.axes[0].get_aspect() == 1.0  # equal aspect
    assert "von Mises stress (MPa)" in fig.axes[-1].get_ylabel()
    matplotlib.pyplot.close(fig)


def test_negative_fields_keep_true_minimum():
    ax = matplotlib.pyplot.figure().add_subplot()
    values = np.array([-5.0, 0.0, 5.0])
    sc = fringe_scatter(ax, np.zeros((3, 2)), values, field="pressure")
    assert sc.get_clim() == (-5.0, 5.0)
    matplotlib.pyplot.close(ax.figure)


def test_banded_fringe_is_discrete():
    ax = matplotlib.pyplot.figure().add_subplot()
    sc = fringe_scatter(ax, POSITIONS[0], VALUES[0], bands=9)
    assert sc.get_cmap().N == 9
    matplotlib.pyplot.close(ax.figure)


def test_compare_rollout_shares_scale_and_extent():
    fig = compare_rollout(
        POSITIONS,
        VALUES,
        POSITIONS + 1.0,
        VALUES * 2.0,  # prediction hotter than GT
        frames=[0, 2],
        times_us=np.array([0.0, 10.0, 20.0]),
        title="t",
        wall_x=-2.0,
    )
    panels = fig.axes[:4]
    clims = {ax.collections[0].get_clim() for ax in panels}
    assert len(clims) == 1  # GT sets one shared fringe range
    assert clims.pop()[1] == pytest.approx(VALUES[[0, 2]].max())
    assert len({ax.get_xlim() for ax in panels}) == 1
    matplotlib.pyplot.close(fig)


def test_animate_rollout_writes_gif(tmp_path):
    out = animate_rollout(
        POSITIONS, VALUES, tmp_path / "roll.gif", times_us=np.zeros(T), fps=5, dpi=50
    )
    assert out.exists() and out.stat().st_size > 0


def _sph_case(tmp_path):
    # 3 SPH particles + 1 shell node, 2 frames, SI units (mirrors
    # tests/datasets/test_canonical.py).
    coords = np.array([[0.0, 0.0], [1e-3, 0.0], [0.0, 1e-3], [5e-3, 5e-3]])
    disp = np.zeros((2, 4, 2), dtype=np.float32)
    disp[1, :3, 0] = 2e-3
    stress = np.zeros((2, 3, 6), dtype=np.float32)
    stress[1, :, 0] = 300e6
    eps = np.zeros((2, 3), dtype=np.float32)
    eps[1] = 0.25
    case = Case(
        metadata=Metadata(case_id="T-test", dimension=2, source_units="g-mm-ms"),
        nodes=Nodes(coords=coords, node_id=np.arange(1, 5, dtype=np.int64)),
        elements={
            "sph": ElementBlock(
                connectivity=np.arange(3, dtype=np.int64).reshape(3, 1),
                element_id=np.arange(1, 4, dtype=np.int64),
                part_id=np.ones(3, dtype=np.int64),
            ),
        },
        materials=[Material(2, "MAT_ELASTIC_PLASTIC_HYDRO", {"data": [[2]]}, None)],
        response=Response(
            time=np.array([0.0, 2e-6]),
            node={"displacement": disp},
            element={"sph": {"stress": stress, "effective_plastic_strain": eps}},
        ),
    )
    path = tmp_path / "case.h5"
    write_case(case, path)
    return path


def test_load_case_field_von_mises_and_strain(tmp_path):
    path = _sph_case(tmp_path)
    vm = load_case_field(path, "von_mises_stress")
    assert vm.positions.shape == (2, 3, 2)
    np.testing.assert_allclose(vm.values[1], 300.0)  # MPa
    np.testing.assert_allclose(vm.times_us, [0.0, 2.0])

    eps = load_case_field(path, "effective_plastic_strain")
    np.testing.assert_allclose(eps.values[1], 0.25)

    disp = load_case_field(path, "displacement_magnitude")
    np.testing.assert_allclose(disp.values[1], 2.0)  # mm


def test_load_case_field_missing_data_raises(tmp_path):
    with pytest.raises(KeyError, match="velocity"):
        load_case_field(_sph_case(tmp_path), "velocity_magnitude")


def test_custom_field_spec_is_accepted():
    spec = FieldSpec("anything", "my field", "kJ", "{:.2f}")
    fig = snapshot(POSITIONS[0], VALUES[0], field=spec)
    assert "my field (kJ)" in fig.axes[-1].get_ylabel()
    matplotlib.pyplot.close(fig)


def test_cli_stem_parsing_and_frame_selection():
    assert split_and_case("test_extrap-T-20-60-200") == ("test_extrap", "T-20-60-200")
    with pytest.raises(ValueError):
        split_and_case("nonsense")
    assert snapshot_frames(152, 11, 4) == [11, 57, 104, 151]


def test_snapshot_frames_degenerate_columns():
    assert snapshot_frames(152, 11, 1) == [151]  # single column: final frame
    with pytest.raises(ValueError, match="columns"):
        snapshot_frames(152, 11, 0)


def test_compare_rollout_single_frame_column():
    fig = compare_rollout(POSITIONS, VALUES, POSITIONS, VALUES, frames=[1])
    assert len(fig.axes) == 3  # GT + prediction panels + fringe bar
    matplotlib.pyplot.close(fig)


def test_3d_trajectories_are_projected_to_xy(tmp_path):
    z = np.full((T, P, 1), 500.0)  # huge z must not leak into the extents
    pos3 = np.concatenate([POSITIONS, z], axis=-1)
    fig = compare_rollout(pos3, VALUES, pos3, VALUES, frames=[0, 2])
    assert fig.axes[0].get_ylim()[1] < 50.0
    matplotlib.pyplot.close(fig)

    out = animate_rollout(pos3, VALUES, tmp_path / "proj.gif", fps=5, dpi=40)
    assert out.exists()
