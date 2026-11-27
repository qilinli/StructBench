"""Several runs beside the reference on one case (``viz.compare``), the
several-panel drawing under it (``viz.fringe.compare_panels``), and the
rigid planes a definition declares."""

import json
from dataclasses import replace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from PIL import Image  # noqa: E402

from structbench.benchmarks import BenchmarkSpec, RigidPlane  # noqa: E402
from structbench.benchmarks.card import BenchmarkCard  # noqa: E402
from structbench.core import (  # noqa: E402
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
    write_case,
)
from structbench.viz import Panel, PlaneTrack, compare_panels  # noqa: E402
from structbench.viz import compare as compare_module  # noqa: E402
from structbench.viz.__main__ import main  # noqa: E402
from structbench.viz.compare import RunView, compare_runs, pick_cases  # noqa: E402
from structbench.viz.fringe import _plane_band  # noqa: E402

T, NX, NY = 9, 3, 2  # frames; a 3 x 2 quad grid and one node no element uses
PLATE = 2
AUX_FIELDS = ("stress_axisymmetric", "effective_plastic_strain")
CASE = "val_0000"


def _grid():
    xs, ys = np.meshgrid(np.arange(NX + 1), np.arange(NY + 1), indexing="ij")
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)  # mm
    cells = []
    for i in range(NX):
        for j in range(NY):
            n0 = i * (NY + 1) + j
            cells.append([n0, n0 + NY + 1, n0 + NY + 2, n0 + 1])
    coords = np.concatenate([coords, [[0.0, 4.0]]])  # the plate's node, above
    return coords, np.asarray(cells, dtype=np.int64)


def _name_plate(traj):
    types = np.zeros(traj.positions.shape[1], dtype=np.int64)
    types[np.setdiff1d(np.arange(len(types)), np.unique(traj.cells))] = PLATE
    return replace(traj, particle_type=types)


def _write_case(root):
    rng = np.random.default_rng(3)
    coords, cells = _grid()
    n_nodes, n_cells = len(coords), len(cells)
    disp = np.zeros((T, n_nodes, 2), dtype=np.float32)
    disp[:, -1, 1] = -np.arange(T) * 0.1e-3  # the plate comes down 0.1 mm a frame
    stress = np.zeros((T, n_cells, 6), dtype=np.float32)
    stress[:, :, :4] = rng.normal(size=(T, n_cells, 4)) * 20e6
    peeq = np.cumsum(rng.random((T, n_cells)) * 1e-3, axis=0).astype(np.float32)
    case = Case(
        metadata=Metadata(case_id=CASE, dimension=2, source_units="t-mm-s"),
        nodes=Nodes(
            coords=coords * 1e-3,
            node_id=np.arange(1, n_nodes + 1, dtype=np.int64),
            node_type=np.zeros(n_nodes, dtype=np.int64),
        ),
        elements={
            "solid": ElementBlock(
                connectivity=cells,
                element_id=np.arange(1, n_cells + 1, dtype=np.int64),
                part_id=np.ones(n_cells, dtype=np.int64),
            )
        },
        materials=[Material(1, "elastic_plastic_isotropic", {}, None)],
        response=Response(
            time=np.arange(T, dtype=np.float64) * 1e-6,
            node={"displacement": disp},
            element={"solid": {"stress": stress, "effective_plastic_strain": peeq}},
        ),
    )
    write_case(case, root / f"{CASE}.h5")
    return peeq


def _spec(**changes):
    card = BenchmarkCard(
        name="CompareFixture",
        version="0.0",
        description="synthetic comparison fixture",
        provenance="synthetic (test fixture)",
        data_license="n/a (synthetic test data)",
        solver="synthetic",
        discretisation="FEM",
        materials=("elastic_plastic_isotropic",),
        loading="synthetic",
        erosion=False,
        source_units="t-mm-s",
        geometry="synthetic quads",
        n_cases=2,
        splits={"train": 1, "val": 1},
        task="fixture",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=(),
        fields=("node/displacement", "element/solid/stress"),
        particles_per_case="13-13",
        n_frames=T,
        output_dt_ms=0.001,
        input_frames=2,
        protocol_rationale="synthetic fixture; not a benchmark",
    )
    spec = BenchmarkSpec(
        card=card,
        splits={"train": ("train_0000",), "val": (CASE,)},
        eval_splits=("val",),
        aux_field="von_mises_stress",
        qois={},
        boundary_feature_fn=None,
        dataset_id="compare-fixture",
        mesh_transform=_name_plate,
        scored_frames=7,
        state_site="element",
        state_channels=AUX_FIELDS,
        element_formulation="plane_strain",
        property_tolerances={"yield": 1e-5, "peeq": 1e-7},
        rigid_planes=(
            RigidPlane((0.0, 1.0), offset=0.0, label="floor"),
            RigidPlane((0.0, -1.0), node_type=PLATE, label="plate"),
        ),
    )
    return replace(spec, **changes) if changes else spec


def _run(root, name, *, error, plate_speed, with_state):
    """A run directory as evaluation leaves it: one stored rollout, one
    metrics file. The run's plate comes down at its own speed."""
    coords, cells = _grid()
    run = root / name
    (run / "rollouts").mkdir(parents=True)
    positions = np.repeat(coords[None], T, axis=0).astype(np.float32)
    positions[:, -1, 1] -= np.arange(T) * plate_speed
    aux = np.zeros((T, len(coords), 5), dtype=np.float32)
    aux[..., 4] = np.arange(len(coords))[None] * 1e-3  # plastic strain, nodal
    arrays = {
        "predicted_positions": positions,
        "predicted_aux": aux,
        "position_rmse": np.zeros(T - 2),
        "aux_rmse": np.zeros(T - 2),
        "input_frames": np.int64(2),
        "scored_frames": np.int64(7),
    }
    if with_state:
        state = np.zeros((T, len(cells), 5), dtype=np.float32)
        state[..., 4] = 0.5
        arrays["predicted_state"] = state
    np.savez(run / "rollouts" / f"val-{CASE}.npz", **arrays)
    (run / "metrics-val.json").write_text(
        json.dumps({"cases": {CASE: {"rollout_rel_l2_displacement": error}}})
    )
    return run


def _view(run, label, spec):
    return RunView(run, label, spec, {"train": {"aux_fields": list(AUX_FIELDS)}})


def test_a_rigid_plane_is_fixed_or_moves_with_a_node():
    assert RigidPlane((0, 1), offset=0.0).normal == (0.0, 1.0)
    assert RigidPlane((1, 0), node_type=3).node_type == 3
    with pytest.raises(ValueError, match="unit vector"):
        RigidPlane((0.0, 2.0), offset=0.0)
    with pytest.raises(ValueError, match="exactly one"):
        RigidPlane((0.0, 1.0))
    with pytest.raises(ValueError, match="exactly one"):
        RigidPlane((0.0, 1.0), offset=0.0, node_type=1)
    with pytest.raises(ValueError, match="must be an int"):
        RigidPlane((0.0, 1.0), node_type=1.5)
    with pytest.raises(ValueError, match="one moving plane per node type"):
        _spec(
            rigid_planes=(
                RigidPlane((0.0, 1.0), node_type=2),
                RigidPlane((1.0, 0.0), node_type=2),
            )
        )
    with pytest.raises(ValueError, match="share one dimension"):
        _spec(
            rigid_planes=(
                RigidPlane((0.0, 1.0), offset=0.0),
                RigidPlane((0.0, 0.0, 1.0), offset=0.0),
            )
        )
    assert _spec(rigid_planes=()).rigid_planes == ()


def test_the_band_lies_behind_the_plane_and_follows_it():
    floor = PlaneTrack((0.0, 1.0), 0.0)
    band = _plane_band(floor, 0, reach=10.0, width=1.5)
    assert sorted(set(band[:, 1].round(9))) == [-1.5, 0.0]  # below y = 0
    assert band[:, 0].min() == -10.0 and band[:, 0].max() == 10.0
    plate = PlaneTrack((0.0, -1.0), np.array([-4.0, -3.5]))  # y = 4, then 3.5
    assert plate.at(1) == -3.5
    assert sorted(set(_plane_band(plate, 0, 10.0, 1.5)[:, 1].round(9))) == [4.0, 5.5]
    assert sorted(set(_plane_band(plate, 1, 10.0, 1.5)[:, 1].round(9))) == [3.5, 5.0]
    wall = PlaneTrack((1.0, 0.0), -2.0)  # the wall at x = -2, body to its left
    assert sorted(set(_plane_band(wall, 0, 10.0, 1.5)[:, 0].round(9))) == [-3.5, -2.0]


def test_compare_panels_draws_elements_and_planes(tmp_path):
    coords, cells = _grid()
    positions = np.repeat(coords[None], T, axis=0)
    values = np.linspace(0.0, 1.0, T * len(cells)).reshape(T, len(cells))
    planes = (PlaneTrack((0.0, 1.0), 0.0, "floor"),)
    panels = [
        Panel("Reference", positions, values, planes),
        Panel("Run", positions, values * 2.0, planes),
        Panel("Other", positions, values * 0.5, planes),
    ]
    gif = compare_panels(
        panels,
        tmp_path / "c.gif",
        cells=cells,
        field="effective_plastic_strain",
        times_us=np.arange(T) * 500.0,
        final_path=tmp_path / "c.png",
        max_frames=4,
    )
    with Image.open(gif) as image:
        # every third frame of nine, and the last: 0, 3, 6, 8
        assert image.n_frames == 4
    assert (tmp_path / "c.png").stat().st_size > 0
    with pytest.raises(ValueError, match="one value per element"):
        compare_panels(
            [Panel("Reference", positions, np.zeros((T, 99)))],
            tmp_path / "bad.gif",
            cells=cells,
        )
    with pytest.raises(ValueError, match="at least one panel"):
        compare_panels([], tmp_path / "none.gif")
    # without connectivity the same panels are points, one value per node
    nodal = np.arange(T * len(coords), dtype=float).reshape(T, len(coords))
    points = compare_panels(
        [Panel("Reference", positions, nodal), Panel("Run", positions, nodal)],
        tmp_path / "p.gif",
    )
    with Image.open(points) as image:
        assert image.n_frames == T


def test_pick_cases_takes_the_case_at_a_rank_of_the_mean_error(tmp_path):
    errors = {
        "a": {"c1": 0.1, "c2": 0.5, "c3": 0.3, "c4": 0.9, "only_a": 0.0},
        "b": {"c1": 0.3, "c2": 0.1, "c3": 0.5, "c4": 0.9},
    }
    runs = []
    for name, per_case in errors.items():
        run = tmp_path / name
        run.mkdir()
        (run / "metrics-test.json").write_text(
            json.dumps(
                {
                    "cases": {
                        c: {"rollout_rel_l2_displacement": e}
                        for c, e in per_case.items()
                    }
                }
            )
        )
        runs.append(run)
    # means: c1 0.2, c2 0.3, c3 0.4, c4 0.9; a case one run lacks is left out
    assert pick_cases(runs, ["test"]) == [("test", "c3")]
    assert pick_cases(runs, ["test"], quantile=0.0) == [("test", "c1")]
    assert pick_cases(runs, ["test"], quantile=1.0) == [("test", "c4")]
    with pytest.raises(ValueError, match=r"quantile must be in \[0, 1\]"):
        pick_cases(runs, ["test"], quantile=1.5)
    with pytest.raises(ValueError, match="no case"):
        pick_cases(runs, ["val"])


def test_compare_runs_puts_each_run_beside_the_reference(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    stored_peeq = _write_case(data)
    spec = _spec()
    two_site = _run(tmp_path, "two", error=0.125, plate_speed=0.2, with_state=True)
    nodal = _run(tmp_path, "nodal", error=0.25, plate_speed=0.05, with_state=False)
    views = [_view(two_site, "Two-site", spec), _view(nodal, "Nodal", spec)]

    seen = {}
    real = compare_module.compare_panels

    def recording(panels, out_path, **kwargs):
        seen["panels"], seen["kwargs"] = panels, kwargs
        return real(panels, out_path, **kwargs)

    monkeypatch.setattr(compare_module, "compare_panels", recording)
    gif, final = compare_runs(
        views, "val", CASE, data, tmp_path / "out", channel="effective_plastic_strain"
    )
    assert gif.name == f"{CASE}.gif" and final.name == f"{CASE}-final.png"
    assert gif.stat().st_size > 0 and final.stat().st_size > 0

    reference, first, second = seen["panels"]
    _, cells = _grid()
    # the clock ends at the scored bound, not at the stored record's end
    assert reference.positions.shape[0] == 7
    # the reference shows the stored element state
    np.testing.assert_allclose(reference.values, stored_peeq[:7], rtol=1e-6)
    # a run that stored its element state shows it; another the mean of its
    # nodal values over each element's nodes
    assert first.values.shape == (7, len(cells)) and np.all(first.values == 0.5)
    np.testing.assert_allclose(second.values[0], cells.mean(axis=1) * 1e-3, rtol=1e-5)
    assert first.title == "Two-site\ndisp. error 0.125"
    assert second.title == "Nodal\ndisp. error 0.250"
    # drawn by element, the scale the reference's
    assert seen["kwargs"]["cells"].shape == cells.shape
    assert seen["kwargs"]["field"].key == "effective_plastic_strain"
    # the floor stays; the plate is where each panel's own node puts it
    for panel, speed in ((reference, 0.1), (first, 0.2), (second, 0.05)):
        floor, plate = panel.planes
        assert floor.offset == 0.0 and floor.label == "floor"
        np.testing.assert_allclose(
            plate.offset, -(4.0 - np.arange(7) * speed), atol=1e-5
        )

    # a percentile of the reference's last shown frame caps the scale
    compare_runs(
        views,
        "val",
        CASE,
        data,
        tmp_path / "out",
        channel="effective_plastic_strain",
        vmax_percentile=50.0,
    )
    assert seen["kwargs"]["vmax"] == pytest.approx(np.percentile(stored_peeq[6], 50.0))

    with pytest.raises(ValueError, match="no run carries the channel 'nope'"):
        compare_runs(views, "val", CASE, data, tmp_path / "out", channel="nope")
    with pytest.raises(FileNotFoundError, match="no stored rollout"):
        compare_runs(views, "test", CASE, data, tmp_path / "out")
    # without a channel: the headline field if a run carries it, else the
    # first run's first channel
    compare_runs(views, "val", CASE, data, tmp_path / "out")
    assert seen["kwargs"]["field"].key == "s_rr"
    with pytest.raises(ValueError, match="the case has 0 such nodes"):
        compare_runs(
            [_view(two_site, "Two-site", _spec(mesh_transform=None))],
            "val",
            CASE,
            data,
            tmp_path / "out",
            channel="effective_plastic_strain",
        )


def test_the_command_line_picks_and_says_what_it_needs(tmp_path, capsys):
    runs = []
    for name, error in (("a", 0.1), ("b", 0.3)):
        runs.append(
            _run(tmp_path, name, error=error, plate_speed=0.1, with_state=False)
        )
    base = ["--run", str(runs[0]), "--run", str(runs[1]), "--data-root", str(tmp_path)]
    main([*base, "--split", "val", "--pick-only"])
    assert f"picked val: {CASE}" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="give --case with --split"):
        main(base)
    with pytest.raises(SystemExit, match="--case needs its --split"):
        main([*base, "--case", CASE])
    with pytest.raises(SystemExit, match="needs --out"):
        main([*base, "--split", "val", "--case", CASE])
