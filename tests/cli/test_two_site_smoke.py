"""End-to-end two-site smoke (ADR-0075 plan 3a): a synthetic plane-strain quad
mesh benchmark with an element-site state block, trained with the anchored
flow map under both two-site decoders, then evaluated. The gate: the
return-map arm's violation rates read zero at tolerance along the rollout,
the corner-mean arm's state is scored the same way, and every record and
artifact carries the state beside the nodal view."""

import json

import numpy as np
import pytest
import torch

from structbench.benchmarks import ConstitutiveLaw
from structbench.benchmarks.card import BenchmarkCard
from structbench.benchmarks.registry import BenchmarkSpec
from structbench.cli.train import train
from structbench.config import TrainConfig, TransolverConfig
from structbench.core import (
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
    write_case,
)
from structbench.eval import terminal_peak_displacement
from structbench.models.common.two_site import sigma_y_rows, von_mises4

T, NX, NY = 8, 3, 2  # frames; a 3 x 2 quad grid: 12 nodes, 6 elements
STATE_CHANNELS = ("stress_axisymmetric", "effective_plastic_strain")
LAW = ConstitutiveLaw(124_000.0, 0.34, ((0.0, 10.0), (200.0, 10_200.0)))


def _grid():
    xs, ys = np.meshgrid(np.arange(NX + 1), np.arange(NY + 1), indexing="ij")
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)  # mm
    cells = []
    for i in range(NX):
        for j in range(NY):
            n0 = i * (NY + 1) + j
            cells.append([n0, n0 + NY + 1, n0 + NY + 2, n0 + 1])
    node_type = np.where(coords[:, 1] == 0.0, 1, 0).astype(np.int64)  # bottom row fixed
    return coords, np.asarray(cells, dtype=np.int64), node_type


def _write_cases(root, ids):
    rng = np.random.default_rng(5)
    coords, cells, node_type = _grid()
    n_nodes, n_cells = len(coords), len(cells)
    for cid in ids:
        disp = np.zeros((T, n_nodes, 2), dtype=np.float32)
        drift = rng.normal(size=(T, n_nodes, 2)).astype(np.float32) * 2e-6  # m
        drift[:, node_type == 1] = 0.0
        disp[1:] = np.cumsum(drift, axis=0)[1:]
        stress = np.zeros((T, n_cells, 6), dtype=np.float32)
        stress[:, :, :4] = rng.normal(size=(T, n_cells, 4)) * 20e6  # Pa, inside yield
        peeq = np.cumsum(rng.random((T, n_cells)) * 1e-3, axis=0).astype(np.float32)
        case = Case(
            metadata=Metadata(case_id=cid, dimension=2, source_units="t-mm-s"),
            nodes=Nodes(
                coords=coords * 1e-3,
                node_id=np.arange(1, n_nodes + 1, dtype=np.int64),
                node_type=node_type,
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
        write_case(case, root / f"{cid}.h5")


def _spec(ids):
    qois = {"terminal_peak_deflection": terminal_peak_displacement(exclude_types=(1,))}
    card = BenchmarkCard(
        name="TwoSiteSmoke",
        version="0.0",
        description="synthetic two-site smoke benchmark",
        provenance="synthetic (test fixture)",
        data_license="n/a (synthetic test data)",
        solver="synthetic",
        discretisation="FEM",
        materials=("elastic_plastic_isotropic",),
        loading="synthetic",
        erosion=False,
        source_units="t-mm-s",
        geometry="synthetic quads",
        n_cases=sum(len(v) for v in ids.values()),
        splits={k: len(v) for k, v in ids.items()},
        task="smoke",
        aux_field="von_mises_stress",
        aux_unit="MPa",
        qois=tuple(qois),
        fields=("node/displacement", "element/solid/stress"),
        particles_per_case="12-12",
        n_frames=T,
        output_dt_ms=0.001,
        input_frames=2,
        protocol_rationale="synthetic smoke fixture; not a benchmark",
    )
    return BenchmarkSpec(
        card=card,
        splits={k: tuple(v) for k, v in ids.items()},
        eval_splits=("val",),
        aux_field="von_mises_stress",
        qois=qois,
        boundary_feature_fn=None,
        dataset_id="two-site-smoke",
        kinematic_types=(1,),
        scripted_types=(1,),
        state_site="element",
        state_channels=STATE_CHANNELS,
        element_formulation="plane_strain",
        constitutive_law=lambda cid: LAW,
    )


def _run(tmp_path, decoder: str):
    torch.manual_seed(0)
    ids = {"train": [f"train_{i:04d}" for i in range(3)], "val": ["val_0000"]}
    spec = _spec(ids)
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_cases(data_root, [c for v in ids.values() for c in v])
    cfg = TransolverConfig(
        dim=2,
        node_type_size=2,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        slice_num=8,
        normalizer_warmup_steps=5,
        noise_std=0.0,
        time_conditioned=True,
        aux_input=True,
        flow_map=True,
        flow_map_eval_intervals=(1, 3),
        flow_map_canonical_interval=3,
        two_site_decoder=decoder,
    )
    tcfg = TrainConfig(
        benchmark="TwoSiteSmoke",
        batch_size=2,
        training_steps=20,
        val_every=10,
        aux_fields=STATE_CHANNELS,
    )
    out = tmp_path / "run"
    train(spec, cfg, tcfg, data_root, out, "cpu", family="transolver")
    return spec, data_root, out, ids


@pytest.mark.parametrize("decoder", ["return_map", "corner_mean"])
def test_two_site_train_and_evaluate_smoke(tmp_path, monkeypatch, decoder):
    import structbench.cli.train as cli_train

    spec, data_root, out, ids = _run(tmp_path, decoder)
    record = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert record["model"]["two_site_decoder"] == decoder
    assert any(p.name.startswith("model-best-") for p in out.glob("model-*.pt"))

    monkeypatch.setattr(cli_train, "get_benchmark", lambda name: spec)
    metrics = cli_train.evaluate(ids["val"], data_root, out, "cpu", split_name="val")
    case = metrics["cases"][ids["val"][0]]
    labels = ("s_rr", "s_zz", "s_tt", "s_rz", "effective_plastic_strain")
    assert np.isfinite(case["rollout_rel_l2_state"]) and np.isfinite(
        case["rollout_state_rmse"]
    )
    assert tuple(case["rollout_rel_l2_state_channels"]) == labels
    assert tuple(case["rollout_state_rmse_channels"]) == labels
    # Per interval, both anchor modes, the state rides the flow-map prefixes.
    for prefix in (
        "rollout_m1",
        "rollout_oracle_m1",
        "rollout_m3",
        "rollout_oracle_m3",
    ):
        assert np.isfinite(case[f"{prefix}_rel_l2_state"])
        assert tuple(case[f"{prefix}_rel_l2_state_channels"]) == labels
    assert np.isfinite(metrics["mean"]["rollout_rel_l2_state"])
    assert np.isfinite(metrics["mean"]["rollout_m3_state_rmse"])

    npz = np.load(out / "rollouts" / f"val-{ids['val'][0]}.npz")
    state = npz["predicted_state"]
    assert state.shape == (T, NX * NY, 5)
    assert npz["state_rmse"].shape == (T - 2,)
    if decoder == "return_map":
        # By construction along the whole rollout: von Mises at or below
        # sigma_y(peeq) and peeq never below the seed state.
        st = torch.from_numpy(state)
        knots_x = torch.tensor([LAW.hardening[0]] * (NX * NY))
        knots_y = torch.tensor([LAW.hardening[1]] * (NX * NY))
        for f in range(2, T):
            sy, _ = sigma_y_rows(st[f, :, 4], knots_x, knots_y)
            assert bool((von_mises4(st[f, :, :4]) <= sy * (1 + 1e-5)).all()), f
            assert bool((st[f, :, 4] >= st[1, :, 4] - 1e-7).all()), f
