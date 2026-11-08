"""The gradient task family's evaluation mode (ADR-0076) on a tiny run.

Eleven tiny free-flight cases along one Taylor line (``T-20-80-<v>``) and a
tiny untrained CGN: the numbers mean nothing, the contract does — every case
on the line is rolled out from seeds at ``v ± δ`` for both deltas, the
reference sensitivities come from the stored cases with their judged flags,
the headline pools the scored sets, and the report lands in the run dir.
"""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from structbench.benchmarks import get_benchmark
from structbench.cli.train import (
    CGNConfig,
    TrainConfig,
    build_simulator,
    evaluate_gradient,
)
from structbench.core import (
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
    write_case,
)
from structbench.datasets import compute_stats, load_case_trajectory

SMALL_CGN = {
    "input_frames": 3,
    "connectivity_radius": 2.0,
    "hidden_dim": 8,
    "message_passing_steps": 1,
    "nmlp_layers": 1,
    "particle_type_embedding_size": 4,
    "noise_std": 0.1,
    "dim": 2,
}
LINE = [f"T-20-80-{v}" for v in range(100, 201, 10)]


def _write_free_flight_case(data_root, case_id, v_m_s, n_frames=8):
    """Three SPH particles in free flight toward −x at ``v_m_s`` (SI), a shell
    node for the wall, a stress that grows so the aux std is non-degenerate."""
    coords = np.array([[0.0, 0.0], [1e-3, 0.0], [0.0, 1e-3], [5e-3, 5e-3]])
    t = np.arange(n_frames, dtype=np.float64) * 2e-6
    disp = np.zeros((n_frames, 4, 2), dtype=np.float32)
    for p in range(3):
        disp[:, p, 0] = -v_m_s * t
    stress = np.zeros((n_frames, 3, 6), dtype=np.float32)
    stress[:, :, 0] = 100e6 + 5e6 * np.arange(n_frames)[:, None] * (v_m_s / 100.0)
    case = Case(
        metadata=Metadata(case_id=case_id, dimension=2, source_units="g-mm-ms"),
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
            time=t,
            node={"displacement": disp},
            element={"sph": {"stress": stress}},
        ),
    )
    write_case(case, data_root / f"{case_id}.h5")


def _tiny_run(tmp_path, benchmark="taylor_impact_2d"):
    data_root = tmp_path / "data"
    data_root.mkdir()
    for cid in LINE:
        _write_free_flight_case(data_root, cid, float(cid.rsplit("-", 1)[1]))
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    trajs = [load_case_trajectory(data_root / f"{cid}.h5") for cid in LINE[:3]]
    stats = compute_stats(trajs)
    stats.save(out_dir / "normalization_stats.npz")
    cgn = CGNConfig(**SMALL_CGN)
    stats_t = {
        key: {
            "mean": torch.tensor(getattr(stats, f"{key}_mean"), dtype=torch.float32),
            "std": torch.tensor(getattr(stats, f"{key}_std"), dtype=torch.float32),
        }
        for key in ("velocity", "acceleration", "aux")
    }
    # The checkpoint must match what evaluate_gradient rebuilds from the run
    # record: a boundary feature only where the benchmark declares one.
    has_wall = get_benchmark(benchmark).boundary_feature_fn is not None
    simulator = build_simulator(
        stats_t,
        cgn,
        n_particle_types=2,
        boundary_feature_fn=(lambda p: p[:, 0:1]) if has_wall else None,
        device="cpu",
    )
    simulator.save(str(out_dir / "model-best-000002.pt"))
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "run": {"benchmark": benchmark, "seed": 0, "commit": "test"},
                "model": {"family": "cgn", **asdict(cgn)},
                "train": {
                    k: v
                    for k, v in asdict(TrainConfig()).items()
                    if k not in ("benchmark", "seed")
                },
                "protocol": {
                    "input_frames": SMALL_CGN["input_frames"],
                    "horizon": "full",
                    "eval_times": "native",
                    "standard": True,
                },
                "n_particle_types": 2,
                "data_root": str(data_root),
            }
        ),
        encoding="utf-8",
    )
    return data_root, out_dir


def test_gradient_mode_reports_one_line_against_its_own_differences(tmp_path):
    data_root, out_dir = _tiny_run(tmp_path)
    report = evaluate_gradient(data_root, out_dir, "cpu", case_ids=LINE)
    spec = get_benchmark("taylor_impact_2d")
    fam = spec.gradient_family
    assert report["parameter"] == fam.parameter
    assert report["deltas"] == list(fam.deltas)
    assert list(report["lines"]) == ["L80"]  # the other lines got no model values
    line = report["lines"]["L80"]
    assert line["values"] == [float(v) for v in range(100, 201, 10)]
    assert line["sets"][0] == "train" and line["sets"][-1] == "test_extrap"
    for qoi in fam.noise:
        ref = line["reference"][qoi]
        assert len(ref) == 11
        assert ref[0]["one_sided"] and ref[-1]["one_sided"]
        assert all(isinstance(p["judged"], bool) for p in ref)
        assert all(m is not None and np.isfinite(m) for m in line["model"][qoi])
        assert all(f is not None and f >= 0 for f in line["floor"][qoi])
    # headline pools the scored sets' points (130, 170, 200): three per quantity
    for qoi in fam.noise:
        n = report["metrics"][f"gradient_n_judged_{qoi}"]
        n += report["metrics"][f"gradient_n_reported_{qoi}"]
        assert n == 3
        n_all = report["metrics_all"][f"gradient_n_judged_{qoi}"]
        n_all += report["metrics_all"][f"gradient_n_reported_{qoi}"]
        assert n_all == 11
    # the free-flight reference: the tiny bar never deforms, so its extents
    # do not change with velocity and the reference sensitivity is zero —
    # which the judged rule must refuse (an error can never be ≤ half of 0)
    assert all(not p["judged"] for p in line["reference"]["final_length"])
    written_path = out_dir / "metrics-gradient.json"
    written = json.loads(written_path.read_text(encoding="utf-8"))
    assert written["metrics"].keys() == report["metrics"].keys()


def test_gradient_mode_refuses_cases_off_the_sweep_and_benchmarks_without_a_family(
    tmp_path,
):
    data_root, out_dir = _tiny_run(tmp_path)
    with pytest.raises(ValueError, match="not on the benchmark's sweep"):
        evaluate_gradient(data_root, out_dir, "cpu", case_ids=["C-1"])
    (tmp_path / "wave").mkdir()
    data_root, out_dir = _tiny_run(tmp_path / "wave", benchmark="wave_propagation_1d")
    with pytest.raises(ValueError, match="declares no gradient family"):
        evaluate_gradient(data_root, out_dir, "cpu", case_ids=LINE)
