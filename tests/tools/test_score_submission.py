"""tools/score_submission.py (ADR-0074 clause 4; open-evaluation plan §5):
the prediction contract, the registry's metrics per case, coverage by set,
the regeneration guard and the rate limit, on tiny Taylor-named cases."""

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from structbench.benchmarks import get_benchmark
from structbench.core import (
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
    read_case,
    write_case,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "score_submission", REPO_ROOT / "tools" / "score_submission.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_submission"] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()
SPEC = get_benchmark("taylor_impact_2d")
INPUT_FRAMES = SPEC.card.input_frames  # 6
# A uniform clock, so the loader keeps every frame (it drops a terminal frame
# only when it is off the sampling interval, ADR-0028): T' = 10, scored [6, 10).
N_FRAMES = 10


def _reference_case(case_id: str, v: float) -> Case:
    coords = np.array([[0.0, 0.0], [1e-3, 0.0], [0.0, 1e-3], [5e-3, 5e-3]])
    t = np.arange(N_FRAMES, dtype=float) * 2e-6
    disp = np.zeros((N_FRAMES, 4, 2), dtype=np.float32)
    for p in range(3):
        disp[:, p, 0] = -v * t + 1e-5 * (p + 1) * np.arange(N_FRAMES)
    stress = np.zeros((N_FRAMES, 3, 6), dtype=np.float32)
    stress[:, :, 0] = 100e6 + 5e6 * np.arange(N_FRAMES)[:, None]
    return Case(
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


def _prediction(ref: Case, *, disp_error: float = 0.0, aux_factor: float = 1.0) -> Case:
    """The scored span of ``ref`` as a prediction, perturbed."""
    r = ref.response
    assert r is not None
    lo, hi = INPUT_FRAMES, N_FRAMES
    vm = np.sqrt(0.5 * ((r.element["sph"]["stress"][lo:hi, :, 0]) ** 2) * 2)  # = |s_xx|
    return Case(
        metadata=ref.metadata,
        nodes=ref.nodes,
        elements=ref.elements,
        materials=ref.materials,
        response=Response(
            time=r.time[lo:hi],
            node={"displacement": r.node["displacement"][lo:hi] + disp_error},
            element={"sph": {"von_mises_stress": (vm * aux_factor).astype(np.float32)}},
        ),
    )


_TOML = """
method = "Tiny"
family = "cgn"
scheme = "autoregressive"
division = "closed"
submitter = "Someone"
commit = "abc1234"
config_sha256 = "0"
seeds = [1]
checkpoint_sha256 = "0"
writeup = "https://example.org"
"""


def _setup(tmp_path, *, disp_error=0.0, aux_factor=1.0, toml=_TOML):
    reference, submission = tmp_path / "reference", tmp_path / "submission"
    reference.mkdir(parents=True)
    submission.mkdir(parents=True)
    cases = {"T-20-80-130": 130.0, "T-20-80-170": 170.0, "T-20-80-200": 200.0}
    for cid, v in cases.items():
        ref = _reference_case(cid, v)
        write_case(ref, reference / f"{cid}.h5")
        write_case(
            _prediction(ref, disp_error=disp_error, aux_factor=aux_factor),
            submission / f"{cid}.h5",
        )
    (submission / "submission.toml").write_text(toml, encoding="utf-8")
    return reference, submission


def test_a_faithful_prediction_scores_zero_and_trips_the_guard(tmp_path):
    reference, submission = _setup(tmp_path)
    report = tool.score(SPEC, submission, reference)
    assert report["n_cases"] == 3 and report["problems"] == []
    for row in report["cases"].values():
        assert row["rollout_rel_l2_disp"] == pytest.approx(0.0, abs=1e-6)
        assert row["rollout_rel_l2_aux"] == pytest.approx(0.0, abs=1e-6)
        assert row["qoi_final_length_error"] == pytest.approx(0.0, abs=1e-5)
        # ADR-0075 clause 6 (plan 3c): error growth is undefined on a zero
        # error and recorded null, never a nan in the mean.
        assert row["prop_error_growth"] is None
    assert "prop_error_growth" not in report["mean"]
    # coverage by set from the Taylor design: interp within, extrap beyond
    assert report["cases"]["T-20-80-130"]["set"] == "test_interp"
    assert report["by_set"]["test_interp"]["within, ≤ 1"]["n"] == 2
    assert report["by_set"]["test_extrap"]["beyond, ≤ 1"]["n"] == 1
    # the von Mises floor on the Taylor card is 0.12: a perfect aux is flagged
    kinds = {f["type"] for f in report["flags"]}
    assert kinds == {"below_noise_floor"}
    assert report["flags"][0]["field"] == "von_mises_stress"


def test_an_imperfect_prediction_is_scored_and_not_flagged(tmp_path):
    reference, submission = _setup(tmp_path, disp_error=2e-4, aux_factor=0.5)
    report = tool.score(SPEC, submission, reference)
    assert report["flags"] == []
    row = report["cases"]["T-20-80-170"]
    assert row["rollout_rel_l2_aux"] == pytest.approx(0.5, abs=1e-3)
    assert row["rollout_pos_rmse_mm"] > 0.0 and row["rollout_rel_l2_disp"] > 0.0
    assert set(report["mean"]) >= {"rollout_rel_l2_disp", "rollout_rel_l2_aux"}
    # A constant displacement error does not grow: the one property a
    # prediction file supports reads zero per frame (ADR-0075 clause 6), to
    # the float32 rounding of the stored positions.
    assert row["prop_error_growth"] == pytest.approx(0.0, abs=1e-5)
    assert report["mean"]["prop_error_growth"] == pytest.approx(0.0, abs=1e-5)


def test_prediction_contract_is_enforced(tmp_path):
    reference, submission = _setup(tmp_path)
    # wrong span: a prediction over every frame is refused, and named
    full = read_case(reference / "T-20-80-130.h5")
    write_case(full, submission / "T-20-80-130.h5")
    # a train case is not scored
    write_case(_reference_case("T-20-80-100", 100.0), submission / "T-20-80-100.h5")
    report = tool.score(SPEC, submission, reference)
    assert report["n_cases"] == 2
    assert any("time axis must equal" in p for p in report["problems"])
    assert any("not a scored case" in p for p in report["problems"])


def test_submission_toml_is_required_and_complete(tmp_path):
    reference, submission = _setup(tmp_path, toml='method = "Tiny"\n')
    with pytest.raises(ValueError, match="lacks"):
        tool.score(SPEC, submission, reference)
    open_toml = _TOML.replace('division = "closed"', 'division = "open"')
    reference, submission = _setup(tmp_path / "open", toml=open_toml)
    with pytest.raises(ValueError, match="declares its data"):
        tool.score(SPEC, submission, reference)


def test_rate_limit_reads_and_extends_the_ledger(tmp_path):
    reference, submission = _setup(tmp_path)
    ledger = tmp_path / "ledger.json"
    now = dt.datetime(2026, 11, 8, tzinfo=dt.UTC)
    recent = [
        {
            "submitter": "Someone",
            "method": "Tiny",
            "blind_tag": "blind-v1",
            "scored_at": (now - dt.timedelta(days=d)).isoformat(),
        }
        for d in (1, 10, 29)
    ]
    old = [dict(recent[0], scored_at=(now - dt.timedelta(days=40)).isoformat())]
    ledger.write_text(json.dumps(recent + old), encoding="utf-8")
    report = tool.score(
        SPEC, submission, reference, blind_tag="blind-v1", ledger=ledger, now=now
    )
    assert any(
        f["type"] == "rate_limited" and f["recent"] == 3 for f in report["flags"]
    )
    assert len(json.loads(ledger.read_text(encoding="utf-8"))) == 5
    # another tag is another window
    report = tool.score(
        SPEC, submission, reference, blind_tag="blind-v2", ledger=ledger, now=now
    )
    assert not any(f["type"] == "rate_limited" for f in report["flags"])
