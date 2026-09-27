"""The converge stage: runs paired across roots, levels compared, the record."""

import json

import numpy as np
import pytest
from conftest import MINIMAL_TOML, write_definition

pytest.importorskip("scipy")

from structbench.core import (  # noqa: E402
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
)
from structbench.core.io import write_case  # noqa: E402
from structbench.datagen import converge, definition  # noqa: E402

TOML = (
    MINIMAL_TOML.replace('production = "1"', 'production = "2"')
    .replace('pilot = ["1", "2"]', 'pilot = ["1", "2", "4"]')
    .replace(
        "[splits.pilot]",
        "[splits.conv]\npoints = [{ L = 1.0, v0 = 10.0 }]\nprobe = true\n\n"
        "[splits.pilot]",
    )
)
#: The QoI is the x-displacement, a constant 1 + C h^2 with h = L / k: second
#: order exactly (up to float32 storage).
PROBLEM = '''\
"""A toy problem whose displacement converges at second order in the element size."""

from structbench.datagen.abaqus import deck

C = 0.04


def input_deck(params, variant):
    return deck.heading("toy")


def mesh(params):
    k = int(params.get("refine", 1))
    return deck.structured_quad_mesh(k, k, 0.0, float(params["L"]), 0.0, 1.0)


def qoi(case):
    return {"length": float(case.response.node["displacement"][-1, :, 0].max())}
'''
BASE = {"L": 1.0, "v0": 10.0, "E": 1000.0}


def _case(
    problem, params, level, *, with_stress=True, elements_across=None, h_level=None
):
    k = elements_across or int(level)
    grid = problem.mesh({**params, "refine": k})
    coords = np.asarray(grid.coords, float)
    conn = np.asarray(grid.connectivity) - 1
    n, e = len(coords), len(conn)
    h = float(params["L"]) / (h_level or int(level))
    u = np.zeros((3, n, 2), np.float32)
    u[:, :, 0] = 1.0 + 0.04 * h * h
    element = {"stress": np.ones((3, e, 6), np.float32)} if with_stress else {}
    return Case(
        metadata=Metadata(case_id="x", dimension=2, source_units="t-mm-s"),
        nodes=Nodes(coords=coords, node_id=np.arange(1, n + 1)),
        elements={
            "solid": ElementBlock(
                connectivity=conn,
                element_id=np.arange(1, e + 1),
                part_id=np.ones(e, np.int64),
            )
        },
        materials=[Material(material_id=1, source_model="toy", source_params={})],
        response=Response(
            time=np.linspace(0.0, 1.0, 3),
            node={"displacement": u},
            element={"solid": element},
            globals_={},
        ),
    )


class Sweep:
    """A run root with hand-written provenance files and canonical cases."""

    def __init__(self, root, problem):
        self.root, self.problem = root, problem
        (root / "canonical").mkdir(parents=True)

    def run(
        self,
        case_id,
        split,
        level,
        params=BASE,
        *,
        canonical=True,
        refine_key="refine",
        **case_kw,
    ):
        prov = {
            "case_id": case_id,
            "split": split,
            "variant": None,
            "params": {**params, **({refine_key: level} if refine_key else {})},
        }
        (self.root / case_id).mkdir()
        (self.root / case_id / "provenance.json").write_bytes(json.dumps(prov).encode())
        if canonical:
            c = _case(self.problem, params, level, **case_kw)
            c = Case(
                **{
                    **c.__dict__,
                    "metadata": Metadata(
                        case_id=case_id, dimension=2, source_units="t-mm-s"
                    ),
                }
            )
            write_case(c, self.root / "canonical" / f"{case_id}.h5")
        return self


@pytest.fixture
def toy(tmp_path):
    ds = write_definition(tmp_path / "ds", toml=TOML, problem=PROBLEM)
    defn, problem = definition.load_definition(ds), definition.load_problem(ds)
    return ds, defn, problem


def _three_levels(tmp_path, problem, name="runs"):
    s = Sweep(tmp_path / name, problem)
    s.run("TOY-train-0000", "train", "2")
    s.run("TOY-conv-0000", "conv", "1")
    s.run("TOY-conv-0001", "conv", "4")
    return s


def test_levels_are_paired_and_the_production_run_supplies_its_level(tmp_path, toy):
    _, defn, problem = toy
    s = _three_levels(tmp_path, problem)
    record = converge.converge(defn, problem, [s.root])
    (entry,) = record["cases"]
    assert entry["case_id"] == "TOY-train-0000"
    assert entry["levels"] == {
        "1": "TOY-conv-0000",
        "2": "TOY-train-0000",
        "4": "TOY-conv-0001",
    }
    assert entry["production_level"] == "2"
    x = entry["extrapolation"]["length"]
    assert x["status"] == "monotone" and x["order"] == pytest.approx(2.0, abs=1e-3)
    assert x["extrapolated"] == pytest.approx(1.0, abs=1e-5)
    assert x["error_vs_extrapolated"]["2"] == pytest.approx(0.01, rel=1e-2)
    assert x["ratio"] == pytest.approx(2.0)
    # the coarse displacement field differs from the finest by the constant offset
    assert entry["fields"]["displacement"]["2"] == pytest.approx(
        0.0075 / 1.0025,
        rel=1e-3,  # the constant offset over the finest norm
    )
    assert entry["fields"]["stress"]["1"] == pytest.approx(0.0, abs=1e-9)
    assert record["summary"]["qoi"]["length"]["statuses"] == {"monotone": 1}


def test_a_duplicate_run_at_one_level_is_resolved_production_first_then_root_order(
    tmp_path, toy
):
    _, defn, problem = toy
    first = _three_levels(tmp_path, problem, "first")
    first.run("TOY-conv-0002", "conv", "2")  # a probe duplicating the production level
    second = Sweep(tmp_path / "second", problem).run("TOY-train-0000", "train", "2")
    record = converge.converge(defn, problem, [first.root, second.root])
    (entry,) = record["cases"]
    assert entry["levels"]["2"] == "TOY-train-0000"
    assert any("TOY-conv-0002" in n and "level 2" in n for n in record["notes"])
    # the older root's production run is chosen when the first root has none
    # at that level
    third = Sweep(tmp_path / "third", problem)
    third.run("TOY-conv-0000", "conv", "1").run("TOY-conv-0001", "conv", "4")
    fourth = Sweep(tmp_path / "fourth", problem).run("TOY-train-0000", "train", "2")
    (entry,) = converge.converge(defn, problem, [third.root, fourth.root])["cases"]
    assert entry["levels"]["2"] == "TOY-train-0000"


def test_labels_without_a_constant_ratio_get_no_extrapolation_but_field_errors(
    tmp_path, toy
):
    _, defn, problem = toy
    s = Sweep(tmp_path / "runs", problem)
    s.run("TOY-train-0000", "train", "2").run("TOY-conv-0000", "conv", "1").run(
        "TOY-conv-0001", "conv", "6"
    )
    (entry,) = converge.converge(defn, problem, [s.root])["cases"]
    assert entry["extrapolation"]["length"] is None
    assert any("ratio" in n for n in entry["notes"])
    assert set(entry["fields"]["displacement"]) == {"1", "2"}


def test_a_field_stored_on_one_level_only_is_skipped_with_a_note(tmp_path, toy):
    _, defn, problem = toy
    s = Sweep(tmp_path / "runs", problem)
    s.run("TOY-train-0000", "train", "2").run("TOY-conv-0000", "conv", "1")
    s.run("TOY-conv-0001", "conv", "4", with_stress=False)
    (entry,) = converge.converge(defn, problem, [s.root])["cases"]
    assert "stress" not in entry["fields"] and "displacement" in entry["fields"]
    assert any("stress" in n for n in entry["notes"])


def test_a_case_whose_levels_do_not_nest_is_reported_and_the_others_proceed(
    tmp_path, toy
):
    _, defn, problem = toy
    s = _three_levels(tmp_path, problem)
    other = {**BASE, "L": 2.0}
    s.run("TOY-train-0001", "train", "2", other).run(
        "TOY-conv-0002", "conv", "1", other
    )
    s.run(
        "TOY-conv-0003", "conv", "4", other, elements_across=7
    )  # 7 x 7: nests neither
    record = converge.converge(defn, problem, [s.root])
    good, bad = record["cases"]
    assert good["case_id"] == "TOY-train-0000" and "displacement" in good["fields"]
    assert bad["case_id"] == "TOY-train-0001"
    assert any("nest" in n for n in bad["notes"]) and "4" not in bad["fields"].get(
        "displacement", {}
    )


def test_the_record_is_byte_identical_and_names_no_path(tmp_path, toy):
    _, defn, problem = toy
    s = _three_levels(tmp_path, problem)
    a = converge.to_json(converge.converge(defn, problem, [s.root]))
    b = converge.to_json(converge.converge(defn, problem, [s.root]))
    assert a == b and b"\r\n" not in a and a.endswith(b"\n")
    assert str(tmp_path).encode() not in a and b"runs" not in a
    md = converge.render_markdown(json.loads(a))
    assert md == converge.render_markdown(json.loads(b)) and "TOY-train-0000" in md


def test_runs_without_the_refine_key_are_skipped_and_noted(tmp_path, toy):
    _, defn, problem = toy
    s = _three_levels(tmp_path, problem)
    s.run("TOY-pilot-0000", "pilot", "1", refine_key=None)
    record = converge.converge(defn, problem, [s.root])
    assert len(record["cases"]) == 1
    assert any("TOY-pilot-0000" in n and "refine" in n for n in record["notes"])


def test_the_command_writes_json_and_markdown_and_exits_one_without_pairs(
    tmp_path, toy, capsys
):
    ds, _, problem = toy
    s = _three_levels(tmp_path, problem)
    rc = converge.main(["--dataset", str(ds), "--sweep", str(s.root)])
    assert rc == 0
    assert (s.root / "converge" / "convergence.json").is_file()
    assert (s.root / "converge" / "convergence.md").is_file()
    assert "1 case" in capsys.readouterr().out
    lonely = Sweep(tmp_path / "lonely", problem).run("TOY-train-0000", "train", "2")
    assert converge.main(["--dataset", str(ds), "--sweep", str(lonely.root)]) == 1
    assert (
        converge.main(["--dataset", str(tmp_path / "nowhere"), "--sweep", str(s.root)])
        == 2
    )


def test_runs_without_the_refine_key_are_noted_once_per_split(tmp_path, toy):
    _, defn, problem = toy
    s = _three_levels(tmp_path, problem)
    s.run("TOY-pilot-0000", "pilot", "1", refine_key=None)
    s.run("TOY-pilot-0001", "pilot", "1", refine_key=None)
    notes = converge.converge(defn, problem, [s.root])["notes"]
    about = [n for n in notes if "refine" in n and "pilot" in n]
    assert (
        len(about) == 1
        and "TOY-pilot-0000" in about[0]
        and "TOY-pilot-0001" in about[0]
    )
