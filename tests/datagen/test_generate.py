"""Tests for the sweep generator (ADR-0069, ADR-0071). A toy dataset, never a real one."""  # noqa: E501

import csv
import json
import subprocess

import pytest
from conftest import MINIMAL_PROBLEM, MINIMAL_TOML, write_definition

pytest.importorskip("scipy")

from structbench.datagen import definition, generate  # noqa: E402

TOY_PROBLEM = """
from structbench.datagen.abaqus import deck


def input_deck(params, variant):
    mesh = deck.structured_quad_mesh(1, 1, 0.0, params["a"], 0.0, 1.0)
    nodes = deck.node_block(mesh.node_labels, mesh.coords)
    return deck.heading(f"toy {variant}") + nodes


def mesh(params):
    k = int(params.get("refine", 1))
    return deck.structured_quad_mesh(k, k, 0.0, params["a"], 0.0, 1.0)


def qoi(case):
    return {"width": float(case.nodes.coords[:, 0].max())}
"""
TOY_TOML = """
[dataset]
name = "toy"
case_prefix = "TOY"
units = "t-mm-s"
solver = "abaqus"

[declaration]
unit_system = "t-mm-s"
discretisation = "FEM"
erosion = false
fields = ["node/displacement"]

[fixed]
k = 2.0

[variables]
a = [1.0, 2.0]

[splits.main]
n = {n}
seed = 5

[splits.listed]
points = [{{ a = 1.5 }}]
variants = ["x", "y"]

[splits.pilot]
points = [{{ a = 1.25 }}]
probe = true

[levels]
refine_key = "refine"
production = "1"
pilot = ["1", "2"]

[pilot]
split = "pilot"
fine_cases = []
min_free_gb = 1.0
accepted_gaps = []

[qoi]
names = ["width"]
units = ["m"]
"""
FEASIBLE_BELOW_AMAX = """
def feasible(params):
    return params['a'] < params['amax']
"""


def _dataset(tmp_path, n=3, problem=TOY_PROBLEM, toml=None):
    ds = write_definition(
        tmp_path / "ds", toml=toml or TOY_TOML.format(n=n), problem=problem
    )
    git = ["git", "-C", str(ds), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "toy", "--allow-empty"], check=True)
    return ds


def _run(ds, work, *extra):
    return generate.main(["--dataset", str(ds), "--work-root", str(work), *extra])


def test_writes_cases_provenance_and_manifest(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    assert _run(ds, work) == 0
    ids = sorted(p.name for p in (work / "toy").iterdir() if p.is_dir())
    assert ids == [
        "TOY-listed-0000-x",
        "TOY-listed-0000-y",
        "TOY-main-0000",
        "TOY-main-0001",
        "TOY-main-0002",
        "TOY-pilot-0000",
    ]
    prov = json.loads((work / "toy/TOY-listed-0000-y/provenance.json").read_text())
    assert prov["variant"] == "y" and prov["split"] == "listed"
    assert prov["units"] == "t-mm-s"
    assert prov["params"] == {"a": 1.5, "k": 2.0}
    assert set(prov["dataset_repository"]) == {"commit", "dirty"}
    rows = list(csv.DictReader((work / "toy/manifest.csv").open(encoding="utf-8")))
    assert [r["case_id"] for r in rows][:3] == ids[2:5] and set(rows[0]) >= {"a", "k"}


def test_provenance_records_both_definition_hashes(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    assert _run(ds, work) == 0
    prov = json.loads((work / "toy/TOY-main-0000/provenance.json").read_text())
    assert prov["format"] == "abaqus-provenance/2"
    assert prov["definition_sha256"] == definition.load_definition(ds).sha256()
    assert prov["problem_sha256"] == definition.problem_sha256(ds)
    assert "sweep_sha256" not in prov


def test_loading_the_problem_leaves_the_dataset_repository_clean(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    _run(ds, work)
    _run(ds, work, "--force")  # a second load must not find its own cache either
    prov = json.loads((work / "toy/TOY-main-0000/provenance.json").read_text())
    assert prov["dataset_repository"]["dirty"] is False
    assert not (ds / "__pycache__").exists()


def test_a_problems_feasibility_check_filters_sampled_points(tmp_path):
    problem = TOY_PROBLEM + "\ndef feasible(params):\n    return params['a'] < 1.5\n"
    ds, work = _dataset(tmp_path, n=4, problem=problem), tmp_path / "work"
    assert _run(ds, work) == 0
    for k in range(4):
        prov = json.loads((work / f"toy/TOY-main-{k:04d}/provenance.json").read_text())
        assert prov["params"]["a"] < 1.5
    # a listed point is deliberate and bypasses the check (a = 1.5 is infeasible)
    listed = json.loads((work / "toy/TOY-listed-0000-x/provenance.json").read_text())
    assert listed["params"]["a"] == 1.5


def test_limits_reach_feasible_but_not_the_deck(tmp_path):
    toml = TOY_TOML.format(n=4) + "\n[limits]\namax = 1.5\n"
    ds = _dataset(tmp_path, problem=TOY_PROBLEM + FEASIBLE_BELOW_AMAX, toml=toml)
    d = definition.load_definition(ds)
    specs = generate.plan_cases(d, definition.load_problem(ds).feasible)
    main = [s for s in specs if s.split == "main"]
    assert len(main) == 4 and all(s.params["a"] < 1.5 for s in main)
    assert all("amax" not in s.params for s in main)


def test_rerun_is_byte_identical_and_skips(tmp_path, capsys):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    _run(ds, work)
    before = (work / "toy/TOY-main-0001/TOY-main-0001.inp").read_bytes()
    assert _run(ds, work) == 0
    assert (work / "toy/TOY-main-0001/TOY-main-0001.inp").read_bytes() == before
    assert "unchanged=6" in capsys.readouterr().out


def test_raising_n_appends_only(tmp_path):
    work = tmp_path / "work"
    _run(_dataset(tmp_path, n=3), work)
    first = (work / "toy/TOY-main-0002/TOY-main-0002.inp").read_bytes()
    assert _run(_dataset(tmp_path, n=5), work) == 0
    assert (work / "toy/TOY-main-0002/TOY-main-0002.inp").read_bytes() == first
    assert (work / "toy/TOY-main-0004").is_dir()


def test_changed_deck_conflicts_then_force_rewrites_but_never_a_run(tmp_path):
    work = tmp_path / "work"
    _run(_dataset(tmp_path), work)
    (work / "toy/TOY-main-0000/run.json").write_text("{}")
    changed = _dataset(tmp_path, problem=TOY_PROBLEM.replace("toy {variant}", "toy2"))
    assert _run(changed, work) == 1  # conflicts
    assert _run(changed, work, "--force") == 1  # the case with run.json stays locked
    assert b"toy2" in (work / "toy/TOY-main-0001/TOY-main-0001.inp").read_bytes()
    assert b"toy2" not in (work / "toy/TOY-main-0000/TOY-main-0000.inp").read_bytes()


def _definition(tmp_path, toml):
    return definition.load_definition(write_definition(tmp_path / "d", toml=toml))


def test_unsafe_case_ids_are_refused(tmp_path):
    long_prefix = 'case_prefix = "' + "X" * 40 + '"'
    d = _definition(tmp_path, MINIMAL_TOML.replace('case_prefix = "TOY"', long_prefix))
    with pytest.raises(ValueError, match="job name"):
        generate.plan_cases(d)


def test_two_sampled_splits_may_not_share_a_seed(tmp_path):
    # Same seed, same box: the second split would silently copy the first.
    twin = "[splits.t]\nn = 2\nseed = 1\n\n[splits.pilot]"
    with pytest.raises(ValueError, match="seed 1"):
        toml = MINIMAL_TOML.replace("[splits.pilot]", twin)
        generate.plan_cases(_definition(tmp_path, toml))


def test_unknown_units_are_refused_before_anything_is_written(tmp_path):
    toml = MINIMAL_TOML.replace('units = "t-mm-s"', 'units = "t-mm-sec"', 1)
    with pytest.raises(definition.DefinitionError, match="t-mm-sec"):
        _definition(tmp_path, toml)


def test_fixed_and_sampled_names_must_not_collide(tmp_path):
    toml = MINIMAL_TOML.replace("E = 1000.0", "E = 1000.0\nL = 1.0")
    with pytest.raises(ValueError, match="both fixed and sampled"):
        generate.plan_cases(_definition(tmp_path, toml))


def test_the_minimal_definition_generates(tmp_path):
    ds = _dataset(tmp_path, problem=MINIMAL_PROBLEM, toml=MINIMAL_TOML)
    assert _run(ds, tmp_path / "work") == 0
    assert (tmp_path / "work/toy/TOY-pilot-0001/TOY-pilot-0001.inp").is_file()
