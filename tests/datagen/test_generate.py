"""Tests for the sweep generator (ADR-0069, ADR-0071). A toy dataset, never a real one."""  # noqa: E501

import csv
import hashlib
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


def _run(ds, work, *extra, gate=False):
    """Generate; the toy sweeps have no preflight stamp, so the gate is off
    unless a test is about it."""
    args = ["--dataset", str(ds), "--work-root", str(work), *extra]
    return generate.main(args if gate else [*args, "--no-preflight"])


def _stamp(work, ds, *, passed=True, definition_sha=None, problem_sha=None):
    sweep = work / "toy"
    (sweep / "preflight").mkdir(parents=True, exist_ok=True)
    stamp = {
        "format": "preflight-stamp/1",
        "passed": passed,
        "created_utc": "2026-09-27T00:00:00+00:00",
        "definition_sha256": definition_sha or definition.load_definition(ds).sha256(),
        "problem_sha256": problem_sha or definition.problem_sha256(ds),
    }
    (sweep / "preflight" / "stamp.json").write_text(json.dumps(stamp), encoding="utf-8")


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
    assert prov["params"] == {"a": 1.5, "k": 2.0, "refine": "1"}  # production level
    assert set(prov["dataset_repository"]) == {"commit", "dirty"}
    rows = list(csv.DictReader((work / "toy/manifest.csv").open(encoding="utf-8")))
    assert [r["case_id"] for r in rows][:3] == ids[2:5] and set(rows[0]) >= {"a", "k"}


def test_provenance_records_both_definition_hashes(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    assert _run(ds, work) == 0
    prov = json.loads((work / "toy/TOY-main-0000/provenance.json").read_text())
    assert prov["format"] == "abaqus-provenance/3"
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


def test_non_probe_splits_run_at_the_production_level(tmp_path):
    specs = generate.plan_cases(_definition(tmp_path, MINIMAL_TOML))
    by_split = {}
    for s in specs:
        by_split.setdefault(s.split, set()).add(s.params.get("refine"))
    assert by_split["train"] == {"1"}  # levels.production, not left to the deck
    assert by_split["pilot"] == {None}  # a probe split chooses its own levels


def test_a_production_split_may_not_name_another_level(tmp_path):
    other = MINIMAL_TOML.replace("n = 4\n", 'n = 4\ncategorical = { refine = ["2"] }\n')
    with pytest.raises(ValueError, match="splits.train.*levels.production"):
        generate.plan_cases(_definition(tmp_path, other))
    same = MINIMAL_TOML.replace("n = 4\n", 'n = 4\ncategorical = { refine = ["1"] }\n')
    specs = generate.plan_cases(_definition(tmp_path, same))
    assert {s.params["refine"] for s in specs if s.split == "train"} == {"1"}


def test_generate_exits_two_with_one_line_on_a_refused_definition(tmp_path, capsys):
    long_prefix = 'case_prefix = "' + "X" * 40 + '"'
    toml = MINIMAL_TOML.replace('case_prefix = "TOY"', long_prefix)
    ds = _dataset(tmp_path, problem=MINIMAL_PROBLEM, toml=toml)
    assert _run(ds, tmp_path / "work") == 2
    err = capsys.readouterr().err
    assert "job name" in err and "Traceback" not in err and err.count("\n") == 1
    assert not (tmp_path / "work").exists()


def test_generate_records_the_package_version_outside_a_checkout(tmp_path, monkeypatch):
    import structbench

    monkeypatch.setattr(generate, "_REPO", tmp_path / "site-packages")
    ds, work = (
        _dataset(tmp_path, problem=MINIMAL_PROBLEM, toml=MINIMAL_TOML),
        tmp_path / "w",
    )
    assert _run(ds, work) == 0
    prov = json.loads((work / "toy/TOY-train-0000/provenance.json").read_text())
    assert prov["repository"] == {
        "version": structbench.__version__,
        "commit": None,
        "dirty": None,
    }


def test_the_minimal_definition_generates(tmp_path):
    ds = _dataset(tmp_path, problem=MINIMAL_PROBLEM, toml=MINIMAL_TOML)
    assert _run(ds, tmp_path / "work") == 0
    assert (tmp_path / "work/toy/TOY-pilot-0001/TOY-pilot-0001.inp").is_file()


# --- the preflight gate (plan 2b) --------------------------------------------


def test_production_splits_are_refused_without_a_passing_stamp(tmp_path, capsys):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    assert _run(ds, work, gate=True) == 2
    err = capsys.readouterr().err
    assert "no preflight stamp" in err and "main" in err and "--no-preflight" in err
    assert not (work / "toy" / "TOY-main-0000").exists()
    _stamp(work, ds, passed=False)
    assert _run(ds, work, gate=True) == 2
    assert "did not pass" in capsys.readouterr().err
    _stamp(work, ds, problem_sha="0" * 64)
    assert _run(ds, work, gate=True) == 2
    assert "another problem.py" in capsys.readouterr().err
    _stamp(work, ds, definition_sha="0" * 64)
    assert _run(ds, work, gate=True) == 2
    assert "another dataset.toml" in capsys.readouterr().err
    assert not (work / "toy" / "TOY-main-0000").exists()


def test_probe_splits_are_never_gated(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    assert _run(ds, work, "--split", "pilot", gate=True) == 0
    prov = json.loads((work / "toy/TOY-pilot-0000/provenance.json").read_text())
    assert prov["preflight"] is None and prov["probe"] is None


def test_a_dry_run_is_not_gated(tmp_path, capsys):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    assert _run(ds, work, "--dry-run", gate=True) == 0
    assert "main: 3 cases" in capsys.readouterr().out


def test_a_passing_stamp_is_recorded_in_every_guarded_case(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    _stamp(work, ds)
    assert _run(ds, work, gate=True) == 0
    prov = json.loads((work / "toy/TOY-main-0000/provenance.json").read_text())
    assert set(prov["preflight"]) == {"stamp_sha256", "created_utc"}
    stamp_path = work / "toy/preflight/stamp.json"
    assert prov["preflight"]["stamp_sha256"] == definition.file_sha256(stamp_path)
    assert prov["preflight"]["created_utc"] == "2026-09-27T00:00:00+00:00"
    pilot = json.loads((work / "toy/TOY-pilot-0000/provenance.json").read_text())
    assert pilot["preflight"] is None


def test_no_preflight_records_the_omission(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    assert _run(ds, work) == 0
    prov = json.loads((work / "toy/TOY-main-0000/provenance.json").read_text())
    assert prov["preflight"] == {"skipped": True}
    assert prov["format"] == "abaqus-provenance/3" and prov["probe"] is None


def test_case_id_for_appends_a_suffix_within_the_job_name_rule(tmp_path):
    defn = definition.load_definition(_dataset(tmp_path))
    assert generate.case_id_for(defn, "pilot", 3, None, "-L2") == "TOY-pilot-0003-L2"
    assert generate.case_id_for(defn, "pilot", 3, "x", "-T0p5") == (
        "TOY-pilot-0003-x-T0p5"
    )
    with pytest.raises(ValueError, match="job name"):
        generate.case_id_for(defn, "pilot", 3, None, "-" + "x" * 40)


def test_materialise_writes_probe_metadata_and_a_given_deck(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    defn, problem = definition.load_definition(ds), definition.load_problem(ds)
    spec = generate.CaseSpec(
        "TOY-pilot-0000-E",
        "pilot",
        0,
        None,
        None,
        {"k": 2.0, "a": 1.25, "refine": "1"},
        probe={"role": "conformance", "level": "1", "factor": None},
    )
    counts, problems = generate.materialise(
        work / "toy",
        [spec],
        defn=defn,
        problem=problem,
        dataset_dir=ds,
        decks={"TOY-pilot-0000-E": "*HEADING\nhanded in\n"},
    )
    assert (dict(counts), problems) == ({"written": 1}, [])
    folder = work / "toy" / "TOY-pilot-0000-E"
    assert (folder / "TOY-pilot-0000-E.inp").read_text() == "*HEADING\nhanded in\n"
    prov = json.loads((folder / "provenance.json").read_text())
    assert prov["probe"] == {"role": "conformance", "level": "1", "factor": None}
    assert prov["preflight"] is None and prov["split"] == "pilot"
    assert prov["inp_sha256"] == hashlib.sha256(b"*HEADING\nhanded in\n").hexdigest()
