"""Tests for the sweep generator (ADR-0069). A toy dataset, never a real one."""

import csv
import json
import subprocess

import abaqus_paths  # noqa: F401
import pytest

pytest.importorskip("scipy")

import generate  # noqa: E402

TOY_MODEL = """
import deck

def build(params, variant):
    mesh = deck.structured_quad_mesh(1, 1, 0.0, params["a"], 0.0, 1.0)
    nodes = deck.node_block(mesh.node_labels, mesh.coords)
    return deck.heading(f"toy {variant}") + nodes
"""
TOY_SWEEP = """
[dataset]
name = "toy"
case_prefix = "TOY"
units = "t-mm-s"

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
"""


def _dataset(tmp_path, n=3, model=TOY_MODEL):
    ds = tmp_path / "ds"
    ds.mkdir(exist_ok=True)
    (ds / "sweep.toml").write_text(TOY_SWEEP.format(n=n), encoding="utf-8")
    (ds / "model.py").write_text(model, encoding="utf-8")
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
    ]
    prov = json.loads((work / "toy/TOY-listed-0000-y/provenance.json").read_text())
    assert prov["variant"] == "y" and prov["split"] == "listed"
    assert prov["units"] == "t-mm-s"
    assert prov["params"] == {"a": 1.5, "k": 2.0}
    assert set(prov["dataset_repository"]) == {"commit", "dirty"}
    rows = list(csv.DictReader((work / "toy/manifest.csv").open(encoding="utf-8")))
    assert [r["case_id"] for r in rows][:3] == ids[2:] and set(rows[0]) >= {"a", "k"}


def test_loading_the_model_leaves_the_dataset_repository_clean(tmp_path):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    _run(ds, work)
    _run(ds, work, "--force")  # a second load must not find its own cache either
    prov = json.loads((work / "toy/TOY-main-0000/provenance.json").read_text())
    assert prov["dataset_repository"]["dirty"] is False
    assert not (ds / "__pycache__").exists()


def test_rerun_is_byte_identical_and_skips(tmp_path, capsys):
    ds, work = _dataset(tmp_path), tmp_path / "work"
    _run(ds, work)
    before = (work / "toy/TOY-main-0001/TOY-main-0001.inp").read_bytes()
    assert _run(ds, work) == 0
    assert (work / "toy/TOY-main-0001/TOY-main-0001.inp").read_bytes() == before
    assert "unchanged=5" in capsys.readouterr().out


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
    changed = _dataset(tmp_path, model=TOY_MODEL.replace("toy {variant}", "toy2"))
    assert _run(changed, work) == 1  # conflicts
    assert _run(changed, work, "--force") == 1  # the case with run.json stays locked
    assert b"toy2" in (work / "toy/TOY-main-0001/TOY-main-0001.inp").read_bytes()
    assert b"toy2" not in (work / "toy/TOY-main-0000/TOY-main-0000.inp").read_bytes()


def test_unsafe_case_ids_are_refused():
    sweep = {
        "dataset": {"name": "x", "case_prefix": "X" * 40, "units": "t-mm-s"},
        "variables": {"a": [0.0, 1.0]},
        "splits": {"s": {"n": 1, "seed": 1}},
    }
    with pytest.raises(ValueError, match="job name"):
        generate.plan_cases(sweep)


def test_two_sampled_splits_may_not_share_a_seed():
    # Same seed, same box: the second split would silently copy the first.
    sweep = {
        "dataset": {"name": "x", "case_prefix": "X", "units": "t-mm-s"},
        "variables": {"a": [0.0, 1.0]},
        "splits": {"s": {"n": 2, "seed": 5}, "t": {"n": 2, "seed": 5}},
    }
    with pytest.raises(ValueError, match="seed 5"):
        generate.plan_cases(sweep)


def test_unknown_units_are_refused_before_anything_is_written():
    sweep = {
        "dataset": {"name": "x", "case_prefix": "X", "units": "t-mm-sec"},
        "variables": {"a": [0.0, 1.0]},
        "splits": {"s": {"n": 1, "seed": 1}},
    }
    with pytest.raises(ValueError, match="t-mm-sec"):
        generate.plan_cases(sweep)


def test_fixed_and_sampled_names_must_not_collide():
    sweep = {
        "dataset": {"name": "x", "case_prefix": "X", "units": "t-mm-s"},
        "fixed": {"a": 1.0},
        "variables": {"a": [0.0, 1.0]},
        "splits": {"s": {"n": 1, "seed": 1}},
    }
    with pytest.raises(ValueError, match="both fixed and sampled"):
        generate.plan_cases(sweep)
