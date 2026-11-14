"""Tests for tools/publish_checkpoints.py (ADR-0074 clause 8; plan of
2026-11-09): staging a release's blessed checkpoints from a synthetic
``models/`` mirror. ``tools/`` is not a package, so the module is loaded from
its path like the other tool tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from structbench.benchmarks.registry import get_benchmark
from structbench.benchmarks.results import BaselineResult

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "publish_checkpoints.py"
_spec = importlib.util.spec_from_file_location("publish_checkpoints", _TOOL_PATH)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
# dataclasses resolve string annotations through sys.modules[cls.__module__]
sys.modules[_spec.name] = tool
_spec.loader.exec_module(tool)

BUNDLE = "cgn-abc1234"
FILE = "model-best-000002.pt"
POINTER = f"models/taylor_impact_2d/{BUNDLE}/{FILE}"
CONFIG = json.dumps({"model": {"family": "cgn"}, "data_root": "/home/someone/data"})
METRICS = json.dumps(
    {"split": "test_interp", "checkpoint_path": f"runs/fleet/s1/{FILE}", "mean": {}}
)


def _mirror(tmp_path: Path, *, listed_digest: str | None = None) -> tuple[Path, str]:
    """A models mirror with one Taylor bundle; returns (root, digest)."""
    bundle = tmp_path / "models" / "taylor_impact_2d" / BUNDLE
    bundle.mkdir(parents=True)
    (bundle / FILE).write_bytes(b"weights")
    (bundle / "config.json").write_text(CONFIG, encoding="utf-8")
    (bundle / "normalization_stats.npz").write_bytes(b"stats")
    (bundle / "metrics-test_interp.json").write_text(METRICS, encoding="utf-8")
    # what a private bundle also holds and a publication must not carry
    (bundle / "model-best-000001.pt").write_bytes(b"earlier")
    (bundle / "train.log").write_text("/home/someone/data\n", encoding="utf-8")
    (bundle / "metrics-val.json").write_text("{}", encoding="utf-8")
    (bundle / "provenance").mkdir()
    (bundle / "provenance" / "MANIFEST.tsv").write_text("s1\n", encoding="utf-8")
    (bundle / "rollouts").mkdir()
    (bundle / "rollouts" / "case.npz").write_bytes(b"rollout")
    digest = hashlib.sha256(b"weights").hexdigest()
    cfg_digest = hashlib.sha256(CONFIG.encode()).hexdigest()
    (bundle / "SHA256SUMS").write_text(
        f"{cfg_digest}  ./config.json\n{listed_digest or digest}  ./{FILE}\n",
        encoding="utf-8",
    )
    return tmp_path / "models", digest


def _result(digest: str | None, **over) -> BaselineResult:
    kwargs = dict(
        family="cgn",
        label="CGN",
        run_commit="abc1234",
        run_date="2026-07-08",
        scheme="autoregressive",
        metrics={"test_interp": {"rollout_rel_l2_displacement": 0.13}},
        checkpoint=POINTER,
        checkpoint_sha256=digest,
    )
    kwargs.update(over)
    return BaselineResult(**kwargs)


def test_parse_pointer_splits_the_archive_path():
    assert tool.parse_pointer(POINTER) == ("taylor_impact_2d", BUNDLE, FILE)
    with pytest.raises(ValueError, match="archive-relative"):
        tool.parse_pointer("https://huggingface.co/x/y/resolve/v1/b/f.pt")
    with pytest.raises(ValueError, match="archive-relative"):
        tool.parse_pointer("models/taylor_impact_2d/f.pt")


def test_read_sha256sums_handles_the_sha256sum_format(tmp_path):
    root, digest = _mirror(tmp_path)
    sums = tool.read_sha256sums(root / "taylor_impact_2d" / BUNDLE)
    assert sums[FILE] == digest and "config.json" in sums


def test_verify_bundle_passes_a_consistent_row(tmp_path):
    root, digest = _mirror(tmp_path)
    check = tool.verify_bundle(root, _result(digest))
    assert check.ok and check.problems == []
    assert check.digest == digest
    assert check.bundle_dir == root / "taylor_impact_2d" / BUNDLE


def test_verify_bundle_names_each_problem(tmp_path):
    root, digest = _mirror(tmp_path, listed_digest="0" * 64)
    check = tool.verify_bundle(root, _result(digest))
    assert not check.ok and any("SHA256SUMS" in p for p in check.problems)
    root2, digest2 = _mirror(tmp_path / "two")
    check = tool.verify_bundle(root2, _result("f" * 64))
    assert not check.ok and any("registry" in p for p in check.problems)
    check = tool.verify_bundle(root2, _result(None))
    assert not check.ok and any("no checkpoint_sha256" in p for p in check.problems)
    check = tool.verify_bundle(tmp_path / "nowhere", _result(digest2))
    assert not check.ok and any("bundle" in p for p in check.problems)


def test_public_url_is_the_immutable_resolve_form():
    assert tool.public_url("taylor_impact_2d", "v0.4.0", BUNDLE, FILE) == (
        "https://huggingface.co/StructBench/taylor-impact-2d/resolve/v0.4.0/"
        f"{BUNDLE}/{FILE}"
    )
    assert "deforming-plate" in tool.public_url("deforming_plate", "v0.4.0", "b", "f")


def test_plan_release_stages_blessed_rows_only_and_skips_published(tmp_path):
    root, digest = _mirror(tmp_path)
    blessed = _result(digest)
    provisional = _result(
        digest,
        family="mgn",
        label="MGN",
        provisional=True,
        checkpoint="models/taylor_impact_2d/mgn-0000000/m.pt",
    )
    published = _result(
        None,
        family="transolver",
        label="Transolver",
        checkpoint="https://huggingface.co/StructBench/taylor-impact-2d/resolve/v0.3.0/t/m.pt",
    )
    spec = replace(
        get_benchmark("taylor_impact_2d"), results=(blessed, provisional, published)
    )
    plan = tool.plan_release("taylor_impact_2d", spec, root, "v0.4.0")
    assert [r.label for r in plan.rows] == ["CGN"]
    assert plan.skipped == {"MGN": "provisional", "Transolver": "already public"}
    assert plan.rows[0].url.endswith(f"/v0.4.0/{BUNDLE}/{FILE}")
    assert plan.ok
    with_prov = tool.plan_release(
        "taylor_impact_2d", spec, root, "v0.4.0", include_provisional=True
    )
    assert [r.label for r in with_prov.rows] == ["CGN", "MGN"]
    assert not with_prov.ok  # the MGN bundle does not exist on the mirror


def test_stage_writes_readme_script_and_pointers(tmp_path):
    root, digest = _mirror(tmp_path)
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_result(digest),))
    plan = tool.plan_release("taylor_impact_2d", spec, root, "v0.4.0")
    out = tmp_path / "stage"
    tool.stage(plan, out)
    folder = out / "taylor_impact_2d"
    readme = (folder / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\nlicense: apache-2.0\n")
    assert "| CGN | cgn | autoregressive |" in readme and digest in readme
    assert "docs/benchmarks/taylor_impact_2d.md" in readme
    # the card ends with the software citation CITATION.cff states
    assert "## Citation" in readme and "@software{structbench," in readme
    assert "  author  = {Li, Qilin}," in readme and "  doi     = {10.5281/" in readme
    script =(folder / "publish.sh").read_text(encoding="utf-8")
    assert "hf repos create StructBench/taylor-impact-2d --type model" in script
    assert f'"$(dirname "$0")/{BUNDLE}" {BUNDLE} --type model' in script
    assert "README.md" in script and str(root) not in script
    assert (
        "--public" in script
        and "tag create StructBench/taylor-impact-2d v0.4.0" in script
    )
    pointers = json.loads((folder / "pointers.json").read_text(encoding="utf-8"))
    assert pointers["CGN"] == {
        "checkpoint": POINTER,
        "url": plan.rows[0].url,
        "checkpoint_sha256": digest,
    }


def test_stage_publishes_the_minimal_subset_without_local_paths(tmp_path):
    root, digest = _mirror(tmp_path)
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_result(digest),))
    plan = tool.plan_release("taylor_impact_2d", spec, root, "v0.4.0")
    staged = tool.stage(plan, tmp_path / "stage") / BUNDLE
    assert sorted(p.name for p in staged.iterdir()) == [
        "SHA256SUMS",
        "config.json",
        "metrics-test_interp.json",
        FILE,
        "normalization_stats.npz",
    ]
    # the checkpoint is the registered file, byte for byte
    assert hashlib.sha256((staged / FILE).read_bytes()).hexdigest() == digest
    config = json.loads((staged / "config.json").read_text(encoding="utf-8"))
    assert config == {"model": {"family": "cgn"}}
    metrics = json.loads(
        (staged / "metrics-test_interp.json").read_text(encoding="utf-8")
    )
    assert metrics["checkpoint_path"] == FILE
    assert all("someone" not in p.read_text(errors="ignore") for p in staged.iterdir())
    # SHA256SUMS covers exactly the published files, with their staged digests
    sums = tool.read_sha256sums(staged)
    assert sorted(sums) == sorted(
        p.name for p in staged.iterdir() if p.name != "SHA256SUMS"
    )
    for name, listed in sums.items():
        assert hashlib.sha256((staged / name).read_bytes()).hexdigest() == listed
    # the private bundle is untouched
    assert json.loads((root / "taylor_impact_2d" / BUNDLE / "config.json").read_text())[
        "data_root"
    ]


def test_stage_refuses_a_local_path_it_does_not_know(tmp_path):
    root, digest = _mirror(tmp_path)
    bundle = root / "taylor_impact_2d" / BUNDLE
    (bundle / "metrics-test_interp.json").write_text(
        json.dumps({"mean": {}, "stats_file": "/home/someone/derived/norm.npz"}),
        encoding="utf-8",
    )
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_result(digest),))
    plan = tool.plan_release("taylor_impact_2d", spec, root, "v0.4.0")
    with pytest.raises(ValueError, match="stats_file holds a local path"):
        tool.stage(plan, tmp_path / "stage")


def test_run_dir_stands_in_for_a_bundle_that_is_not_on_the_mirror(tmp_path):
    run = tmp_path / "runs" / "s1"
    run.mkdir(parents=True)
    (run / FILE).write_bytes(b"weights")
    (run / "config.json").write_text(CONFIG, encoding="utf-8")
    digest = hashlib.sha256(b"weights").hexdigest()
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_result(digest),))
    empty = tmp_path / "models"
    assert not tool.plan_release("taylor_impact_2d", spec, empty, "v0.4.0").ok
    plan = tool.plan_release(
        "taylor_impact_2d", spec, empty, "v0.4.0", run_dirs={BUNDLE: run}
    )
    assert plan.ok  # no SHA256SUMS needed; the registry digest identifies the file
    staged = tool.stage(plan, tmp_path / "stage") / BUNDLE
    assert sorted(p.name for p in staged.iterdir()) == [
        "SHA256SUMS",
        "config.json",
        FILE,
    ]
    wrong = replace(spec, results=(_result("f" * 64),))
    assert not tool.plan_release(
        "taylor_impact_2d", wrong, empty, "v0.4.0", run_dirs={BUNDLE: run}
    ).ok


def test_main_check_fails_on_a_bad_mirror(tmp_path, monkeypatch, capsys):
    root, digest = _mirror(tmp_path, listed_digest="0" * 64)
    spec = replace(get_benchmark("taylor_impact_2d"), results=(_result(digest),))
    monkeypatch.setattr(tool, "get_benchmark", lambda name: spec)
    rc = tool.main(
        [
            "--benchmark",
            "taylor_impact_2d",
            "--models-root",
            str(root),
            "--out",
            str(tmp_path / "stage"),
            "--tag",
            "v0.4.0",
            "--check",
        ]
    )
    assert rc == 1
    assert "SHA256SUMS" in capsys.readouterr().out
    assert not (tmp_path / "stage").exists()
