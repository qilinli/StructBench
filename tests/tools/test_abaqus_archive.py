"""Tests for archive.py: finished cases -> the data tree; ODB retention (Task 10).

Temporary folders only. The data tree is never listed, only written at the
paths the plan names.
"""

import hashlib
import json
from pathlib import Path

import abaqus_paths  # noqa: F401
import archive
import numpy as np

_TOML = """
[dataset]
name = "toy"

[retention]
odb_fraction = 0.5
odb_seed = 7
odb_cases = ["T-0003"]
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sweep(tmp_path: Path, n: int = 4) -> tuple[Path, Path, Path]:
    sweep, dataset, data = tmp_path / "toy", tmp_path / "dataset", tmp_path / "data"
    dataset.mkdir()
    data.mkdir()
    (dataset / "sweep.toml").write_text(_TOML, encoding="utf-8")
    ids = [f"T-{i:04d}" for i in range(n)]
    for cid in ids:
        folder = sweep / cid
        folder.mkdir(parents=True)
        for name in (f"{cid}.inp", f"{cid}.sta", f"{cid}.msg", f"{cid}.dat"):
            (folder / name).write_text(f"{name} text\n", encoding="utf-8")
        (folder / "run.json").write_text('{"status": "completed"}', encoding="utf-8")
        prov = {"split": "train", "units": "t-mm-s"}
        (folder / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
        (folder / "runner.log").write_text("Licensed to: seat 4242\n", encoding="utf-8")
        odb = folder / f"{cid}.odb"
        odb.write_bytes(cid.encode() * 100)
        manifest = {"format": "abaqus-npz/1", "odb_sha256": _sha(odb)}
        np.savez(folder / f"{cid}.npz", manifest=np.array(json.dumps(manifest)))
    (sweep / "canonical").mkdir()
    for cid in ids:
        (sweep / "canonical" / f"{cid}.h5").write_bytes(b"h5" + cid.encode())
    (sweep / "datacheck").mkdir()
    record = {"cases": [{"case_id": cid} for cid in ids]}
    (sweep / "datacheck" / "measurements.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    return sweep, dataset, data


def _argv(sweep, dataset, data, *extra):
    return [
        "--sweep",
        str(sweep),
        "--dataset",
        str(dataset),
        "--data-root",
        str(data),
        *extra,
    ]


def test_the_dry_run_writes_nothing(tmp_path, capsys):
    sweep, dataset, data = _sweep(tmp_path)
    assert archive.main(_argv(sweep, dataset, data)) == 0
    assert list(data.iterdir()) == []
    assert "dry run" in capsys.readouterr().out


def test_yes_copies_the_case_files_and_never_the_runner_log(tmp_path):
    sweep, dataset, data = _sweep(tmp_path)
    assert archive.main(_argv(sweep, dataset, data, "--yes")) == 0
    raw = data / "raw" / "toy" / "abaqus" / "T-0000"
    names = {p.name for p in raw.iterdir()}
    assert {"T-0000.inp", "T-0000.npz", "run.json", "provenance.json"} <= names
    assert {"T-0000.sta", "T-0000.msg", "T-0000.dat"} <= names
    assert "runner.log" not in names
    assert (data / "canonical" / "toy" / "T-0000.h5").read_bytes() == b"h5T-0000"
    assert (sweep / "T-0000" / "T-0000.odb").exists()  # never deletes the local copy
    kept = {p.parent.name for p in (data / "raw" / "toy" / "abaqus").glob("*/*.odb")}
    assert kept == archive.retained(
        [f"T-{i:04d}" for i in range(4)], fraction=0.5, seed=7, named=["T-0003"]
    )


def test_retention_is_deterministic_for_a_seed():
    ids = [f"T-{i:04d}" for i in range(100)]
    first = archive.retained(ids, fraction=0.05, seed=3, named=[])
    assert first == archive.retained(ids, fraction=0.05, seed=3, named=[])
    assert len(first) == 5
    assert "T-0042" in archive.retained(ids, fraction=0.05, seed=3, named=["T-0042"])


def test_a_rerun_skips_verified_files_and_refuses_a_mismatch(tmp_path, capsys):
    """Review Focus 5: never overwrite silently after a partial copy."""
    sweep, dataset, data = _sweep(tmp_path)
    archive.main(_argv(sweep, dataset, data, "--yes"))
    target = data / "raw" / "toy" / "abaqus" / "T-0001" / "run.json"
    target.write_text("tampered", encoding="utf-8")
    capsys.readouterr()
    assert archive.main(_argv(sweep, dataset, data, "--yes")) == 1
    assert target.read_text(encoding="utf-8") == "tampered"
    out = capsys.readouterr().out
    assert "T-0001" in out and "mismatch" in out
    assert "copied=0" in out


def test_prune_deletes_only_an_odb_its_export_vouches_for(tmp_path, capsys):
    sweep, dataset, data = _sweep(tmp_path)
    keep = archive.retained(
        [f"T-{i:04d}" for i in range(4)], fraction=0.5, seed=7, named=["T-0003"]
    )
    drop = sorted({f"T-{i:04d}" for i in range(4)} - keep)
    bad, good = drop[0], drop[1:]
    (sweep / bad / f"{bad}.odb").write_bytes(b"not the exported file")
    assert archive.main(_argv(sweep, dataset, data, "--yes", "--prune-odb")) == 1
    assert (sweep / bad / f"{bad}.odb").exists()  # its sha256 disagrees
    for cid in good:
        assert not (sweep / cid / f"{cid}.odb").exists()
    for cid in keep:
        assert (sweep / cid / f"{cid}.odb").exists()
    assert bad in capsys.readouterr().out
