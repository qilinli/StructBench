"""Tests for tools/build_hf_bundle.py (the Hugging Face staging-bundle builder).

``tools/`` is not a package (mirrors ``tests/tools/test_blessing_pooled_rmse.py``),
so the module under test is loaded from its file path via ``importlib``.
``build()`` runs against tiny synthetic ``.h5`` files that carry only the two
datasets the builder reads (``nodes/coords``, ``response/time/t``) — the
canonical schema itself is exercised elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from structbench.benchmarks.registry import get_benchmark

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "build_hf_bundle.py"
_spec = importlib.util.spec_from_file_location("build_hf_bundle", _TOOL_PATH)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def _write_stub(path: Path, *, n_nodes: int = 5, n_frames: int = 3) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("nodes/coords", data=np.zeros((n_nodes, 2), np.float32))
        f.create_dataset("response/time/t", data=np.arange(n_frames, dtype=float))


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        benchmark="taylor_impact_2d",
        data_root="",
        raw_root=None,
        out="",
        allow_missing=False,
        no_sha256=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------
# Case-id grammars and deck locators.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("benchmark", "case_id", "expected"),
    [
        (
            "taylor_impact_2d",
            "T-20-80-150",
            {"bar_width_mm": 20, "bar_length_mm": 80, "impact_speed_ms": 150},
        ),
        (
            "wave_propagation_1d",
            "W1D-300-4",
            {"bar_length_mm": 300, "initial_speed_ms": 4},
        ),
        (
            "notch_beam_2d_impact",
            "NB-I-320-Sphere-a-120",
            {
                "beam_height_mm": 80,
                "beam_width_mm": 320,
                "impactor": "Sphere",
                "impactor_cross_section": "disk",
                "notch_position": "a",
                "impact_speed_ms": 120,
            },
        ),
        (
            "notch_beam_2d_impact",
            "S_100_800_V60_extrapolation",
            {
                "beam_height_mm": 100,
                "beam_width_mm": 800,
                "impactor": "Sphere",
                "impactor_cross_section": "disk",
                "notch_position": "",
                "impact_speed_ms": 60,
                "probe_label": "extrapolation",
            },
        ),
    ],
)
def test_case_params(benchmark, case_id, expected):
    assert tool._case_params(benchmark, case_id) == expected


def test_case_params_held_aside_id_is_outside_the_grid():
    """Taylor's Convergence run has no speed token: the parser must raise, and
    build() relies on that to leave its parameter columns blank."""
    with pytest.raises(ValueError):
        tool._case_params("taylor_impact_2d", "T-20-80-Convergence")


@pytest.mark.parametrize(
    ("benchmark", "case_id", "relative"),
    [
        ("taylor_impact_2d", "T-20-80-150", "lsdyna/2080/150/Taylor.k"),
        ("taylor_impact_2d", "T-20-80-Convergence", "lsdyna/2080/Convergence/Taylor.k"),
        ("wave_propagation_1d", "W1D-300-4", "300_4/WavePropagation.k"),
        (
            "notch_beam_2d_impact",
            "NB-I-320-Sphere-a-120",
            "InitialVelocity/Sphere/80320/Aa120/Beam1.k",
        ),
        (
            "notch_beam_2d_impact",
            "S_100_800_V60_extrapolation",
            "2DGeneralizibility/S_100_800_V60_extrapolation/Beam1.k",
        ),
    ],
)
def test_deck_path(benchmark, case_id, relative, tmp_path):
    assert tool._deck_path(benchmark, case_id, tmp_path) == tmp_path / relative


# --------------------------------------------------------------------------
# The HF README: front-matter, section separation, citation.
# --------------------------------------------------------------------------


def test_hf_readme_front_matter_and_sections():
    spec = get_benchmark("wave_propagation_1d")
    text = tool._hf_readme(spec, "wave_propagation_1d", "wave-propagation-1d")
    assert text.startswith("---\nlicense: cc-by-4.0\n")
    front, body = text[3:].split("\n---\n", 1)
    assert "size_categories:\n- n<1K" in front
    # The viewer is pointed at the manifest only (one subset, one split).
    assert (
        "configs:\n- config_name: manifest\n  data_files:\n"
        "  - split: cases\n    path: cases.csv"
    ) in front
    # Title once, then Download, then the archive README body as its own
    # paragraph (a single newline would merge it into the Download text).
    assert body.count("# Wave1D-Propagation — StructBench canonical dataset") == 1
    assert "<https://github.com/qilinli/StructBench>.\n\n" in body
    assert 'hf_hub_download("StructBench/wave-propagation-1d"' in body
    # The mirror-only section (manifest, decks, case-id grammar) sits between
    # the archive's Files section and its layout table.
    assert "## Manifest and input decks (Hugging Face mirror)" in body
    assert "`W1D-<L>-<V>`" in body
    assert (
        body.index("## Files")
        < body.index("## Manifest")
        < body.index("## HDF5 layout")
    )
    assert "## Citation" in body and "@software{structbench," in body
    assert "  url     = {https://github.com/qilinli/StructBench}," in body


def test_citation_bibtex_reads_citation_cff():
    bib = tool._citation_bibtex()
    assert bib.startswith("@software{structbench,") and bib.endswith("}")
    assert "author  = {Li, Qilin}" in bib
    assert "version = {" in bib and "year    = {" in bib


# --------------------------------------------------------------------------
# build(): manifest, held-aside handling, decks, line endings.
# --------------------------------------------------------------------------


def _taylor_tree(
    tmp_path: Path, *, extras: tuple[str, ...] = ()
) -> tuple[Path, Path, Path]:
    spec = get_benchmark("taylor_impact_2d")
    data = tmp_path / "canonical"
    data.mkdir()
    for ids in spec.splits.values():
        for cid in ids:
            _write_stub(data / f"{cid}.h5")
    for cid in extras:
        _write_stub(data / f"{cid}.h5", n_nodes=25)
    raw = tmp_path / "raw"
    for cid in ("T-20-80-150", "T-20-80-Convergence"):
        deck = tool._deck_path("taylor_impact_2d", cid, raw)
        deck.parent.mkdir(parents=True, exist_ok=True)
        deck.write_text(f"*KEYWORD {cid}\n", encoding="utf-8")
    return data, raw, tmp_path / "out"


def test_build_taylor_bundle_with_held_aside_run(tmp_path, capsys):
    data, raw, out = _taylor_tree(tmp_path, extras=("T-20-80-Convergence",))
    rc = tool.build(_args(data_root=str(data), raw_root=str(raw), out=str(out)))
    assert rc == 0
    staged = out / "taylor_impact_2d"

    with (staged / "cases.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        columns = reader.fieldnames
    # Parameters first, integrity columns last.
    assert columns == [
        "case_id",
        "split",
        "bar_width_mm",
        "bar_length_mm",
        "impact_speed_ms",
        "n_nodes",
        "n_frames",
        "file_bytes",
        "sha256",
    ]
    spec = get_benchmark("taylor_impact_2d")
    n_expected = sum(len(ids) for ids in spec.splits.values())
    assert len(rows) == n_expected + 1
    held = [r for r in rows if r["split"] == "held_aside"]
    assert [r["case_id"] for r in held] == ["T-20-80-Convergence"]
    assert held[0]["bar_width_mm"] == "" and held[0]["n_nodes"] == "25"
    # SHA-256 is of the file on disk.
    digest = hashlib.sha256((data / "T-20-80-Convergence.h5").read_bytes()).hexdigest()
    assert held[0]["sha256"] == digest
    protocol = next(r for r in rows if r["case_id"] == "T-20-80-150")
    registry_split = next(s for s, ids in spec.splits.items() if "T-20-80-150" in ids)
    assert (protocol["split"], protocol["impact_speed_ms"]) == (registry_split, "150")

    # The held-aside run's deck ships too ("one per case").
    decks = sorted(p.name for p in (staged / "decks").iterdir())
    assert decks == ["README.md", "T-20-80-150.k", "T-20-80-Convergence.k"]

    # LF line endings regardless of platform; card.json alongside.
    for name in ("README.md", "decks/README.md", "card.json"):
        assert b"\r\n" not in (staged / name).read_bytes()
    assert (staged / "README.md").read_text(encoding="utf-8").startswith("---\n")

    summary = capsys.readouterr().out
    assert f"{n_expected}/{n_expected} protocol cases + 1 held aside" in summary


def test_build_refuses_stray_case_files(tmp_path):
    """An undeclared .h5 would be published with the directory: hard error,
    downgraded to a warning only under --allow-missing (smoke runs)."""
    data, raw, out = _taylor_tree(tmp_path, extras=("T-99-stray",))
    assert tool.build(_args(data_root=str(data), out=str(out))) == 1
    assert not (out / "taylor_impact_2d" / "cases.csv").exists()
    rc = tool.build(_args(data_root=str(data), out=str(out), allow_missing=True))
    assert rc == 0
    rows = list(
        csv.DictReader((out / "taylor_impact_2d" / "cases.csv").open(encoding="utf-8"))
    )
    assert any(
        r["case_id"] == "T-99-stray" and r["split"] == "held_aside" for r in rows
    )


def test_build_missing_protocol_case_is_an_error(tmp_path):
    data, raw, out = _taylor_tree(tmp_path)
    (data / "T-20-80-150.h5").unlink()
    assert tool.build(_args(data_root=str(data), out=str(out))) == 1
    assert tool.build(_args(data_root=str(data), out=str(out), allow_missing=True)) == 0


def test_build_refuses_deforming_plate(tmp_path):
    assert (
        tool.build(
            _args(
                benchmark="deforming_plate", data_root=str(tmp_path), out=str(tmp_path)
            )
        )
        == 2
    )
