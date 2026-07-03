"""Tests for _resolve_run_spec in viz/__main__.py.

Deliberately kept in a separate file from test_fringe.py so these tests run
even when matplotlib is not installed (test_fringe.py has a module-level
importorskip that would skip everything in that file).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_viz_resolves_spec_from_run_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"benchmark": "wave_propagation_1d"}), encoding="utf-8"
    )
    from structbench.viz.__main__ import _resolve_run_spec

    spec, resolved = _resolve_run_spec(tmp_path)
    assert spec.aux_field == "axial_stress"
    assert resolved["benchmark"] == "wave_propagation_1d"


def test_viz_resolve_defaults_to_taylor(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({}), encoding="utf-8")
    from structbench.viz.__main__ import _resolve_run_spec

    spec, _ = _resolve_run_spec(tmp_path)
    assert spec.aux_field == "von_mises_stress"


def test_viz_resolve_unknown_benchmark_raises_key_error(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"benchmark": "not_a_benchmark"}), encoding="utf-8"
    )
    from structbench.viz.__main__ import _resolve_run_spec

    with pytest.raises(KeyError, match="not_a_benchmark"):
        _resolve_run_spec(tmp_path)


def test_viz_resolve_missing_config_raises(tmp_path: Path) -> None:
    from structbench.viz.__main__ import _resolve_run_spec

    with pytest.raises(FileNotFoundError, match="config.json"):
        _resolve_run_spec(tmp_path)
