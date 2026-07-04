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


# ---------------------------------------------------------------------------
# split_and_case stem parser — covers all three rollout families
# ---------------------------------------------------------------------------


def _parse(stem: str) -> tuple[str, str]:
    from structbench.viz.__main__ import split_and_case

    return split_and_case(stem)


class TestSplitAndCase:
    """split_and_case must parse stems from every benchmark family."""

    # --- Taylor family ---
    def test_taylor_val(self) -> None:
        assert _parse("val-T-20-60-100") == ("val", "T-20-60-100")

    def test_taylor_test_interp(self) -> None:
        assert _parse("test_interp-T-20-60-200") == ("test_interp", "T-20-60-200")

    def test_taylor_test_extrap(self) -> None:
        assert _parse("test_extrap-T-20-60-200") == ("test_extrap", "T-20-60-200")

    # --- Notch-beam family ---
    def test_notch_bend_val(self) -> None:
        assert _parse("val-NB-B-320-Ab-16") == ("val", "NB-B-320-Ab-16")

    def test_notch_impact_test_interp(self) -> None:
        assert _parse("test_interp-NB-I-480-Bullet-b-80") == (
            "test_interp",
            "NB-I-480-Bullet-b-80",
        )

    def test_notch_probe(self) -> None:
        assert _parse("probe-C_60_240_V22_extrapolation") == (
            "probe",
            "C_60_240_V22_extrapolation",
        )

    # --- Wave 1-D family ---
    def test_wave_val(self) -> None:
        assert _parse("val-W1D-300-2") == ("val", "W1D-300-2")

    def test_wave_test_extrap(self) -> None:
        assert _parse("test_extrap-W1D-500-8") == ("test_extrap", "W1D-500-8")

    # --- Error path ---
    def test_no_separator_raises(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            _parse("nocaseid")
