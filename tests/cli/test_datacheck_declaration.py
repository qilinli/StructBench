"""`datacheck measure --declaration`: an unregistered sweep (plan 2, Task 8).

A private sweep has no registry entry until its paper is out, so it declares
what a benchmark card would in its own ``sweep.toml``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from structbench.cli.datacheck import declared_from_toml, main
from structbench.core.io import abaqus_export_to_case, write_case

_SPEC = importlib.util.spec_from_file_location(
    "abaqus_adapter_fixture",
    Path(__file__).resolve().parents[1] / "core" / "test_abaqus_adapter.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE)

_TOML = """
[dataset]
name = "toy_sweep"

[declaration]
unit_system = "t-mm-s"
fields = [
    "node/displacement", "node/velocity", "node/acceleration",
    "solid/stress", "solid/effective_plastic_strain", "global/kinetic_energy",
]
discretisation = "FEM"
erosion = false
"""


def _canonical(root: Path, tmp: Path) -> None:
    np.savez(tmp / "x.npz", **_FIXTURE._arrays())
    root.mkdir()
    for name in ("T-0000", "T-0001"):
        case = abaqus_export_to_case(
            tmp / "x.npz",
            _FIXTURE._ADAPTER_DECK,
            source_units="t-mm-s",
            dimension=2,
            case_id=name,
        )
        write_case(case, root / f"{name}.h5")


def test_the_declaration_is_read_from_the_sweep(tmp_path):
    (tmp_path / "sweep.toml").write_text(_TOML, encoding="utf-8")
    name, declared = declared_from_toml(tmp_path / "sweep.toml")
    assert name == "toy_sweep"
    assert declared.unit_system == "t-mm-s"
    assert declared.discretisation == "FEM" and declared.erosion is False
    assert "solid/stress" in declared.fields
    assert declared.yield_table is None  # per-case hardening: none declared


def test_an_unknown_declaration_key_is_named(tmp_path):
    (tmp_path / "sweep.toml").write_text(_TOML + "colour = 'blue'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="colour"):
        declared_from_toml(tmp_path / "sweep.toml")


def test_measure_with_a_declaration_records_every_case(tmp_path):
    (tmp_path / "sweep.toml").write_text(_TOML, encoding="utf-8")
    root = tmp_path / "canonical"
    _canonical(root, tmp_path)
    out = tmp_path / "m.json"
    argv = ["measure", "--declaration", str(tmp_path / "sweep.toml")]
    argv += ["--data-root", str(root), "--out", str(out)]
    assert main(argv) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["benchmark"] is None
    assert [c["case_id"] for c in record["cases"]] == ["T-0000", "T-0001"]


@pytest.mark.parametrize(
    "flags", [[], ["--benchmark", "taylor_impact_2d", "--declaration", "x.toml"]]
)
def test_exactly_one_source_of_declaration(tmp_path, flags):
    argv = [
        "measure",
        *flags,
        "--data-root",
        str(tmp_path),
        "--out",
        str(tmp_path / "m.json"),
    ]
    assert main(argv) == 2


def test_a_discretisation_outside_the_card_vocabulary_is_refused(tmp_path):
    """The card's words are SPH, FEM and coupled; "solid" is a block name."""
    toml = _TOML.replace('discretisation = "FEM"', 'discretisation = "solid"')
    (tmp_path / "sweep.toml").write_text(toml, encoding="utf-8")
    with pytest.raises(ValueError, match="FEM"):
        declared_from_toml(tmp_path / "sweep.toml")
