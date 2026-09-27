"""load_problem's sibling imports stay with their dataset; [levels].pilot is checked."""

import sys

import pytest
from conftest import MINIMAL_PROBLEM, MINIMAL_TOML, write_definition

from structbench.datagen import definition

WITH_HELPER = MINIMAL_PROBLEM.replace(
    "from structbench.datagen.abaqus import deck",
    "from structbench.datagen.abaqus import deck\nfrom helpers import SCALE",
)


def test_a_sibling_module_does_not_leak_into_another_dataset(tmp_path):
    for name, scale in (("a", 1.0), ("b", 2.0)):
        ds = write_definition(tmp_path / name, problem=WITH_HELPER)
        (ds / "helpers.py").write_bytes(f"SCALE = {scale}\n".encode())
    assert definition.load_problem(tmp_path / "a").SCALE == 1.0
    assert definition.load_problem(tmp_path / "b").SCALE == 2.0
    assert "helpers" not in sys.modules


def test_a_sibling_is_gone_from_sys_modules_even_when_the_problem_fails(tmp_path):
    ds = write_definition(
        tmp_path / "d",
        problem=WITH_HELPER.replace("def qoi", "def qoi_"),  # a hook is missing
    )
    (ds / "helpers.py").write_bytes(b"SCALE = 5.0\n")
    with pytest.raises(definition.DefinitionError, match="qoi"):
        definition.load_problem(ds)
    assert "helpers" not in sys.modules and str(ds) not in sys.path


@pytest.mark.parametrize(
    "pilot, expected",
    [
        ('["2", "1"]', "increasing"),
        ('["1", "1"]', "unique"),
        ('["1", "0"]', "increasing"),
    ],
)
def test_pilot_levels_are_unique_and_numeric_ones_increase(tmp_path, pilot, expected):
    toml = MINIMAL_TOML.replace('pilot = ["1", "2"]', f"pilot = {pilot}").replace(
        'production = "1"', 'production = "1"'
    )
    with pytest.raises(definition.DefinitionError, match=expected):
        definition.load_definition(write_definition(tmp_path / "d", toml=toml))
