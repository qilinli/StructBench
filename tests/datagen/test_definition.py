"""The dataset definition contract (ADR-0071): dataset.toml and problem.py."""

import hashlib

import pytest
from conftest import MINIMAL_PROBLEM, MINIMAL_TOML, write_definition

from structbench.datagen import definition


def test_a_minimal_definition_loads_every_table(definition_dir):
    d = definition.load_definition(definition_dir)
    assert (d.name, d.case_prefix, d.units, d.solver) == (
        "toy",
        "TOY",
        "t-mm-s",
        "abaqus",
    )
    assert d.variables == {"L": (1.0, 2.0), "v0": (10.0, 20.0)}
    assert [s.name for s in d.splits] == ["train", "pilot"]
    assert d.split("pilot").probe is True and d.split("train").probe is False
    assert d.levels.production == "1" and d.levels.pilot == ("1", "2")
    assert d.pilot.split == "pilot"
    assert d.pilot.accepted_gaps == ("solver_identity_complete",)
    assert d.qoi.names == ("length",) and d.qoi.units == ("m",)
    assert d.limits == {} and d.retention == {}
    assert len(d.sha256()) == 64


@pytest.mark.parametrize(
    "drop, expected",
    [
        ("[levels]", "levels"),
        ("[pilot]", "pilot"),
        ("[qoi]", "qoi"),
        ('refine_key = "refine"', "levels.refine_key"),
        ("min_free_gb = 5.0", "pilot.min_free_gb"),
    ],
)
def test_missing_table_is_named(tmp_path, drop, expected):
    toml = MINIMAL_TOML.replace(drop, "")
    with pytest.raises(definition.DefinitionError, match=expected):
        definition.load_definition(write_definition(tmp_path / "d", toml=toml))


def test_pilot_split_must_be_an_explicit_probe(tmp_path):
    sampled = MINIMAL_TOML.replace('split = "pilot"', 'split = "train"')
    with pytest.raises(definition.DefinitionError, match="pilot.split"):
        definition.load_definition(write_definition(tmp_path / "a", toml=sampled))
    unmarked = MINIMAL_TOML.replace("probe = true\n", "")
    with pytest.raises(definition.DefinitionError, match="probe"):
        definition.load_definition(write_definition(tmp_path / "b", toml=unmarked))


def test_production_level_must_be_a_pilot_level(tmp_path):
    toml = MINIMAL_TOML.replace('production = "1"', 'production = "4"')
    with pytest.raises(definition.DefinitionError, match="levels.production"):
        definition.load_definition(write_definition(tmp_path / "d", toml=toml))


def test_refine_key_may_not_be_a_variable_or_a_constant(tmp_path):
    toml = MINIMAL_TOML.replace("E = 1000.0", "E = 1000.0\nrefine = 2")
    with pytest.raises(definition.DefinitionError, match="refine"):
        definition.load_definition(write_definition(tmp_path / "d", toml=toml))


def test_qoi_names_and_units_must_pair(tmp_path):
    toml = MINIMAL_TOML.replace('units = ["m"]', 'units = ["m", "m"]')
    with pytest.raises(definition.DefinitionError, match="qoi.units"):
        definition.load_definition(write_definition(tmp_path / "d", toml=toml))


def test_unknown_unit_label_and_solver_are_refused(tmp_path):
    bad_units = MINIMAL_TOML.replace('units = "t-mm-s"', 'units = "furlongs"')
    with pytest.raises(definition.DefinitionError, match="dataset.units"):
        definition.load_definition(write_definition(tmp_path / "u", toml=bad_units))
    bad_solver = MINIMAL_TOML.replace('solver = "abaqus"', 'solver = "ansys"')
    with pytest.raises(definition.DefinitionError, match="dataset.solver"):
        definition.load_definition(write_definition(tmp_path / "s", toml=bad_solver))


def test_problem_must_define_the_three_required_hooks(definition_dir):
    problem = definition.load_problem(definition_dir)
    assert callable(problem.input_deck)
    assert callable(problem.mesh) and callable(problem.qoi)
    broken = MINIMAL_PROBLEM.replace("def qoi", "def qoi_")
    (definition_dir / "problem.py").write_bytes(broken.encode())
    with pytest.raises(definition.DefinitionError, match="qoi"):
        definition.load_problem(definition_dir)


def test_problem_hash_is_of_the_file_bytes(definition_dir):
    expected = hashlib.sha256((definition_dir / "problem.py").read_bytes()).hexdigest()
    assert definition.problem_sha256(definition_dir) == expected
