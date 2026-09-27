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


def test_definition_hashes_do_not_change_with_line_endings(tmp_path):
    # a checkout's CRLF conversion must not make a definition read as another
    lf = write_definition(tmp_path / "lf")
    crlf = tmp_path / "crlf"
    crlf.mkdir()
    for name in ("dataset.toml", "problem.py"):
        (crlf / name).write_bytes((lf / name).read_bytes().replace(b"\n", b"\r\n"))
    assert (
        definition.load_definition(lf).sha256()
        == definition.load_definition(crlf).sha256()
    )
    assert definition.problem_sha256(lf) == definition.problem_sha256(crlf)


def test_malformed_toml_is_a_definition_error(tmp_path):
    ds = write_definition(tmp_path / "d", toml="[dataset\nname = 'x'\n")
    with pytest.raises(definition.DefinitionError, match="dataset.toml"):
        definition.load_definition(ds)


def test_a_problem_that_fails_to_import_is_a_definition_error(definition_dir):
    broken = "import no_such_module_xyz\n" + MINIMAL_PROBLEM
    (definition_dir / "problem.py").write_bytes(broken.encode())
    with pytest.raises(
        definition.DefinitionError, match="problem.py.*no_such_module_xyz"
    ):
        definition.load_problem(definition_dir)


DATACLASS_HEADER = (
    '"""A toy problem with a dataclass."""\n'
    "from __future__ import annotations\n"
    "from dataclasses import dataclass\n"
    "\n\n"
    "@dataclass(frozen=True)\n"
    "class Measure:\n"
    "    value: float = 1.0\n"
)


def test_a_problem_that_defines_a_dataclass_loads(definition_dir):
    # dataclasses resolve string annotations through sys.modules[cls.__module__];
    # a module executed by path without being registered there cannot define one
    problem = MINIMAL_PROBLEM.replace(
        '"""A toy problem: one quad per level, byte-stable."""', DATACLASS_HEADER
    )
    (definition_dir / "problem.py").write_bytes(problem.encode())
    module = definition.load_problem(definition_dir)
    assert module.Measure().value == 1.0


def test_a_problem_may_import_a_sibling_module(definition_dir):
    import sys

    (definition_dir / "helpers.py").write_bytes(b"SCALE = 3.0\n")
    problem = MINIMAL_PROBLEM.replace(
        "from structbench.datagen.abaqus import deck",
        "from structbench.datagen.abaqus import deck\nfrom helpers import SCALE",
    )
    (definition_dir / "problem.py").write_bytes(problem.encode())
    module = definition.load_problem(definition_dir)
    assert module.SCALE == 3.0
    assert str(definition_dir) not in sys.path  # the entry is removed afterwards


def test_levels_symmetry_defaults_to_axisymmetric_and_refuses_others(tmp_path):
    d = definition.load_definition(write_definition(tmp_path / "a"))
    assert d.levels.symmetry == "axisymmetric"
    planar = MINIMAL_TOML.replace(
        'refine_key = "refine"', 'refine_key = "refine"\nsymmetry = "planar"'
    )
    d = definition.load_definition(write_definition(tmp_path / "b", toml=planar))
    assert d.levels.symmetry == "planar"
    odd = MINIMAL_TOML.replace(
        'refine_key = "refine"', 'refine_key = "refine"\nsymmetry = "spherical"'
    )
    with pytest.raises(definition.DefinitionError, match="levels.symmetry"):
        definition.load_definition(write_definition(tmp_path / "c", toml=odd))


# --- the preflight's probe fields (plan 2b) ---------------------------------

_FIXED_CLOCK = "[fixed]\nE = 1000.0\nframe_interval = 1.0\nn_intervals = 20"
_GAPS = 'accepted_gaps = ["solver_identity_complete"]'


def test_pilot_probe_fields_have_defaults(definition_dir):
    d = definition.load_definition(definition_dir)
    p = d.pilot
    assert (p.increment_key, p.increment_factors) == ("dt_scale", (0.5,))
    assert (p.frame_key, p.frame_count_key) == ("frame_interval", "n_intervals")
    assert (p.frame_factor, p.frame_tolerance, p.settling_margin) == (0.5, 0.05, 0.25)
    assert p.contact_force_global is None
    assert d.qoi.tolerance == (0.01,)


@pytest.mark.parametrize(
    "extra, message",
    [
        ("increment_factors = [1.0]", "increment_factors"),
        ("frame_factor = 1.0", "frame_factor"),
        ("frame_factor = 0.3", "whole number"),  # fixed.n_intervals = 20
        ('frame_key = "L"', "sampled"),
        ("nonsense = 1", "unknown field"),
        ('contact_force_global = ""', "contact_force_global"),
        ("settling_margin = 1.0", "settling_margin"),
        ("frame_tolerance = 0", "frame_tolerance"),
    ],
)
def test_bad_pilot_probe_fields_are_refused(tmp_path, extra, message):
    toml = MINIMAL_TOML.replace("[fixed]\nE = 1000.0", _FIXED_CLOCK)
    toml = toml.replace(_GAPS, f"{_GAPS}\n{extra}")
    with pytest.raises(definition.DefinitionError, match=message):
        definition.load_definition(write_definition(tmp_path / "d", toml=toml))


def test_qoi_tolerance_is_one_positive_number_per_name(tmp_path):
    two = MINIMAL_TOML.replace(
        'units = ["m"]', 'units = ["m"]\ntolerance = [0.02, 0.02]'
    )
    with pytest.raises(definition.DefinitionError, match="qoi.tolerance"):
        definition.load_definition(write_definition(tmp_path / "d", toml=two))
    zero = MINIMAL_TOML.replace('units = ["m"]', 'units = ["m"]\ntolerance = [0.0]')
    with pytest.raises(definition.DefinitionError, match="qoi.tolerance"):
        definition.load_definition(write_definition(tmp_path / "e", toml=zero))
    one = MINIMAL_TOML.replace('units = ["m"]', 'units = ["m"]\ntolerance = [0.02]')
    d = definition.load_definition(write_definition(tmp_path / "f", toml=one))
    assert d.qoi.tolerance == (0.02,)
    odd = MINIMAL_TOML.replace('units = ["m"]', 'units = ["m"]\nextra = 1')
    with pytest.raises(definition.DefinitionError, match="unknown field"):
        definition.load_definition(write_definition(tmp_path / "g", toml=odd))


def test_contact_force_global_drops_the_global_prefix(tmp_path):
    toml = MINIMAL_TOML.replace(
        _GAPS,
        f'{_GAPS}\ncontact_force_global = "global/reaction_force_2_reference_node"',
    )
    p = definition.load_definition(write_definition(tmp_path / "d", toml=toml)).pilot
    assert p.contact_force_global == "reaction_force_2_reference_node"
