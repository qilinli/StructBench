"""`new` scaffolds a definition from the example; `check` validates one, no solver."""

import pytest
from conftest import MINIMAL_PROBLEM, MINIMAL_TOML, write_definition

pytest.importorskip("scipy")

from structbench.datagen import template  # noqa: E402


def test_the_example_declares_its_probes_and_the_documented_defaults():
    """The scaffold is the example: what a new dataset starts from must be the
    documented contract (plan 2b)."""
    from structbench.datagen import definition

    defn = definition.load_definition(template.EXAMPLE_DIR)
    p = defn.pilot
    assert p.contact_force_global == "reaction_force_2_reference_node"
    assert (p.increment_key, p.increment_factors) == ("dt_scale", (0.5,))
    assert (p.frame_key, p.frame_count_key, p.frame_factor) == (
        "frame_interval",
        "n_intervals",
        0.5,
    )
    assert (p.frame_tolerance, p.settling_margin) == (0.05, 0.25)
    assert defn.qoi.tolerance == (0.01, 0.01)


def test_scaffold_writes_the_four_files_and_renames_the_dataset(tmp_path):
    written = template.scaffold(tmp_path / "my_set", "my_set")
    names = sorted(p.name for p in written)
    assert names == ["DATA_CARD.md", "README.md", "dataset.toml", "problem.py"]
    toml = (tmp_path / "my_set" / "dataset.toml").read_text(encoding="utf-8")
    assert 'name = "my_set"' in toml and "abaqus_conformance" not in toml


def test_scaffold_writes_lf_endings(tmp_path):
    for path in template.scaffold(tmp_path / "s", "s"):
        assert b"\r\n" not in path.read_bytes()


def test_scaffold_refuses_a_non_empty_target(tmp_path):
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "note.txt").write_text("x")
    with pytest.raises(FileExistsError):
        template.scaffold(tmp_path / "busy", "busy")


def test_a_fresh_scaffold_passes_check(tmp_path):
    template.scaffold(tmp_path / "ok", "ok")
    assert template.check_definition(tmp_path / "ok") == []


def test_check_lists_definition_errors_instead_of_raising(tmp_path):
    ds = write_definition(tmp_path / "d", toml=MINIMAL_TOML.replace("[qoi]", "[qoi_]"))
    problems = template.check_definition(ds)
    assert len(problems) == 1 and "qoi" in problems[0]


def test_check_reports_an_unstable_deck(tmp_path):
    unstable = MINIMAL_PROBLEM.replace(
        'return deck.heading("toy")',
        "import time\n    return deck.heading(str(time.perf_counter_ns()))",
    )
    ds = write_definition(tmp_path / "d", problem=unstable)
    assert any("byte-stable" in p for p in template.check_definition(ds))


HASH_ORDERED = MINIMAL_PROBLEM.replace(
    'return deck.heading("toy")',
    'return deck.heading(" ".join({"alpha", "bravo", "charlie", "delta", "echo", '
    '"foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima"}))',
)


def test_check_reports_a_deck_that_depends_on_hash_order(tmp_path):
    # identical within one interpreter, different in the next: the review's gap
    ds = write_definition(tmp_path / "d", problem=HASH_ORDERED)
    problems = template.check_definition(ds)
    assert any("byte-stable" in p and "hash seed" in p for p in problems)


@pytest.mark.parametrize(
    "toml, expected",
    [
        ("[dataset\nname = 'x'\n", "dataset.toml"),
        (MINIMAL_TOML.replace("L = [1.0, 2.0]", "L = [2.0, 1.0]"), "variables.L"),
        (MINIMAL_TOML.replace("L = [1.0, 2.0]", "L = 1.0"), "variables.L"),
        (
            MINIMAL_TOML.replace("[splits.train]\nn = 4", "[splits.train]"),
            "splits.train",
        ),
    ],
)
def test_check_reports_a_malformed_definition_in_a_sentence(tmp_path, toml, expected):
    problems = template.check_definition(write_definition(tmp_path / "d", toml=toml))
    assert len(problems) == 1 and expected in problems[0]


def test_check_reports_a_problem_that_does_not_import(tmp_path):
    broken = "import no_such_module_xyz\n" + MINIMAL_PROBLEM
    problems = template.check_definition(
        write_definition(tmp_path / "d", problem=broken)
    )
    assert len(problems) == 1
    assert "problem.py" in problems[0] and "no_such_module_xyz" in problems[0]
    unparsable = MINIMAL_PROBLEM.replace("def mesh(params):", "def mesh(params:")
    problems = template.check_definition(
        write_definition(tmp_path / "e", problem=unparsable)
    )
    assert len(problems) == 1 and "SyntaxError" in problems[0]


def test_check_reports_a_mesh_hook_that_raises(tmp_path):
    raising = MINIMAL_PROBLEM.replace(
        "def mesh(params):\n", 'def mesh(params):\n    raise RuntimeError("no grid")\n'
    )
    problems = template.check_definition(
        write_definition(tmp_path / "d", problem=raising)
    )
    assert problems == ["problem.mesh: raised RuntimeError: no grid"]


@pytest.mark.parametrize(
    "toml, expected",
    [
        (
            MINIMAL_TOML.replace(
                'case_prefix = "TOY"', 'case_prefix = "' + "X" * 40 + '"'
            ),
            "job name",
        ),
        (MINIMAL_TOML.replace("n = 4\n", 'n = 4\nwithin = "hot"\n'), "region"),
        (
            MINIMAL_TOML.replace("E = 1000.0", "E = 1000.0\nL = 1.0"),
            "both fixed and sampled",
        ),
    ],
)
def test_check_reports_what_generate_would_refuse(tmp_path, toml, expected):
    problems = template.check_definition(write_definition(tmp_path / "d", toml=toml))
    assert any(expected in p for p in problems), problems


def test_check_names_the_level_that_does_not_nest(tmp_path):
    skewed = MINIMAL_PROBLEM.replace(
        "mesh = deck.structured_quad_mesh(k, k,",
        "mesh = deck.structured_quad_mesh(k, 2 * k + 1,",
    ).replace(
        "return deck.structured_quad_mesh(k, k,",
        "return deck.structured_quad_mesh(k, 2 * k + 1,",
    )
    ds = write_definition(tmp_path / "d", problem=skewed)
    problems = template.check_definition(ds)
    assert any("nest" in p and "2" in p for p in problems)


def test_check_reports_qoi_keys_that_differ_from_the_declaration(tmp_path):
    wrong = MINIMAL_PROBLEM.replace('{"length":', '{"span":')
    ds = write_definition(tmp_path / "d", problem=wrong)
    assert any("qoi" in p and "span" in p for p in template.check_definition(ds))


def test_the_check_command_exits_one_on_problems_and_zero_when_clean(tmp_path, capsys):
    template.scaffold(tmp_path / "ok", "ok")
    assert template.main_check([str(tmp_path / "ok")]) == 0
    bad = MINIMAL_TOML.replace("[levels]", "[lvls]")
    ds = write_definition(tmp_path / "bad", toml=bad)
    assert template.main_check([str(ds)]) == 1
    assert "levels" in capsys.readouterr().out
