"""`new` scaffolds a definition from the example; `check` validates one, no solver."""

import pytest
from conftest import MINIMAL_PROBLEM, MINIMAL_TOML, write_definition

pytest.importorskip("scipy")

from structbench.datagen import template  # noqa: E402


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
