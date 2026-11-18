"""A solver is experimental or admitted; the later stages run Abaqus only (ADR-0079)."""

import pytest
from conftest import MINIMAL_TOML, write_definition

pytest.importorskip("scipy")

from structbench.datagen import (  # noqa: E402
    archive,
    card,
    cli,
    converge,
    definition,
    generate,
    preflight,
)

TOY = MINIMAL_TOML.replace('solver = "abaqus"', 'solver = "toysolver"')


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setattr(definition, "EXPERIMENTAL_SOLVERS", ("toysolver",))
    ds = write_definition(tmp_path / "toy", toml=TOY)
    (ds / card.CARD_FILE).write_bytes(card.template_card("toy").encode("utf-8"))
    return ds


def test_no_solver_is_both_and_the_stages_run_only_known_ones():
    assert not set(definition.SOLVERS) & set(definition.EXPERIMENTAL_SOLVERS)
    known = definition.SOLVERS + definition.EXPERIMENTAL_SOLVERS
    assert set(definition.STAGE_SOLVERS) <= set(known)


def test_a_solver_in_neither_list_is_still_refused(tmp_path):
    with pytest.raises(definition.DefinitionError, match="dataset.solver"):
        definition.load_definition(write_definition(tmp_path / "t", toml=TOY))


def test_check_accepts_an_experimental_solver_and_says_so(toy, capsys):
    assert cli.main(["check", str(toy)]) == 0
    out = capsys.readouterr().out
    assert "note: solver 'toysolver' is experimental (ADR-0079)" in out


def test_check_says_nothing_of_an_admitted_solver(tmp_path, capsys):
    assert cli.main(["check", str(write_definition(tmp_path / "a"))]) == 0
    assert "experimental" not in capsys.readouterr().out


def test_the_card_says_the_solver_is_experimental(toy):
    assert card.main(["--dataset", str(toy)]) == 0
    text = (toy / card.CARD_FILE).read_text(encoding="utf-8")
    assert "**The solver 'toysolver' is experimental (ADR-0079)" in text


@pytest.mark.parametrize(
    "stage, argv",
    [
        (generate, ["--dataset", "{ds}", "--work-root", "{tmp}"]),
        (preflight, ["--dataset", "{ds}", "--work-root", "{tmp}"]),
        (converge, ["--dataset", "{ds}", "--sweep", "{tmp}"]),
        (archive, ["--dataset", "{ds}", "--sweep", "{tmp}", "--data-root", "{tmp}/o"]),
    ],
)
def test_a_later_stage_refuses_a_solver_it_has_no_code_for(
    toy, tmp_path, capsys, stage, argv
):
    args = [a.format(ds=toy, tmp=tmp_path) for a in argv]
    assert stage.main(args) == 2
    err = capsys.readouterr().err
    assert "has no code for 'toysolver'" in err and "ADR-0079" in err
    assert not (tmp_path / "toy" / "preflight").exists()
