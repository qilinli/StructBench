"""`card` renders the generated parts of DATA_CARD.md from records, no solver."""

import json
from importlib import resources

import pytest
from conftest import MINIMAL_PROBLEM, write_definition

pytest.importorskip("scipy")

from structbench.datagen import card, cli, definition, template  # noqa: E402

EXAMPLES = ("abaqus_conformance", "abaqus_cyclic_element", "abaqus_drop_mass")


def _dataset(tmp_path, **kwargs):
    ds = write_definition(tmp_path / "toy", **kwargs)
    (ds / card.CARD_FILE).write_bytes(card.template_card("toy").encode("utf-8"))
    return ds


def _card(ds):
    return (ds / card.CARD_FILE).read_text(encoding="utf-8")


def test_the_template_carries_every_part_once_and_no_old_marker():
    text = card.template_card("x")
    for part in card.PARTS:
        assert text.count(card.start_marker(part)) == 1
        assert text.count(card.end_marker(part)) == 1
    assert "GENERATED" not in text and text.startswith("# x — data card\n")


def test_a_definition_with_no_sweep_renders_a_card_that_says_so(tmp_path):
    ds = _dataset(tmp_path)
    assert card.main(["--dataset", str(ds)]) == 0
    text = _card(ds)
    # every part is filled, and each part that needs a record states its absence
    assert card.blank_card(text) != text
    assert text.count("No sweep was read") == 7  # every part but the first
    assert "| `train` | Sobol, seed 1 | 4 | — |" in text
    assert "| `pilot` | listed points | 2 | probe |" in text
    assert "`length`" in text and definition.load_definition(ds).sha256() in text
    assert b"\r" not in (ds / card.CARD_FILE).read_bytes()


def test_rendering_twice_gives_the_same_bytes_and_check_agrees(tmp_path, capsys):
    ds = _dataset(tmp_path)
    assert card.main(["--dataset", str(ds)]) == 0
    first = (ds / card.CARD_FILE).read_bytes()
    assert card.main(["--dataset", str(ds)]) == 0
    assert (ds / card.CARD_FILE).read_bytes() == first
    assert card.main(["--dataset", str(ds), "--check"]) == 0


def test_the_authors_text_is_never_touched(tmp_path):
    ds = _dataset(tmp_path)
    text = (
        _card(ds).replace(
            "<!-- author: the model, and why this one -->",
            "A toy, because it is small.\n\n| a | b |\n|---|---|\n| 1 | 2 |",
        )
        + "\nA last line of the author's.\n"
    )
    (ds / card.CARD_FILE).write_bytes(text.encode("utf-8"))
    assert card.main(["--dataset", str(ds)]) == 0
    rendered = _card(ds)
    assert "A toy, because it is small.\n\n| a | b |" in rendered
    assert rendered.endswith("\nA last line of the author's.\n")
    assert card.blank_card(rendered) == card.blank_card(text)


def test_check_writes_nothing_and_exits_one_on_drift(tmp_path, capsys):
    ds = _dataset(tmp_path)
    assert card.main(["--dataset", str(ds), "--check"]) == 1  # never rendered
    assert _card(ds) == card.template_card("toy")
    assert card.main(["--dataset", str(ds)]) == 0
    # the definition changes: the committed card is no longer its render
    toml = (ds / "dataset.toml").read_text(encoding="utf-8")
    (ds / "dataset.toml").write_bytes(toml.replace("n = 4", "n = 5").encode())
    assert card.main(["--dataset", str(ds), "--check"]) == 1
    assert "differs" in capsys.readouterr().out


@pytest.mark.parametrize(
    "broken, said",
    [
        (lambda t: t.replace(card.end_marker("qoi"), ""), "card:qoi: no end marker"),
        (
            lambda t: t.replace(card.start_marker("qoi"), "").replace(
                card.end_marker("qoi"), ""
            ),
            "card:qoi: no marker pair",
        ),
        (lambda t: t + "<!-- card:extra:start -->\n", "card:extra: not a part"),
        (lambda t: t + "<!-- GENERATED from dataset.toml -->\n", "before 2026-11-18"),
    ],
)
def test_a_card_the_stage_cannot_place_its_parts_in_is_refused(
    tmp_path, capsys, broken, said
):
    ds = _dataset(tmp_path)
    before = broken(_card(ds))
    (ds / card.CARD_FILE).write_bytes(before.encode("utf-8"))
    assert card.main(["--dataset", str(ds)]) == 2
    assert said in capsys.readouterr().err
    assert _card(ds) == before  # refused means unchanged


def test_no_card_and_no_sweep_directory_are_refused(tmp_path, capsys):
    ds = write_definition(tmp_path / "toy")
    assert card.main(["--dataset", str(ds)]) == 2
    assert "no DATA_CARD.md" in capsys.readouterr().err
    ds = _dataset(tmp_path)
    assert card.main(["--dataset", str(ds), "--sweep", str(tmp_path / "none")]) == 2


def test_the_draws_feasible_rejects_are_counted_from_the_definition(tmp_path):
    picky = (
        MINIMAL_PROBLEM + "\n\ndef feasible(params):\n    return params['L'] < 1.5\n"
    )
    ds = _dataset(tmp_path, problem=picky)
    records = card.load_records(ds, None)
    assert records.planned == {"train": 4, "pilot": 2}
    # listed points are deliberate and bypass the hook; Sobol draws do not
    assert records.rejected["pilot"] == 0 and records.rejected["train"] > 0
    assert f"| `train` | 4 | {records.rejected['train']} |" in card.part_survivors(
        records
    )


def _case(sweep, cid, split, hashes, *, status=None, command=("abaqus", "job=x")):
    folder = sweep / cid
    folder.mkdir(parents=True)
    prov = {
        "split": split,
        "units": "t-mm-s",
        "preflight": None if split == "pilot" else {"skipped": True},
        "repository": {"version": "0.5.0", "commit": "a" * 40},
        "dataset_repository": {"commit": "b" * 40, "dirty": False},
        **hashes,
    }
    (folder / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    if status is not None:
        run = {"status": status, "abaqus_version": "2025", "command": list(command)}
        (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")


def test_a_sweeps_records_are_reported_with_what_is_missing_from_them(tmp_path):
    ds = _dataset(tmp_path)
    hashes = dict(card.load_records(ds, None).hashes)
    sweep = tmp_path / "runs" / "toy"
    _case(sweep, "TOY-train-0000", "train", hashes, status="completed")
    _case(sweep, "TOY-train-0001", "train", hashes, status="error")
    _case(sweep, "TOY-train-0002", "train", hashes)
    _case(
        sweep,
        "TOY-pilot-0000",
        "pilot",
        {**hashes, "problem_sha256": "0" * 64},
        status="completed",
        command=("python", str(tmp_path / "stand_in.py"), "job=x"),
    )
    stamp = {
        "format": "preflight-stamp/1",
        "passed": True,
        "created_utc": "2026-11-01T00:00:00+00:00",
        "cases": {"level": ["TOY-pilot-0000-L1"]},
        "steps": {},
        **hashes,
        "definition_sha256": "1" * 64,
    }
    (sweep / "preflight").mkdir()
    (sweep / "preflight" / "stamp.json").write_text(json.dumps(stamp), "utf-8")

    assert card.main(["--dataset", str(ds), "--sweep", str(sweep)]) == 0
    text = _card(ds)
    # the survivor log names every case that did not reach a canonical file
    assert "| `train` | 4 | 0 | 3 | 1 | 0 | 0 | 0 |" in text
    for line in (
        "- TOY-train-0001: error",
        "- TOY-train-0002: not run",
        "- TOY-train-0000: completed, no canonical file",
    ):
        assert line in text
    # a record for another definition is shown as such, not hidden
    assert "**The stamp is for another definition** (`definition_sha256` differs)" in (
        text
    )
    assert "**1 of them under another definition**" in text
    # a stand-in run is not a simulation, and the card says so
    assert "**1 runs were launched with a script in the solver's place" in text
    assert "- generated without a preflight: 3 cases" in text
    # the absent records are named with the stage that writes them
    assert "The sweep holds no verification record" in text
    assert "The sweep holds no convergence record" in text
    # a public file: no path of this machine, no trace of the work root
    assert str(tmp_path) not in text and tmp_path.name not in text


@pytest.mark.parametrize("name", EXAMPLES)
def test_a_shipped_examples_card_is_its_render(name, capsys):
    """The drift test: an example's definition changes, its card is re-rendered."""
    with resources.as_file(
        resources.files("structbench.datagen") / "examples" / name
    ) as path:
        assert card.main(["--dataset", str(path), "--check"]) == 0
        text = (path / card.CARD_FILE).read_text(encoding="utf-8")
    assert "<!-- author" not in text  # the author's sections are written
    assert "No sweep was read" in text  # an example ships a definition, no data


def test_new_scaffolds_an_empty_card_that_card_then_fills(tmp_path):
    assert cli.main(["new", str(tmp_path / "mine")]) == 0
    scaffolded = _card(tmp_path / "mine")
    assert scaffolded == card.template_card("mine")
    assert cli.main(["card", "--dataset", str(tmp_path / "mine")]) == 0
    assert template.check_definition(tmp_path / "mine") == []
    assert "Dataset `mine`, solver `abaqus`" in _card(tmp_path / "mine")
