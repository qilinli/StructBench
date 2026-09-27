"""The shipped reference sets: format, provenance, and measures that reproduce."""

import json
import re
from importlib import resources

import pytest

from structbench.validation import reference
from structbench.validation.measures import taylor

REFS = resources.files("structbench.validation") / "references"


def test_taylor_copper_is_listed_and_loads():
    assert "taylor_copper" in reference.list_references()
    ref = reference.load_reference("taylor_copper")
    assert ref.family == "taylor_rod" and ref.units["length"] == "mm"
    assert [t.id for t in ref.tests] == ["1", "2", "3", "4", "5", "6"]
    assert ref.test("3").v0_ms == 162 and ref.test("3").L0_mm == 34.5
    assert len(ref.sha256) == 64


def test_every_source_has_a_doi_and_says_whether_it_was_consulted():
    ref = reference.load_reference("taylor_copper")
    for s in ref.sources:
        assert re.fullmatch(r"10\.\d{4,9}/\S+", s["doi"]), s
        assert isinstance(s["consulted"], bool)
    assert any(s["licence"] == "CC BY 4.0" for s in ref.sources)


def test_the_stated_measures_are_what_the_functions_return_on_the_outline():
    ref = reference.load_reference("taylor_copper")
    for t in ref.tests:
        rz = t.outline_rz_mm
        assert t.Lf_mm == pytest.approx(taylor.final_length(rz), abs=1e-6)
        assert t.Rf_mm == pytest.approx(taylor.largest_radius(rz), abs=1e-6)
        expected = taylor.lateral_radii(rz, t.Lf_mm, ref.fractions)
        assert list(t.Wf_mm) == pytest.approx(expected, abs=1e-6)
        assert rz[:, 1].min() == pytest.approx(0.0, abs=1e-6)


def test_the_json_is_lf_sorted_and_names_its_extraction_tool():
    raw = (REFS / "taylor_copper.json").read_bytes()
    assert b"\r\n" not in raw
    data = json.loads(raw)
    assert data["format"] == reference.REFERENCE_FORMAT
    assert data["extraction"]["tool"] == "tools/validation/digitize_zelepugin2023.py"
    assert "source_simulation" not in raw.decode("utf-8")
    assert json.dumps(
        data, indent=1, sort_keys=True, ensure_ascii=False
    ) + "\n" == raw.decode("utf-8")


def test_the_companion_names_every_source():
    md = (REFS / "taylor_copper.md").read_text(encoding="utf-8")
    for s in reference.load_reference("taylor_copper").sources:
        assert s["doi"] in md, s["doi"]


def test_an_unknown_set_and_a_wrong_format_are_refused(tmp_path):
    with pytest.raises(reference.ReferenceError, match="taylor_copper"):
        reference.load_reference("no_such_set")
    bad = tmp_path / "bad.json"
    bad.write_bytes(b'{"format": "something/9", "name": "bad"}')
    with pytest.raises(reference.ReferenceError, match="something/9"):
        reference.load_reference_file(bad)


def test_architecture_names_the_validation_layer():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    text = (root / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "### `validation/`" in text and "structbench-validate" in text
    assert "{verification, validation}" in text


def test_a_test_whose_wf_count_differs_from_the_fractions_is_refused(tmp_path):
    import json

    data = json.loads((REFS / "taylor_copper.json").read_bytes())
    data["tests"][0]["Wf_mm"] = data["tests"][0]["Wf_mm"][:-1]
    bad = tmp_path / "bad.json"
    bad.write_bytes(json.dumps(data).encode())
    with pytest.raises(reference.ReferenceError, match="Wf_mm"):
        reference.load_reference_file(bad)
