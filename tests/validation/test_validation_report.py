"""The record: byte-stable JSON, Markdown regenerated from it."""

from conftest import rod_case
from test_validation_compare import toy_reference

from structbench.core.io import write_case
from structbench.validation import compare, report

VARIANTS = {"a": "setup A", "b": "setup B"}


def _pairs(tmp_path):
    # one measured pair, one aborted, one whose file is missing
    if not (tmp_path / "case.h5").is_file():
        write_case(rod_case(), tmp_path / "case.h5")
    return [
        compare.Pair("a", "1", tmp_path / "case.h5", None, None),
        compare.Pair("a", "2", None, "aborted", "distortion"),
        compare.Pair("b", "1", tmp_path / "none.h5", None, None),
    ]


def _record(tmp_path):
    ref = toy_reference(tmp_path)
    cmp = compare.compare(ref, _pairs(tmp_path), VARIANTS)
    return ref, report.build_record(ref, cmp, VARIANTS, {"solver": "toy"}, ("none",))


def test_the_record_is_byte_identical_on_regeneration(tmp_path):
    ref, rec = _record(tmp_path)
    first = report.to_json(rec)
    _, again = _record(tmp_path)
    assert first == report.to_json(again)
    assert b"\r\n" not in first and first.endswith(b"\n")
    assert report.from_json(first) == rec
    assert rec.format == report.RECORD_FORMAT and rec.reference["sha256"] == ref.sha256


def test_markdown_renders_from_the_json_and_names_sources_and_gaps(tmp_path):
    ref, rec = _record(tmp_path)
    md = report.render_markdown(rec, ref)
    assert md == report.render_markdown(report.from_json(report.to_json(rec)), ref)
    assert "10.1000/toy" in md and "aborted" in md and "missing" in md
    assert "setup A" in md and "toy" in md and "%" in md
    assert "none" in md  # the record's caveat
