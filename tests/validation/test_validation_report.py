"""The record: byte-stable JSON, Markdown regenerated from it."""

from rods import rod_case
from test_validation_compare import toy_reference

from structbench.core.io import write_case
from structbench.validation import compare, report

VARIANTS = {"a": "setup A", "b": "setup B"}


def _pairs(tmp_path):
    # one measured pair, one aborted, one whose file is missing
    if not (tmp_path / "case.h5").is_file():
        write_case(rod_case(), tmp_path / "case.h5")
    return [
        compare.Pair("a", "1", tmp_path / "case.h5", "case.h5", None, None),
        compare.Pair("a", "2", None, None, "aborted", "distortion"),
        compare.Pair("b", "1", tmp_path / "none.h5", "none.h5", None, None),
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


def test_variants_and_setup_render_in_a_fixed_order_after_a_round_trip(tmp_path):
    ref = toy_reference(tmp_path)
    variants = {"fine": "fine mesh", "coarse": "coarse mesh"}  # not alphabetical
    pairs = [
        compare.Pair("fine", "1", None, None, "aborted", "x"),
        compare.Pair("coarse", "1", None, None, "aborted", "y"),
    ]
    cmp = compare.compare(ref, pairs, variants)
    rec = report.build_record(ref, cmp, variants, {"solver": "s", "element": "e"}, ())
    md = report.render_markdown(rec, ref)
    assert md == report.render_markdown(report.from_json(report.to_json(rec)), ref)
    assert md.index("coarse mesh") < md.index("fine mesh")
    assert md.index("| element |") < md.index("| solver |")


def test_the_record_does_not_depend_on_where_the_pairs_file_lives(tmp_path):
    ref = toy_reference(tmp_path)
    records = []
    for name in ("here", "there"):
        d = tmp_path / name
        d.mkdir()
        p = d / "pairs.toml"
        lines = [
            b'reference = "toy"',
            b"[variants]",
            b'a = "A"',
            b"[[pair]]",
            b'variant = "a"',
            b'test = "1"',
            b'case = "cases/gone.h5"',
        ]
        p.write_bytes(b"\n".join(lines) + b"\n")
        _, variants, pairs = compare.load_pairs(p)
        cmp = compare.compare(ref, pairs, variants)
        records.append(report.to_json(report.build_record(ref, cmp, variants, {}, ())))
    assert records[0] == records[1]


def test_rendering_with_another_reference_set_is_refused(tmp_path):
    import dataclasses

    import pytest

    ref, rec = _record(tmp_path)
    other = dataclasses.replace(ref, sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        report.render_markdown(rec, other)


def test_the_record_names_the_reference_format(tmp_path):
    from structbench.validation.reference import REFERENCE_FORMAT

    _, rec = _record(tmp_path)
    assert rec.reference["format"] == REFERENCE_FORMAT
