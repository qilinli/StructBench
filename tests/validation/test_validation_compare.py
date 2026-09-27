"""compare: pairs against a reference set; deviations reported, gaps named."""

import json

import pytest
from rods import rod_case, two_body_case

from structbench.core.io import write_case
from structbench.validation import compare, reference
from structbench.validation.measures import taylor

PAIRS = """\
reference = "toy"

[variants]
a = "setup A"

[[pair]]
variant = "a"
test = "1"
case = "{case}"

[[pair]]
variant = "a"
test = "2"
status = "aborted"
reason = "distortion"
"""


def toy_reference(tmp_path):
    """Two tests whose measured outline is the synthetic rod's own, read as metres."""
    rz = taylor.outline(rod_case()) * compare.CANONICAL_TO_MM
    lf = taylor.final_length(rz)
    tests = [
        {
            "id": tid,
            "source": "S1",
            "via": None,
            "material": "toy",
            "L0_mm": 5000.0,
            "D0_mm": 2000.0,
            "v0_ms": 100.0,
            "T0_K": 298.0,
            "outline_rz_mm": rz.tolist(),
            "Lf_mm": lf,
            "Rf_mm": taylor.largest_radius(rz),
            "Wf_mm": taylor.lateral_radii(rz, lf, taylor.FRACTIONS),
        }
        for tid in ("1", "2")
    ]
    data = {
        "format": reference.REFERENCE_FORMAT,
        "name": "toy",
        "family": "taylor_rod",
        "title": "a toy set",
        "units": {"length": "mm", "velocity": "m/s", "temperature": "K"},
        "sources": [
            {
                "id": "S1",
                "citation": "c",
                "doi": "10.1000/toy",
                "licence": None,
                "consulted": True,
                "role": "r",
            }
        ],
        "extraction": {
            "tool": "none",
            "method": "synthetic",
            "checks": [],
            "date": "x",
        },
        "measures": {"fractions": list(taylor.FRACTIONS)},
        "tests": tests,
        "caveats": ["synthetic"],
    }
    path = tmp_path / "toy.json"
    path.write_bytes(json.dumps(data, sort_keys=True).encode())
    return reference.load_reference_file(path)


def write_pairs(tmp_path, text):
    path = tmp_path / "pairs.toml"
    path.write_bytes(text.encode())
    return path


def test_an_identical_case_deviates_by_zero_and_an_aborted_pair_has_no_numbers(
    tmp_path,
):
    ref = toy_reference(tmp_path)
    case_path = tmp_path / "case.h5"
    write_case(rod_case(), case_path)
    name, variants, pairs = compare.load_pairs(
        write_pairs(tmp_path, PAIRS.format(case=case_path.as_posix()))
    )
    assert name == "toy" and variants == {"a": "setup A"}
    cmp = compare.compare(ref, pairs, variants)
    done, aborted = cmp.results
    assert done.status == "completed" and done.case_id == "rod"
    assert done.dev_length == pytest.approx(0.0, abs=1e-9)
    assert done.dev_radius == pytest.approx(0.0, abs=1e-9)
    assert done.dev_lateral_rms == pytest.approx(0.0, abs=1e-9)
    assert done.length_mm == pytest.approx(4000.0)
    assert aborted.status == "aborted" and aborted.reason == "distortion"
    assert aborted.length_mm is None and aborted.dev_length is None
    assert cmp.summary["a"]["length"]["n"] == 1
    assert cmp.summary["a"]["length"]["max"] == pytest.approx(0.0, abs=1e-9)


def test_relative_paths_resolve_against_the_pairs_file(tmp_path):
    (tmp_path / "cases").mkdir()
    write_case(rod_case(), tmp_path / "cases" / "c.h5")
    _, _, pairs = compare.load_pairs(
        write_pairs(tmp_path, PAIRS.format(case="cases/c.h5"))
    )
    assert pairs[0].case == (tmp_path / "cases" / "c.h5").resolve()


def test_a_missing_case_file_is_reported_not_raised(tmp_path):
    ref = toy_reference(tmp_path)
    pairs = [compare.Pair("a", "1", tmp_path / "nowhere.h5", "nowhere.h5", None, None)]
    (result,) = compare.compare(ref, pairs, {"a": "A"}).results
    assert result.status == "missing" and result.case_id is None
    assert "nowhere.h5" in (result.reason or "") and result.case_path == "nowhere.h5"


def test_a_case_without_one_outline_is_unmeasurable(tmp_path):
    ref = toy_reference(tmp_path)
    (tmp_path / "two.h5").write_bytes(b"")
    pairs = [compare.Pair("a", "1", tmp_path / "two.h5", "two.h5", None, None)]
    (result,) = compare.compare(
        ref, pairs, {"a": "A"}, case_loader=lambda p: two_body_case()
    ).results
    assert result.status == "unmeasurable" and "OutlineError" in (result.reason or "")


def test_an_unknown_test_is_refused_by_name(tmp_path):
    ref = toy_reference(tmp_path)
    text = PAIRS.format(case="x.h5").replace('test = "1"', 'test = "9"')
    _, variants, pairs = compare.load_pairs(write_pairs(tmp_path, text))
    with pytest.raises(compare.PairsError, match="test '9'"):
        compare.compare(ref, pairs, variants)


def test_an_unknown_variant_is_refused_by_name(tmp_path):
    text = PAIRS.format(case="x.h5").replace(
        'variant = "a"\ntest = "1"', 'variant = "q"\ntest = "1"'
    )
    with pytest.raises(compare.PairsError, match="variant 'q'"):
        compare.load_pairs(write_pairs(tmp_path, text))


def test_a_duplicate_pair_is_refused(tmp_path):
    text = PAIRS.format(case="x.h5").replace('test = "2"', 'test = "1"')
    with pytest.raises(compare.PairsError, match="twice"):
        compare.load_pairs(write_pairs(tmp_path, text))


def test_a_pair_needs_a_case_or_a_status(tmp_path):
    text = PAIRS.format(case="x.h5").replace('status = "aborted"\n', "")
    with pytest.raises(compare.PairsError, match="case file or a status"):
        compare.load_pairs(write_pairs(tmp_path, text))


def test_the_reference_sets_own_fractions_are_used(tmp_path):
    # a set declaring other heights, compared with an identical outline: zero deviation
    ref = toy_reference(tmp_path)
    import dataclasses

    from structbench.validation import reference as refmod

    rz = ref.test("1").outline_rz_mm
    lf = taylor.final_length(rz)
    other = (0.1, 0.9)
    tests = tuple(
        dataclasses.replace(t, Wf_mm=tuple(taylor.lateral_radii(rz, lf, other)))
        for t in ref.tests
    )
    ref2 = refmod.ReferenceSet(
        **{**dataclasses.asdict(ref), "fractions": other, "tests": tests}
    )
    (tmp_path / "c.h5").unlink(missing_ok=True)
    write_case(rod_case(), tmp_path / "c.h5")
    (r,) = compare.compare(
        ref2,
        [compare.Pair("a", "1", tmp_path / "c.h5", "c.h5", None, None)],
        {"a": "A"},
    ).results
    assert r.status == "completed" and len(r.lateral_radii_mm) == 2
    assert r.dev_lateral_rms == pytest.approx(0.0, abs=1e-9)


def test_case_paths_are_recorded_as_written_not_resolved(tmp_path):
    ref = toy_reference(tmp_path)
    text = PAIRS.format(case="cases/nowhere.h5")
    _, variants, pairs = compare.load_pairs(write_pairs(tmp_path, text))
    assert pairs[0].case_text == "cases/nowhere.h5"
    (missing, _) = compare.compare(ref, pairs, variants).results
    assert missing.status == "missing" and missing.case_path == "cases/nowhere.h5"
    assert str(tmp_path) not in (missing.reason or "")


def test_an_unreadable_case_file_is_unmeasurable_with_its_path(tmp_path):
    ref = toy_reference(tmp_path)
    (tmp_path / "junk.h5").write_bytes(b"not an hdf5 file")
    pairs = [compare.Pair("a", "1", tmp_path / "junk.h5", "junk.h5", None, None)]
    (r,) = compare.compare(ref, pairs, {"a": "A"}).results
    assert r.status == "unmeasurable" and r.case_path == "junk.h5"
    assert "OSError" in (r.reason or "")


def test_only_aborted_is_an_accepted_status(tmp_path):
    text = PAIRS.format(case="x.h5").replace(
        'status = "aborted"', 'status = "completed"'
    )
    with pytest.raises(compare.PairsError, match="aborted"):
        compare.load_pairs(write_pairs(tmp_path, text))
    both = PAIRS.format(case="x.h5").replace(
        'case = "x.h5"', 'case = "x.h5"\nstatus = "aborted"'
    )
    with pytest.raises(compare.PairsError, match="not both"):
        compare.load_pairs(write_pairs(tmp_path, both))


def test_results_carry_the_measured_values(tmp_path):
    ref = toy_reference(tmp_path)
    write_case(rod_case(), tmp_path / "m.h5")
    (r,) = compare.compare(
        ref, [compare.Pair("a", "1", tmp_path / "m.h5", "m.h5", None, None)], {"a": "A"}
    ).results
    t = ref.test("1")
    assert r.measured_length_mm == t.Lf_mm and r.measured_radius_mm == t.Rf_mm
    assert r.measured_lateral_mm == tuple(t.Wf_mm)
