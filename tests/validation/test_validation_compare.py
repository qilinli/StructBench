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
    pairs = [compare.Pair("a", "1", tmp_path / "nowhere.h5", None, None)]
    (result,) = compare.compare(ref, pairs, {"a": "A"}).results
    assert result.status == "missing" and result.case_id is None
    assert "nowhere.h5" in (result.reason or "")


def test_a_case_without_one_outline_is_unmeasurable(tmp_path):
    ref = toy_reference(tmp_path)
    (tmp_path / "two.h5").write_bytes(b"")
    pairs = [compare.Pair("a", "1", tmp_path / "two.h5", None, None)]
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
