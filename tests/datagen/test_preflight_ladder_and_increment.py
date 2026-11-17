"""The space step on a ladder that is not in constant ratio, and increment
changes a dataset accepts with a reason -- on the shipped example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")

from test_preflight_e2e import _dataset, _git, _preflight, _records  # noqa: E402

from structbench.datagen import definition, preflight, template  # noqa: E402
from structbench.datagen.definition import DefinitionError  # noqa: E402

_GAPS = 'accepted_gaps = ["solver_identity_complete"]'
ACCEPTED = (
    'accepted_increment = { "ACX-pilot-0000.final_length" = '
    '"a second instability late in the run, whose timing the increment moves", '
    '"ACX-pilot-0001.face_radius" = "never needed" }'
)


def _edit(ds: Path, old: str, new: str, message: str) -> None:
    toml = (ds / "dataset.toml").read_text(encoding="utf-8")
    assert old in toml
    (ds / "dataset.toml").write_bytes(toml.replace(old, new).encode("utf-8"))
    _git(ds, message)


def _stamp(work: Path) -> dict:
    return json.loads((work / "e2e" / "preflight" / "stamp.json").read_text("utf-8"))


# --- the declaration -------------------------------------------------------------


def test_accepted_increment_defaults_to_empty_and_carries_its_reasons(tmp_path):
    ds = _dataset(tmp_path)
    assert dict(definition.load_definition(ds).pilot.accepted_increment) == {}
    _edit(ds, _GAPS, f"{_GAPS}\n{ACCEPTED}", "accepted")
    assert dict(definition.load_definition(ds).pilot.accepted_increment) == {
        "ACX-pilot-0000.final_length": "a second instability late in the run, "
        "whose timing the increment moves",
        "ACX-pilot-0001.face_radius": "never needed",
    }
    assert template.check_definition(ds) == []


@pytest.mark.parametrize(
    ("line", "said"),
    [
        ('accepted_increment = ["ACX-pilot-0000.final_length"]', "table"),
        ('accepted_increment = { "final_length" = "why" }', "pilot case id"),
        ('accepted_increment = { "ACX-pilot-0000.width" = "why" }', "no quantity"),
        ('accepted_increment = { "ACX-pilot-0000.final_length" = " " }', "reason"),
        ('accepted_increment = { "ACX-pilot-0000.final_length" = 3 }', "reason"),
    ],
)
def test_a_malformed_accepted_increment_is_refused(tmp_path, line, said):
    ds = _dataset(tmp_path)
    _edit(ds, _GAPS, f"{_GAPS}\n{line}", "malformed")
    with pytest.raises(DefinitionError, match=said):
        definition.load_definition(ds)


def test_check_refuses_a_case_that_is_not_a_pilot(tmp_path):
    ds = _dataset(tmp_path)
    for stranger in ("ACX-pilot-0007", "ACX-pilot-0000-L2", "ACX-train-0000"):
        line = f'accepted_increment = {{ "{stranger}.final_length" = "why" }}'
        toml = (ds / "dataset.toml").read_text(encoding="utf-8")
        (ds / "dataset.toml").write_bytes(
            toml.replace(_GAPS, f"{_GAPS}\n{line}").encode("utf-8")
        )
        (finding,) = template.check_definition(ds)
        assert "accepted_increment" in finding and "not a pilot case" in finding
        (ds / "dataset.toml").write_bytes(toml.encode("utf-8"))


# --- the increment step ----------------------------------------------------------


def _increment(tmp_path, changes, declared=ACCEPTED):
    """The step on the example's pilots, each quantity moved by ``changes``."""
    ds = _dataset(tmp_path)
    if declared:
        _edit(ds, _GAPS, f"{_GAPS}\n{declared}", "accepted")
    defn = definition.load_definition(ds)
    specs = preflight.preflight_cases(defn)
    qois: dict[str, dict[str, float]] = {}
    for s in preflight.by_role(specs, "level") + preflight.by_role(specs, "increment"):
        plain = f"ACX-pilot-{s.index:04d}"
        scaled = (s.probe or {}).get("role") == "increment"
        qois[s.case_id] = {
            name: 1.0 + (changes.get(f"{plain}.{name}", 0.0) if scaled else 0.0)
            for name in defn.qoi.names
        }
    return ds, defn, specs, preflight.step_increment(qois, {}, specs, defn, None)


def test_an_accepted_exceedance_passes_and_is_listed_with_its_change(tmp_path):
    _, _, _, step = _increment(tmp_path, {"ACX-pilot-0000.final_length": 0.08})
    assert step.verdict == "pass", step.summary
    assert "except the changes the dataset accepts" in step.summary
    assert "ACX-pilot-0000.final_length changes by 0.08" in step.summary
    (entry,) = step.detail["accepted"].items()
    assert entry[0] == "ACX-pilot-0000.final_length"
    assert entry[1]["changes"] == {"0.25": pytest.approx(0.08)}
    assert entry[1]["tolerance"] == 0.01
    assert entry[1]["reason"].startswith("a second instability")
    # The other declared entry was measured and not exceeded.
    assert step.detail["unused_accepted"] == ["ACX-pilot-0001.face_radius"]
    assert step.detail["unaccepted"] == []


def test_an_exceedance_not_listed_still_fails(tmp_path):
    changes = {
        "ACX-pilot-0000.final_length": 0.08,  # accepted
        "ACX-pilot-0000.face_radius": 0.03,  # another quantity of the same pilot
        "ACX-pilot-0001.final_length": 0.02,  # the same quantity of another pilot
    }
    _, _, _, step = _increment(tmp_path, changes)
    assert step.verdict == "fail"
    assert "ACX-pilot-0000: face_radius changes by 0.03" in step.summary
    assert "ACX-pilot-0001: final_length changes by 0.02" in step.summary
    assert "ACX-pilot-0000: final_length" not in step.summary
    assert step.detail["unaccepted"] == [
        "ACX-pilot-0000.face_radius",
        "ACX-pilot-0001.final_length",
    ]
    assert list(step.detail["accepted"]) == ["ACX-pilot-0000.final_length"]


def test_without_a_declaration_the_step_is_as_before(tmp_path):
    _, _, _, quiet = _increment(tmp_path, {}, declared="")
    assert quiet.verdict == "pass" and "except" not in quiet.summary
    assert quiet.detail["accepted"] == {} and quiet.detail["unused_accepted"] == []
    _, _, _, loud = _increment(
        tmp_path / "b", {"ACX-pilot-0000.final_length": 0.08}, declared=""
    )
    assert loud.verdict == "fail"


def test_an_accepted_change_never_covers_a_missing_measurement(tmp_path):
    ds, defn, specs, _ = _increment(tmp_path, {})
    step = preflight.step_increment({}, {}, specs, defn, None)
    assert step.verdict == "not_assessable"
    assert "unused_accepted" not in step.detail  # nothing was measured


def test_the_stamp_and_the_report_list_accepted_unaccepted_and_unused(tmp_path):
    changes = {
        "ACX-pilot-0000.final_length": 0.08,
        "ACX-pilot-0001.final_length": 0.02,
    }
    ds, defn, specs, step = _increment(tmp_path, changes)
    stamp = preflight.stamp_record(
        defn,
        ds,
        [step],
        specs,
        created_utc="2026-11-17T00:00:00+00:00",
        siblings_sha256="0" * 64,
    )
    assert stamp["passed"] is False
    assert stamp["accepted_increment"]["ACX-pilot-0000.final_length"]["changes"] == {
        "0.25": pytest.approx(0.08)
    }
    assert stamp["unaccepted_increment"] == ["ACX-pilot-0001.final_length"]
    assert stamp["unused_accepted_increment"] == ["ACX-pilot-0001.face_radius"]
    report = preflight.render_report(stamp)
    assert (
        "- increment change `ACX-pilot-0000.final_length` accepted (0.08 at the "
        "increment key = 0.25; tolerance 0.01): a second instability late in the "
        "run, whose timing the increment moves"
    ) in report
    assert "`ACX-pilot-0001.final_length` not accepted: it fails the step" in report
    assert "`ACX-pilot-0001.face_radius` unused" in report


# --- the whole preflight, and the re-judging ---------------------------------------


def test_an_unequal_ladder_passes_the_space_step_with_the_manufactured_order(
    tmp_path,
):
    """The fake solver's response is second order in the spacing; the levels
    1, 2, 6 are in ratios 2 and 3."""
    ds, work = _dataset(tmp_path), tmp_path / "work"
    _edit(ds, 'pilot = ["1", "2", "4"]', 'pilot = ["1", "2", "6"]', "ladder")
    assert template.check_definition(ds) == []
    assert _preflight(ds, work) == 0
    stamp = _stamp(work)
    space = stamp["steps"]["space"]
    assert space["verdict"] == "pass", space["summary"]
    (case,) = space["detail"]["cases"].values()  # the one pilot at three levels
    for row in case.values():
        assert row["status"] == "monotone"
        assert row["order"] == pytest.approx(2.0, abs=0.05)
        assert row["error_at_production"] < 0.01
    record = json.loads(
        (work / "e2e" / "preflight" / "converge" / "convergence.json").read_text(
            "utf-8"
        )
    )
    x = next(c for c in record["cases"] if c["extrapolation"]["final_length"])
    assert x["extrapolation"]["final_length"]["coarse_ratio"] == 2.0
    assert x["extrapolation"]["final_length"]["ratio"] == 3.0

    # Re-judging with the new key: nothing runs, and the entry is unused.
    pre = work / "e2e" / "preflight"
    before = _records(pre)
    _edit(ds, _GAPS, f"{_GAPS}\n{ACCEPTED}", "accepted")
    assert _preflight(ds, work) == 2  # the runs are another definition's
    assert _preflight(ds, work, "--rejudge") == 0
    assert _records(pre) == before
    again = _stamp(work)
    assert again["passed"] is True
    assert again["steps"]["space"]["detail"] == space["detail"]
    assert again["accepted_increment"] == {}
    assert again["unused_accepted_increment"] == [
        "ACX-pilot-0000.final_length",
        "ACX-pilot-0001.face_radius",
    ]
    assert "unused: the runs do not exceed" in (pre / "report.md").read_text("utf-8")
    assert sys.executable  # the fake solver ran under this interpreter
