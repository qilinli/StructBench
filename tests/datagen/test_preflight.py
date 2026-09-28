"""The preflight (plan 2b): its case set, its steps on records, the stamp, the report."""  # noqa: E501

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MINIMAL_PROBLEM, MINIMAL_TOML, write_definition

pytest.importorskip("scipy")

from structbench.core import (  # noqa: E402
    Case,
    ElementBlock,
    Material,
    Metadata,
    Nodes,
    Response,
)
from structbench.datagen import definition, generate, preflight, verify  # noqa: E402
from structbench.verification.criteria import CheckResult  # noqa: E402
from structbench.verification.results import Verdict  # noqa: E402

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "abaqus_adapter_fixture", _HERE.parent / "core" / "test_abaqus_adapter.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE)

_GAPS = 'accepted_gaps = ["solver_identity_complete"]'
#: Three levels, production in the middle, the probe keys as constants.
PF_TOML = (
    MINIMAL_TOML.replace(
        "[fixed]\nE = 1000.0",
        "[fixed]\nE = 1000.0\ndt_scale = 0.5\nframe_interval = 1.0\nn_intervals = 20",
    )
    .replace('production = "1"', 'production = "2"')
    .replace('pilot = ["1", "2"]', 'pilot = ["1", "2", "4"]')
)
#: A problem whose deck reads every probe key (the minimal one ignores them).
READS_PROBES = MINIMAL_PROBLEM.replace(
    '    return deck.heading("toy") + deck.node_block(mesh.node_labels, mesh.coords)',
    '    tag = "toy %s %s %s" % (\n'
    '        params.get("dt_scale"), params.get("frame_interval"), '
    'params.get("n_intervals")\n    )\n'
    "    return deck.heading(tag) + deck.node_block(mesh.node_labels, mesh.coords)",
)


def _load(tmp_path, toml=PF_TOML, problem=None, name="d"):
    ds = write_definition(tmp_path / name, toml=toml, problem=problem)
    return ds, definition.load_definition(ds), definition.load_problem(ds)


def _toy_case(fn, frames=21, *, force=None, case_id="x"):
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    t = np.linspace(0.0, 1.0, frames)
    u = np.zeros((frames, 4, 2), np.float32)
    u[:, :, 0] = fn(t)[:, None]
    globals_ = {} if force is None else {"reaction_force": np.asarray(force, float)}
    return Case(
        metadata=Metadata(case_id=case_id, dimension=2, source_units="t-mm-s"),
        nodes=Nodes(coords=xy, node_id=np.arange(1, 5)),
        elements={
            "solid": ElementBlock(
                connectivity=np.array([[0, 1, 3, 2]]),
                element_id=np.array([1]),
                part_id=np.array([1]),
            )
        },
        materials=[Material(material_id=1, source_model="toy", source_params={})],
        response=Response(
            time=t,
            node={"displacement": u},
            element={"solid": {"stress": np.zeros((frames, 1, 6), np.float32)}},
            globals_=globals_,
        ),
    )


# --- the case set --------------------------------------------------------------


def test_preflight_cases_cover_levels_factors_frame_and_conformance(tmp_path):
    _, defn, _ = _load(tmp_path)
    specs = preflight.preflight_cases(defn)
    assert [s.case_id for s in specs] == [
        "TOY-pilot-0000-L1",
        "TOY-pilot-0000-L2",
        "TOY-pilot-0000-L4",
        "TOY-pilot-0001-L1",
        "TOY-pilot-0001-L2",
        "TOY-pilot-0000-T0p25",
        "TOY-pilot-0001-T0p25",
        "TOY-pilot-0000-F",
        "TOY-pilot-0000-E",
    ]
    by = {s.case_id: s for s in specs}
    assert by["TOY-pilot-0000-L4"].params["refine"] == "4"
    production = by["TOY-pilot-0000-L2"].params
    assert production["refine"] == "2" and production["dt_scale"] == 0.5
    assert by["TOY-pilot-0000-T0p25"].params == {**production, "dt_scale": 0.25}
    frame = by["TOY-pilot-0000-F"].params
    assert frame == {**production, "frame_interval": 0.5, "n_intervals": 40}
    assert by["TOY-pilot-0000-E"].params == production
    assert by["TOY-pilot-0000-E"].probe == {
        "role": "conformance",
        "level": "2",
        "factor": None,
    }
    assert by["TOY-pilot-0000-T0p25"].probe == {
        "role": "increment",
        "level": "2",
        "factor": 0.5,
    }
    assert by["TOY-pilot-0000-F"].probe == {
        "role": "frame",
        "level": "2",
        "factor": 0.5,
    }
    assert by["TOY-pilot-0001-L1"].probe == {
        "role": "level",
        "level": "1",
        "factor": None,
    }
    assert all(s.split == "pilot" and s.index in (0, 1) for s in specs)
    assert preflight.batch_a(specs, "2") == [
        "TOY-pilot-0000-L2",
        "TOY-pilot-0001-L2",
        "TOY-pilot-0000-E",
    ]
    assert [s.case_id for s in preflight.by_role(specs, "frame")] == [
        "TOY-pilot-0000-F"
    ]


def test_when_production_is_the_finest_level_every_pilot_runs_it(tmp_path):
    _, defn, _ = _load(
        tmp_path, PF_TOML.replace('production = "2"', 'production = "4"')
    )
    ids = [
        s.case_id for s in preflight.by_role(preflight.preflight_cases(defn), "level")
    ]
    assert ids == [
        "TOY-pilot-0000-L1",
        "TOY-pilot-0000-L2",
        "TOY-pilot-0000-L4",
        "TOY-pilot-0001-L1",
        "TOY-pilot-0001-L2",
        "TOY-pilot-0001-L4",
    ]


def test_preflight_cases_without_frame_keys_skip_the_frame_probe(tmp_path):
    toml = PF_TOML.replace("\nframe_interval = 1.0\nn_intervals = 20", "")
    _, defn, _ = _load(tmp_path, toml)
    assert not preflight.by_role(preflight.preflight_cases(defn), "frame")
    step = preflight.step_frame(None, None, defn)
    assert step.verdict == "not_assessable" and "frame_interval" in step.summary


def test_frame_probe_disabled_is_not_applicable(tmp_path):
    _, defn, _ = _load(tmp_path, PF_TOML.replace(_GAPS, f"{_GAPS}\nframe_factor = 0"))
    assert not preflight.by_role(preflight.preflight_cases(defn), "frame")
    assert preflight.step_frame(None, None, defn).verdict == "not_applicable"


def test_without_the_increment_key_in_fixed_the_probe_is_not_assessable(tmp_path):
    """Review finding 3: no assumed production value; the key must be a
    constant in [fixed] for the probe to mean anything."""
    _, defn, _ = _load(tmp_path, PF_TOML.replace("\ndt_scale = 0.5", ""))
    assert preflight.production_value(defn) is None
    specs = preflight.preflight_cases(defn)
    assert not preflight.by_role(specs, "increment")
    step = preflight.step_increment({}, {}, specs, defn, None)
    assert step.verdict == "not_assessable"
    assert "dt_scale" in step.summary and "[fixed]" in step.summary


def test_no_increment_factors_means_no_increment_cases(tmp_path):
    _, defn, _ = _load(
        tmp_path, PF_TOML.replace(_GAPS, f"{_GAPS}\nincrement_factors = []")
    )
    specs = preflight.preflight_cases(defn)
    assert not preflight.by_role(specs, "increment")
    assert (
        preflight.step_increment({}, {}, specs, defn, None).verdict == "not_applicable"
    )


def test_label_suffix_and_not_run():
    assert preflight.label_suffix("level", "2") == "-L2"
    assert preflight.label_suffix("level", "1.5") == "-L1p5"
    assert preflight.label_suffix("increment", 0.25) == "-T0p25"
    assert preflight.label_suffix("frame", None) == "-F"
    assert preflight.label_suffix("conformance", None) == "-E"
    s = preflight.not_run("space", "an earlier step failed")
    assert (s.name, s.verdict) == ("space", "not_assessable")
    assert "earlier" in s.summary


# --- the steps on records ------------------------------------------------------


def test_deck_for_widens_the_conformance_deck_or_says_why_not(tmp_path):
    _, defn, problem = _load(tmp_path)
    (e,) = preflight.by_role(preflight.preflight_cases(defn), "conformance")
    with pytest.raises(ValueError, match="ENERGY OUTPUT"):
        preflight.deck_for(e, problem)  # the minimal problem writes no step
    (lv, *_) = preflight.by_role(preflight.preflight_cases(defn), "level")
    assert preflight.deck_for(lv, problem) == problem.input_deck(dict(lv.params), None)


def test_deck_regression_fails_when_a_probe_key_is_ignored(tmp_path):
    ds, defn, problem = _load(tmp_path, name="ignores")
    specs = preflight.preflight_cases(defn)
    step = preflight.step_deck_regression(ds, specs, problem, defn)
    assert step.verdict == "fail"
    assert "'dt_scale' leaves the deck unchanged" in step.summary
    assert step.detail["unread_keys"] == ["dt_scale", "frame_interval"]
    ds2, defn2, problem2 = _load(tmp_path, problem=READS_PROBES, name="reads")
    specs2 = preflight.preflight_cases(defn2)
    step2 = preflight.step_deck_regression(ds2, specs2, problem2, defn2)
    assert step2.verdict == "pass", step2.summary
    assert step2.detail["checked"] == len(specs2) and step2.detail["unstable"] == []


def _record(status="monotone", error=0.004, coarsest=0.005):
    x = {
        "status": status,
        "order": 2.0 if status == "monotone" else None,
        "extrapolated": 1.0 if status == "monotone" else None,
        "gci_fine": 0.001 if status == "monotone" else None,
        "error_vs_extrapolated": (
            {"1": 0.02, "2": error, "4": 0.00125} if status == "monotone" else None
        ),
        "coarsest_vs_finest": coarsest,
        "levels": ["1", "2", "4"],
        "ratio": 2.0,
    }
    entry = {
        "case_id": "TOY-pilot-0000-L2",
        "levels": {"1": "a", "2": "TOY-pilot-0000-L2", "4": "c"},
        "production_level": "2",
        "qoi": {"length": {"1": 1.02, "2": 1.005, "4": 1.00125}},
        "extrapolation": {"length": x},
        "fields": {"displacement": {"1": 0.02, "2": 0.005}},
        "notes": [],
    }
    fields = {"displacement": {"1": {"lowest": 0.02, "median": 0.02, "highest": 0.02}}}
    return {"cases": [entry], "summary": {"qoi": {}, "fields": fields}}


def test_step_space_reads_the_convergence_record(tmp_path):
    _, defn, _ = _load(tmp_path)  # tolerance 0.01 per QoI
    step = preflight.step_space(_record(), defn)
    assert step.verdict == "pass", step.summary
    detail = step.detail["cases"]["TOY-pilot-0000-L2"]["length"]
    assert detail == {
        "status": "monotone",
        "order": 2.0,
        "error_at_production": 0.004,
        "coarsest_vs_finest": 0.005,
    }
    assert step.detail["fields"] == _record()["summary"]["fields"]
    assert preflight.step_space(_record(error=0.02), defn).verdict == "fail"
    assert preflight.step_space(_record(status="oscillatory"), defn).verdict == "review"
    empty = {"cases": [], "summary": {"qoi": {}, "fields": {}}}
    none = preflight.step_space(empty, defn)
    assert none.verdict == "not_assessable" and "three levels" in none.summary
    two = _record()
    two["cases"][0]["extrapolation"]["length"] = None
    assert preflight.step_space(two, defn).verdict == "not_assessable"


def test_step_increment_compares_qois_within_tolerance(tmp_path):
    _, defn, _ = _load(tmp_path)
    specs = preflight.preflight_cases(defn)
    qois = {
        "TOY-pilot-0000-L2": {"length": 1.0},
        "TOY-pilot-0000-T0p25": {"length": 1.004},
        "TOY-pilot-0001-L2": {"length": 2.0},
        "TOY-pilot-0001-T0p25": {"length": 2.0},
    }
    fields = {"TOY-pilot-0000-T0p25": {"node/displacement": 0.001}}
    step = preflight.step_increment(qois, fields, specs, defn, None)
    assert step.verdict == "pass", step.summary
    assert step.detail["pilots"]["TOY-pilot-0000"]["0.25"]["length"] == pytest.approx(
        0.004
    )
    assert step.detail["fields"] == fields and step.detail["production_value"] == 0.5
    qois["TOY-pilot-0000-T0p25"] = {"length": 1.02}
    bad = preflight.step_increment(qois, fields, specs, defn, None)
    assert bad.verdict == "fail" and "TOY-pilot-0000" in bad.summary
    del qois["TOY-pilot-0001-T0p25"]
    missing = preflight.step_increment(qois, fields, specs, defn, None)
    assert (
        missing.verdict == "not_assessable"
        and "TOY-pilot-0001-T0p25" in missing.summary
    )


def test_step_duration_measures_settling_and_separation(tmp_path):
    toml = PF_TOML.replace(_GAPS, f'{_GAPS}\ncontact_force_global = "reaction_force"')
    _, defn, _ = _load(tmp_path, toml)
    specs = preflight.preflight_cases(defn)
    t = np.linspace(0.0, 1.0, 21)
    force = np.where(t < 0.3, np.sin(np.pi * t / 0.3), 0.0)
    cases = {
        "TOY-pilot-0000-L2": _toy_case(lambda t: 1 - np.exp(-t / 0.1), force=force),
        "TOY-pilot-0001-L2": _toy_case(lambda t: 1 - np.exp(-t / 0.05), force=force),
    }
    problem = SimpleNamespace(
        qoi=lambda c: {"length": float(c.response.node["displacement"][-1, 0, 0])}
    )
    step = preflight.step_duration(cases, specs, defn, problem)
    assert step.verdict == "pass", step.summary
    d = step.detail["pilots"]["TOY-pilot-0000-L2"]
    # |u - u_final| = exp(-t/0.1) > 1 % up to t = 0.45 (frame 9); the pulse ends at 0.3
    assert d["settling_frame"] == 9 and d["separation_frame"] == 5
    assert d["share_after_settling"] == pytest.approx(0.55)
    assert step.detail["pilots"]["TOY-pilot-0001-L2"]["settling_frame"] == 4
    assert step.detail["frames"] == 21 and step.detail["slowest_settling_frame"] == 9
    assert step.detail["settling_limit_frame"] == 15  # (1 - 0.25) * 20
    strict = definition.load_definition(
        write_definition(
            tmp_path / "strict",
            toml=PF_TOML.replace(_GAPS, f"{_GAPS}\nsettling_margin = 0.6"),
        )
    )
    assert preflight.step_duration(cases, specs, strict, problem).verdict == "fail"
    assert preflight.step_duration({}, specs, defn, problem).verdict == "not_assessable"


def test_step_frame_passes_a_smooth_response_and_fails_a_jagged_one(tmp_path):
    _, defn, _ = _load(tmp_path)  # frame_tolerance 0.05
    smooth = _toy_case(lambda t: 1 - np.exp(-t / 0.3), 41, force=np.linspace(0, 1, 41))
    production = _toy_case(
        lambda t: 1 - np.exp(-t / 0.3), 21, force=np.linspace(0, 1, 21)
    )
    step = preflight.step_frame(smooth, production, defn)
    assert step.verdict == "pass", step.summary
    assert step.detail["stride"] == 2
    assert step.detail["interpolation"]["node/displacement"] < 0.05
    assert step.detail["common_instants"]["node/displacement"] == pytest.approx(
        0.0, abs=1e-6
    )
    # the force rises linearly over 21 production frames: 10 % at frame 2, 90 % at 18
    assert step.detail["shortest_rise_frames"] == 16
    assert step.detail["judged"] == [
        "global/reaction_force",
        "node/displacement",
        "solid/stress",
    ]
    jagged = _toy_case(lambda t: np.sin(2 * np.pi * 10 * t), 41)
    bad = preflight.step_frame(jagged, None, defn)
    assert bad.verdict == "fail" and bad.detail["common_instants"] is None


def test_step_frame_judges_the_clock_the_factor_names(tmp_path):
    """Review finding 2: at frame_factor 0.25 the stored clock is every fourth
    frame of the probe export, not every second; the judged error must be the
    21-frame clock's whichever export measured it."""

    def wave(t):
        return np.sin(2 * np.pi * 4.0 * t)  # five stored frames per period

    _, quarter, _ = _load(
        tmp_path, PF_TOML.replace(_GAPS, f"{_GAPS}\nframe_factor = 0.25"), name="q"
    )
    _, half, _ = _load(tmp_path, name="h")
    from_quarter = preflight.step_frame(
        _toy_case(wave, 81), _toy_case(wave, 21), quarter
    )
    from_half = preflight.step_frame(_toy_case(wave, 41), _toy_case(wave, 21), half)
    assert (from_quarter.detail["stride"], from_half.detail["stride"]) == (4, 2)
    judged_q = from_quarter.detail["interpolation"]["node/displacement"]
    judged_h = from_half.detail["interpolation"]["node/displacement"]
    assert judged_q == pytest.approx(judged_h, rel=0.3)
    assert from_quarter.verdict == from_half.verdict == "fail"
    # the mistake the finding names: interpolating across two probe frames
    from structbench.verification import temporal

    two_frames = temporal.midpoint_interpolation_errors(_toy_case(wave, 81))
    assert two_frames["node/displacement"] < 0.5 * judged_q


def _jagged(frames=41):
    return np.sin(2 * np.pi * 10 * np.linspace(0, 1, frames)).astype(np.float32)


def test_with_nothing_reported_acceleration_and_stress_are_judged(tmp_path):
    """Plan 3a: an empty frame_reported judges every stored field."""
    _, defn, _ = _load(tmp_path)
    case = _toy_case(lambda t: 1 - np.exp(-t / 0.3), 41)
    case.response.node["acceleration"] = np.repeat(
        _jagged()[:, None, None], 4, 1
    ).repeat(2, 2)
    step = preflight.step_frame(case, None, defn)
    assert step.verdict == "fail" and "node/acceleration" in step.summary
    stressed = _toy_case(lambda t: 1 - np.exp(-t / 0.3), 41)
    stressed.response.element["solid"]["stress"] = np.repeat(
        _jagged()[:, None, None], 6, 2
    )
    step = preflight.step_frame(stressed, None, defn)
    assert step.verdict == "fail" and "solid/stress" in step.summary


def test_reported_fields_carry_their_reason_and_everything_else_is_judged(tmp_path):
    declared = (
        'frame_reported = { "node/acceleration" = "second derivative", '
        '"global/absent" = "not stored in this case" }'
    )
    _, defn, _ = _load(tmp_path, PF_TOML.replace(_GAPS, f"{_GAPS}\n{declared}"))
    case = _toy_case(lambda t: 1 - np.exp(-t / 0.3), 41, force=np.linspace(0, 1, 41))
    case.response.node["acceleration"] = np.repeat(
        _jagged()[:, None, None], 4, 1
    ).repeat(2, 2)
    step = preflight.step_frame(case, None, defn)
    assert step.verdict == "pass", step.summary
    reported = step.detail["reported"]
    assert set(reported) == {"node/acceleration"}
    assert reported["node/acceleration"]["reason"] == "second derivative"
    assert reported["node/acceleration"]["error"] > 0.5
    assert step.detail["reported_absent"] == ["global/absent"]
    assert step.detail["judged"] == [
        "global/reaction_force",
        "node/displacement",
        "solid/stress",
    ]
    assert "node/acceleration" in step.summary  # the report says what was not judged


def test_a_non_finite_reported_field_does_not_block(tmp_path):
    declared = 'frame_reported = { "global/reaction_force" = "contact chatter" }'
    _, defn, _ = _load(tmp_path, PF_TOML.replace(_GAPS, f"{_GAPS}\n{declared}"))
    force = np.linspace(0.0, 1.0, 41)
    force[3] = np.nan
    case = _toy_case(lambda t: 1 - np.exp(-t / 0.3), 41, force=force)
    step = preflight.step_frame(case, None, defn)
    assert step.verdict == "pass", step.summary


def _history(arrays, **terms):
    out = dict(arrays)
    for term, value in terms.items():
        key = f"history/S/Assembly Assembly-1/{term}"
        series = out[key].copy()
        series[1:, 1] = value
        out[key] = series
    return out


def test_step_conformance_names_missing_and_extra_terms(tmp_path):
    # the fixture: AE=1 CD=2 FD=3 IE=4 KE=5 PD=6 SE=7 VD=8 WK=9 ETOTAL=10, no PW.
    # closing: CD (outside the identity) to 0, PW to 0, ETOTAL to KE+IE+VD+FD-WK = 11
    base = _FIXTURE._arrays()
    base["history/S/Assembly Assembly-1/ALLPW"] = base[
        "history/S/Assembly Assembly-1/ALLWK"
    ].copy()
    passing = _history(base, ALLCD=0.0, ALLPW=0.0, ETOTAL=11.0)
    np.savez(tmp_path / "ok.npz", **passing)
    step = preflight.step_conformance(tmp_path / "ok.npz", "t-mm-s")
    assert step.verdict == "pass", step.summary
    assert step.detail["identity_residual"] == pytest.approx(0.0, abs=1e-12)
    assert step.detail["missing"] == [] and step.detail["extra_nonzero"] == []
    np.savez(tmp_path / "extra.npz", **_FIXTURE._arrays())
    extra = preflight.step_conformance(tmp_path / "extra.npz", "t-mm-s")
    assert extra.verdict == "fail" and "ALLCD" in extra.summary
    missing = {k: v for k, v in passing.items() if not k.endswith("/ALLVD")}
    np.savez(tmp_path / "missing.npz", **missing)
    gone = preflight.step_conformance(tmp_path / "missing.npz", "t-mm-s")
    assert gone.verdict == "fail" and "ALLVD" in gone.summary
    off = _history(passing, ETOTAL=12.0)
    np.savez(tmp_path / "off.npz", **off)
    unbalanced = preflight.step_conformance(tmp_path / "off.npz", "t-mm-s")
    assert unbalanced.verdict == "fail" and "identity" in unbalanced.summary
    assert preflight.step_conformance(None, "t-mm-s").verdict == "not_assessable"


def test_the_conformance_step_fails_a_deck_that_omits_required_evidence(tmp_path):
    """The decks alone decide it: a deck that asks for too little fails even
    before any run exists, so nothing is launched on it."""
    findings = ["X-L2: misses output:time_marks"]
    step = preflight.step_conformance(None, "t-mm-s", deck_findings=findings)
    assert step.verdict == "fail" and "time_marks" in step.summary
    assert step.detail["deck_findings"] == findings
    base = _FIXTURE._arrays()
    base["history/S/Assembly Assembly-1/ALLPW"] = base[
        "history/S/Assembly Assembly-1/ALLWK"
    ].copy()
    np.savez(tmp_path / "ok.npz", **_history(base, ALLCD=0.0, ALLPW=0.0, ETOTAL=11.0))
    ok = preflight.step_conformance(tmp_path / "ok.npz", "t-mm-s", decks_checked=29)
    assert ok.verdict == "pass" and "all 29 decks ask for" in ok.summary
    assert ok.detail["deck_findings"] == []


def test_deck_request_findings_name_each_deck_and_what_it_misses():
    from importlib import resources

    root = resources.files("structbench.datagen") / "examples" / "abaqus_conformance"
    with resources.as_file(root) as ds:
        defn, problem = definition.load_definition(ds), definition.load_problem(ds)
    specs = preflight.preflight_cases(defn)
    decks = {s.case_id: preflight.deck_for(s, problem) for s in specs}
    assert preflight.deck_request_findings(decks, defn.units) == []
    first = sorted(decks)[0]
    broken = {**decks, first: decks[first].replace(", TIME MARKS=YES", "")}
    assert preflight.deck_request_findings(broken, defn.units) == [
        f"{first}: misses output:time_marks"
    ]
    hidden = {first: decks[first] + "*INCLUDE, INPUT=more.inp\n"}
    (finding,) = preflight.deck_request_findings(hidden, defn.units)
    assert finding.startswith(f"{first}: its output requests cannot be read")


def _report(rows):
    by: dict[str, list[CheckResult]] = {}
    for case_id, quantity, verdict, value in rows:
        by.setdefault(case_id, []).append(
            CheckResult(quantity, Verdict(verdict), value, "1")
        )
    cases = [SimpleNamespace(case_id=c, results=tuple(r)) for c, r in by.items()]
    return SimpleNamespace(cases=cases)


def test_step_energy_and_verification_read_the_report(tmp_path):
    _, defn, _ = _load(tmp_path)  # accepts solver_identity_complete
    ok = _report(
        [
            ("A", "plastic_dissipation_excess_max", "pass", 0.0),
            ("A", "energy_gain_max", "not_assessable", None),
            ("A", "solver_identity_complete", "fail", 0.0),
        ]
    )
    assert preflight.step_energy(ok, ["A"]).verdict == "pass"
    v = preflight.step_verification(ok, defn)
    assert v.verdict == "pass" and v.detail["accepted_gaps"] == {
        "solver_identity_complete": {"fail": 1}
    }
    bad = _report(
        [
            ("A", "plastic_dissipation_excess_max", "fail", 0.1),
            ("B", "plastic_dissipation_excess_max", "pass", 0.0),
        ]
    )
    e = preflight.step_energy(bad, ["A", "B"])
    assert e.verdict == "fail" and "A" in e.summary
    assert e.detail["rows"]["plastic_dissipation_excess_max"] == {"fail": 1, "pass": 1}
    assert preflight.step_energy(bad, ["B"]).verdict == "pass"
    w = preflight.step_verification(bad, defn)
    assert w.verdict == "fail" and w.detail["failing"] == [
        "A: plastic_dissipation_excess_max"
    ]
    assert preflight.step_energy(_report([]), ["A"]).verdict == "not_assessable"


#: A toy problem whose mesh grows with the rod: 2 L refine elements along, refine
#: across, so production cases of different lengths cost different amounts.
SIZED_PROBLEM = MINIMAL_PROBLEM.replace(
    '    return deck.structured_quad_mesh(k, k, 0.0, float(params["L"]), 0.0, 1.0)\n'
    "\n\ndef qoi",
    '    n = int(round(2 * float(params["L"]))) * k\n'
    '    return deck.structured_quad_mesh(n, k, 0.0, float(params["L"]), 0.0, 1.0)\n'
    "\n\ndef qoi",
)


def _budget_inputs(tmp_path):
    assert SIZED_PROBLEM != MINIMAL_PROBLEM
    _, defn, problem = _load(tmp_path, problem=SIZED_PROBLEM)
    specs = preflight.preflight_cases(defn)
    # pilot 0 (L 1) at level 2: 15 nodes, 8 elements; pilot 1 (L 2): 27 and 16
    runs = {
        "TOY-pilot-0000-L1": {"wall_s": 10.0},
        "TOY-pilot-0000-L2": {"wall_s": 80.0},
        "TOY-pilot-0001-L2": {"wall_s": 160.0},
        "TOY-pilot-0000-L4": {"wall_s": 900.0},
    }
    sizes = {"TOY-pilot-0000-L2": 15_000_000, "TOY-pilot-0001-L2": 27_000_000}
    train = [s for s in generate.plan_cases(defn) if s.split == "train"]
    return defn, problem, specs, runs, sizes, train


def _mesh_size(problem, spec):
    grid = problem.mesh(dict(spec.params))
    return len(grid.node_labels), len(grid.connectivity)


def test_the_budget_sizes_production_by_its_own_meshes(tmp_path):
    """Plan 3a: pilots at the box's corners no longer set the estimate."""
    defn, problem, specs, runs, sizes, train = _budget_inputs(tmp_path)
    step = preflight.step_budget(defn, problem, specs, runs, sizes, free=100.0)
    assert step.verdict == "pass", step.summary
    b = step.detail
    nodes = sum(_mesh_size(problem, s)[0] for s in train)
    elements = sum(_mesh_size(problem, s)[1] for s in train)
    counts = (b["production_cases"], b["completed_cases"], b["remaining_cases"])
    assert counts == (4, 0, 4)
    assert b["bytes_per_node"] == pytest.approx(1e6)
    assert b["wall_s_per_element"] == pytest.approx(10.0)
    assert b["estimated_gb"] == pytest.approx(nodes * 1e6 / 1e9)
    assert b["estimated_wall_h"] == pytest.approx(elements * 10.0 / 3600)
    assert (b["wall_s_median"], b["wall_s_max"]) == (120.0, 160.0)
    assert b["bytes_per_case"] == 21_000_000  # kept for run's estimate line
    assert b["per_level"]["1"] == {"n": 1, "wall_s_median": 10.0, "wall_s_max": 10.0}
    assert (b["min_free_gb"], b["free_gb"]) == (5.0, 100.0)
    tight = preflight.step_budget(defn, problem, specs, runs, sizes, free=5.0)
    assert tight.verdict == "fail"
    none = preflight.step_budget(defn, problem, specs, {}, {}, free=100.0)
    assert none.verdict == "not_assessable"


def test_the_budget_counts_only_what_production_has_left(tmp_path):
    defn, problem, specs, runs, sizes, train = _budget_inputs(tmp_path)
    # completed production: case id -> the sha256 of the deck that ran
    current = {
        s.case_id: hashlib.sha256(
            problem.input_deck(dict(s.params), s.variant).encode("utf-8")
        ).hexdigest()
        for s in train
    }
    done = preflight.step_budget(
        defn, problem, specs, runs, sizes, free=1.0, completed=current
    )
    assert done.verdict == "pass" and "nothing left" in done.summary
    assert done.detail["remaining_cases"] == 0 and done.detail["estimated_gb"] == 0.0
    first_two = {cid: current[cid] for cid in list(current)[:2]}
    half = preflight.step_budget(
        defn, problem, specs, runs, sizes, free=100.0, completed=first_two
    )
    nodes_left = sum(_mesh_size(problem, s)[0] for s in train[2:])
    assert half.detail["remaining_cases"] == 2
    assert half.detail["estimated_gb"] == pytest.approx(nodes_left * 1e6 / 1e9)
    # review finding 2: runs of another definition's decks must be made again
    stale = dict.fromkeys(current, "0" * 64)
    other = preflight.step_budget(
        defn, problem, specs, runs, sizes, free=100.0, completed=stale
    )
    assert other.detail["remaining_cases"] == 4
    assert other.detail["completed_cases"] == 0
    assert "nothing left" not in other.summary


def test_a_pilot_mesh_without_nodes_is_not_assessable(tmp_path):
    defn, _, specs, runs, sizes, _ = _budget_inputs(tmp_path)
    empty = SimpleNamespace(
        mesh=lambda p: SimpleNamespace(
            node_labels=np.zeros(0), connectivity=np.zeros((0, 4))
        )
    )
    step = preflight.step_budget(defn, empty, specs, runs, sizes, free=100.0)
    assert step.verdict == "not_assessable" and "no nodes" in step.summary


# --- the stamp and the report --------------------------------------------------


def test_stamp_record_and_report_are_deterministic(tmp_path):
    ds, defn, _ = _load(tmp_path)
    specs = preflight.preflight_cases(defn)
    steps = [preflight.Step(name, "pass", "fine", {"n": 1}) for name in preflight.STEPS]
    steps[3] = preflight.Step("space", "review", "order not observed", {})
    stamp = preflight.stamp_record(
        defn,
        ds,
        steps,
        specs,
        created_utc="2026-09-27T00:00:00+00:00",
        siblings_sha256="0" * 64,
    )
    assert stamp["format"] == "preflight-stamp/1" and stamp["passed"] is False
    assert stamp["definition_sha256"] == defn.sha256()
    assert stamp["problem_sha256"] == definition.problem_sha256(ds)
    assert stamp["siblings_sha256"] == "0" * 64
    assert stamp["dataset"] == "toy" and stamp["created_utc"].startswith("2026")
    assert set(stamp["cases"]) == set(preflight.ROLES)
    assert stamp["cases"]["conformance"] == ["TOY-pilot-0000-E"]
    assert list(stamp["steps"]) == list(preflight.STEPS)
    assert stamp["steps"]["space"]["verdict"] == "review"
    assert set(stamp["structbench"]) == {"version", "commit"}
    assert stamp["budget"] == {"n": 1}
    text = preflight.render_report(stamp)
    assert text == preflight.render_report(json.loads(json.dumps(stamp)))
    assert "review" in text and "space" in text and "TOY-pilot-0000-E" in text
    preflight.write_outputs(tmp_path / "out", stamp)
    raw = (tmp_path / "out" / "stamp.json").read_bytes()
    assert raw.endswith(b"\n") and b"\r\n" not in raw and json.loads(raw) == stamp
    assert (tmp_path / "out" / "report.md").read_bytes() == text.encode("utf-8")
    assert preflight.passed([preflight.Step("x", "not_applicable", "", {})]) is True
    assert preflight.passed([preflight.Step("x", "not_assessable", "", {})]) is False


def test_judge_sweep_returns_the_report_that_validate_sweep_prints(tmp_path):
    from test_verify import _sweep

    sweep, dataset = _sweep(tmp_path)
    record, report = verify.judge_sweep(sweep, dataset)
    assert [c.case_id for c in report.cases] == ["T-0000", "T-0001"]
    assert [c.case_id for c in record.cases] == ["T-0000", "T-0001"]
    out = sweep / "datacheck"
    assert (out / "measurements.json").is_file() and (out / "report.md").is_file()
    assert verify.validate_sweep(sweep, dataset) == 1


# --- the review's findings (plan 2b fix pass) ----------------------------------


def _closed_arrays():
    base = _FIXTURE._arrays()
    base["history/S/Assembly Assembly-1/ALLPW"] = base[
        "history/S/Assembly Assembly-1/ALLWK"
    ].copy()
    return _history(base, ALLCD=0.0, ALLPW=0.0, ETOTAL=11.0)


def test_non_finite_values_never_pass(tmp_path):
    """Review finding 5: NaN is never inside a tolerance."""
    _, defn, _ = _load(tmp_path)
    specs = preflight.preflight_cases(defn)
    qois = {
        "TOY-pilot-0000-L2": {"length": 1.0},
        "TOY-pilot-0000-T0p25": {"length": float("nan")},
        "TOY-pilot-0001-L2": {"length": 2.0},
        "TOY-pilot-0001-T0p25": {"length": 2.0},
    }
    step = preflight.step_increment(qois, {}, specs, defn, None)
    assert step.verdict == "not_assessable"
    assert "TOY-pilot-0000-T0p25" in step.summary and "length" in step.summary
    cases = {
        "TOY-pilot-0000-L2": _toy_case(lambda t: t),
        "TOY-pilot-0001-L2": _toy_case(lambda t: t),
    }
    nan_qoi = SimpleNamespace(qoi=lambda c: {"length": float("nan")})
    step = preflight.step_duration(cases, specs, defn, nan_qoi)
    assert step.verdict == "not_assessable" and "length" in step.summary
    record = _record()
    errors = record["cases"][0]["extrapolation"]["length"]["error_vs_extrapolated"]
    errors["2"] = float("nan")
    assert preflight.step_space(record, defn).verdict == "not_assessable"
    np.savez(tmp_path / "nan.npz", **_history(_closed_arrays(), ALLKE=float("nan")))
    conformance = preflight.step_conformance(tmp_path / "nan.npz", "t-mm-s")
    assert conformance.verdict == "fail" and "ALLKE" in conformance.summary
    force = np.linspace(0.0, 1.0, 41)
    force[3] = np.nan
    frame = preflight.step_frame(
        _toy_case(lambda t: 1 - np.exp(-t / 0.3), 41, force=force), None, defn
    )
    assert frame.verdict == "not_assessable" and "reaction_force" in frame.summary


def test_a_qoi_that_raises_on_a_short_prefix_is_a_finding_not_a_crash(tmp_path):
    """Review finding 6."""
    _, defn, _ = _load(tmp_path)
    specs = preflight.preflight_cases(defn)

    def qoi(c):
        if len(c.response.time) < 3:
            raise ValueError("needs three frames")
        return {"length": 1.0}

    cases = {
        "TOY-pilot-0000-L2": _toy_case(lambda t: t),
        "TOY-pilot-0001-L2": _toy_case(lambda t: t),
    }
    step = preflight.step_duration(cases, specs, defn, SimpleNamespace(qoi=qoi))
    assert step.verdict == "not_assessable"
    assert "needs three frames" in step.summary and "TOY-pilot-0000-L2" in step.summary


def test_energy_and_increment_summaries_say_what_was_not_assessed(tmp_path):
    """Review finding 8: a pass must not read as if every row were judged."""
    _, defn, _ = _load(tmp_path)
    specs = preflight.preflight_cases(defn)
    report = _report(
        [
            ("A", "plastic_dissipation_excess_max", "pass", 0.0),
            ("A", "energy_gain_max", "not_assessable", None),
            ("A", "energy_loss_max", "not_assessable", None),
        ]
    )
    step = preflight.step_energy(report, ["A"])
    assert step.verdict == "pass"
    assert "not assessed" in step.summary and "energy_gain_max" in step.summary
    qois = {
        "TOY-pilot-0000-L2": {"length": 1.0},
        "TOY-pilot-0000-T0p25": {"length": 1.0},
        "TOY-pilot-0001-L2": {"length": 2.0},
        "TOY-pilot-0001-T0p25": {"length": 2.0},
    }
    without = preflight.step_increment(qois, {}, specs, defn, None)
    assert without.verdict == "pass"
    assert "energy rows hold" not in without.summary
    assert "not assessed" in without.summary


def test_the_stamp_carries_no_absolute_path(tmp_path):
    """Review finding 14."""
    garbage = tmp_path / "broken.npz"
    garbage.write_bytes(b"not an npz")
    step = preflight.step_conformance(garbage, "t-mm-s")
    assert step.verdict == "not_assessable"
    assert str(tmp_path) not in step.summary and "broken.npz" in step.summary


# --- accepted reviews (plan 3a, Task 3) -------------------------------------------


def test_step_space_names_its_reviews(tmp_path):
    _, defn, _ = _load(tmp_path)
    step = preflight.step_space(_record(status="oscillatory"), defn)
    assert step.verdict == "review" and step.reviews == ("space.length",)
    assert preflight.step_space(_record(), defn).reviews == ()


def _steps(**override):
    steps = {n: preflight.Step(n, "pass", "fine", {}) for n in preflight.STEPS}
    steps.update(override)
    return list(steps.values())


def test_accepted_reviews_rescue_a_review_and_nothing_else():
    review = preflight.Step("space", "review", "no order", {}, ("space.length",))
    ok = {"space.length": "moves under 0.1 %"}
    assert preflight.passed(_steps(space=review), ok) is True
    assert preflight.passed(_steps(space=review), {}) is False
    two = preflight.Step("space", "review", "", {}, ("space.length", "space.width"))
    assert preflight.passed(_steps(space=two), ok) is False
    failing = preflight.Step("frame", "fail", "unresolved", {})
    assert preflight.passed(_steps(space=review, frame=failing), ok) is False
    missing = preflight.Step("budget", "not_assessable", "no pilot", {})
    assert preflight.passed(_steps(space=review, budget=missing), ok) is False
    keyless = preflight.Step("space", "review", "", {})
    assert preflight.passed(_steps(space=keyless), ok) is False
    assert preflight.accepted(_steps(space=review), ok) == ok
    assert preflight.accepted(_steps(space=two), ok) == ok


def test_the_stamp_lists_accepted_unaccepted_and_unused_reviews(tmp_path):
    acceptances = (
        'accepted_reviews = { "space.length" = "moves under 0.1 % between levels", '
        '"increment.length" = "never needed" }'
    )
    ds, defn, _ = _load(tmp_path, PF_TOML.replace(_GAPS, f"{_GAPS}\n{acceptances}"))
    specs = preflight.preflight_cases(defn)
    frame = preflight.Step(
        "frame",
        "pass",
        "resolved",
        {
            "reported": {
                "node/acceleration": {"error": 0.9, "reason": "second derivative"}
            }
        },
    )
    two = preflight.Step("space", "review", "", {}, ("space.length", "space.width"))
    stamp = preflight.stamp_record(
        defn,
        ds,
        _steps(space=two, frame=frame),
        specs,
        created_utc="2026-09-28T00:00:00+00:00",
        siblings_sha256="0" * 64,
    )
    assert stamp["accepted_reviews"] == {
        "space.length": "moves under 0.1 % between levels"
    }
    assert stamp["unaccepted_reviews"] == ["space.width"]
    assert stamp["unused_acceptances"] == ["increment.length"]
    assert stamp["steps"]["space"]["reviews"] == ["space.length", "space.width"]
    assert stamp["passed"] is False
    report = preflight.render_report(stamp)
    assert "moves under 0.1 % between levels" in report
    assert "second derivative" in report and "space.width" in report
    assert "**Not passed.** Blocking: space (review)" in report
    one = preflight.Step("space", "review", "", {}, ("space.length",))
    stamp = preflight.stamp_record(
        defn,
        ds,
        _steps(space=one, frame=frame),
        specs,
        created_utc="2026-09-28T00:00:00+00:00",
        siblings_sha256="0" * 64,
    )
    assert stamp["passed"] is True and stamp["unaccepted_reviews"] == []
    report = preflight.render_report(stamp)
    assert "**Passed.**" in report and "moves under 0.1 % between levels" in report
    # the verdict is recomputable from the stamp alone
    again = [
        preflight.Step(n, s["verdict"], s["summary"], s["detail"], tuple(s["reviews"]))
        for n, s in stamp["steps"].items()
    ]
    assert preflight.passed(again, stamp["accepted_reviews"]) is stamp["passed"]


# --- the plan 3a review's findings -------------------------------------------------


def test_the_frame_step_needs_a_judged_node_field(tmp_path):
    """Review finding 1: reporting every node field judges nothing the clock is for."""
    declared = 'frame_reported = { "node/displacement" = "why" }'
    _, defn, _ = _load(tmp_path, PF_TOML.replace(_GAPS, f"{_GAPS}\n{declared}"))
    case = _toy_case(lambda t: 1 - np.exp(-t / 0.3), 41, force=np.linspace(0, 1, 41))
    step = preflight.step_frame(case, None, defn)
    assert step.verdict == "not_assessable" and "node field" in step.summary
    everything = (
        'frame_reported = { "node/displacement" = "why", "solid/stress" = "why", '
        '"global/reaction_force" = "why" }'
    )
    _, all_out, _ = _load(
        tmp_path, PF_TOML.replace(_GAPS, f"{_GAPS}\n{everything}"), name="all"
    )
    assert preflight.step_frame(case, None, all_out).verdict == "not_assessable"


def test_a_review_beyond_tolerance_is_a_failure(tmp_path):
    """Review finding 3 (ruling): a quantity with no observed order whose levels
    differ by more than its tolerance does not converge; only 'barely moves'
    is left to a person."""
    _, defn, _ = _load(tmp_path)  # tolerance 0.01
    for status in ("diverging", "oscillatory", "flat"):
        far = preflight.step_space(_record(status=status, coarsest=0.3), defn)
        assert far.verdict == "fail", (status, far.summary)
        assert "30 %" in far.summary and "across levels" in far.summary
    near = preflight.step_space(_record(status="oscillatory", coarsest=0.005), defn)
    assert near.verdict == "review" and near.reviews == ("space.length",)
    assert "0.5 % across levels" in near.summary
    unknown = preflight.step_space(_record(status="diverging", coarsest=None), defn)
    assert unknown.verdict == "fail"


def test_a_failing_space_step_still_names_its_reviews(tmp_path):
    """Review finding 4: an acceptance is not 'unused' when its review occurred
    beside a failure."""
    _, defn, _ = _load(tmp_path)
    record = _record(error=0.02)  # a monotone case outside its tolerance: fail
    other = _record(status="oscillatory")["cases"][0]
    record["cases"].append({**other, "case_id": "TOY-pilot-0001-L2"})
    step = preflight.step_space(record, defn)
    assert step.verdict == "fail" and step.reviews == ("space.length",)
    acc = {"space.length": "why"}
    assert preflight.passed([step], acc) is False
    assert preflight.accepted([step], acc) == acc


def test_a_mesh_hook_that_raises_in_the_budget_is_a_finding(tmp_path):
    """Review finding 5."""
    defn, problem, specs, runs, sizes, _ = _budget_inputs(tmp_path)

    def mesh(params):
        if 1.0 < float(params["L"]) < 2.0:
            raise ValueError("no mesh for this length")
        return problem.mesh(params)

    step = preflight.step_budget(
        defn, SimpleNamespace(mesh=mesh), specs, runs, sizes, free=100.0
    )
    assert step.verdict == "not_assessable"
    assert "mesh() raised" in step.summary and "no mesh for this length" in step.summary


def test_completed_reads_each_run_and_the_deck_it_ran(tmp_path):
    sweep = tmp_path / "sweep"
    for cid, status, sha in (("A", "completed", "abc"), ("B", "error", "def")):
        (sweep / cid).mkdir(parents=True)
        (sweep / cid / "run.json").write_text(json.dumps({"status": status}))
        (sweep / cid / "provenance.json").write_text(json.dumps({"inp_sha256": sha}))
    (sweep / "preflight").mkdir()
    assert preflight._completed(sweep) == {"A": "abc"}
