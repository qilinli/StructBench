"""The preflight: the pilot split through the three resolutions, the energy
account, the budget and the verification instrument, ending in a stamp
(ADR-0071, part two (b)).

    structbench-datagen preflight --dataset <dir> --work-root <runs>
        [--abaqus EXE] [--workers N] [--timeout S]

The stage plans a case set from the pilot points -- every pilot at every
``[levels].pilot`` level but the finest (the ``fine_cases`` there too), the
pilots at production with the solver increment scaled, one pilot exported at
a finer frame interval, one conformance run with every energy term requested
-- and drives ``run``, ``export``, ``convert``, ``verify`` and ``converge``
over it in ``<work-root>/<name>/preflight/``, a sweep of its own. Ten steps
turn the records into verdicts; ``report.md`` and ``stamp.json`` are written
beside the cases, and ``generate`` opens splits that are not probes only
against a passing stamp for the current definition. Re-running resumes:
every stage skips what is already done, and a preflight folder holding
another definition's runs is refused, never cleared. Exit codes: 0 passed,
1 a step failed or needs review (the stamp says so), 2 refused, 3 stopped by
the environment.

The step functions below are pure: each turns records (a convergence record,
a verification report, canonical cases, run records) into a ``Step`` with a
verdict from ADR-0066's vocabulary, so they are tested without a solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from structbench.core import Case
from structbench.core.io import read_case
from structbench.core.io.abaqus import (
    LEDGER_CLOSED_TERMS,
    assembly_history,
    read_abaqus_export,
)
from structbench.datagen import converge as convergence_stage
from structbench.datagen import sampling
from structbench.datagen.abaqus.deck import with_all_energy
from structbench.datagen.converge import level_order
from structbench.datagen.convert import convert_sweep
from structbench.datagen.definition import (
    Definition,
    load_definition,
    load_problem,
    problem_sha256,
    siblings_sha256,
)
from structbench.datagen.export import export_cases
from structbench.datagen.generate import (
    CaseSpec,
    case_id_for,
    materialise,
    package_state,
    plan_cases,
)
from structbench.datagen.run import NOT_LAUNCHED, case_state, free_gb, run_sweep
from structbench.datagen.template import (
    check_definition,
    deck_sha256_in_fresh_interpreter,
)
from structbench.datagen.verify import judge_sweep
from structbench.verification.results import Verdict
from structbench.verification.temporal import (
    common_instant_errors,
    interpolation_errors,
    qoi_history,
    rise_time_frames,
    separation_frame,
    settling_frame,
)

STAMP_FORMAT = "preflight-stamp/1"
PREFLIGHT_DIR = "preflight"
ROLES = ("level", "increment", "frame", "conformance")
#: The steps in the order the report prints them.
STEPS = (
    "deck_regression",
    "feasibility",
    "conformance",
    "space",
    "increment",
    "frame",
    "duration",
    "energy",
    "budget",
    "verification",
)
ENERGY_ROWS = (
    "energy_gain_max",
    "energy_loss_max",
    "energy_residual_final",
    "kinetic_energy_closure",
    "plastic_dissipation_excess_max",
    "stored_globals_match_ledger",
)
#: The terms the ledger identity needs (``abaqus_ledger`` asks for the same).
REQUIRED_TERMS = frozenset({"ALLKE", "ALLIE", "ALLVD", "ALLWK", "ETOTAL"})
#: The identity must close to this relative residual: float32 storage of the
#: history, with margin.
IDENTITY_TOLERANCE = 1e-5
#: The verdicts that let the stamp pass.
PASSING = frozenset({"pass", "not_applicable"})


@dataclass(frozen=True)
class Step:
    name: str
    verdict: str  # pass | fail | review | not_applicable | not_assessable
    summary: str
    detail: dict[str, Any]
    #: The "<step>.<name>" keys behind a review verdict, which a dataset may
    #: accept with a reason ([pilot].accepted_reviews; plan 3a).
    reviews: tuple[str, ...] = ()


def _rescued(step: Step, acceptances: Mapping[str, str]) -> bool:
    """A review whose every key a person accepted; never a fail or an absence."""
    return (
        step.verdict == "review"
        and bool(step.reviews)
        and all(k in acceptances for k in step.reviews)
    )


def passed(steps: Sequence[Step], acceptances: Mapping[str, str] | None = None) -> bool:
    """Every step passes or does not apply, or is a review the dataset accepted."""
    acc = acceptances or {}
    return all(s.verdict in PASSING or _rescued(s, acc) for s in steps)


def accepted(steps: Sequence[Step], acceptances: Mapping[str, str]) -> dict[str, str]:
    """The review keys that occurred and are accepted, with their reasons."""
    keys = sorted({k for s in steps for k in s.reviews})
    return {k: acceptances[k] for k in keys if k in acceptances}


def not_run(name: str, why: str) -> Step:
    return Step(name, "not_assessable", f"not run: {why}", {})


# --- the case set --------------------------------------------------------------


def label_suffix(role: str, label: str | float | None) -> str:
    """The id suffix of a preflight case: -L<level>, -T<value>, -F, -E."""
    if role == "level":
        return "-L" + str(label).replace(".", "p")
    if role == "increment":
        return "-T" + f"{float(label):g}".replace(".", "p")  # type: ignore[arg-type]
    if role == "frame":
        return "-F"
    if role == "conformance":
        return "-E"
    raise ValueError(f"unknown preflight role {role!r}")


def production_value(defn: Definition) -> float | None:
    """The production value of the increment key: ``[fixed]``'s, or None when
    the key is not a declared constant (then the probe cannot be built)."""
    value = defn.fixed.get(defn.pilot.increment_key)
    return None if value is None else float(value)


def by_role(specs: Sequence[CaseSpec], role: str) -> list[CaseSpec]:
    return [s for s in specs if (s.probe or {}).get("role") == role]


def batch_a(specs: Sequence[CaseSpec], production: str) -> list[str]:
    """The cases run first: the pilots at the production level, and the
    conformance run. Their steps decide whether the rest is worth launching."""
    levels = [
        s.case_id
        for s in by_role(specs, "level")
        if (s.probe or {}).get("level") == production
    ]
    return levels + [s.case_id for s in by_role(specs, "conformance")]


def preflight_cases(defn: Definition) -> list[CaseSpec]:
    """The preflight's case set, from the pilot points.

    Role ``level``: every pilot at every ``[levels].pilot`` level except the
    finest, always at the production level, and at the finest too when its
    plain id is in ``[pilot].fine_cases``. Role ``increment``: every pilot at
    the production level with the increment key at the production value times
    each factor. Role ``frame``: one pilot (the first fine case, else the
    first pilot) at the production level with the frame interval scaled by
    ``frame_factor`` and the frame count divided by it, when the probe is
    enabled and both keys are constants. Role ``conformance``: the first pilot
    at the production level (its deck is widened by ``deck_for``).
    """
    pilot = defn.pilot
    split = defn.split(pilot.split)
    key, production = defn.levels.refine_key, defn.levels.production
    levels = list(defn.levels.pilot)
    finest = levels[-1]
    pilots: list[tuple[int, str | None, dict[str, Any], str]] = []
    for point in sampling.sample_split(defn.variables, defn.regions, split, None):
        for variant in split.variants or (None,):
            plain = case_id_for(defn, split.name, point.index, variant)
            pilots.append((point.index, variant, {**defn.fixed, **point.params}, plain))

    def spec(
        index: int,
        variant: str | None,
        params: dict[str, Any],
        role: str,
        level: str,
        factor: float | None,
        suffix: str,
    ) -> CaseSpec:
        return CaseSpec(
            case_id_for(defn, split.name, index, variant, suffix),
            split.name,
            index,
            variant,
            split.seed,
            params,
            probe={"role": role, "level": level, "factor": factor},
        )

    specs: list[CaseSpec] = []
    for index, variant, base, plain in pilots:
        own = [
            lv
            for lv in levels
            if lv != finest or lv == production or plain in pilot.fine_cases
        ]
        for lv in own:
            params = {**base, key: lv}
            specs.append(
                spec(
                    index, variant, params, "level", lv, None, label_suffix("level", lv)
                )
            )
    base_value = production_value(defn)
    scaled = (
        [(factor, base_value * factor) for factor in pilot.increment_factors]
        if base_value is not None
        else []
    )
    for factor, value in scaled:
        for index, variant, base, _plain in pilots:
            params = {**base, key: production, pilot.increment_key: value}
            suffix = label_suffix("increment", value)
            specs.append(
                spec(index, variant, params, "increment", production, factor, suffix)
            )
    keys_fixed = pilot.frame_key in defn.fixed and pilot.frame_count_key in defn.fixed
    if pilot.frame_factor > 0.0 and keys_fixed:
        chosen = next((p for p in pilots if p[3] in pilot.fine_cases), pilots[0])
        index, variant, base, plain = chosen
        interval = float(base[pilot.frame_key]) * pilot.frame_factor
        count = round(float(base[pilot.frame_count_key]) / pilot.frame_factor)
        params = {
            **base,
            key: production,
            pilot.frame_key: interval,
            pilot.frame_count_key: count,
        }
        specs.append(
            spec(index, variant, params, "frame", production, pilot.frame_factor, "-F")
        )
    index, variant, base, plain = pilots[0]
    specs.append(
        spec(
            index,
            variant,
            {**base, key: production},
            "conformance",
            production,
            None,
            "-E",
        )
    )
    return specs


def deck_for(spec: CaseSpec, problem: ModuleType) -> str:
    """The case's deck; the conformance run's with every energy term requested."""
    text: str = problem.input_deck(dict(spec.params), spec.variant)
    if (spec.probe or {}).get("role") == "conformance":
        return with_all_energy(text)
    return text


def _pilot_key(spec: CaseSpec) -> tuple[int, str | None]:
    return spec.index, spec.variant


# --- the steps -----------------------------------------------------------------


def step_deck_regression(
    dataset_dir: Path, specs: Sequence[CaseSpec], problem: ModuleType, defn: Definition
) -> Step:
    """Every preflight deck rebuilds byte for byte in a fresh interpreter, and
    every probe key changes the deck it is meant to change."""
    decks: dict[str, str] = {}
    unstable: list[str] = []
    failures: list[str] = []
    for spec in specs:
        text: str = problem.input_deck(dict(spec.params), spec.variant)
        decks[spec.case_id] = text
        try:
            fresh = deck_sha256_in_fresh_interpreter(
                dataset_dir, spec.params, spec.variant
            )
        except RuntimeError as exc:
            failures.append(f"{spec.case_id}: {exc}")
            continue
        if fresh != hashlib.sha256(text.encode("utf-8")).hexdigest():
            unstable.append(spec.case_id)
    production = defn.levels.production
    production_deck = {
        _pilot_key(s): decks[s.case_id]
        for s in by_role(specs, "level")
        if (s.probe or {}).get("level") == production
    }
    unread: set[str] = set()
    per_pilot: dict[tuple[int, str | None], dict[str, str]] = {}
    for s in by_role(specs, "level"):
        per_pilot.setdefault(_pilot_key(s), {})[str((s.probe or {})["level"])] = decks[
            s.case_id
        ]
    for by_level in per_pilot.values():
        if len(set(by_level.values())) < len(by_level):
            unread.add(defn.levels.refine_key)
    for s in by_role(specs, "increment"):
        if decks[s.case_id] == production_deck.get(_pilot_key(s)):
            unread.add(defn.pilot.increment_key)
    for s in by_role(specs, "frame"):
        if decks[s.case_id] == production_deck.get(_pilot_key(s)):
            unread.add(defn.pilot.frame_key)
    detail = {
        "checked": len(specs),
        "unstable": unstable,
        "unread_keys": sorted(unread),
        "failures": failures,
    }
    problems = []
    for key in sorted(unread):
        problems.append(
            f"{key!r} leaves the deck unchanged: problem.input_deck does not read it"
        )
    if unstable:
        problems.append(
            f"{len(unstable)} deck(s) differ in a fresh interpreter: "
            + ", ".join(unstable)
        )
    problems += failures
    if problems:
        return Step("deck_regression", "fail", "; ".join(problems), detail)
    return Step(
        "deck_regression",
        "pass",
        f"all {len(specs)} decks rebuild byte for byte and every probe key is read",
        detail,
    )


def step_feasibility(
    specs: Sequence[CaseSpec],
    states: Mapping[str, str],
    defn: Definition,
    problem: ModuleType,
) -> Step:
    """Every pilot completes at the production level (and the conformance run)."""
    production = defn.levels.production
    batch = batch_a(specs, production)
    not_done = [cid for cid in batch if states.get(cid) != "done"]
    feasible = getattr(problem, "feasible", None)
    checks: dict[str, bool | None] = {}
    for s in by_role(specs, "level"):
        if (s.probe or {}).get("level") == production:
            checks[s.case_id] = (
                None
                if feasible is None
                else bool(feasible({**defn.fixed, **defn.limits, **s.params}))
            )
    detail = {
        "cases": {cid: states.get(cid, "missing") for cid in batch},
        "limits": dict(defn.limits),
        "feasible": checks,
    }
    if not_done:
        return Step(
            "feasibility",
            "fail",
            f"{len(not_done)} of {len(batch)} production-level runs did not "
            "complete: "
            + ", ".join(f"{cid} ({states.get(cid, 'missing')})" for cid in not_done),
            detail,
        )
    return Step(
        "feasibility",
        "pass",
        f"all {len(batch)} production-level runs completed",
        detail,
    )


def step_conformance(npz_path: Path | None, units: str) -> Step:
    """The conformance run: the ledger identity closes with the standard terms
    and no term outside it is non-zero."""
    if npz_path is None or not Path(npz_path).is_file():
        return Step(
            "conformance", "not_assessable", "the conformance run has no export", {}
        )
    path = Path(npz_path)
    try:
        history = assembly_history(read_abaqus_export(path))
    except Exception as exc:  # an unreadable export is a finding about it
        why = str(exc).replace(str(path), path.name).replace(str(path.parent), "…")
        return Step(
            "conformance",
            "not_assessable",
            f"the conformance export {path.name} could not be read: "
            f"{type(exc).__name__}: {why}",
            {},
        )
    missing = sorted(REQUIRED_TERMS - set(history))
    extra = sorted(
        term
        for term, values in history.items()
        if term not in LEDGER_CLOSED_TERMS and bool(np.any(values != 0.0))
    )
    non_finite = sorted(
        term
        for term, values in history.items()
        if not bool(np.all(np.isfinite(values)))
    )
    residual: float | None = None
    problems: list[str] = []
    if non_finite:
        problems.append("non-finite values in " + ", ".join(non_finite))
    if missing:
        problems.append("missing " + ", ".join(missing))
    elif not non_finite:
        total = history["ETOTAL"] + history["ALLWK"]
        balance = history["ALLKE"] + history["ALLIE"] + history["ALLVD"]
        if "ALLFD" in history and "ALLPW" in history:
            balance = balance + history["ALLFD"] - history["ALLPW"]
        scale = float(np.max(np.abs(total)))
        if scale > 0.0:
            residual = float(np.max(np.abs(balance - total)) / scale)
            if residual > IDENTITY_TOLERANCE:
                problems.append(
                    f"the ledger identity misses by {residual:.3g} "
                    f"(> {IDENTITY_TOLERANCE:g})"
                )
        else:
            problems.append(
                "the solver total is zero throughout; the identity cannot be checked"
            )
    if extra:
        problems.append("non-zero outside the identity: " + ", ".join(extra))
    detail = {
        "terms": sorted(history),
        "missing": missing,
        "extra_nonzero": extra,
        "identity_residual": residual,
        "tolerance": IDENTITY_TOLERANCE,
        "units": units,
    }
    if problems:
        return Step("conformance", "fail", "; ".join(problems), detail)
    return Step(
        "conformance",
        "pass",
        f"the ledger identity closes to {residual:.2g} with the standard terms "
        "and no other term is non-zero",
        detail,
    )


def step_space(record: Mapping[str, Any], defn: Definition) -> Step:
    """Space: each QoI at the production level against its extrapolated value."""
    tol = dict(zip(defn.qoi.names, defn.qoi.tolerance, strict=True))
    production = defn.levels.production
    cases_detail: dict[str, dict[str, Any]] = {}
    fails: list[str] = []
    reviews: list[str] = []
    review_keys: set[str] = set()
    unassessable: list[str] = []
    assessed = 0
    for entry in record.get("cases", []):
        per: dict[str, Any] = {}
        for name in defn.qoi.names:
            x = (entry.get("extrapolation") or {}).get(name)
            if not x:
                continue
            assessed += 1
            errors = x.get("error_vs_extrapolated") or {}
            err = errors.get(production)
            per[name] = {
                "status": x.get("status"),
                "order": x.get("order"),
                "error_at_production": err,
            }
            if x.get("status") != "monotone" or err is None:
                reviews.append(f"{entry['case_id']}: {name} {x.get('status')}")
                review_keys.add(f"space.{name}")
            elif not math.isfinite(err):
                unassessable.append(f"{entry['case_id']}: {name} is not finite")
            elif err > tol[name]:
                fails.append(
                    f"{entry['case_id']}: {name} {err:.3g} > {tol[name]:g} at level "
                    f"{production}"
                )
        if per:
            cases_detail[entry["case_id"]] = per
    detail = {
        "cases": cases_detail,
        "fields": dict(record.get("summary", {}).get("fields", {})),
        "tolerance": tol,
    }
    if assessed == 0:
        return Step(
            "space",
            "not_assessable",
            "no pilot ran at three levels in constant ratio, so no quantity of "
            "interest could be extrapolated",
            detail,
        )
    if fails:
        return Step("space", "fail", "; ".join(fails), detail)
    if unassessable:
        return Step("space", "not_assessable", "; ".join(unassessable), detail)
    if reviews:
        return Step(
            "space",
            "review",
            "no observed order for " + "; ".join(reviews),
            detail,
            tuple(sorted(review_keys)),
        )
    return Step(
        "space",
        "pass",
        f"every quantity of interest at level {production} is within its "
        f"tolerance of the extrapolated value ({assessed} extrapolations)",
        detail,
    )


def _energy_rows(report: Any, case_ids: Collection[str]) -> tuple[dict, dict, list]:
    ids = set(case_ids)
    rows: dict[str, Counter[str]] = {}
    values: dict[str, list[float]] = {}
    fails: list[str] = []
    for case in report.cases:
        if case.case_id not in ids:
            continue
        for r in case.results:
            if r.quantity not in ENERGY_ROWS:
                continue
            rows.setdefault(r.quantity, Counter())[str(r.verdict)] += 1
            if r.value is not None:
                values.setdefault(r.quantity, []).append(float(r.value))
            if r.verdict is Verdict.FAIL:
                fails.append(f"{case.case_id}: {r.quantity}")
    return rows, values, fails


def step_increment(
    qois: Mapping[str, Mapping[str, float]],
    fields: Mapping[str, Mapping[str, float]],
    specs: Sequence[CaseSpec],
    defn: Definition,
    report: Any,
) -> Step:
    """Time (a): the QoIs at another stable-increment scale, and the energy
    rows of those runs."""
    key = defn.pilot.increment_key
    if not defn.pilot.increment_factors:
        return Step(
            "increment",
            "not_applicable",
            "the dataset declares no increment factors",
            {},
        )
    if production_value(defn) is None:
        return Step(
            "increment",
            "not_assessable",
            f"{key!r} is not in [fixed]; the increment probe scales the production "
            "value and needs it declared as a constant",
            {},
        )
    increments = by_role(specs, "increment")
    tol = dict(zip(defn.qoi.names, defn.qoi.tolerance, strict=True))
    production = defn.levels.production
    base_of = {
        _pilot_key(s): s.case_id
        for s in by_role(specs, "level")
        if (s.probe or {}).get("level") == production
    }
    pilots: dict[str, dict[str, dict[str, float]]] = {}
    fails: list[str] = []
    missing: list[str] = []
    for s in increments:
        plain = case_id_for(defn, s.split, s.index, s.variant)
        base = base_of.get(_pilot_key(s))
        if base is None or base not in qois or s.case_id not in qois:
            missing.append(s.case_id)
            continue
        value = float(s.params[key])
        diffs: dict[str, float] = {}
        for name in defn.qoi.names:
            a, b = qois[base].get(name), qois[s.case_id].get(name)
            if a is None or b is None:
                missing.append(f"{s.case_id} ({name})")
                continue
            if not (math.isfinite(float(a)) and math.isfinite(float(b))):
                missing.append(f"{s.case_id} ({name} is not finite)")
                continue
            rel = abs(float(b) - float(a)) / max(abs(float(a)), 1e-300)
            diffs[name] = rel
            if rel > tol[name]:
                fails.append(
                    f"{plain}: {name} changes by {rel:.3g} at {key}={value:g} "
                    f"(> {tol[name]:g})"
                )
        pilots.setdefault(plain, {})[f"{value:g}"] = diffs
    energy = None
    energy_note = "energy rows not assessed (no verification report)"
    if report is not None:
        rows, _, energy_fails = _energy_rows(report, [s.case_id for s in increments])
        energy = {q: dict(sorted(c.items())) for q, c in rows.items()}
        fails += energy_fails
        unassessed = _unassessed_rows(rows)
        energy_note = "no energy row fails" + (
            f" (not assessed: {', '.join(unassessed)})" if unassessed else ""
        )
    detail = {
        "pilots": pilots,
        "fields": {cid: dict(v) for cid, v in fields.items()},
        "production_value": production_value(defn),
        "energy_rows": energy,
    }
    if missing:
        return Step(
            "increment",
            "not_assessable",
            "no finite quantities of interest for " + ", ".join(missing),
            detail,
        )
    if fails:
        return Step("increment", "fail", "; ".join(fails), detail)
    return Step(
        "increment",
        "pass",
        f"every quantity of interest is within tolerance at {key} scaled by "
        + ", ".join(f"{f:g}" for f in defn.pilot.increment_factors)
        + f"; {energy_note}",
        detail,
    )


def _unassessed_rows(rows: Mapping[str, Counter[str]]) -> list[str]:
    """The energy rows no case received a verdict on (absent, or only
    ``not_assessable`` / ``not_applicable``)."""
    return [
        q
        for q in ENERGY_ROWS
        if q not in rows or not (set(rows[q]) - {"not_assessable", "not_applicable"})
    ]


def step_frame(
    frame_case: Case | None, production_case: Case | None, defn: Definition
) -> Step:
    """Time (b): the stored frame interval against a half-interval export."""
    p = defn.pilot
    if p.frame_factor <= 0.0:
        return Step(
            "frame",
            "not_applicable",
            "the frame probe is disabled by the dataset (frame_factor = 0)",
            {},
        )
    if frame_case is None:
        absent = [k for k in (p.frame_key, p.frame_count_key) if k not in defn.fixed]
        if absent:
            return Step(
                "frame",
                "not_assessable",
                "the frame probe needs "
                + " and ".join(repr(k) for k in absent)
                + " in [fixed]",
                {},
            )
        return Step(
            "frame",
            "not_assessable",
            "the frame probe's case has no canonical file",
            {},
        )
    assert frame_case.response is not None
    # The stored clock is every stride-th frame of the probe export: the
    # judged error is that clock's, whatever the factor (review finding 2).
    stride = round(1.0 / p.frame_factor)
    try:
        errors = interpolation_errors(frame_case, stride)
    except ValueError as exc:
        return Step("frame", "not_assessable", str(exc), {})
    # Judged: every stored field but those the dataset reports, each with its
    # reason (plan 3a); a declared key the case does not store is noted, not
    # silently accepted.
    declared = dict(p.frame_reported)
    judged = sorted(k for k in errors if k not in declared)
    reported = {
        k: {"error": errors[k], "reason": declared[k]}
        for k in sorted(errors)
        if k in declared
    }
    reported_absent = sorted(k for k in declared if k not in errors)
    not_finite = [k for k in judged if not math.isfinite(errors[k])]
    over = [f"{k} {errors[k]:.3g}" for k in judged if errors[k] > p.frame_tolerance]
    common: dict[str, float] | None = None
    note = None
    if production_case is not None and production_case.response is not None:
        try:
            common = common_instant_errors(production_case, frame_case)
        except ValueError as exc:
            note = str(exc)
    rise: dict[str, int] = {}
    for name, series in frame_case.response.globals_.items():
        r = rise_time_frames(np.asarray(series, dtype=np.float64)[::stride])
        if r is not None:
            rise[name] = r
    detail = {
        "stride": stride,
        "interpolation": errors,
        "common_instants": common,
        "judged": judged,
        "reported": reported,
        "reported_absent": reported_absent,
        "rise_frames": rise,
        "shortest_rise_frames": min(rise.values()) if rise else None,
        "tolerance": p.frame_tolerance,
        "factor": p.frame_factor,
        "note": note,
    }
    if not_finite:
        return Step(
            "frame",
            "not_assessable",
            "non-finite values in " + ", ".join(not_finite),
            detail,
        )
    if over:
        return Step(
            "frame",
            "fail",
            "the stored clock does not resolve "
            + ", ".join(over)
            + f" (interpolation error > {p.frame_tolerance:g})",
            detail,
        )
    return Step(
        "frame",
        "pass",
        f"the stored frame interval resolves all {len(judged)} judged fields to "
        f"{p.frame_tolerance:g} (linear interpolation of a {p.frame_factor:g}× "
        f"interval export, stride {stride}); reported, not judged: "
        + (", ".join(reported) or "none"),
        detail,
    )


def step_duration(
    cases: Mapping[str, Case],
    specs: Sequence[CaseSpec],
    defn: Definition,
    problem: ModuleType,
) -> Step:
    """Duration: when each pilot settles (and, if named, when contact ends)
    against the stored horizon and the margin."""
    production = defn.levels.production
    ids = [
        s.case_id
        for s in by_role(specs, "level")
        if (s.probe or {}).get("level") == production
    ]
    missing = [cid for cid in ids if cid not in cases]
    if missing:
        return Step(
            "duration",
            "not_assessable",
            "no canonical file for " + ", ".join(missing),
            {"missing": missing},
        )
    tol = dict(zip(defn.qoi.names, defn.qoi.tolerance, strict=True))
    global_name = defn.pilot.contact_force_global
    pilots: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    problems: list[str] = []
    frames: int | None = None
    horizon: float | None = None
    for cid in ids:
        case = cases[cid]
        assert case.response is not None
        r = case.response
        n = len(r.time)
        if frames is None:
            frames, horizon = n, float(r.time[-1] - r.time[0])
        elif n != frames:
            notes.append(f"{cid}: {n} frames where the first pilot has {frames}")
        try:
            history = qoi_history(case, problem.qoi)
        except Exception as exc:  # a QoI that cannot take a prefix is a finding
            problems.append(f"{cid}: qoi() raised {type(exc).__name__}: {exc}")
            continue
        bad = [q for q, v in history.items() if not bool(np.all(np.isfinite(v)))]
        if bad:
            problems.append(f"{cid}: {', '.join(bad)} not finite")
            continue
        settle = settling_frame(history, tol)
        separation: int | None = None
        if global_name is not None:
            series = r.globals_.get(global_name)
            if series is None:
                notes.append(f"{cid}: no stored global {global_name!r}")
            else:
                separation = separation_frame(series)
        pilots[cid] = {
            "settling_frame": settle,
            "separation_frame": separation,
            "share_after_settling": 1.0 - settle / (n - 1) if n > 1 else 0.0,
        }
    assert frames is not None
    if problems:
        return Step(
            "duration",
            "not_assessable",
            "; ".join(problems),
            {"problems": problems, "pilots": pilots, "frames": frames},
        )
    slowest = max(p["settling_frame"] for p in pilots.values())
    limit = int(math.floor((1.0 - defn.pilot.settling_margin) * (frames - 1)))
    detail = {
        "pilots": pilots,
        "frames": frames,
        "horizon": horizon,
        "slowest_settling_frame": slowest,
        "settling_limit_frame": limit,
        "margin": defn.pilot.settling_margin,
        "contact_force_global": global_name,
        "notes": notes,
    }
    if slowest > limit:
        who = ", ".join(
            f"{cid} (frame {p['settling_frame']})"
            for cid, p in pilots.items()
            if p["settling_frame"] > limit
        )
        return Step(
            "duration",
            "fail",
            f"settles after frame {limit} of {frames - 1} (margin "
            f"{defn.pilot.settling_margin:g}): {who}",
            detail,
        )
    return Step(
        "duration",
        "pass",
        f"the slowest pilot settles at frame {slowest} of {frames - 1}, within the "
        f"margin (limit frame {limit})",
        detail,
    )


def step_energy(report: Any, case_ids: Collection[str]) -> Step:
    """The energy account over the named cases: no energy row fails."""
    rows, values, fails = _energy_rows(report, case_ids)
    if not rows:
        return Step(
            "energy",
            "not_assessable",
            "no energy row was judged for the preflight cases",
            {},
        )
    detail = {
        "rows": {q: dict(sorted(c.items())) for q, c in rows.items()},
        "spread": {
            q: {
                "lowest": min(v),
                "median": float(np.median(v)),
                "highest": max(v),
                "n": len(v),
            }
            for q, v in values.items()
        },
    }
    if fails:
        return Step("energy", "fail", "; ".join(fails), detail)
    unassessed = _unassessed_rows(rows)
    return Step(
        "energy",
        "pass",
        f"no energy row fails on the {len(set(case_ids))} preflight cases"
        + (f"; not assessed: {', '.join(unassessed)}" if unassessed else ""),
        detail,
    )


def step_budget(
    defn: Definition,
    problem: ModuleType,
    specs: Sequence[CaseSpec],
    runs: Mapping[str, Mapping[str, Any]],
    sizes: Mapping[str, int],
    free: float,
    *,
    completed: Collection[str] = (),
) -> Step:
    """What production still has to generate, against the free space and the
    margin (plan 3a).

    The pilots give a rate, not a size: bytes per mesh node and wall time per
    element, each the median over the production-level pilots; the rate is
    applied to the meshes of the production cases not yet ``completed``, so
    pilots placed at the corners of the box do not set the size, and a sweep
    already generated is not budgeted again. The wall estimate ignores the
    increment count and the case's severity, and says so.
    """
    production = defn.levels.production
    per_level: dict[str, list[float]] = {}
    for s in by_role(specs, "level"):
        wall = runs.get(s.case_id, {}).get("wall_s")
        if wall is not None:
            per_level.setdefault(str((s.probe or {})["level"]), []).append(float(wall))
    prod_specs = [
        s for s in by_role(specs, "level") if (s.probe or {}).get("level") == production
    ]
    base = {"free_gb": free, "min_free_gb": defn.pilot.min_free_gb}
    per_node: list[float] = []
    per_element: list[float] = []
    walls: list[float] = []
    pilot_sizes: list[float] = []
    for s in prod_specs:
        wall = runs.get(s.case_id, {}).get("wall_s")
        if wall is None or s.case_id not in sizes:
            continue
        nodes, elements = _mesh_counts(problem, s.params)
        if nodes == 0 or elements == 0:
            return Step(
                "budget",
                "not_assessable",
                f"the mesh of {s.case_id} has no nodes or no elements",
                base,
            )
        per_node.append(float(sizes[s.case_id]) / nodes)
        per_element.append(float(wall) / elements)
        walls.append(float(wall))
        pilot_sizes.append(float(sizes[s.case_id]))
    if not per_node:
        return Step(
            "budget",
            "not_assessable",
            "no production-level pilot has both a run record and a size",
            base,
        )
    probe_of = {s.name: s.probe for s in defn.splits}
    planned = [
        s
        for s in plan_cases(defn, getattr(problem, "feasible", None))
        if not probe_of[s.split]
    ]
    done = set(completed)
    remaining = [s for s in planned if s.case_id not in done]
    counts = [_mesh_counts(problem, s.params) for s in remaining]
    bytes_per_node = float(statistics.median(per_node))
    wall_per_element = float(statistics.median(per_element))
    estimated_gb = sum(n for n, _ in counts) * bytes_per_node / 1e9
    detail = {
        **base,
        "production_cases": len(planned),
        "completed_cases": len(planned) - len(remaining),
        "remaining_cases": len(remaining),
        "bytes_per_node": bytes_per_node,
        "wall_s_per_element": wall_per_element,
        "estimated_gb": estimated_gb,
        "estimated_wall_h": sum(e for _, e in counts) * wall_per_element / 3600.0,
        # per pilot, for run's estimate line
        "wall_s_median": float(statistics.median(walls)),
        "wall_s_max": max(walls),
        "bytes_per_case": float(statistics.median(pilot_sizes)),
        "per_level": {
            lv: {
                "n": len(per_level[lv]),
                "wall_s_median": float(statistics.median(per_level[lv])),
                "wall_s_max": max(per_level[lv]),
            }
            for lv in level_order(defn, set(per_level))
        },
    }
    if not remaining:
        return Step(
            "budget",
            "pass",
            f"nothing left to generate: all {len(planned)} production cases completed",
            detail,
        )
    needed = estimated_gb + defn.pilot.min_free_gb
    if free < needed:
        return Step(
            "budget",
            "fail",
            f"{free:.1f} GB free, {estimated_gb:.1f} GB estimated for the "
            f"{len(remaining)} production cases left plus the "
            f"{defn.pilot.min_free_gb:g} GB margin",
            detail,
        )
    return Step(
        "budget",
        "pass",
        f"the {len(remaining)} production cases left are estimated at "
        f"{estimated_gb:.1f} GB and {detail['estimated_wall_h']:.1f} core-hours "
        f"(sized by their meshes; the wall ignores increments and severity); "
        f"{free:.1f} GB free",
        detail,
    )


def _mesh_counts(problem: ModuleType, params: Mapping[str, Any]) -> tuple[int, int]:
    grid = problem.mesh(dict(params))
    return len(grid.node_labels), len(grid.connectivity)


def step_verification(report: Any, defn: Definition) -> Step:
    """The instrument over the preflight cases: no fail outside the accepted gaps."""
    accepted = set(defn.pilot.accepted_gaps)
    failing: list[str] = []
    gaps: dict[str, Counter[str]] = {}
    for case in report.cases:
        for r in case.results:
            if r.quantity in accepted:
                gaps.setdefault(r.quantity, Counter())[str(r.verdict)] += 1
            elif r.verdict is Verdict.FAIL:
                failing.append(f"{case.case_id}: {r.quantity}")
    detail = {
        "failing": failing,
        "accepted_gaps": {q: dict(sorted(c.items())) for q, c in gaps.items()},
    }
    if failing:
        return Step("verification", "fail", "; ".join(failing), detail)
    listed = ", ".join(sorted(gaps)) or "none seen"
    return Step(
        "verification",
        "pass",
        f"no verification row fails outside the accepted gaps ({listed})",
        detail,
    )


# --- the stamp and the report --------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Plain JSON: tuples to lists, numpy scalars to Python, non-finite to null."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stamp_record(
    defn: Definition,
    dataset_dir: Path,
    steps: Sequence[Step],
    specs: Sequence[CaseSpec],
    *,
    created_utc: str,
    siblings_sha256: str,
    runs_generated_under: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The stamp: the definition's hashes, the verdicts, the budget, the cases,
    and the hashes the runs were generated under (different from the stamp's
    own only after ``--rejudge``)."""
    by_name = {s.name: s for s in steps}
    ordered = [by_name[n] for n in STEPS if n in by_name]
    ordered += [s for s in steps if s.name not in STEPS]
    state = package_state()
    acceptances = dict(defn.pilot.accepted_reviews)
    occurred = sorted({k for s in steps for k in s.reviews})
    record = {
        "format": STAMP_FORMAT,
        "dataset": defn.name,
        "definition_sha256": defn.sha256(),
        "problem_sha256": problem_sha256(dataset_dir),
        "siblings_sha256": siblings_sha256,
        "structbench": {"version": state["version"], "commit": state["commit"]},
        "created_utc": created_utc,
        "passed": passed(steps, acceptances),
        "accepted_reviews": accepted(steps, acceptances),
        "unaccepted_reviews": [k for k in occurred if k not in acceptances],
        "unused_acceptances": sorted(k for k in acceptances if k not in occurred),
        "steps": {
            s.name: {
                "verdict": s.verdict,
                "summary": s.summary,
                "detail": s.detail,
                "reviews": list(s.reviews),
            }
            for s in ordered
        },
        "budget": by_name["budget"].detail if "budget" in by_name else {},
        "cases": {role: [s.case_id for s in by_role(specs, role)] for role in ROLES},
        "runs_generated_under": [dict(h) for h in runs_generated_under],
    }
    return _json_safe(record)


def render_report(stamp: Mapping[str, Any]) -> str:
    """The reader's document, from the stamp alone."""
    sb = stamp.get("structbench", {})
    commit = sb.get("commit") or "no commit"
    out = [f"# Preflight of {stamp['dataset']}", ""]
    out.append(
        f"*Stamp `{stamp['format']}`, {stamp['created_utc']}; StructBench "
        f"{sb.get('version', '?')} ({commit}); `dataset.toml` "
        f"{stamp['definition_sha256'][:12]}…, `problem.py` "
        f"{stamp['problem_sha256'][:12]}….*"
    )
    out.append("")
    acceptances = stamp.get("accepted_reviews") or {}

    def rescued(s: Mapping[str, Any]) -> bool:
        keys = s.get("reviews") or []
        return (
            s["verdict"] == "review"
            and bool(keys)
            and all(k in acceptances for k in keys)
        )

    if stamp.get("passed"):
        out.append(
            "**Passed.** Every step passed, does not apply, or is a review the "
            "dataset accepted (below); `generate` opens the production splits "
            "against this stamp."
        )
    else:
        blocking = [
            name
            for name, s in stamp["steps"].items()
            if s["verdict"] not in PASSING and not rescued(s)
        ]
        out.append(
            "**Not passed.** Blocking: "
            + ", ".join(f"{n} ({stamp['steps'][n]['verdict']})" for n in blocking)
            + "."
        )
    out += ["", "## Verdicts", "", "| Step | Verdict | Summary |", "|---|---|---|"]
    for name, s in stamp["steps"].items():
        summary = str(s["summary"]).replace("|", "\\|")
        verdict = f"{s['verdict']} (accepted)" if rescued(s) else s["verdict"]
        out.append(f"| {name} | {verdict} | {summary} |")
    reported = ((stamp["steps"].get("frame") or {}).get("detail") or {}).get(
        "reported"
    ) or {}
    unaccepted = stamp.get("unaccepted_reviews") or []
    unused = stamp.get("unused_acceptances") or []
    if acceptances or reported or unaccepted or unused:
        out += ["", "## Accepted by the dataset", ""]
        for key, reason in acceptances.items():
            out.append(f"- review `{key}` accepted: {reason}")
        for key, row in reported.items():
            reason = row.get("reason", "") if isinstance(row, Mapping) else ""
            out.append(f"- `{key}` reported, not judged by the frame step: {reason}")
        for key in unaccepted:
            out.append(f"- review `{key}` not accepted: it blocks the stamp")
        for key in unused:
            out.append(f"- acceptance `{key}` declared but no such review occurred")
    out += ["", "## Cases", ""]
    for role in ROLES:
        ids = stamp.get("cases", {}).get(role, [])
        out.append(f"- {role}: " + (", ".join(ids) if ids else "none"))
    budget = stamp.get("budget") or {}
    if budget:
        out += ["", "## Budget", ""]
        for key in sorted(budget):
            if key == "per_level":
                continue
            out.append(f"- {key}: {budget[key]}")
        for lv, row in (budget.get("per_level") or {}).items():
            out.append(f"- level {lv}: {row}")
    out += ["", "## Details", ""]
    for name, s in stamp["steps"].items():
        out += [f"### {name}", "", str(s["summary"]), ""]
        if s.get("detail"):
            text = json.dumps(s["detail"], indent=2, sort_keys=True, ensure_ascii=False)
            out += ["```json", text, "```", ""]
    return "\n".join(out).rstrip("\n") + "\n"


def write_outputs(pre: Path, stamp: Mapping[str, Any]) -> None:
    """``stamp.json`` (sorted keys, LF) and ``report.md`` beside the cases."""
    pre.mkdir(parents=True, exist_ok=True)
    text = json.dumps(stamp, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (pre / "stamp.json").write_bytes(text.encode("utf-8"))
    (pre / "report.md").write_bytes(render_report(stamp).encode("utf-8"))


# --- the driver ----------------------------------------------------------------


def _line(step: Step) -> str:
    return f"  {step.name}: {step.verdict} - {step.summary}"


def _stale_cases(pre: Path, hashes: Mapping[str, str]) -> list[str]:
    """Cases in the preflight folder generated under other definition hashes.

    A changed ``dataset.toml``, ``problem.py`` or imported sibling invalidates
    every run there, whether or not the decks changed (review finding 1).
    """
    stale: list[str] = []
    if not pre.is_dir():
        return stale
    for prov_path in sorted(pre.glob("*/provenance.json")):
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        if any(prov.get(k) != v for k, v in hashes.items()):
            stale.append(prov_path.parent.name)
    return stale


def _states(pre: Path, ids: Sequence[str]) -> dict[str, str]:
    return {
        cid: case_state(pre / cid) if (pre / cid).is_dir() else "missing" for cid in ids
    }


def _run_records(pre: Path, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for cid in ids:
        path = pre / cid / "run.json"
        if path.is_file():
            out[cid] = json.loads(path.read_text(encoding="utf-8"))
    return out


#: The definition hashes a run's provenance records, and the stamp compares.
HASH_KEYS = ("definition_sha256", "problem_sha256", "siblings_sha256")


def _generated_under(pre: Path, specs: Sequence[CaseSpec]) -> list[dict[str, Any]]:
    """The distinct definition hashes the preflight's runs were generated under."""
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for s in specs:
        path = pre / s.case_id / "provenance.json"
        if path.is_file():
            prov = json.loads(path.read_text(encoding="utf-8"))
            key = tuple(prov.get(k) for k in HASH_KEYS)
            seen[key] = dict(zip(HASH_KEYS, key, strict=True))
    return [seen[k] for k in sorted(seen, key=str)]


def _rejudge_refusal(
    pre: Path, specs: Sequence[CaseSpec], decks: Mapping[str, str], defn: Definition
) -> str | None:
    """Why the runs in ``pre`` are not the same evidence under the current
    definition, or None: the case set must be the planned one, every case must
    have run, its deck must be the current deck byte for byte, and its units
    the current units."""
    present = (
        {p.name for p in pre.iterdir() if (p / "provenance.json").is_file()}
        if pre.is_dir()
        else set()
    )
    planned = {s.case_id for s in specs}
    new = sorted(planned - present)
    if new:
        return f"new case {new[0]} (re-judging runs nothing; run without --rejudge)"
    gone = sorted(present - planned)
    if gone:
        return f"case {gone[0]} is no longer planned; move it aside"
    for s in specs:
        folder = pre / s.case_id
        prov = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        if prov.get("units") != defn.units:
            return (
                f"{s.case_id} ran in units {prov.get('units')!r}, the definition "
                f"now says {defn.units!r}"
            )
        deck = folder / f"{s.case_id}.inp"
        if not deck.is_file() or deck.read_bytes() != decks[s.case_id].encode("utf-8"):
            return f"the deck of {s.case_id} differs from the current definition's"
        if not (folder / "run.json").is_file():
            return f"{s.case_id} has not run (run the preflight without --rejudge)"
    return None


def _completed(sweep: Path) -> set[str]:
    """Case ids whose run completed in the sweep (the production already made)."""
    done = set()
    for run in sweep.glob("*/run.json"):
        try:
            if json.loads(run.read_text(encoding="utf-8")).get("status") == "completed":
                done.add(run.parent.name)
        except (OSError, ValueError):
            continue  # a record being written is not yet a completed run
    return done


def _sizes(pre: Path, ids: Sequence[str]) -> dict[str, int]:
    """Bytes per case: the case folder (ODB, export, side files) and its
    canonical file."""
    out = {}
    for cid in ids:
        folder = pre / cid
        if not folder.is_dir():
            continue
        total = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        h5 = pre / "canonical" / f"{cid}.h5"
        if h5.is_file():
            total += h5.stat().st_size
        out[cid] = total
    return out


def preflight(
    dataset_dir: Path,
    work_root: Path,
    *,
    abaqus: str = "abaqus",
    workers: int = 2,
    timeout: float | None = None,
    solver_args: Sequence[str] | None = None,
    exporter_args: Sequence[str] | None = None,
    rejudge: bool = False,
    echo: Callable[[str], None] = print,
) -> int:
    """Run the preflight; see the module docstring for the steps and exit codes.

    ``solver_args`` and ``exporter_args`` are inserted after the executable on
    the solver's and the exporter's command lines (tests: a fake solver
    script in place of Abaqus and of ``python <exporter>``). ``rejudge``
    re-evaluates the steps on the runs already in the preflight folder, with
    no solver, when the current definition reproduces every deck byte for byte
    and the units are unchanged (plan 3a); the stamp records the definition
    hashes the runs were generated under.
    """
    dataset_dir = Path(dataset_dir).resolve()
    work_root = Path(work_root)
    findings = check_definition(dataset_dir)
    if findings:
        for finding in findings:
            print(f"{dataset_dir.name}: {finding}", file=sys.stderr)
        return 2
    defn = load_definition(dataset_dir)
    problem = load_problem(dataset_dir)
    specs = preflight_cases(defn)
    try:
        decks = {s.case_id: deck_for(s, problem) for s in specs}
    except ValueError as exc:
        print(
            f"{dataset_dir.name}: the conformance run cannot be built: {exc}",
            file=sys.stderr,
        )
        return 2
    pre = work_root / defn.name / PREFLIGHT_DIR
    hashes = {
        "definition_sha256": defn.sha256(),
        "problem_sha256": problem_sha256(dataset_dir),
        "siblings_sha256": siblings_sha256(dataset_dir, problem),
    }
    stale = _stale_cases(pre, hashes)
    if stale and not rejudge:
        print(
            f"{pre} holds runs of another definition ({len(stale)} case(s), e.g. "
            f"{stale[0]}); move it aside (nothing is deleted), or pass --rejudge "
            "if only the judging changed and the decks are the same",
            file=sys.stderr,
        )
        return 2
    if rejudge:
        why = _rejudge_refusal(pre, specs, decks, defn)
        if why:
            print(f"{pre}: cannot re-judge: {why}", file=sys.stderr)
            return 2
    created = datetime.now(UTC).isoformat(timespec="seconds")
    steps: list[Step] = []

    def add(step: Step) -> None:
        steps.append(step)
        echo(_line(step))

    def finish() -> int:
        done = {s.name for s in steps}
        full = list(steps) + [
            not_run(n, "an earlier step did not pass") for n in STEPS if n not in done
        ]
        stamp = stamp_record(
            defn,
            dataset_dir,
            full,
            specs,
            created_utc=created,
            siblings_sha256=hashes["siblings_sha256"],
            runs_generated_under=_generated_under(pre, specs),
        )
        write_outputs(pre, stamp)
        echo(("passed" if stamp["passed"] else "not passed") + f" -> {pre}")
        return 0 if stamp["passed"] else 1

    # Before any deck is written: a failed regression leaves nothing behind
    # that a corrected problem.py would then be refused against.
    add(step_deck_regression(dataset_dir, specs, problem, defn))
    if steps[-1].verdict == "fail":
        return finish()
    try:
        counts, problems = materialise(
            pre, specs, defn=defn, problem=problem, dataset_dir=dataset_dir, decks=decks
        )
    except RuntimeError as exc:
        print(
            f"{dataset_dir.name}: {exc}; commit the definition first", file=sys.stderr
        )
        return 2
    if problems:
        print(
            f"{pre} holds decks of another definition ({len(problems)} case(s) "
            f"differ, e.g. {problems[0]}); move it aside (nothing is deleted)",
            file=sys.stderr,
        )
        return 2
    echo(
        f"preflight of {defn.name}: {len(specs)} cases in {pre} ("
        + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        + ")"
    )
    exe = None if rejudge else shutil.which(abaqus)
    if exe is None and not rejudge:
        print(f"abaqus executable {abaqus!r} not found", file=sys.stderr)
        return 2
    solver = [exe or abaqus, *(solver_args or [])]
    production = defn.levels.production
    ids = [s.case_id for s in specs]
    first = batch_a(specs, production)
    rest = [cid for cid in ids if cid not in first]

    def launch(cases: list[str]) -> int | None:
        """Run and export ``cases``; an exit code when the preflight must stop."""
        if not cases or rejudge:  # re-judging runs nothing
            return None
        try:
            results = run_sweep(
                pre,
                solver,
                cases=cases,
                workers=workers,
                timeout=timeout,
                min_free_gb=defn.pilot.min_free_gb,
                echo=echo,
            )
        except (RuntimeError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 2
        if any(r.status == NOT_LAUNCHED for r in results):
            print(
                f"free space below {defn.pilot.min_free_gb:g} GB: the preflight "
                "stopped; free space and run it again (it resumes)",
                file=sys.stderr,
            )
            return 3
        rc = export_cases(pre, abaqus, cases=cases, exporter_args=exporter_args)
        if rc != 0:
            echo(f"  export returned {rc}; cases without an export are reported")
        return None

    stop = launch(first)
    if stop is not None:
        return stop
    add(step_feasibility(specs, _states(pre, ids), defn, problem))
    (conformance,) = by_role(specs, "conformance")
    npz = pre / conformance.case_id / f"{conformance.case_id}.npz"
    add(step_conformance(npz if npz.is_file() else None, defn.units))
    if any(s.verdict not in PASSING for s in steps):
        return finish()  # nothing more is launched on a step that did not pass
    stop = launch(rest)
    if stop is not None:
        return stop

    conversion = convert_sweep(pre, dataset_id=defn.name)
    for cid, reason in sorted(conversion.failed.items()):
        echo(f"  convert {cid}: {reason}")
    _, report = judge_sweep(pre, dataset_dir)
    level_ids = [s.case_id for s in by_role(specs, "level")]
    conv = convergence_stage.converge(
        defn, problem, [pre], anchor=defn.pilot.split, cases=level_ids
    )
    out = pre / "converge"
    out.mkdir(exist_ok=True)
    (out / "convergence.json").write_bytes(convergence_stage.to_json(conv))
    (out / "convergence.md").write_bytes(
        convergence_stage.render_markdown(conv).encode("utf-8")
    )
    add(step_space(conv, defn))

    canonical = pre / "canonical"

    def load(cid: str) -> Case | None:
        path = canonical / f"{cid}.h5"
        return read_case(path) if path.is_file() else None

    def qoi_of(cid: str, case: Case) -> dict[str, float] | None:
        try:
            return {k: float(v) for k, v in problem.qoi(case).items()}
        except Exception as exc:  # a QoI that raises is a finding, not a crash
            echo(f"  {cid}: qoi() raised {type(exc).__name__}: {exc}")
            return None

    prod_specs = [
        s for s in by_role(specs, "level") if (s.probe or {}).get("level") == production
    ]
    base_id = {_pilot_key(s): s.case_id for s in prod_specs}
    prod_cases: dict[str, Case] = {}
    for s in prod_specs:
        loaded = load(s.case_id)
        if loaded is not None:
            prod_cases[s.case_id] = loaded
    qois: dict[str, dict[str, float]] = {}
    for cid, case in prod_cases.items():
        values = qoi_of(cid, case)
        if values is not None:
            qois[cid] = values
    inc_fields: dict[str, dict[str, float]] = {}
    for s in by_role(specs, "increment"):
        loaded = load(s.case_id)
        if loaded is None:
            continue
        values = qoi_of(s.case_id, loaded)
        if values is not None:
            qois[s.case_id] = values
        base = prod_cases.get(base_id.get(_pilot_key(s), ""))
        if base is not None:
            try:
                inc_fields[s.case_id] = common_instant_errors(base, loaded)
            except ValueError as exc:
                echo(f"  {s.case_id}: {exc}")
    frame_case = production_case = None
    for s in by_role(specs, "frame"):
        frame_case = load(s.case_id)
        production_case = prod_cases.get(base_id.get(_pilot_key(s), ""))
    add(step_increment(qois, inc_fields, specs, defn, report))
    add(step_frame(frame_case, production_case, defn))
    add(step_duration(prod_cases, specs, defn, problem))
    add(step_energy(report, ids))
    add(
        step_budget(
            defn,
            problem,
            specs,
            _run_records(pre, ids),
            _sizes(pre, ids),
            free_gb(work_root),
            completed=_completed(work_root / defn.name),
        )
    )
    add(step_verification(report, defn))
    return finish()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="structbench-datagen preflight",
        description="run the pilot split through the preflight and write its stamp",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--abaqus", default="abaqus")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--solver-args", nargs="*", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--exporter-args", nargs="*", default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="re-evaluate the steps on the runs already made, with no solver, "
        "when the current definition reproduces every deck and the units",
    )
    args = parser.parse_args(argv)
    return preflight(
        args.dataset,
        args.work_root,
        abaqus=args.abaqus,
        workers=args.workers,
        timeout=args.timeout,
        solver_args=args.solver_args,
        exporter_args=args.exporter_args,
        rejudge=args.rejudge,
    )


if __name__ == "__main__":
    sys.exit(main())
