"""The verification report a benchmark user reads (ADR-0066 clause 8).

Written for someone deciding whether to trust a dataset, not for the
instrument's author: a bottom line first, findings in plain words, results
grouped by what they are about, the numbers nobody has judged set apart from
the checks nobody could make, and one row per case for whatever varies.

It is generated from the committed measurements and the criteria alone — no
data is read — so it can be regenerated whenever either changes.
"""

from __future__ import annotations

from collections import defaultdict

from ..core import PLATFORM_REASONS
from .criteria import CheckResult, Criterion, DatasetReport
from .quantities import Quantity, get_quantity
from .results import Category, Verdict

__all__ = ["render_markdown"]

_PLATFORM = {str(r) for r in PLATFORM_REASONS}

_CATEGORIES = {
    Category.DATA_INTEGRITY: "Data integrity",
    Category.NUMERICAL_HEALTH: "Numerical health of the runs",
    Category.CONSERVATION: "Energy and mass conservation",
    Category.CONSTITUTIVE: "Material behaviour",
    Category.UNITS: "Units and magnitudes",
}
_EVIDENCE = {
    "E1": "the solver input",
    "E2": "the solver's version and precision",
    "E3": "the solver's termination record and messages",
    "E4": "the time-integration record (time step, added mass)",
    "E5": "the global energy ledger",
    "E6": "energy and mass per part and per contact interface",
    "E7": "applied loads and reaction forces over time",
    "E8": "field output at the material points",
    "E9": "a sampling clock shared by the energy ledger and the fields",
    "E10a": "a declared unit system",
    "E10b": "declared unit anchors",
}
_REASONS = {
    "unsupported": "this instrument cannot measure it yet",
    "no_declaration_home": "the benchmark has nowhere to declare it yet",
    "not_ingested": "the adapter did not keep the data it needs",
    "stale_definition": "measured under an older definition; needs re-measuring",
    "unparsable": "the instrument could not read what the run supplied",
    "source_unreadable": "the case file could not be read",
    "not_requested": "the run never asked the solver for it",
    "not_computed": "the solver was told not to compute it",
    "not_available_from_solver": "this solver cannot provide it",
}
# outcome of one quantity over the whole dataset, most important first
_FINDING, _PASS, _MEASURED, _UNCHECKED, _NA = (
    "finding",
    "pass",
    "measured",
    "unchecked",
    "not_applicable",
)
Rows = list[tuple[str, CheckResult]]


def _num(value: float, digits: int = 4) -> str:
    text = f"{value:.{digits}g}"
    return f"{value:.0f}" if "e+" in text and abs(value) < 1e9 else text


def _scaled(value: float, unit: str) -> str:
    steps = {
        "Pa": ((1e9, "GPa"), (1e6, "MPa"), (1e3, "kPa")),
        "m": ((1.0, "m"), (1e-3, "mm"), (1e-6, "µm")),
    }.get(unit, ())
    for factor, name in steps:
        if abs(value) >= factor:
            return f"{_num(value / factor)} {name}"
    return _num(value, 6) if unit == "1" else f"{_num(value)} {unit}"


def _fmt(q: Quantity, value: float) -> str:
    if q.display == "percent":
        return f"{100.0 * value:.3g} %"
    if q.display == "count":
        return f"{value:.0f}"
    if q.display == "flag":
        return "yes" if value >= 0.5 else "no"
    return _scaled(value, q.unit)


def _span(q: Quantity, values: list[float]) -> str:
    lo, hi = min(values), max(values)
    return _fmt(q, lo) if lo == hi else f"{_fmt(q, lo)} to {_fmt(q, hi)}"


def _outcome(rows: Rows) -> str:
    verdicts = {r.verdict for _, r in rows}
    if verdicts & {Verdict.FAIL, Verdict.REVIEW}:
        return _FINDING
    if Verdict.PASS in verdicts:
        return _PASS
    if any(r.value is not None for _, r in rows):
        return _MEASURED
    return _NA if verdicts == {Verdict.NOT_APPLICABLE} else _UNCHECKED


def _worst(q: Quantity, rows: Rows) -> str:
    valued = [(cid, r.value) for cid, r in rows if r.value is not None]
    if len(valued) < 2 or len({v for _, v in valued}) == 1:
        return "all cases" if len(valued) > 1 else ""
    pick = min if q.lower_is_worse else max
    return f"`{pick(valued, key=lambda cv: cv[1])[0]}`"


def _bound(q: Quantity, c: Criterion) -> str:
    if q.display == "flag":
        text = "must be yes"
    elif c.lo is not None and c.hi is not None:
        text = f"between {_fmt(q, c.lo)} and {_fmt(q, c.hi)}"
    elif c.hi is not None:
        text = "must be zero" if c.hi == 0 else f"at most {_fmt(q, c.hi)}"
    else:
        assert c.lo is not None
        text = f"at least {_fmt(q, c.lo)}"
    tokens = sorted(str(x) for x in c.scope.run_traits) + sorted(
        c.scope.declared_intent
    )
    if not tokens:
        return text
    return f"{text}, for {', '.join(x.replace('_', ' ') for x in tokens)} runs"


def _pl(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _cases(ids: list[str], total: int, limit: int = 5) -> str:
    if len(ids) == total and total > 1:
        return f"all {total} cases"
    shown = ", ".join(f"`{i}`" for i in ids[:limit])
    return shown if len(ids) <= limit else f"{shown} and {len(ids) - limit} more"


def _why_unchecked(rows: Rows) -> str:
    reasons = []
    for _, r in rows:
        if r.verdict is not Verdict.NOT_ASSESSABLE:
            continue
        if r.reason == "source_missing" and r.missing:
            need = " and ".join(_EVIDENCE.get(item, item) for item in r.missing)
            reasons.append(f"the runs did not supply {need}")
        else:
            reasons.append(_REASONS.get(r.reason, r.reason.replace("_", " ")))
    return "; ".join(dict.fromkeys(reasons))


def render_markdown(report: DatasetReport) -> str:
    """The reader-facing report for one dataset (or one run)."""
    data = report.measurements
    n_cases = len(report.cases)
    by_quantity: dict[str, Rows] = defaultdict(list)
    for case in report.cases:
        for result in case.results:
            by_quantity[result.quantity].append((case.case_id, result))
    quantities = {name: get_quantity(name) for name in by_quantity}
    outcome = {name: _outcome(rows) for name, rows in by_quantity.items()}
    applied = {(c.quantity, c.label()): c for c in report.criteria}
    by_label = {f"{c.label()} [{c.source}]": c for c in report.criteria}

    def names(kind: str, category: Category | None = None) -> list[str]:
        return sorted(
            (
                n
                for n, o in outcome.items()
                if o == kind and category in (None, quantities[n].category)
            ),
            key=lambda n: quantities[n].title,
        )

    def criterion_of(name: str) -> Criterion | None:
        labels = {r.criterion for _, r in by_quantity[name] if r.criterion}
        return applied.get((name, labels.pop())) if len(labels) == 1 else None

    def levels_text(name: str, labels: set[str]) -> str:
        q = quantities[name]
        found = [by_label[label] for label in sorted(labels) if label in by_label]
        return "; ".join(f"{_bound(q, c)} [{c.source}]" for c in found)

    def context(name: str) -> str:
        rows = by_quantity[name]
        shown = {r.unratified_level for _, r in rows if r.unratified_level}
        other = {lv for _, r in rows for lv in r.out_of_scope_levels}
        if shown:
            return "a published level exists, not confirmed by this platform: " + (
                levels_text(name, shown)
            )
        if other:
            return "published levels exist only for other kinds of run: " + (
                levels_text(name, other)
            )
        return "no criterion"

    title = data.benchmark or "unregistered runs"
    checks = len(by_quantity)
    out = [
        f"# {title} — reference-data verification",
        "",
        f"Dataset revision {data.dataset_revision or 'not recorded'} · {n_cases}"
        f" case{'s' if n_cases != 1 else ''} · {checks} checks per case ·"
        f" structbench {data.structbench_version}",
        "",
        "This report says what was checked about the simulation runs behind this"
        " dataset, what was found, and what could not be checked. It is generated"
        " from a committed record of measurements; no verdict here rests on a"
        " number fitted to this dataset. A pass is a necessary sign of a healthy"
        " run, not evidence that the simulation matches reality.",
        "",
        "## Summary",
        "",
    ]

    findings, passed = names(_FINDING), names(_PASS)
    measured, unchecked, skipped = names(_MEASURED), names(_UNCHECKED), names(_NA)
    out.append(
        f"- **{_pl(len(passed), 'check passes', 'checks pass')}** wherever"
        f" {'it applies' if len(passed) == 1 else 'they apply'}."
    )
    if findings:
        named = "; ".join(quantities[n].title.lower() for n in findings)
        out.append(f"- **{_pl(len(findings), 'finding', 'findings')}**: {named}.")
    else:
        out.append("- **No findings.**")
    out += [
        f"- **{_pl(len(measured), 'quantity is', 'quantities are')} measured but not"
        " judged**: the platform has no confirmed criterion. The values are below"
        " for you to weigh.",
        f"- **{_pl(len(unchecked), 'check', 'checks')} could not be made**, because"
        " the runs did not keep the evidence or the instrument cannot do it yet.",
        f"- {_pl(len(skipped), 'check does', 'checks do')} not apply to these runs.",
        "",
        "| | Pass | Finding | Measured, not judged | Not checked | Not applicable |",
        "|---|---|---|---|---|---|",
    ]
    for category, heading in _CATEGORIES.items():
        counts = [
            len(names(kind, category))
            for kind in (_PASS, _FINDING, _MEASURED, _UNCHECKED, _NA)
        ]
        out.append(f"| {heading} | " + " | ".join(str(c or "") for c in counts) + " |")

    out += ["", "## Findings", ""]
    for name in findings:
        q, rows = quantities[name], by_quantity[name]
        bad = [
            (cid, r) for cid, r in rows if r.verdict in (Verdict.FAIL, Verdict.REVIEW)
        ]
        values = [r.value for _, r in bad if r.value is not None]
        criterion = criterion_of(name)
        word = "review" if bad[0][1].verdict is Verdict.REVIEW else "fail"
        out += [
            f"### {q.title} — {word}",
            "",
            f"- Found: {_span(q, values)} in {_cases([c for c, _ in bad], n_cases)}.",
            f"- Required: {_bound(q, criterion) if criterion else 'see criteria'}.",
            f"- What it means: {q.meaning}.",
            "",
        ]
    if not findings:
        out += ["None.", ""]

    out += ["## Results by category", ""]
    for category, heading in _CATEGORIES.items():
        listed = [
            n
            for kind in (_FINDING, _PASS, _MEASURED, _UNCHECKED)
            for n in names(kind, category)
        ]
        out += [f"### {heading}", ""]
        if listed:
            out += [
                "| Check | Result | Worst case | Verdict | Basis |",
                "|---|---|---|---|---|",
            ]
        for name in listed:
            q, rows = quantities[name], by_quantity[name]
            values = [r.value for _, r in rows if r.value is not None]
            kind = outcome[name]
            criterion = criterion_of(name)
            if kind in (_FINDING, _PASS):
                n_bad = sum(
                    r.verdict in (Verdict.FAIL, Verdict.REVIEW) for _, r in rows
                )
                good = sum(r.verdict is Verdict.PASS for _, r in rows)
                verdict = f"**fail** ({n_bad} of {n_cases})" if n_bad else "pass"
                if not n_bad and good < n_cases:
                    verdict = f"pass ({good} of {n_cases})"
                basis = _bound(q, criterion) if criterion else ""
                if criterion is not None and criterion.provisional:
                    basis += " (provisional tolerance)"
            elif kind == _MEASURED:
                verdict = "not judged"
                basis = (
                    "no criterion"
                    if context(name) == "no criterion"
                    else ("published level shown below, for context")
                )
            else:
                verdict, basis = "not checked", _why_unchecked(rows)
            shown = _span(q, values) if values else "—"
            out.append(
                f"| {q.title} | {shown} | {_worst(q, rows)} | {verdict} | {basis} |"
            )
        idle = names(_NA, category)
        if idle:
            out += [
                "",
                "Does not apply to these runs: "
                + "; ".join(quantities[n].title.lower() for n in idle)
                + ".",
            ]
        out.append("")

    out += [
        "## Measured, not judged",
        "",
        "Numbers the platform reports without a verdict. Where a published limit"
        " exists it is shown for context only: it has not been confirmed against"
        " its source by the maintainer, so it is not this platform's standard.",
        "",
    ]
    for name in measured:
        q, rows = quantities[name], by_quantity[name]
        values = [r.value for _, r in rows if r.value is not None]
        worst = _worst(q, rows)
        where = f" (worst: {worst})" if worst.startswith("`") else ""
        out.append(
            f"- **{q.title}**: {_span(q, values)}{where}. A problem here would mean"
            f" that {q.meaning}. *{context(name)[:1].upper()}{context(name)[1:]}.*"
        )
    if not measured:
        out.append("None.")

    out += ["", "## What could not be checked", ""]
    gaps: dict[str, list[str]] = defaultdict(list)
    for name in unchecked:
        gaps[_why_unchecked(by_quantity[name])].append(quantities[name].title.lower())
    for why, titles in sorted(gaps.items()):
        out.append(f"- Because {why}: {'; '.join(titles)}.")
    if not unchecked:
        out.append("Nothing: every applicable check was made.")

    varying = [
        n
        for n in measured
        if len({r.value for _, r in by_quantity[n] if r.value is not None}) > 1
    ]
    if varying:
        out += [
            "",
            "## Per-case values",
            "",
            "The unjudged quantities that differ between cases, one row per case.",
            "",
            "| Case | " + " | ".join(quantities[n].title for n in varying) + " |",
            "|---|" + "---|" * len(varying),
        ]
        for case in report.cases:
            mine = {r.quantity: r.value for r in case.results}
            cells = [
                _fmt(quantities[n], v) if (v := mine.get(n)) is not None else "—"
                for n in varying
            ]
            out.append(f"| `{case.case_id}` | " + " | ".join(cells) + " |")

    out += [
        "",
        "## How to read this report",
        "",
        "- **pass / fail** — the value was compared with a bound that is"
        " definitional (zero, equality, the input's own value) or computed from"
        " storage precision. None was fitted to this dataset.",
        "- **not judged** — the value is reported, and no bound is applied. Sourced"
        " limits from the literature are shown for context until the maintainer"
        " has confirmed them against their sources; none has been.",
        "- **not checked** — the check needs evidence the runs did not keep, or the"
        " instrument cannot make it yet. The reason is given.",
        "- **not applicable** — the quantity does not exist for this kind of run"
        " (hourglass energy in a particle model, for instance).",
        "",
        "Bounds applied:",
        "",
    ]
    for name in sorted(passed + findings, key=lambda n: quantities[n].title):
        criterion = criterion_of(name)
        if criterion is not None:
            q = quantities[name]
            tag = " (provisional)" if criterion.provisional else ""
            out.append(
                f"- *{q.title}* — {_bound(q, criterion)}{tag}. {criterion.rationale}"
            )
    levels = sorted(
        (c for c in report.criteria if not c.ratified and c.quantity in measured),
        key=lambda c: (quantities[c.quantity].title, c.label()),
    )
    if levels:
        out += [
            "",
            "Published levels shown for context (not applied). The bracketed ids are"
            " claims in the source dossier,"
            " `docs/plans/2026-09-21-reference-data-verification-sources.md`:",
            "",
        ]
        for c in levels:
            q = quantities[c.quantity]
            out.append(f"- *{q.title}* — {_bound(q, c)} [{c.source}]. {c.rationale}")
    return "\n".join(out) + "\n"
