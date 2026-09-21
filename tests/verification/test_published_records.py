"""The published verification records regenerate from themselves (ADR-0066 cl. 8).

``docs/datachecks/<benchmark>.json`` is the committed record: measurements,
never verdicts. Its ``.md`` sibling is generated, so it must equal
``render_markdown(judge(from_json(json)))`` — no data is needed to check that,
and a change to a criterion or to the report shows up here as drift.

To regenerate a report after such a change (no data needed)::

    python -m structbench.cli.datacheck judge \\
        --measurements docs/datachecks/NAME.json --report docs/datachecks/NAME.md

To re-measure (needs the canonical archive, and the run-evidence record the
dataset's glue writes), see ``structbench.cli.datacheck measure``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from structbench.benchmarks import available_benchmarks, get_benchmark
from structbench.verification.criteria import judge
from structbench.verification.quantities import CATALOGUE
from structbench.verification.report import from_json, render_markdown, to_json

_DIR = Path(__file__).resolve().parents[2] / "docs" / "datachecks"
_RECORDS = sorted(_DIR.glob("*.json"))


def test_there_is_at_least_one_published_record() -> None:
    assert _RECORDS, f"no records under {_DIR}"


@pytest.mark.parametrize("path", _RECORDS, ids=lambda p: p.stem)
def test_the_report_is_what_the_record_and_the_criteria_generate(path: Path) -> None:
    record = from_json(path.read_text(encoding="utf-8"))
    published = path.with_suffix(".md").read_text(encoding="utf-8")
    assert published == render_markdown(judge(record))


@pytest.mark.parametrize("path", _RECORDS, ids=lambda p: p.stem)
def test_the_record_is_canonical_and_complete(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    record = from_json(text)
    assert to_json(record) == text  # written by to_json, not by hand
    assert path.stem in available_benchmarks()
    assert record.benchmark == get_benchmark(path.stem).card.name
    names = [q.name for q in CATALOGUE]
    for case in record.cases:
        assert [m.quantity for m in case.measurements] == names, case.case_id
        assert case.file_sha256 is not None and len(case.file_sha256) == 64


@pytest.mark.parametrize("path", _RECORDS, ids=lambda p: p.stem)
def test_nothing_private_is_published(path: Path) -> None:
    for artefact in (path, path.with_suffix(".md")):
        text = artefact.read_text(encoding="utf-8")
        assert not re.search(r"[A-Za-z]:\\|/home/|/Users/|OneDrive", text), artefact
        assert not re.search(r"(?i)licen[cs]e|hostname", text), artefact
