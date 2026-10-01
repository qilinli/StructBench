"""The decision log keeps the shape docs/decisions/README.md asks of it.

The index and the files agree; every record a header names exists; and a
record numbered 0074 or later opens with its Your-call block and fits the
page (the rule of 2026-10-01, which earlier records predate).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "docs" / "decisions"
FIRST_UNDER_THE_RULE = 74
WORD_CAP = 900

_FILES = sorted(p for p in LOG.glob("0*.md"))
_NUM = re.compile(r"(?<![\d.])0\d{3}(?![\d.])")


def _number(path: Path) -> str:
    return path.name[:4]


def _body(text: str) -> str:
    """The record before its first appended note (a `---` rule on its own line)."""
    return text.split("\n---\n", 1)[0]


def test_every_record_has_an_index_row_and_every_row_a_record():
    index = (LOG / "README.md").read_text(encoding="utf-8")
    rows = set(re.findall(r"^\| (0\d{3}) \|", index, re.M))
    files = {_number(p) for p in _FILES}
    assert rows == files, (sorted(files - rows), sorted(rows - files))


@pytest.mark.parametrize("path", _FILES, ids=_number)
def test_every_record_a_header_names_exists(path: Path):
    header = path.read_text(encoding="utf-8").split("\n## ", 1)[0]
    cited = set()
    for line in header.splitlines():
        if line.startswith(("**Amends**", "**Status**", "**Touches**")):
            cited.update(_NUM.findall(line))
    existing = {_number(p) for p in _FILES}
    assert cited <= existing, sorted(cited - existing)


@pytest.mark.parametrize(
    "path", [p for p in _FILES if int(_number(p)) >= FIRST_UNDER_THE_RULE], ids=_number
)
def test_a_new_record_opens_with_its_your_call_block_and_fits_the_page(path: Path):
    text = path.read_text(encoding="utf-8")
    header = text.split("\n## ", 1)[0]
    assert "**Your call**:" in header, f"{path.name}: no Your-call block in the header"
    words = len(_body(text).split())
    assert words <= WORD_CAP, f"{path.name}: {words} words before the first note"
