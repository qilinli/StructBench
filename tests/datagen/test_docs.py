"""The documents that describe the pipeline exist where the design says."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_conformance_document_lives_under_docs_and_nothing_points_at_the_old_path():
    assert (ROOT / "docs" / "datagen" / "abaqus-conformance.md").is_file()
    assert not (ROOT / "data_generation" / "abaqus").exists()
    # The LS-DYNA block keeps its name under data_generation/lsdyna/; only the
    # Abaqus one moved, so code and top-level docs may not name the old Abaqus
    # path, and code may not name the bare file either.
    code = [p for p in [*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/**/*.py")]]
    prose = [*ROOT.glob("docs/*.md"), ROOT / "data_generation" / "README.md"]
    stale_code = [
        p
        for p in code
        if p.name != "test_docs.py"
        and "STANDARD_INPUT_BLOCK" in p.read_text(encoding="utf-8", errors="replace")
    ]
    old_abaqus_path = "abaqus/STANDARD_INPUT_BLOCK"
    stale_prose = [
        p
        for p in prose
        if old_abaqus_path in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert stale_code == [] and stale_prose == []


def test_architecture_names_datagen_in_the_layering():
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "### `datagen/`" in text
    assert "datagen" in text.split("`cli/` depends on most other modules")[0]


def test_no_living_document_or_code_still_says_datagen_validate():
    # ADR-0072: the stage is verify. Historical plans under docs/plans keep
    # their text; living docs, the snapshot, the example and the code may not.
    import re

    files = [
        *ROOT.glob("docs/*.md"),
        *ROOT.glob("docs/datagen/*.md"),
        ROOT / "CLAUDE.md",
        ROOT / "data_generation" / "README.md",
        *ROOT.glob("src/structbench/datagen/**/*.py"),
        *ROOT.glob("src/structbench/datagen/**/*.md"),
    ]
    stale = re.compile(
        r"structbench-datagen (\|\s*)?validate\b|datagen[./]validate\b"
        r"|\bvalidate \| archive"
    )
    hits = [
        str(p.relative_to(ROOT))
        for p in files
        if stale.search(p.read_text(encoding="utf-8", errors="replace"))
    ]
    assert hits == [], hits


def test_the_guide_stub_exists():
    text = (ROOT / "docs" / "DATA_GENERATION.md").read_text(encoding="utf-8")
    assert "structbench-datagen" in text and "dataset.toml" in text
