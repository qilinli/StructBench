"""The documents that describe the pipeline exist where the design says."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_conformance_documents_live_under_docs_and_nothing_names_the_old_file():
    for solver in ("abaqus", "lsdyna"):
        assert (ROOT / "docs" / "datagen" / f"{solver}-conformance.md").is_file()
    assert not (ROOT / "tools" / "ingest" / "abaqus").exists()
    assert not (
        ROOT / "tools" / "ingest" / "lsdyna" / "STANDARD_INPUT_BLOCK.md"
    ).is_file()
    # Both blocks moved (Abaqus 2026-09-27, LS-DYNA 2026-09-29), so neither code
    # nor a living document may name the old file. ADRs keep a "(moved ...; was
    # ...)" note as records, and historical plans under docs/plans keep their
    # text; neither is scanned.
    files = [
        *ROOT.glob("src/**/*.py"),
        *ROOT.glob("tests/**/*.py"),
        *ROOT.glob("docs/*.md"),
        *ROOT.glob("docs/datagen/*.md"),
        ROOT / "README.md",
        ROOT / "CLAUDE.md",
        ROOT / "tools" / "ingest" / "README.md",
    ]
    stale = [
        p
        for p in files
        if p.name != "test_docs.py"
        and "STANDARD_INPUT_BLOCK" in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert stale == []


def test_the_conformance_documents_relative_links_resolve():
    link = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
    broken = [
        (doc.name, target)
        for doc in sorted((ROOT / "docs" / "datagen").glob("*.md"))
        for target in link.findall(doc.read_text(encoding="utf-8"))
        if "://" not in target and not (doc.parent / target).exists()
    ]
    assert broken == []


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
        ROOT / "tools" / "ingest" / "README.md",
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


def test_architecture_names_the_convergence_engine_and_the_adapter_names_its_exporter():
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "`convergence.py`" in text and "`dataset.py`" in text
    adapter = (ROOT / "src" / "structbench" / "core" / "io" / "abaqus.py").read_text(
        encoding="utf-8"
    )
    assert "data_generation/abaqus/odb_export.py" not in adapter
    assert "structbench/datagen/abaqus/odb_export.py" in adapter
