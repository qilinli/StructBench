"""``structbench.datagen`` imports ``core``, ``datasets``, ``verification`` and
``validation`` only (ADR-0071, plan 2a) -- never ``cli``, ``benchmarks``, ``eval``
or ``models``. The resolver is the verification boundary test's, so relative
imports count.
"""

from __future__ import annotations

import pathlib

from test_verification_import_boundary import _first_party_targets

_SRC = pathlib.Path("src")
_ALLOWED = {"core", "datasets", "verification", "validation", "datagen"}


def test_datagen_imports_only_its_lower_layers() -> None:
    modules = sorted((_SRC / "structbench" / "datagen").rglob("*.py"))
    assert modules, "datagen package not found - run pytest from the repo root"
    offenders = {
        str(path): sorted(_first_party_targets(path, _SRC) - _ALLOWED)
        for path in modules
        if _first_party_targets(path, _SRC) - _ALLOWED
    }
    assert not offenders, f"forbidden first-party imports: {offenders}"
