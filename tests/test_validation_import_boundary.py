"""``structbench.validation`` imports only ``core`` (ADR-0072).

The resolver is the verification boundary test's, so relative imports count.
"""

from __future__ import annotations

import pathlib

from test_verification_import_boundary import _first_party_targets

_SRC = pathlib.Path("src")
_ALLOWED = {"core", "validation"}


def test_validation_imports_only_core() -> None:
    modules = sorted((_SRC / "structbench" / "validation").rglob("*.py"))
    assert modules, "validation package not found - run pytest from the repo root"
    offenders = {
        str(path): sorted(_first_party_targets(path, _SRC) - _ALLOWED)
        for path in modules
        if _first_party_targets(path, _SRC) - _ALLOWED
    }
    assert not offenders, f"forbidden first-party imports: {offenders}"
