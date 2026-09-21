"""``structbench.verification`` imports only ``core`` and ``datasets`` (ADR-0066).

Unlike ``test_torch_free_core.py`` this resolves *relative* imports, because a
``from ..benchmarks import x`` is exactly the edge the rule forbids.
"""

from __future__ import annotations

import ast
import pathlib

_SRC = pathlib.Path("src")
_ALLOWED = {"core", "datasets", "verification"}


def _first_party_targets(path: pathlib.Path, src: pathlib.Path) -> set[str]:
    """Top-level ``structbench`` subpackages one module imports.

    ``path`` is a module under ``src/structbench``; relative imports are
    resolved against its own package.
    """
    package = list(path.relative_to(src).parts[:-1])  # e.g. [structbench, verification]
    targets: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                if node.module:
                    names = [".".join([*base, node.module])]
                else:  # "from .. import core"
                    names = [".".join([*base, alias.name]) for alias in node.names]
            else:
                names = [node.module or ""]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "structbench" and len(parts) > 1:
                targets.add(parts[1])
    return targets


def test_the_resolver_sees_relative_and_absolute_imports(
    tmp_path: pathlib.Path,
) -> None:
    module = tmp_path / "src" / "structbench" / "verification" / "measures" / "m.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "from ...benchmarks import get_benchmark\n"
        "from ..results import Verdict\n"
        "from ... import eval\n"
        "import structbench.models.cgn\n"
        "import numpy as np\n",
        encoding="utf-8",
    )
    assert _first_party_targets(module, tmp_path / "src") == {
        "benchmarks",
        "verification",
        "eval",
        "models",
    }


def test_verification_imports_only_core_and_datasets() -> None:
    modules = sorted((_SRC / "structbench" / "verification").rglob("*.py"))
    assert modules, "verification package not found - run pytest from the repo root"
    offenders = {
        str(path): sorted(_first_party_targets(path, _SRC) - _ALLOWED)
        for path in modules
        if _first_party_targets(path, _SRC) - _ALLOWED
    }
    assert not offenders, f"forbidden first-party imports: {offenders}"
