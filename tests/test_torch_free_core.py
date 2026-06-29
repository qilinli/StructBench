"""ADR-0018: the core substrate must not depend on the ML stack."""

import ast
import pathlib

CORE = pathlib.Path("src/structbench/core")


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def test_core_does_not_import_torch():
    offenders = {
        str(p): {"torch", "torch_geometric"} & _imported_modules(p)
        for p in CORE.rglob("*.py")
        if {"torch", "torch_geometric"} & _imported_modules(p)
    }
    assert not offenders, f"core/ must stay torch-free (ADR-0018): {offenders}"
