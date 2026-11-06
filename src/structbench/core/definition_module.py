"""Import a definition module from a directory, by path.

A dataset definition (``problem.py``, ADR-0071) and a benchmark definition
(``benchmark.py``) are Python files that live outside the package in a
directory of their own and are loaded by path. This module is the one
importer both use, so the rules are the same: no bytecode is written beside
the definition (a ``__pycache__`` inside its repository would read as
uncommitted work), the directory is importable only while the module loads
so the file may import a sibling, and the siblings it imported are unloaded
afterwards so two definitions' siblings of one name never meet. The hashes
here are LF-normalised, so a checkout's line endings do not change them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


class DefinitionModuleError(ValueError):
    """The definition file is missing, cannot be loaded, or raised on import."""


def file_sha256(path: Path) -> str:
    """The file's sha256 with LF line endings, whatever a checkout wrote."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_definition_module(
    directory: Path, filename: str, *, qualifier: str
) -> ModuleType:
    """Import ``<directory>/<filename>`` by path and return the module.

    Parameters
    ----------
    directory:
        The definition's directory; importable while the module loads.
    filename:
        The file to import, e.g. ``"problem.py"`` or ``"benchmark.py"``.
    qualifier:
        Prefix of the module's name in ``sys.modules`` while it loads.

    Returns
    -------
    ModuleType
        The loaded module, with ``__siblings__``: the modules of the same
        directory it imported, as sorted POSIX paths relative to it.

    Raises
    ------
    DefinitionModuleError
        If the file is missing or cannot be loaded, or if importing it raises.
    """
    path = directory / filename
    if not path.is_file():
        raise DefinitionModuleError(f"{filename}: not found in {directory}")
    spec = importlib.util.spec_from_file_location(f"{qualifier}_{directory.name}", path)
    if spec is None or spec.loader is None:
        raise DefinitionModuleError(f"{filename}: cannot be loaded from {directory}")
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    sys.path.insert(0, str(directory))
    known = set(sys.modules)
    # Registered before execution, as importlib's recipe says: dataclasses
    # resolve a class's annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # a broken definition is a definition problem
        sys.modules.pop(spec.name, None)
        raise DefinitionModuleError(f"{filename}: {type(exc).__name__}: {exc}") from exc
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous
        if str(directory) in sys.path:
            sys.path.remove(str(directory))
        root = directory.resolve()
        siblings: list[str] = []
        for name in set(sys.modules) - known - {spec.name}:
            file = getattr(sys.modules.get(name), "__file__", None)
            if file and root in Path(file).resolve().parents:
                siblings.append(Path(file).resolve().relative_to(root).as_posix())
                del sys.modules[name]  # a sibling stays with its definition
    module.__siblings__ = tuple(sorted(siblings))  # type: ignore[attr-defined]
    return module


def definition_sha256s(
    directory: Path, filename: str, module: ModuleType
) -> dict[str, str]:
    """``{relative path: sha256}`` of the definition file and its siblings.

    The file comes first, then the siblings ``module`` imported in path
    order; every hash is LF-normalised. This is what a record stores to tie
    a run or a case to the definition that produced it.
    """
    out = {filename: file_sha256(directory / filename)}
    for relative in getattr(module, "__siblings__", ()):
        out[relative] = file_sha256(directory / relative)
    return out
