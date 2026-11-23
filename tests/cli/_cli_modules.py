"""The modules behind ``structbench-train``, for tests that replace a name."""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Any

import pytest

import structbench.cli.evaluation
import structbench.cli.train
import structbench.cli.training


def _modules() -> list[ModuleType]:
    found = [structbench.cli.train]
    for package in (structbench.cli.training, structbench.cli.evaluation):
        for info in pkgutil.iter_modules(package.__path__):
            found.append(importlib.import_module(f"{package.__name__}.{info.name}"))
    return found


CLI_MODULES: tuple[ModuleType, ...] = tuple(_modules())


def patch_cli(monkeypatch: pytest.MonkeyPatch, name: str, value: Any) -> None:
    """Replace ``name`` in every ``structbench-train`` module that holds it.

    The command's code is spread over ``cli/train.py``, ``cli/training/`` and
    ``cli/evaluation/``, and each module binds the names it imports, so a
    replacement has to reach every binding to take effect.
    """
    holders = [module for module in CLI_MODULES if hasattr(module, name)]
    assert holders, f"no structbench-train module holds {name!r}"
    for module in holders:
        monkeypatch.setattr(module, name, value)
