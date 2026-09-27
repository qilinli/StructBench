"""The dataset definition: ``dataset.toml`` and ``problem.py`` (ADR-0071).

``load_definition`` parses and validates the tables a dataset must supply;
``load_problem`` imports the dataset's code and checks it defines the hooks the
stages call. Both raise ``DefinitionError`` naming the table and field, so a
user who has just filled the template learns what is wrong before any solver
runs. The contract is the design's table ("The dataset definition").
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from structbench.core.io import unit_factors
from structbench.datagen import sampling

DEFINITION_FILE = "dataset.toml"
PROBLEM_FILE = "problem.py"
SOLVERS = ("abaqus",)
REQUIRED_HOOKS = ("input_deck", "mesh", "qoi")
OPTIONAL_HOOKS = ("feasible",)


class DefinitionError(ValueError):
    """A dataset definition that does not meet the contract; says which field."""


@dataclass(frozen=True)
class Levels:
    refine_key: str
    production: str
    pilot: tuple[str, ...]


@dataclass(frozen=True)
class Pilot:
    split: str
    fine_cases: tuple[str, ...]
    min_free_gb: float
    accepted_gaps: tuple[str, ...]


@dataclass(frozen=True)
class QoiSpec:
    names: tuple[str, ...]
    units: tuple[str, ...]


@dataclass(frozen=True)
class Definition:
    path: Path
    raw: dict[str, Any]
    name: str
    case_prefix: str
    units: str
    solver: str
    fixed: dict[str, Any]
    variables: dict[str, tuple[float, float]]
    regions: dict[str, dict[str, tuple[float, float]]]
    splits: tuple[sampling.Split, ...]
    limits: dict[str, Any]
    levels: Levels
    pilot: Pilot
    qoi: QoiSpec
    retention: dict[str, Any]

    def split(self, name: str) -> sampling.Split:
        for s in self.splits:
            if s.name == name:
                return s
        raise KeyError(name)

    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    table = raw.get(name)
    if not isinstance(table, dict):
        raise DefinitionError(f"{name}: table missing from {DEFINITION_FILE}")
    return table


def _field(table: dict[str, Any], where: str, key: str, kind: type) -> Any:
    if key not in table:
        raise DefinitionError(f"{where}.{key}: missing")
    value = table[key]
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind):
        raise DefinitionError(f"{where}.{key}: expected {kind.__name__}")
    return value


def _strings(table: dict[str, Any], where: str, key: str) -> tuple[str, ...]:
    value = _field(table, where, key, list)
    if not all(isinstance(v, str) for v in value):
        raise DefinitionError(f"{where}.{key}: expected a list of strings")
    return tuple(value)


def load_definition(dataset_dir: Path) -> Definition:
    """Parse and validate ``<dataset_dir>/dataset.toml``."""
    path = dataset_dir / DEFINITION_FILE
    if not path.is_file():
        raise DefinitionError(f"{DEFINITION_FILE}: not found in {dataset_dir}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    dataset = _table(raw, "dataset")
    name = _field(dataset, "dataset", "name", str)
    prefix = _field(dataset, "dataset", "case_prefix", str)
    units = _field(dataset, "dataset", "units", str)
    try:
        unit_factors(units)
    except ValueError as exc:
        raise DefinitionError(f"dataset.units: {exc}") from exc
    solver = _field(dataset, "dataset", "solver", str)
    if solver not in SOLVERS:
        raise DefinitionError(f"dataset.solver: {solver!r} is not one of {SOLVERS}")
    _table(raw, "declaration")  # its contents are validated by declared_from_toml

    fixed = dict(_table(raw, "fixed")) if "fixed" in raw else {}
    limits = dict(raw.get("limits", {}))
    variables = sampling.parse_bounds(_table(raw, "variables"), "variables")
    regions = {
        r: sampling.parse_bounds(box, f"regions.{r}")
        for r, box in raw.get("regions", {}).items()
    }
    if "splits" not in raw:
        raise DefinitionError("splits: at least one [splits.<name>] table is required")
    try:
        splits = tuple(sampling.parse_splits(raw))
    except (ValueError, KeyError) as exc:
        raise DefinitionError(f"splits: {exc}") from exc

    lv = _table(raw, "levels")
    levels = Levels(
        _field(lv, "levels", "refine_key", str),
        _field(lv, "levels", "production", str),
        _strings(lv, "levels", "pilot"),
    )
    if levels.production not in levels.pilot:
        raise DefinitionError(
            f"levels.production: {levels.production!r} is not among levels.pilot"
        )
    tables = (("fixed", fixed), ("variables", variables), ("limits", limits))
    for where, table in tables:
        if levels.refine_key in table:
            raise DefinitionError(
                f"{where}.{levels.refine_key}: the refine key is reserved for levels"
            )

    pt = _table(raw, "pilot")
    pilot = Pilot(
        _field(pt, "pilot", "split", str),
        _strings(pt, "pilot", "fine_cases"),
        _field(pt, "pilot", "min_free_gb", float),
        _strings(pt, "pilot", "accepted_gaps"),
    )
    names = {s.name: s for s in splits}
    if pilot.split not in names:
        raise DefinitionError(f"pilot.split: no [splits.{pilot.split}] table")
    if not names[pilot.split].points:
        raise DefinitionError(f"pilot.split: [splits.{pilot.split}] must list points")
    if not names[pilot.split].probe:
        raise DefinitionError(
            f"splits.{pilot.split}: the pilot split needs probe = true"
        )

    q = _table(raw, "qoi")
    qoi = QoiSpec(_strings(q, "qoi", "names"), _strings(q, "qoi", "units"))
    if not qoi.names:
        raise DefinitionError("qoi.names: at least one quantity of interest")
    if len(qoi.units) != len(qoi.names):
        raise DefinitionError("qoi.units: one unit per name")

    retention = dict(raw.get("retention", {}))
    unknown = set(retention) - {"odb_fraction", "odb_seed", "odb_cases"}
    if unknown:
        raise DefinitionError(f"retention: unknown keys {sorted(unknown)}")

    return Definition(
        path,
        raw,
        name,
        prefix,
        units,
        solver,
        fixed,
        variables,
        regions,
        splits,
        limits,
        levels,
        pilot,
        qoi,
        retention,
    )


def load_problem(dataset_dir: Path) -> ModuleType:
    """Import ``<dataset_dir>/problem.py`` by path; no bytecode is written.

    A ``__pycache__`` inside the dataset's repository would make its provenance
    read as uncommitted work.
    """
    path = dataset_dir / PROBLEM_FILE
    if not path.is_file():
        raise DefinitionError(f"{PROBLEM_FILE}: not found in {dataset_dir}")
    spec = importlib.util.spec_from_file_location(
        f"dataset_problem_{dataset_dir.name}", path
    )
    if spec is None or spec.loader is None:
        raise DefinitionError(f"{PROBLEM_FILE}: cannot be loaded from {dataset_dir}")
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    # Registered before execution, as importlib's recipe says: dataclasses
    # resolve a class's annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous
    for hook in REQUIRED_HOOKS:
        if not callable(getattr(module, hook, None)):
            raise DefinitionError(f"{PROBLEM_FILE}: defines no {hook}()")
    return module


def problem_sha256(dataset_dir: Path) -> str:
    return hashlib.sha256((dataset_dir / PROBLEM_FILE).read_bytes()).hexdigest()
