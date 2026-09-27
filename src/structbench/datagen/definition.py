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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from structbench.core.io import unit_factors
from structbench.datagen import sampling

DEFINITION_FILE = "dataset.toml"
PROBLEM_FILE = "problem.py"
SOLVERS = ("abaqus",)
#: The volume weights the convergence engine restricts element fields with.
SYMMETRIES = ("axisymmetric", "planar")
REQUIRED_HOOKS = ("input_deck", "mesh", "qoi")
OPTIONAL_HOOKS = ("feasible",)


class DefinitionError(ValueError):
    """A dataset definition that does not meet the contract; says which field."""


@dataclass(frozen=True)
class Levels:
    refine_key: str
    production: str
    pilot: tuple[str, ...]
    symmetry: str = "axisymmetric"  # one of SYMMETRIES


#: Every key ``[pilot]`` may carry; a misspelt optional key is refused, not ignored.
PILOT_KEYS = frozenset(
    {
        "split",
        "fine_cases",
        "min_free_gb",
        "accepted_gaps",
        "increment_key",
        "increment_factors",
        "frame_key",
        "frame_count_key",
        "frame_factor",
        "frame_tolerance",
        "settling_margin",
        "contact_force_global",
    }
)
QOI_KEYS = frozenset({"names", "units", "tolerance"})


@dataclass(frozen=True)
class Pilot:
    """The preflight's targets (ADR-0071, part two (b)).

    The probe fields have defaults so that a definition written before them
    still loads. ``increment_factors`` scale the production value of
    ``increment_key`` (``[fixed]``'s, or 1.0 when absent); an empty list
    disables the increment probe. ``frame_factor`` scales the stored frame
    interval for the frame probe; 0 disables it. ``contact_force_global``
    names a stored global whose separation frame the duration step measures.
    """

    split: str
    fine_cases: tuple[str, ...]
    min_free_gb: float
    accepted_gaps: tuple[str, ...]
    increment_key: str = "dt_scale"
    increment_factors: tuple[float, ...] = (0.5,)
    frame_key: str = "frame_interval"
    frame_count_key: str = "n_intervals"
    frame_factor: float = 0.5
    frame_tolerance: float = 0.05
    settling_margin: float = 0.25
    contact_force_global: str | None = None


@dataclass(frozen=True)
class QoiSpec:
    names: tuple[str, ...]
    units: tuple[str, ...]
    tolerance: tuple[float, ...] = ()  # relative, one per name; 0.01 each by default


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
        return file_sha256(self.path)


def file_sha256(path: Path) -> str:
    """The file's sha256 with LF line endings, whatever a checkout wrote."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _number(label: str) -> float | None:
    """The label as a number, or None when it is not one."""
    try:
        return float(label)
    except ValueError:
        return None


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


def _optional(
    table: dict[str, Any], where: str, key: str, kind: type, default: Any
) -> Any:
    return _field(table, where, key, kind) if key in table else default


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _placements(
    variables: Mapping[str, Any], splits: Sequence[sampling.Split]
) -> dict[str, list[tuple[str, str]]]:
    """Where each parameter name is set outside ``[fixed]``: (how, where)."""
    out: dict[str, list[tuple[str, str]]] = {}
    for name in variables:
        out.setdefault(name, []).append(("sampled", "[variables]"))
    for s in splits:
        for name in s.extra:
            out.setdefault(name, []).append(("sampled", f"splits.{s.name}.extra"))
        for name in s.categorical:
            out.setdefault(name, []).append(
                ("pinned", f"splits.{s.name}'s categorical")
            )
        for point in s.points:
            for name in point:
                out.setdefault(name, []).append(
                    ("set", f"the points of splits.{s.name}")
                )
    return out


def _probe_fields(
    pt: dict[str, Any],
    fixed: dict[str, Any],
    placements: Mapping[str, list[tuple[str, str]]],
) -> dict[str, Any]:
    """The optional ``[pilot]`` probe fields, checked against the definition.

    The increment and frame keys must be constants: the preflight scales the
    value it finds in ``[fixed]``, so a split that samples, pins or points the
    key elsewhere -- a probe split included, the pilot split above all --
    would make the pilots stand for a production that never runs that way.
    """
    out: dict[str, Any] = {
        "increment_key": _optional(pt, "pilot", "increment_key", str, "dt_scale"),
        "frame_key": _optional(pt, "pilot", "frame_key", str, "frame_interval"),
        "frame_count_key": _optional(
            pt, "pilot", "frame_count_key", str, "n_intervals"
        ),
    }
    factors = _optional(pt, "pilot", "increment_factors", list, [0.5])
    if not all(_is_number(f) for f in factors) or any(
        f <= 0.0 or f == 1.0 for f in factors
    ):
        raise DefinitionError(
            "pilot.increment_factors: each factor scales the production value and "
            "must be positive and not 1"
        )
    out["increment_factors"] = tuple(float(f) for f in factors)
    factor = _optional(pt, "pilot", "frame_factor", float, 0.5)
    if not 0.0 <= factor < 1.0:
        raise DefinitionError(
            "pilot.frame_factor: a fraction of the stored frame interval, "
            "0 <= factor < 1 (0 disables the frame probe)"
        )
    if factor > 0.0:
        stride = 1.0 / factor
        if abs(stride - round(stride)) > 1e-9:
            raise DefinitionError(
                f"pilot.frame_factor: 1 / {factor:g} is not a whole number of "
                "frames, so the stored clock would not be a stride of the probe's; "
                "use 0.5, 0.25, 0.2, 0.1, ..."
            )
    out["frame_factor"] = factor
    out["frame_tolerance"] = _optional(pt, "pilot", "frame_tolerance", float, 0.05)
    if out["frame_tolerance"] <= 0.0:
        raise DefinitionError("pilot.frame_tolerance: must be positive")
    out["settling_margin"] = _optional(pt, "pilot", "settling_margin", float, 0.25)
    if not 0.0 <= out["settling_margin"] < 1.0:
        raise DefinitionError("pilot.settling_margin: 0 <= margin < 1")
    name = pt.get("contact_force_global")
    if name is not None:
        if not isinstance(name, str) or not name:
            raise DefinitionError(
                "pilot.contact_force_global: a stored global's name, or omit the key"
            )
        name = name.removeprefix("global/")
    out["contact_force_global"] = name
    for key in ("increment_key", "frame_key", "frame_count_key"):
        for how, where in placements.get(out[key], []):
            remedy = {
                "sampled": "the probe scales the constant in [fixed]",
                "pinned": "make it a constant in [fixed]",
                "set": "the pilots must run at the constant in [fixed]",
            }[how]
            raise DefinitionError(
                f"pilot.{key}: {out[key]!r} is {how} by {where}; {remedy}"
            )
    count_key = out["frame_count_key"]
    if factor > 0.0 and out["frame_key"] in fixed and count_key in fixed:
        count = fixed[count_key]
        if not _is_number(count):
            raise DefinitionError(f"fixed.{count_key}: expected a number of frames")
        frames = count / factor
        if abs(frames - round(frames)) > 1e-9:
            raise DefinitionError(
                f"pilot.frame_factor: fixed.{count_key} / {factor:g} is not a whole "
                "number of frames"
            )
    return out


def load_definition(dataset_dir: Path) -> Definition:
    """Parse and validate ``<dataset_dir>/dataset.toml``."""
    path = dataset_dir / DEFINITION_FILE
    if not path.is_file():
        raise DefinitionError(f"{DEFINITION_FILE}: not found in {dataset_dir}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise DefinitionError(f"{DEFINITION_FILE}: {exc}") from exc

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
    try:
        variables = sampling.parse_bounds(_table(raw, "variables"), "variables")
        regions = {
            r: sampling.parse_bounds(box, f"regions.{r}")
            for r, box in raw.get("regions", {}).items()
        }
    except ValueError as exc:
        raise DefinitionError(str(exc)) from exc
    if "splits" not in raw:
        raise DefinitionError("splits: at least one [splits.<name>] table is required")
    try:
        splits = tuple(sampling.parse_splits(raw))
    except ValueError as exc:
        raise DefinitionError(str(exc)) from exc

    lv = _table(raw, "levels")
    symmetry = str(lv.get("symmetry", "axisymmetric"))
    if symmetry not in SYMMETRIES:
        raise DefinitionError(
            f"levels.symmetry: {symmetry!r} is not one of {SYMMETRIES}"
        )
    levels = Levels(
        _field(lv, "levels", "refine_key", str),
        _field(lv, "levels", "production", str),
        _strings(lv, "levels", "pilot"),
        symmetry,
    )
    if len(set(levels.pilot)) != len(levels.pilot):
        raise DefinitionError("levels.pilot: labels must be unique")
    numbers = [_number(label) for label in levels.pilot]
    if None not in numbers:
        values = [v for v in numbers if v is not None]
        if any(v <= 0.0 for v in values) or any(
            b <= a for a, b in zip(values[:-1], values[1:], strict=True)
        ):
            raise DefinitionError(
                "levels.pilot: numeric labels are refinement factors and must be "
                "positive and increasing, coarse to fine"
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
    _table(raw, "qoi")  # a missing table is named before a stray key is blamed
    unknown = sorted(set(pt) - PILOT_KEYS)
    if unknown:
        raise DefinitionError(f"pilot.{unknown[0]}: unknown field")
    pilot = Pilot(
        _field(pt, "pilot", "split", str),
        _strings(pt, "pilot", "fine_cases"),
        _field(pt, "pilot", "min_free_gb", float),
        _strings(pt, "pilot", "accepted_gaps"),
        **_probe_fields(pt, fixed, _placements(variables, splits)),
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
    unknown = sorted(set(q) - QOI_KEYS)
    if unknown:
        raise DefinitionError(f"qoi.{unknown[0]}: unknown field")
    qoi_names, qoi_units = _strings(q, "qoi", "names"), _strings(q, "qoi", "units")
    if not qoi_names:
        raise DefinitionError("qoi.names: at least one quantity of interest")
    if len(qoi_units) != len(qoi_names):
        raise DefinitionError("qoi.units: one unit per name")
    tolerance = q.get("tolerance", [0.01] * len(qoi_names))
    if (
        not isinstance(tolerance, list)
        or len(tolerance) != len(qoi_names)
        or not all(_is_number(t) and t > 0.0 for t in tolerance)
    ):
        raise DefinitionError(
            "qoi.tolerance: one positive relative tolerance per name (default 0.01)"
        )
    qoi = QoiSpec(qoi_names, qoi_units, tuple(float(t) for t in tolerance))

    retention = dict(raw.get("retention", {}))
    stray = sorted(set(retention) - {"odb_fraction", "odb_seed", "odb_cases"})
    if stray:
        raise DefinitionError(f"retention: unknown keys {stray}")

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
    read as uncommitted work. The dataset's directory is importable while the
    module loads, so ``problem.py`` may import a sibling (its measures, say);
    the entry is removed afterwards, and so are the sibling modules it
    imported, so two datasets' siblings of one name never meet.
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
    sys.path.insert(0, str(dataset_dir))
    known = set(sys.modules)
    # Registered before execution, as importlib's recipe says: dataclasses
    # resolve a class's annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # a broken problem.py is a definition problem
        sys.modules.pop(spec.name, None)
        raise DefinitionError(f"{PROBLEM_FILE}: {type(exc).__name__}: {exc}") from exc
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous
        if str(dataset_dir) in sys.path:
            sys.path.remove(str(dataset_dir))
        root = dataset_dir.resolve()
        siblings: list[str] = []
        for name in set(sys.modules) - known - {spec.name}:
            file = getattr(sys.modules.get(name), "__file__", None)
            if file and root in Path(file).resolve().parents:
                siblings.append(Path(file).resolve().relative_to(root).as_posix())
                del sys.modules[name]  # a sibling stays with its dataset
    for hook in REQUIRED_HOOKS:
        if not callable(getattr(module, hook, None)):
            raise DefinitionError(f"{PROBLEM_FILE}: defines no {hook}()")
    #: The dataset's own modules problem.py imported, relative to its directory;
    #: the stamp and the provenance hash them beside the two definition files.
    module.__siblings__ = tuple(sorted(siblings))  # type: ignore[attr-defined]
    return module


def problem_sha256(dataset_dir: Path) -> str:
    return file_sha256(dataset_dir / PROBLEM_FILE)


#: The siblings hash of a problem.py that imports no sibling.
NO_SIBLINGS_SHA256 = hashlib.sha256(b"").hexdigest()


def siblings_sha256(dataset_dir: Path, problem: ModuleType) -> str:
    """One hash over the sibling modules ``problem.py`` imported: each file's
    relative path and LF-normalised bytes, in path order."""
    digest = hashlib.sha256()
    for relative in getattr(problem, "__siblings__", ()):
        digest.update(relative.encode("utf-8") + b"\n")
        digest.update((dataset_dir / relative).read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\n")
    return digest.hexdigest()
