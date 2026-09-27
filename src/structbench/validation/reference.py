"""Reference-experiment sets (``validation-reference/1``): loading and access.

A set is one JSON file under ``references/`` with its sources (DOIs, whether
each was consulted directly, the licence its curves travel under), the
extraction tool and its checks, the tests' conditions and measured outlines,
and the measures the shared functions return on those outlines.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REFERENCE_FORMAT = "validation-reference/1"
_REFERENCES = resources.files("structbench.validation") / "references"


class ReferenceError(ValueError):
    """A reference set that is missing or does not meet the format."""


@dataclass(frozen=True)
class Test:
    """One physical test: its conditions, measured outline and measures (mm)."""

    id: str
    source: str
    via: str | None
    material: str
    L0_mm: float
    D0_mm: float
    v0_ms: float
    T0_K: float
    outline_rz_mm: NDArray[np.float64]
    Lf_mm: float
    Rf_mm: float
    Wf_mm: tuple[float, ...]


@dataclass(frozen=True)
class ReferenceSet:
    name: str
    family: str
    title: str
    units: dict[str, str]
    sources: tuple[dict[str, Any], ...]
    extraction: dict[str, Any]
    fractions: tuple[float, ...]
    tests: tuple[Test, ...]
    caveats: tuple[str, ...]
    sha256: str

    def test(self, test_id: str) -> Test:
        for t in self.tests:
            if t.id == test_id:
                return t
        raise KeyError(test_id)


def list_references() -> list[str]:
    """The names of the shipped sets."""
    return sorted(
        p.name[: -len(".json")]
        for p in _REFERENCES.iterdir()
        if p.name.endswith(".json")
    )


def load_reference(name: str) -> ReferenceSet:
    """A shipped set by name."""
    path = _REFERENCES / f"{name}.json"
    if not path.is_file():
        raise ReferenceError(
            f"no reference set {name!r}; shipped: {', '.join(list_references())}"
        )
    with resources.as_file(path) as p:
        return load_reference_file(p)


def _test(raw: dict[str, Any]) -> Test:
    return Test(
        id=str(raw["id"]),
        source=str(raw["source"]),
        via=raw.get("via"),
        material=str(raw["material"]),
        L0_mm=float(raw["L0_mm"]),
        D0_mm=float(raw["D0_mm"]),
        v0_ms=float(raw["v0_ms"]),
        T0_K=float(raw["T0_K"]),
        outline_rz_mm=np.asarray(raw["outline_rz_mm"], dtype=np.float64),
        Lf_mm=float(raw["Lf_mm"]),
        Rf_mm=float(raw["Rf_mm"]),
        Wf_mm=tuple(float(w) for w in raw["Wf_mm"]),
    )


def load_reference_file(path: Path) -> ReferenceSet:
    """A set from any path; the sha256 of its bytes travels with it."""
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReferenceError(f"{path.name}: {exc}") from exc
    if data.get("format") != REFERENCE_FORMAT:
        raise ReferenceError(
            f"{path.name}: format {data.get('format')!r} is not {REFERENCE_FORMAT}"
        )
    try:
        tests = tuple(_test(t) for t in data["tests"])
        ids = [t.id for t in tests]
        if len(set(ids)) != len(ids):
            raise ReferenceError(f"{path.name}: duplicate test ids")
        return ReferenceSet(
            name=str(data["name"]),
            family=str(data["family"]),
            title=str(data["title"]),
            units={str(k): str(v) for k, v in data["units"].items()},
            sources=tuple(data["sources"]),
            extraction=dict(data["extraction"]),
            fractions=tuple(float(f) for f in data["measures"]["fractions"]),
            tests=tests,
            caveats=tuple(str(c) for c in data.get("caveats", ())),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ReferenceError):
            raise
        raise ReferenceError(f"{path.name}: {type(exc).__name__}: {exc}") from exc
