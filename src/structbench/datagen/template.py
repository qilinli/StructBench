"""Scaffold a dataset definition (``new``) and check one (``check``), ADR-0071.

``scaffold`` copies the shipped example and renames it; ``check_definition``
runs every contract check that needs no solver: the tables, the problem's
hooks, everything ``generate`` would refuse, a byte-stable deck (also in a
fresh interpreter with another hash seed, so a deck built from set or dict
order is caught), nesting mesh levels, and QoI names that match the
declaration. Problems come back as plain sentences, one per line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from structbench.core import Case, ElementBlock, Metadata, Nodes, Response
from structbench.datagen import generate
from structbench.datagen.definition import (
    Definition,
    DefinitionError,
    load_definition,
    load_problem,
)

EXAMPLE_DIR = resources.files("structbench.datagen") / "examples" / "abaqus_conformance"
SCAFFOLD_FILES = ("dataset.toml", "problem.py", "README.md", "DATA_CARD.md")
EXAMPLE_NAME = "abaqus_conformance"

#: Run under another hash seed: prints the sha256 of one deck.
_FRESH_DECK = """\
import hashlib, json, sys
from pathlib import Path
from structbench.datagen.definition import load_problem
problem = load_problem(Path(sys.argv[1]))
variant = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
text = problem.input_deck(json.loads(sys.argv[2]), variant)
sys.stdout.write(hashlib.sha256(text.encode("utf-8")).hexdigest())
"""


def scaffold(target: Path, name: str, *, solver: str = "abaqus") -> list[Path]:
    """Write the template into ``target`` (which must be empty or absent)."""
    if solver != "abaqus":
        raise ValueError(f"no template for solver {solver!r}")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty")
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for filename in SCAFFOLD_FILES:
        text = (EXAMPLE_DIR / filename).read_text(encoding="utf-8")
        text = text.replace(EXAMPLE_NAME, name).replace("\r\n", "\n")
        if filename == "README.md":
            body = text.split("\n", 2)[2]
            text = (
                f"# {name}\n\nStarted from the example definition; edit "
                "dataset.toml and problem.py, then run `structbench-datagen check`.\n\n"
            ) + body
        path = target / filename
        path.write_bytes(text.encode("utf-8"))
        written.append(path)
    return written


def mesh_nests(coarse: NDArray[np.float64], fine: NDArray[np.float64]) -> bool:
    """Every coarse node coincides with a fine node (to 1e-9 of the extent)."""
    from scipy.spatial import cKDTree

    scale = float(np.ptp(fine, axis=0).max()) or 1.0
    distance, _ = cKDTree(fine).query(coarse)
    return bool(distance.max() <= 1e-9 * scale)


def _synthetic_case(defn: Definition, grid: Any) -> Case:
    n, e = len(grid.node_labels), len(grid.element_labels)
    return Case(
        metadata=Metadata(case_id="check", dimension=2, source_units=defn.units),
        nodes=Nodes(
            coords=np.asarray(grid.coords, float), node_id=np.asarray(grid.node_labels)
        ),
        elements={
            "solid": ElementBlock(
                connectivity=np.asarray(grid.connectivity) - 1,
                element_id=np.asarray(grid.element_labels),
                part_id=np.ones(e, np.int64),
            )
        },
        materials=[],
        response=Response(
            time=np.array([0.0, 1.0]),
            node={"displacement": np.zeros((2, n, 2), np.float32)},
            element={
                "solid": {"effective_plastic_strain": np.zeros((2, e), np.float32)}
            },
            globals_={},
        ),
    )


def deck_sha256_in_fresh_interpreter(
    dataset_dir: Path, params: dict[str, Any], variant: str | None = None
) -> str:
    """The deck's sha256 from a new interpreter whose hash seed differs from ours.

    Python randomises ``str`` hashes per process, so a deck that walks a set
    or an unordered dict is identical within one interpreter and different in
    the next; only a second process with another seed can show it.
    """
    seed = "2" if os.environ.get("PYTHONHASHSEED") == "1" else "1"
    command = [
        sys.executable,
        "-c",
        _FRESH_DECK,
        str(dataset_dir),
        json.dumps(params),
        json.dumps(variant),
    ]
    env = {**os.environ, "PYTHONHASHSEED": seed}
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines() or ["no output"]
        raise RuntimeError(detail[-1])
    return proc.stdout.strip()


def check_definition(dataset_dir: Path) -> list[str]:
    """Every contract problem in plain words; an empty list means it passes."""
    try:
        defn = load_definition(dataset_dir)
        problem = load_problem(dataset_dir)
    except DefinitionError as exc:
        return [str(exc)]
    try:
        specs = generate.plan_cases(defn, getattr(problem, "feasible", None))
    except (ValueError, KeyError) as exc:  # what generate would refuse
        return [str(exc.args[0] if exc.args else exc)]
    except Exception as exc:
        return [f"problem.feasible: raised {type(exc).__name__}: {exc}"]
    problems: list[str] = []
    base = next(dict(s.params) for s in specs if s.split == defn.pilot.split)
    key = defn.levels.refine_key

    params = {**base, key: defn.levels.production}
    try:
        first = problem.input_deck(dict(params), None)
        second = problem.input_deck(dict(params), None)
    except Exception as exc:  # a broken hook is a problem to report, not a crash
        return problems + [f"problem.input_deck: raised {type(exc).__name__}: {exc}"]
    if first != second:
        problems.append(
            "problem.input_deck: not byte-stable "
            "(two calls with the same parameters differ)"
        )
    else:
        try:
            fresh = deck_sha256_in_fresh_interpreter(dataset_dir, params)
        except RuntimeError as exc:
            return problems + [
                f"problem.input_deck: failed in a fresh interpreter: {exc}"
            ]
        if fresh != hashlib.sha256(first.encode("utf-8")).hexdigest():
            problems.append(
                "problem.input_deck: not byte-stable (a fresh interpreter with "
                "another hash seed gives a different deck: it depends on set or "
                "dict order)"
            )

    grids: dict[str, Any] = {}
    coarsest: NDArray[np.float64] | None = None
    for level in defn.levels.pilot:
        try:
            grids[level] = problem.mesh({**base, key: level})
            coords = np.asarray(grids[level].coords, float)
        except Exception as exc:
            return problems + [f"problem.mesh: raised {type(exc).__name__}: {exc}"]
        if coarsest is None:
            coarsest = coords
        elif not mesh_nests(coarsest, coords):
            problems.append(
                f"problem.mesh: level {level} does not nest level "
                f"{defn.levels.pilot[0]}"
            )

    try:
        out = problem.qoi(_synthetic_case(defn, grids[defn.levels.production]))
    except Exception as exc:
        return problems + [f"problem.qoi: raised {type(exc).__name__}: {exc}"]
    if tuple(out) != defn.qoi.names:
        problems.append(
            f"problem.qoi: returns {sorted(out)} but qoi.names declares "
            f"{list(defn.qoi.names)}"
        )
    return problems


def main_new(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="structbench-datagen new", description=scaffold.__doc__
    )
    parser.add_argument("target", type=Path)
    parser.add_argument("--solver", default="abaqus")
    args = parser.parse_args(argv)
    try:
        written = scaffold(args.target, args.target.name, solver=args.solver)
    except (FileExistsError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    for path in written:
        print(f"wrote {path}")
    print("next: edit dataset.toml and problem.py, then `structbench-datagen check`")
    return 0


def main_check(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="structbench-datagen check", description=check_definition.__doc__
    )
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    problems = check_definition(args.dataset)
    for line in problems:
        print(line)
    if problems:
        return 1
    defn = load_definition(args.dataset)
    print(
        f"ok: {defn.name}: {len(defn.splits)} splits, {len(defn.variables)} "
        f"variables, pilot {defn.pilot.split!r} at levels {list(defn.levels.pilot)}"
    )
    return 0
