"""Scaffold a dataset definition (``new``) and check one (``check``), ADR-0071.

``scaffold`` copies the shipped example and renames it; ``check_definition``
runs every contract check that needs no solver: the tables, the problem's
hooks, a byte-stable deck, nesting mesh levels, and QoI names that match the
declaration. Problems come back as plain sentences, one per line.
"""

from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from structbench.core import Case, ElementBlock, Metadata, Nodes, Response
from structbench.datagen import sampling
from structbench.datagen.definition import (
    Definition,
    DefinitionError,
    load_definition,
    load_problem,
)

EXAMPLE_DIR = resources.files("structbench.datagen") / "examples" / "abaqus_conformance"
SCAFFOLD_FILES = ("dataset.toml", "problem.py", "README.md", "DATA_CARD.md")
EXAMPLE_NAME = "abaqus_conformance"


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


def check_definition(dataset_dir: Path) -> list[str]:
    """Every contract problem in plain words; an empty list means it passes."""
    try:
        defn = load_definition(dataset_dir)
    except DefinitionError as exc:
        return [str(exc)]
    try:
        problem = load_problem(dataset_dir)
    except DefinitionError as exc:
        return [str(exc)]
    problems: list[str] = []
    pilot = defn.split(defn.pilot.split)
    points = sampling.sample_split(defn.variables, defn.regions, pilot)
    base = {**defn.fixed, **points[0].params}
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

    coarsest: NDArray[np.float64] | None = None
    for level in defn.levels.pilot:
        coords = np.asarray(problem.mesh({**base, key: level}).coords, float)
        if coarsest is None:
            coarsest = coords
        elif not mesh_nests(coarsest, coords):
            problems.append(
                f"problem.mesh: level {level} does not nest level "
                f"{defn.levels.pilot[0]}"
            )

    grid = problem.mesh(params)
    try:
        out = problem.qoi(_synthetic_case(defn, grid))
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
