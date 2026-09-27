"""The shipped example definition: the single-rod conformance case."""

import hashlib
from importlib import resources

import numpy as np
import pytest

pytest.importorskip("scipy")

from structbench.core import Case, ElementBlock, Metadata, Nodes, Response  # noqa: E402
from structbench.datagen import definition, sampling  # noqa: E402

EXAMPLE = resources.files("structbench.datagen") / "examples" / "abaqus_conformance"


def _dir():
    with resources.as_file(EXAMPLE) as path:
        return path


def _production_params(d, problem):
    point = sampling.sample_split(d.variables, d.regions, d.split(d.pilot.split))[0]
    return {**d.fixed, **point.params, d.levels.refine_key: d.levels.production}


def test_the_example_loads_and_names_its_pilots():
    d = definition.load_definition(_dir())
    assert d.name == "abaqus_conformance" and d.solver == "abaqus"
    assert d.split(d.pilot.split).probe and len(d.split(d.pilot.split).points) == 3
    assert d.levels.production in d.levels.pilot


def test_the_example_deck_is_byte_stable_and_requests_the_ledger():
    d, problem = definition.load_definition(_dir()), definition.load_problem(_dir())
    params = _production_params(d, problem)
    a, b = problem.input_deck(params, None), problem.input_deck(params, None)
    assert hashlib.sha256(a.encode()).digest() == hashlib.sha256(b.encode()).digest()
    assert "ALLPW" in a and "*DYNAMIC, EXPLICIT, SCALE FACTOR=0.5" in a


def test_the_example_qoi_returns_the_declared_names():
    d, problem = definition.load_definition(_dir()), definition.load_problem(_dir())
    mesh = problem.mesh(_production_params(d, problem))
    n, e = len(mesh.node_labels), len(mesh.element_labels)
    case = Case(
        metadata=Metadata(case_id="x", dimension=2, source_units=d.units),
        nodes=Nodes(coords=mesh.coords * 1e-3, node_id=mesh.node_labels),
        elements={
            "solid": ElementBlock(
                connectivity=mesh.connectivity - 1,
                element_id=mesh.element_labels,
                part_id=np.ones(e, np.int64),
            )
        },
        materials=[],
        response=Response(
            time=np.array([0.0, 1e-6]),
            node={"displacement": np.zeros((2, n, 2), np.float32)},
            element={"solid": {"effective_plastic_strain": np.zeros((2, e))}},
            globals_={},
        ),
    )
    out = problem.qoi(case)
    assert tuple(out) == d.qoi.names
    assert all(isinstance(v, float) for v in out.values())
