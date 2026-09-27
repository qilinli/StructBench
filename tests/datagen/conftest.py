"""Shared fixtures: a minimal valid dataset definition (ADR-0071)."""

from pathlib import Path

import pytest

pytest.importorskip("scipy")

MINIMAL_TOML = """\
[dataset]
name = "toy"
case_prefix = "TOY"
units = "t-mm-s"
solver = "abaqus"

[declaration]
unit_system = "t-mm-s"
discretisation = "FEM"
erosion = false
fields = ["node/displacement", "solid/stress", "global/kinetic_energy"]

[fixed]
E = 1000.0

[variables]
L = [1.0, 2.0]
v0 = [10.0, 20.0]

[splits.train]
n = 4
seed = 1

[splits.pilot]
points = [{ L = 1.0, v0 = 10.0 }, { L = 2.0, v0 = 20.0 }]
probe = true

[levels]
refine_key = "refine"
production = "1"
pilot = ["1", "2"]

[pilot]
split = "pilot"
fine_cases = ["TOY-pilot-0000"]
min_free_gb = 5.0
accepted_gaps = ["solver_identity_complete"]

[qoi]
names = ["length"]
units = ["m"]
"""

MINIMAL_PROBLEM = '''\
"""A toy problem: one quad per level, byte-stable."""

from structbench.datagen.abaqus import deck


def input_deck(params, variant):
    k = int(params.get("refine", 1))
    mesh = deck.structured_quad_mesh(k, k, 0.0, float(params["L"]), 0.0, 1.0)
    return deck.heading("toy") + deck.node_block(mesh.node_labels, mesh.coords)


def mesh(params):
    k = int(params.get("refine", 1))
    return deck.structured_quad_mesh(k, k, 0.0, float(params["L"]), 0.0, 1.0)


def qoi(case):
    x = case.nodes.coords
    return {"length": float(x[:, 0].max() - x[:, 0].min())}
'''


def write_definition(
    dataset_dir: Path, *, toml: str | None = None, problem: str | None = None
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset.toml").write_bytes((toml or MINIMAL_TOML).encode())
    (dataset_dir / "problem.py").write_bytes((problem or MINIMAL_PROBLEM).encode())
    return dataset_dir


@pytest.fixture
def definition_dir(tmp_path: Path) -> Path:
    return write_definition(tmp_path / "toy")
