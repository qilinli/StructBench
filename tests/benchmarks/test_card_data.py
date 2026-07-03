"""Card-vs-data validation (ADR-0025) — runs only when data is present.

Set STRUCTBENCH_DATA_ROOT to the canonical HDF5 directory of the Taylor
dataset (the folder holding <case_id>.h5) to enable.
"""

import os
from pathlib import Path

import pytest

from structbench.benchmarks import get_benchmark
from structbench.core.io import read_case

DATA_ROOT = os.environ.get("STRUCTBENCH_DATA_ROOT")

pytestmark = pytest.mark.skipif(
    DATA_ROOT is None, reason="STRUCTBENCH_DATA_ROOT not set"
)


def test_taylor_card_matches_one_canonical_case():
    spec = get_benchmark("taylor_impact_2d")
    case_id = spec.splits["train"][0]
    case = read_case(Path(DATA_ROOT) / f"{case_id}.h5")

    lo, hi = (int(x) for x in spec.card.particles_per_case.split("-"))
    n_particles = case.elements["sph"].element_id.shape[0]
    assert lo <= n_particles <= hi

    assert case.response is not None
    assert case.response.time.shape[0] == spec.card.n_frames
