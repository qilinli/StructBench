"""Card-vs-data validation (ADR-0027) — runs only when data is present.

Each benchmark has its own env var:
  STRUCTBENCH_DATA_ROOT        — Taylor canonical HDF5 directory
  STRUCTBENCH_WAVE1D_DATA_ROOT — Wave-1d canonical HDF5 directory

Each parametrized case skips independently when its var is unset.
"""

import os
from pathlib import Path

import pytest

from structbench.benchmarks import get_benchmark
from structbench.core.io import read_case

_BENCHMARK_ROOTS = {
    "taylor_impact_2d": os.environ.get("STRUCTBENCH_DATA_ROOT"),
    "wave_propagation_1d": os.environ.get("STRUCTBENCH_WAVE1D_DATA_ROOT"),
}


@pytest.mark.parametrize("name", sorted(_BENCHMARK_ROOTS))
def test_card_matches_one_canonical_case(name: str) -> None:
    root = _BENCHMARK_ROOTS[name]
    if root is None:
        pytest.skip(f"data root env var for {name} not set")
    spec = get_benchmark(name)
    case = read_case(Path(root) / f"{spec.splits['train'][0]}.h5")
    lo, hi = (int(x) for x in spec.card.particles_per_case.split("-"))
    assert lo <= case.elements["sph"].element_id.shape[0] <= hi
    assert case.response is not None
    assert case.response.time.shape[0] == spec.card.n_frames
